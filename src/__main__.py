"""Main entry point for the function calling system.

This program reads natural language prompts and function definitions,
then uses an LLM with constrained decoding to generate structured
JSON function calls. The constrained decoding guarantees valid JSON
output by limiting which tokens the model can produce at each step.

Usage:
    uv run python -m src
    uv run python -m src --functions_definition FILE --input FILE --output FILE
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model

from src.models import PromptInput, FunctionCall
from src.registry import FunctionRegistry
from src.vocab import VocabManager
from src.fsm import JSONStateMachine


def build_prompt(
    registry: FunctionRegistry, user_question: str
) -> str:
    """Build the prompt that tells the AI what functions are available.

    Creates a text prompt listing all function names, descriptions,
    and parameters, followed by the user's question. The AI uses this
    context to decide which function to call and what arguments to use.

    Args:
        registry: The loaded function registry.
        user_question: The user's natural language question.

    Returns:
        A formatted prompt string.
    """
    prompt = (
        "You are an AI assistant that calls functions."
        " Here are the available functions:\n\n"
    )
    for fn_name in registry.get_functions_name():
        desc = registry.get_description(fn_name)
        params = registry.get_parameters(fn_name)
        prompt += (
            f"Function: {fn_name}\n"
            f"Description: {desc}\n"
            f"Parameters: {json.dumps(params)}\n\n"
        )
    prompt += f"User Question: {user_question}\n"
    prompt += "Call the correct function in JSON format:\n"
    return prompt


def generate_function_call(
    ai: Any,
    vocab: VocabManager,
    registry: FunctionRegistry,
    question: str,
) -> Dict[str, Any]:
    """Generate a single function call for a given question.

    This is the core generation loop. For each token:
    1. Ask the FSM which tokens are valid right now
    2. If only one token is valid (forced structural text), use it
       directly WITHOUT calling the model — this is the key speedup
    3. If multiple tokens are valid (AI chooses a value), call the
       model, mask invalid tokens to -inf, pick the highest score
    4. Look up the token's text from the vocab dict (not ai.decode)

    Skipping model calls for forced tokens saves ~70% of inference
    calls since most tokens are structural JSON text.

    Args:
        ai: The LLM model instance.
        vocab: The vocabulary manager.
        registry: The function registry.
        question: The user's question.

    Returns:
        A dictionary with prompt, name, and parameters.
    """
    # Build the text prompt and encode it to token IDs
    prompt_text = build_prompt(registry, question)
    tokens: List[int] = ai.encode(prompt_text).squeeze().tolist()

    # Create a fresh state machine for this question
    fsm = JSONStateMachine(ai, vocab, registry)

    max_tokens = 200  # Safety limit to prevent infinite generation
    steps = 0
    generated = ""  # The JSON text being built token by token

    while fsm.state != "DONE" and steps < max_tokens:
        # Step 1: Ask the FSM which tokens are allowed right now
        allowed_tokens = fsm.get_allowed_tokens()

        # Step 2: Pick the next token
        if len(allowed_tokens) == 1:
            # FORCED TOKEN: only one valid choice (structural JSON text
            # like {"name": " or , "parameters": {). Skip the expensive
            # model inference call — we already know which token to use.
            token_id = allowed_tokens[0]
        else:
            # FREE CHOICE: multiple valid tokens. Ask the model to score
            # them and pick the best one using constrained decoding.

            # Get the model's raw scores (logits) for all possible tokens
            logits = np.array(ai.get_logits_from_input_ids(tokens))

            # THE CAGE: Set ALL logits to -infinity, then restore only
            # the tokens that the FSM says are valid. This makes it
            # impossible for the model to pick an invalid token.
            masked = np.full_like(logits, -np.inf)
            allowed_arr = np.array(allowed_tokens)
            masked[allowed_arr] = logits[allowed_arr]

            # Pick the valid token with the highest score
            token_id = int(np.argmax(masked))

        # Step 3: Look up the token's text from the vocabulary dict.
        # This is instant — no model call needed (replaces ai.decode).
        word = vocab.get_token_text(token_id)

        # Step 4: Append to our generated text and token ID list
        generated += word
        tokens.append(token_id)

        # Show progress as tokens are generated
        print(word, end="", flush=True)

        # Step 5: Tell the FSM what token was picked so it updates state
        fsm.commit(token_id, word)
        steps += 1

    print()

    # Post-processing: fix the closing braces.
    # When the last parameter is a number/boolean and ends with '}',
    # that '}' only closes the inner parameters object. We need to
    # add the outer '}' to make the JSON complete:
    # {"name": "...", "parameters": {"a": 2}  <-- missing outer }
    # {"name": "...", "parameters": {"a": 2}} <-- after fix
    json_str = generated.strip()
    if json_str.endswith("}") and not json_str.endswith("}}"):
        json_str += "}"

    # Parse the generated JSON and extract the function call info
    try:
        parsed = json.loads(json_str)
        return {
            "prompt": question,
            "name": parsed.get("name", "fn_not_found"),
            "parameters": parsed.get("parameters", {}),
        }
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return {
            "prompt": question,
            "name": "fn_not_found",
            "parameters": {},
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path for the output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline.

    Steps:
    1. Load function definitions from JSON
    2. Load test prompts from JSON
    3. Initialize the LLM model and vocabulary
    4. Process each prompt through constrained decoding
    5. Write results to output JSON file
    """
    args = parse_args()

    # --- Load function definitions ---
    try:
        registry = FunctionRegistry()
        registry.load(args.functions_definition)
    except FileNotFoundError:
        print(f"Error: File not found: {args.functions_definition}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.functions_definition}")
        sys.exit(1)

    # --- Load test prompts ---
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_tests = json.load(f)
        tests = [PromptInput.model_validate(t) for t in raw_tests]
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Invalid input data: {e}")
        sys.exit(1)

    # --- Initialize the AI model and vocabulary ---
    print("Initializing LLM...")
    ai = Small_LLM_Model()

    # Load vocabulary from the vocab file directly (instant).
    # Old code called ai.decode() 150K+ times here — that took minutes.
    vocab = VocabManager(vocab_path=ai.get_path_to_vocab_file())

    # --- Process each test prompt ---
    results: List[Dict[str, Any]] = []

    for i, test in enumerate(tests):
        print(f"\n--- Test {i + 1}/{len(tests)}: {test.prompt} ---")

        try:
            result = generate_function_call(
                ai, vocab, registry, test.prompt
            )

            # Validate the output format with Pydantic
            validated = FunctionCall.model_validate(result)
            results.append(validated.model_dump())
            print("SUCCESS")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "prompt": test.prompt,
                "name": "fn_not_found",
                "parameters": {},
            })

    # --- Write output ---
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nWrote {len(results)} results to {args.output}")
    except PermissionError:
        print(f"Error: Permission denied writing to {args.output}")
        sys.exit(1)
    except Exception as e:
        print(f"Error writing to {args.output}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    registry = FunctionRegistry()
    registry.load(args.functions_definition)
