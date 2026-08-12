"""Main entry point for the function calling system."""
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
    """Build the prompt that tells the AI what functions exist.

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

    Args:
        ai: The LLM model instance.
        vocab: The vocabulary manager.
        registry: The function registry.
        question: The user's question.

    Returns:
        A dictionary with prompt, name, and parameters.
    """
    prompt_text = build_prompt(registry, question)
    fsm = JSONStateMachine(ai, vocab, registry)

    max_tokens = 200
    steps = 0
    generated = ""

    tokens: List[int] = ai.encode(prompt_text).squeeze().tolist()

    while fsm.state != "DONE" and steps < max_tokens:
        allowed_tokens = fsm.get_allowed_tokens()

        logits = np.array(ai.get_logits_from_input_ids(tokens))

        # The Cage: start everything at -inf, rescue only allowed
        masked = np.full_like(logits, -np.inf)
        allowed_arr = np.array(allowed_tokens)
        masked[allowed_arr] = logits[allowed_arr]

        token_id = int(np.argmax(masked))
        word = ai.decode([token_id])

        generated += word
        tokens.append(token_id)

        print(word, end="", flush=True)

        fsm.commit(token_id, word)
        steps += 1

    print()

    json_str = generated.strip()
    if json_str.endswith("}") and not json_str.endswith("}}"):
        json_str += "}"

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
        The parsed arguments.
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
    """Run the function calling pipeline."""
    args = parse_args()

    # Load function definitions
    try:
        registry = FunctionRegistry()
        registry.load(args.functions_definition)
    except FileNotFoundError:
        print(f"Error: File not found: {args.functions_definition}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.functions_definition}")
        sys.exit(1)

    # Load test prompts
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

    # Initialize the AI model and vocabulary
    print("Initializing LLM...")
    ai = Small_LLM_Model()
    vocab = VocabManager(
        vocab_path=ai.get_path_to_vocab_file(),
        ai_decode_fn=ai.decode
    )

    # Process each test prompt
    results: List[Dict[str, Any]] = []

    for i, test in enumerate(tests):
        print(f"\n--- Test {i + 1}/{len(tests)}: {test.prompt} ---")

        try:
            result = generate_function_call(
                ai, vocab, registry, test.prompt
            )

            # Validate output with pydantic
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

    # Write output
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
    main()
