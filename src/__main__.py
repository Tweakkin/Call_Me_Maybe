"""Main entry point for the function calling system.

Reads natural language prompts and function definitions, then uses
an LLM with constrained decoding to generate structured JSON function
calls. The constrained decoding guarantees valid JSON output by
limiting which tokens the model can produce at each step.

Usage:
    uv run python -m src
    uv run python -m src --functions_definition FILE --input FILE --output FILE
"""
import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model

from src.models import PromptInput, FunctionCall
from src.registry import FunctionRegistry
from src.vocab import VocabManager
from src.fsm import JSONStateMachine


def build_prompt(
    registry: FunctionRegistry, user_question: str
) -> str:
    """Build the prompt listing available functions and the user's question."""
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

    The generation loop:
    1. Ask the FSM which tokens are valid
    2. If only one token is valid (forced), use it directly (no AI call)
    3. If multiple are valid, call the AI, mask invalid tokens, pick best
    4. Look up the token's text from the vocab dict
    """
    # Build prompt and encode it to token IDs
    prompt_text = build_prompt(registry, question)
    tokens: List[int] = ai.encode(prompt_text).squeeze().tolist()

    # Create a fresh state machine
    fsm = JSONStateMachine(ai, vocab, registry)

    max_tokens = 200
    steps = 0
    generated = ""

    while fsm.state != "DONE" and steps < max_tokens:
        allowed_tokens = fsm.get_allowed_tokens()

        if len(allowed_tokens) == 1:
            # FORCED: only one valid choice, skip the AI call
            token_id = allowed_tokens[0]
        else:
            # FREE CHOICE: ask the AI to score tokens, then mask
            # everything except the allowed ones to -infinity
            logits = np.array(ai.get_logits_from_input_ids(tokens))
            masked = np.full_like(logits, -np.inf)
            allowed_arr = np.array(allowed_tokens)
            masked[allowed_arr] = logits[allowed_arr]
            token_id = int(np.argmax(masked))

        # Look up the token's text from our vocab dict
        word = vocab.get_token_text(token_id)
        generated += word
        tokens.append(token_id)
        print(word, end="", flush=True)

        # Tell the FSM what token was picked
        fsm.commit(token_id, word)
        steps += 1

    print()

    # Post-processing: when the last parameter is a number/boolean
    # and ends with }, that only closes the inner parameters object.
    # We add the outer } to complete the JSON.
    json_str = generated.strip()
    if json_str.endswith("}") and not json_str.endswith("}}"):
        json_str += "}"

    # Parse the JSON and return the result
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


def main() -> None:
    """Load everything, process each prompt, write results."""
    # Start by Handling command line This one is to handle (--help)
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    # Then Handle "--functions_definitions"
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON.",
    )
    # Then Handle "--input"
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON.",
    )
    # Then we handle "--output"
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path for the output JSON file.",
    )
    # This returns an object with the properties that we handled
    args = parser.parse_args()

    # Load functions and validate their format using Pydantic
    registry = FunctionRegistry()
    registry.load(args.functions_definition)

    # Open input file, turns raw text into Python List of dicts
    with open(args.input, "r", encoding="utf-8") as f:
        # ex : [ {"prompt": "What is 2+2?"}, {"prompt": "Say hello"} ]
        raw_tests = json.load(f)
    # Pydantic validate if dict has key named prompt, and type str
    tests = [PromptInput.model_validate(t) for t in raw_tests]

    # Initialize the AI model and vocabulary
    print("Initializing LLM...")
    ai = Small_LLM_Model()
    # Load vocab.json, decode tokens, flip and split into cached lists.
    # We pass the function names so the fn_candidates list is built
    # during this same single pass (no second loop needed later).
    vocab = VocabManager(
        vocab_path=ai.get_path_to_vocab_file(),
        fn_names=registry.get_functions_name(),
    )

    # Process each test prompt
    results: List[Dict[str, Any]] = []

    # Main Engine Loop, Goes through My prompts One by One
    for i, test in enumerate(tests):
        print(f"\n--- Test {i + 1}/{len(tests)}: {test.prompt} ---")

        result = generate_function_call(
            ai, vocab, registry, test.prompt
        )
        validated = FunctionCall.model_validate(result)
        results.append(validated.model_dump())
        print("SUCCESS")

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
