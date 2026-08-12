*This project has been created as part of the 42 curriculum by yboukhmi.*

# Call Me Maybe

## Description

A function calling system that translates natural language prompts into structured JSON function calls using constrained decoding. Instead of hoping the LLM produces valid JSON, we force it token-by-token using logit masking.

The system uses the Qwen3-0.6B model (0.6 billion parameters) and achieves 100% valid JSON output through constrained decoding.

## Instructions

### Installation

```bash
uv sync
```

### Running

```bash
uv run python -m src
```

With custom paths:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/results.json
```

### Linting

```bash
make lint
```

## Algorithm Explanation

The system uses **constrained decoding** to guarantee valid JSON output:

1. **Vocabulary Loading**: We load the model's entire vocabulary (150,000+ tokens) and decode each token into its text representation.
2. **State Machine (FSM)**: A finite state machine tracks where we are in the JSON structure (e.g., expecting a function name, a number parameter, a string value).
3. **Logit Masking (The Cage)**: At each generation step, we set all invalid token logits to negative infinity. Only tokens that maintain valid JSON structure survive.
4. **Token Selection**: `np.argmax` picks the highest-scoring token from the remaining valid options.

This ensures every generated token is structurally valid, making it impossible for the model to produce broken JSON.

## Design Decisions

- **Cached Vocabulary Filters**: Number and string token lists are computed once at startup and cached, avoiding redundant 150,000-iteration loops during generation.
- **NumPy Masking**: The constraint cage uses vectorized NumPy operations instead of Python loops for massive speed improvements.
- **String Length Limits**: String parameters are capped at 50 tokens to prevent the model from generating infinitely long values.
- **Encode Once**: The prompt is encoded into token IDs once. New tokens are appended to the list instead of re-encoding the entire string each step.
- **Forced Parameter Order**: Instead of letting the AI choose which parameter to output next, the FSM forces the parameter keys in the exact order they appear in the JSON schema. This guarantees schema compliance and drastically simplifies the state machine.

## Performance Analysis

- **Accuracy**: 100% correct function selection on simple prompts. Complex multi-parameter string functions (like regex) may produce imperfect parameter values due to the small model size.
- **Speed**: All 11 test prompts complete in under 5 minutes on standard hardware.
- **Reliability**: 100% valid JSON output. Every result is parseable and schema-compliant.

## Challenges Faced

1. **Tokenizer Quirks**: The BPE tokenizer uses special characters (like `Ġ` for spaces). We decode every token through the model's decoder to get clean text.
2. **Performance**: Naive Python loops over 150,000 tokens were extremely slow. Solved with NumPy vectorized operations and caching.
3. **Infinite String Generation**: The model would sometimes repeat patterns endlessly. Fixed with a maximum token limit per string value.

## Testing Strategy

1. Run the full test suite: `uv run python -m src`
2. Verify all outputs parse as valid JSON
3. Check function names match available definitions
4. Confirm parameter types match the schema (numbers are numbers, strings are strings)
5. Test with edge cases: large numbers, special characters, multi-parameter functions

## Example Usage

Input prompt: "What is the sum of 2 and 3?"

Output:
```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
}
```

## Resources

- [Qwen3-0.6B Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained Decoding Overview](https://huggingface.co/blog/constrained-beam-search)
- AI was used to assist with code structure, debugging tokenizer issues, and writing documentation.
