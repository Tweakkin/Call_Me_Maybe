# Call Me Maybe - Evaluation Guide 4

This guide is a practical defense sheet for the `Call Me Maybe` project. It is
based on the project subject, the current README, the input files, and the
implementation in `src/`. It is intended to help explain the project clearly
during an evaluation, not to replace understanding the code.

## 1. The project in one minute

The project converts a natural-language request into a structured function
call. For example:

```text
What is the sum of 2 and 3?
```

becomes a JSON object containing the original prompt, the selected function
name, and typed parameters:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The difficult part is not producing JSON-looking text. The difficult part is
guaranteeing that a small language model cannot produce tokens that break the
required JSON structure or the function schema. The implementation does this
with constrained decoding: it uses a finite-state machine to determine the
legal next tokens, masks every other logit to negative infinity, and selects
the best remaining token.

The model is still responsible for the semantic decisions:

- Which available function matches the user request?
- What value should each parameter have?

The decoder is responsible for the output contract:

- Where punctuation, keys, quotes, and braces must appear.
- Which function names are possible.
- Which token categories are legal for each parameter type.
- When generation must finish.

This distinction is central. The code does not choose a function with a
keyword heuristic; it constrains the model while allowing the model to select
among registered function names.

## 2. The execution path

The command-line entry point is `src/__main__.py`. The complete flow is:

1. Parse `--functions_definition`, `--input`, and `--output`.
2. Load and validate function definitions through `FunctionRegistry`.
3. Load and validate each input prompt as a `PromptInput`.
4. Initialize `Small_LLM_Model` from the provided `llm_sdk`.
5. Read the model vocabulary and build cached token categories in
   `VocabManager`.
6. Build a prompt containing all available function descriptions and the user
   question.
7. Create a fresh `JSONStateMachine` for each question.
8. Generate one token at a time:
   - ask the FSM for allowed token IDs;
   - use a forced token directly if there is only one choice;
   - otherwise obtain logits from the model;
   - mask disallowed logits to `-inf`;
   - choose the highest remaining logit;
   - commit the chosen token to the FSM.
9. Complete the outer JSON object when needed.
10. Parse the generated JSON and validate the result with `FunctionCall`.
11. Write the list of results to the requested output path.

The important modules are:

| Module | Responsibility |
|---|---|
| `src/__main__.py` | CLI, prompt construction, generation loop, output |
| `src/fsm.py` | JSON/schema state transitions and allowed-token decisions |
| `src/vocab.py` | Vocabulary decoding and cached token filtering |
| `src/registry.py` | Loading and looking up function definitions |
| `src/models.py` | Pydantic input and output models |
| `llm_sdk/` | The supplied model wrapper |

## 3. The data contract

### Function definitions

`data/input/functions_definition.json` is an array. Each entry contains:

```json
{
  "name": "fn_greet",
  "description": "Generate a greeting message for a person by name.",
  "parameters": {
    "name": {
      "type": "string"
    }
  },
  "returns": {
    "type": "string"
  }
}
```

`FunctionDef` validates the top-level fields. `ParameterDef` currently stores
the type as a string so the registry can inspect it later. The registry keeps
definitions in a dictionary keyed by function name, which makes function and
parameter lookups straightforward.

Parameter order is preserved from the JSON object. The FSM places parameters
in a queue and emits them in that order. This is a deliberate simplification:
the model chooses values, but it does not choose the order of keys.

### Test prompts

`data/input/function_calling_tests.json` is an array of objects such as:

```json
{
  "prompt": "Reverse the string 'hello'"
}
```

Each object is validated by `PromptInput`.

### Output

The output is an array of objects with:

- `prompt`: the original string;
- `name`: the selected registered function name;
- `parameters`: a dictionary of generated values.

The generated text is parsed with `json.loads`, and the resulting wrapper is
validated with `FunctionCall`. The output format is therefore separate from
the function's `returns` field: the project describes a call; it does not
execute the called function.

## 4. How constrained decoding works

### Normal generation versus constrained generation

Without constraints, the model returns logits for the complete vocabulary.
Taking the largest logit could produce a missing quote, an unknown function,
prose, or invalid punctuation.

The implementation creates a mask:

```text
masked_logits = -infinity for every token
masked_logits[allowed_token_ids] = model_logits[allowed_token_ids]
```

`argmax` then cannot select an invalid token. A token with a logit of
negative infinity is effectively unavailable.

This does not make the model more intelligent. It restricts the set of
possible outputs so that the model's remaining choice is structurally safe.

### Forced tokens

At the beginning, the FSM queues the exact text:

```text
{"name": "
```

It also queues punctuation and parameter keys as it progresses. When exactly
one token is allowed, the generation loop uses it without requesting new
logits. This is both faster and deterministic.

### Function-name state

`VocabManager` precomputes `fn_candidates`, containing tokens whose decoded
characters could occur in one of the registered function names. The FSM
maintains `typed_name`. A candidate is allowed only if the current partial
name remains a prefix of at least one known function name, including the
closing quote.

When the closing quote is committed, the FSM stores the selected name and
loads its parameter names from the registry.

### Number state

Number tokens are vocabulary entries whose stripped text contains only digits,
`.` or `-`. The FSM allows these tokens while a number is being generated.
After the final parameter it allows a closing brace; otherwise it allows a
comma and moves to the next parameter.

This is a lexical constraint, not a complete numeric grammar. For example,
the token filter alone does not prove that a sequence has exactly one decimal
point or that a minus sign is in the correct position. This is an important
limitation to understand when discussing the implementation.

### Boolean state

Boolean candidates are found by comparing stripped token text with prefixes or
completed forms of `true` and `false`. This allows the model to construct a
boolean value from one or more vocabulary tokens.

### String state

String tokens cannot contain a quote or a control character. A quote token is
also allowed so the value can close. `string_length` stops a string after the
configured token limit, preventing an endless value.

The source currently sets the FSM string limit to 20 tokens. The README
describes a 50-token design decision, so these two documents should not be
presented as identical facts. If asked, describe the implementation value
accurately and identify the documentation discrepancy.

### Vocabulary caching

The vocabulary may contain more than 150,000 entries. `VocabManager` scans it
once, decodes BPE markers such as `Ġ` (space) and `Ċ` (newline), and caches
lists for numbers, strings, quotes, booleans, commas, closing braces, and
function-name candidates.

This avoids rescanning the entire vocabulary at every generation step. The
main loop also looks up token text from the cached dictionary rather than
decoding the complete sequence repeatedly.

## 5. Questions an evaluator may ask

### Conceptual questions

**Why use function calling instead of asking the LLM for the answer?**

Because downstream software needs an executable description, not prose. A
function call separates intent from execution and allows another component to
perform a calculation, API request, database query, or other action.

**Why is prompting alone insufficient?**

A small model may omit quotes, add explanations, invent keys, choose an
unknown function, or use the wrong type. A prompt expresses a preference;
logit masking enforces a token-level constraint.

**What is a logit?**

A logit is the model's unnormalized score for a possible next token. The code
does not need to convert logits to probabilities because selecting the maximum
score gives the same result as selecting the maximum softmax probability.

**Why use negative infinity?**

After `argmax`, any invalid entry set to negative infinity cannot win against
an allowed finite score. This is a simple and effective hard mask.

**What does the model decide and what does the FSM decide?**

The model decides semantic content within the legal choices. The FSM decides
the legal structural choices. The registry supplies the schema, and the
vocabulary manager maps schema categories to token IDs.

**Why is this a finite-state machine?**

At every step, generation is in a finite named state such as expecting a
function name, a number, a boolean, a string, or completion. The current
state and the committed token determine the next state and allowed tokens.

### Code-reading questions

**Where does the function list come from?**

`FunctionRegistry.load()` reads the JSON file, validates each entry with
Pydantic, and stores it in `functions`. `get_functions_name()` returns the
registered keys.

**Why is `FunctionRegistry` a Pydantic model?**

It follows the subject's requirement that classes use Pydantic for validation
and gives the registry typed, structured function definitions.

**Why build a new FSM for every prompt?**

Each prompt needs an independent generation state. Reusing one would risk
leaking the previous function name, parameter queue, or string length.

**Why append token IDs instead of re-encoding generated text?**

The prompt is encoded once. Each selected token ID is appended to the existing
input sequence, avoiding repeated tokenization and preserving exact token
boundaries.

**Why does the code sometimes add a final `}`?**

The inner parameters object and outer result object close at different points.
For the last number or boolean, the generated closing brace can close the
inner object first; the post-processing step adds the outer brace when the
generated text ends with only one `}`.

**What is the purpose of `returns` if it is not in the output?**

It describes the result type of the function that would eventually be called.
This project only generates the call request, so the return type is not
serialized into the call result.

### Reliability and edge-case questions

**Does the implementation guarantee 100% valid JSON?**

The intended constrained path is designed to guarantee the JSON structure,
but the claim should be stated carefully. The guarantee depends on token
classification, correct FSM transitions, complete allowed-token sets, and
successful completion before the generation cap. The parser and Pydantic
validation are final checks, not substitutes for correct constraints.

**What happens if the input JSON is missing or malformed?**

The current file-loading code uses normal `open()` and `json.load()` calls.
These raise errors for missing or malformed files. The subject asks for clear,
graceful error handling, so this is a point an evaluator may inspect and a
possible improvement request. Do not claim that every failure is currently
converted into a friendly message unless that behavior has been implemented.

**What happens if no function name matches?**

The function-name candidate set can become empty, leaving no valid token to
select. This is an important failure mode. A robust implementation would
detect it and report that the definitions or vocabulary cannot support the
requested generation instead of allowing an obscure downstream failure.

**Are numbers fully validated?**

No. The current number filter accepts tokens made of digits, `.` and `-`, but
does not maintain a detailed numeric sub-state. Pydantic's `FunctionCall`
uses `Any` for parameter values and therefore does not enforce the per-function
parameter types after parsing. This should be acknowledged if questioned.

**Are all function parameter types supported?**

The FSM has explicit paths for `string` and `boolean`; all other types go
through the number path. This works for the expected simple schema but is not a
general implementation of arbitrary JSON Schema types or nested objects.

**What happens when generation reaches `max_tokens`?**

The loop stops after 200 steps. The code then attempts to parse the generated
text. If it is incomplete, parsing fails and the fallback result uses
`fn_not_found` with empty parameters. This is a bounded failure rather than
an infinite generation loop, but it is not a successful function call.

## 6. Performance discussion

The largest avoidable cost is repeatedly scanning the vocabulary. The
implementation addresses this with:

- one vocabulary pass at startup;
- cached token lists;
- vectorized NumPy masking;
- direct token-ID appending;
- skipping model inference for forced single-token choices.

The trade-off is startup memory: the decoded vocabulary and several cached
lists are kept in memory. Generation still requires a model-logit request for
steps where multiple tokens are valid, so the model and hardware remain the
main runtime factors.

The subject's target is to process the supplied tests in under five minutes,
achieve at least 90% function-selection and argument-extraction accuracy, and
produce valid JSON for every result. These are separate metrics:

- syntactically valid JSON does not prove the right function was selected;
- the right function does not prove parameter values have the right types;
- fast execution does not prove semantic accuracy.

## 7. A strong live walkthrough

When demonstrating one prompt, explain the following sequence:

1. `main()` loads definitions and prompts.
2. `build_prompt()` gives the model the available functions and user request.
3. The FSM queues the opening JSON prefix.
4. Forced prefix tokens are emitted without model calls.
5. In function-name state, the model chooses among prefix-compatible tokens.
6. Once a name is complete, the registry provides the parameter order and types.
7. The FSM forces each key and punctuation.
8. The model chooses a value only from the token category for that type.
9. The FSM closes the parameter object and the outer object.
10. `json.loads()` and Pydantic validate the result before it is written.

A useful debugging demonstration is to print or inspect `allowed_tokens` at
one step and show that a token such as ordinary prose is absent. The key
point is not merely that the final text looks correct; it is that invalid
choices were unavailable during generation.

## 8. Suggested evaluator modifications

The subject warns that a small modification may be requested. Practice
explaining how you would approach these without hardcoding the supplied
examples:

1. Add a new function definition with one string parameter. The registry and
   parameter queue should handle it without changing function-specific logic.
2. Add a boolean parameter. Confirm that the boolean token filter and
   terminator handling produce valid JSON.
3. Add a function with multiple parameters of mixed types. Check parameter
   order, commas, quotes, and the final braces.
4. Change the input and output paths using command-line arguments.
5. Add a test prompt with a large number, an empty string, or special
   characters and explain which token filter handles it.
6. Change the maximum generation length and explain the safety/performance
   trade-off.
7. Replace greedy selection with sampling only if the mask remains applied;
   otherwise constrained decoding would no longer be deterministic.

For every modification, first identify the data contract, then identify the
FSM state affected, then test both a normal case and a boundary case.

## 9. Honest limitations to know before the evaluation

These are not reasons to hide the project; they are areas where a precise
explanation is stronger than an exaggerated claim:

- The implementation uses a simplified token-category filter rather than a
  complete JSON grammar.
- Number-token filtering does not fully validate decimal or sign syntax.
- Parameter values are stored as `Any` in `FunctionCall`, so schema-specific
  type validation after decoding is limited.
- Unknown or unsupported parameter types fall through to the number path.
- Input-file failures are not visibly normalized into a project-specific
  error format in the current entry point.
- The README says string values are capped at 50 tokens, while the current
  FSM value is 20.
- The project subject describes strict validation and graceful failures; an
  evaluator may ask how the implementation could be strengthened in those
  areas.

## 10. Short answers worth memorizing

**What is the core idea?**  
Use the LLM for semantic selection, but use an FSM and logit masking to make
invalid structured output impossible or immediately rejectable.

**Why inspect the vocabulary?**  
Constraints operate on token IDs, so the program must know which vocabulary
tokens represent numbers, strings, punctuation, booleans, and function-name
prefixes.

**Why not use a regular expression on the final output?**  
A final check can reject bad output, but it cannot prevent the model from
generating it. Constrained decoding applies the rule before every token is
selected.

**What is the main optimization?**  
Cache decoded vocabulary categories once and use NumPy to mask the model's
logits instead of scanning more than 150,000 tokens repeatedly.

**What does success mean?**  
The output is parseable JSON, follows the expected call shape, uses a
registered function, contains the required parameters, and achieves good
semantic accuracy on prompts that were not hardcoded.

**What would you improve first?**  
Add explicit schema-aware validation and richer FSM states for numbers and
unsupported JSON types, then make input and generation failures report clear
errors without returning a misleading success-shaped fallback.

