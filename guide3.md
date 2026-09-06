# Call Me Maybe — Ultimate Defense Guide

*Read this top to bottom once. Then use the Q&A as flashcards. During the defense, walk through sections 1→2→3→4 in order, then drill into any file they ask about.*

---

## 1. The Opening Pitch (Say This First)

> "This project turns natural-language questions into structured function calls — the kind an API or a program could actually execute. Instead of asking an LLM to write JSON and hoping it's valid, I force it to be valid: I use a small local model (Qwen3-0.6B) and, at every single token it generates, I block every token that would break the JSON schema. Only the tokens that keep the output syntactically and semantically correct are allowed to be picked. This is called **constrained decoding**, and it guarantees 100% valid, schema-compliant JSON — even though a 0.6-billion-parameter model would normally get it right maybe 30% of the time."

If they ask "what did you build specifically":

> "I built four cooperating pieces: a **function registry** that loads and validates the available functions, a **vocabulary manager** that pre-classifies the model's 150,000+ tokens into categories (numbers, strings, quotes, commas, etc.), a **finite state machine (FSM)** that knows where we are inside the JSON and tells the generator which token categories are legal right now, and a **generation loop** that ties it all together: ask the FSM what's allowed, ask the model to score only those tokens, pick the best one, repeat until the JSON is complete."

---

## 2. What Problem Does This Solve?

- LLMs are trained to produce fluent text, not machine-readable output.
- Small models (0.6B parameters) are especially bad at spontaneously producing valid JSON — the subject says they succeed ~30% of the time with prompting alone.
- Production systems need 99%+ reliability.
- The trick: **constrained decoding** — edit the probability landscape *before* the model picks a token, so it structurally cannot produce anything invalid.
- The task: given *"What is the sum of 2 and 3?"*, output not the answer (`5`) but the **function call**: `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`.

---

## 3. The 7 Steps From Start to Finish

```
Step 1: Parse command line arguments (argparse)
Step 2: Load function definitions into the registry (registry.py)
Step 3: Validate input prompts with Pydantic (models.py)
Step 4: Initialize the AI model (llm_sdk)
Step 5: Load vocabulary and build cached token lists (vocab.py)
Step 6: For each prompt, generate a JSON function call (fsm.py + main loop)
Step 7: Validate output and write results to file
```

---

## 4. Codebase Mindmap

```
Call_Me_Maybe/
│
├── llm_sdk/__init__.py        → Small_LLM_Model: talks to the actual neural network
│                                  (encode text, get logits, locate vocab file)
│
├── data/input/
│   ├── functions_definition.json  → catalog of callable functions + typed parameters
│   └── function_calling_tests.json → the natural-language prompts to convert
│
├── Makefile                   → install, run, debug, clean, lint, lint-strict
│
└── src/                       → YOUR code (everything you're evaluated on)
    ├── models.py     → Pydantic schemas: validate every JSON blob that flows in/out
    ├── registry.py   → FunctionRegistry: loads function definitions, answers
    │                    "what functions exist / what params does fn X take"
    ├── vocab.py      → VocabManager: loads vocab.json ONCE, pre-sorts
    │                    every token into buckets (number / string / quote /
    │                    comma / close-brace / boolean / fn-name-candidate)
    ├── fsm.py        → JSONStateMachine: the "cage". Knows where we are
    │                    inside the JSON and returns legal next-token IDs
    └── __main__.py   → orchestrator: CLI args, wires everything together
                         in the generation loop, writes the output file
```

**One-sentence role for every file:**

| File | Role |
|---|---|
| `llm_sdk/__init__.py` | The only file that touches PyTorch/HuggingFace — gives you tokens in, logits out. You must NOT touch its private members. |
| `src/models.py` | Defines the shape of every JSON object the program reads or writes, and validates it automatically (Pydantic). |
| `src/registry.py` | In-memory "database" of callable functions: names, descriptions, parameter names/types. |
| `src/vocab.py` | Turns the model's raw 150K-token vocabulary into fast lookup tables of *categories* of tokens. |
| `src/fsm.py` | The actual "constrainer" — a state machine that says which token IDs are legal at each step. |
| `src/__main__.py` | The conductor: reads CLI args and files, builds the prompt, runs the generation loop, writes output. |

---

## 5. File-by-File, Method-by-Method

### 5.1 `llm_sdk/__init__.py` — Provided, Read-Only, Do NOT Modify

Class **`Small_LLM_Model`**

| Method | What It Does |
|---|---|
| `encode(text) → Tensor` | Turns text into a tensor of token IDs. We use this to encode the prompt and forced text. |
| `get_logits_from_input_ids(input_ids) → List[float]` | Feeds the whole token sequence through the neural network, returns scores for every possible next token (151K scores). This is the "brain". |
| `get_path_to_vocab_file() → str` | Returns the path to `vocab.json` so `VocabManager` can read it. |
| `decode(ids) → str` | Token IDs → text. Not used in our code (we use `VocabManager.get_token_text()` instead — faster). |

**Subject rule:** Only use public methods. Using anything starting with `_` (like `self._model`) is forbidden.

---

### 5.2 `src/models.py` — The Blueprints (4 Pydantic Classes)

| Class | Fields | Purpose |
|---|---|---|
| `PromptInput` | `prompt: str` | Validates each input question from the tests file. |
| `ParameterDef` | `type: str` | Validates one parameter's type definition, e.g. `{"type": "number"}`. |
| `FunctionDef` | `name, description, parameters, returns` | Validates a complete function definition from the definitions file. |
| `FunctionCall` | `prompt, name, parameters` | Validates the final output format before writing to the results file. |

**No methods.** These are pure data shapes. Pydantic does all the work automatically.

**Why Pydantic and not just dicts?** The subject *requires* it: "All classes must use pydantic for validation." It also turns malformed input into readable errors early.

---

### 5.3 `src/registry.py` — The Rulebook (1 Class, 5 Methods)

State: `functions: Dict[str, FunctionDef]` — maps function name → validated definition.

| Method | What It Does |
|---|---|
| `load(path)` | Opens JSON file, validates each function with `FunctionDef.model_validate()`, stores them in a dict. |
| `get_functions_name()` | Returns `["fn_add_numbers", "fn_greet", ...]` — used to build the prompt and tell the FSM which names are valid. |
| `get_description(name)` | Returns the description string (used in the prompt so the AI understands what each function does). |
| `get_parameters(name)` | Returns `{"a": {"type": "number"}, "b": {"type": "number"}}` — used by the FSM to know parameter types. |
| `get_parameter_names(name)` | Returns `["a", "b"]` — the parameter names in order. This is the `params_queue` the FSM iterates through. |

---

### 5.4 `src/vocab.py` — The Dictionary (1 Function, 1 Class)

#### `swap_tokens(bpe_text)` — Standalone Function
The tokenizer's `vocab.json` stores spaces as `Ġ` and newlines as `Ċ`. This function just does two `.replace()` calls to turn them back into real characters. Example: `"Ġhello"` → `" hello"`.

#### `VocabManager.__init__(vocab_path, fn_names)` — The Single Pass Loop
In one loop through 151,000 tokens, it does 4 things:
1. **Clean** each token with `swap_tokens()`
2. **Flip** the dictionary from `text→ID` to `ID→text` (stored in `decoded_vocab`)
3. **Sort** each token into the right cached list
4. **Filter** tokens that could appear in function names (`fn_candidates`)

**Classification rules** (for each token's `stripped = text.strip()`):
- **Number**: every character is in `"0123456789.-"`
- **Quote**: exactly `"`
- **Comma**: contains `,` and every character is whitespace or `,`
- **Close brace**: contains `}` and every character is whitespace or `}`
- **Boolean**: `stripped` is a prefix of, or starts with, one of `["true", "false", "true,", "false,", "true}", "false}"]`
- **String**: no `"` and no control character (`ord(c) < 32`)
- **Function name candidate**: every character is inside the set of characters from all valid function names (plus `"`)

#### Getter Methods (9 total)
All just return the pre-built list. Zero computation. Instant.

`get_token_text()`, `get_all_tokens()`, `get_fn_candidates()`, `get_number_tokens()`, `get_string_tokens()`, `get_quote_tokens()`, `get_boolean_tokens()`, `get_comma_tokens()`, `get_close_brace_tokens()`

---

### 5.5 `src/fsm.py` — The Traffic Cop (1 Class, 4 Methods)

This is the heart of constrained decoding. It never generates text itself — it only answers *"which token IDs are legal right now?"* and updates its internal state when told which token was chosen.

#### States (5 total)

| State | What Happens |
|---|---|
| `EXPECT_FUNCTION_NAME` | AI picks tokens that spell a valid function name |
| `EXPECT_NUMBER_VALUE` | AI picks number tokens (digits, dot, minus) |
| `EXPECT_BOOLEAN_VALUE` | AI picks boolean tokens (true/false) |
| `EXPECT_STRING_VALUE` | AI picks string-safe tokens |
| `DONE` | Generation is complete |

#### The 4 Methods

| Method | Nickname | Role | Called By |
|---|---|---|---|
| `get_allowed_tokens()` | The Menu Giver | Returns the list of legal token IDs. **100% read-only** — never changes any variables. If forced tokens are queued, returns the first one. Otherwise checks the current state and returns the matching list. | Main loop (before each token) |
| `commit(t_id, text)` | The Note Taker | Updates variables after a token is picked. If forcing, pops the used token. If the AI is typing, tracks progress and detects when a value is finished. | Main loop (after each token) |
| `setup_next_parameter(prefix, empty_text)` | The Router | Pops the next parameter from the queue, checks its type in the registry, and tells `force_text` to type the parameter key. If queue is empty, closes the JSON. | `commit()` (when a value finishes) |
| `force_text(text, next_state)` | The Typist | Encodes exact structural text into token IDs and queues them. Saves what state to jump to when the queue is empty. | `setup_next_parameter()` and `__init__()` |

**Key design idea — "forced text":** Large chunks of the JSON skeleton (`{"name": "`, `, "parameters": {"a": `, closing braces) are 100% predictable — they don't need the AI at all. The FSM pre-encodes these literal strings into token IDs and replays them one at a time. The AI is only consulted when there's a genuine choice to make.

#### Key Variables (10 total)

| Variable | What It Tracks |
|---|---|
| `state` | Current FSM state |
| `next_state` | State to jump to after forced tokens finish |
| `forced_tokens` | Queue of token IDs being forced right now |
| `typed_name` | The function name the AI has typed so far (e.g. `"fn_ad"`) |
| `chosen_function` | The final complete function name (e.g. `"fn_add_numbers"`) |
| `fn_candidates` | Pre-filtered tokens that could appear in function names |
| `valid_functions` | List of all valid function names from registry |
| `params_queue` | Remaining parameter names to process (e.g. `["b"]`) |
| `string_length` | How many tokens the AI has typed for the current string |
| `string_limit` | Maximum tokens allowed per string (20) |

---

### 5.6 `src/__main__.py` — The Orchestra Conductor (3 Functions)

#### `build_prompt(registry, question)`
Takes a user question and wraps it in a big instruction text that lists all available functions with their names, descriptions, and parameter types. This is what the AI reads to understand the task.

#### `generate_function_call(ai, vocab, registry, question)`
The main engine. For each question:
1. Build the prompt and encode it to token IDs (`ai.encode(...).squeeze().tolist()`)
2. Create a fresh FSM (which immediately queues `{"name": "`)
3. **The Loop** (while `fsm.state != "DONE"` and under 200-token safety cap):
   - Ask FSM: `get_allowed_tokens()`
   - If only 1 allowed → use it directly (**skip the AI call** — saves time and guarantees byte-perfect structural text)
   - If multiple allowed → call `ai.get_logits_from_input_ids(tokens)`, mask all illegal tokens to `-infinity`, pick highest with `np.argmax`
   - Look up the token's text via `vocab.get_token_text()` (not `ai.decode()` — cheaper)
   - Append to `generated` string, append token ID to `tokens` list, print live
   - Call `fsm.commit(token_id, word)`
4. Post-processing: if the string ends with single `}` (not `}}`), add the missing outer `}`
5. `json.loads()` the result and return `{"prompt", "name", "parameters"}`

#### `main()`
1. Parse CLI args (`--functions_definition`, `--input`, `--output` with defaults)
2. Load function definitions into registry
3. Open and validate input prompts with Pydantic
4. Initialize AI model (`Small_LLM_Model()`)
5. Initialize `VocabManager` (passing `fn_names` for the single-pass filter)
6. Loop through each prompt, call `generate_function_call`
7. Validate each result with `FunctionCall.model_validate()`
8. Create output directory if needed, write all results to JSON file

---

## 6. The Core Algorithm — Constrained Decoding (5 Lines)

```python
logits = np.array(ai.get_logits_from_input_ids(tokens))  # AI scores all 151K tokens
masked = np.full_like(logits, -np.inf)                    # Create array of -infinity
allowed_arr = np.array(allowed_tokens)                     # The FSM's approved list
masked[allowed_arr] = logits[allowed_arr]                  # Copy only approved scores
token_id = int(np.argmax(masked))                          # Pick the highest approved score
```

We don't change how the AI thinks. We just cross out all the illegal answers before picking the winner.

- `np.array()` — converts a Python list into a fast NumPy array
- `np.full_like(x, val)` — creates a new array same size as `x`, filled with `val`
- `np.argmax()` — returns the **position** (index) of the highest value

---

## 7. Full Trace Example: "What is the sum of 2 and 3?"

| Step | FSM State | What Happens | Generated Text |
|---|---|---|---|
| 1 | (boot) | FSM forces `{"name": "` — no AI call | `{"name": "` |
| 2 | `EXPECT_FUNCTION_NAME` | AI picks tokens to spell `fn_add_numbers"` | `{"name": "fn_add_numbers"` |
| 3 | (commit detects `"`) | Saves function, loads `params_queue = ["a", "b"]`, pops `"a"`, forces `, "parameters": {"a": ` | `...fn_add_numbers", "parameters": {"a": ` |
| 4 | `EXPECT_NUMBER_VALUE` | AI picks `2`, then `,` | `..."a": 2,` |
| 5 | (commit detects `,`) | Pops `"b"` from queue, forces ` "b": ` | `...2, "b": ` |
| 6 | `EXPECT_NUMBER_VALUE` | AI picks `3`, then `}` | `..."b": 3}` |
| 7 | (commit detects `}`) | Queue empty → state = DONE | Loop ends |
| 8 | Post-processing | Adds missing outer `}` | `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}` |

---

## 8. Design Decisions — The "Why" Not Just the "What"

| Decision | Why It Matters |
|---|---|
| **Forced parameter order** | FSM follows the exact order from `functions_definition.json`. Removes ambiguity and simplifies the FSM (just a queue, not a set). |
| **Cached vocabulary filters** | Classifying 150K tokens once at startup instead of on every generation step makes the system fast enough (< 5 min for 11 prompts). |
| **NumPy masking instead of Python loops** | `masked[allowed_arr] = logits[allowed_arr]` is vectorized — looping over 150K logits in pure Python would be far slower. |
| **Encode once, append after** | Re-encoding the whole growing prompt string every step would be redundant. Token IDs are just appended to a list. |
| **String length cap (20 tokens)** | Prevents the small model from generating a string that never terminates. |
| **Skip AI call when only 1 token allowed** | Not just an optimization — also guarantees literal JSON scaffolding is byte-perfect. |
| **`vocab.get_token_text()` instead of `ai.decode()`** | `ai.decode` calls back into the HF tokenizer per token. The dict lookup is instant. |
| **Greedy decoding (`argmax`)** | Deterministic and reproducible. For structured output, always taking the highest-scoring valid token is the right call. |

---

## 9. Guarantees This Design Provides

- ✅ **100% valid JSON syntax** → forced-text controls every brace, quote, comma, and key literally
- ✅ **100% schema compliance** → `EXPECT_NUMBER_VALUE` only allows digits, `EXPECT_STRING_VALUE` only string-safe tokens, etc.
- ✅ **Function name always real** → `EXPECT_FUNCTION_NAME` only allows tokens that are valid prefixes of a real function name
- ⚠️ **NOT guaranteed:** the *correctness* of which function is chosen or what the values actually are — that's still up to the model's judgment. The cage guarantees **shape**, not **semantic correctness**.

---

## 10. Evaluator Q&A (25+ Questions)

**Q: What is function calling?**
A: It translates a natural-language request into a structured, machine-executable call (function name + typed arguments) instead of a prose answer, so external systems can act on it.

**Q: What is constrained decoding, in your own words?**
A: At every generation step, before picking the next token, you compute which tokens keep the output valid, set every other token's logit to negative infinity, and only then pick the highest-scoring token. It moves the guarantee from "hope the model behaves" to "make it impossible to misbehave."

**Q: Why a state machine instead of regex or retrying until valid JSON?**
A: A state machine gives a cheap, always-current answer to "what's legal right now" before every single token. Retry-until-valid doesn't scale (a 0.6B model may rarely get it right) and doesn't guarantee anything.

**Q: Walk me through what happens when I run `uv run python -m src`.**
A: (Use the 7 Steps in Section 3, then drill into the trace in Section 7 if they ask for more detail.)

**Q: What if `functions_definition.json` is missing or malformed?**
A: `registry.load()` will raise `FileNotFoundError` or `json.JSONDecodeError`, or Pydantic raises `ValidationError` if an entry doesn't match the schema.

**Q: Why does VocabManager build all lists in a single pass?**
A: The vocab is 150,000+ tokens. Looping once and sorting each token is far cheaper than doing 7 separate passes or re-scanning on every generated token.

**Q: What's `Ġ` and `Ċ`?**
A: BPE tokenizer artifacts marking "this token starts with a space" (`Ġ`) or "this token is a newline" (`Ċ`). `swap_tokens()` converts them back to real characters.

**Q: Why cap strings at 20 tokens?**
A: To prevent the model from looping/repeating a pattern indefinitely inside a string value.

**Q: What happens with zero parameters?**
A: `params_queue` is empty. `setup_next_parameter` sees the empty queue and forces `, "parameters": {}}` directly to DONE.

**Q: What about boolean parameters?**
A: `setup_next_parameter` detects `p_type == "boolean"`, forces the key without a trailing quote, and sets state to `EXPECT_BOOLEAN_VALUE` where only true/false tokens are allowed.

**Q: Why `ai.get_logits_from_input_ids` instead of a `generate()` method?**
A: You need the raw logits *before* any sampling happens so you can mask them. A black-box `generate()` wouldn't expose that intervention point.

**Q: Is `np.argmax` greedy decoding?**
A: Yes, fully greedy/deterministic. Same prompt always produces the same output. For reliable structured output, that's exactly what you want.

**Q: What's the difference between `ai.encode`/`ai.decode` and VocabManager?**
A: `ai.encode/decode` go through the HF tokenizer per call. VocabManager pre-computes a static `ID→text` dictionary once at startup, so lookups are just `dict.get()`.

**Q: How do you handle regex parameters?**
A: The FSM doesn't know what regex is. It sees `"type": "string"` and switches to `EXPECT_STRING_VALUE`. The AI fills in the actual pattern. Regex is just a normal string to our code.

**Q: What's `max_tokens = 200`?**
A: A safety ceiling so that if some edge case prevents the FSM from reaching DONE, the program still terminates instead of hanging forever.

**Q: Why not use `dspy`, `outlines`, `transformers`?**
A: The subject explicitly forbids them. The entire point is to build constrained decoding yourself from raw logits/tokens.

**Q: Where does Pydantic get used?**
A: All 4 data classes in `models.py`, plus `FunctionRegistry` and `VocabManager` inherit from `BaseModel`. It's a subject requirement.

**Q: What's the Makefile for?**
A: `install` → `uv sync`; `run` → `uv run python -m src`; `debug` → runs with pdb; `clean` → removes caches; `lint` → flake8 + mypy; `lint-strict` → full `--strict` mypy.

**Q: How would you add support for a new type like `"array"`?**
A: Add a new `EXPECT_ARRAY_VALUE` state, add token category lists in VocabManager (open/close bracket), extend `setup_next_parameter` with a branch for `"array"`, and add the state handling in `get_allowed_tokens` and `commit`.

**Q: What if two functions share a prefix (e.g. `fn_add` and `fn_add_v2`)?**
A: Both stay valid candidates as long as the typed text is a prefix of either. The model's own token choices disambiguate. The FSM only prevents invalid choices.

**Q: What's `fn_candidates` for?**
A: A pre-filtered dictionary of only tokens whose characters appear in valid function names. Instead of checking all 151K tokens during function name generation, we only check a few hundred.

**Q: Why does post-processing add a `}`?**
A: When the last parameter is a number/boolean, the AI types `3}` — that closes the inner `parameters` object. The outer JSON object still needs its own `}`.

**Q: What about multiple models?**
A: `Small_LLM_Model.__init__` already accepts a `model_name` parameter. You could add a `--model` CLI flag and pass it through. The rest of the pipeline is model-agnostic.

---

## 11. Things to Be Honest About if Pressed

Be upfront — evaluators probe for "did you actually understand it" over "does it look polished":

- The current `main()` does **not** wrap file loading in `try/except`. If a file is missing, the program crashes with a raw traceback instead of a friendly message. Good honest answer: *"That's a real gap — I would wrap `registry.load()` and the input file open in a try/except that prints a clear message and exits cleanly."*

- `ai.decode()` is available but not used anywhere (we use `VocabManager.get_token_text()` instead). This is intentional for performance. But `ai.encode` **is** still used (for the prompt and forced text), so the "recode the tokenizer" bonus is only partially satisfied.

- The system is greedy/deterministic (`argmax`). If the model's top choice is wrong, there's no retry or backtracking mechanism.

---

## 12. Quick Self-Test Before Your Defense

Try to answer these out loud, from memory:

1. Name the five files in `src/` and one sentence on each.
2. What are the 5 FSM states?
3. What are the 4 FSM methods and their nicknames?
4. What are the 7 token categories cached in VocabManager?
5. Trace "Greet shrek" through the FSM the way Section 7 traced the addition example.
6. What are the 5 lines of the constrained decoding algorithm and what does each do?
7. What's the one thing not fully compliant with the "never crash" requirement, and how would you fix it in 5 minutes live?

If you can do all seven without hesitation, you're ready.
