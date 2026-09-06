# Call Me Maybe — Peer Evaluation Defense Guide

## What Is This Project?

This project takes a **natural language question** (like "What is 2 plus 3?") and generates a **structured JSON function call** (like `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`).

We do NOT execute the function. We just generate the JSON that **asks** to call it.

The key challenge: LLMs output free text. They can hallucinate, invent fake functions, or produce broken JSON. We solve this using **Constrained Decoding** — a technique where we limit which tokens the AI is allowed to pick at every single step, guaranteeing the output is always valid JSON with real function names.

---

## The 7 Steps From Start to Finish

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

## File Map — What Each File Does

### `src/models.py` — The Blueprints (4 classes)
**Role:** Defines the shape of our data. Pydantic validates that data matches these shapes.

| Class | What It Validates | Fields |
|---|---|---|
| `PromptInput` | Each input question | `prompt: str` |
| `ParameterDef` | A single parameter definition | `type: str` |
| `FunctionDef` | A complete function definition | `name, description, parameters, returns` |
| `FunctionCall` | The final output format | `prompt, name, parameters` |

**No methods.** These are pure data shapes. Pydantic does all the work automatically.

---

### `src/registry.py` — The Rulebook (1 class, 5 methods)
**Role:** Loads the function definitions JSON file and provides lookup methods so the FSM can ask "What parameters does fn_add have?"

| Method | What It Does |
|---|---|
| `load(path)` | Opens JSON file, validates each function with Pydantic, stores them in a dict |
| `get_functions_name()` | Returns list of all function names: `["fn_add", "fn_greet", ...]` |
| `get_description(name)` | Returns description string for a function |
| `get_parameters(name)` | Returns parameter info: `{"a": {"type": "number"}, "b": {"type": "number"}}` |
| `get_parameter_names(name)` | Returns just the names: `["a", "b"]` |

---

### `src/vocab.py` — The Dictionary (1 function, 1 class)
**Role:** Loads `vocab.json` (151,000 tokens), cleans the text, flips the dictionary, and sorts tokens into cached lists in ONE single loop.

#### `swap_tokens(bpe_text)` (standalone function)
Replaces two special characters: `Ġ` → space, `Ċ` → newline. That's it.

#### `VocabManager.__init__()` — The Single Pass Loop
In one loop through 151,000 tokens, it does 4 things:
1. **Clean** each token with `swap_tokens()`
2. **Flip** the dictionary from `text→ID` to `ID→text`
3. **Sort** each token into the right cached list (numbers, strings, quotes, booleans, commas, close braces)
4. **Filter** tokens that could appear in function names (`fn_candidates`)

#### Getter Methods (7 total)
`get_token_text()`, `get_all_tokens()`, `get_fn_candidates()`, `get_number_tokens()`, `get_string_tokens()`, `get_quote_tokens()`, `get_boolean_tokens()`, `get_comma_tokens()`, `get_close_brace_tokens()`

All getters just return the pre-built list. Zero computation. Instant.

---

### `src/fsm.py` — The Traffic Cop (1 class, 4 methods)
**Role:** Controls which tokens the AI is allowed to pick at each step. Guarantees the output is valid JSON.

#### States
| State | What Happens |
|---|---|
| `EXPECT_FUNCTION_NAME` | AI picks tokens that spell a valid function name |
| `EXPECT_NUMBER_VALUE` | AI picks number tokens (digits, dot, minus) |
| `EXPECT_BOOLEAN_VALUE` | AI picks boolean tokens (true/false) |
| `EXPECT_STRING_VALUE` | AI picks string-safe tokens |
| `DONE` | Generation is complete |

#### The 4 Methods

| Method | Role | Called By |
|---|---|---|
| `get_allowed_tokens()` | **The Menu Giver.** Returns the list of legal token IDs. 100% read-only — never changes any variables. If forced tokens are queued, returns the first one. Otherwise, checks the current state and returns the matching cached list from vocab. | Main loop (before each token) |
| `commit(t_id, text)` | **The Note Taker.** Updates variables after a token is picked. If forcing, pops the used token. If the AI is typing, tracks progress and detects when a value is finished. | Main loop (after each token) |
| `setup_next_parameter(prefix, empty_text)` | **The Router.** Pops the next parameter from the queue, checks its type, and tells `force_text` to type the parameter key. If queue is empty, closes the JSON. | `commit()` (when a value finishes) |
| `force_text(text, next_state)` | **The Typist.** Encodes exact structural text into token IDs and queues them. Saves what state to jump to when the queue is empty. | `setup_next_parameter()` and `__init__()` |

#### Key Variables
| Variable | What It Tracks |
|---|---|
| `state` | Current FSM state |
| `next_state` | State to jump to after forced tokens finish |
| `forced_tokens` | Queue of token IDs being forced right now |
| `typed_name` | The function name the AI has typed so far (e.g. `"fn_ad"`) |
| `chosen_function` | The final complete function name (e.g. `"fn_add"`) |
| `fn_candidates` | Pre-filtered tokens that could appear in function names |
| `valid_functions` | List of all valid function names from registry |
| `params_queue` | Remaining parameter names to process (e.g. `["b"]`) |
| `string_length` | How many tokens the AI has typed for the current string |
| `string_limit` | Maximum tokens allowed per string (20) |

---

### `src/__main__.py` — The Orchestra Conductor (3 functions)
**Role:** Ties everything together. Loads files, initializes components, runs the generation loop, writes output.

#### `build_prompt(registry, question)`
Takes a user question and wraps it in a big instruction text that lists all available functions. This is what the AI reads to understand the task.

#### `generate_function_call(ai, vocab, registry, question)`
The main engine. For each question:
1. Build the prompt and encode it to token IDs
2. Create a fresh FSM
3. **The Loop:** Repeat until DONE:
   - Ask FSM: "What tokens are allowed?"
   - If only 1 allowed → use it directly (skip AI, saves time)
   - If multiple allowed → ask AI to score them, mask illegal ones to -infinity, pick highest
   - Look up the token's text, append to generated string
   - Tell FSM what was picked (`commit`)
4. Post-processing: add missing outer `}` if needed
5. Parse JSON and return the result dict

#### `main()`
The entry point:
1. Parse command line args (`argparse`)
2. Load function definitions into registry
3. Validate input prompts with Pydantic
4. Initialize AI model
5. Initialize VocabManager (single pass loop)
6. Loop through each prompt, call `generate_function_call`
7. Validate each result with `FunctionCall.model_validate()`
8. Write all results to output JSON file

---

## The Core Algorithm — Constrained Decoding

This is the heart of the entire project. In 5 lines:

```python
logits = np.array(ai.get_logits_from_input_ids(tokens))  # AI scores all 151K tokens
masked = np.full_like(logits, -np.inf)                    # Create array of -infinity
allowed_arr = np.array(allowed_tokens)                     # The FSM's approved list
masked[allowed_arr] = logits[allowed_arr]                  # Copy only approved scores
token_id = int(np.argmax(masked))                          # Pick the highest approved score
```

We don't change how the AI thinks. We just cross out all the illegal answers before picking the winner.

---

## Example Walkthrough: "What is 2 plus 3?"

| Step | FSM State | What Happens | Generated Text |
|---|---|---|---|
| 1 | (boot) | FSM forces `{"name": "` | `{"name": "` |
| 2 | `EXPECT_FUNCTION_NAME` | AI picks tokens to spell `fn_add_numbers"` | `{"name": "fn_add_numbers"` |
| 3 | (commit detects `"`) | FSM saves function, pops `"a"` from queue, forces `, "parameters": {"a": ` | `{"name": "fn_add_numbers", "parameters": {"a": ` |
| 4 | `EXPECT_NUMBER_VALUE` | AI picks `2`, then `,` | `{"name": "fn_add_numbers", "parameters": {"a": 2,` |
| 5 | (commit detects `,`) | FSM pops `"b"`, forces ` "b": ` | `... "a": 2, "b": ` |
| 6 | `EXPECT_NUMBER_VALUE` | AI picks `3`, then `}` | `... "b": 3}` |
| 7 | (commit detects `}`) | Queue empty → state = DONE | Loop ends |
| 8 | Post-processing | Adds missing outer `}` | `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}` |

---

## Quick Defense Answers

**Q: Why Pydantic?**
A: The subject requires it. Pydantic validates data automatically — if a field is missing or the wrong type, it raises an error instantly. We use it to validate input prompts, function definitions, and output format.

**Q: Why not just let the AI generate freely?**
A: The AI could hallucinate fake function names, produce broken JSON, or output random text. Constrained decoding guarantees 100% valid output by limiting the AI's choices at every single step.

**Q: What is the FSM?**
A: A Finite State Machine. It tracks where we are in building the JSON and tells the generation loop which tokens are legal at each step. It has 5 states and 4 methods.

**Q: How does VocabManager help performance?**
A: Instead of scanning 151,000 tokens every time the FSM needs to know which tokens are numbers, we pre-sort them once at startup into cached lists. The FSM just grabs the pre-built list instantly.

**Q: What does `fn_candidates` do?**
A: It's a tiny pre-filtered dictionary containing only tokens whose characters appear in valid function names. Instead of checking all 151K tokens during function name generation, we only check a few hundred.

**Q: How do you handle regex/string parameters?**
A: The FSM doesn't know what regex is. It just sees `"type": "string"` in the registry and switches to `EXPECT_STRING_VALUE`. The AI (which is smart enough to understand regex from the prompt) fills in the actual pattern. Our code treats regex exactly like any other string.

**Q: What is the 20-token string limit?**
A: A safety net. If the AI keeps typing string tokens forever, `string_length` counts each one. At 20, the FSM forces a closing quote `"` to stop the string. This prevents infinite loops.

**Q: Why does post-processing add a `}`?**
A: When the last parameter is a number/boolean, the AI types `3}` — that `}` closes the inner parameters object. But the outer JSON object still needs its own `}`. The post-processing adds it.
