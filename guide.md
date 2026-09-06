# Call Me Maybe — Full Project Study Guide

*Use this to rehearse your defense. Read it top to bottom once, then use the Q&A section as flashcards.*

---

## 1. The 30-second pitch (say this first)

> "This project turns natural-language questions into structured function calls — the kind an API or a program could actually execute. Instead of asking an LLM to write JSON and hoping it's valid, I force it to be valid: I use a small local model (Qwen3-0.6B) and, at every single token it generates, I block every token that would break the JSON schema. Only the tokens that keep the output syntactically and semantically correct are allowed to be picked. This is called **constrained decoding**, and it guarantees 100% valid, schema-compliant JSON — even though a 0.6-billion-parameter model would normally get this right maybe 30% of the time if you just prompted it."

Then, if asked "what did you build specifically":

> "I built four cooperating pieces: a **function registry** that loads and validates the available functions, a **vocabulary manager** that pre-classifies the model's 150,000+ tokens into categories (numbers, strings, quotes, commas, etc.), a **finite state machine (FSM)** that knows where we are inside the JSON we're building and tells the generator which token categories are legal right now, and a **generation loop** that ties it all together: ask the FSM what's allowed, ask the model to score only those tokens, pick the best one, and repeat until the JSON is complete."

---

## 2. What problem does this actually solve?

- LLMs are trained to produce fluent text, not machine-readable output.
- Small models (0.6B parameters here) are especially bad at spontaneously producing valid JSON — the subject PDF states they succeed as little as ~30% of the time with prompting alone.
- Production systems still need 99%+ reliability. The trick used everywhere in industry (and here) is **constrained decoding**: instead of trusting the model's free choice, you edit the probability landscape *before* the model picks a token, so it structurally cannot produce anything invalid.
- The task: given a prompt like *"What is the sum of 2 and 3?"*, output not the answer (`5`) but the **function call**: `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`.

---

## 3. The mindmap (mental map of the whole codebase)

```
Call_Me_Maybe/
│
├── llm_sdk/__init__.py        → Small_LLM_Model: talks to the actual neural network
│                                  (encode text, get next-token logits, locate vocab file)
│
├── data/input/
│   ├── functions_definition.json  → catalog of callable functions + typed parameters
│   └── function_calling_tests.json → the natural-language prompts to convert
│
└── src/                        → YOUR code (everything you're evaluated on)
    ├── models.py     → Pydantic schemas: validate every JSON blob that flows in/out
    ├── registry.py   → FunctionRegistry: loads functions_definition.json, answers
    │                    "what functions exist / what params does fn X take"
    ├── vocab.py      → VocabManager: loads the model's vocab.json ONCE, pre-sorts
    │                    every token into buckets (number / string / quote / comma /
    │                    close-brace / boolean / fn-name-candidate)
    ├── fsm.py        → JSONStateMachine: the "cage". Knows exactly where we are
    │                    inside the JSON grammar and returns the list of legal
    │                    next-token IDs at every step
    └── __main__.py   → orchestrator: CLI args, wires registry+vocab+FSM+model
                         together in the generation loop, writes the output file
```

**One-sentence role for every file:**

| File | Role |
|---|---|
| `llm_sdk/__init__.py` | The only file that touches PyTorch/HuggingFace — gives you tokens in, logits out. You must NOT touch its private members. |
| `src/models.py` | Defines the shape of every JSON object the program reads or writes, and validates it automatically (Pydantic). |
| `src/registry.py` | In-memory "database" of callable functions: names, descriptions, parameter names/types. |
| `src/vocab.py` | Turns the model's raw 150K-token vocabulary into fast lookup tables of *categories* of tokens. |
| `src/fsm.py` | The actual "constrainer" — a state machine that, given where we are in the JSON, says which token IDs are legal. |
| `src/__main__.py` | The conductor: reads CLI args and files, builds the prompt, runs the token-by-token generation loop, parses the result, writes output. |

---

## 4. File-by-file, method-by-method

### 4.1 `llm_sdk/__init__.py` — provided, read-only, do not modify

Class **`Small_LLM_Model`**

| Method | What it does | Why you need it |
|---|---|---|
| `__init__(model_name="Qwen/Qwen3-0.6B", ...)` | Picks a device (mps > cuda > cpu), loads the tokenizer and the causal LM from Hugging Face, puts the model in `eval()` mode and freezes gradients. | Sets up everything so inference is fast and deterministic (no training). |
| `encode(text) -> Tensor` | Runs the HF tokenizer on a string, returns a `[1, seq_len]` tensor of token IDs. | You use this to turn your prompt (and any "forced" literal text like `'{"name": "'`) into token IDs. |
| `decode(ids) -> str` | Inverse of encode: token IDs → text, skipping special tokens. | Not actually used in the generation loop (the project decodes manually via `VocabManager` instead, see design decision below) — but useful for debugging. |
| `get_logits_from_input_ids(input_ids) -> List[float]` | Feeds the whole sequence through the model, returns the raw (un-softmaxed) logits for the **next** token only. | This is the "brain": one call = one probability distribution over the whole vocabulary for what comes next. |
| `get_path_to_vocab_file() -> str` | Downloads/locates `vocab.json` from the Hugging Face hub cache and returns its path. | Lets `VocabManager` read the token_id ↔ token_text mapping without hardcoding it. |
| `get_path_to_merges_file()` / `get_path_to_tokenizer_file()` | Same idea for BPE merges / full tokenizer.json. | Not used in this implementation, but available if you needed raw BPE logic yourself. |

**Key exam point:** the subject explicitly forbids using *private* methods/attributes of `llm_sdk` (anything starting with `_`, like `self._model` or `self._tokenizer`). Everything you use goes through the four public methods above.

---

### 4.2 `src/models.py` — Pydantic schemas

| Class | Fields | Purpose |
|---|---|---|
| `PromptInput` | `prompt: str` | Validates one entry of `function_calling_tests.json`. If a test entry is missing `"prompt"` or it's not a string, Pydantic raises a clear validation error instead of crashing with a vague `KeyError`. |
| `ParameterDef` | `type: str` | Validates one parameter's type declaration, e.g. `{"type": "number"}`. |
| `FunctionDef` | `name: str`, `description: str`, `parameters: Dict[str, ParameterDef]`, `returns: ParameterDef` | Validates one full entry of `functions_definition.json`. |
| `FunctionCall` | `prompt: str`, `name: str`, `parameters: Dict[str, Any]` | Validates one **output** object before it's written to the results file — a final sanity check that the shape matches the spec exactly. |

**Why Pydantic and not just `dict`s?** The subject *requires* it ("All classes must use pydantic for validation"), and practically: it turns malformed input files into readable errors early, instead of the program crashing deep inside the generation loop.

---

### 4.3 `src/registry.py` — `FunctionRegistry(BaseModel)`

State: `functions: Dict[str, FunctionDef]` — maps function name → validated definition.

| Method | What it does |
|---|---|
| `load(path)` | Opens the JSON file, parses it, and for every entry runs `FunctionDef.model_validate(fn)` (Pydantic validation), then stores it keyed by name. Raises `FileNotFoundError` / `json.JSONDecodeError` naturally if the file is missing or broken — these are caught in `__main__.py`. |
| `get_functions_name()` | Returns `list(self.functions.keys())`, e.g. `['fn_add_numbers', 'fn_greet', ...]`. Used to (a) list functions in the prompt text, and (b) tell the FSM/vocab which function names are legal. |
| `get_description(name)` | Returns the human-readable description of one function (used to build the prompt so the LLM understands what each function does). |
| `get_parameters(name)` | Returns `{"a": {"type": "number"}, ...}` — used both to write the prompt and, critically, by the FSM to know **what type** the current parameter should be (number/string/boolean) so it can pick the right token category. |
| `get_parameter_names(name)` | Returns just `["a", "b"]`, the parameter names **in the order they appear in the schema**. This order is what the FSM iterates through — see design decision "Forced Parameter Order" below. |

---

### 4.4 `src/vocab.py` — `VocabManager(BaseModel)`

This is the *"understand the model's language"* module. It's built once at startup and never changes during generation (all lists are cached).

Free function:
- `swap_tokens(bpe_text) -> str` — the tokenizer's `vocab.json` stores tokens in raw BPE form, where a leading space is written as the special character `Ġ` and a newline as `Ċ`. This function just does `.replace("Ġ", " ").replace("Ċ", "\n")` to turn a raw BPE token like `"Ġhello"` into readable `" hello"`. This is a documented "tokenizer quirk" the README calls out.

State (all built in one single pass through the vocab in `__init__`):
- `decoded_vocab: Dict[token_id, text]` — the full vocabulary, every token decoded to readable text.
- `number_tokens`, `string_tokens`, `quote_tokens`, `boolean_tokens`, `comma_tokens`, `close_brace_tokens` — pre-filtered lists of token IDs belonging to each category.
- `fn_candidates: Dict[token_id, text]` — tokens made only of characters that appear somewhere in a valid function name (plus `"`), i.e. tokens that *could* plausibly be part of building a function name character-by-character.

| Method | What it does |
|---|---|
| `__init__(vocab_path, fn_names=[])` | Reads `vocab.json` (`{bpe_token_string: token_id}`), and for **every single token** (150K+): decodes it via `swap_tokens`, stores it in `decoded_vocab`, then classifies it into zero or more of the category lists below, using simple character-set checks. Also builds `fn_candidates` in this same pass — no second loop needed. |
| `get_token_text(token_id)` | Instant dict lookup — this is what replaces calling the (slow) `ai.decode()` after every generated token. |
| `get_all_tokens()` | Returns the full `decoded_vocab` dict. |
| `get_fn_candidates()` | Returns tokens that could legally extend a function name currently being typed. |
| `get_number_tokens()` | Digits `0-9`, `.`, `-` only (e.g. `"42"`, `"-"`, `"."`). |
| `get_string_tokens()` | Any token with no `"` character and no control characters (safe inside a JSON string). |
| `get_quote_tokens()` | Tokens that are exactly `"`. |
| `get_boolean_tokens()` | Tokens that are a prefix of, or match, `"true"`/`"false"` (with optional trailing `,`/`}`). |
| `get_comma_tokens()` | Tokens made only of `,` and whitespace. |
| `get_close_brace_tokens()` | Tokens made only of `}` and whitespace. |

**Classification logic, in plain words**, for each raw token text `stripped = text.strip()`:
- **Number**: every character in `"0123456789.-"`.
- **Quote**: exactly `"`.
- **Comma**: contains `,` and every char is whitespace or `,`.
- **Close brace**: contains `}` and every char is whitespace or `}`.
- **Boolean**: `stripped` is a prefix of, or has as a prefix, one of `["true","false","true,","false,","true}","false}"]` — this lets multi-token boolean words be built up correctly token-by-token.
- **String**: no `"` and no ASCII control character (`ord(c) < 32`).
- **Function-name candidate**: every character of the token is inside the set of characters that appear in any valid function name (plus the closing `"`).

---

### 4.5 `src/fsm.py` — `JSONStateMachine`

This is the heart of the constrained decoding "cage." It never generates text itself — it only answers *"which token IDs are legal right now?"* and updates its internal state once told which token was chosen.

**States:** `EXPECT_FUNCTION_NAME → EXPECT_NUMBER_VALUE / EXPECT_STRING_VALUE / EXPECT_BOOLEAN_VALUE → ... → DONE`

**Key design idea — "forced text":** Large chunks of the JSON skeleton (`{"name": "`, `", "parameters": {"a": `, closing braces, etc.) are 100% predictable — they don't need the AI at all. The FSM pre-encodes these literal strings into token IDs with `force_text()` and just replays them one at a time. The AI is only actually consulted when there's a genuine choice to make (which function? which digit? which letter of a string?).

| Method | What it does |
|---|---|
| `__init__(ai, vocab, registry)` | Stores references, initializes `state=""`, `valid_functions` (all function names), `fn_candidates` (from vocab), an empty `params_queue`, string-length tracking (`string_length`, `string_limit=20`). Immediately calls `force_text('{"name": "', "EXPECT_FUNCTION_NAME")` to kick things off — generation always starts by forcing the literal opening of the JSON object. |
| `force_text(text, next_state)` | Encodes `text` into token IDs via `ai.encode`, stores them as a queue in `self.forced_tokens`, and remembers what state to switch to once the queue is exhausted. |
| `setup_next_parameter(prefix, empty_text)` | Called whenever we finish one parameter and need to start the next (or close the object). Pops the next parameter name off `params_queue`. If the queue is empty: either force `empty_text` (e.g. closing braces) then go to `DONE`, or go straight to `DONE`. Otherwise, looks up that parameter's type via the registry and forces the right JSON key text (`"paramName": "` for strings, `"paramName": ` for numbers/booleans) with the matching next state. |
| `get_allowed_tokens()` | **Read-only.** Rule 1: if there are queued forced tokens, only the *next* forced token ID is "allowed" (i.e. the FSM overrides the model completely). Rule 2: otherwise, branch on `self.state`: <br>• `EXPECT_FUNCTION_NAME`: for every candidate token, check if appending it to the text typed so far is still a valid *prefix* of some real function name (+ closing quote) — only those survive.<br>• `EXPECT_NUMBER_VALUE`: all number tokens, plus close-brace tokens if this is the last parameter, or comma tokens otherwise.<br>• `EXPECT_BOOLEAN_VALUE`: all boolean tokens.<br>• `EXPECT_STRING_VALUE`: if we've hit the 20-token string cap, only allow closing quote; otherwise allow string tokens + quote token. |
| `commit(t_id, text)` | **The only method that mutates state.** If we were replaying forced tokens: pop one off the queue; if the queue just became empty, switch to `next_state`. Otherwise (real AI choice): <br>• In `EXPECT_FUNCTION_NAME`: append the token text to `typed_name`; if it now ends in `"`, the function name is complete — store `chosen_function`, load its parameter names into `params_queue`, and call `setup_next_parameter` to open the `"parameters": {` block (or close immediately with `{}}` if the function takes no parameters).<br>• In `EXPECT_NUMBER_VALUE`/`EXPECT_BOOLEAN_VALUE`: if the terminator character (`}` if last param, else `,`) appears in the generated text, move to the next parameter.<br>• In `EXPECT_STRING_VALUE`: increment `string_length`; if the token contains `"`, the string just closed — reset the counter and move to the next parameter (forcing `}}` if this was the last one). |

**Why this matters for the "100% valid JSON" guarantee:** at every single step, the FSM's `get_allowed_tokens()` never returns a token that would break either (a) JSON syntax, or (b) the specific parameter's declared type. `np.argmax` in `__main__.py` can only ever pick from that pre-filtered set.

---

### 4.6 `src/__main__.py` — the orchestrator

| Function | What it does |
|---|---|
| `build_prompt(registry, user_question)` | Builds the natural-language prompt: an instruction line, then for each function its name/description/parameters (as JSON), then the user's question and a "Call the correct function in JSON format" instruction. This is what actually goes into the model so it has context on what functions exist. |
| `generate_function_call(ai, vocab, registry, question)` | The generation loop, described step by step below. |
| `main()` | CLI parsing (`argparse` for `--functions_definition`, `--input`, `--output` with sensible defaults under `data/input/` and `data/output/`), loads and Pydantic-validates the function definitions and the test prompts, instantiates `Small_LLM_Model` and `VocabManager`, loops over every prompt calling `generate_function_call`, validates each result against `FunctionCall`, and finally writes the full results array to the output JSON file (creating the output directory if needed). |

**`generate_function_call` step by step (this is what you must be able to narrate live):**

1. Build the prompt text and encode it once into a list of token IDs (`ai.encode(...).squeeze().tolist()`). Encoding is done once; from here on new tokens are just appended (an explicit "Design Decision" in the README — avoids re-encoding the growing string every step).
2. Create a fresh `JSONStateMachine` — this immediately queues the forced tokens for `{"name": "`.
3. Loop while `fsm.state != "DONE"` and under a 200-token safety cap (`max_tokens`):
   a. Ask `fsm.get_allowed_tokens()`.
   b. **If exactly one token is allowed** → it's forced, use it directly, **skip calling the model** (this is a real performance/design decision, not just a state machine detail).
   c. **If multiple tokens are allowed** → call `ai.get_logits_from_input_ids(tokens)` to get real logits from the neural network, build a `-inf`-filled array the same size as the vocab, copy back in only the logits at the allowed indices, and take `np.argmax` of that masked array. This is the actual "cage": everything not allowed becomes mathematically impossible to select because its score is negative infinity.
   d. Look up the chosen token's text via `vocab.get_token_text(token_id)` (not `ai.decode` — cheaper, and it's the cached table built earlier).
   e. Append the token both to the running generated string and to the running token-ID list (for the next logits call), print it live, and call `fsm.commit(token_id, word)` to advance the state machine.
4. After the loop: post-process the string — if it ends in a single `}` (closing only the inner `parameters` object) rather than `}}`, append one more `}` to close the outer object too.
5. `json.loads()` the result. On success, return `{"prompt", "name", "parameters"}`. On `JSONDecodeError` (should basically never happen given the cage, but it's defensive), print a warning and fall back to `{"name": "fn_not_found", "parameters": {}}`.

---

## 5. The full step-by-step example (memorize this trace)

Prompt: **"What is the sum of 2 and 3?"**

1. `build_prompt` lists all functions (`fn_add_numbers`, `fn_greet`, ...) with descriptions/params, then appends the user question.
2. FSM starts by forcing `{"name": "` — no AI call, these tokens are just replayed.
3. State → `EXPECT_FUNCTION_NAME`. FSM computes which `fn_candidates` tokens are valid prefixes of a real function name. Model logits are requested, masked to only those candidates, argmax picks e.g. `"fn_add"`, then continues character by character (`_numbers`), until the accumulated `typed_name` ends in `"` → `chosen_function = "fn_add_numbers"`.
4. `commit` loads `params_queue = ["a", "b"]`, calls `setup_next_parameter(', "parameters": {', ...)`, which — since parameter `a` is type `number` — forces the literal text `, "parameters": {"a": ` and sets state to `EXPECT_NUMBER_VALUE`.
5. State → `EXPECT_NUMBER_VALUE`. Allowed tokens = digits/`.`/`-` plus comma tokens (since `b` is still queued). Model picks `"2"`, then a comma-containing token → triggers `setup_next_parameter(" ", "")` for `b`, forcing `"b": `.
6. State → `EXPECT_NUMBER_VALUE` again, but now `params_queue` is empty, so allowed tokens = digits plus close-brace tokens. Model picks `"3"`, then a token containing `}` → `setup_next_parameter(" ", "")` finds an empty queue and no `empty_text` → state becomes `DONE` directly.
7. Loop ends. Generated string ≈ `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}` — ends in a single `}`, so `__main__.py` appends one more `}`.
8. `json.loads` succeeds → `{"prompt": "...", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}`.

---

## 6. Design decisions — the "why," not just the "what"

| Decision | Why it matters |
|---|---|
| **Forced parameter order** | Instead of letting the model choose which parameter to fill next, the FSM always follows the exact order from `functions_definition.json`. This removes an entire class of ambiguity/bugs and massively simplifies the FSM (no need to track "which params are already filled" as a set — just a queue). |
| **Cached vocabulary filters** | Classifying 150K tokens is expensive; doing it in one pass at startup instead of on every generation step (or worse, on every single one of possibly thousands of tokens across many prompts) is what makes the system fast enough (<5 min for 11 prompts). |
| **NumPy masking instead of Python loops** | `masked[allowed_arr] = logits[allowed_arr]` is a vectorized operation; looping over 150K logits in pure Python per token would be far slower. |
| **Encode once, append after** | Re-encoding the whole growing prompt string every step would be redundant work; token IDs are just appended to a Python list. |
| **String length cap (20 tokens)** | Prevents the (unreliable, tiny) model from generating a string parameter that never terminates. |
| **Skip the AI call when only one token is allowed** | Not just an optimization — it also guarantees the literal JSON scaffolding (`{"name": "`, closing braces, etc.) is always byte-perfect, since the model never even gets a chance to deviate from it. |
| **Using `vocab.get_token_text()` instead of `ai.decode()`** | `ai.decode` would call back into the (slower) HF tokenizer machinery per token; the dict lookup is instant and was already built. |

---

## 7. Guarantees this design provides — and where they come from

- **100% valid JSON syntax** → the FSM's forced-text mechanism controls every brace, quote, comma, and key literally; free-choice tokens are always masked down to the categories that keep the JSON well-formed.
- **100% schema compliance (correct types)** → `EXPECT_NUMBER_VALUE` only ever allows digit/`.`/`-` tokens, `EXPECT_STRING_VALUE` only string-safe tokens, `EXPECT_BOOLEAN_VALUE` only true/false-prefix tokens — so a `"number"`-typed parameter genuinely cannot become text, and vice versa.
- **Function name is always one of the real, defined functions** → `EXPECT_FUNCTION_NAME` allows a token only if appending it keeps the typed-so-far string a valid prefix of *some* real function name.
- **What's NOT 100% guaranteed**: the *correctness* of which function is chosen and what the numeric/string values actually are — that's still up to the (small, 0.6B) model's judgment, guided only by logits, not the cage. The cage guarantees *shape*, not *semantic correctness*. The README is explicit that complex multi-parameter string functions (e.g. regex) may produce imperfect values because of the small model size.

---

## 8. Likely evaluation questions and how to answer them

**Q: What is function calling and why doesn't the model just answer the question directly?**
A: Function calling translates a natural-language request into a structured, machine-executable call (function name + typed arguments) instead of a prose answer, so external systems/APIs can act on it.

**Q: What is constrained decoding, in your own words?**
A: At every generation step, before picking the next token, you compute which tokens would keep the output both syntactically valid and schema-compliant, set every other token's logit to negative infinity, and only then pick the highest-scoring token (`argmax`). It moves the guarantee of correctness from "hope the model behaves" to "make it structurally impossible to misbehave."

**Q: Why use a state machine instead of, say, regex or just retrying until valid JSON appears?**
A: A state machine gives an explicit, cheap, always-current answer to "what's legal right now," which is exactly what you need before every single token — it's essentially the finite automaton for the JSON grammar restricted further by whatever type the current parameter needs. Retry-until-valid doesn't scale (a 0.6B model may rarely get it right) and doesn't guarantee anything.

**Q: Walk me through what happens when I run `uv run python -m src`.**
A: (Use the Section 5 trace, but at a higher level first, then drill down if asked.)

**Q: What happens if `functions_definition.json` is missing or malformed?**
A: `FunctionRegistry.load` will raise `FileNotFoundError` or `json.JSONDecodeError` naturally when trying to open/parse the file, or a Pydantic `ValidationError` if an entry doesn't match `FunctionDef`'s schema. (Be ready to discuss whether/where you'd want to add a `try/except` with a friendly message — the subject asks for graceful error handling; check if the current code wraps these calls with try/except in `main()` — as written, it does not, so mention this as something you'd wrap for full robustness if asked "is this fully compliant with the graceful-error-handling requirement?")

**Q: Why does `VocabManager` build all its lists in a single pass instead of iterating the vocab multiple times?**
A: Performance — the vocab is 150,000+ tokens; looping over it once (`O(n)`) and sorting each token into whichever buckets it belongs to is far cheaper than doing 6 separate passes (one per category) or worse, re-scanning it on every generated token.

**Q: What's `Ġ` and `Ċ`, and why do you need to handle them?**
A: They're BPE tokenizer artifacts marking "this token starts with a space" (`Ġ`) or "this token is a newline" (`Ċ`). `swap_tokens()` converts them back to a real space/newline so the accumulated generated text reads correctly and so the string/number classification logic works on the real characters, not the encoded markers.

**Q: Why cap string generation at 20 tokens?**
A: To prevent the model from looping/repeating a pattern indefinitely inside a string value — a documented failure mode in the README ("Infinite String Generation").

**Q: What would happen if a function had zero parameters?**
A: In `commit()`, after the function name closes, `params_queue` would be empty; `setup_next_parameter(', "parameters": {', ', "parameters": {}}')` sees the empty queue and forces the literal `', "parameters": {}}'` directly to `DONE` — no AI calls needed for the parameters section at all.

**Q: What would happen with a boolean parameter?**
A: `setup_next_parameter` detects `p_type == "boolean"`, forces the key text without a trailing quote (`'"paramName": '`), and sets state to `EXPECT_BOOLEAN_VALUE`, where only tokens that are (prefixes of) `true`/`false` (optionally followed by `,`/`}`) are allowed.

**Q: Why call `ai.get_logits_from_input_ids` instead of some higher-level `generate()` method?**
A: You need the raw logits *before* any sampling happens, so you can mask them yourself — the whole point of constrained decoding is intervening between "model scores tokens" and "a token gets picked," which a black-box `generate()` call wouldn't expose.

**Q: Is `np.argmax` "greedy decoding"? Why not sample randomly?**
A: Yes, it's fully greedy/deterministic. Given the goal is 100% reliable, reproducible structured output (not creative variety), always taking the highest-scoring valid token is the right call — determinism also makes debugging and testing much easier.

**Q: What's the difference between the model's tokenizer `encode`/`decode` and what `VocabManager` does?**
A: `ai.encode`/`ai.decode` go through the real HF tokenizer object at runtime, per call. `VocabManager` instead pre-computes a static dictionary (`token_id -> text`) for the *entire* vocabulary once at startup by reading `vocab.json` directly, so later lookups are just `dict.get()` — no repeated tokenizer overhead, and it also lets you pre-classify tokens into number/string/etc. buckets, which the tokenizer API doesn't give you directly.

**Q: What are the Pydantic classes for, concretely?**
A: They validate every JSON boundary in the program: incoming test prompts (`PromptInput`), incoming function definitions (`ParameterDef`/`FunctionDef`), and outgoing results (`FunctionCall`) — catching malformed data with clear errors instead of the program crashing or silently producing bad output deep inside the generation logic.

**Q: How would you add support for a new parameter type, e.g. `"array"`?**
A: You'd (1) add handling in `FunctionRegistry`/`ParameterDef` if needed (already generic via `type: str`), (2) add a new `EXPECT_ARRAY_VALUE` state and a `setup_next_parameter` branch that forces `[` and picks the state, (3) add corresponding token-category lists in `VocabManager` (e.g. `get_open_bracket_tokens`, `get_close_bracket_tokens`), (4) extend `get_allowed_tokens` and `commit` in the FSM for that state. This is a good answer for the "small live modification" part of the evaluation.

**Q: What happens if two functions share a name prefix (e.g. `fn_add_numbers` and `fn_add_numbers_v2`)?**
A: During `EXPECT_FUNCTION_NAME`, both would remain valid candidates as long as the typed-so-far text is a prefix of either — the FSM keeps both alive in `get_allowed_tokens()` (via `(fn + '"').startswith(potential)`) until the model's own token choices disambiguate which one it's actually building, character by character; the model is still responsible for the *choice*, the cage only prevents an invalid choice.

**Q: How do you guarantee the output is never malformed at all, even in edge cases?**
A: Structurally, the FSM never allows a syntax-breaking token to be chosen. As a last defensive layer, `__main__.py` also fixes a known asymmetry (single `}` vs `}}` at the very end) and wraps the final parse in `try/except json.JSONDecodeError`, falling back to a safe `fn_not_found` result rather than crashing.

**Q: What's `max_tokens = 200` for?**
A: A hard safety ceiling on the generation loop so that if some edge case caused the FSM to never reach `DONE` (a bug, or a pathological input), the program still terminates instead of hanging forever.

**Q: Why is this project not using `dspy`, `outlines`, `transformers`' built-in constrained generation, etc.?**
A: The subject explicitly forbids those — the entire point of the exercise is to build the constrained-decoding mechanism (vocabulary classification + FSM + logit masking) yourself, from the raw logits/tokens the `llm_sdk` gives you, rather than relying on a library that already solves it.

**Q: Where does `pydantic` get used and why is it required?**
A: All four data-shape classes in `models.py`, plus `FunctionRegistry` and `VocabManager` themselves inherit from `pydantic.BaseModel`. It's a project requirement ("All classes must use pydantic for validation") and it gives you runtime type validation with almost no boilerplate.

**Q: What's the Makefile for and what does each rule do?**
A: `install` → `uv sync` (installs dependencies from `pyproject.toml`/`uv.lock`); `run` → `uv run python -m src` (executes the program with defaults); `debug` → runs the same entry point through Python's `pdb` debugger; `clean` → removes `__pycache__`, `.mypy_cache`, `.pytest_cache` and any stray `.pyc` files; `lint` → runs `flake8` (style) and `mypy` with a specific strict-ish flag set (untyped defs disallowed, etc.); `lint-strict` → the same but with mypy's full `--strict` mode.

**Q: Where would you add a new bonus feature, e.g. supporting a second model?**
A: `Small_LLM_Model.__init__` already accepts a `model_name` parameter, so `main()` could expose a `--model` CLI flag and pass it straight through — the rest of the pipeline (`VocabManager`, `JSONStateMachine`) is model-agnostic as long as the new model exposes the same four SDK methods.

---

## 9. Things you should be ready to be honest about if pressed

Be upfront and confident about these — the evaluators specifically probe for "did you actually understand it" over "does it look polished":

- The current `main()` does **not** wrap file loading (`registry.load`, opening the tests file) in `try/except` — if a file is missing or malformed, the program will raise an uncaught exception and crash rather than printing a graceful message, which is technically a gap against the "must never crash unexpectedly" general rule. Good, confident answer if asked: *"That's a real gap — I would wrap the `registry.load()` call and the input-file `open()`/`json.load()` in a try/except that prints a clear message and exits cleanly, rather than letting a raw traceback surface."*
- `ai.decode()` exists in the SDK but isn't used anywhere in `src/` (the project uses `VocabManager.get_token_text()` instead) — this is intentional per the "Encode Once" / caching design decision, and also aligns with a bonus objective ("avoiding direct use of encode and decode in the main code, instead using get_logits_from_input_ids and get_path_to_vocab_file") — though note `ai.encode` *is* still used (for the prompt and for forced text), so this bonus is only partially satisfied.
- The system is greedy/deterministic (`argmax`), not probabilistic — good for reliability, but it means the same prompt always produces the same output, and if the model's top choice for a free-choice slot is "wrong" (e.g. picks the wrong function), there's no retry/backtracking mechanism.

---

## 10. Quick self-test before your defense

Try to answer these out loud, from memory, without looking:

1. Name the five files in `src/` and one sentence on each.
2. What are the five FSM states and what triggers a transition between them?
3. What are the six token categories cached in `VocabManager`?
4. Trace "Greet shrek" through the FSM the way Section 5 traced the addition example.
5. Why is `np.argmax` applied to a masked array instead of the raw logits?
6. What's the one thing in this codebase that isn't fully compliant with the "never crash" requirement, and how would you fix it in under 5 minutes live?

If you can do all six without hesitation, you're ready.