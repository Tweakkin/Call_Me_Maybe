"""Finite State Machine for constrained JSON generation.

Tracks where we are in building a JSON function call and tells the
generation loop which tokens are valid at each step.

States:
    EXPECT_FUNCTION_NAME -> AI picks tokens to build a function name
    EXPECT_NUMBER_VALUE  -> AI picks number tokens
    EXPECT_BOOLEAN_VALUE -> AI picks boolean tokens
    EXPECT_STRING_VALUE  -> AI picks string tokens
    DONE                 -> Generation complete
"""
from typing import Any, Dict, List

from src.vocab import VocabManager
from src.registry import FunctionRegistry


class JSONStateMachine:
    """Tracks position in JSON and returns allowed tokens at each step."""

    def __init__(
        self,
        ai: Any,
        vocab: VocabManager,
        registry: FunctionRegistry,
    ) -> None:
        self.ai = ai
        self.vocab = vocab
        self.registry = registry

        self.state: str = ""
        self.next_state: str = ""
        self.forced_tokens: List[int] = []

        self.chosen_function: str = ""
        self.valid_functions: List[str] = registry.get_functions_name()
        self.typed_name: str = ""
        self.fn_candidates: Dict[int, str] = vocab.get_fn_candidates()

        self.params_queue: List[str] = []
        self.string_length: int = 0
        self.string_limit: int = 20

        # Boot up the FSM by forcing the JSON opening text
        self.force_text('{"name": "', "EXPECT_FUNCTION_NAME")

    def force_text(self, text: str, next_state: str) -> None:
        """Encode exact text into Token IDs and queue them up."""
        encoded = self.ai.encode(text)
        self.forced_tokens = encoded[0].tolist()
        self.next_state = next_state

    def setup_next_parameter(self, prefix: str, empty_text: str) -> None:
        """Set up the next parameter, or close the JSON if none are left."""
        if len(self.params_queue) == 0:
            if empty_text:
                self.force_text(empty_text, "DONE")
            else:
                self.state = "DONE"
            return

        p_name = self.params_queue.pop(0)
        params_info = self.registry.get_parameters(self.chosen_function)
        p_type = params_info[p_name]["type"]

        if p_type == "string":
            self.force_text(f'{prefix}"{p_name}": "', "EXPECT_STRING_VALUE")
        elif p_type == "boolean":
            self.force_text(f'{prefix}"{p_name}": ', "EXPECT_BOOLEAN_VALUE")
        else:
            self.force_text(f'{prefix}"{p_name}": ', "EXPECT_NUMBER_VALUE")

    def get_allowed_tokens(self) -> List[int]:
        """Return token IDs allowed at this step (100% read-only)."""

        # Rule 1: If we have forced tokens queued up, only return the first one
        if self.forced_tokens:
            return [self.forced_tokens[0]]

        # Rule 2: Otherwise, ask the AI based on the current state
        if self.state == "EXPECT_FUNCTION_NAME":
            allowed: List[int] = []
            for t_id, t_text in self.fn_candidates.items():
                potential = self.typed_name + t_text
                for fn in self.valid_functions:
                    if (fn + '"').startswith(potential):
                        allowed.append(t_id)
                        break
            return allowed

        if self.state == "EXPECT_NUMBER_VALUE":
            allowed = list(self.vocab.get_number_tokens())
            if len(self.params_queue) == 0:
                allowed.extend(self.vocab.get_close_brace_tokens())
            else:
                allowed.extend(self.vocab.get_comma_tokens())
            return allowed

        if self.state == "EXPECT_BOOLEAN_VALUE":
            return list(self.vocab.get_boolean_tokens())

        if self.state == "EXPECT_STRING_VALUE":
            if self.string_length >= self.string_limit:
                return list(self.vocab.get_quote_tokens())
            allowed = list(self.vocab.get_string_tokens())
            allowed.extend(self.vocab.get_quote_tokens())
            return allowed

        return []

    def commit(self, t_id: int, text: str) -> None:
        """Update FSM state based on what token was picked."""

        # If we are working through forced tokens, pop the used one and return
        if self.forced_tokens:
            self.forced_tokens.pop(0)
            # If we just finished forcing tokens, transition to the next state
            if len(self.forced_tokens) == 0 and self.next_state:
                self.state = self.next_state
                self.next_state = ""
            return

        # Otherwise, track the AI's generation
        if self.state == "EXPECT_FUNCTION_NAME":
            self.typed_name += text
            if self.typed_name.endswith('"'):
                self.chosen_function = self.typed_name[:-1]
                fn = self.chosen_function
                self.params_queue = self.registry.get_parameter_names(fn)
                self.setup_next_parameter(
                    ', "parameters": {', ', "parameters": {}}'
                )

        elif self.state in ("EXPECT_NUMBER_VALUE", "EXPECT_BOOLEAN_VALUE"):
            terminator = "}" if len(self.params_queue) == 0 else ","
            if terminator in text:
                self.setup_next_parameter(" ", "")

        elif self.state == "EXPECT_STRING_VALUE":
            self.string_length += 1
            if '"' in text:
                self.string_length = 0
                self.setup_next_parameter(", ", "}}")
