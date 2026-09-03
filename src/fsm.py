"""Finite State Machine for constrained JSON generation.

Tracks where we are in building a JSON function call and tells the
generation loop which tokens are valid at each step.

The JSON we're building looks like:
    {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}

Structural parts (like {"name": ") are FORCED one token at a time.
Value parts (function name, parameter values) are FREE — the AI picks
from a filtered set of valid tokens.

States:
    EXPECT_START         -> Forces the opening: {"name": "
    FORCING_SEQUENCE     -> Forces a specific token sequence
    EXPECT_FUNCTION_NAME -> AI picks tokens to build a function name
    EXPECT_NEXT_PARAM    -> Moves to next parameter or closes JSON
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
        """Initialize the state machine.

        Args:
            ai: The LLM model (used for encode() on forced text).
            vocab: The vocabulary manager with cached token lists.
            registry: The function registry with available functions.
        """
        self.ai = ai
        self.vocab = vocab
        self.registry = registry

        # Current FSM state
        self.state: str = "EXPECT_START"

        # Which function the AI chose
        self.chosen_function: str = ""

        # Forced sequence tracking — when we force exact text like
        # {"name": ", we encode it into token IDs and emit one at a time
        self.sequence_to_force: List[int] = []
        self.sequence_index: int = 0
        self.next_state: str = ""

        # Function name tracking
        self.valid_functions: List[str] = registry.get_functions_name()
        self.current_fn_string: str = ""

        # Pre-filtered tokens that could appear in a function name.
        # Built during VocabManager's single pass — no extra loop needed.
        self.fn_candidates: Dict[int, str] = vocab.get_fn_candidates()

        # Parameter tracking
        self.params_queue: List[str] = []
        self.string_token_count: int = 0
        self.max_string_tokens: int = 50

    def _force_string(self, text: str, next_state: str) -> None:
        """Encode text into token IDs and force them one at a time.

        Since forced tokens have only one valid choice, the main loop
        skips the expensive model call for them.
        """
        encoded = self.ai.encode(text)
        self.sequence_to_force = encoded[0].tolist()
        self.sequence_index = 0
        self.state = "FORCING_SEQUENCE"
        self.next_state = next_state

    def _setup_param(self, p_name: str, prefix: str) -> None:
        """Force a parameter key and transition to the right value state.

        Args:
            p_name: The parameter name (like "a" or "b").
            prefix: Text before the parameter name (like ', ').
        """
        fn = self.chosen_function
        p_type = self.registry.get_parameters(fn)[p_name]["type"]

        if p_type == "string":
            self._force_string(
                f'{prefix}"{p_name}": "', "EXPECT_STRING_VALUE"
            )
        elif p_type == "boolean":
            self._force_string(
                f'{prefix}"{p_name}": ', "EXPECT_BOOLEAN_VALUE"
            )
        else:
            self._force_string(
                f'{prefix}"{p_name}": ', "EXPECT_NUMBER_VALUE"
            )

    def get_allowed_tokens(self) -> List[int]:
        """Return token IDs the AI is allowed to pick from.

        Based on the current state, returns only tokens that would
        keep the JSON valid.
        """
        if self.state == "EXPECT_START":
            self._force_string('{"name": "', "EXPECT_FUNCTION_NAME")
            return [self.sequence_to_force[0]]

        if self.state == "FORCING_SEQUENCE":
            return [self.sequence_to_force[self.sequence_index]]

        if self.state == "EXPECT_FUNCTION_NAME":
            # Check each candidate token to see if appending it
            # would still be a prefix of some valid function name
            allowed: List[int] = []
            for t_id, t_text in self.fn_candidates.items():
                potential = self.current_fn_string + t_text
                for fn in self.valid_functions:
                    if fn.startswith(potential):
                        allowed.append(t_id)
                        break
                    if (fn + '"').startswith(potential):
                        allowed.append(t_id)
                        break
            return allowed

        if self.state == "EXPECT_NEXT_PARAM":
            if len(self.params_queue) == 0:
                # All parameters done — close with }}
                self._force_string("}}", "DONE")
            else:
                p_name = self.params_queue.pop(0)
                self._setup_param(p_name, ", ")
            return [self.sequence_to_force[0]]

        if self.state == "EXPECT_NUMBER_VALUE":
            # Number tokens + terminator (} if last param, , if more)
            allowed = list(self.vocab.get_number_tokens())
            if len(self.params_queue) == 0:
                allowed.extend(self.vocab.get_close_brace_tokens())
            else:
                allowed.extend(self.vocab.get_comma_tokens())
            return allowed

        if self.state == "EXPECT_BOOLEAN_VALUE":
            return list(self.vocab.get_boolean_tokens())

        if self.state == "EXPECT_STRING_VALUE":
            # If string is too long, force it to close with a quote
            if self.string_token_count >= self.max_string_tokens:
                return list(self.vocab.get_quote_tokens())
            # Otherwise allow any string-safe token + closing quote
            allowed = list(self.vocab.get_string_tokens())
            allowed.extend(self.vocab.get_quote_tokens())
            return allowed

        return []

    def commit(self, t_id: int, text: str) -> None:
        """Update the FSM after a token has been picked.

        Called after each token is generated. Advances the state
        based on what token was picked.
        """
        if self.state == "FORCING_SEQUENCE":
            self.sequence_index += 1
            if self.sequence_index >= len(self.sequence_to_force):
                self.state = self.next_state

        elif self.state == "EXPECT_FUNCTION_NAME":
            self.current_fn_string += text

            # Check if function name is complete (ends with ")
            if self.current_fn_string.endswith('"'):
                # Strip the closing quote to get the function name
                self.chosen_function = self.current_fn_string[:-1]
                fn = self.chosen_function
                self.params_queue = self.registry.get_parameter_names(fn)

                if len(self.params_queue) == 0:
                    # No parameters — force empty parameters object
                    self._force_string(
                        ', "parameters": {', "EXPECT_NEXT_PARAM"
                    )
                else:
                    # Set up the first parameter
                    p_name = self.params_queue.pop(0)
                    self._setup_param(
                        p_name, ', "parameters": {'
                    )

        elif self.state in ("EXPECT_NUMBER_VALUE", "EXPECT_BOOLEAN_VALUE"):
            # Check if the value ended with a terminator
            terminator = "}" if len(self.params_queue) == 0 else ","
            if terminator in text:
                if len(self.params_queue) == 0:
                    # Last parameter — done (outer } added later)
                    self.state = "DONE"
                else:
                    # More parameters — set up the next one
                    p_name = self.params_queue.pop(0)
                    self._setup_param(p_name, " ")

        elif self.state == "EXPECT_STRING_VALUE":
            self.string_token_count += 1
            if '"' in text:
                # Closing quote found — string value is done
                self.string_token_count = 0
                self.state = "EXPECT_NEXT_PARAM"
