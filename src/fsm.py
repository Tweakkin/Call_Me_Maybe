"""Finite State Machine for constrained JSON generation."""
from typing import Any, List, Optional

from src.vocab import VocabManager
from src.registry import FunctionRegistry


class JSONStateMachine:
    """Tracks position in the JSON structure and returns allowed tokens.

    States:
        EXPECT_START: Beginning, forces '{"name": "'
        FORCING_SEQUENCE: Forces a specific token sequence
        EXPECT_FUNCTION_NAME: AI picks function name tokens
        EXPECT_NEXT_PARAM: Transitions to next parameter or closes
        EXPECT_NUMBER_VALUE: AI picks number tokens
        EXPECT_BOOLEAN_VALUE: AI picks boolean tokens
        EXPECT_STRING_VALUE: AI picks string tokens
        DONE: Generation complete
    """

    MAX_STRING_TOKENS: int = 50

    def __init__(
        self,
        ai: Any,
        vocab: VocabManager,
        registry: FunctionRegistry,
    ) -> None:
        """Initialize the state machine.

        Args:
            ai: The LLM model instance (for encode).
            vocab: The vocabulary manager.
            registry: The function registry.
        """
        self.ai = ai
        self.vocab = vocab
        self.registry = registry

        self.state: str = "EXPECT_START"
        self.chosen_function: Optional[str] = None

        # For forcing exact token sequences
        self.sequence_to_force: List[int] = []
        self.sequence_index: int = 0
        self.next_state_after_sequence: Optional[str] = None

        # For function name building
        self.valid_functions: List[str] = registry.get_functions_name()
        self.current_fn_string: str = ""

        # For parameter tracking
        self.params_queue: List[str] = []
        self.current_param_type: Optional[str] = None
        self.string_value_length: int = 0

    def _force_string(self, text: str, next_state: str) -> None:
        """Set up a forced token sequence.

        Args:
            text: The exact text to force the AI to output.
            next_state: The state to transition to after the sequence.
        """
        encoded = self.ai.encode(text)
        self.sequence_to_force = encoded[0].tolist()
        self.sequence_index = 0
        self.state = "FORCING_SEQUENCE"
        self.next_state_after_sequence = next_state

    def get_allowed_tokens(self) -> List[int]:
        """Return the list of allowed token IDs for the current state.

        Returns:
            A list of token IDs the AI is allowed to pick from.
        """
        if self.state == "EXPECT_START":
            self._force_string('{"name": "', "EXPECT_FUNCTION_NAME")
            return self.get_allowed_tokens()

        elif self.state == "FORCING_SEQUENCE":
            return [self.sequence_to_force[self.sequence_index]]

        elif self.state == "EXPECT_FUNCTION_NAME":
            return self._get_fn_name_tokens()

        elif self.state == "EXPECT_NEXT_PARAM":
            return self._handle_next_param()

        elif self.state == "EXPECT_NUMBER_VALUE":
            return self._get_number_value_tokens()

        elif self.state == "EXPECT_BOOLEAN_VALUE":
            return self._get_boolean_value_tokens()

        elif self.state == "EXPECT_STRING_VALUE":
            return self._get_string_value_tokens()

        return []

    def _get_fn_name_tokens(self) -> List[int]:
        """Find tokens that build towards a valid function name.

        Returns:
            A list of allowed token IDs.
        """
        allowed: List[int] = []
        for t_id, t_text in self.vocab.get_all_tokens().items():
            potential = self.current_fn_string + t_text
            for fn in self.valid_functions:
                if fn.startswith(potential):
                    allowed.append(t_id)
                    break
                if (fn + '"').startswith(potential):
                    allowed.append(t_id)
                    break
        return allowed

    def _handle_next_param(self) -> List[int]:
        """Transition to the next parameter or close the JSON.

        Returns:
            A list of allowed token IDs.
        """
        if len(self.params_queue) == 0:
            self._force_string("}}",  "DONE")
        else:
            p_name = self.params_queue.pop(0)
            fn = self.chosen_function or ""
            p_type = self.registry.get_parameters(fn)[p_name]["type"]
            self.current_param_type = p_type

            if p_type == "string":
                self._force_string(
                    f', "{p_name}": "', "EXPECT_STRING_VALUE"
                )
            elif p_type == "boolean":
                self._force_string(
                    f', "{p_name}": ', "EXPECT_BOOLEAN_VALUE"
                )
            else:
                self._force_string(
                    f', "{p_name}": ', "EXPECT_NUMBER_VALUE"
                )
        return self.get_allowed_tokens()

    def _get_number_value_tokens(self) -> List[int]:
        """Get allowed tokens for a number parameter value.

        Returns:
            A list of number token IDs plus a terminator.
        """
        allowed: List[int] = list(self.vocab.get_number_tokens())

        terminator = "}" if len(self.params_queue) == 0 else ","
        for t_id, t_text in self.vocab.get_all_tokens().items():
            if terminator in t_text:
                if all(c in " \n\r\t" + terminator for c in t_text):
                    allowed.append(t_id)
        return allowed

    def _get_boolean_value_tokens(self) -> List[int]:
        """Get allowed tokens for a boolean parameter value.

        Returns:
            A list of boolean token IDs.
        """
        return list(self.vocab.get_boolean_tokens())

    def _get_string_value_tokens(self) -> List[int]:
        """Get allowed tokens for a string parameter value.

        Returns:
            A list of string token IDs plus the closing quote.
        """
        if self.string_value_length >= self.MAX_STRING_TOKENS:
            return list(self.vocab.get_quote_tokens())

        allowed: List[int] = list(self.vocab.get_string_tokens())
        allowed.extend(self.vocab.get_quote_tokens())
        return allowed

    def commit(self, t_id: int, text: str) -> None:
        """Advance the state machine after a token is picked.

        Args:
            t_id: The chosen token ID.
            text: The decoded text of the chosen token.
        """
        if self.state == "FORCING_SEQUENCE":
            self.sequence_index += 1
            if self.sequence_index >= len(self.sequence_to_force):
                self.state = self.next_state_after_sequence or "DONE"

        elif self.state == "EXPECT_FUNCTION_NAME":
            self._commit_fn_name(text)

        elif self.state == "EXPECT_NUMBER_VALUE":
            self._commit_number_or_bool(text)

        elif self.state == "EXPECT_BOOLEAN_VALUE":
            self._commit_number_or_bool(text)

        elif self.state == "EXPECT_STRING_VALUE":
            self.string_value_length += 1
            if '"' in text:
                self.string_value_length = 0
                self.state = "EXPECT_NEXT_PARAM"

    def _commit_fn_name(self, text: str) -> None:
        """Process a function name token.

        Args:
            text: The decoded token text.
        """
        self.current_fn_string += text
        if not self.current_fn_string.endswith('"'):
            return

        self.chosen_function = self.current_fn_string[:-1]
        fn = self.chosen_function
        self.params_queue = self.registry.get_parameter_names(fn)

        if len(self.params_queue) == 0:
            self._force_string(
                ', "parameters": {', "EXPECT_NEXT_PARAM"
            )
        else:
            p_name = self.params_queue.pop(0)
            p_type = self.registry.get_parameters(fn)[p_name]["type"]
            self.current_param_type = p_type

            if p_type == "string":
                self._force_string(
                    f', "parameters": {{"{p_name}": "',
                    "EXPECT_STRING_VALUE",
                )
            elif p_type == "boolean":
                self._force_string(
                    f', "parameters": {{"{p_name}": ',
                    "EXPECT_BOOLEAN_VALUE",
                )
            else:
                self._force_string(
                    f', "parameters": {{"{p_name}": ',
                    "EXPECT_NUMBER_VALUE",
                )

    def _commit_number_or_bool(self, text: str) -> None:
        """Process a number or boolean value token.

        Args:
            text: The decoded token text.
        """
        terminator = "}" if len(self.params_queue) == 0 else ","
        if terminator not in text:
            return

        if len(self.params_queue) == 0:
            self.state = "DONE"
        else:
            p_name = self.params_queue.pop(0)
            fn = self.chosen_function or ""
            p_type = self.registry.get_parameters(fn)[p_name]["type"]
            self.current_param_type = p_type

            if p_type == "string":
                self._force_string(
                    f' "{p_name}": "', "EXPECT_STRING_VALUE"
                )
            elif p_type == "boolean":
                self._force_string(
                    f' "{p_name}": ', "EXPECT_BOOLEAN_VALUE"
                )
            else:
                self._force_string(
                    f' "{p_name}": ', "EXPECT_NUMBER_VALUE"
                )
