"""Finite State Machine for constrained JSON generation.

The FSM tracks where we are in building a JSON function call and tells
the generation loop which tokens are valid at each step. This ensures
the output is always valid JSON that matches the function schema.

The JSON structure we're building looks like:
    {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}

How it works:
- Structural parts (like {"name": " or , "parameters": {) are FORCED:
  the FSM encodes them into token IDs and emits them one at a time.
  The AI has no choice here, and we skip the expensive model call.
- Value parts (function name, parameter values) are FREE: the AI picks
  from a filtered set of valid tokens using logit masking.

States:
    EXPECT_START         -> Forces the opening: {"name": "
    FORCING_SEQUENCE     -> Forces a specific sequence of tokens
    EXPECT_FUNCTION_NAME -> AI picks tokens to build a valid function name
    EXPECT_NEXT_PARAM    -> Transitions to next parameter or closes JSON
    EXPECT_NUMBER_VALUE  -> AI picks number tokens for a number parameter
    EXPECT_BOOLEAN_VALUE -> AI picks boolean tokens
    EXPECT_STRING_VALUE  -> AI picks string tokens for a string parameter
    DONE                 -> Generation complete
"""
from typing import Any, Dict, List, Optional, Set

from src.vocab import VocabManager
from src.registry import FunctionRegistry


class JSONStateMachine:
    """Tracks position in JSON structure and returns allowed tokens.

    The FSM forces structural parts as exact token sequences, and lets
    the AI choose freely only for values (function name, parameter
    values). This guarantees valid JSON output every time.
    """

    # Maximum number of tokens allowed for a single string value.
    # This prevents the model from generating infinitely long strings.
    MAX_STRING_TOKENS: int = 50

    def __init__(
        self,
        ai: Any,
        vocab: VocabManager,
        registry: FunctionRegistry,
    ) -> None:
        """Initialize the state machine.

        Args:
            ai: The LLM model instance (used for encode() on forced text).
            vocab: The vocabulary manager with cached token lists.
            registry: The function registry with available functions.
        """
        self.ai = ai
        self.vocab = vocab
        self.registry = registry

        # Current state of the FSM
        self.state: str = "EXPECT_START"

        # Which function the AI chose (set after function name is done)
        self.chosen_function: Optional[str] = None

        # --- Forced sequence tracking ---
        # When we need to force exact text (like '{"name": "'), we
        # encode it into token IDs and force them one at a time.
        self.sequence_to_force: List[int] = []
        self.sequence_index: int = 0
        self.next_state_after_sequence: Optional[str] = None

        # --- Function name tracking ---
        self.valid_functions: List[str] = registry.get_functions_name()
        # Text built so far while the AI picks function name tokens
        self.current_fn_string: str = ""
        # Pre-filtered vocab: only tokens whose characters could appear
        # in a function name. Cuts search from 150K+ to a few hundred.
        self._fn_name_candidates: Dict[int, str] = (
            self._build_fn_name_candidates()
        )

        # --- Parameter tracking ---
        # Queue of parameter names still to process (in order)
        self.params_queue: List[str] = []
        # Type of the current parameter being generated
        # How many tokens generated for the current string value
        self.string_value_length: int = 0

    def _build_fn_name_candidates(self) -> Dict[int, str]:
        """Pre-filter the vocab to only tokens useful for function names.

        We collect all characters that appear in any function name (like
        a-z, _, 0-9), plus the closing quote. Then we keep only tokens
        whose text is made entirely of those characters.

        This typically cuts the search from 150K+ tokens to a few hundred,
        which makes _get_fn_name_tokens() much faster.

        Returns:
            A dict of {token_id: token_text} for candidate tokens.
        """
        # Collect every character used in any function name
        valid_chars: Set[str] = set()
        for fn_name in self.valid_functions:
            valid_chars.update(fn_name)
        # Also allow the closing quote (marks end of function name)
        valid_chars.add('"')

        # Keep only tokens whose text contains only valid_chars
        candidates: Dict[int, str] = {}
        for t_id, t_text in self.vocab.get_all_tokens().items():
            if t_text and all(c in valid_chars for c in t_text):
                candidates[t_id] = t_text
        return candidates

    def _force_string(self, text: str, next_state: str) -> None:
        """Set up a forced token sequence.

        The given text is encoded into token IDs using the model's
        tokenizer. The FSM will force these tokens one at a time.
        Since forced tokens have only one valid choice, the main loop
        skips the expensive model call for them.

        Args:
            text: The exact text to force (e.g., '{"name": "').
            next_state: The state to go to after the sequence is done.
        """
        encoded = self.ai.encode(text)
        self.sequence_to_force = encoded[0].tolist()
        self.sequence_index = 0
        self.state = "FORCING_SEQUENCE"
        self.next_state_after_sequence = next_state

    def get_allowed_tokens(self) -> List[int]:
        """Return the list of token IDs the AI is allowed to pick from.

        This is the core of constrained decoding. Based on the current
        state, we return only tokens that would maintain valid JSON.

        Returns:
            A list of allowed token IDs.
        """
        if self.state == "EXPECT_START":
            # Begin the JSON: force {"name": "
            self._force_string('{"name": "', "EXPECT_FUNCTION_NAME")
            return self.get_allowed_tokens()

        if self.state == "FORCING_SEQUENCE":
            # Only one token allowed: the next in the forced sequence
            return [self.sequence_to_force[self.sequence_index]]

        if self.state == "EXPECT_FUNCTION_NAME":
            return self._get_fn_name_tokens()

        if self.state == "EXPECT_NEXT_PARAM":
            return self._handle_next_param()

        if self.state == "EXPECT_NUMBER_VALUE":
            return self._get_number_value_tokens()

        if self.state == "EXPECT_BOOLEAN_VALUE":
            return self._get_boolean_value_tokens()

        if self.state == "EXPECT_STRING_VALUE":
            return self._get_string_value_tokens()

        return []

    def _get_fn_name_tokens(self) -> List[int]:
        """Find tokens that build towards a valid function name.

        We check each candidate token (pre-filtered at init) to see if
        appending it to the current function name text would still be a
        prefix of some valid function name (or complete it with a quote).

        Returns:
            A list of allowed token IDs.
        """
        allowed: List[int] = []
        for t_id, t_text in self._fn_name_candidates.items():
            # What the name would look like if we added this token
            potential = self.current_fn_string + t_text
            for fn in self.valid_functions:
                # Does any function name start with this text?
                if fn.startswith(potential):
                    allowed.append(t_id)
                    break
                # Does this complete a function name with closing quote?
                if (fn + '"').startswith(potential):
                    allowed.append(t_id)
                    break
        return allowed

    def _handle_next_param(self) -> List[int]:
        """Move to the next parameter, or close the JSON if all done.

        If there are more parameters in the queue, force the key text
        (like , "b": ) and transition to the appropriate value state.
        If all parameters are done, force the closing braces }}.

        Returns:
            A list of allowed token IDs.
        """
        if len(self.params_queue) == 0:
            # All parameters done — close both the parameters object
            # and the outer object with }}
            self._force_string("}}", "DONE")
        else:
            # Get the next parameter from the queue
            p_name = self.params_queue.pop(0)
            fn = self.chosen_function or ""
            p_type = self.registry.get_parameters(fn)[p_name]["type"]

            # Force the parameter key, then let the AI pick the value
            if p_type == "string":
                self._force_string(
                    f', "{p_name}": "', "EXPECT_STRING_VALUE"
                )
            elif p_type == "boolean":
                self._force_string(
                    f', "{p_name}": ', "EXPECT_BOOLEAN_VALUE"
                )
            else:
                # Default to number
                self._force_string(
                    f', "{p_name}": ', "EXPECT_NUMBER_VALUE"
                )
        return self.get_allowed_tokens()

    def _get_number_value_tokens(self) -> List[int]:
        """Get allowed tokens for a number parameter value.

        Allows number tokens (digits, dots, minus) plus a terminator
        that signals the end of the number:
        - '}' if this is the last parameter
        - ',' if there are more parameters after this one

        Uses cached terminator token lists (built at startup in
        VocabManager) instead of scanning all 150K+ tokens each time.

        Returns:
            A list of allowed token IDs.
        """
        # Start with all number tokens
        allowed: List[int] = list(self.vocab.get_number_tokens())

        # Add the right terminator tokens from the cached lists
        if len(self.params_queue) == 0:
            # Last parameter: allow '}' to close parameters object
            allowed.extend(self.vocab.get_close_brace_tokens())
        else:
            # More parameters coming: allow ',' to separate them
            allowed.extend(self.vocab.get_comma_tokens())
        return allowed

    def _get_boolean_value_tokens(self) -> List[int]:
        """Get allowed tokens for a boolean parameter value.

        Returns:
            A list of boolean token IDs (true/false and variants).
        """
        return list(self.vocab.get_boolean_tokens())

    def _get_string_value_tokens(self) -> List[int]:
        """Get allowed tokens for a string parameter value.

        Allows any string-safe token plus the closing quote.
        If the string has hit MAX_STRING_TOKENS, only the closing
        quote is allowed (forces the string to end).

        Returns:
            A list of allowed token IDs.
        """
        if self.string_value_length >= self.MAX_STRING_TOKENS:
            # String too long — force it to close with a quote
            return list(self.vocab.get_quote_tokens())

        # Allow any string-safe token plus the closing quote
        allowed: List[int] = list(self.vocab.get_string_tokens())
        allowed.extend(self.vocab.get_quote_tokens())
        return allowed

    def commit(self, t_id: int, text: str) -> None:
        """Update the FSM state after a token has been picked.

        Called after each token is generated. Advances the FSM to the
        next state based on what was picked.

        Args:
            t_id: The chosen token ID.
            text: The decoded text of the chosen token.
        """
        if self.state == "FORCING_SEQUENCE":
            # Move to the next token in the forced sequence
            self.sequence_index += 1
            if self.sequence_index >= len(self.sequence_to_force):
                # Forced sequence complete — move to next state
                self.state = self.next_state_after_sequence or "DONE"

        elif self.state == "EXPECT_FUNCTION_NAME":
            self._commit_fn_name(text)

        elif self.state == "EXPECT_NUMBER_VALUE":
            self._commit_number_or_bool(text)

        elif self.state == "EXPECT_BOOLEAN_VALUE":
            self._commit_number_or_bool(text)

        elif self.state == "EXPECT_STRING_VALUE":
            # Count tokens in the current string value
            self.string_value_length += 1
            if '"' in text:
                # Closing quote found — string value is done
                self.string_value_length = 0
                self.state = "EXPECT_NEXT_PARAM"

    def _commit_fn_name(self, text: str) -> None:
        """Process a function name token.

        Appends the token text to the name being built. When a closing
        quote (") is found, the name is complete and we set up the
        parameters section.

        Args:
            text: The decoded token text.
        """
        self.current_fn_string += text

        # Not done yet if there's no closing quote
        if not self.current_fn_string.endswith('"'):
            return

        # Function name complete — strip the closing quote
        self.chosen_function = self.current_fn_string[:-1]
        fn = self.chosen_function

        # Get the parameter names for this function (in order)
        self.params_queue = self.registry.get_parameter_names(fn)

        if len(self.params_queue) == 0:
            # No parameters — force empty parameters object
            self._force_string(
                ', "parameters": {', "EXPECT_NEXT_PARAM"
            )
        else:
            # Set up the first parameter
            p_name = self.params_queue.pop(0)
            p_type = self.registry.get_parameters(fn)[p_name]["type"]

            # Force the parameters key and first parameter name,
            # then let AI pick the value
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

        Checks if the token contains a terminator (} or ,). If yes,
        the value is complete and we either finish or move to the
        next parameter.

        Note: when the last parameter is a number/bool and ends with '}',
        that '}' only closes the inner parameters object. The outer '}'
        is added by post-processing in __main__.py.

        Args:
            text: The decoded token text.
        """
        # Which character signals the end of this value?
        terminator = "}" if len(self.params_queue) == 0 else ","

        if terminator not in text:
            # Still generating the number/boolean value
            return

        if len(self.params_queue) == 0:
            # Last parameter — we're done
            # (outer '}' added by post-processing)
            self.state = "DONE"
        else:
            # More parameters — set up the next one
            p_name = self.params_queue.pop(0)
            fn = self.chosen_function or ""
            p_type = self.registry.get_parameters(fn)[p_name]["type"]

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
