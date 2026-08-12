"""Vocabulary manager for constrained decoding."""
import json
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, ConfigDict


class VocabManager(BaseModel):
    """Loads the model vocabulary and builds filtered token lists.

    The vocab file maps token text to token IDs. We decode every token
    to get clean text, then build cached lists of number and string
    tokens for the constraint cage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    decoded_vocab: Dict[int, str] = {}
    _number_tokens: List[int] = []
    _string_tokens: List[int] = []
    _quote_tokens: List[int] = []
    _boolean_tokens: List[int] = []

    def __init__(self, **data: Any) -> None:
        """Initialize and populate the vocabulary."""
        super().__init__(**data)

        vocab_path: str = data["vocab_path"]
        ai_decode_fn: Callable = data["ai_decode_fn"]

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab: Dict[str, int] = json.load(f)

        for token_text, token_id in raw_vocab.items():
            self.decoded_vocab[token_id] = ai_decode_fn([token_id])

        # Cache these once at startup so generation is fast
        self._number_tokens = self._build_number_tokens()
        self._string_tokens = self._build_string_tokens()
        self._quote_tokens = self._build_quote_tokens()
        self._boolean_tokens = self._build_boolean_tokens()

    def get_all_tokens(self) -> Dict[int, str]:
        """Return the full decoded vocabulary dictionary."""
        return self.decoded_vocab

    def get_number_tokens(self) -> List[int]:
        """Return cached list of token IDs that represent numbers."""
        return self._number_tokens

    def get_string_tokens(self) -> List[int]:
        """Return cached list of token IDs safe for JSON strings."""
        return self._string_tokens

    def get_quote_tokens(self) -> List[int]:
        """Return cached list of token IDs for the closing quote."""
        return self._quote_tokens

    def get_boolean_tokens(self) -> List[int]:
        """Return cached list of token IDs for booleans."""
        return self._boolean_tokens

    def _build_number_tokens(self) -> List[int]:
        """Find all tokens made entirely of digits, dots, or minus.

        Returns:
            A list of valid number token IDs.
        """
        allowed: List[int] = []
        for t_id, text in self.decoded_vocab.items():
            stripped = text.strip()
            if stripped == "":
                continue
            if all(c in "0123456789.-" for c in stripped):
                allowed.append(t_id)
        return allowed

    def _build_string_tokens(self) -> List[int]:
        """Find all tokens that do not contain a double quote or controls.

        Returns:
            A list of valid string token IDs.
        """
        allowed: List[int] = []
        for t_id, text in self.decoded_vocab.items():
            if text == "":
                continue
            if '"' not in text:
                # Ban ASCII control characters to ensure valid JSON
                if not any(ord(c) < 32 for c in text):
                    allowed.append(t_id)
        return allowed

    def _build_quote_tokens(self) -> List[int]:
        """Find all tokens that are just a double quote.

        Returns:
            A list of quote token IDs.
        """
        allowed: List[int] = []
        for t_id, text in self.decoded_vocab.items():
            if text.strip() == '"':
                allowed.append(t_id)
        return allowed

    def _build_boolean_tokens(self) -> List[int]:
        """Find tokens that build towards 'true' or 'false'.

        Returns:
            A list of boolean token IDs.
        """
        allowed: List[int] = []
        targets = ["true", "false", "true,", "false,", "true}", "false}"]
        for t_id, text in self.decoded_vocab.items():
            stripped = text.strip()
            if stripped == "":
                continue
            for t in targets:
                if t.startswith(stripped) or stripped.startswith(t):
                    allowed.append(t_id)
                    break
        return allowed
