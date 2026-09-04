"""Vocabulary manager for constrained decoding.

Loads the model's vocab.json file and builds filtered lists of token IDs
for each type (numbers, strings, booleans, commas, close braces).

These cached lists let the FSM instantly know which tokens are valid
at each step, without scanning all 150K+ tokens every time.
"""
import json
from typing import Dict, List
from pydantic import BaseModel


def swap_tokens(bpe_text: str) -> str:
    """Convert a raw BPE token string into readable text.

    The vocab file stores spaces as the special character 'Ġ' and
    newlines as 'Ċ'. Everything else (letters, digits, punctuation)
    is stored as-is. So we just replace those two special characters.

    Example: 'Ġhello' -> ' hello'
    """
    return bpe_text.replace("Ġ", " ").replace("Ċ", "\n")


class VocabManager(BaseModel):
    """Loads the AI vocabulary and builds filtered token lists.

    At startup: reads vocab.json, decodes each token, and builds
    cached lists (numbers, strings, commas, etc.) in a single pass.

    During generation: the FSM calls the get_*() methods to instantly
    get valid token lists without scanning all 150K+ tokens each time.
    """

    # Maps token_id -> decoded text for every token in the vocabulary.
    # Example: {12345: " hello", 67890: "world"}
    decoded_vocab: Dict[int, str] = {}

    # Cached filtered token lists, built once at startup.
    number_tokens: List[int] = []
    string_tokens: List[int] = []
    quote_tokens: List[int] = []
    boolean_tokens: List[int] = []
    comma_tokens: List[int] = []
    close_brace_tokens: List[int] = []
    fn_candidates: Dict[int, str] = {}

    def __init__(self, vocab_path: str, fn_names: List[str] = []) -> None:
        """Load vocabulary and build all cached token lists in one pass.

        Args:
            vocab_path: Path to the model's vocab.json file.
            fn_names: List of valid function names. Used to pre-filter
                      tokens that could appear in a function name.
        """
        super().__init__()

        # Read the vocab file.
        # Format: {"Ġhello": 12345, "world": 67890, ...}
        # Keys are BPE token strings, values are token IDs.
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab: Dict[str, int] = json.load(f)

        # Collect every character that appears in any function name,
        # plus the closing quote. Used to filter fn_candidates below.
        valid_chars: set = set()
        for fn_name in fn_names:
            valid_chars.update(fn_name)
        valid_chars.add('"')

        # The targets we check against when building boolean tokens.
        bool_targets = ["true", "false", "true,", "false,", "true}", "false}"]

        # Single pass: decode each token and sort it into the right lists.
        # This replaces calling ai.decode() 150K+ times (which was slow)
        # and avoids looping through the vocab multiple times.
        for bpe_text, token_id in raw_vocab.items():
            text = swap_tokens(bpe_text)
            self.decoded_vocab[token_id] = text

            stripped = text.strip()
            if not stripped:
                continue

            # Number token: every character is a digit, dot, or minus
            if all(c in "0123456789.-" for c in stripped):
                self.number_tokens.append(token_id)

            # Quote token: just a double-quote character
            if stripped == '"':
                self.quote_tokens.append(token_id)

            # Comma token: only commas and whitespace
            if "," in text and all(c in " \n\r\t," for c in text):
                self.comma_tokens.append(token_id)

            # Close brace token: only '}' and whitespace
            if "}" in text and all(c in " \n\r\t}" for c in text):
                self.close_brace_tokens.append(token_id)

            # Boolean token: matches or is a prefix of true/false
            for target in bool_targets:
                if target.startswith(stripped) or stripped.startswith(target):
                    self.boolean_tokens.append(token_id)
                    break

            # String token: no double-quote and no control characters
            if '"' not in text and not any(ord(c) < 32 for c in text):
                self.string_tokens.append(token_id)

            # Function name candidate: every character is in valid_chars
            if fn_names and all(c in valid_chars for c in text):
                self.fn_candidates[token_id] = text

    def get_token_text(self, token_id: int) -> str:
        """Return the decoded text for a token ID.

        Used in the generation loop instead of calling ai.decode().
        """
        return self.decoded_vocab.get(token_id, "")

    def get_all_tokens(self) -> Dict[int, str]:
        """Return the full token_id -> text dictionary."""
        return self.decoded_vocab

    def get_fn_candidates(self) -> Dict[int, str]:
        """Return pre-filtered tokens that could appear in a function name."""
        return self.fn_candidates

    def get_number_tokens(self) -> List[int]:
        """Return cached token IDs for number parts (digits, dot, minus)."""
        return self.number_tokens

    def get_string_tokens(self) -> List[int]:
        """Return cached token IDs safe to use inside a JSON string."""
        return self.string_tokens

    def get_quote_tokens(self) -> List[int]:
        """Return cached token IDs for the double-quote character."""
        return self.quote_tokens

    def get_boolean_tokens(self) -> List[int]:
        """Return cached token IDs for true/false values."""
        return self.boolean_tokens

    def get_comma_tokens(self) -> List[int]:
        """Return cached token IDs for commas (parameter separators)."""
        return self.comma_tokens

    def get_close_brace_tokens(self) -> List[int]:
        """Return cached token IDs for '}' (closes the JSON object)."""
        return self.close_brace_tokens
