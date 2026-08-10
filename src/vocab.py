import json
from typing import Dict, List, Callable

class VocabManager:
    def __init__(self, vocab_path: str, ai_decode_fn: Callable):
        """
        Loads the vocabulary from the JSON file and decodes every single Token ID
        into its actual text string. This takes a few seconds at startup.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            
        print("Loading and decoding 150,000+ vocabulary tokens... (please wait)")
        self.decoded_vocab: Dict[int, str] = {}
        for token_text, token_id in raw_vocab.items():
            self.decoded_vocab[token_id] = ai_decode_fn([token_id])
            
    def get_all_tokens(self) -> Dict[int, str]:
        return self.decoded_vocab
        
    def get_number_tokens(self) -> List[int]:
        """
        Finds every Token ID in the vocabulary that represents a valid number part.
        This includes digits, negative signs, and decimal points.
        """
        allowed = []
        for t_id, text in self.decoded_vocab.items():
            text = text.strip()  # Remove spaces
            if text == "":
                continue
            
            # Check if every character is a valid piece of a float/integer
            is_valid = True
            for char in text:
                if char not in "0123456789.-":
                    is_valid = False
                    break
            
            if is_valid:
                allowed.append(t_id)
                
        return allowed
        
    def get_string_tokens(self) -> List[int]:
        """
        Finds every Token ID that is safe to put INSIDE a JSON string value.
        The main rule is: it cannot contain an unescaped double quote (").
        """
        allowed = []
        for t_id, text in self.decoded_vocab.items():
            if '"' not in text:
                allowed.append(t_id)
        return allowed
