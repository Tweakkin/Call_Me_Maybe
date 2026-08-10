from typing import List
from .vocab import VocabManager
from .registry import FunctionRegistry

class JSONStateMachine:
    def __init__(self, ai, vocab: VocabManager, registry: FunctionRegistry):
        self.ai = ai
        self.vocab = vocab
        self.registry = registry
        
        self.state = "EXPECT_START"
        self.chosen_function = None
        
        # For forcing specific sequences of tokens
        self.sequence_to_force = []
        self.sequence_index = 0
        self.next_state_after_sequence = None
        
        # For function names
        self.valid_functions = registry.get_functions_name()
        self.current_fn_string = ""
        
        # For parameters
        self.params_queue = []
        self.current_param_type = None
        
    def _force_string(self, text: str, next_state: str):
        """Helper to force the AI to type an exact string (like '{"name": "')"""
        self.sequence_to_force = self.ai.encode(text)[0].tolist()
        self.sequence_index = 0
        self.state = "FORCING_SEQUENCE"
        self.next_state_after_sequence = next_state
        
    def get_allowed_tokens(self) -> List[int]:
        """Returns the list of Token IDs allowed for the current state."""
        
        if self.state == "EXPECT_START":
            self._force_string('{"name": "', "EXPECT_FUNCTION_NAME")
            return self.get_allowed_tokens()
            
        elif self.state == "FORCING_SEQUENCE":
            return [self.sequence_to_force[self.sequence_index]]
            
        elif self.state == "EXPECT_FUNCTION_NAME":
            allowed = []
            for t_id, t_text in self.vocab.get_all_tokens().items():
                potential_str = self.current_fn_string + t_text
                
                # Check if this token builds towards a valid function name OR closes it with a quote
                is_valid = False
                for fn in self.valid_functions:
                    if fn.startswith(potential_str):
                        is_valid = True
                        break
                    target_closed = fn + '"'
                    if target_closed.startswith(potential_str):
                        is_valid = True
                        break
                        
                if is_valid:
                    allowed.append(t_id)
            return allowed
            
        elif self.state == "EXPECT_NEXT_PARAM":
            if len(self.params_queue) == 0:
                self._force_string('}', "DONE")
            else:
                p_name = self.params_queue.pop(0)
                self.current_param_type = self.registry.get_parameters(self.chosen_function)[p_name]["type"]
                
                # If it's a string, open the quotes for the value. If number, no quotes.
                if self.current_param_type == "string":
                    self._force_string(f', "{p_name}": "', "EXPECT_STRING_VALUE")
                else:
                    self._force_string(f', "{p_name}": ', "EXPECT_NUMBER_VALUE")
            return self.get_allowed_tokens()
            
        elif self.state == "EXPECT_NUMBER_VALUE":
            # Allow numbers
            allowed = list(self.vocab.get_number_tokens())
            
            # The number generation ends when the AI types a comma (if more params) or a brace (if done)
            # Actually, to make it strict, we don't let the AI type the comma/brace! 
            # We just let it type numbers, and we will force the comma/brace in EXPECT_NEXT_PARAM.
            # But wait, how do we know the AI is done thinking of the number? 
            # We MUST allow the AI to output a terminator so we know it's finished.
            terminator = "}" if len(self.params_queue) == 0 else ","
            for t_id, t_text in self.vocab.get_all_tokens().items():
                if terminator in t_text and all(c in " \n\r\t" + terminator for c in t_text):
                    allowed.append(t_id)
            return allowed
            
        elif self.state == "EXPECT_STRING_VALUE":
            # Allow all string characters (no quotes)
            allowed = list(self.vocab.get_string_tokens())
            
            # Allow the closing quote so the AI can finish the string
            for t_id, t_text in self.vocab.get_all_tokens().items():
                if t_text.strip() == '"':
                    allowed.append(t_id)
            return allowed
            
        return []

    def commit(self, t_id: int, text: str):
        """Called after the AI successfully picks a token. Advances the state machine."""
        
        if self.state == "FORCING_SEQUENCE":
            self.sequence_index += 1
            if self.sequence_index >= len(self.sequence_to_force):
                self.state = self.next_state_after_sequence
                
        elif self.state == "EXPECT_FUNCTION_NAME":
            self.current_fn_string += text
            if self.current_fn_string.endswith('"'):
                self.chosen_function = self.current_fn_string[:-1]
                self.params_queue = self.registry.get_parameter_names(self.chosen_function)
                
                # The function is chosen. Next, we force the parameters dictionary to open.
                # Note: EXPECT_NEXT_PARAM handles the comma for the first param differently, 
                # so we just force the opening of the dictionary here.
                if len(self.params_queue) == 0:
                    self._force_string(', "parameters": {', "EXPECT_NEXT_PARAM")
                else:
                    # For the very first parameter, we don't want a leading comma inside the dict
                    p_name = self.params_queue.pop(0)
                    self.current_param_type = self.registry.get_parameters(self.chosen_function)[p_name]["type"]
                    
                    if self.current_param_type == "string":
                        self._force_string(f', "parameters": {{"{p_name}": "', "EXPECT_STRING_VALUE")
                    else:
                        self._force_string(f', "parameters": {{"{p_name}": ', "EXPECT_NUMBER_VALUE")

        elif self.state == "EXPECT_NUMBER_VALUE":
            terminator = "}" if len(self.params_queue) == 0 else ","
            if terminator in text:
                # The AI finished the number and typed the terminator.
                # The terminator is already on the screen, so we skip forcing it.
                if len(self.params_queue) == 0:
                    self.state = "DONE"
                else:
                    # There are more parameters. The comma is printed. Force the NEXT parameter key.
                    p_name = self.params_queue.pop(0)
                    self.current_param_type = self.registry.get_parameters(self.chosen_function)[p_name]["type"]
                    if self.current_param_type == "string":
                        self._force_string(f' "{p_name}": "', "EXPECT_STRING_VALUE")
                    else:
                        self._force_string(f' "{p_name}": ', "EXPECT_NUMBER_VALUE")
                        
        elif self.state == "EXPECT_STRING_VALUE":
            if '"' in text:
                # The AI finished the string.
                self.state = "EXPECT_NEXT_PARAM"
