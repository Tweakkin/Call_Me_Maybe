import json
from typing import Dict, Any, List

class FunctionRegistry:
    """
    Loads and manages the function definitions from functions_definition.json.
    This tells the State Machine what functions are available and what parameters they need.
    """
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        
    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Store functions by their name for easy lookup
        for fn in data:
            self.functions[fn["name"]] = fn
            
    def get_functions_name(self) -> List[str]:
        return list(self.functions.keys())
        
    def get_description(self, name: str) -> str:
        return self.functions.get(name, {}).get("description", "")
        
    def get_parameters(self, name: str) -> Dict[str, Any]:
        """
        Returns a dict of parameter names to their type info.
        Example: {'a': {'type': 'number'}, 'b': {'type': 'number'}}
        """
        return self.functions.get(name, {}).get("parameters", {})
        
    def get_parameter_names(self, name: str) -> List[str]:
        return list(self.get_parameters(name).keys())
