"""Loads and manages function definitions.

This module reads the functions_definition.json file and provides
lookup methods for function names, descriptions, and parameters.
The FunctionRegistry is used by the FSM to know which functions
exist and what parameters they expect.
"""
import json
from typing import Any, Dict, List
from pydantic import BaseModel

from src.models import FunctionDef


class FunctionRegistry(BaseModel):
    """Stores and provides access to function definitions.

    After loading a JSON file of function definitions, this class
    provides methods to look up function names, descriptions,
    parameter names, and parameter types.

    Example function definition:
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "number"}
        }
    """

    # Maps function name -> FunctionDef for all loaded functions
    functions: Dict[str, FunctionDef] = {}

    def load(self, path: str) -> None:
        """Load function definitions from a JSON file.

        Reads the file, validates each function definition using
        Pydantic, and stores them in the functions dict.

        Args:
            path: Path to the functions_definition.json file.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for fn in data:
            validated = FunctionDef.model_validate(fn)
            self.functions[validated.name] = validated

    def get_functions_name(self) -> List[str]:
        """Return a list of all registered function names.

        Example return: ['fn_add_numbers', 'fn_greet', 'fn_reverse_string']
        """
        return list(self.functions.keys())

    def get_description(self, name: str) -> str:
        """Return the description of a function.

        Args:
            name: The function name to look up.

        Returns:
            The description string, or empty string if not found.
        """
        fn = self.functions.get(name)
        if fn is None:
            return ""
        return fn.description

    def get_parameters(self, name: str) -> Dict[str, Any]:
        """Return parameter info for a function.

        Args:
            name: The function name to look up.

        Returns:
            A dict like {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
            Returns empty dict if function not found.
        """
        fn = self.functions.get(name)
        if fn is None:
            return {}
        return {
            k: v.model_dump() for k, v in fn.parameters.items()
        }

    def get_parameter_names(self, name: str) -> List[str]:
        """Return just the parameter names for a function (in order).

        Args:
            name: The function name to look up.

        Returns:
            A list like ['a', 'b']. Returns empty list if not found.
        """
        fn = self.functions.get(name)
        if fn is None:
            return []
        return list(fn.parameters.keys())
