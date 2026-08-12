"""Loads and manages function definitions."""
import json
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict

from src.models import FunctionDef


class FunctionRegistry(BaseModel):
    """Loads and manages the function definitions.

    Reads functions_definition.json and provides lookup methods
    for function names, descriptions, and parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    functions: Dict[str, FunctionDef] = {}

    def __init__(self, **data: Any) -> None:
        """Initialize an empty registry."""
        super().__init__(**data)

    def load(self, path: str) -> None:
        """Load function definitions from a JSON file.

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
        """Return a list of all registered function names."""
        return list(self.functions.keys())

    def get_description(self, name: str) -> str:
        """Return the description of a function.

        Args:
            name: The function name.

        Returns:
            The description string, or empty string if not found.
        """
        fn = self.functions.get(name)
        if fn is None:
            return ""
        return fn.description

    def get_parameters(self, name: str) -> Dict[str, Any]:
        """Return a dict of parameter names to their type info.

        Args:
            name: The function name.

        Returns:
            A dict like {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
        """
        fn = self.functions.get(name)
        if fn is None:
            return {}
        return {
            k: v.model_dump() for k, v in fn.parameters.items()
        }

    def get_parameter_names(self, name: str) -> List[str]:
        """Return a list of parameter names for a function.

        Args:
            name: The function name.

        Returns:
            A list of parameter name strings.
        """
        fn = self.functions.get(name)
        if fn is None:
            return []
        return list(fn.parameters.keys())
