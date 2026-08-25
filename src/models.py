"""Pydantic models for input validation and output formatting.

These models use Pydantic to validate data at runtime:
- PromptInput: validates each test prompt from the input file
- ParameterDef: validates a parameter's type definition
- FunctionDef: validates a complete function definition
- FunctionCall: validates the output format for a function call result
"""
from pydantic import BaseModel
from typing import Any, Dict


class PromptInput(BaseModel):
    """Validates a single test prompt from the input file.

    Expected JSON format: {"prompt": "What is the sum of 2 and 3?"}
    """

    prompt: str


class ParameterDef(BaseModel):
    """Validates a single parameter definition.

    Expected JSON format: {"type": "number"} or {"type": "string"}
    """

    type: str


class FunctionDef(BaseModel):
    """Validates a function definition from the definitions file.

    Expected JSON format:
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"}
    }
    """

    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class FunctionCall(BaseModel):
    """Validates the output format for a single function call result.

    This is what we write to the output JSON file.
    Expected JSON format:
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2, "b": 3}
    }
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]
