"""Pydantic models for input validation and output formatting."""
from pydantic import BaseModel
from typing import Any, Dict


class PromptInput(BaseModel):
    """Validates a single test prompt from the input file."""

    prompt: str


class ParameterDef(BaseModel):
    """Validates a single parameter definition."""

    type: str


class FunctionDef(BaseModel):
    """Validates a function definition from the definitions file."""

    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class FunctionCall(BaseModel):
    """The output format for a single function call result."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
