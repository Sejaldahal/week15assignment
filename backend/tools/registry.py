"""
Tool registry: maps a tool name to (json_schema, callable). The
orchestrator passes the schemas to the LLM provider and, when the model
requests a call, looks up and executes the corresponding function here -
after Pydantic validation inside each tool's own `execute()`.
"""
from __future__ import annotations

from typing import Any, Callable

from backend.tools import calculator, time_tool

_REGISTRY: dict[str, Callable[[dict], Any]] = {
    calculator.SCHEMA["name"]: calculator.execute,
    time_tool.SCHEMA["name"]: time_tool.execute,
}

TOOL_SCHEMAS: list[dict] = [calculator.SCHEMA, time_tool.SCHEMA]


def execute_tool(name: str, args: dict) -> Any:
    if name not in _REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _REGISTRY[name](args)
    except Exception as exc:  # noqa: BLE001 - never let a bad tool call crash the request
        return {"error": f"Tool execution failed: {exc}"}
