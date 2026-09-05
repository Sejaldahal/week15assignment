"""
Calculator tool. Evaluates arithmetic expressions using a restricted AST
walk - NOT eval() or exec() - so arbitrary code execution is impossible.
Only numeric literals and +, -, *, /, %, **, parentheses, and unary +/-
are permitted.
"""
from __future__ import annotations

import ast
import operator
from pydantic import BaseModel, Field

SCHEMA = {
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression and return the numeric result.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression, e.g. '25 * 4 + 10'",
            }
        },
        "required": ["expression"],
    },
}

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorInput(BaseModel):
    expression: str = Field(..., min_length=1, max_length=200)


class CalculatorError(Exception):
    pass


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        try:
            return _ALLOWED_BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def execute(args: dict) -> dict:
    validated = CalculatorInput.model_validate(args)
    try:
        tree = ast.parse(validated.expression, mode="eval")
        result = _eval_node(tree)
    except (SyntaxError, CalculatorError) as exc:
        return {"error": str(exc)}
    return {"expression": validated.expression, "result": result}
