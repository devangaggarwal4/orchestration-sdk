"""Deterministic fake tools for agentic eval tasks."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from hybrid_router.types import ToolSpec

WEATHER = {"paris": 18, "oslo": 4, "madrid": 27, "tokyo": 21, "london": 12}
ORDERS = {
    "A-1042": {"status": "shipped", "carrier": "DHL", "eta": "2026-10-14"},
    "A-2077": {"status": "processing", "carrier": None, "eta": "2026-10-20"},
}
DOCS = [
    (
        "gateway limits",
        "The Gateway service accepts requests up to 25 MB. See changelog for history.",
    ),
    ("gateway changelog", "Gateway v3.2 raised the maximum request size from 10 MB to 25 MB."),
    ("gateway auth", "Gateway uses OAuth2 client credentials; tokens expire after 1 hour."),
    ("billing", "Invoices are issued on the 1st of each month in EUR."),
]


def _weather(city: str) -> str:
    temp = WEATHER.get(city.strip().lower())
    return f"{city}: {temp}°C, clear" if temp is not None else f"error: unknown city {city!r}"


def _order(order_id: str) -> str:
    o = ORDERS.get(order_id.strip().upper())
    if o is None:
        return f"error: no order {order_id!r}"
    return f"order {order_id}: status={o['status']}, carrier={o['carrier']}, eta={o['eta']}"


def _search(query: str) -> str:
    words = {w for w in query.lower().split() if len(w) > 2}
    hits = [text for title, text in DOCS if words & set((title + " " + text).lower().split())]
    return "\n".join(hits[:2]) or "no results"


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _calc(expression: str) -> str:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"error: {e}"


def _spec(name: str, description: str, prop: str, prop_desc: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {prop: {"type": "string", "description": prop_desc}},
            "required": [prop],
        },
    )


TOOLS: dict[str, tuple[ToolSpec, Callable[..., str]]] = {
    "get_weather": (
        _spec("get_weather", "Current temperature for a city.", "city", "City name"),
        _weather,
    ),
    "lookup_order": (
        _spec("lookup_order", "Status of an order.", "order_id", "e.g. A-1042"),
        _order,
    ),
    "search_docs": (
        _spec("search_docs", "Search internal documentation.", "query", "Keywords"),
        _search,
    ),
    "calculator": (
        _spec("calculator", "Evaluate an arithmetic expression.", "expression", "e.g. (3+4)*2"),
        _calc,
    ),
}


def run_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Returns (result text, is_error)."""
    if name not in TOOLS:
        return f"error: unknown tool {name!r}", True
    _, fn = TOOLS[name]
    try:
        out = fn(**args)
    except TypeError as e:
        return f"error: bad arguments: {e}", True
    return out, out.startswith("error:")
