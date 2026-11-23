"""Small strict helpers shared by clarification algorithms."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from agentinputlab.contracts import AlgorithmOutputError


def parse_json_object(text: str, *, label: str) -> Mapping[str, Any]:
    """Parse one model-produced JSON object without accepting trailing prose."""

    if not isinstance(text, str) or not text.strip():
        raise AlgorithmOutputError(f"{label} response must be non-empty text")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AlgorithmOutputError(f"{label} response must be one JSON object") from exc
    if not isinstance(value, dict):
        raise AlgorithmOutputError(f"{label} response must be one JSON object")
    return value


def string_list(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AlgorithmOutputError(f"{label} must be a JSON array")
    items: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise AlgorithmOutputError(f"{label} entries must be non-empty strings")
        item = raw.strip()
        if item not in items:
            items.append(item)
    if not minimum <= len(items) <= maximum:
        raise AlgorithmOutputError(
            f"{label} must contain between {minimum} and {maximum} unique entries"
        )
    return tuple(items)


def visible_problem(state: Any) -> str:
    context = "\n\n".join(block.text for block in state.context) or "(none)"
    tools = "\n".join(f"- {name}" for name in state.tool_names) or "(none)"
    history = "\n".join(f"{item.role}: {item.content}" for item in state.messages)
    return (
        f"# User request\n{state.query}\n\n"
        f"# Instruction\n{state.instruction or '(none)'}\n\n"
        f"# Visible context\n{context}\n\n"
        f"# Available tools\n{tools}\n\n"
        f"# Visible conversation\n{history or '(none)'}"
    )
