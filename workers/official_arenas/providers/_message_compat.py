"""Narrow message-shape compatibility helpers for self-hosted model templates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def requires_qwen38_initial_system_message(model: Any) -> bool:
    """Return whether the served alias uses Qwen3.8's strict system placement."""

    if not isinstance(model, str):
        return False
    alias = model.strip().casefold().replace("_", "-")
    return alias == "qwen3.8" or alias.startswith("qwen3.8-")


def normalize_qwen38_reasoning_effort(model: Any, effort: Any) -> Any:
    """Map STATE-Bench's strongest effort label to Qwen3.8's equivalent."""

    if requires_qwen38_initial_system_message(model) and effort == "high":
        return "xhigh"
    return effort


def coalesce_chat_system_messages(messages: Sequence[Any]) -> list[Any]:
    """Move chat system content into one leading message without reordering peers."""

    copied = [dict(item) if isinstance(item, Mapping) else item for item in messages]
    system_items: list[dict[str, Any]] = []
    non_system_items: list[Any] = []
    for item in copied:
        if isinstance(item, Mapping) and item.get("role") == "system":
            content = item.get("content")
            if not isinstance(content, str):
                raise TypeError("Qwen3.8 system message content must be text")
            system_items.append(dict(item))
        else:
            non_system_items.append(item)
    if not system_items:
        return copied
    merged = dict(system_items[0])
    merged["content"] = "\n\n".join(item["content"] for item in system_items)
    return [merged, *non_system_items]


def fold_responses_system_messages_into_instructions(
    input_items: Sequence[Any],
    instructions: Any,
) -> tuple[list[Any], Any]:
    """Fold Responses input system items into its single instructions field.

    Qwen3.8's chat template accepts only the leading system slot.  The Responses
    transport renders ``instructions`` there, so later system-role input items
    must be removed and appended to that field in their original order.
    """

    copied = [dict(item) if isinstance(item, Mapping) else item for item in input_items]
    system_contents: list[str] = []
    non_system_items: list[Any] = []
    for item in copied:
        if isinstance(item, Mapping) and item.get("role") == "system":
            content = item.get("content")
            if not isinstance(content, str):
                raise TypeError("Qwen3.8 Responses system item content must be text")
            system_contents.append(content)
        else:
            non_system_items.append(item)
    if not system_contents:
        return copied, instructions
    if instructions is not None and not isinstance(instructions, str):
        raise TypeError("Qwen3.8 Responses instructions must be text or None")
    parts = []
    if instructions:
        parts.append(instructions)
    parts.extend(system_contents)
    return non_system_items, "\n\n".join(parts)
