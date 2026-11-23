"""Immutable input state presented to an intervention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError
from .serialization import content_hash, freeze_mapping, to_jsonable


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    try:
        return freeze_mapping(value)
    except TypeError as exc:
        raise ValidationError(f"state metadata must be canonically serializable: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValidationError("message.role must not be empty")


@dataclass(frozen=True, slots=True)
class ContextBlock:
    text: str
    block_id: str = ""
    source: str = ""
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        if not self.text:
            raise ValidationError("context block text must not be empty")


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Per-run resource ceilings.

    ``max_context_tokens`` is the cumulative number of provider-reported LLM
    input tokens across every model call in one run.  It is not a per-request
    window-size limit; a run with two 600-token requests consumes 1,200 context
    tokens for this budget.
    """

    max_model_calls: int = 8
    max_retrieval_calls: int = 8
    max_tool_calls: int = 8
    max_output_tokens: int = 4096
    max_context_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_model_calls",
            "max_retrieval_calls",
            "max_tool_calls",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a non-negative integer")
        if self.max_context_tokens is not None and (
            isinstance(self.max_context_tokens, bool)
            or not isinstance(self.max_context_tokens, int)
            or self.max_context_tokens < 0
        ):
            raise ValidationError("max_context_tokens must be a non-negative integer or None")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValidationError("max_cost_usd must be finite and non-negative or None")


@dataclass(frozen=True, slots=True)
class FrozenAgentState:
    """A hash-addressed snapshot used for paired pre-action experiments."""

    query: str
    task_id: str = ""
    instruction: str = ""
    context: tuple[ContextBlock, ...] = ()
    messages: tuple[Message, ...] = ()
    tool_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not isinstance(self.instruction, str):
            raise ValidationError("state query and instruction must be strings")
        if not self.query.strip() and not self.instruction.strip():
            raise ValidationError("state requires a non-empty query or instruction")
        context = tuple(self.context)
        messages = tuple(self.messages)
        tool_names = tuple(self.tool_names)
        if not all(isinstance(item, ContextBlock) for item in context):
            raise ValidationError("state context entries must be ContextBlock instances")
        if not all(isinstance(item, Message) for item in messages):
            raise ValidationError("state messages must be Message instances")
        if not all(isinstance(item, str) and item for item in tool_names):
            raise ValidationError("state tool_names must contain non-empty strings")
        if not isinstance(self.budget, ExecutionBudget):
            raise ValidationError("state budget must be an ExecutionBudget")
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tool_names", tool_names)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def canonical_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @property
    def state_hash(self) -> str:
        return content_hash(self)
