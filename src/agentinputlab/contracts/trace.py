"""Serializable method-specific execution traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import TracePhase
from .serialization import content_hash, freeze_mapping, to_jsonable


@dataclass(frozen=True, slots=True)
class TraceEvent:
    phase: TracePhase
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_mapping(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    method_id: str
    release_id: str
    events: tuple[TraceEvent, ...] = ()
    counters: Mapping[str, int | float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        events = tuple(self.events)
        warnings = tuple(self.warnings)
        if not all(isinstance(event, TraceEvent) for event in events):
            raise TypeError("trace events must be TraceEvent instances")
        if not all(isinstance(warning, str) for warning in warnings):
            raise TypeError("trace warnings must be strings")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "counters", freeze_mapping(self.counters))
        object.__setattr__(self, "warnings", warnings)

    def append(self, event: TraceEvent) -> ExecutionTrace:
        return ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(*self.events, event),
            counters=dict(self.counters),
            warnings=self.warnings,
        )

    @property
    def trace_hash(self) -> str:
        return content_hash(self)

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
