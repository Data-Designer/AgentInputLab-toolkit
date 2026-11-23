"""In-process tool connector intended for tests and trusted local tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.serialization import content_hash

from .base import ToolCall, ToolConnector, ToolResult, ToolSpec


@dataclass(slots=True)
class InMemoryToolConnector(ToolConnector):
    specs: tuple[ToolSpec, ...]
    functions: Mapping[str, Callable[..., Any]]
    connector_id: str = "memory-tools"
    calls: list[ToolCall] = field(default_factory=list, init=False)

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self.specs)

    def call(self, request: ToolCall) -> ToolResult:
        self.calls.append(request)
        function = self.functions.get(request.name)
        if function is None:
            return ToolResult(call_id=request.call_id, error=f"unknown tool: {request.name}")
        try:
            return ToolResult(
                call_id=request.call_id,
                output=function(**dict(request.arguments)),
            )
        except Exception as exc:
            return ToolResult(call_id=request.call_id, error=f"{type(exc).__name__}: {exc}")

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "connector_id": self.connector_id,
            "tool_schema_hash": content_hash(self.specs),
        }
