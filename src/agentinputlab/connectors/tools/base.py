"""Tool-calling contracts independent of MCP, OpenAPI, or benchmark runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import deep_freeze_jsonable, freeze_mapping


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    read_only: bool = True
    network: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("tool spec requires a non-empty name")
        object.__setattr__(self, "input_schema", freeze_mapping(self.input_schema))


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("tool call requires a non-empty name")
        object.__setattr__(self, "arguments", freeze_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    output: Any = None
    error: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", deep_freeze_jsonable(self.output))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class ToolConnector(Protocol):
    def list_tools(self) -> tuple[ToolSpec, ...]: ...

    def call(self, request: ToolCall) -> ToolResult: ...

    def fingerprint(self) -> Mapping[str, Any]: ...
