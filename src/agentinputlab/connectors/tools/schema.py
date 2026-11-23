"""Immutable, schema-only tool connector for planning interventions.

The connector exposes an audited snapshot of an arena's tool declarations but
cannot execute them.  Execution remains inside the official arena after the
input intervention has been materialized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash

from .base import ToolCall, ToolConnector, ToolResult, ToolSpec


@dataclass(frozen=True, slots=True)
class SchemaOnlyToolConnector(ToolConnector):
    """Expose frozen tool schemas while denying every execution request."""

    specs: tuple[ToolSpec, ...]
    connector_id: str = "schema-only-tools"

    def __post_init__(self) -> None:
        if not self.connector_id.strip():
            raise ValidationError("schema-only connector requires a non-empty ID")
        object.__setattr__(self, "specs", tuple(self.specs))
        if not self.specs:
            raise ValidationError("schema-only connector requires at least one tool")
        if any(not isinstance(spec, ToolSpec) for spec in self.specs):
            raise TypeError("schema-only connector specs must contain ToolSpec values")
        names = tuple(spec.name for spec in self.specs)
        if len(names) != len(set(names)):
            raise ValidationError("schema-only connector tool names must be unique")

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return self.specs

    def call(self, request: ToolCall) -> ToolResult:
        if not isinstance(request, ToolCall):
            raise TypeError("schema-only connector requires ToolCall")
        return ToolResult(
            call_id=request.call_id,
            error=(
                "schema-only connector cannot execute tools; execution belongs to "
                "the official downstream arena"
            ),
            metadata={"denied": True, "connector_id": self.connector_id},
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": "agentinputlab.schema_only_tool_connector.v1",
            "connector_id": self.connector_id,
            "execution_policy": "deny_all.v1",
            "tool_schema_hash": content_hash(self.specs),
        }

    @classmethod
    def from_openai_definitions(
        cls,
        definitions: Sequence[Mapping[str, Any]],
        *,
        connector_id: str = "schema-only-tools",
    ) -> SchemaOnlyToolConnector:
        """Normalize OpenAI-style or bare function declarations."""

        if isinstance(definitions, (str, bytes)) or not isinstance(definitions, Sequence):
            raise ValidationError("tool definitions must be a sequence")
        specs: list[ToolSpec] = []
        for index, raw in enumerate(definitions):
            if not isinstance(raw, Mapping):
                raise ValidationError(f"tool definition {index} must be an object")
            function = raw.get("function", raw)
            if not isinstance(function, Mapping):
                raise ValidationError(f"tool definition {index} function must be an object")
            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters", function.get("input_schema", {}))
            if not isinstance(name, str) or not name.strip():
                raise ValidationError(f"tool definition {index} requires a name")
            if not isinstance(description, str) or not description.strip():
                raise ValidationError(f"tool definition {index} requires a description")
            if not isinstance(parameters, Mapping):
                raise ValidationError(f"tool definition {index} parameters must be an object")
            specs.append(
                ToolSpec(
                    name=name,
                    description=description,
                    input_schema=dict(parameters),
                    # Schema-only snapshots do not assert mutability or networking.
                    read_only=False,
                    network=False,
                )
            )
        return cls(tuple(specs), connector_id=connector_id)


__all__ = ["SchemaOnlyToolConnector"]
