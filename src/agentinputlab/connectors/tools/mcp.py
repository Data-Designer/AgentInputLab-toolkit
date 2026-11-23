"""MCP tool connector with an explicitly injected JSON-RPC-like transport."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, freeze_mapping

from .base import ToolCall, ToolConnector, ToolResult, ToolSpec
from .openapi import normalize_json_schema, validate_json_value


@runtime_checkable
class MCPTransport(Protocol):
    """Minimal injected client seam.

    A transport can wrap stdio, HTTP, a benchmark runtime, or an offline replay.
    AgentInputLab does not construct a network client implicitly.
    """

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


def normalize_mcp_tool(raw: Mapping[str, Any], *, network: bool = False) -> ToolSpec:
    """Normalize one MCP ``tools/list`` record into the common ToolSpec."""

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("MCP tool name must be a non-empty string")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValidationError(f"MCP tool description must be a string: {name}")
    schema = raw.get("inputSchema", raw.get("input_schema", {}))
    if not isinstance(schema, Mapping):
        raise ValidationError(f"MCP tool inputSchema must be an object: {name}")
    annotations = raw.get("annotations", {})
    if annotations is None:
        annotations = {}
    if not isinstance(annotations, Mapping):
        raise ValidationError(f"MCP tool annotations must be an object: {name}")
    # Missing annotations are treated conservatively, not as read-only.
    read_only = annotations.get("readOnlyHint") is True
    return ToolSpec(
        name=name.strip(),
        description=description,
        input_schema=normalize_json_schema(schema),
        read_only=read_only,
        network=network,
    )


def _unwrap(response: Mapping[str, Any]) -> Mapping[str, Any]:
    error = response.get("error")
    if error:
        raise ValidationError(f"MCP transport returned an error: {canonical_json(error)}")
    result = response.get("result", response)
    if not isinstance(result, Mapping):
        raise ValidationError("MCP response result must be an object")
    return result


@dataclass(slots=True)
class MCPToolConnector(ToolConnector):
    """Expose MCP server tools without binding to a specific MCP SDK."""

    transport: MCPTransport
    connector_id: str = "mcp"
    server_identity: Mapping[str, Any] = field(default_factory=dict)
    declared_tools: Iterable[Mapping[str, Any]] = ()
    allow_tools: tuple[str, ...] = ()
    allow_mutating: bool = False
    allow_network_calls: bool = False
    network: bool = True
    max_pages: int = 100
    _tools: tuple[ToolSpec, ...] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        declared = tuple(self.declared_tools)
        self.allow_tools = tuple(self.allow_tools)
        self.server_identity = freeze_mapping(self.server_identity)
        if self.max_pages < 1:
            raise ValidationError("MCP max_pages must be positive")
        if declared:
            self._tools = self._normalize_tools(declared)
        self.declared_tools = tuple(freeze_mapping(item) for item in declared)

    def _normalize_tools(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> tuple[ToolSpec, ...]:
        tools = tuple(
            tool
            for tool in (normalize_mcp_tool(item, network=self.network) for item in records)
            if "*" in self.allow_tools or tool.name in self.allow_tools
        )
        names = [item.name for item in tools]
        if len(names) != len(set(names)):
            raise ValidationError("MCP tools/list contains duplicate tool names")
        return tuple(sorted(tools, key=lambda item: item.name))

    def refresh_tools(self) -> tuple[ToolSpec, ...]:
        if self.network and not self.allow_network_calls:
            raise ValidationError("MCP network discovery is disabled by policy")
        records: list[Mapping[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(self.max_pages):
            params = {"cursor": cursor} if cursor else {}
            payload = _unwrap(self.transport.request("tools/list", params))
            page = payload.get("tools", [])
            if not isinstance(page, list) or not all(isinstance(item, Mapping) for item in page):
                raise ValidationError("MCP tools/list result.tools must be an array of objects")
            records.extend(page)
            next_cursor = payload.get("nextCursor")
            if next_cursor is None:
                self._tools = self._normalize_tools(records)
                return self._tools
            if not isinstance(next_cursor, str) or not next_cursor:
                raise ValidationError("MCP nextCursor must be a non-empty string")
            if next_cursor in seen_cursors:
                raise ValidationError("MCP tools/list pagination repeated a cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        raise ValidationError("MCP tools/list exceeded max_pages")

    def list_tools(self) -> tuple[ToolSpec, ...]:
        if self._tools is None:
            return self.refresh_tools()
        return self._tools

    def call(self, request: ToolCall) -> ToolResult:
        if self.network and not self.allow_network_calls:
            return ToolResult(
                call_id=request.call_id,
                error=f"network connector is disabled by policy: {self.connector_id}",
            )
        known = {tool.name: tool for tool in self.list_tools()}
        tool = known.get(request.name)
        if tool is None:
            return ToolResult(call_id=request.call_id, error=f"unknown tool: {request.name}")
        if not tool.read_only and not self.allow_mutating:
            return ToolResult(
                call_id=request.call_id,
                error=f"mutating tool is disabled by policy: {request.name}",
            )
        if tool.network and not self.allow_network_calls:
            return ToolResult(
                call_id=request.call_id,
                error=f"network tool is disabled by policy: {request.name}",
            )
        properties = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])
        unknown = sorted(set(request.arguments) - set(properties))
        missing = sorted(set(required) - set(request.arguments))
        if unknown:
            return ToolResult(
                call_id=request.call_id,
                error=f"unknown arguments for {request.name}: {', '.join(unknown)}",
            )
        if missing:
            return ToolResult(
                call_id=request.call_id,
                error=f"missing required arguments for {request.name}: {', '.join(missing)}",
            )
        try:
            for name, value in request.arguments.items():
                schema = properties.get(name, {})
                if isinstance(schema, Mapping):
                    validate_json_value(value, schema, path=name)
        except ValidationError as exc:
            return ToolResult(call_id=request.call_id, error=str(exc))
        try:
            payload = _unwrap(
                self.transport.request(
                    "tools/call",
                    {"name": request.name, "arguments": dict(request.arguments)},
                )
            )
            if payload.get("isError") is True:
                return ToolResult(
                    call_id=request.call_id,
                    error=canonical_json(payload.get("content", payload)),
                    metadata={"connector": "mcp", "tool": request.name},
                )
            output = payload.get("content", payload)
            return ToolResult(
                call_id=request.call_id,
                output=output,
                metadata={"connector": "mcp", "tool": request.name},
            )
        except Exception as exc:
            return ToolResult(
                call_id=request.call_id,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"connector": "mcp", "tool": request.name},
            )

    def fingerprint(self) -> Mapping[str, Any]:
        tools = self._tools or ()
        return {
            "connector_id": self.connector_id,
            "protocol": "mcp",
            "implementation": "agentinputlab.mcp.v1",
            "server_identity_hash": content_hash(self.server_identity),
            "tool_schema_hash": content_hash(tools),
            "tool_count": len(tools),
            "schema_state": "observed" if self._tools is not None else "unresolved",
            "network": self.network,
            "allow_tools": sorted(self.allow_tools),
            "allow_mutating": self.allow_mutating,
            "allow_network_calls": self.allow_network_calls,
        }


__all__ = ["MCPToolConnector", "MCPTransport", "normalize_mcp_tool"]
