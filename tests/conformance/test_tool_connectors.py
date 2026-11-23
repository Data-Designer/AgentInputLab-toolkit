"""Conformance tests for dependency-free MCP/OpenAPI/Python tool seams."""

from __future__ import annotations

import math
import unittest
from collections.abc import Mapping
from typing import Literal

from agentinputlab.connectors.tools import (
    MCPToolConnector,
    OpenAPIToolConnector,
    ToolCall,
    normalize_json_schema,
    python_tool_spec,
    validate_json_value,
)


class FakeMCPTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, Mapping[str, object]]] = []

    def request(
        self,
        method: str,
        params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        params = params or {}
        self.requests.append((method, params))
        if method == "tools/list":
            return {
                "result": {
                    "tools": [
                        {
                            "name": "lookup",
                            "description": "Look up a record.",
                            "inputSchema": {
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        }
                    ]
                }
            }
        if method == "tools/call":
            return {
                "result": {
                    "content": [{"type": "text", "text": f"found:{params['arguments']['query']}"}]
                }
            }
        raise AssertionError(method)


class FakeOpenAPITransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, object],
        headers: Mapping[str, str],
        body: object,
    ) -> object:
        request = {
            "method": method,
            "path": path,
            "query": dict(query),
            "headers": dict(headers),
            "body": body,
        }
        self.requests.append(request)
        return {"ok": True, **request}


OPENAPI_DOCUMENT = {
    "openapi": "3.1.0",
    "info": {"title": "Fixture", "version": "1"},
    "components": {
        "schemas": {
            "CreatePet": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["dog", "cat"]},
                },
                "required": ["name"],
            }
        }
    },
    "paths": {
        "/pets": {
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/CreatePet"}}
                    },
                },
            }
        },
        "/pets/{pet_id}": {
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet",
                "parameters": [
                    {
                        "name": "pet_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "verbose",
                        "in": "query",
                        "schema": {"type": "boolean", "nullable": True},
                    },
                ],
            }
        },
    },
}


class ToolConnectorConformanceTests(unittest.TestCase):
    def test_mcp_construction_and_fingerprint_do_not_call_transport(self) -> None:
        transport = FakeMCPTransport()
        connector = MCPToolConnector(
            transport=transport,
            connector_id="fixture",
            server_identity={"version": "1", "name": "server"},
            allow_tools=("lookup",),
        )
        first = connector.fingerprint()
        second = connector.fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(transport.requests, [])
        self.assertEqual(first["schema_state"], "unresolved")

    def test_mcp_schema_discovery_and_call(self) -> None:
        transport = FakeMCPTransport()
        connector = MCPToolConnector(
            transport=transport,
            allow_tools=("lookup",),
            allow_mutating=True,
            network=False,
        )
        tools = connector.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "lookup")
        self.assertFalse(tools[0].read_only, "missing annotation must fail safe")
        result = connector.call(
            ToolCall(name="lookup", arguments={"query": "Ada"}, call_id="call-1")
        )
        self.assertFalse(result.error)
        self.assertEqual(result.call_id, "call-1")
        self.assertEqual(transport.requests[0][0], "tools/list")
        self.assertEqual(transport.requests[1][0], "tools/call")

    def test_mcp_required_argument_is_checked_before_call(self) -> None:
        transport = FakeMCPTransport()
        connector = MCPToolConnector(
            transport=transport,
            allow_tools=("lookup",),
            allow_mutating=True,
            network=False,
        )
        connector.list_tools()
        result = connector.call(ToolCall(name="lookup", arguments={}, call_id="bad"))
        self.assertIn("missing required", result.error)
        self.assertEqual([method for method, _ in transport.requests], ["tools/list"])

    def test_mcp_declared_tools_are_offline_and_normalized(self) -> None:
        transport = FakeMCPTransport()
        connector = MCPToolConnector(
            transport=transport,
            declared_tools=[
                {
                    "name": "safe",
                    "inputSchema": {
                        "required": ["mode"],
                        "properties": {"mode": {"enum": ["z", "a"], "type": "string"}},
                    },
                    "annotations": {"readOnlyHint": True},
                }
            ],
            allow_tools=("safe",),
            network=False,
        )
        tool = connector.list_tools()[0]
        self.assertEqual(transport.requests, [])
        self.assertTrue(tool.read_only)
        self.assertEqual(tool.input_schema["properties"]["mode"]["enum"], ("a", "z"))

    def test_openapi_inspection_is_offline_and_call_maps_parameters(self) -> None:
        transport = FakeOpenAPITransport()
        connector = OpenAPIToolConnector(
            document=OPENAPI_DOCUMENT,
            transport=transport,
            connector_id="pets",
            base_url_identity="fixture://pets-v1",
            allow_operations=("getPet", "createPet"),
            allow_mutating=True,
            network=False,
        )
        fingerprint = connector.fingerprint()
        tools = {tool.name: tool for tool in connector.list_tools()}
        self.assertEqual(transport.requests, [])
        self.assertEqual(fingerprint["operation_count"], 2)
        self.assertTrue(tools["getPet"].read_only)
        self.assertFalse(tools["createPet"].read_only)
        self.assertEqual(
            tools["getPet"].input_schema["properties"]["verbose"]["type"],
            ("boolean", "null"),
        )

        result = connector.call(
            ToolCall(
                name="getPet",
                arguments={"pet_id": "a/b", "verbose": True},
                call_id="openapi-1",
            )
        )
        self.assertFalse(result.error)
        self.assertEqual(transport.requests[0]["path"], "/pets/a%2Fb")
        self.assertEqual(transport.requests[0]["query"], {"verbose": True})

        result = connector.call(
            ToolCall(
                name="createPet",
                arguments={"body": {"name": "Mochi", "kind": "cat"}},
                call_id="openapi-2",
            )
        )
        self.assertFalse(result.error)
        self.assertEqual(transport.requests[1]["body"], {"name": "Mochi", "kind": "cat"})

    def test_openapi_missing_required_argument_fails_before_transport(self) -> None:
        transport = FakeOpenAPITransport()
        connector = OpenAPIToolConnector(
            document=OPENAPI_DOCUMENT,
            transport=transport,
            allow_operations=("getPet",),
            network=False,
        )
        result = connector.call(ToolCall(name="getPet", arguments={}, call_id="bad"))
        self.assertIn("missing required", result.error)
        self.assertEqual(transport.requests, [])

    def test_tool_connectors_default_to_an_empty_allowlist(self) -> None:
        mcp = MCPToolConnector(
            transport=FakeMCPTransport(),
            declared_tools=[
                {
                    "name": "safe",
                    "inputSchema": {"type": "object", "properties": {}},
                    "annotations": {"readOnlyHint": True},
                }
            ],
        )
        openapi = OpenAPIToolConnector(
            document=OPENAPI_DOCUMENT,
            transport=FakeOpenAPITransport(),
        )
        self.assertEqual(mcp.list_tools(), ())
        self.assertEqual(openapi.list_tools(), ())

    def test_mutating_openapi_operation_requires_explicit_policy(self) -> None:
        transport = FakeOpenAPITransport()
        connector = OpenAPIToolConnector(
            document=OPENAPI_DOCUMENT,
            transport=transport,
            allow_operations=("createPet",),
        )
        result = connector.call(
            ToolCall(
                name="createPet",
                arguments={"body": {"name": "Mochi", "kind": "cat"}},
                call_id="mutating",
            )
        )
        self.assertIn("mutating tool is disabled", result.error)
        self.assertEqual(transport.requests, [])

    def test_openapi_schema_validation_precedes_transport(self) -> None:
        transport = FakeOpenAPITransport()
        connector = OpenAPIToolConnector(
            document=OPENAPI_DOCUMENT,
            transport=transport,
            allow_operations=("createPet",),
            allow_mutating=True,
            network=False,
        )
        result = connector.call(
            ToolCall(
                name="createPet",
                arguments={"body": {"name": "Mochi", "kind": "hamster"}},
                call_id="schema",
            )
        )
        self.assertIn("allowed enum", result.error)
        self.assertEqual(transport.requests, [])

    def test_network_connectors_fail_closed_by_default(self) -> None:
        mcp_transport = FakeMCPTransport()
        mcp = MCPToolConnector(
            transport=mcp_transport,
            allow_tools=("lookup",),
        )
        with self.assertRaisesRegex(ValueError, "network discovery"):
            mcp.list_tools()
        self.assertEqual(mcp_transport.requests, [])

        openapi_transport = FakeOpenAPITransport()
        openapi = OpenAPIToolConnector(
            document=OPENAPI_DOCUMENT,
            transport=openapi_transport,
            allow_operations=("getPet",),
        )
        result = openapi.call(ToolCall(name="getPet", arguments={"pet_id": "1"}, call_id="network"))
        self.assertIn("network tool is disabled", result.error)
        self.assertEqual(openapi_transport.requests, [])

    def test_schema_allof_and_typed_additional_properties_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_json_value("wrong", {"allOf": [{"type": "integer"}]})
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_json_value(
                {"bad": "wrong"},
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {"type": "integer"},
                },
            )

    def test_openapi_rejects_paths_with_embedded_query_or_fragment(self) -> None:
        for path in ("/safe?redirect=https://evil.test", "/safe#fragment", "/safe\\evil"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "root-relative"):
                OpenAPIToolConnector(
                    document={
                        "openapi": "3.1.0",
                        "info": {"title": "bad", "version": "1"},
                        "paths": {path: {"get": {"operationId": "bad"}}},
                    },
                    transport=FakeOpenAPITransport(),
                    allow_operations=("bad",),
                )

    def test_python_tool_schema_normalization(self) -> None:
        def search(
            query: str,
            limit: int = 5,
            mode: Literal["semantic", "lexical"] = "semantic",
        ) -> list[str]:
            """Search a fixture corpus."""

            return [query] * limit

        spec = python_tool_spec(search)
        self.assertEqual(spec.name, "search")
        self.assertEqual(spec.input_schema["required"], ("query",))
        self.assertEqual(spec.input_schema["properties"]["limit"]["type"], "integer")
        self.assertEqual(
            spec.input_schema["properties"]["mode"]["enum"],
            ("lexical", "semantic"),
        )
        self.assertEqual(spec.input_schema["properties"]["limit"]["default"], 5)
        self.assertFalse(spec.read_only, "Python tools require an explicit read-only declaration")

    def test_schema_fingerprint_is_order_independent(self) -> None:
        left = normalize_json_schema(
            {
                "properties": {
                    "b": {"enum": [2, 1]},
                    "a": {"type": "string"},
                },
                "required": ["b", "a"],
            }
        )
        right = normalize_json_schema(
            {
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "string"},
                    "b": {"enum": [1, 2]},
                },
            }
        )
        self.assertEqual(left, right)
        self.assertFalse(math.isnan(float(len(str(left)))))


if __name__ == "__main__":
    unittest.main()
