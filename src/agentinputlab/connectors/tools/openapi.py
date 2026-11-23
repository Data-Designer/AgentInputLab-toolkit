"""OpenAPI and Python-callable tool normalization with injected transport.

No HTTP implementation is bundled here.  Calling an OpenAPI operation requires
an explicit transport object, which keeps imports, construction, schema
inspection, and fingerprinting offline by default.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Mapping as ABCMapping
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Literal,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)
from urllib.parse import quote, urlsplit

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import (
    canonical_json,
    content_hash,
    freeze_mapping,
    to_jsonable,
)

from .base import ToolCall, ToolConnector, ToolResult, ToolSpec

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "connection",
        "cookie",
        "host",
        "proxy-authorization",
        "set-cookie",
        "transfer-encoding",
        "x-api-key",
    }
)


def _json_pointer(document: Mapping[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValidationError(f"only local OpenAPI references are supported: {ref}")
    value: Any = document
    for raw_segment in ref[2:].split("/"):
        segment = raw_segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or segment not in value:
            raise ValidationError(f"unresolved OpenAPI reference: {ref}")
        value = value[segment]
    return value


def _normalize_schema_node(
    value: Any,
    *,
    document: Mapping[str, Any] | None,
    resolving: frozenset[str],
) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalize_schema_node(item, document=document, resolving=resolving) for item in value
        ]
    if not isinstance(value, Mapping):
        return to_jsonable(value)

    if "$ref" in value:
        ref = value["$ref"]
        if not isinstance(ref, str):
            raise ValidationError("JSON Schema $ref must be a string")
        if document is None or not ref.startswith("#/") or ref in resolving:
            return {"$ref": ref}
        resolved = _json_pointer(document, ref)
        normalized = _normalize_schema_node(
            resolved,
            document=document,
            resolving=resolving | {ref},
        )
        siblings = {key: item for key, item in value.items() if key != "$ref"}
        if not siblings:
            return normalized
        return {
            "allOf": [
                normalized,
                _normalize_schema_node(siblings, document=document, resolving=resolving),
            ]
        }

    result: dict[str, Any] = {}
    nullable = value.get("nullable") is True
    for key in sorted(value):
        if key in {"nullable", "readOnly", "writeOnly", "xml", "externalDocs"}:
            continue
        item = value[key]
        if key == "properties":
            if not isinstance(item, Mapping):
                raise ValidationError("JSON Schema properties must be an object")
            result[key] = {
                name: _normalize_schema_node(
                    item[name],
                    document=document,
                    resolving=resolving,
                )
                for name in sorted(item)
            }
        elif key == "required":
            if not isinstance(item, (list, tuple)) or not all(
                isinstance(name, str) for name in item
            ):
                raise ValidationError("JSON Schema required must be an array of strings")
            result[key] = sorted(set(item))
        elif key == "enum":
            if not isinstance(item, (list, tuple)):
                raise ValidationError("JSON Schema enum must be an array")
            result[key] = sorted(
                (to_jsonable(member) for member in item),
                key=canonical_json,
            )
        else:
            result[key] = _normalize_schema_node(
                item,
                document=document,
                resolving=resolving,
            )

    if "properties" in result and "type" not in result:
        result["type"] = "object"
    if nullable:
        existing = result.get("type")
        if isinstance(existing, str):
            result["type"] = [existing, "null"]
        elif isinstance(existing, list):
            result["type"] = sorted(set(existing) | {"null"})
        elif "anyOf" in result:
            result["anyOf"] = [*result["anyOf"], {"type": "null"}]
        else:
            result["anyOf"] = [{"type": "null"}, {}]
    return result


def normalize_json_schema(
    schema: Mapping[str, Any] | None,
    *,
    document: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a stable, local-ref-resolved JSON Schema mapping."""

    if schema is None:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    if not isinstance(schema, Mapping):
        raise ValidationError("tool input schema must be an object")
    normalized = _normalize_schema_node(schema, document=document, resolving=frozenset())
    if not isinstance(normalized, Mapping):
        raise ValidationError("normalized tool input schema must be an object")
    if not normalized:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    return normalized


def validate_json_value(value: Any, schema: Mapping[str, Any], *, path: str = "value") -> None:
    """Validate the conservative JSON-Schema subset exposed by connectors."""

    if "allOf" in schema:
        variants = schema["allOf"]
        if not isinstance(variants, (list, tuple)) or not all(
            isinstance(variant, Mapping) for variant in variants
        ):
            raise ValidationError(f"{path}: allOf must be an array of schemas")
        for variant in variants:
            validate_json_value(value, variant, path=path)
    if "oneOf" in schema:
        variants = schema["oneOf"]
        if not isinstance(variants, (list, tuple)):
            raise ValidationError(f"{path}: oneOf must be an array")
        matches = 0
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            try:
                validate_json_value(value, variant, path=path)
                matches += 1
            except ValidationError:
                continue
        if matches != 1:
            raise ValidationError(f"{path}: value must match exactly one allowed schema")
    if "anyOf" in schema:
        variants = schema["anyOf"]
        if not isinstance(variants, (list, tuple)):
            raise ValidationError(f"{path}: anyOf must be an array")
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            try:
                validate_json_value(value, variant, path=path)
                break
            except ValidationError:
                continue
        else:
            raise ValidationError(f"{path}: value does not match any allowed schema")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: value is not in the allowed enum")
    expected = schema.get("type")
    expected_types = {expected} if isinstance(expected, str) else set(expected or ())
    if expected_types and not _matches_json_type(value, expected_types):
        raise ValidationError(f"{path}: expected JSON type {'|'.join(sorted(expected_types))}")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            required = set(schema.get("required", ()))
            missing = sorted(required - set(value))
            if missing:
                raise ValidationError(f"{path}: missing required keys {', '.join(missing)}")
            if schema.get("additionalProperties") is False:
                unknown = sorted(set(value) - set(properties))
                if unknown:
                    raise ValidationError(f"{path}: unknown keys {', '.join(unknown)}")
            for key, item in value.items():
                child = properties.get(key)
                if isinstance(child, Mapping):
                    validate_json_value(item, child, path=f"{path}.{key}")
                elif key not in properties and isinstance(
                    schema.get("additionalProperties"), Mapping
                ):
                    validate_json_value(
                        item,
                        schema["additionalProperties"],
                        path=f"{path}.{key}",
                    )
    if isinstance(value, (list, tuple)):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                validate_json_value(item, item_schema, path=f"{path}[{index}]")
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValidationError(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValidationError(f"{path}: array is longer than maxItems")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValidationError(f"{path}: string is shorter than minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValidationError(f"{path}: string is longer than maxLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise ValidationError(f"{path}: string does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValidationError(f"{path}: number must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"{path}: number is above maximum")


def _matches_json_type(value: Any, expected: set[str]) -> bool:
    return any(
        (
            (kind == "null" and value is None)
            or (kind == "boolean" and isinstance(value, bool))
            or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (
                kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            or (kind == "string" and isinstance(value, str))
            or (kind == "array" and isinstance(value, (list, tuple)))
            or (kind == "object" and isinstance(value, Mapping))
        )
        for kind in expected
    )


def _annotation_schema(annotation: Any) -> Mapping[str, Any]:
    if annotation in {Any, inspect.Signature.empty}:
        return {}
    if annotation is None or annotation is type(None):
        return {"type": "null"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        values = [to_jsonable(item) for item in arguments]
        return {"enum": sorted(values, key=canonical_json)}
    if origin in {Union, types.UnionType}:
        variants = [_annotation_schema(item) for item in arguments]
        return {"anyOf": variants}
    if origin in {list, tuple, set, frozenset, Sequence, Iterable}:
        item_schema = _annotation_schema(arguments[0]) if arguments else {}
        return {"type": "array", "items": item_schema}
    if origin in {dict, Mapping, ABCMapping}:
        value_schema = _annotation_schema(arguments[1]) if len(arguments) > 1 else {}
        return {"type": "object", "additionalProperties": value_schema}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [to_jsonable(item.value) for item in annotation]
        return {"enum": sorted(values, key=canonical_json)}
    if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation):
        properties: dict[str, Any] = {}
        required: list[str] = []
        for item in dataclasses.fields(annotation):
            properties[item.name] = _annotation_schema(item.type)
            if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
                required.append(item.name)
        result: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = sorted(required)
        return result
    return {"type": "string", "x-python-type": getattr(annotation, "__name__", str(annotation))}


def python_tool_spec(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
    network: bool = False,
) -> ToolSpec:
    """Derive a deterministic tool schema from a typed Python callable."""

    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = dict(getattr(function, "__annotations__", {}))

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValidationError("variadic Python tools require an explicit ToolSpec")
        schema = dict(_annotation_schema(hints.get(parameter_name, parameter.annotation)))
        if parameter.default is inspect.Signature.empty:
            required.append(parameter_name)
        else:
            with suppress(TypeError):
                schema["default"] = to_jsonable(parameter.default)
        properties[parameter_name] = schema
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        input_schema["required"] = sorted(required)
    tool_name = name or function.__name__
    if not tool_name or not isinstance(tool_name, str):
        raise ValidationError("Python tool name must be a non-empty string")
    return ToolSpec(
        name=tool_name,
        description=description if description is not None else (inspect.getdoc(function) or ""),
        input_schema=normalize_json_schema(input_schema),
        read_only=read_only,
        network=network,
    )


@runtime_checkable
class OpenAPITransport(Protocol):
    """Injected request executor; implementations may be HTTP, mock, or replay."""

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any],
        headers: Mapping[str, str],
        body: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _Operation:
    tool: ToolSpec
    method: str
    path: str
    argument_locations: Mapping[str, tuple[str, str]]
    body_argument: str | None

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "tool": self.tool,
            "method": self.method,
            "path": self.path,
            "argument_locations": self.argument_locations,
            "body_argument": self.body_argument,
        }


def _resolve_object(value: Any, document: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("OpenAPI component must be an object")
    if "$ref" in value:
        ref = value["$ref"]
        if not isinstance(ref, str):
            raise ValidationError("OpenAPI $ref must be a string")
        resolved = _json_pointer(document, ref)
        if not isinstance(resolved, Mapping):
            raise ValidationError(f"OpenAPI reference does not resolve to an object: {ref}")
        return resolved
    return value


def _operation_name(method: str, path: str, operation: Mapping[str, Any]) -> str:
    operation_id = operation.get("operationId")
    if isinstance(operation_id, str) and operation_id.strip():
        return _SAFE_NAME.sub("_", operation_id.strip()).strip("_")
    stem = _SAFE_NAME.sub("_", path.strip("/")).strip("_") or "root"
    return f"{method.lower()}_{stem}"


def _operation_description(operation: Mapping[str, Any]) -> str:
    summary = operation.get("summary")
    description = operation.get("description")
    parts = [
        item.strip() for item in (summary, description) if isinstance(item, str) and item.strip()
    ]
    return "\n\n".join(parts)


@dataclass(slots=True)
class OpenAPIToolConnector(ToolConnector):
    """Expose selected OpenAPI operations through the common tool contract."""

    document: Mapping[str, Any]
    transport: OpenAPITransport
    connector_id: str = "openapi"
    base_url_identity: str = ""
    allow_operations: tuple[str, ...] = ()
    allow_mutating: bool = False
    allow_network_calls: bool = False
    network: bool = True
    _operations: Mapping[str, _Operation] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.allow_operations = tuple(self.allow_operations)
        self._operations = self._build_operations()
        self.document = freeze_mapping(self.document)

    def _build_operations(self) -> Mapping[str, _Operation]:
        paths = self.document.get("paths", {})
        if not isinstance(paths, Mapping):
            raise ValidationError("OpenAPI document.paths must be an object")
        operations: dict[str, _Operation] = {}
        for path in sorted(paths):
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or path.startswith("//")
                or "\\" in path
                or urlsplit(path).scheme
                or urlsplit(path).netloc
                or urlsplit(path).query
                or urlsplit(path).fragment
            ):
                raise ValidationError(f"OpenAPI paths must be root-relative: {path!r}")
            path_item = _resolve_object(paths[path], self.document)
            inherited = path_item.get("parameters", [])
            if not isinstance(inherited, list):
                raise ValidationError(f"OpenAPI path parameters must be an array: {path}")
            for method in sorted(key for key in path_item if key.lower() in _HTTP_METHODS):
                raw_operation = _resolve_object(path_item[method], self.document)
                tool_name = _operation_name(method, path, raw_operation)
                if "*" not in self.allow_operations and tool_name not in self.allow_operations:
                    continue
                if tool_name in operations:
                    raise ValidationError(
                        f"duplicate normalized OpenAPI operation name: {tool_name}"
                    )
                own_parameters = raw_operation.get("parameters", [])
                if not isinstance(own_parameters, list):
                    raise ValidationError(
                        f"OpenAPI operation parameters must be an array: {tool_name}"
                    )
                # OpenAPI operation parameters override path-level parameters
                # with the same ``(name, in)`` identity.
                merged_parameters: dict[tuple[Any, Any], Mapping[str, Any]] = {}
                for item in [*inherited, *own_parameters]:
                    parameter = _resolve_object(item, self.document)
                    merged_parameters[(parameter.get("name"), parameter.get("in"))] = parameter
                parameters = list(merged_parameters.values())
                name_counts = Counter(
                    item.get("name") for item in parameters if isinstance(item.get("name"), str)
                )
                properties: dict[str, Any] = {}
                required: list[str] = []
                locations: dict[str, tuple[str, str]] = {}
                for parameter in parameters:
                    original_name = parameter.get("name")
                    location = parameter.get("in")
                    if not isinstance(original_name, str) or location not in {
                        "path",
                        "query",
                        "header",
                        "cookie",
                    }:
                        raise ValidationError(f"invalid OpenAPI parameter in operation {tool_name}")
                    if location == "cookie" or (
                        location == "header" and original_name.lower() in _SENSITIVE_HEADERS
                    ):
                        raise ValidationError(
                            f"sensitive OpenAPI parameter is not model-exposed: {original_name}"
                        )
                    argument_name = (
                        original_name
                        if name_counts[original_name] == 1
                        else f"{location}__{original_name}"
                    )
                    properties[argument_name] = normalize_json_schema(
                        parameter.get("schema", {}),
                        document=self.document,
                    )
                    locations[argument_name] = (location, original_name)
                    if parameter.get("required") is True or location == "path":
                        required.append(argument_name)

                body_argument: str | None = None
                request_body = raw_operation.get("requestBody")
                if request_body is not None:
                    request_body = _resolve_object(request_body, self.document)
                    content = request_body.get("content", {})
                    if not isinstance(content, Mapping) or not content:
                        raise ValidationError(f"requestBody has no content schemas: {tool_name}")
                    media_type = (
                        "application/json" if "application/json" in content else sorted(content)[0]
                    )
                    media = _resolve_object(content[media_type], self.document)
                    body_argument = "body" if "body" not in properties else "request_body"
                    properties[body_argument] = normalize_json_schema(
                        media.get("schema", {}),
                        document=self.document,
                    )
                    if request_body.get("required") is True:
                        required.append(body_argument)

                input_schema: dict[str, Any] = {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                }
                if required:
                    input_schema["required"] = sorted(set(required))
                tool = ToolSpec(
                    name=tool_name,
                    description=_operation_description(raw_operation),
                    input_schema=normalize_json_schema(input_schema),
                    read_only=method.lower() in {"get", "head", "options"},
                    network=self.network,
                )
                operations[tool_name] = _Operation(
                    tool=tool,
                    method=method.upper(),
                    path=path,
                    argument_locations=locations,
                    body_argument=body_argument,
                )
        return operations

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._operations[name].tool for name in sorted(self._operations))

    def call(self, request: ToolCall) -> ToolResult:
        operation = self._operations.get(request.name)
        if operation is None:
            return ToolResult(call_id=request.call_id, error=f"unknown tool: {request.name}")
        if not operation.tool.read_only and not self.allow_mutating:
            return ToolResult(
                call_id=request.call_id,
                error=f"mutating tool is disabled by policy: {request.name}",
            )
        if operation.tool.network and not self.allow_network_calls:
            return ToolResult(
                call_id=request.call_id,
                error=f"network tool is disabled by policy: {request.name}",
            )
        arguments = dict(request.arguments)
        properties = operation.tool.input_schema.get("properties", {})
        required = operation.tool.input_schema.get("required", [])
        unknown = sorted(set(arguments) - set(properties))
        missing = sorted(set(required) - set(arguments))
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
            for name, value in arguments.items():
                schema = properties.get(name, {})
                if isinstance(schema, Mapping):
                    validate_json_value(value, schema, path=name)
        except ValidationError as exc:
            return ToolResult(call_id=request.call_id, error=str(exc))

        path = operation.path
        query: dict[str, Any] = {}
        headers: dict[str, str] = {}
        cookies: dict[str, str] = {}
        for argument_name, (location, original_name) in operation.argument_locations.items():
            if argument_name not in arguments:
                continue
            value = arguments[argument_name]
            if location == "path":
                path = path.replace("{" + original_name + "}", quote(str(value), safe=""))
            elif location == "query":
                query[original_name] = value
            elif location == "header":
                headers[original_name] = str(value)
            elif location == "cookie":
                cookies[original_name] = str(value)
        if cookies:
            headers["Cookie"] = "; ".join(f"{name}={cookies[name]}" for name in sorted(cookies))
        body = arguments.get(operation.body_argument) if operation.body_argument else None
        try:
            output = self.transport.request(
                operation.method,
                path,
                query=query,
                headers=headers,
                body=body,
            )
            return ToolResult(
                call_id=request.call_id,
                output=output,
                metadata={"connector": "openapi", "operation": request.name},
            )
        except Exception as exc:
            return ToolResult(
                call_id=request.call_id,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"connector": "openapi", "operation": request.name},
            )

    def fingerprint(self) -> Mapping[str, Any]:
        operation_fingerprints = [
            self._operations[name].fingerprint() for name in sorted(self._operations)
        ]
        return {
            "connector_id": self.connector_id,
            "protocol": "openapi",
            "implementation": "agentinputlab.openapi.v1",
            "document_hash": content_hash(self.document),
            "operation_schema_hash": content_hash(operation_fingerprints),
            "operation_count": len(operation_fingerprints),
            "base_url_identity_hash": content_hash(self.base_url_identity),
            "network": self.network,
            "allow_operations": sorted(self.allow_operations),
            "allow_mutating": self.allow_mutating,
            "allow_network_calls": self.allow_network_calls,
        }


__all__ = [
    "OpenAPIToolConnector",
    "OpenAPITransport",
    "normalize_json_schema",
    "python_tool_spec",
    "validate_json_value",
]
