"""Explicit Responses transport for strict required-field integer schemas."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .base import GenerationResponse, Usage


def request_payload(chat: Mapping[str, Any]) -> dict[str, Any]:
    required = {"model", "messages", "max_tokens", "temperature", "response_format"}
    if not required <= chat.keys() or chat.keys() - required - {"seed", "thinking", "stream"}:
        raise ValueError("Unsupported strict Responses options")
    if chat.get("thinking", {"type": "disabled"}) != {"type": "disabled"}:
        raise ValueError("Strict Responses requires non-thinking mode")
    if "stream" in chat and chat["stream"] is not False:
        raise ValueError("Strict Responses requires non-streaming output")
    messages = chat["messages"]
    if not messages or any(set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"}
                           or not isinstance(m["content"], str) for m in messages):
        raise ValueError("Strict Responses expects plain text messages")
    fmt = chat["response_format"]
    if set(fmt) != {"type", "json_schema"} or fmt["type"] != "json_schema":
        raise ValueError("Strict Responses requires a JSON schema")
    strict = fmt["json_schema"]
    if (set(strict) != {"name", "schema", "strict"} or strict["strict"] is not True
            or not isinstance(strict["name"], str) or not strict["name"]
            or not isinstance(strict["schema"], Mapping) or strict["schema"].get("type") != "object"
            or not _schema_supported(strict["schema"])):
        raise ValueError("Unsupported strict Responses schema")
    return {
        "model": chat["model"],
        "input": messages,
        "max_output_tokens": chat["max_tokens"],
        "temperature": chat["temperature"],
        **({"seed": chat["seed"]} if "seed" in chat else {}),
        "reasoning": {"effort": "none"},
        "stream": False,
        "store": False,
        "text": {"format": {"type": "json_schema", **strict}},
    }


def _schema_supported(schema: Any) -> bool:
    if schema == {"type": "integer"}:
        return True
    if (not isinstance(schema, Mapping) or set(schema) != {"type", "properties", "required", "additionalProperties"}
            or schema["type"] != "object" or schema["additionalProperties"] is not False
            or not isinstance(schema["properties"], Mapping) or not isinstance(schema["required"], list)):
        return False
    required = schema["required"]
    return (all(isinstance(k, str) for k in required) and len(required) == len(set(required))
            and set(required) == set(schema["properties"])
            and all(_schema_supported(child) for child in schema["properties"].values()))


def _conforms(value: Any, schema: Mapping[str, Any]) -> bool:
    if schema["type"] == "integer":
        return type(value) is int
    return (isinstance(value, dict) and value.keys() == schema["properties"].keys()
            and all(_conforms(value[key], child) for key, child in schema["properties"].items()))


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate structured field")
    return result


def parse_response(data: Mapping[str, Any], payload: Mapping[str, Any], aliases: tuple[str, ...]) -> GenerationResponse:
    if data["object"] != "response" or data["model"] not in (payload["model"], *aliases) or data.get("error"):
        raise ValueError("Unbound Responses identity or provider failure")
    [message] = data["output"]
    if message["type"] != "message" or message["role"] != "assistant" or not message["content"]:
        raise ValueError("Missing assistant response")
    parts = message["content"]
    if any(p["type"] != "output_text" or not isinstance(p["text"], str) for p in parts):
        raise ValueError("Unsupported Responses content")
    text = "".join(p["text"] for p in parts)
    usage = data["usage"]
    n, m = usage["input_tokens"], usage["output_tokens"]
    if (type(n) is not int or type(m) is not int or n < 0 or not 0 <= m <= payload["max_output_tokens"]
            or usage.get("total_tokens", n + m) != n + m):
        raise ValueError("Missing or inconsistent Responses usage")
    if data["status"] == "completed":
        if not _conforms(json.loads(text, object_pairs_hook=_object), payload["text"]["format"]["schema"]):
            raise ValueError("Strict output does not match the requested schema")
        finish = "stop"
    elif data["status"] == "incomplete" and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        finish = "length"
    else:
        raise ValueError("Unexpected Responses terminal")
    return GenerationResponse(text=text, model=data["model"], finish_reason=finish,
                              usage=Usage(input_tokens=n, output_tokens=m), raw=data)
