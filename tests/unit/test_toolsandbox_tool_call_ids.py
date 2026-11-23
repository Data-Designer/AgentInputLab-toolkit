"""Opaque API IDs must round-trip through ToolSandbox's Python code format."""
import ast
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from workers.official_arenas.providers.toolsandbox import _normalize_tool_call_ids


def response_with_ids(*ids):
    calls = [SimpleNamespace(id=value, type="function", function=SimpleNamespace(
        name="get_cellular_service_status", arguments='{"flag": true}')) for value in ids]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="unchanged text", tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=7))


@pytest.mark.parametrize("identifier", ["chatcmpl-tool-example", "call.1", "123call", "call 1", "call/1", "call\n1", "x);unsafe()#"])
def test_opaque_id_is_encoded_without_changing_tool_or_usage(identifier):
    response = response_with_ids(identifier)
    before = deepcopy(response)
    repairs = _normalize_tool_call_ids(response)
    call = response.choices[0].message.tool_calls[0]
    assert len(repairs) == 1 and call.id.isidentifier()
    assert repairs[0] == {"kind": "tool_call_id_to_python_identifier",
        "original_id": identifier, "normalized_id": call.id}
    assert len(call.id) < 64
    assert call.function == before.choices[0].message.tool_calls[0].function
    assert response.choices[0].message.content == before.choices[0].message.content
    assert response.usage == before.usage
    # The exact upstream conversion template now produces syntactically valid
    # code; do not execute model-provided tools during a protocol unit test.
    code = (f"{call.id}_parameters = {json.loads(call.function.arguments)}\n"
        f"{call.id}_response = {call.function.name}(**{call.id}_parameters)\n"
        f"print(repr({call.id}_response))")
    assert isinstance(ast.parse(code), ast.Module)
    assert _normalize_tool_call_ids(response) == []


def test_valid_ids_and_response_alternatives_remain_unchanged():
    response = response_with_ids("call_1", "call_2", "class")
    response.choices.append(SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(id="not-selected")])) )
    before = deepcopy(response)
    assert _normalize_tool_call_ids(response) == []
    assert response == before


def test_alias_does_not_collide_with_an_existing_valid_id():
    raw = "call-1"
    reserved = "ail_call_" + hashlib.sha256(raw.encode()).hexdigest()[:32]
    response = response_with_ids(raw, reserved)
    repairs = _normalize_tool_call_ids(response)
    calls = response.choices[0].message.tool_calls
    assert calls[0].id == reserved + "_1" and calls[1].id == reserved
    assert repairs[0]["original_id"] == raw


def test_duplicate_ids_are_not_deduplicated_and_distinct_ids_stay_distinct():
    response = response_with_ids("call-1", "call-1", "call.1")
    repairs = _normalize_tool_call_ids(response)
    ids = [call.id for call in response.choices[0].message.tool_calls]
    assert len(ids) == 3 and ids[0] == ids[1] and ids[1] != ids[2]
    assert len(repairs) == 2


@pytest.mark.parametrize("response", [None, SimpleNamespace(), SimpleNamespace(choices=[]),
    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))])])
def test_no_tool_calls_are_not_fabricated(response):
    assert _normalize_tool_call_ids(response) == []


def test_missing_or_empty_ids_are_not_fabricated():
    response = response_with_ids(None, "")
    assert _normalize_tool_call_ids(response) == []
    assert [call.id for call in response.choices[0].message.tool_calls] == [None, ""]
