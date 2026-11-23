"""Fixtures test message/usage contracts, not real neural compatibility."""
import copy
import json
from types import SimpleNamespace

import pytest

from workers.official_arenas.protocol import OfficialArenaError, ProviderArmResult
from workers.official_arenas.providers._live_history import NativeHistorySession
from workers.official_arenas.providers.tau3 import Tau3Provider
from workers.official_arenas.providers.tau_retail import TauRetailProvider


class Native:
    def __init__(self, **wire):
        self.wire = wire
        self.private = {"native_trace_metadata": ["retained"]}

    def model_copy(self, *, update, deep):
        assert deep
        value = copy.deepcopy(self)
        value.wire.update(update)
        return value


def encode(messages):
    return [copy.deepcopy(m.wire) for m in messages]


def messages():
    return [Native(role="system", content="policy"),
            Native(role="assistant", content="original greeting", tool_calls=None),
            Native(role="user", content="original request"),
            Native(role="assistant", content=None, tool_calls=[{
                "id": "c1", "type": "function", "name": "read",
                "function": {"name": "read", "arguments": '{"x": 1}'}}]),
            Native(role="tool", content="native result", tool_call_id="c1"),
            Native(role="assistant", content="earlier answer", tool_calls=None),
            Native(role="user", content="new question")]


def make(transform=None, usage=None):
    seen = []
    usage = usage or dict(model_calls=1, retrieval_calls=0, input_tokens=21, output_tokens=11,
                          unpriced_model_calls=1, cost_usd=0.0)
    def run(raw, **kwargs):
        seen.append(copy.deepcopy(kwargs))
        if transform is not None:
            raw = transform(raw)
        else:
            raw[0]["content"] += "\nsummary"
        return {"messages": raw, "usage": usage}
    hook = SimpleNamespace(transform=run, fingerprint=lambda: {"fixture": True}, records=[])
    return NativeHistorySession(hook, {"query": "original request"},
        {"max_model_calls": 6, "max_output_tokens": 3000}), seen


def test_native_tool_frames_ids_arguments_and_private_metadata_roundtrip():
    session, seen = make()
    native = messages()
    before = encode(native)
    result = session.messages(native, encode=encode, agent_usage={"model_calls": 2, "output_tokens": 50})
    assert seen[0]["remaining_budget"] == {"max_model_calls": 4, "max_output_tokens": 2950}
    assert encode(native) == before
    assert encode(result)[1:] == before[1:]
    assert result[0].wire["content"] == "policy\nsummary"
    assert all(out.private == old.private and out is not old for out, old in zip(result, native))
    assert session.remaining({"model_calls": 2, "output_tokens": 50}) == {
        "max_model_calls": 3, "max_output_tokens": 2939}
    assert session.combined_usage({"model_calls": 2, "input_tokens": 100, "output_tokens": 50})["model_calls"] == 3


def test_only_source_order_subsequence_tail_can_be_retained():
    session, _ = make(transform=lambda raw: [*raw[:3], *raw[-2:]])
    result = session.messages(messages(), encode=encode, agent_usage={})
    assert encode(result) == [*encode(messages())[:3], *encode(messages())[-2:]]


@pytest.mark.parametrize("mutate", [
    lambda raw: [raw[0], dict(raw[1], content="changed greeting"), *raw[2:]],
    lambda raw: [*raw[:3], *reversed(raw[3:])],
    lambda raw: [*raw, {"role": "assistant", "content": "invented"}],
    lambda raw: [*raw[:3], raw[4], *raw[5:]],  # Orphaned tool result.
    lambda raw: [dict(raw[0], name="injected"), *raw[1:]],
])
def test_changed_greeting_tools_or_nontext_fields_are_rejected(mutate):
    session, _ = make(transform=mutate)
    with pytest.raises(OfficialArenaError):
        session.messages(messages(), encode=encode, agent_usage={})


@pytest.mark.parametrize("usage", [
    {"model_calls": 6, "retrieval_calls": 0, "input_tokens": 1, "output_tokens": 1},
    {"model_calls": 1, "retrieval_calls": 0, "input_tokens": 1, "output_tokens": 3000},
    {"model_calls": 0, "output_tokens": 1},
    {"model_calls": 1, "output_tokens": 1},
    {"model_calls": 1, "retrieval_calls": 1, "input_tokens": 1, "output_tokens": 1},
])
def test_invalid_usage_cannot_hide_helper_cost_or_spend_reserved_call(usage):
    session, _ = make(usage=usage)
    with pytest.raises(Exception):
        session.messages(messages(), encode=encode, agent_usage={})


def test_result_and_finally_share_one_create_only_receipt(tmp_path):
    session, _ = make()
    session.messages(messages(), encode=encode, agent_usage={})
    target = {"model_calls": 1, "input_tokens": 100, "output_tokens": 20, "tool_calls": 1}
    agent = SimpleNamespace(_history_session=session, _history_target_usage=lambda: target)
    native = ProviderArmResult(metrics={"official_reward": 0.0, "cost": 0.0},
                               trace={"native": True}, usage=target, runtime_instance_id="a" * 64)
    result = Tau3Provider._history_result(native, agent, tmp_path)
    session.write_receipt(tmp_path / "live-history.json", target_usage=target)
    value = json.loads((tmp_path / "live-history.json").read_text())
    assert value["native_environment_history_NOT_modified"]
    assert result.usage["model_calls"] == 2 and result.usage["output_tokens"] == 31
    assert result.trace["native"] and result.runtime_instance_id != "a" * 64
    assert native.usage == target and native.metrics["official_reward"] == 0.0
    with pytest.raises(OfficialArenaError, match="changed before finalization"):
        session.write_receipt(tmp_path / "live-history.json", target_usage={"model_calls": 9})


@pytest.mark.parametrize("provider", [Tau3Provider, TauRetailProvider])
@pytest.mark.parametrize("hook", [object(), SimpleNamespace(transform=lambda: None)])
def test_tau_rejects_incomplete_hook_before_loading_any_native_runtime(provider, hook):
    with pytest.raises(OfficialArenaError, match="transform and fingerprint"):
        provider(history_transform=hook)


@pytest.mark.parametrize("provider", [Tau3Provider, TauRetailProvider])
def test_optional_history_hook_is_explicit_and_default_remains_none(provider):
    assert provider().history_transform is None
    hook = SimpleNamespace(transform=lambda *a, **k: None, fingerprint=lambda: {"fixture": True})
    assert provider(history_transform=hook).history_transform is hook
