"""Protocol fixtures, not claims of real-model acceptance."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from workers.official_arenas.protocol import OfficialArenaError, ProviderArmResult
from workers.official_arenas.providers._live_history import ResponsesHistorySession
from workers.official_arenas.providers.statebench import (
    StateBenchProvider, _install_stateless_tool_history, _intervened_agent_class, _SharedBudgetStop,
)


class Item:
    def __init__(self, **wire):
        self.wire = wire
        self.private = {"original_sdk_metadata": [1, 2]}

    def model_dump(self, **kwargs):
        return copy.deepcopy(self.wire)

    def model_copy(self, *, update, deep):
        assert deep
        item = copy.deepcopy(self)
        item.wire.update(update)
        return item


def native():
    return [dict(role="user", content="original"),
        Item(type="reasoning", id="r1", summary=[], encrypted_content="opaque"),
        Item(type="function_call", call_id="c1", id="f1", name="read", arguments='{"x":1}'),
        Item(type="function_call", call_id="c2", id="f2", name="read", arguments='{"x":2}'),
        dict(type="function_call_output", call_id="c2", output='{"value":2}'),
        dict(type="function_call_output", call_id="c1", output='{"value":1}'),
        Item(type="message", role="assistant", id="m1", content=[dict(type="output_text", text="done", annotations=[])]),
        dict(role="system", content="original injected context"),
        dict(role="user", content="follow-up")]


def dump(items):
    return [item.model_dump() if isinstance(item, Item) else copy.deepcopy(item) for item in items]


def hook(mutate=None, usage=None):
    seen = []
    def transform(rows, **kwargs):
        seen.append(copy.deepcopy(dict(rows=rows, **kwargs)))
        if mutate:
            rows = mutate(rows)
        return dict(messages=rows, usage=usage or dict(model_calls=0, output_tokens=0))
    return SimpleNamespace(transform=transform, fingerprint=lambda: {"fixture": True}, records=[]), seen


def session(mutate=None, usage=None):
    h, seen = hook(mutate, usage)
    return ResponsesHistorySession(h, {"query": "original"},
        dict(max_model_calls=8, max_output_tokens=4096)), seen


def test_noop_roundtrip_retains_every_native_item_and_all_sdk_metadata():
    s, seen = session()
    original = native()
    before = dump(original)
    actual, instructions = s.responses(original, instructions="real policy", agent_usage={})
    assert dump(actual) == before == dump(original)
    assert instructions == "real policy"
    assert seen[0]["rows"][1] == dict(role="user", content="original")
    assert len(seen[0]["rows"]) == 4  # One atomic assistant/function/results episode.
    assert "opaque" in seen[0]["rows"][2]["content"]
    for new, old in zip(actual, original):
        assert new is not old
        if isinstance(old, Item):
            assert new.private == old.private


def test_first_user_summary_does_not_rewrite_native_policy_tools_or_annotations():
    s, _ = session(lambda rows: [rows[0], dict(rows[1], content="original\nsummary"), *rows[2:]],
                   dict(model_calls=1, input_tokens=15, output_tokens=20, retrieval_calls=0))
    original = native()
    actual, instructions = s.responses(original, instructions="policy", agent_usage=dict(model_calls=2))
    assert actual[0]["content"] == "original\nsummary"
    assert dump(actual)[1:] == dump(original)[1:]
    assert instructions == "policy"
    assert s.remaining(dict(model_calls=2, output_tokens=100)) == dict(max_model_calls=5, max_output_tokens=3976)


def test_only_complete_native_episodes_can_be_removed():
    n = native()
    n += [Item(type="message", role="assistant", content=[dict(type="output_text", text="answer")]),
          dict(role="user", content="third turn")]
    s, _ = session(lambda rows: [*rows[:2], *rows[-2:]])
    actual, _ = s.responses(n, instructions="policy", agent_usage={})
    assert dump(actual) == [dump(n)[0], dump(n)[7], *dump(n)[-2:]]  # Injected policy stays native.


@pytest.mark.parametrize("mutate", [
    lambda rows: [dict(rows[0], content="new policy"), *rows[1:]],
    lambda rows: [rows[0], dict(rows[1], name="injected"), *rows[2:]],
    lambda rows: [*rows[:2], dict(role="assistant", content="invented"), rows[-1]],
    lambda rows: [*rows[:2], *reversed(rows[2:])],
    lambda rows: rows[:-1],
])
def test_invalid_optimizer_output_never_reaches_native_wire(mutate):
    s, _ = session(mutate)
    with pytest.raises(OfficialArenaError):
        s.responses(native(), instructions="policy", agent_usage={})


@pytest.mark.parametrize("mutate", [
    lambda n: n[:4] + n[5:],  # Missing c2 result.
    lambda n: n[:2] + n[3:],  # Orphaned c1 result.
    lambda n: n[:4] + [dict(type="function_call_output", call_id="other", output="x")] + n[4:],
    lambda n: [dict(role="user", content=[dict(type="input_image", image_url="x")])],
    lambda n: n[:-1],  # Not an actual user boundary.
])
def test_unsupported_source_shape_is_rejected_before_optimizer(mutate):
    s, seen = session()
    with pytest.raises(OfficialArenaError):
        s.responses(mutate(native()), instructions="policy", agent_usage={})
    assert seen == []


def test_single_input_text_block_preserves_extra_fields():
    n = [dict(type="message", role="user", content=[dict(type="input_text", text="original", metadata="kept")])]
    s, _ = session(lambda rows: [rows[0], dict(rows[1], content="original summary")])
    actual, _ = s.responses(n, instructions="policy", agent_usage={})
    assert actual == [dict(type="message", role="user", content=[dict(type="input_text", text="original summary", metadata="kept")])]
    assert n[0]["content"][0]["text"] == "original"


@pytest.mark.parametrize("model", ["qwen3-4b", "qwen3.8-27b-fp8", "gpt-oss-20b"])
def test_stateless_transport_hook_runs_only_at_native_user_boundary(model):
    calls, views = [], []
    def complete(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(id=str(len(calls)), output=[dict(type="message", role="assistant", content="reply")])
    c = SimpleNamespace(deployment=model, complete_with_tools=complete)
    _install_stateless_tool_history(c)
    def view(kwargs):
        views.append(copy.deepcopy(kwargs))
        return dict(kwargs, input=[*kwargs["input"][:-1], dict(kwargs["input"][-1], content="summary")])
    c._agentinputlab_responses_history = view
    first = [dict(role="system", content="context"), dict(role="user", content="original")]
    c.complete_with_tools(instructions="policy", input=first, max_tokens=512)
    c.complete_with_tools(instructions="policy", input=[dict(type="function_call_output", call_id="c", output="x")], previous_response_id="1", max_tokens=512)
    assert len(views) == 1
    assert first[-1]["content"] == "original"
    assert any(isinstance(r, dict) and r.get("content") == "summary" for r in calls[1]["input"])
    assert calls[1]["previous_response_id"] is None
    if model.startswith("qwen3.8"):
        assert calls[0]["instructions"] == calls[1]["instructions"] == "policy\n\ncontext"
    else:
        assert calls[0]["instructions"] == calls[1]["instructions"] == "policy"


def test_state_budget_wrapper_reserves_target_after_helper_and_on_tool_followup(monkeypatch):
    class Agent:
        def __init__(self, client):
            self.client, self.tool_handlers = client, {}
    monkeypatch.setitem(sys.modules, "state_bench.agents.state_bench", SimpleNamespace(StateBenchAgent=Agent))
    wire = []
    def complete(**kwargs):
        wire.append(kwargs)
        return SimpleNamespace(id=str(len(wire)), output=[], usage=SimpleNamespace(
            input_tokens=10, output_tokens=min(400, kwargs["max_tokens"])))
    client = SimpleNamespace(deployment="qwen3-4b", complete_with_tools=complete)
    _install_stateless_tool_history(client)
    h, seen = hook(usage=dict(model_calls=1, input_tokens=20, output_tokens=400, retrieval_calls=0))
    agent = _intervened_agent_class()(client, materialized={"query": "original"}, max_api_calls=3,
        max_output_tokens=1000, max_input_tokens=5000, max_output_tokens_per_call=512, max_tool_calls=8, history_transform=h)
    client.complete_with_tools(instructions="policy", input=[dict(role="user", content="original")], max_tokens=512)
    assert wire[0]["max_tokens"] == 512
    client.complete_with_tools(instructions="policy", input=[], previous_response_id="1", max_tokens=512)
    assert wire[1]["max_tokens"] == 200
    assert len(seen) == 1
    with pytest.raises(_SharedBudgetStop):
        client.complete_with_tools(instructions="policy", input=[], previous_response_id="2", max_tokens=512)
    assert len(wire) == 2
    assert agent.agentinputlab_usage()["model_calls"] == 2
    assert agent._history_session.combined_usage(agent.agentinputlab_usage())["model_calls"] == 3


def test_provider_keeps_failure_receipt_and_combines_success_usage(tmp_path, monkeypatch):
    s, _ = session()
    s.usage = dict(model_calls=1, input_tokens=10, output_tokens=20)
    p = StateBenchProvider(history_transform=s.hook)
    def run(*args, _history_context, **kwargs):
        _history_context["agent"] = SimpleNamespace(_history_session=s, agentinputlab_usage=lambda: {"model_calls": 2})
        return ProviderArmResult(metrics={"task_completion_pass": 0.0}, trace={"native": True}, usage={"model_calls": 2}, runtime_instance_id="a"*64)
    monkeypatch.setattr(p, "_run_arm", run)
    result = p.run_arm({}, replay_record={}, arm_payload={}, seed=17, role="test", work_dir=tmp_path)
    assert result.usage["model_calls"] == 3 and result.metrics["task_completion_pass"] == 0.0
    assert json.loads((tmp_path / "live-history.json").read_text())["responses_view"]["protocol"].startswith("native.responses")
    (tmp_path / "live-history.json").unlink()
    def fail(*args, **kwargs):
        run(*args, **kwargs)
        raise RuntimeError("real native failure")
    monkeypatch.setattr(p, "_run_arm", fail)
    with pytest.raises(RuntimeError, match="real native failure"):
        p.run_arm({}, replay_record={}, arm_payload={}, seed=17, role="test", work_dir=tmp_path)
    assert (tmp_path / "live-history.json").exists()


def test_state_history_is_opt_in_and_rejects_incomplete_hook():
    assert StateBenchProvider().history_transform is None
    with pytest.raises(OfficialArenaError):
        StateBenchProvider(history_transform=object())
