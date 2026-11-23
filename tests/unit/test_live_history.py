import json
from types import MappingProxyType, SimpleNamespace

import pytest

from agentinputlab.contracts import ExecutionTrace, InputPatch, Message, PatchOp, PatchOperation
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.integrations import live_history
from agentinputlab.integrations.live_history import ACONLiveHistory, _balanced_tool_frames, _view
from agentinputlab.integrations.live_history import NativeHistoryInputError, validate_native_history_prefix
from agentinputlab.observability.models import ResourceUsage


@pytest.mark.parametrize("rows", [[], [{"role": "user", "content": "native one-pass prompt"}],
    [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}],
    [{"role": "developer", "content": "policy"}, {"role": "user", "content": "task"}]])
def test_public_prefix_preflight_rejects_unsupported_native_inputs_without_io(rows):
    from copy import deepcopy
    before = deepcopy(rows)
    with pytest.raises(NativeHistoryInputError) as caught:
        validate_native_history_prefix(rows)
    assert caught.value.code == "native_system_user_boundary_missing"
    assert isinstance(caught.value, ValidationError)
    assert rows == before  # No invented SYSTEM, user, answer or fake success.
    assert "native one-pass prompt" not in str(caught.value)


def test_public_prefix_preflight_supports_real_boundaries_without_constructing_a_model():
    assert validate_native_history_prefix([{"role": "system"}, {"role": "user"}]) == 1
    greeting = [{"role": "system"}, {"role": "assistant", "content": "native greeting"}, {"role": "user"}]
    assert validate_native_history_prefix(greeting, prefix_policy="preserve_pre_user_assistant") == 2
    with pytest.raises(NativeHistoryInputError):
        validate_native_history_prefix(greeting)


@pytest.mark.parametrize("rows,policy,code", [
    ("not messages", "strict_system_user", "invalid_message_sequence"),
    ([None], "strict_system_user", "invalid_message_sequence"),
    ([], "invented", "unknown_prefix_policy"),
    ([{"role": "system"}, {"role": "assistant", "content": "greeting"}],
        "preserve_pre_user_assistant", "native_first_user_missing"),
    ([{"role": "system"}, {"role": "assistant", "content": "greeting", "tool_calls": [{}]}, {"role": "user"}],
        "preserve_pre_user_assistant", "native_greeting_prefix_unsupported"),
])
def test_public_prefix_preflight_exposes_specific_input_contract_codes(rows, policy, code):
    with pytest.raises(NativeHistoryInputError) as caught:
        validate_native_history_prefix(rows, prefix_policy=policy)
    assert caught.value.code == code


def runtime(monkeypatch):
    method = SimpleNamespace(release_id="curate.acon.history_runtime",
                             controls=SimpleNamespace(preserve_last_k_turns=1))

    class FixtureRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, method, state, context):
            from agentinputlab.algorithms.curate.acon.common import preserved_turn_indices
            indices = preserved_turn_indices(state.messages, 1)
            patched = (state.messages[0], Message("user", "original\n<HISTORY_SUMMARY>\nsummary\n</HISTORY_SUMMARY>"),
                       *(state.messages[index] for index in indices))
            patch = InputPatch(method_id="acon", release_id=method.release_id,
                source_state_hash=state.state_hash,
                operations=(PatchOperation(PatchOp.REPLACE_MESSAGES, patched),),
                trace=ExecutionTrace(method_id="acon", release_id=method.release_id),
                implementation_identity="unit_fixture", profile_kind="controlled_adapter",
                metadata={"applied": True})
            return SimpleNamespace(patch=patch, usage=ResourceUsage(model_calls=1, output_tokens=20))

    monkeypatch.setattr(live_history, "InterventionRunner", FixtureRunner)
    # Only serializable dataclasses are accepted by the normal record serializer.
    real_to_jsonable = live_history.to_jsonable
    monkeypatch.setattr(live_history, "to_jsonable", lambda x: {"fixture": True}
                        if isinstance(x, SimpleNamespace) else real_to_jsonable(x))
    return ACONLiveHistory(method, SimpleNamespace(external_runtime=SimpleNamespace(fingerprint=lambda: {})))


def messages():
    return [{"role": "system", "content": "system"},
            {"role": "user", "content": "original"},
            {"role": "assistant", "content": "earlier response"},
            {"role": "user", "content": "follow-up"}]


def test_tool_frames_visible_to_optimizer_and_unchanged_when_retained():
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]}
    reply = {"role": "tool", "content": "visible observation", "tool_call_id": "c1"}
    assert "visible observation" in _view(reply).content
    assert _view(reply).role == "assistant"
    assert "read" in _view(call).content
    assert _balanced_tool_frames([call, reply])
    assert not _balanced_tool_frames([reply])
    assert not _balanced_tool_frames([call])
    assert not _balanced_tool_frames([{"role": "function", "content": "legacy"}])


def test_native_trace_fingerprint_is_plain_json_not_nested_mappingproxy(monkeypatch):
    rt = runtime(monkeypatch)
    rt.context.external_runtime.fingerprint = lambda: MappingProxyType({"identity": MappingProxyType({"source": "pinned"})})
    encoded = json.dumps(rt.fingerprint())
    assert json.loads(encoded)["worker"]["identity"]["source"] == "pinned"


def test_causal_append_and_reserved_downstream_budget(monkeypatch):
    rt = runtime(monkeypatch)
    raw = messages()
    output = rt.transform(raw, state={"query": "goal"},
                          remaining_budget={"max_model_calls": 8, "max_output_tokens": 4096})
    assert output["usage"]["model_calls"] == 1
    assert output["messages"][-2:] == raw[-2:]
    assert "HISTORY_SUMMARY" in output["messages"][1]["content"]
    appended = [*raw, {"role": "assistant", "content": "new response"}]
    output2 = rt.transform(appended, state={"query": "goal"},
                           remaining_budget={"max_model_calls": 6, "max_output_tokens": 3000})
    assert output2["messages"][-1] == appended[-1]
    assert output2["messages"][1] == output["messages"][1]
    assert output2["usage"]["model_calls"] == 0
    assert rt.records[-1]["reason"] == "intra_user_turn_tool_protocol_preserved"
    with pytest.raises(ValidationError, match="append-only"):
        rt.transform(raw[:2], state={"query": "goal"},
                     remaining_budget={"max_model_calls": 6, "max_output_tokens": 3000})


def test_downstream_mutation_cannot_modify_private_compressed_history(monkeypatch):
    rt = runtime(monkeypatch)
    raw = messages()
    output = rt.transform(raw, state={"query": "goal"},
                          remaining_budget={"max_model_calls": 8, "max_output_tokens": 4096})
    output["messages"].insert(0, {"role": "system", "content": "native downstream preamble"})
    appended = [*raw, {"role": "assistant", "content": "response"}]
    later = rt.transform(appended, state={"query": "goal"},
                         remaining_budget={"max_model_calls": 6, "max_output_tokens": 3000})
    assert later["messages"][0]["content"] == "system"
    assert later["messages"][1]["role"] == "user"


@pytest.mark.parametrize("calls,tokens", [(1, 4096), (2, 1023)])
def test_no_optimizer_side_effect_when_final_agent_call_would_be_lost(monkeypatch, calls, tokens):
    rt = runtime(monkeypatch)
    output = rt.transform(messages(), state={"query": "goal"},
                          remaining_budget={"max_model_calls": calls, "max_output_tokens": tokens})
    assert output["usage"] == {"model_calls": 0, "output_tokens": 0}
    assert output["messages"] == messages()


def test_orphaned_tool_tail_is_not_compressed(monkeypatch):
    rt = runtime(monkeypatch)
    raw = [*messages()[:2], {"role": "tool", "content": "observation", "tool_call_id": "missing"},
           {"role": "user", "content": "follow-up"}]
    output = rt.transform(raw, state={"query": "goal"},
                          remaining_budget={"max_model_calls": 8, "max_output_tokens": 4096})
    assert output["messages"] == raw
    assert output["usage"]["model_calls"] == 0
    assert rt.records[-1]["reason"] == "preserved_tail_has_unpaired_tool_frames"


def test_native_assistant_greeting_requires_explicit_prefix_policy(monkeypatch):
    rt = runtime(monkeypatch)
    raw = messages()
    raw.insert(1, {"role": "assistant", "content": "How may I help you?", "name": "native"})
    with pytest.raises(ValidationError, match="system then user"):
        rt.transform(raw, state={"query": "goal"},
                     remaining_budget={"max_model_calls": 8, "max_output_tokens": 4096})
    assert not rt.records
    assert "prefix_policy" not in rt.fingerprint()  # Default ABI remains unchanged.


def test_explicit_native_greeting_preserved_verbatim_and_never_moved(monkeypatch):
    old = runtime(monkeypatch)
    rt = ACONLiveHistory(old.method, old.context, prefix_policy="preserve_pre_user_assistant")
    raw = messages()
    greeting = {"role": "assistant", "content": "Native fixed greeting", "name": "native"}
    raw.insert(1, greeting)
    output = rt.transform(raw, state={"query": "goal"},
                          remaining_budget={"max_model_calls": 8, "max_output_tokens": 4096})
    assert output["messages"][1] == greeting
    assert output["messages"][2]["role"] == "user"
    assert "HISTORY_SUMMARY" in output["messages"][2]["content"]
    assert output["messages"][-2:] == raw[-2:]
    assert rt.fingerprint()["prefix_policy"] == "preserve_pre_user_assistant"
    assert rt.records[-1]["immutable_native_prefix_length"] == 1
    native_view, prefix = rt._native_optimizer_view(raw)
    assert prefix == [greeting] and native_view == messages()
    # A later raw append uses the compressed view but retains the exact greeting.
    appended = [*raw, {"role": "assistant", "content": "new response"}]
    later = rt.transform(appended, state={"query": "goal"},
                         remaining_budget={"max_model_calls": 6, "max_output_tokens": 3000})
    assert later["messages"][1] == greeting
    assert later["messages"][2] == output["messages"][2]
    assert later["usage"] == {"model_calls": 0, "output_tokens": 0}
    changed = [dict(row) for row in appended]
    changed[1]["content"] = "changed greeting"
    with pytest.raises(ValidationError, match="append-only"):
        rt.transform(changed, state={"query": "goal"},
                     remaining_budget={"max_model_calls": 6, "max_output_tokens": 3000})


@pytest.mark.parametrize("prefix", [
    {"role": "assistant", "content": ""},
    {"role": "assistant", "content": "call", "tool_calls": [{"id": "x"}]},
    {"role": "assistant", "content": "call", "function_call": {"name": "x"}},
    {"role": "tool", "content": "result", "tool_call_id": "x"},
    {"role": "system", "content": "second system"},
])
def test_prefix_policy_never_reorders_or_hides_non_greeting_frames(monkeypatch, prefix):
    old = runtime(monkeypatch)
    rt = ACONLiveHistory(old.method, old.context, prefix_policy="preserve_pre_user_assistant")
    raw = messages()
    raw.insert(1, prefix)
    with pytest.raises(ValidationError, match="plain assistant greetings"):
        rt.transform(raw, state={"query": "goal"},
                     remaining_budget={"max_model_calls": 8, "max_output_tokens": 4096})
    assert not rt.records


@pytest.mark.parametrize("policy", ["silent_drop", None, [], True])
def test_invalid_prefix_policy_rejected_before_native_work(monkeypatch, policy):
    old = runtime(monkeypatch)
    with pytest.raises(ValidationError, match="prefix policy"):
        ACONLiveHistory(old.method, old.context, prefix_policy=policy)
