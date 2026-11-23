"""The ToolTalk hook shares the real agent budget; fixtures are not neural evidence."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from workers.official_arenas.protocol import OfficialArenaError
from workers.official_arenas.providers import tooltalk


@pytest.fixture
def predictor(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=19, completion_tokens=7),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content="done", tool_calls=None,
            ))],
        )

    openai = ModuleType("openai")
    openai.OpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setattr(tooltalk, "_tool_schemas", lambda: [{"type": "function", "function": {"name": "read"}}])
    state = dict(query="original request", instruction="task instruction", context=[], metadata={})
    budget = dict(max_model_calls=8, max_tool_calls=8, max_output_tokens=4096)

    def make(hook=None, **changes):
        return tooltalk._Predictor(model="fixture-not-a-backbone", base_url="http://not-contacted.invalid/v1",
            api_key="fixture", materialized=state, budget={**budget, **changes}, optimizer_usage={},
            seed=17, max_output_tokens_per_call=512, history_transform=hook)

    return make, calls


META = dict(location="fixed", timestamp="fixed", username="user")
HISTORY = [{"role": "user", "text": "native request"}]


class Hook:
    def __init__(self, usage=None):
        self.calls = []
        self.usage = usage or dict(model_calls=1, retrieval_calls=0, input_tokens=31,
                                  output_tokens=11, unpriced_model_calls=1, cost_usd=0.0)

    def fingerprint(self):
        return {"implementation": "unit-fixture"}

    def transform(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        messages[0]["content"] += "\nfixture summary"
        return dict(messages=messages, usage=self.usage)


def test_without_hook_preserves_original_messages_controls_and_usage(predictor):
    make, calls = predictor
    agent = make()
    assert agent(META, HISTORY) == {"role": "assistant", "text": "done"}
    assert calls[0]["messages"] == agent._messages(META, HISTORY)
    assert calls[0]["seed"] == 17 and calls[0]["max_tokens"] == 512
    assert agent.usage() == dict(model_calls=1, tool_calls=0, input_tokens=19,
                                output_tokens=7, cost_usd=0.0, unpriced_model_calls=1)


def test_live_helper_calls_are_shared_and_not_double_billed(predictor):
    make, calls = predictor; hook = Hook(); agent = make(hook)
    agent(META, HISTORY)
    assert hook.calls[0][1]["remaining_budget"] == dict(max_model_calls=8, max_output_tokens=4096)
    assert calls[0]["messages"][0]["content"].endswith("fixture summary")
    usage = agent.usage()
    assert usage["model_calls"] == 2 and usage["input_tokens"] == 50 and usage["output_tokens"] == 18
    assert usage["unpriced_model_calls"] == 2
    agent(META, [*HISTORY, {"role": "assistant", "text": "done"}, {"role": "user", "text": "follow-up"}])
    assert hook.calls[1][1]["remaining_budget"] == dict(max_model_calls=6, max_output_tokens=4078)
    assert calls[1]["seed"] == 18  # Native target sequence is independent of helper calls.


def test_initial_optimizer_cost_is_not_charged_twice(predictor):
    make, _ = predictor;hook = Hook();agent = make(hook)
    agent.max_model_calls -= 2;agent.max_output_tokens -= 100
    agent(META, HISTORY)
    assert hook.calls[0][1]["remaining_budget"] == dict(max_model_calls=6, max_output_tokens=3996)
    assert agent.usage()["model_calls"] == 2


@pytest.mark.parametrize("usage", [
    dict(model_calls=8, retrieval_calls=0, input_tokens=1, output_tokens=1),
    dict(model_calls=1, retrieval_calls=0, input_tokens=1, output_tokens=4096),
    dict(model_calls=1, retrieval_calls=0, input_tokens=1, output_tokens=1, tool_calls=1),
    dict(model_calls=1, output_tokens=1),
    dict(model_calls=-1, output_tokens=0),
    dict(model_calls=True, output_tokens=0),
    dict(model_calls=0, output_tokens=1),
])
def test_bad_usage_rejected_before_downstream_generation(predictor, usage):
    make, calls = predictor
    with pytest.raises(Exception):make(Hook(usage))(META, HISTORY)
    assert calls == []


def test_shared_budget_stops_before_another_helper_or_target_call(predictor):
    make, calls = predictor;hook = Hook();agent = make(hook, max_model_calls=2)
    agent(META, HISTORY)
    result = agent(META, HISTORY)
    assert result["text"] == tooltalk._BUDGET_STOP
    assert len(hook.calls) == len(calls) == 1 and agent.usage()["model_calls"] == 2


def test_orphaned_tool_frames_rejected_before_target_call(predictor):
    make, calls = predictor;hook = Hook()
    hook.transform = lambda messages, **k: dict(messages=[*messages, {"role": "tool", "content": "reply", "tool_call_id": "missing"}], usage={"model_calls": 0, "output_tokens": 0})
    with pytest.raises(OfficialArenaError, match="orphaned"):make(hook)(META, HISTORY)
    assert not calls


def test_temporary_tool_rollout_is_not_committed_to_append_only_history(predictor):
    make, calls = predictor
    hook = Hook(dict(model_calls=0, output_tokens=0))
    agent = make(hook)
    agent(META, HISTORY)
    temporary = [*HISTORY, {"role": "api", "request": {
        "api_name": "read", "parameters": {}}, "response": {"value": "temporary"},
        "exception": None}]
    agent(META, temporary)
    assert len(hook.calls) == 1
    assert calls[-1]["messages"][0]["content"].endswith("fixture summary")
    assert calls[-1]["messages"][-1]["role"] == "tool"
    assert "temporary" in calls[-1]["messages"][-1]["content"]
    # The native executor discards the temporary prediction and commits its
    # official previous turn before the next user request.
    committed = [*HISTORY, {"role": "assistant", "text": "official prior turn"},
                 {"role": "user", "text": "follow-up"}]
    agent(META, committed)
    assert len(hook.calls) == 2
    assert [row["role"] for row in hook.calls[-1][0]] == ["system", "user", "assistant", "user"]
    assert all(row.get("role") != "tool" for row in hook.calls[-1][0])
    assert agent.usage()["model_calls"] == 3


def test_temporary_rollout_cannot_rewrite_its_native_user_boundary(predictor):
    make, calls = predictor
    agent = make(Hook(dict(model_calls=0, output_tokens=0)))
    agent(META, HISTORY)
    with pytest.raises(OfficialArenaError, match="changed its user boundary"):
        agent({**META, "username": "unexpected rewrite"}, [*HISTORY,
              {"role": "assistant", "text": "temporary"}])
    assert len(calls) == 1


def test_live_history_cannot_start_mid_rollout(predictor):
    make, calls = predictor
    agent = make(Hook(dict(model_calls=0, output_tokens=0)))
    with pytest.raises(OfficialArenaError, match="initial user boundary"):
        agent(META, [*HISTORY, {"role": "assistant", "text": "not a user boundary"}])
    assert not calls


@pytest.mark.parametrize("hook", [object(), SimpleNamespace(transform=lambda: None), SimpleNamespace(fingerprint=lambda: {})])
def test_provider_rejects_invalid_hook_before_loading(hook):
    with pytest.raises(OfficialArenaError, match="transform and fingerprint"):
        tooltalk.ToolTalkProvider(history_transform=hook)
