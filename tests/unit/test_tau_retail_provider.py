from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from workers.official_arenas.protocol import (
    EXECUTABLE_PAYLOAD_PROTOCOL, OfficialArenaError, content_hash, standard_shared_budget,
    visible_shared_budget,
)
from workers.official_arenas.providers import tau_retail as retail


def config():
    def args(port, tokens):
        return {"api_base": f"http://127.0.0.1:{port}/v1", "temperature": 0,
                "seed": 17, "num_retries": 0, "max_tokens": tokens, "timeout": 300}
    return {"task_set_name": "retail", "task_split_name": "test", "limit": 40,
            "capture_seed": 17, "llm_agent": "openai/qwen27", "llm_user": "openai/qwen27",
            "llm_judge": "openai/qwen27", "llm_args_agent": args(8104, 512),
            "llm_args_user": args(8104, 1024),
            "llm_args_judge": {**args(8104, 2048), "response_format": {"type": "json_object"}},
            "max_steps": 8, "max_errors": 3, "timeout": 300,
            "locked_dependency_hashes": {slot: content_hash(slot) for slot in (
                "agent_model", "user_simulator", "nl_assertion_judge", "official_runtime", "scorer")},
            "budget_override": standard_shared_budget()}


def test_retail_identity_is_not_banking():
    provider = retail.factory()
    assert provider.benchmark_id == "tau3.retail"
    assert provider.scorer_id == "tau3.retail.official_v1.0.1.backbone_conditioned_judge"
    assert len(provider.scorer_revision) == 64
    assert provider._validate_config(config())["task_set_name"] == "retail"


@pytest.mark.parametrize("role", ["user", "judge"])
@pytest.mark.parametrize("difference", ["model", "endpoint"])
def test_environment_model_must_follow_the_current_backbone(role, difference):
    candidate = config()
    if difference == "model":
        candidate[f"llm_{role}"] = "openai/another-backbone"
    else:
        candidate[f"llm_args_{role}"]["api_base"] = "http://127.0.0.1:8110/v1"
    with pytest.raises(OfficialArenaError, match="evaluated"):
        retail.factory()._validate_config(candidate)


def test_backbone_roles_may_have_distinct_output_caps():
    candidate = config()
    candidate["llm_args_judge"]["api_base"] += "/"
    assert retail.factory()._validate_config(candidate)["llm_judge"] == candidate["llm_agent"]


@pytest.mark.parametrize("key,value", [
    ("task_set_name", "banking_knowledge"), ("task_split_name", "base"),
    ("limit", 41), ("limit", True), ("capture_seed", 18), ("capture_seed", 17.0),
    ("timeout", None), ("timeout", float("inf")), ("timeout", 0),
])
def test_retail_rejects_implicit_scope_or_budget(key, value):
    candidate = config()
    candidate[key] = value
    with pytest.raises(OfficialArenaError):
        retail.factory()._validate_config(candidate)


@pytest.mark.parametrize("role", ["agent", "user", "judge"])
@pytest.mark.parametrize("key,value", [("num_retries", 1), ("num_retries", False),
                                       ("seed", 18), ("temperature", 0.7),
                                       ("api_base", "https://remote.example/v1"), ("timeout", None),
                                       ("timeout", float("inf")), ("timeout", True), ("timeout", 0)])
def test_all_generation_roles_are_local_fixed_and_no_retry(role, key, value):
    candidate = config()
    candidate[f"llm_args_{role}"][key] = value
    with pytest.raises(OfficialArenaError):
        retail.factory()._validate_config(candidate)


def test_retail_rejects_banking_retrieval_and_unlocked_judge():
    candidate = config()
    candidate["retrieval_config"] = "bm25"
    with pytest.raises(OfficialArenaError):
        retail.factory()._validate_config(candidate)
    candidate = config()
    del candidate["locked_dependency_hashes"]["nl_assertion_judge"]
    with pytest.raises(OfficialArenaError):
        retail.factory()._validate_config(candidate)


class User:
    def __init__(self, content="visible request", tool_call=False):
        self.content, self.tool_call = content, tool_call

    def is_tool_call(self):
        return self.tool_call


def capture_case(*, message=None, error=None, to_role="user", user_tools=None):
    calls, instances = [], []

    class Orchestrator:
        def __init__(self):
            self.message = message or User()
            self.to_role = SimpleNamespace(value=to_role)
            self.environment = SimpleNamespace(user_tools=user_tools)
            self.cleaned, self.steps = False, 0
            instances.append(self)

        def initialize(self):
            pass

        def step(self):
            self.steps += 1
            if error:
                raise error

        def _cleanup(self):
            self.cleaned = True

    def builder(_config, _task, **kwargs):
        calls.append(kwargs)
        return Orchestrator()

    def run():
        return retail.capture_first_user_once(build_orchestrator=builder, run_config=object(),
                                              task=SimpleNamespace(id="5"), seed=17, user_message_type=User)
    return run, calls, instances


def test_capture_is_one_fixed_seed_user_generation():
    run, calls, instances = capture_case()
    instance, message = run()
    assert calls == [{"seed": 17, "simulation_id": "ailab-retail-capture-5-17"}]
    assert instance.steps == 1 and not instance.cleaned
    assert message.content == "visible request"


@pytest.mark.parametrize("case", [
    {"error": json.JSONDecodeError("bad", "x", 0)}, {"message": User("")},
    {"message": User(tool_call=True)}, {"to_role": "agent"}, {"user_tools": [object()]},
])
def test_capture_failure_never_retries_or_runs_agent(case):
    run, calls, instances = capture_case(**case)
    with pytest.raises(Exception):
        run()
    assert len(calls) == 1 and instances[0].cleaned
    assert instances[0].steps == (0 if "to_role" in case or "user_tools" in case else 1)


@pytest.mark.parametrize("results", [[], [{"expectedOutcome": "a", "metExpectation": "true", "reasoning": "x"}],
                                   [{"expectedOutcome": "other", "metExpectation": True, "reasoning": "x"}],
                                   [{"expectedOutcome": "a", "metExpectation": True, "reasoning": "x"}] * 2])
def test_empty_partial_or_untyped_judge_is_not_success(results):
    response = SimpleNamespace(content=json.dumps({"results": results}))
    with pytest.raises(retail.RetailJudgeError):
        retail._check_judge_response(response, ["a"])


def test_valid_judge_preserves_real_negative():
    response = SimpleNamespace(content=json.dumps({"results": [
        {"expectedOutcome": "a", "metExpectation": False, "reasoning": "not met"}]}))
    retail._check_judge_response(response, ["a"])


@pytest.mark.parametrize("content", [None, "", "  \n"])
@pytest.mark.parametrize("tool_calls", [None, []])
def test_empty_agent_output_is_not_repaired_from_reasoning(content, tool_calls):
    response = SimpleNamespace(content=content, tool_calls=tool_calls,
                               reasoning_content="private reasoning is not an answer")
    with pytest.raises(retail.RetailAgentOutputError, match="consumed attempt is retained"):
        retail._require_retail_agent_message(response)


@pytest.mark.parametrize("content,tool_calls", [("real answer", None), (None, [object()])])
def test_valid_agent_message_is_unchanged(content, tool_calls):
    response = SimpleNamespace(content=content, tool_calls=tool_calls)
    assert retail._require_retail_agent_message(response) is None
    assert response.content is content and response.tool_calls is tool_calls


def test_retail_empty_output_stops_after_accounting_and_restores_generator(monkeypatch):
    def original():
        return None
    module = SimpleNamespace(generate=original)
    monkeypatch.setitem(sys.modules, "tau2.utils", SimpleNamespace(llm_utils=module))

    class Base:
        def __init__(self):
            self.calls, self.output_tokens = 0, 0

        def _generate_next_message(self, message, state):
            self.calls += 1
            self.output_tokens += 512
            return SimpleNamespace(content=None, tool_calls=None)

    monkeypatch.setattr(retail, "_intervened_tau_agent_class", lambda: Base)
    agent = retail._retail_agent_class([])()
    assert module.generate is original
    with pytest.raises(retail.RetailAgentOutputError):
        agent._generate_next_message(None, None)
    assert agent.calls == 1 and agent.output_tokens == 512


def test_backbone_judge_is_explicit_scoped_and_restored(monkeypatch):
    response = SimpleNamespace(content=json.dumps({"results": [
        {"expectedOutcome": "a", "metExpectation": True, "reasoning": "met"}]}),
        usage={"prompt_tokens": 123, "completion_tokens": 44})
    original_args = {"unchanged": True}
    original_generate = lambda *a, **k: response
    module = SimpleNamespace(DEFAULT_LLM_NL_ASSERTIONS="old-model", DEFAULT_LLM_NL_ASSERTIONS_ARGS=original_args,
                             generate=original_generate)
    monkeypatch.setitem(sys.modules, "tau2.evaluator", SimpleNamespace(evaluator_nl_assertions=module))
    candidate = config()
    candidate["llm_agent"] = "openai/a-different-backbone"
    with retail.bound_retail_judge(candidate, ["a"]) as counters:
        assert module.DEFAULT_LLM_NL_ASSERTIONS == "openai/qwen27"
        assert module.DEFAULT_LLM_NL_ASSERTIONS_ARGS == candidate["llm_args_judge"]
        module.generate()
        assert counters == {"model_calls": 1, "input_tokens": 123, "output_tokens": 44}
    assert module.DEFAULT_LLM_NL_ASSERTIONS == "old-model"
    assert module.DEFAULT_LLM_NL_ASSERTIONS_ARGS is original_args and module.generate is original_generate
    with pytest.raises(RuntimeError), retail.bound_retail_judge(candidate, ["a"]):
        raise RuntimeError("fixture failure")
    assert module.DEFAULT_LLM_NL_ASSERTIONS == "old-model" and module.generate is original_generate


def test_shared_budget_stop_is_retained_and_cannot_be_reexecuted(tmp_path, monkeypatch):
    provider = retail.factory()
    monkeypatch.setattr(provider, "doctor", lambda _config: ())
    monkeypatch.setattr(retail, "_install_local_tau_cost_accounting", lambda _config: frozenset())
    candidate = config()
    state = {"query": "public request", "budget": visible_shared_budget(candidate, {})}
    payload = {"schema_version": 1, "protocol": EXECUTABLE_PAYLOAD_PROTOCOL,
               "visible_state_hash": content_hash(state),
               "intervention_execution": {"materialized_state": state, "output_state_hash": content_hash(state),
                                          "input_state_hash": content_hash(state), "usage": {"model_calls": 8}}}
    replay = {"checkpoint_id": "5", "protected": {"task_id": "5", "capture_attempt": 1, "capture_seed": 17,
              "locked_dependency_hashes": candidate["locked_dependency_hashes"]}}
    args = {"replay_record": replay, "arm_payload": payload, "seed": 17, "role": "treatment", "work_dir": tmp_path}
    result = provider.run_arm(candidate, **args)
    assert result.metrics["official_reward"] == 0
    assert result.trace["reward_info"] is None
    assert result.usage["model_calls"] == 0
    assert result.trace["generation_controls"]["automatic_retries"] == 0
    assert result.trace["benchmark_id"] == "tau3.retail"
    with pytest.raises(FileExistsError):
        provider.run_arm(candidate, **args)
    assert len(list(tmp_path.iterdir())) == 2


def test_retail_per_call_output_cap_cannot_exceed_shared_budget():
    candidate = config()
    candidate["llm_args_agent"]["max_tokens"] = 513
    with pytest.raises(OfficialArenaError):
        retail.factory()._validate_config(candidate)


@pytest.mark.parametrize("reason,hard_stops,expected", [("timeout", 0, 0), ("max_steps", 0, 0),
    ("agent_stop", 1, 0), ("user_stop", 0, 1), ("agent_stop", 0, 1)])
def test_native_reward_and_timeout_failure_accounting_are_separate(reason, hard_stops, expected):
    raw_reward = 1.0
    assert retail._paper_terminal_success(raw_reward, reason, hard_stops) == expected
    assert raw_reward == 1.0


def test_environment_user_calls_are_itemized_not_added_to_agent_budget():
    initial = User()
    known = User()
    known.usage = {"prompt_tokens": 9, "completion_tokens": 12}
    usage = retail._environment_user_usage([initial, known, User(), object()], 1, User)
    assert usage == {"model_calls": 2, "input_tokens": 9, "output_tokens": 12, "token_usage_incomplete": True}


def failed_capture_fixture(tmp_path, monkeypatch):
    provider, candidate, calls = retail.factory(), config(), []
    candidate["limit"] = 1
    monkeypatch.setattr(provider, "doctor", lambda _config: ())
    monkeypatch.setattr(provider, "_run_config", lambda _config: object())
    monkeypatch.setattr(retail, "_install_local_tau_cost_accounting", lambda _config: frozenset())
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", SimpleNamespace(UserMessage=User))
    monkeypatch.setitem(sys.modules, "tau2.runner.build", SimpleNamespace(build_orchestrator=object()))
    monkeypatch.setitem(sys.modules, "tau2.runner.helpers", SimpleNamespace(
        get_tasks=lambda *a, **kw: [SimpleNamespace(id="5")]))
    def failed_once(**kwargs):
        assert (tmp_path / "first-capture-5-start.json").is_file()
        calls.append(kwargs["seed"])
        raise TimeoutError("fixture first generation timeout")
    monkeypatch.setattr(retail, "capture_first_user_once", failed_once)
    return provider, candidate, calls


def test_capture_failure_is_reserved_before_generation_and_not_retried(tmp_path, monkeypatch):
    provider, candidate, calls = failed_capture_fixture(tmp_path, monkeypatch)
    for _ in range(2):
        with pytest.raises(OfficialArenaError, match="no automatic retry"):
            provider.capture(candidate, output_root=tmp_path)
    assert calls == [17]
    saved = json.loads((tmp_path / "first-capture-5.json").read_text())
    assert saved["terminal"] == "capture_failed" and saved["retry_permitted"] is False
    assert len(saved["exception_sha256"]) == 64


def test_interrupted_or_live_capture_is_not_reexecuted(tmp_path, monkeypatch):
    provider, candidate, calls = failed_capture_fixture(tmp_path, monkeypatch)
    retail._create_only(tmp_path / "first-capture-5-start.json", {"reserved": True})
    with pytest.raises(OfficialArenaError, match="no automatic retry"):
        provider.capture(candidate, output_root=tmp_path)
    assert calls == [] and not (tmp_path / "first-capture-5.json").exists()


def test_capture_receipt_task_identity_and_symlink_are_rejected(tmp_path):
    saved = {"protocol": "retail-first-capture.v1", "config_binding": "a" * 64,
             "task_id": "9", "seed": 17, "attempt": 1, "terminal": "capture_failed"}
    saved["receipt_sha256"] = content_hash(saved)
    path = tmp_path / "wrong.json"
    retail._create_only(path, saved)
    with pytest.raises(OfficialArenaError, match="differs"):
        retail._saved_capture(path, binding="a" * 64, task_id="5")
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(OfficialArenaError, match="symlinked"):
        retail._saved_capture(link, binding="a" * 64, task_id="9")
