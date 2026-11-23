from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from workers.official_arenas.providers.tau3 import (
    Tau3Provider,
    _capture_initial_visible_user,
    _capture_retry_seed,
    _is_gpt_oss_generation_boundary_failure,
    _normalize_empty_assistant_message,
    _normalized_tool_arguments,
    _normalized_tool_name,
)


def test_provider_records_default_capture_policy_in_both_checkpoint_views(monkeypatch, tmp_path):
    from workers.official_arenas.providers import tau3

    class User:
        content = "source-visible request"

        def is_tool_call(self):
            return False

        def model_copy(self, **kwargs):
            return self

        def model_dump(self, **kwargs):
            return {"role": "user", "content": self.content}

    class Assistant:
        pass

    observed = []

    class Orchestrator:
        def __init__(self):
            tool = SimpleNamespace(name="native_tool", openai_schema={"type": "function"})
            self.environment = SimpleNamespace(get_tools=lambda: [tool], get_policy=lambda: "native policy")
            self.user = SimpleNamespace(llm="openai/fixture")

        def initialize(self):
            observed.append("initialize")

        def step(self):
            observed.append("step")
            self.message = User()

        def get_messages(self):
            return [self.message]

        def _cleanup(self):
            observed.append("cleanup")

    def build(config, task, *, seed, simulation_id):
        observed.append((seed, simulation_id))
        return Orchestrator()

    for hook in ("_install_local_tau_cost_accounting", "_install_local_tau_nl_evaluator",
                 "_install_agentinputlab_retriever", "_install_tau_tool_argument_normalizer"):
        monkeypatch.setattr(tau3, hook, lambda *a: None)
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", SimpleNamespace(AssistantMessage=Assistant, UserMessage=User))
    monkeypatch.setitem(sys.modules, "tau2.runner.build", SimpleNamespace(build_orchestrator=build))
    monkeypatch.setitem(sys.modules, "tau2.runner.helpers", SimpleNamespace(get_tasks=lambda *a, **kw: [SimpleNamespace(id="fixed")]))
    config = {"task_set_name": "banking_knowledge", "task_split_name": "test", "limit": 1,
              "retrieval_config": "bm25", "max_steps": 8, "locked_dependency_hashes": {"source": "a" * 64}}
    monkeypatch.setattr(Tau3Provider, "_validate_config", lambda self, value: value)
    monkeypatch.setattr(Tau3Provider, "_run_config", lambda self, value: object())
    row, = Tau3Provider().capture(config, output_root=tmp_path)
    assert observed == [(0, "agentinputlab-capture-fixed-attempt-1"), "initialize", "step", "cleanup"]
    policy = row["protected"]["capture_policy"]
    assert row["visible_state"]["metadata"]["capture_policy"] == policy
    assert policy["policy_id"] == "tau3.single_capture.v1"
    assert policy["max_attempts"] == 1
    assert policy["text_only_fallback_allowed"] is False
    assert policy["attempt_count_is_not_model_call_or_token_usage"] is True
    assert row["protected"]["capture_seed"] == 0


@pytest.mark.parametrize("policy", [{}, {"max_attempts": 1}])
def test_default_or_explicit_single_capture_does_not_retry_generated_failure(policy):
    seen = []

    class Orchestrator:
        user = SimpleNamespace(llm="openai/fixture")

        def initialize(self):
            raise json.JSONDecodeError("bad generated arguments", "", 0)

        def _cleanup(self):
            seen.append("cleaned")

    def build(*args, **kwargs):
        seen.append(kwargs["seed"])
        return Orchestrator()

    with pytest.raises(RuntimeError, match="after 1 simulator attempts"):
        _capture_initial_visible_user(
            build_orchestrator=build,
            run_config=object(),
            task=SimpleNamespace(id="fixed"),
            base_seed=17,
            assistant_message_type=type("Assistant", (), {}),
            user_message_type=type("User", (), {}),
            **policy,
        )
    assert seen == [17, "cleaned"]


@pytest.mark.parametrize("attempts", [None, True, 0, -1, 17, 18, 1.5])
def test_invalid_capture_attempt_limit_fails_before_building(attempts):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not build a native runtime")

    with pytest.raises(RuntimeError, match="capture_max_attempts"):
        _capture_initial_visible_user(
            build_orchestrator=forbidden,
            run_config=object(),
            task=SimpleNamespace(id="fixed"),
            base_seed=17,
            assistant_message_type=object,
            user_message_type=object,
            max_attempts=attempts,
        )


def test_tau3_recognizes_only_gpt_oss_generation_boundary_failures() -> None:
    context = RuntimeError(
        "litellm.BadRequestError: Input length (41259) exceeds model's maximum context length (40960)."
    )
    harmony = RuntimeError("unexpected tokens remaining in message header: Some(...)")

    assert _is_gpt_oss_generation_boundary_failure(context, model_name="gpt-oss-20b")
    assert _is_gpt_oss_generation_boundary_failure(harmony, model_name="gpt-oss-20b")
    assert _is_gpt_oss_generation_boundary_failure(context, model_name="openai/gpt-oss-20b")
    assert _is_gpt_oss_generation_boundary_failure(harmony, model_name="openai/gpt-oss-20b")
    assert not _is_gpt_oss_generation_boundary_failure(context, model_name="other-model")
    assert not _is_gpt_oss_generation_boundary_failure(
        RuntimeError("endpoint unavailable"),
        model_name="gpt-oss-20b",
    )


def test_tau3_tool_arguments_accept_normal_and_double_encoded_objects() -> None:
    expected = '{"card_name":"Crypto-Cash Back Card","user_id":"u-1"}'

    assert _normalized_tool_arguments(expected) == expected
    assert _normalized_tool_arguments(json.dumps(expected)) == expected
    assert _normalized_tool_arguments(json.dumps(json.dumps(expected))) == expected


def test_tau3_tool_arguments_reject_non_object_payloads() -> None:
    assert _normalized_tool_arguments('"plain text"') is None
    assert _normalized_tool_arguments("[1,2,3]") is None
    assert _normalized_tool_arguments("not-json") is None
    assert _normalized_tool_arguments({"already": "decoded"}) is None


def test_tau3_tool_name_repairs_only_advertised_harmony_suffix() -> None:
    advertised = {"request_human_agent_transfer"}

    assert (
        _normalized_tool_name(
            "request_human_agent_transfer<|channel|>commentary",
            advertised_tool_names=advertised,
        )
        == "request_human_agent_transfer"
    )
    assert (
        _normalized_tool_name(
            "invented<|channel|>commentary",
            advertised_tool_names=advertised,
        )
        is None
    )
    assert (
        _normalized_tool_name(
            "request_human_agent_transfer",
            advertised_tool_names=advertised,
        )
        is None
    )


def test_tau3_capture_retries_only_pre_agent_simulator_failures() -> None:
    class User:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class Assistant:
        pass

    seeds = []
    orchestrators = []

    class Orchestrator:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.message = None
            self.cleaned = False
            orchestrators.append(self)

        def initialize(self) -> None:
            pass

        def step(self) -> None:
            self.message = User(None) if self.seed == 7 else User("visible request")

        def _cleanup(self) -> None:
            self.cleaned = True

    def build(run_config, task, *, seed, simulation_id):
        del run_config, task, simulation_id
        seeds.append(seed)
        return Orchestrator(seed)

    orchestrator, message, attempt, seed = _capture_initial_visible_user(
        build_orchestrator=build,
        run_config=object(),
        task=SimpleNamespace(id="task-1"),
        base_seed=7,
        assistant_message_type=Assistant,
        user_message_type=User,
        max_attempts=2,
    )

    assert seeds == [7, _capture_retry_seed(7, 1)]
    assert orchestrators[0].cleaned is True
    assert orchestrator.cleaned is False
    assert message.content == "visible request"
    assert attempt == 2
    assert seed == _capture_retry_seed(7, 1)


def test_tau3_capture_can_recover_after_more_than_four_deterministic_attempts() -> None:
    class User:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class Assistant:
        pass

    seeds = []
    orchestrators = []
    success_seed = _capture_retry_seed(11, 5)

    class Orchestrator:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.message = None
            self.cleaned = False
            orchestrators.append(self)

        def initialize(self) -> None:
            pass

        def step(self) -> None:
            if self.seed == success_seed:
                self.message = User("visible after bounded retries")
                return
            raise json.JSONDecodeError("bad simulator tool arguments", "", 0)

        def _cleanup(self) -> None:
            self.cleaned = True

    def build(run_config, task, *, seed, simulation_id):
        del run_config, task, simulation_id
        seeds.append(seed)
        return Orchestrator(seed)

    orchestrator, message, attempt, seed = _capture_initial_visible_user(
        build_orchestrator=build,
        run_config=object(),
        task=SimpleNamespace(id="task-retry-window"),
        base_seed=11,
        assistant_message_type=Assistant,
        user_message_type=User,
        max_attempts=6,
    )

    assert seeds == [_capture_retry_seed(11, index) for index in range(6)]
    assert all(item.cleaned for item in orchestrators[:-1])
    assert orchestrator.cleaned is False
    assert message.content == "visible after bounded retries"
    assert attempt == 6
    assert seed == success_seed


def test_tau3_capture_uses_only_explicit_gpt_oss_text_only_fallback() -> None:
    class User:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class Assistant:
        pass

    orchestrators = []

    class Simulator:
        llm = "openai/gpt-oss-20b"

        def __init__(self) -> None:
            self.tools = [object()]

    class Orchestrator:
        def __init__(self) -> None:
            self.user = Simulator()
            self.message = None
            self.cleaned = False
            orchestrators.append(self)

        def initialize(self) -> None:
            pass

        def step(self) -> None:
            if self.user.tools:
                raise json.JSONDecodeError("bad simulator tool arguments", "", 0)
            self.message = User("visible request after text-only fallback")

        def _cleanup(self) -> None:
            self.cleaned = True

    def build(run_config, task, *, seed, simulation_id):
        del run_config, task, seed, simulation_id
        return Orchestrator()

    orchestrator, message, attempt, _ = _capture_initial_visible_user(
        build_orchestrator=build,
        run_config=object(),
        task=SimpleNamespace(id="task-gpt-oss-fallback"),
        base_seed=17,
        assistant_message_type=Assistant,
        user_message_type=User,
        max_attempts=17,
        allow_text_only_fallback=True,
    )

    assert attempt == 17
    assert message.content == "visible request after text-only fallback"
    assert orchestrator.user.tools == []
    assert all(item.cleaned for item in orchestrators[:-1])


@pytest.mark.parametrize("policy", [
    {"allow_text_only_fallback": 1},
    {"allow_text_only_fallback": "yes"},
    {"allow_text_only_fallback": True},
    {"allow_text_only_fallback": True, "max_attempts": 16},
])
def test_invalid_text_only_policy_fails_before_building(policy):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not build a simulator")

    with pytest.raises(RuntimeError, match="capture_text_only_fallback"):
        _capture_initial_visible_user(
            build_orchestrator=forbidden, run_config=object(), task=SimpleNamespace(id="fixed"),
            base_seed=17, assistant_message_type=object, user_message_type=object, **policy,
        )


def test_explicit_normal_attempts_never_remove_simulator_tools():
    built = []

    class Orchestrator:
        def __init__(self):
            self.user = SimpleNamespace(llm="openai/gpt-oss-20b", tools=["native tool"])
            self.cleaned = False
            built.append(self)

        def initialize(self):
            raise json.JSONDecodeError("failed native generated arguments", "", 0)

        def _cleanup(self):
            self.cleaned = True

    with pytest.raises(RuntimeError, match="after 16 simulator attempts"):
        _capture_initial_visible_user(
            build_orchestrator=lambda *a, **kw: Orchestrator(), run_config=object(),
            task=SimpleNamespace(id="fixed"), base_seed=17,
            assistant_message_type=object, user_message_type=object, max_attempts=16,
        )
    assert len(built) == 16
    assert all(item.cleaned and item.user.tools == ["native tool"] for item in built)


def test_tau3_capture_allows_long_environment_tool_preamble() -> None:
    class User:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class Assistant:
        pass

    class Environment:
        pass

    class Orchestrator:
        def __init__(self) -> None:
            self.message = None
            self.steps = 0
            self.cleaned = False

        def initialize(self) -> None:
            pass

        def step(self) -> None:
            self.steps += 1
            self.message = (
                User("visible after environment preamble") if self.steps == 17 else Environment()
            )

        def _cleanup(self) -> None:
            self.cleaned = True

    orchestrator = Orchestrator()

    def build(run_config, task, *, seed, simulation_id):
        del run_config, task, seed, simulation_id
        return orchestrator

    captured, message, attempt, seed = _capture_initial_visible_user(
        build_orchestrator=build,
        run_config=object(),
        task=SimpleNamespace(id="task-long-preamble"),
        base_seed=13,
        assistant_message_type=Assistant,
        user_message_type=User,
    )

    assert captured is orchestrator
    assert captured.steps == 17
    assert captured.cleaned is False
    assert message.content == "visible after environment preamble"
    assert attempt == 1
    assert seed == 13


def test_tau3_empty_assistant_message_becomes_explicit_scorable_turn() -> None:
    message = SimpleNamespace(content=None, refusal=None, tool_calls=[])

    assert _normalize_empty_assistant_message(message) is True
    assert message.content == "[AgentInputLab: empty assistant response]"


def test_tau3_empty_assistant_message_preserves_refusal_text() -> None:
    message = SimpleNamespace(content=None, refusal="I cannot help with that.", tool_calls=[])

    assert _normalize_empty_assistant_message(message) is True
    assert message.content == "I cannot help with that."


def test_tau3_nonempty_assistant_message_is_unchanged() -> None:
    with_content = SimpleNamespace(content="Done.", refusal=None, tool_calls=[])
    with_tool = SimpleNamespace(content=None, refusal=None, tool_calls=[object()])

    assert _normalize_empty_assistant_message(with_content) is False
    assert with_content.content == "Done."
    assert _normalize_empty_assistant_message(with_tool) is False
    assert with_tool.content is None
