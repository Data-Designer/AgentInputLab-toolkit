"""Synthetic STATE wiring fixtures; real-response acceptance is separate."""

import contextlib
import hashlib
import json
import sys
import types

import pytest

from agentinputlab.contracts import (
    ArtifactError,
    ExecutionBudget,
    FrozenAgentState,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.fit_configuration import FitConfig
from tests.unit.test_skillopt_retrieval_configuration import configured
from workers.official_arenas.protocol import ProviderArmResult


def configured_state(tmp_path, monkeypatch):
    value, _ = configured(tmp_path, monkeypatch)
    env = value["environment"]
    env.update(kind="statebench", checkout=str(tmp_path), environment_backend=dict(env["backend"]))
    return value


def test_public_state_environment_no_caller_code(tmp_path, monkeypatch):
    value = configured_state(tmp_path, monkeypatch)
    assert FitConfig.from_dict(value).validate()["validated"]


def test_state_environment_key_must_be_declared(tmp_path, monkeypatch):
    value = configured_state(tmp_path, monkeypatch)
    value["environment"]["environment_backend"]["api_key_env"] = "MISSING_JUDGE_KEY"
    with pytest.raises(ValidationError, match="environment_names"):
        FitConfig.from_dict(value)


def test_state_checkout_resolves_relative_to_fit_file(tmp_path, monkeypatch):
    value = configured_state(tmp_path, monkeypatch)
    value["environment"]["checkout"] = "STATE-Bench"
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(value))
    assert FitConfig.from_file(path).value["environment"]["checkout"] == str(
        tmp_path / "STATE-Bench"
    )


@pytest.fixture
def task(tmp_path, monkeypatch):
    import agentinputlab.recipes.skillopt_statebench as native

    base = types.ModuleType("skillopt.envs.base")
    base.EnvAdapter = type("EnvAdapter", (), {})
    monkeypatch.setitem(sys.modules, "skillopt.envs.base", base)
    value = configured_state(tmp_path, monkeypatch)
    env = value["environment"]
    rows = []
    for role, key in (("train", "one"), ("validation", "two")):
        state = FrozenAgentState(
            task_id=key, query="query " + key, budget=ExecutionBudget(max_context_tokens=262144)
        )
        capture = dict(
            checkpoint_id=key,
            visible_state=to_jsonable(state),
            protected=dict(
                locked_dependency_hashes={"evaluated_model": "r", "environment_model": "r"}
            ),
        )
        rows.append(
            dict(
                case_id=key,
                official_split="train",
                role=role,
                state=to_jsonable(state),
                state_hash=state.state_hash,
                protected_checkpoint=capture,
            )
        )
    config = dict(
        split="train",
        agent_model="m",
        evaluator_model="m",
        locked_dependency_hashes={"evaluated_model": "r", "environment_model": "r"},
        budget_override={
            **to_jsonable(rows[0]["state"]["budget"]),
            "max_output_tokens_per_call": 512,
        },
    )
    source = dict(
        benchmark="state_bench.common_support",
        native_source_revision="fixture",
        native_configuration=config,
        fitting_rows=rows,
        heldout_ids=["test"],
        source_files={"fixture.py": "digest"},
        target_model={"model_id": "m"},
        environment_model={"model_id": "m"},
    )
    path = tmp_path / "state-source.json"
    path.write_text(json.dumps(source))
    env["source_input"] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    class Provider:
        benchmark_id = "state_bench.common_support"
        source_revision = "fixture"
        scorer_id = scorer_revision = "native-fixture"

        def __init__(self):
            self.calls = []

        def _validate_config(self, c):
            return dict(c)

        def run_arm(self, c, **kw):
            self.calls.append((c, kw))
            return ProviderArmResult(
                metrics={"task_completion_pass": 0.0},
                usage={"model_calls": 1, "input_tokens": 10, "output_tokens": 4},
                trace={"conversation": [{"role": "assistant", "content": "real visible message"}]},
                runtime_instance_id=content_hash("fixture"),
            )

    provider = Provider()
    monkeypatch.setattr(native, "_provider", lambda: provider)
    monkeypatch.setattr(native, "verify_native_checkout", lambda p, **kw: tmp_path / "checkout")
    monkeypatch.setattr(native, "verify_source_files", lambda *a: None)
    monkeypatch.setattr(native, "native_process_scope", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(native, "_native_inputs", lambda *a: ({}, {}, source["source_files"]))

    @contextlib.contextmanager
    def accounting(*a):
        yield []

    monkeypatch.setattr(native, "_account_native_calls", accounting)
    return env, source, provider


def build(task):
    from agentinputlab.recipes.skillopt_statebench import create_statebench_environment

    env, _, _ = task
    return create_statebench_environment(cfg={}, configuration=env)


def test_native_full_episode_and_zero_score_preserved(task, tmp_path):
    env = build(task)
    result = env.rollout(env.build_train_env(1, 17), "candidate", tmp_path / "out")[0]
    assert result["soft"] == result["hard"] == 0 and result["agent_ok"]
    config, call = task[2].calls[0]
    assert (
        config["request_attempts"] == 1
        and config["agent_base_url"] == task[0]["backend"]["base_url"]
    )
    materialized = call["arm_payload"]["intervention_execution"]["materialized_state"]
    assert materialized["query"] == "query one"
    assert materialized["context"][-1]["text"] == "SKILLOPT CURRENT FITTING SKILL\ncandidate"
    assert call["role"] == "treatment" and call["seed"] == 17
    assert call["arm_payload"]["intervention_execution"]["usage"]["model_calls"] == 0
    assert "test" not in json.dumps(env.fingerprint()["groups"])


@pytest.mark.parametrize("change", ["heldout", "duplicate", "mutated", "diagnostic", "source"])
def test_invalid_rollout_never_calls_provider(task, tmp_path, change):
    env = build(task)
    batch = env.build_train_env(1, 17)
    kwargs = {}
    with pytest.raises(ArtifactError if change == "source" else ValidationError):
        if change == "heldout":
            env.build_eval_env(1, "test", 17)
        elif change == "duplicate":
            batch *= 2
        elif change == "mutated":
            batch[0]["question"] = "other"
        elif change == "diagnostic":
            kwargs["diagnostic_mode"] = True
        else:
            from pathlib import Path

            Path(task[0]["source_input"]["path"]).write_text("{}")
        env.rollout(batch, "skill", tmp_path / "out", **kwargs)
    assert not task[2].calls


@pytest.mark.parametrize("change", ["heldout", "state", "budget", "model", "split"])
def test_source_mismatch_rejected_before_episode(task, change):
    from pathlib import Path

    env, source, provider = task
    if change == "heldout":
        source["heldout_ids"] = ["one"]
    elif change == "state":
        source["fitting_rows"][0]["state"]["query"] = "changed"
    elif change == "budget":
        env["per_call_output_tokens"] = 1024
    elif change == "model":
        env["backend"]["deployment_identity"]["revision"] = "other"
    else:
        source["native_configuration"]["split"] = "test"
    path = Path(env["source_input"]["path"])
    path.write_text(json.dumps(source))
    env["source_input"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValidationError):
        build(task)
    assert not provider.calls


def test_native_error_is_retained_not_zero(task, tmp_path):
    env = build(task)
    task[2].run_arm = lambda *a, **kw: (_ for _ in ()).throw(OSError("native unavailable"))
    batch = env.build_train_env(1, 17)
    with pytest.raises(OSError):
        env.rollout(batch, "skill", tmp_path / "out")
    saved = tmp_path / "out/predictions" / content_hash("one") / "consumed-execution-error.json"
    assert json.loads(saved.read_text())["automatic_retry"] is False


@pytest.mark.parametrize("failure", [False, True])
def test_native_cost_hook_preserves_wire_roles_and_restores(tmp_path, monkeypatch, failure):
    from agentinputlab.recipes.skillopt_statebench import _account_native_calls
    from workers.official_arenas.providers import statebench

    clients, calls = [], []
    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(model_dump=lambda: {"output_tokens": 7})
    )

    def original(**kwargs):
        def generate(**request):
            calls.append(request)
            if failure:
                raise OSError("transport failed")
            return response

        sdk = types.SimpleNamespace(responses=types.SimpleNamespace(create=generate), closed=False)
        sdk.close = lambda: setattr(sdk, "closed", True)
        client = types.SimpleNamespace(_client=sdk)
        clients.append(client)
        return client

    monkeypatch.setattr(statebench, "_openai_compatible_client", original)
    request = {
        "model": "m",
        "input": [{"role": "user", "content": "unchanged"}],
        "max_output_tokens": 512,
    }
    with _account_native_calls(tmp_path, {"agent_model": "m", "evaluator_model": "e"}) as events:
        for model in ("m", "e", "e"):
            client = statebench._openai_compatible_client(model=model)
            if failure:
                with pytest.raises(OSError):
                    client._client.responses.create(**request)
            else:
                assert client._client.responses.create(**request) is response
    assert calls == [request] * 3
    assert [e["role"] for e in events] == ["agent", "user", "judge"]
    assert all(e["usage"] == (None if failure else {"output_tokens": 7}) for e in events)
    assert all(not e["automatic_retry"] for e in events)
    assert all(c._client.closed for c in clients)
    assert statebench._openai_compatible_client is original
    assert len(list(tmp_path.glob("native-call-*.json"))) == 3
