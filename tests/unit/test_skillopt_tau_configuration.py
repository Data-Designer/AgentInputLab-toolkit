"""Public Tau fitting configuration and native episode boundaries."""

import contextlib
import hashlib
import json
from pathlib import Path

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.fit_configuration import FitConfig
from agentinputlab.fit_preparation import _build, _configuration
from tests.unit.test_fit_preparation import fixture_native, skillopt_config, write
from tests.unit.test_skillopt_state_configuration import configured_state
from tests.unit.test_skillopt_state_configuration import task as _state_task
from workers.official_arenas.protocol import ProviderArmResult


def tau_config(tmp_path, monkeypatch):
    value = configured_state(tmp_path, monkeypatch)
    value["environment"].update(
        kind="tau", worker_python="./worker/bin/python", episode_timeout_seconds=180
    )
    return value


def test_tau_public_fit_file(tmp_path, monkeypatch):
    value = tau_config(tmp_path, monkeypatch)
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(value))
    config = FitConfig.from_file(path)
    assert config.validate()["validated"]
    assert config.value["environment"]["worker_python"] == str(tmp_path / "worker/bin/python")


def test_tau_prepare_shares_input_builder_not_ace_plan_feedback(tmp_path, monkeypatch):
    value = skillopt_config(tmp_path, monkeypatch)
    value["environment"].update(
        kind="tau", worker_python="./worker/bin/python", episode_timeout_seconds=180
    )
    del value["native_configuration"], value["environment_model"]
    value.update(
        domain="retail",
        partition_label="declared-development",
        native_configurations={"train-1": {}, "dev-1": {}},
    )
    seen = fixture_native(monkeypatch, "tau")
    config = _configuration(write(tmp_path, value))
    out = tmp_path / "prepared"
    out.mkdir()
    result = _build(config, out)
    assert result["model_calls"] == 0 and not result["fit_executed"]
    assert "factory" not in seen
    assert seen["builder"]["domain"] == "retail"
    assert FitConfig.from_file(result["fit_configuration"]["path"]).validate()["validated"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_python", ""),
        ("episode_timeout_seconds", 0),
        ("stop_policy", "gpt_oss_final_channel"),
    ],
)
def test_tau_rejects_invalid_worker_configuration(tmp_path, monkeypatch, field, value):
    config = tau_config(tmp_path, monkeypatch)
    config["environment"][field] = value
    with pytest.raises(ValidationError):
        FitConfig.from_dict(config)


@pytest.fixture
def tau_task(tmp_path, monkeypatch):
    from agentinputlab.recipes import skillopt_tau as native
    from agentinputlab.recipes.ace_tau import SCHEMA

    env, source, provider = _state_task.__wrapped__(tmp_path, monkeypatch)
    env.update(kind="tau", worker_python="/unused/python", episode_timeout_seconds=180)
    provider.benchmark_id = "tau3.retail"
    locks = dict(agent_model="r", user_simulator="r", nl_assertion_judge="r")
    source.update(
        schema=SCHEMA,
        domain="retail",
        benchmark=provider.benchmark_id,
        source_official_split="train",
        partition_source="declared-dev",
        native_function_schemas={},
    )
    for row in source["fitting_rows"]:
        row["protected_checkpoint"]["protected"]["locked_dependency_hashes"] = locks
        row["native_configuration"] = dict(
            task_split_name="train",
            llm_agent="m",
            llm_user="m",
            llm_judge="m",
            llm_args_agent=dict(
                seed=17, num_retries=0, max_tokens=512, api_base="http://127.0.0.1:9999/v1"
            ),
            llm_args_user=dict(seed=17, num_retries=0, max_tokens=1024),
            llm_args_judge=dict(seed=17, num_retries=0, max_tokens=512),
            locked_dependency_hashes=locks,
            budget_override=source["native_configuration"]["budget_override"],
        )
    path = Path(env["source_input"]["path"])

    def refresh():
        path.write_text(json.dumps(source))
        env["source_input"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    refresh()
    monkeypatch.setattr(native, "_provider", lambda domain: provider)
    monkeypatch.setattr(native, "verify_native_checkout", lambda *a, **kw: tmp_path / "checkout")
    monkeypatch.setattr(native, "verify_source_files", lambda *a: None)
    monkeypatch.setattr(native, "native_process_scope", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(
        native, "_native_inputs", lambda *a: ({}, {}, {}, {}, {}, source["source_files"])
    )
    monkeypatch.setattr(native, "_account_calls", lambda *a: contextlib.nullcontext([]))

    def run_arm(config, **kw):
        provider.calls.append((config, kw))
        return ProviderArmResult(
            metrics=dict(official_reward=0.0),
            usage=dict(model_calls=1, input_tokens=10, output_tokens=4),
            trace=dict(
                messages=[
                    dict(role="assistant", content="visible"),
                    dict(role="evaluator", content="HIDDEN_GOLD"),
                ]
            ),
            runtime_instance_id=content_hash("fixture"),
        )

    provider.run_arm = run_arm
    monkeypatch.setattr(native, "_launch", native._episode)
    return env, source, provider, refresh


def test_tau_full_episode_zero_and_visible_feedback(tau_task, tmp_path):
    from agentinputlab.recipes.skillopt_tau import create_tau_environment

    env, source, provider, _ = tau_task
    adapter = create_tau_environment(cfg={}, configuration=env)
    batch = adapter.build_train_env(1, 17)
    out = tmp_path / "run"
    rows = adapter.rollout(batch, "candidate skill", out)
    assert len(provider.calls) == 1 and rows[0]["soft"] == 0 and rows[0]["hard"] == 0
    assert "visible" in rows[0]["response"] and "HIDDEN_GOLD" not in rows[0]["response"]
    config, payload = provider.calls[0]
    assert payload["role"] == "treatment" and payload["seed"] == 17
    assert payload["arm_payload"]["release_id"] == "recipe.skillopt.offline"
    assert (
        config["llm_args_agent"]["max_tokens"] == 512
        and config["llm_args_user"]["max_tokens"] == 1024
    )
    with pytest.raises(FileExistsError):
        adapter.rollout(batch, "candidate skill", out)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("change", ["heldout", "budget", "seed", "model", "capture", "split"])
def test_tau_source_mismatch_before_episode(tau_task, change):
    from agentinputlab.recipes.skillopt_tau import create_tau_environment

    env, source, provider, refresh = tau_task
    if change == "heldout":
        source["heldout_ids"].append("one")
    elif change == "budget":
        env["per_call_output_tokens"] = 256
    elif change == "seed":
        env["seed"] = 18
    elif change == "model":
        env["environment_backend"]["model_id"] = "another"
    elif change == "capture":
        source["fitting_rows"][0]["state_hash"] = "changed"
    else:
        source["source_official_split"] = "test"
    refresh()
    with pytest.raises(ValidationError):
        create_tau_environment(cfg={}, configuration=env)
    assert not provider.calls


@pytest.mark.parametrize("change", ["heldout", "duplicate", "changed", "diagnostic"])
def test_tau_extra_rollout_rejected(tau_task, tmp_path, change):
    from agentinputlab.recipes.skillopt_tau import create_tau_environment

    env, _, provider, _ = tau_task
    adapter = create_tau_environment(cfg={}, configuration=env)
    batch = adapter.build_train_env(1, 17)
    options = {}
    if change == "heldout":
        batch[0]["id"] = "heldout"
    elif change == "duplicate":
        batch *= 2
    elif change == "changed":
        batch[0]["question"] = "changed"
    else:
        options["diagnostic_mode"] = True
    with pytest.raises(ValidationError):
        adapter.rollout(batch, "skill", tmp_path / "run", **options)
    assert not provider.calls


@pytest.mark.parametrize("role", ["agent", "user", "judge"])
def test_tau_transport_preserves_body_and_separates_role_costs(tmp_path, monkeypatch, role):
    httpx = pytest.importorskip("httpx")

    from agentinputlab.recipes import skillopt_tau as native

    monkeypatch.setattr(native, "_native_role", lambda: role)
    monkeypatch.setenv("TAU_TEST_KEY", "fixture-secret-not-a-credential")
    config = dict(
        seed=17,
        backend=dict(
            model_id="m", base_url="http://127.0.0.1:1234/target/v1", api_key_env="TAU_TEST_KEY"
        ),
        environment_backend=dict(
            model_id="m",
            base_url="http://127.0.0.1:1234/environment/v1",
            api_key_env="TAU_TEST_KEY",
        ),
    )
    controls = {"api_base": "http://127.0.0.1:9/v1", "max_tokens": 512}
    original = dict(llm_args_agent=controls, llm_args_user=controls, llm_args_judge=controls)
    raw = b'{"model":"m","seed":17,"max_tokens":512,"messages":[]}'
    seen = []

    def response(request):
        seen.append(request)
        return httpx.Response(
            200, json=dict(model="m", usage=dict(prompt_tokens=10, completion_tokens=2))
        )

    with (
        httpx.Client(transport=httpx.MockTransport(response)) as client,
        native._account_calls(tmp_path, config, original) as events,
    ):
        client.post("http://127.0.0.1:9/v1/chat/completions", content=raw)
    assert seen[0].content == raw
    assert ("/target/" if role == "agent" else "/environment/") in str(seen[0].url)
    assert events[0]["role"] == role and events[0]["usage"]["completion_tokens"] == 2
    saved = (tmp_path / "native-call-0001.json").read_text()
    assert "fixture-secret" not in saved
