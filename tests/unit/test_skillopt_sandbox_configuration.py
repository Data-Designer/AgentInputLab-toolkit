"""ToolSandbox public fit wiring fixtures, separate from native acceptance."""

import contextlib
import hashlib
import json

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.fit_configuration import FitConfig
from agentinputlab.fit_preparation import _build, _configuration
from tests.unit.test_fit_preparation import fixture_native, skillopt_config, write
from tests.unit.test_skillopt_state_configuration import configured_state
from tests.unit.test_skillopt_state_configuration import task as _state_task
from workers.official_arenas.protocol import ProviderArmResult


def sandbox(tmp_path, monkeypatch):
    value = configured_state(tmp_path, monkeypatch)
    value["environment"].update(
        kind="toolsandbox", worker_python="./worker/bin/python", episode_timeout_seconds=180
    )
    return value


def test_sandbox_public_file_entry(tmp_path, monkeypatch):
    value = sandbox(tmp_path, monkeypatch)
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(value))
    result = FitConfig.from_file(path)
    assert result.validate()["validated"]
    assert result.value["environment"]["worker_python"] == str(tmp_path / "worker/bin/python")


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_timeout_seconds", 0),
        ("episode_timeout_seconds", True),
        ("worker_python", ""),
        ("stop_policy", "gpt_oss_final_channel"),
    ],
)
def test_sandbox_rejects_invalid_native_worker(tmp_path, monkeypatch, field, value):
    config = sandbox(tmp_path, monkeypatch)
    config["environment"][field] = value
    with pytest.raises(ValidationError):
        FitConfig.from_dict(config)


def test_sandbox_prepare_uses_original_source_builder_not_ace_feedback(tmp_path, monkeypatch):
    config = skillopt_config(tmp_path, monkeypatch)
    config["environment"].update(
        kind="toolsandbox", worker_python="./worker/bin/python", episode_timeout_seconds=180
    )
    config["partition_label"] = "explicit_development_groups_not_official_split"
    seen = fixture_native(monkeypatch, "toolsandbox")
    parsed = _configuration(write(tmp_path, config))
    out = tmp_path / "out"
    out.mkdir()
    result = _build(parsed, out)
    assert result["model_calls"] == 0 and not result["fit_executed"]
    assert "factory" not in seen
    assert seen["builder"]["partition_label"] == config["partition_label"]
    assert (
        FitConfig.from_file(result["fit_configuration"]["path"]).value["environment"]["kind"]
        == "toolsandbox"
    )


@pytest.fixture
def state_task(tmp_path, monkeypatch):
    return _state_task.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def native_task(state_task, tmp_path, monkeypatch):
    from pathlib import Path

    from agentinputlab.recipes import skillopt_toolsandbox as native

    env, source, provider = state_task
    env.update(kind="toolsandbox", worker_python="/usr/bin/python3", episode_timeout_seconds=180)
    source["benchmark"] = "toolsandbox.common_support"
    source["native_configuration"].update(user_model="m", scenario_names=["one", "two"])
    source["partition_source"] = "fixture development"
    for row in source["fitting_rows"]:
        row["caller_declared_fit_partition_NOT_official_split"] = True
        protected = row["protected_checkpoint"]["protected"]
        context = {"interactive_console": None}
        protected.update(
            scenario_name=row["case_id"],
            starting_context=context,
            starting_context_hash=content_hash(context),
        )
    path = Path(env["source_input"]["path"])
    path.write_text(json.dumps(source))
    env["source_input"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provider.benchmark_id = source["benchmark"]

    def run_arm(config, **kwargs):
        provider.calls.append((config, kwargs))
        return ProviderArmResult(
            metrics={"official_similarity": 0.0},
            usage={"model_calls": 1, "output_tokens": 4},
            trace={
                "ending_context": {
                    "_dbs": {
                        "SANDBOX": [
                            {"sender": "AGENT", "content": "visible"},
                            {
                                "sender": "SYSTEM",
                                "recipient": "USER",
                                "content": "hidden evaluator",
                            },
                        ]
                    }
                },
                "user_simulator_usage": {"model_calls": 1},
            },
            runtime_instance_id=content_hash("fixture"),
        )

    provider.run_arm = run_arm
    monkeypatch.setattr(native, "_provider", lambda: provider)
    monkeypatch.setattr(native, "verify_native_checkout", lambda *a, **kw: tmp_path / "checkout")
    monkeypatch.setattr(native, "verify_source_files", lambda *a: None)
    monkeypatch.setattr(native, "native_process_scope", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(native, "_native_inputs", lambda *a: ({}, source["source_files"]))
    return env, source, provider


def test_original_native_episode_and_reflection_visibility(native_task, tmp_path, monkeypatch):
    from agentinputlab.recipes import skillopt_toolsandbox as native

    env, source, provider = native_task
    monkeypatch.setattr(native, "_launch", native._episode)
    adapter = native.create_toolsandbox_environment(cfg={}, configuration=env)
    row = adapter.rollout(adapter.build_train_env(1, 17), "skill", tmp_path / "out")[0]
    assert row["hard"] == row["soft"] == 0 and row["native_user_usage"] == {"model_calls": 1}
    assert "visible" in row["response"] and "hidden evaluator" not in row["response"]
    config, call = provider.calls[0]
    assert config["agent_base_url"] == env["backend"]["base_url"] and call["seed"] == 17
    state = call["arm_payload"]["intervention_execution"]["materialized_state"]
    assert state["context"][-1]["text"] == "SKILLOPT CURRENT FITTING SKILL\nskill"
    assert state["budget"] == source["fitting_rows"][0]["state"]["budget"]
    with pytest.raises(FileExistsError):
        adapter.rollout(adapter.build_train_env(1, 17), "skill", tmp_path / "out")
    assert len(provider.calls) == 1


@pytest.mark.parametrize("change", ["heldout", "duplicate", "altered", "diagnostic"])
def test_bad_rollout_never_launches(native_task, tmp_path, monkeypatch, change):
    from agentinputlab.recipes import skillopt_toolsandbox as native

    env, _, provider = native_task
    monkeypatch.setattr(native, "_launch", lambda *a: pytest.fail("unexpected episode"))
    adapter = native.create_toolsandbox_environment(cfg={}, configuration=env)
    batch = adapter.build_train_env(1, 17)
    kw = {}
    with pytest.raises(ValidationError):
        if change == "heldout":
            adapter.build_eval_env(1, "test", 17)
        elif change == "duplicate":
            batch *= 2
        elif change == "altered":
            batch[0]["question"] = "other"
        else:
            kw["diagnostic_mode"] = True
        adapter.rollout(batch, "skill", tmp_path / "out", **kw)
    assert not provider.calls
