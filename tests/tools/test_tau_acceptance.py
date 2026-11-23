from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from run_native_arena_acceptance import load_captured_checkpoint, provider_config


@pytest.mark.parametrize("arena", ["tau3", "retail"])
def test_tau_roles_budget_source_and_single_capture_are_explicit(tmp_path, arena):
    service = tmp_path / "service"
    service.mkdir()
    (service / "started.json").write_text(
        json.dumps({"model_json_hashes": {"config.json": "a" * 64}})
    )
    scorer = tmp_path / "src/tau2/evaluator/evaluator.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# fixture official scorer")
    model = {
        "id": "exact-backbone",
        "base_url": "http://127.0.0.1:8860/v1",
        "service_directory": str(service),
    }
    _, config, binding, environment_binding = provider_config(arena, model, {}, tmp_path)
    assert binding == environment_binding
    assert config["llm_agent"] == config["llm_user"] == "openai/exact-backbone"
    assert config["llm_args_agent"]["max_tokens"] == 512
    assert config["llm_args_agent"]["num_retries"] == 0
    assert config["budget_override"]["max_model_calls"] == 8
    assert config["budget_override"]["max_output_tokens"] == 4096
    assert config["llm_args_user"]["max_tokens"] == 1024
    assert (
        config["locked_dependency_hashes"]["scorer"]
        == hashlib.sha256(scorer.read_bytes()).hexdigest()
    )
    if arena == "retail":
        assert config["llm_judge"] == config["llm_agent"]
        assert config["llm_args_judge"]["max_tokens"] == 2048
        assert config["llm_args_judge"]["response_format"] == {"type": "json_object"}
    else:
        assert config["capture_max_attempts"] == 1
        assert config["capture_text_only_fallback"] is False
        provider = __import__(
            "workers.official_arenas.providers.tau3", fromlist=["Tau3Provider"]
        ).Tau3Provider()
        selected = {**config, "limit": 1, "capture_task_ids": ["task_002"]}
        assert provider._validate_config(selected)["capture_task_ids"] == ["task_002"]
        for invalid in ([], [""], [1], "task_001", ["task_001", "task_001"]):
            with pytest.raises(RuntimeError, match="capture_task_ids"):
                provider._validate_config({**config, "capture_task_ids": invalid})
        for invalid in (None, True, 0, 17):
            with pytest.raises(RuntimeError, match="capture_max_attempts"):
                provider._validate_config({**config, "capture_max_attempts": invalid})
        for invalid in (None, 1, "yes", True):
            with pytest.raises(RuntimeError, match="capture_text_only_fallback"):
                provider._validate_config({**config, "capture_text_only_fallback": invalid})


def test_native_saved_checkpoint_has_exact_identity_and_content_lock(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"checkpoint_id": "5", "protected": {"fixture": True}}))
    job = {
        "checkpoint_file": str(path),
        "checkpoint_id": "5",
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert load_captured_checkpoint(job)["checkpoint_id"] == "5"
    with pytest.raises(ValueError, match="ID differs"):
        load_captured_checkpoint({**job, "checkpoint_id": "9"})
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_captured_checkpoint(job)


def test_native_call_recorder_keeps_cost_and_drops_credentials(monkeypatch):
    from run_tau_acceptance import record_native_calls

    def completion(**kwargs):
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "model": "exact",
                "usage": {"completion_tokens": 512},
                "choices": [],
                "private": "must not enter result",
            }
        )

    module = SimpleNamespace(completion=completion)
    monkeypatch.setitem(__import__("sys").modules, "tau2.utils", SimpleNamespace(llm_utils=module))
    calls = record_native_calls()
    module.completion(model="exact", seed=17, max_tokens=512, api_key="fixture-do-not-save")
    assert len(calls) == 1
    assert calls[0]["response"]["usage"]["completion_tokens"] == 512
    assert "fixture-do-not-save" not in json.dumps(calls)
    assert "must not enter result" not in json.dumps(calls)


def test_real_generated_empty_turn_does_not_gate_other_declared_methods(tmp_path):
    from run_tau_acceptance import probe_allows_fanout

    row = {
        "status": "failed_terminal",
        "stage": "native_episode_and_official_scorer",
        "error": "AssistantMessage must have either content or tool_calls. Got AssistantMessage",
    }
    (tmp_path / "terminal.json").write_text(json.dumps(row))
    assert not probe_allows_fanout(tmp_path)  # no evidence of a consumed model call
    (tmp_path / "native-calls.json").write_text(
        json.dumps({"calls": [{"response": {"choices": []}}]})
    )
    assert probe_allows_fanout(tmp_path)
    (tmp_path / "terminal.json").write_text(json.dumps({**row, "error": "source closure mismatch"}))
    assert not probe_allows_fanout(tmp_path)


def test_target_lanes_advance_without_waiting_for_a_different_models_probe():
    from threading import Event

    from run_tau_acceptance import dispatch_model_shards

    faster_finished = Event()
    models = [{"id": "slow"}, {"id": "fast"}]
    jobs = [
        {"model": model, "release": release, "checkpoint_file": "fixture"}
        for model in models
        for release in ("baseline.no_op", "fixture.method")
    ]
    seen = []

    def one(job):
        if job["model"]["id"] == "slow" and job["release"] == "baseline.no_op":
            assert faster_finished.wait(5), "a global probe barrier would starve the fast target"
        if job["model"]["id"] == "fast" and job["release"] == "fixture.method":
            faster_finished.set()
        seen.append((job["model"]["id"], job["release"]))

    assert dispatch_model_shards(models, jobs, one, lambda job: True) == 0
    assert len(seen) == 4


def test_probe_uses_an_available_checkpoint_without_dropping_capture_failures():
    from run_tau_acceptance import dispatch_model_shards

    model = {"id": "fixture"}
    missing = {"model": model, "release": "baseline.no_op", "case": "failed_capture"}
    available = {
        "model": model,
        "release": "baseline.no_op",
        "case": "available",
        "checkpoint_file": "fixture",
    }
    seen = []
    assert (
        dispatch_model_shards(
            [model], [missing, available], seen.append, lambda job: job is available
        )
        == 0
    )
    assert seen == [available, missing]
