"""Installed transport preparation never executes a worker or a model."""
import hashlib
import json
import sys
from dataclasses import replace

import pytest

from agentinputlab import RunConfig, worker_preparation
from agentinputlab.contracts import ArtifactError, ValidationError
from tests.unit.test_run_subprocess_runtime_configuration import PROFILES


@pytest.fixture
def assets(tmp_path, monkeypatch):
    worker = tmp_path / "installed"
    worker.mkdir()
    (worker / "driver.py").write_text("raise AssertionError('must never execute')")
    source = tmp_path / "source"
    source.mkdir()
    raw = b"fixture source, not a real method"
    (source / "source.py").write_bytes(raw)
    (worker / "compatibility_manifest.json").write_text(json.dumps({
        "source_files": {"source.py": hashlib.sha256(raw).hexdigest()}}))
    monkeypatch.setattr(worker_preparation, "_installed_worker", lambda name: worker)
    return worker, source


def template(assets, kind, release):
    runtime = {"kind": kind, "checkout": str(assets[1]), "python_executable": sys.executable,
               "max_output_tokens": 123}
    if kind.startswith("acon_"):
        runtime.update(optimizer_model_revision="fixture", target_model_revision="fixture",
                       tokenizer_revision="fixture")
    return RunConfig(release, seed=37, runtime=runtime, backend={
        "base_url": "http://localhost:9999/v1", "model_id": "fixture",
        "deployment_identity": {"revision": "fixture"}})


@pytest.mark.parametrize("kind,release", PROFILES)
def test_all_process_profiles_bind_without_execution(assets, tmp_path, kind, release):
    config = template(assets, kind, release)
    proof = worker_preparation.prepare(config, tmp_path / "ready")
    ready = RunConfig.from_file(proof["run_path"])
    assert ready.runtime["python_executable"] == sys.executable
    assert ready.runtime["max_output_tokens"] == 123
    assert ready.seed == 37 and ready.budget == config.budget
    assert ready.parameters == config.parameters and ready.artifacts == config.artifacts
    assert ready.runtime["driver_sha256"] == hashlib.sha256((assets[0] / "driver.py").read_bytes()).hexdigest()
    assert proof["model_calls"] == 0 and not proof["worker_executed"]
    assert not proof["worker_dependencies_verified"]
    with pytest.raises(ValidationError, match="new output"):
        worker_preparation.prepare(config, tmp_path / "ready")


@pytest.mark.parametrize("field,value", [
    ("driver_sha256", "0" * 64), ("python_executable_sha256", "0" * 64),
    ("compatibility_manifest_sha256", "0" * 64), ("model_id", "other"),
    ("base_url", "http://other:9999/v1"), ("driver_path", "/untrusted/driver.py"),
])
def test_conflicting_bindings_never_silently_replaced(assets, tmp_path, field, value):
    config = template(assets, *PROFILES[0])
    with pytest.raises(ValidationError, match="conflicts"):
        worker_preparation.prepare(replace(config, runtime={**config.runtime, field: value}), tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_changed_source_is_rejected(assets, tmp_path):
    (assets[1] / "source.py").write_text("modified")
    with pytest.raises(ArtifactError, match="checksum"):
        worker_preparation.prepare(template(assets, *PROFILES[0]), tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_backend_template_options_propagate_without_overwriting_explicit_worker(assets, tmp_path):
    config = template(assets, "reformer_worker", "rewrite.reformer.repo_native_r_only")
    config = replace(config, backend={**config.backend,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    proof = worker_preparation.prepare(config, tmp_path / "inherited")
    assert RunConfig.from_file(proof["run_path"]).runtime["extra_body"] == config.backend["extra_body"]
    explicit = replace(config, runtime={**config.runtime, "extra_body": {"top_k": 20}})
    proof = worker_preparation.prepare(explicit, tmp_path / "explicit")
    assert RunConfig.from_file(proof["run_path"]).runtime["extra_body"] == explicit.runtime["extra_body"]


@pytest.mark.parametrize("value", [None, 12, False, "", "a\x00b"])
def test_invalid_driver_path_is_validation_error(assets, tmp_path, value):
    config = template(assets, *PROFILES[0])
    with pytest.raises(ValidationError, match="local path"):
        worker_preparation.prepare(replace(config, runtime={**config.runtime,
            "driver_path": value}), tmp_path / "ready")


def test_missing_executable_is_rejected(assets, tmp_path):
    config = template(assets, *PROFILES[0])
    with pytest.raises(ArtifactError, match="Python"):
        worker_preparation.prepare(replace(config, runtime={**config.runtime,
            "python_executable": str(tmp_path / "absent")}), tmp_path / "ready")


def test_cli_relative_source_and_interpreter_symlink(assets, tmp_path, capsys):
    from agentinputlab.cli.main import main

    python = tmp_path / "venv-python"
    python.symlink_to(sys.executable)
    config = template(assets, *PROFILES[0])
    config = replace(config, runtime={**config.runtime, "checkout": "source",
                                      "python_executable": "venv-python"})
    path = tmp_path / "template.json"
    path.write_text(json.dumps(config.to_dict()))
    assert main(["config", "prepare-worker", str(path), "--output-dir", str(tmp_path / "ready")]) == 0
    proof = json.loads(capsys.readouterr().out)
    assert RunConfig.from_file(proof["run_path"]).runtime["python_executable"] == str(python)
