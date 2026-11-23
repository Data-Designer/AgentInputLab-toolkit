"""File entry to the existing RECOMP worker, without neural inference."""
import hashlib
import json
import sys
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.curate.recomp import RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker


def configuration(tmp_path, release="curate.recomp.abstractive"):
    manifest = tmp_path / "checkpoint.json"
    manifest.write_text(json.dumps({"mode": "abstractive", "checkpoint_revision": "checkpoint",
                                    "tokenizer_revision": "tokenizer"}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    parameters = {"checkpoint_revision": "checkpoint", "tokenizer_revision": "tokenizer",
                  "asset_manifest_hash": digest, "asset_license_acknowledged": True}
    if release.endswith("abstractive"):
        parameters["input_template_hash"] = RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256
    return RunConfig(release, parameters=parameters, runtime={
        "kind": "recomp_abstractive_worker", "checkout": str(tmp_path / "source"),
        "checkpoint_root": str(tmp_path / "checkpoint"),
        "checkpoint_manifest_path": str(manifest), "python_executable": sys.executable,
        "device": "cpu", "timeout_seconds": 123})


@pytest.mark.parametrize("release", ["curate.recomp.abstractive", "curate.recomp.selective"])
def test_native_file_entry_binds_original_worker_without_execution(tmp_path, release, monkeypatch):
    monkeypatch.setattr(JsonLineSubprocessWorker, "call", lambda *a: pytest.fail("execute"))
    config = configuration(tmp_path, release)
    config.create_intervention()
    assert config.validate()["runtime_ready"] is None
    worker = config._runtime(load=True)
    assert isinstance(worker, JsonLineSubprocessWorker)
    assert worker.command[0] == sys.executable
    assert worker.command[1].endswith("workers/recomp/driver.py")
    assert "--acknowledge-noassertion-checkpoint" in worker.command
    assert worker.timeout_seconds == 123
    assert worker.identity["checkpoint_manifest_sha256"] == config.parameters["asset_manifest_hash"]
    assert worker.environment["HF_HUB_OFFLINE"] == "1"
    assert worker.environment["CUDA_VISIBLE_DEVICES"] == ""


def test_validate_does_not_read_assets(tmp_path):
    config = configuration(tmp_path)
    (tmp_path / "checkpoint.json").unlink()
    assert config.validate()["runtime_ready"] is None


def test_paths_follow_config_not_cwd(tmp_path, monkeypatch):
    value = configuration(tmp_path).to_dict()
    value["runtime"].update(checkout="source", checkpoint_root="checkpoint",
                             checkpoint_manifest_path="checkpoint.json")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    worker = config._runtime(load=True)
    assert worker.command[worker.command.index("--checkpoint") + 1] == str(tmp_path / "checkpoint")


def test_bad_manifest_rejected_before_worker(tmp_path):
    config = configuration(tmp_path)
    (tmp_path / "checkpoint.json").write_text("{}")
    with pytest.raises(ArtifactError, match="checksum"):
        config._runtime(load=True)


@pytest.mark.parametrize("field,value", [("device", "bad"), ("timeout_seconds", 0),
    ("timeout_seconds", float("inf")), ("python_executable", ""), ("shell_command", "anything")])
def test_bad_runtime_rejected_without_execution(tmp_path, field, value):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError):
        replace(config, runtime={**config.runtime, field: value})._runtime(load=False)


def test_other_profile_cannot_use_abstractive_worker(tmp_path):
    with pytest.raises(ValidationError, match="supported only"):
        replace(configuration(tmp_path), release_id="curate.recomp.extractive")._runtime(load=False)


def test_unacknowledged_assets_are_not_loaded(tmp_path):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError, match="acknowledgement"):
        replace(config, parameters={**config.parameters, "asset_license_acknowledged": False})._runtime(load=True)


def test_manifest_revision_must_match_method(tmp_path):
    config = configuration(tmp_path)
    with pytest.raises(ArtifactError, match="checkpoint_revision"):
        replace(config, parameters={**config.parameters, "checkpoint_revision": "different"})._runtime(load=True)
