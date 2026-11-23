"""Asset-only public preparation; no model loading or generation."""
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab import RunConfig, recomp_configuration
from agentinputlab.contracts import ArtifactError, ValidationError
from workers.recomp import driver


@pytest.fixture
def template(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"synthetic-not-model-weights")
    manifest = {"schema_version": driver.CHECKPOINT_SCHEMA, "mode": "abstractive",
                "checkpoint_revision": "fixture", "tokenizer_revision": "fixture-tokenizer",
                "license": "NOASSERTION", "files": {"model.safetensors":
                    hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()}}
    (tmp_path / "checkpoint.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(driver, "verify_checkout", lambda path: SimpleNamespace(
        manifest_sha256=driver.DEFAULT_MANIFEST_SHA256, evidence_level="synthetic-fixture"))
    monkeypatch.setattr(driver, "_default_model_loader", lambda *a: pytest.fail("model load"))
    return RunConfig("curate.recomp.abstractive", parameters={"asset_license_acknowledged": True},
        runtime={"kind": "recomp_abstractive_worker", "checkout": str(tmp_path / "upstream"),
                 "checkpoint_root": str(root), "checkpoint_manifest_path": str(tmp_path / "checkpoint.json"),
                 "python_executable": sys.executable}, seed=37,
        retriever={"kind": "bm25", "documents": [{"doc_id": "fixture", "text": "visible document"}]})


def test_prepare_derives_parameters_without_model_work(template, tmp_path):
    proof = recomp_configuration.prepare(template, tmp_path / "ready")
    ready = RunConfig.from_file(proof["run_path"])
    assert ready.parameters["checkpoint_revision"] == "fixture"
    assert ready.parameters["tokenizer_revision"] == "fixture-tokenizer"
    assert ready.parameters["asset_manifest_hash"] == hashlib.sha256((tmp_path / "checkpoint.json").read_bytes()).hexdigest()
    assert ready.parameters["input_template_hash"] == driver.ABSTRACTIVE_INPUT_TEMPLATE_SHA256
    assert ready.seed == template.seed and ready.budget == template.budget
    assert proof["model_calls"] == 0 and proof["fit_performed"] is False
    ready.create_intervention()
    assert ready._runtime(load=True) is not None
    with pytest.raises(ValidationError, match="new output"):
        recomp_configuration.prepare(template, tmp_path / "ready")


@pytest.mark.parametrize("field", ["checkpoint_revision", "tokenizer_revision", "asset_manifest_hash", "input_template_hash"])
def test_prepare_never_overwrites_conflicting_identity(template, tmp_path, field):
    with pytest.raises(ValidationError, match="conflicts"):
        recomp_configuration.prepare(replace(template, parameters={**template.parameters, field: "wrong"}), tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_prepare_rejects_changed_checkpoint_without_output(template, tmp_path):
    (tmp_path / "checkpoint/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="checkpoint"):
        recomp_configuration.prepare(template, tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_selective_prepare_requires_reader(template, tmp_path):
    with pytest.raises(ValidationError, match="backend"):
        recomp_configuration.prepare(replace(template, release_id="curate.recomp.selective"), tmp_path / "ready")


def test_public_prepare_command(template, tmp_path, capsys):
    from agentinputlab.cli.main import main
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template.to_dict()))
    assert main(["config", "prepare-recomp", str(path), "--output-dir", str(tmp_path / "ready")]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
