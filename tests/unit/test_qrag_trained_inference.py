"""Synthetic binding fixtures, not evidence of real trained-model acceptance."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import QRAGFitControls, QRAGTrainedPolicyArtifact
from agentinputlab.recipes import qrag_inference as inference
from agentinputlab.recipes.qrag import PROTOCOL, QRAG_UPSTREAM_COMMIT, SOURCE_FILES, _sha


def fitted(tmp_path, monkeypatch):
    root = tmp_path / "fit"
    root.mkdir()
    (root / "native-checkpoints").mkdir()
    for name, value in (
        ("native-final.pt", "final fixture, not weights"),
        ("native-checkpoints/update-00000001.pt", "best fixture, not weights"),
    ):
        (root / name).write_text(value)
    (root / "fit-plan.json").write_text(
        json.dumps(
            dict(
                source_commit=QRAG_UPSTREAM_COMMIT,
                source_files=SOURCE_FILES,
                initial_manifest_sha256="a" * 64,
                controls=asdict(QRAGFitControls()),
            )
        )
    )
    terminal = dict(
        status="trained_candidate",
        optimizer_steps=2,
        updates=[{}, {}],
        checkpoint_storage_events=[
            dict(
                logical_path="/old/native/model_best.pt",
                immutable_snapshot_path="/old/native-checkpoints/update-00000001.pt",
                native_update_step=1,
            )
        ],
    )
    (root / "fit-terminal.json").write_text(json.dumps(terminal))
    metadata = dict(
        protocol=PROTOCOL,
        files={
            n: _sha(root / n) for n in ("fit-plan.json", "fit-terminal.json", "native-final.pt")
        },
    )
    (root / "artifact.json").write_text(
        json.dumps(dict(metadata, artifact_id=content_hash(metadata)))
    )
    source, initial = tmp_path / "source", tmp_path / "initial"
    source.mkdir()
    initial.mkdir()
    (source / "native.py").write_text("fixture")
    (initial / "model.safetensors").write_text("fixture, not weights")
    manifest = tmp_path / "initial-manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(inference, "validate_qrag_training_source", lambda p: source)
    monkeypatch.setattr(inference, "validate_qrag_initial_model", lambda *a: (initial, {}))
    artifact = QRAGTrainedPolicyArtifact.load(root)
    args = dict(
        checkout=source,
        initial_checkpoint=initial,
        initial_manifest=manifest,
        initial_manifest_sha256="a" * 64,
        source_and_asset_acknowledged=True,
        checkpoint_sha256=_sha(root / "native-checkpoints/update-00000001.pt"),
    )
    return artifact, args


def test_native_best_default_and_final_have_distinct_identities(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    best = artifact.runtime(**args)
    assert best.selection == "native_best"
    assert best.checkpoint.name == "update-00000001.pt"
    final = artifact.runtime(
        **{**args, "checkpoint_sha256": _sha(artifact.checkpoint)}, selection="final"
    )
    assert final.checkpoint == artifact.checkpoint
    assert final.deployment_id != best.deployment_id
    assert best.fingerprint()["trained_artifact_id"] == artifact.artifact_id
    assert best.fingerprint()["optimizer_state_loaded"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda:2"),
        ("artifact_id", "0" * 64),
        ("selection", "final"),
        ("deployment_id", "0" * 64),
        ("controls", QRAGFitControls(seed=18)),
    ],
)
def test_mutated_runtime_binding_rejected_before_load(tmp_path, monkeypatch, field, value):
    artifact, args = fitted(tmp_path, monkeypatch)
    runtime = artifact.runtime(**args)
    setattr(runtime, field, value)
    with pytest.raises(ValidationError, match="changed"):
        runtime.fingerprint()


def test_modified_weights_and_source_and_bad_checksum_are_rejected(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="checksum"):
        artifact.runtime(**{**args, "checkpoint_sha256": "0" * 64})
    runtime = artifact.runtime(**args)
    runtime.checkpoint.write_text("modified")
    with pytest.raises(ValidationError, match="changed"):
        runtime._load()
    with pytest.raises(ValidationError, match="identity differs"):
        replace(artifact, artifact_id="0" * 64).runtime(**args)


@pytest.mark.parametrize("selection", ["untrained", None, "model_last"])
def test_invalid_checkpoint_choice_is_not_a_fallback(tmp_path, monkeypatch, selection):
    artifact, args = fitted(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="selection"):
        artifact.runtime(**args, selection=selection)


def test_acknowledgement_and_expected_checkpoint_are_required(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="acknowledgement"):
        artifact.runtime(**{**args, "source_and_asset_acknowledged": False})
    with pytest.raises(ValidationError, match="SHA-256"):
        artifact.runtime(**{**args, "checkpoint_sha256": "latest"})


def test_safe_inference_state_loader_strictness():
    torch = pytest.importorskip("torch")
    modules = {
        name: torch.nn.Linear(2, 1)
        for name in ("critic", "policy", "v_net_target", "action_embed_target")
    }
    agent = SimpleNamespace(**modules, random_policy=torch.nn.Identity(), alpha=1.0)
    checkpoint = {
        k: {n: v.detach().clone() for n, v in m.state_dict().items()} for k, m in modules.items()
    }
    checkpoint.update(random_policy={}, alpha=0.05, critic_optim={"never": "load"}, scheduler={})
    inference._load_inference_states(torch, agent, checkpoint)
    assert agent.alpha == 0.05 and all(not m.training for m in modules.values())
    assert all(not p.requires_grad for m in modules.values() for p in m.parameters())
    checkpoint["critic"]["weight"] = checkpoint["critic"]["weight"].double()
    with pytest.raises(ValidationError, match="dtype"):
        inference._load_inference_states(torch, agent, checkpoint)


def test_runtime_context_cannot_be_substituted(tmp_path, monkeypatch):
    from agentinputlab import FrozenAgentState, InterventionContext

    artifact, args = fitted(tmp_path, monkeypatch)
    method = inference.QRAGTrainedPolicyIntervention(artifact.runtime(**args))
    assert method.release_id == "training.qrag.value_policy"
    with pytest.raises(ValidationError, match="substituted"):
        method.propose(
            FrozenAgentState(query="fixture"), InterventionContext(external_runtime=object())
        )


def test_missing_optional_runtime_has_install_guidance(monkeypatch):
    import sys

    from agentinputlab.contracts import ExternalDependencyError

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ExternalDependencyError, match="agentinputlab\\[qrag-fit\\]"):
        inference._torch()


def test_replaced_checkpoint_symlink_rejected(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    runtime = artifact.runtime(**args)
    original = tmp_path / "moved-snapshot.pt"
    runtime.checkpoint.rename(original)
    runtime.checkpoint.symlink_to(original)
    with pytest.raises(ValidationError, match="changed|symlinked"):
        runtime.fingerprint()
