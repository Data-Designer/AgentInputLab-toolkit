"""Synthetic closure/dispatch fixtures, not trained-model acceptance evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab import ContextBlock, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.curate.llmlingua.adapters import LLMLingua2OfficialAdapter
from agentinputlab.algorithms.curate.llmlingua.multilingual import (
    LLMLingua2BertMultilingualOfficialAdapter,
)
from agentinputlab.contracts import InputPatch, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import LLMLingua2TrainedArtifact, LLMLingua2TrainedCompressor
from agentinputlab.recipes.llmlingua2_training import PROTOCOL, RELEASE, _sha
from agentinputlab.runner import InterventionRunner


def artifact(tmp_path):
    root = tmp_path / "fit"
    root.mkdir()
    (root / "checkpoint").mkdir()
    files = {
        "model.safetensors": "fixture-not-weights",
        "config.json": '{"model_type":"bert","num_labels":2}',
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
    }
    for name, value in files.items():
        (root / "checkpoint" / name).write_text(value)
    (root / "checkpoint-manifest.json").write_text(
        json.dumps(
            dict(
                schema="agentinputlab.llmlingua2.token_checkpoint.v1",
                files={n: _sha(root / "checkpoint" / n) for n in files},
            )
        )
    )
    (root / "fit-plan.json").write_text("{}")
    (root / "fit-terminal.json").write_text(
        json.dumps(dict(status="trained_candidate", optimizer_steps=1, changed_parameter_tensors=1))
    )
    native = root / "native-bert-base-multilingual-cased"
    native.mkdir()
    metadata = dict(
        protocol=PROTOCOL,
        release_id=RELEASE,
        **{n + "_sha256": _sha(root / n) for n in ("fit-plan.json", "fit-terminal.json")},
        checkpoint_manifest_sha256=_sha(root / "checkpoint-manifest.json"),
        native_directory=native.name,
        native_files={},
    )
    (root / "artifact.json").write_text(
        json.dumps(dict(metadata, artifact_id=content_hash(metadata)))
    )
    return LLMLingua2TrainedArtifact.load(root)


def test_typed_deployment_keeps_distinct_identity_and_no_factory(tmp_path):
    fitted = artifact(tmp_path)
    method = fitted.intervention(config={"rate": 0.6})
    assert isinstance(method, LLMLingua2TrainedCompressor)
    assert method.profile_kind == "controlled_adapter" and method.release_id == RELEASE
    assert method.model_revision == "native-fit-" + fitted.artifact_id
    assert method.model_name == str(fitted.checkpoint) and method.config.rate == 0.6
    assert method.compressor_factory is None
    with pytest.raises(ValidationError, match="pinned checkpoint"):
        LLMLingua2BertMultilingualOfficialAdapter(model_revision=method.model_revision)
    with pytest.raises(ValidationError, match="typed trained"):
        LLMLingua2TrainedCompressor(fitted.root)
    with pytest.raises(ValidationError, match="identity differs"):
        LLMLingua2TrainedCompressor(replace(fitted, artifact_id="0" * 64))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "elsewhere"),
        ("model_revision", "changed"),
        ("release_id", "curate.llmlingua2.bert_multilingual_official"),
        ("profile_kind", "official_native"),
        ("model_config", {"trust_remote_code": True}),
        ("compressor_factory", object()),
    ],
)
def test_changed_deployment_binding_fails_before_native_load(tmp_path, field, value):
    method = artifact(tmp_path).intervention()
    setattr(method, field, value)
    with pytest.raises(ValidationError, match="binding changed"):
        method._load(use_llmlingua2=True)


def test_changed_checkpoint_or_replaced_directory_is_not_reloaded(tmp_path):
    fitted = artifact(tmp_path)
    method = fitted.intervention()
    (fitted.checkpoint / "model.safetensors").write_text("changed")
    with pytest.raises(ValidationError, match="binding changed"):
        method._verify_binding()


def test_native_patch_and_evidence_are_preserved_without_hiding_runtime(tmp_path, monkeypatch):
    method = artifact(tmp_path).intervention()
    state = FrozenAgentState(
        query="fixture", context=(ContextBlock(text="original visible context"),)
    )

    # Explicitly synthetic dispatch seam; no classifier is invoked in this unit test.
    def fixture_dispatch(self, state, context):
        return InputPatch.no_op(state, method_id=self.method_id, release_id=self.release_id)

    monkeypatch.setattr(LLMLingua2OfficialAdapter, "propose", fixture_dispatch)
    record = InterventionRunner().run(method, state, InterventionContext(seed=17))
    assert record.patch.apply(state).state_hash == state.state_hash
    assert record.patch.metadata["trained_artifact_id"] == method._artifact_id
    assert record.patch.metadata["generative_model_calls"] == 0
    assert record.patch.trace.events[0].name == "native_fitted_compressor"
    assert record.patch.trace.counters["cpu_seconds"] >= 0
    assert (
        record.patch.implementation_identity == "native_llmlingua2_dispatcher_with_fitted_candidate"
    )


def test_native_family_dispatch_does_not_rebind_loaded_weights(tmp_path, monkeypatch):
    fitted = artifact(tmp_path)
    method = fitted.intervention()
    native = SimpleNamespace(
        model_name=str(fitted.checkpoint),
        model=SimpleNamespace(config=SimpleNamespace(model_type="bert")),
    )
    monkeypatch.setattr(LLMLingua2OfficialAdapter, "_load", lambda self, **kwargs: native)
    monkeypatch.setattr(
        "agentinputlab.recipes.llmlingua2_inference.importlib.metadata.version",
        lambda name: "0.2.2",
    )
    model_identity = id(native.model)
    assert method._load(use_llmlingua2=True) is native
    assert (
        native.model_name == "bert-base-multilingual-cased" and id(native.model) == model_identity
    )
    assert (
        method.model_name == str(fitted.checkpoint)
        and method.model_revision == "native-fit-" + fitted.artifact_id
    )
    assert method._load(use_llmlingua2=True) is native
