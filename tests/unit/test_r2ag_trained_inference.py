"""Synthetic deployment/budget fixtures; not native generation evidence."""

import inspect
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import R2AGReaderDocument, R2AGTrainedArtifact
from agentinputlab.recipes import r2ag_inference as inference
from agentinputlab.recipes import r2ag_training as training


def test_gemma_signature_shim_preserves_native_values_and_return_object():
    class Model:
        config = SimpleNamespace(model_type="gemma4_unified")

        def __init__(self):
            self.calls, self.result = [], object()

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            raise AssertionError("signature test must not run a model")

        def prepare_inputs_for_generation(
            self, input_ids, use_cache=True, is_first_iteration=False, **kwargs
        ):
            self.calls.append((input_ids, use_cache, is_first_iteration, kwargs))
            return self.result

    model = Model()
    assert "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters
    binding = inference._generation_signature_compat(model)
    assert binding["applied"]
    assert "inputs_embeds" in inspect.signature(model.prepare_inputs_for_generation).parameters
    ids, embeds, mask = object(), object(), object()
    assert (
        model.prepare_inputs_for_generation(
            ids, use_cache=False, is_first_iteration=True, inputs_embeds=embeds, attention_mask=mask
        )
        is model.result
    )
    assert model.calls == [(ids, False, True, dict(inputs_embeds=embeds, attention_mask=mask))]
    assert inference._generation_signature_compat(model) == binding


def test_other_models_and_already_supported_signature_are_not_patched():
    model = SimpleNamespace(config=SimpleNamespace(model_type="other"))
    assert not inference._generation_signature_compat(model)["applied"]
    def fn(input_ids, inputs_embeds=None):
        return None
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="gemma4_unified"), prepare_inputs_for_generation=fn
    )
    assert not inference._generation_signature_compat(model)["applied"]
    assert model.prepare_inputs_for_generation is fn


def test_incompatible_gemma_signature_is_not_blindly_adapted():
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="gemma4_unified"),
        forward=lambda input_ids=None, inputs_embeds=None: None,
        prepare_inputs_for_generation=lambda input_ids: None,
    )
    with pytest.raises(ExternalDependencyError, match="contract differs"):
        inference._generation_signature_compat(model)


def test_generation_forward_meter_counts_real_current_inputs_not_full_cached_mask():
    observed = dict(
        reader_forward_attempts=0,
        reader_input_tokens_attempted=0,
        reader_input_token_unknown_attempts=0,
    )
    inference._count_generation_reader_request(
        observed, dict(inputs_embeds=SimpleNamespace(shape=(1, 20, 4)))
    )
    inference._count_generation_reader_request(
        observed,
        dict(
            input_ids=SimpleNamespace(shape=(1, 1)), attention_mask=SimpleNamespace(shape=(1, 21))
        ),
    )
    inference._count_generation_reader_request(observed, {})
    assert observed == dict(
        reader_forward_attempts=3,
        reader_input_tokens_attempted=21,
        reader_input_token_unknown_attempts=1,
    )


def fitted(tmp_path, monkeypatch):
    root = tmp_path / "fit"
    root.mkdir()
    (root / "fit-plan.json").write_text(
        json.dumps(
            dict(
                protocol=training.PROTOCOL,
                source_commit=training.R2AG_UPSTREAM_COMMIT,
                source_files=training.SOURCE_FILES,
                base_model_manifest_sha256="a" * 64,
            )
        )
    )
    (root / "fit-terminal.json").write_text(
        json.dumps(dict(status="trained_candidate", optimizer_steps=2))
    )
    (root / "parameter-proof.json").write_text("{}")
    metadata = dict(
        protocol=training.PROTOCOL,
        files={
            n: training._sha(root / n)
            for n in ("fit-plan.json", "fit-terminal.json", "parameter-proof.json")
        },
    )
    (root / "artifact.json").write_text(
        json.dumps(dict(metadata, artifact_id=content_hash(metadata)))
    )
    source, base = tmp_path / "source", tmp_path / "base"
    source.mkdir()
    base.mkdir()
    for name in training.SOURCE_FILES:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("source fixture, not native code")
    (base / "model.safetensors").write_text("fixture, not weights")
    manifest = tmp_path / "base-manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(training, "validate_r2ag_training_source", lambda *a: source)
    monkeypatch.setattr(inference, "_safe_model", lambda *a: (base, {}))
    artifact = R2AGTrainedArtifact.load(root)
    args = dict(
        checkout=source,
        base_model_root=base,
        base_model_manifest=manifest,
        base_model_manifest_sha256="a" * 64,
        source_and_asset_acknowledged=True,
    )
    return artifact, args


def test_typed_runtime_retains_artifact_and_separate_feature_cost(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    runtime = artifact.runtime(**args)
    assert runtime.fingerprint()["trained_artifact_id"] == artifact.artifact_id
    assert runtime.fingerprint()["embedding_features_precomputed_and_cost_separate"] is True
    assert runtime.fingerprint()["optimizer_loaded"] is False
    with pytest.raises(ValidationError, match="identity"):
        replace(artifact, artifact_id="0" * 64).runtime(**args)


@pytest.mark.parametrize("field,value", [("device", "cuda:2"), ("deployment_id", "0" * 64)])
def test_mutated_deployment_rejected_before_load(tmp_path, monkeypatch, field, value):
    artifact, args = fitted(tmp_path, monkeypatch)
    runtime = artifact.runtime(**args)
    setattr(runtime, field, value)
    with pytest.raises(ValidationError, match="changed"):
        runtime._load()


def test_model_tamper_and_symlink_are_not_silent_substitution(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    runtime = artifact.runtime(**args)
    p = runtime.base / "model.safetensors"
    moved = tmp_path / "moved"
    p.rename(moved)
    p.symlink_to(moved)
    with pytest.raises(ValidationError, match="symlinked"):
        runtime.fingerprint()


@pytest.mark.parametrize("features", [(True, 0, 1), (1, 0), (1, float("nan"), 0), (2, 0, 0)])
def test_document_requires_native_three_finite_features(features):
    with pytest.raises(ValidationError, match="features"):
        R2AGReaderDocument("id", "title", "text", features)


def test_ten_documents_and_generation_cap_checked_before_optional_load(tmp_path, monkeypatch):
    artifact, args = fitted(tmp_path, monkeypatch)
    runtime = artifact.runtime(**args)
    docs = [R2AGReaderDocument(str(i), "", "text", (1, 0, 0)) for i in range(10)]
    with pytest.raises(ValidationError, match="ten"):
        runtime.read(query="q", documents=docs[:9], output_root=tmp_path / "not-created")
    with pytest.raises(ValidationError, match="512"):
        runtime.read(
            query="q", documents=docs, max_new_tokens=513, output_root=tmp_path / "not-created"
        )
    assert not (tmp_path / "not-created").exists()


def test_missing_optional_runtime_has_install_instruction(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ExternalDependencyError, match=r"agentinputlab\[r2ag-fit\]"):
        inference._torch()


def test_safe_components_reject_tensor_dtype_and_nonfinite_values():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(r_former=torch.nn.Linear(3, 2), llama_proj=torch.nn.Linear(2, 4))
    state = {
        n: {k: v.detach().clone() for k, v in getattr(model, n).state_dict().items()}
        for n in ("r_former", "llama_proj")
    }
    inference._load_components(torch, model, state)
    state["llama_proj"]["weight"] = state["llama_proj"]["weight"].double()
    with pytest.raises(ValidationError, match="dtype"):
        inference._load_components(torch, model, state)
    state["llama_proj"]["weight"] = state["llama_proj"]["weight"].float().fill_(float("nan"))
    with pytest.raises(ValidationError, match="value"):
        inference._load_components(torch, model, state)
