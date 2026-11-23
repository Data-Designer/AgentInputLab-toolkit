"""Synthetic source/schema/control tests; not native neural training evidence."""

import json
from dataclasses import replace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ArtifactError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import R2AGFitControls, R2AGOfflineTraining
from agentinputlab.recipes.r2ag_training import (
    FEATURE_PROTOCOL,
    R2AG_UPSTREAM_COMMIT,
    R2AGTrainedArtifact,
    _audit_hotpot,
    _count_reader_request,
    _native_formatter,
    _observe_embedding,
    _quantized_reader_dependencies,
    _reader_training_precision,
    _reader_training_snapshot_preflight,
    _safe_model,
    _sha,
    validate_r2ag_features,
)


def test_factory_controls_preserve_declared_native_hyperparameters():
    method = create_intervention("acquire.r2ag.official_retrained")
    assert isinstance(method, R2AGOfflineTraining)
    c = method.controls
    assert c.epochs == c.batch_size == c.gradient_accumulation_steps == 2
    assert c.learning_rate == 2e-4 and c.max_grad_norm == 0.3 and c.warmup_ratio == 0.03
    assert c.max_steps is None and c.seed == 17


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("seed", -1),
        ("batch_size", True),
        ("max_steps", 0),
        ("max_steps", 10001),
        ("device", "cuda:2"),
        ("learning_rate", float("nan")),
        ("max_grad_norm", 0),
        ("warmup_ratio", 1),
        ("gradient_accumulation_steps", 0),
    ],
)
def test_invalid_controls_fail_before_optional_runtime(field, value):
    with pytest.raises(ValidationError):
        replace(R2AGFitControls(), **{field: value})


def rows(tmp_path, count=10):
    result = []
    for split in ("train", "dev"):
        p = tmp_path / (split + ".json")
        p.write_text(
            json.dumps(
                [
                    dict(
                        _id=split,
                        question=split + " question",
                        answer="answer",
                        context=[[str(i), ["fact"]] for i in range(count)],
                        supporting_facts=[["0", 0]],
                    )
                ]
            )
        )
        result.append(p)
    return result


def test_real_support_labels_and_exact_native_document_count(tmp_path):
    train, dev = rows(tmp_path)
    assert _audit_hotpot(train, dev, 100000)["files"]["train"]["rows"] == 1
    r = json.loads(train.read_text())
    r[0]["supporting_facts"] = []
    train.write_text(json.dumps(r))
    with pytest.raises(ValidationError, match="support"):
        _audit_hotpot(train, dev, 100000)
    train, dev = rows(tmp_path, 9)
    with pytest.raises(ValidationError, match="ten"):
        _audit_hotpot(train, dev, 100000)


def test_formatter_import_skips_module_level_GPU_and_seed_side_effects(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "runner.py").write_text("""raise RuntimeError("must not execute module top level")
class RRAGRunner:
    RETRIEVAL_TOKEN = "old"
    UNK_TOKEN = "old"
    @classmethod
    def set_retrieval_token(cls, value): cls.RETRIEVAL_TOKEN = value
    @classmethod
    def set_unk_token(cls, value): cls.UNK_TOKEN = value
    @staticmethod
    def format_instruction(example): return [x.replace(RRAGRunner.RETRIEVAL_TOKEN, RRAGRunner.UNK_TOKEN) for x in example["instruction"]]
""")
    formatter = _native_formatter(source)
    assert formatter.RETRIEVAL_TOKEN == formatter.UNK_TOKEN == "<R>"
    assert formatter.format_instruction(dict(instruction=["<R> source"])) == ["<R> source"]


def test_feature_closure_has_no_silent_file_substitution(tmp_path):
    names = ("feature-plan.json", "train-features.json", "dev-features.json")
    for name in names:
        (tmp_path / name).write_text("{}")
    manifest = tmp_path / "feature-manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                protocol=FEATURE_PROTOCOL,
                source_commit=R2AG_UPSTREAM_COMMIT,
                files={n: _sha(tmp_path / n) for n in names},
            )
        )
    )
    assert validate_r2ag_features(tmp_path, _sha(manifest))[0] == tmp_path
    (tmp_path / "train-features.json").write_text("changed")
    with pytest.raises(ValidationError, match="differs"):
        validate_r2ag_features(tmp_path, _sha(manifest))


def test_explicit_acknowledgement_before_native_model_or_training(tmp_path):
    with pytest.raises(ValidationError, match="acknowledgement"):
        R2AGOfflineTraining().fit(
            checkout=tmp_path,
            base_model_root=tmp_path,
            base_model_manifest=tmp_path / "missing",
            base_model_manifest_sha256="0" * 64,
            features_root=tmp_path,
            feature_manifest_sha256="0" * 64,
            output_root=tmp_path / "out",
        )


def test_trained_artifact_requires_closed_candidate(tmp_path):
    terminal = tmp_path / "fit-terminal.json"
    terminal.write_text(json.dumps(dict(status="failed_terminal", optimizer_steps=1)))
    metadata = dict(
        protocol="agentinputlab.r2ag.native_reader_fit.v1", files={terminal.name: _sha(terminal)}
    )
    (tmp_path / "artifact.json").write_text(
        json.dumps(dict(metadata, artifact_id=content_hash(metadata)))
    )
    with pytest.raises(ValidationError, match="closed"):
        R2AGTrainedArtifact.load(tmp_path)


def test_embedding_observer_covers_direct_forward_and_restores_on_failure():
    class Encoder:
        def __init__(self):
            self.output = object()

        def forward(self, features, *, fail=False):
            if fail:
                raise RuntimeError("native failure")
            return self.output

    encoder = Encoder()
    original = encoder.forward
    counters = dict(embedding_forward_attempts=0, embedding_forward_responses=0, input_tokens=0)
    with (
        pytest.raises(RuntimeError, match="native failure"),
        _observe_embedding(encoder, counters),
    ):
        assert encoder.forward({}) is encoder.output
        encoder.forward({}, fail=True)
    assert encoder.forward == original
    assert counters == dict(
        embedding_forward_attempts=2, embedding_forward_responses=1, input_tokens=0
    )


def snapshot(tmp_path, config=None):
    root = tmp_path / "snapshot"
    root.mkdir(exist_ok=True)
    for n, data in {
        "config.json": json.dumps(config or {"model_type": "qwen3"}),
        "tokenizer_config.json": "{}",
        "model.safetensors": "synthetic safe-layout fixture, not weights",
        "metal/model.bin": "synthetic ancillary payload, never deserialized",
    }.items():
        p = root / n
        p.parent.mkdir(exist_ok=True)
        p.write_text(data)
    manifest = tmp_path / "snapshot-manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="agentinputlab.r2ag.model_snapshot.v1",
                files={
                    p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*") if p.is_file()
                },
            )
        )
    )
    return root, manifest


def test_fp8_training_is_rejected_without_importing_or_executing_a_reader(tmp_path):
    root, _ = snapshot(tmp_path, {"quantization_config": {"quant_method": "fp8"}})
    with pytest.raises(ExternalDependencyError, match="input backward formula"):
        _reader_training_precision(root)


def fit_fixture(tmp_path, monkeypatch, root, manifest, digest=None):
    import agentinputlab.recipes.r2ag_training as module

    monkeypatch.setattr(module, "validate_r2ag_training_source", lambda _: tmp_path)
    return R2AGOfflineTraining().fit(
        checkout=tmp_path,
        base_model_root=root,
        base_model_manifest=manifest,
        base_model_manifest_sha256=digest or _sha(manifest),
        features_root=tmp_path / "unused-features",
        feature_manifest_sha256="0" * 64,
        output_root=tmp_path / "out",
        source_and_asset_acknowledged=True,
    )


def test_public_fp8_fit_rejects_verified_header_before_weight_io(tmp_path, monkeypatch):
    import agentinputlab.recipes.r2ag_training as module

    root, manifest = snapshot(tmp_path, {"quantization_config": {"quant_method": "fp8"}})

    def forbidden(*_args, **_kwargs):
        pytest.fail("known unsupported fit must not scan weights, features or import ML runtimes")

    monkeypatch.setattr(module, "_safe_model", forbidden)
    monkeypatch.setattr(module, "validate_r2ag_features", forbidden)
    monkeypatch.setattr(module.importlib.metadata, "version", forbidden)
    with pytest.raises(ExternalDependencyError, match="input backward formula"):
        fit_fixture(tmp_path, monkeypatch, root, manifest)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("damage", ["manifest", "config", "config-symlink"])
def test_training_header_integrity_precedes_unsupported_claim(tmp_path, monkeypatch, damage):
    root, manifest = snapshot(tmp_path, {"quantization_config": {"quant_method": "fp8"}})
    digest = _sha(manifest)
    if damage == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif damage == "config":
        (root / "config.json").write_text("{}")
    else:
        target = tmp_path / "config-target.json"
        (root / "config.json").rename(target)
        (root / "config.json").symlink_to(target)
    with pytest.raises(ArtifactError, match="checksum|symlink"):
        fit_fixture(tmp_path, monkeypatch, root, manifest, digest)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("damage", ["schema", "missing-config", "non-object-files"])
def test_training_header_requires_matching_schema_and_declared_config(tmp_path, damage):
    root, manifest = snapshot(tmp_path, {"quantization_config": {"quant_method": "fp8"}})
    spec = json.loads(manifest.read_text())
    if damage == "schema":
        spec["schema"] = "not-an-r2ag-model"
    elif damage == "missing-config":
        del spec["files"]["config.json"]
    else:
        spec["files"] = []
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValidationError, match="pinned model config"):
        _reader_training_snapshot_preflight(root, manifest, _sha(manifest))


@pytest.mark.parametrize("quantization", [None, {"quant_method": "mxfp4"}])
def test_other_reader_fits_still_verify_all_weights(tmp_path, monkeypatch, quantization):
    root, manifest = snapshot(
        tmp_path, {"model_type": "gpt_oss", "quantization_config": quantization}
    )
    (root / "model.safetensors").write_text("corrupt synthetic weights")
    with pytest.raises(ValidationError, match="differs"):
        fit_fixture(tmp_path, monkeypatch, root, manifest)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("quantization", [None, {}, {"quant_method": "mxfp4"}])
def test_other_precision_preflight_does_not_claim_training_certification(tmp_path, quantization):
    root, _ = snapshot(tmp_path, {"quantization_config": quantization})
    # Passing this config-only check is not a neural or backward test.
    assert _reader_training_precision(root) is None


def test_only_gptoss_mxfp4_metal_attachment_is_allowed(tmp_path):
    root, manifest = snapshot(
        tmp_path, {"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}
    )
    assert _safe_model(root, manifest, _sha(manifest))[0] == root
    cfg = root / "config.json"
    cfg.write_text(json.dumps({"model_type": "gpt_oss"}))
    spec = json.loads(manifest.read_text())
    spec["files"]["config.json"] = _sha(cfg)
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValidationError, match="unsafe"):
        _safe_model(root, manifest, _sha(manifest))


def test_root_pickle_weight_remains_rejected_even_for_gptoss(tmp_path):
    root, manifest = snapshot(
        tmp_path, {"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}
    )
    (root / "pytorch_model.bin").write_text("must never load")
    spec = json.loads(manifest.read_text())
    spec["files"]["pytorch_model.bin"] = _sha(root / "pytorch_model.bin")
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValidationError, match="unsafe"):
        _safe_model(root, manifest, _sha(manifest))


def test_unquantized_preflight_does_not_import_or_execute_kernels(tmp_path, monkeypatch):
    import sys

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setitem(sys.modules, "kernels", None)
    assert _quantized_reader_dependencies(tmp_path) == dict(quant_method=None, kernels=[])
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": None}))
    assert _quantized_reader_dependencies(tmp_path) == dict(quant_method=None, kernels=[])
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": "not a config"}))
    with pytest.raises(ValidationError, match="object or null"):
        _quantized_reader_dependencies(tmp_path)


def test_fp8_preflight_records_pinned_origin_without_a_model_forward(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}})
    )
    calls = []
    repo = "kernels-community/finegrained-fp8"
    info = SimpleNamespace(repo_id=repo, revision="b" * 40)
    kernel = ModuleType("kernels")
    kernel.get_loaded_kernels = lambda: [
        SimpleNamespace(repo_info=info),
        SimpleNamespace(repo_info=info),
    ]
    hub = ModuleType("transformers.integrations.hub_kernels")
    hub.get_kernel = lambda *a, **k: calls.append("unexpected download")
    fp8 = ModuleType("transformers.integrations.finegrained_fp8")
    fp8.load_finegrained_fp8_kernel = lambda: calls.append("software load")
    for name, module in {
        "kernels": kernel,
        "transformers.integrations.hub_kernels": hub,
        "transformers.integrations.finegrained_fp8": fp8,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    result = _quantized_reader_dependencies(tmp_path)
    assert result == dict(
        quant_method="fp8",
        kernels=[dict(repository=repo, revision="b" * 40)],
        preflight_neural_forwards=0,
    )
    assert calls == ["software load"]


def test_reader_token_meter_records_failed_request_before_response():
    from types import SimpleNamespace

    mask = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1533))
    observed = dict(
        reader_forward_attempts=0,
        reader_input_tokens_attempted=0,
        reader_input_token_unknown_attempts=0,
        reader_forward_responses=0,
    )
    _count_reader_request(observed, {"attention_mask": mask})
    assert observed == dict(
        reader_forward_attempts=1,
        reader_input_tokens_attempted=1533,
        reader_input_token_unknown_attempts=0,
        reader_forward_responses=0,
    )
    _count_reader_request(observed, {})
    assert observed["reader_input_token_unknown_attempts"] == 1
    assert observed["reader_input_tokens_attempted"] == 1533
