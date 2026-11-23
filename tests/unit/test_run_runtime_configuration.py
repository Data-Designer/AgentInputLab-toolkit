"""Configuration binding regressions, not model-quality acceptance."""

import json
from pathlib import Path

import pytest

from agentinputlab import InterventionContext, RunConfig
from agentinputlab.contracts.errors import ArtifactError, ValidationError


def qrag():
    return {"kind": "qrag_common_support", "source_root": "/abs/source",
            "checkpoint_root": "/abs/policy", "embedder_root": "/abs/embedder",
            "compatibility_manifest_path": "/abs/compatibility.json",
            "compatibility_manifest_sha256": "a" * 64,
            "embedder_manifest_sha256": "b" * 64, "device": "cpu"}


def r2ag():
    return {"kind": "r2ag_common_support", "source_root": "/abs/source",
            "artifact_root": "/abs/artifact", "manifest_sha256": "a" * 64,
            "base_model_root": "/abs/base", "base_model_load_set_sha256": "b" * 64,
            "feature_model_root": "/abs/features", "feature_model_load_set_sha256": "c" * 64,
            "reader_device": "cpu", "feature_device": "cpu"}


def recomp():
    return {"kind": "recomp_extractive_worker", "checkout": "/abs/source",
            "checkpoint_root": "/abs/checkpoint",
            "checkpoint_manifest_path": "/abs/checkpoint.json",
            "checkpoint_manifest_sha256": "a" * 64,
            "compatibility_manifest_path": "/abs/compatibility.json",
            "compatibility_manifest_sha256": "b" * 64, "device": "cpu"}


@pytest.mark.parametrize("release,runtime", [
    ("rewrite.query2doc.querygym_live", {"kind": "querygym_query2doc", "checkout": "/abs/source"}),
    ("acquire.qrag.common_support", qrag()),
    ("acquire.r2ag.common_support", r2ag()),
    ("curate.recomp.extractive", recomp()),
    ("curate.recomp.common_support_extractive", recomp()),
])
def test_native_runtime_shape_validation_does_not_load(release, runtime, monkeypatch):
    from agentinputlab.experiments import production_builders as builders

    def forbidden(*args, **kwargs):
        pytest.fail("validation must not construct runtime, read assets, or call models")

    monkeypatch.setattr(builders, "_build_external_runtime", forbidden)
    assert RunConfig(release, runtime=runtime).validate()["runtime_ready"] is None
    with pytest.raises(ValidationError, match="supported only"):
        RunConfig("baseline.no_op", runtime=runtime).validate()


@pytest.mark.parametrize("runtime", [
    {"kind": "querygym_query2doc"},
    {"kind": "querygym_query2doc", "checkout": "/x", "factory": "arbitrary"},
    {**qrag(), "embedder_manifest_sha256": "invalid"},
    {**qrag(), "checkpoint_root": "relative"},
])
def test_runtime_rejects_missing_or_invalid_input(runtime):
    release = ("rewrite.query2doc.querygym_live" if runtime["kind"] == "querygym_query2doc"
               else "acquire.qrag.common_support")
    with pytest.raises(ValidationError):
        RunConfig(release, runtime=runtime).validate()


def test_runtime_kind_has_actionable_type_error():
    with pytest.raises(ValidationError, match="runtime.kind must be a string"):
        RunConfig("baseline.no_op", runtime={"kind": []}).validate()


def test_runtime_paths_relative_to_config_not_cwd(tmp_path, monkeypatch):
    runtime = qrag()
    for key in ("source_root", "checkpoint_root", "embedder_root", "compatibility_manifest_path"):
        runtime[key] = key
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"release_id": "acquire.qrag.common_support", "runtime": runtime}))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    config.validate()
    for key in ("source_root", "checkpoint_root", "embedder_root", "compatibility_manifest_path"):
        assert config.runtime[key] == str(tmp_path / key)


def test_querygym_uses_configured_backend_and_rejects_duplicate_runtime(monkeypatch):
    from agentinputlab.experiments import production_builders as builders
    from tests.unit.test_benchmark_configuration import Backend

    backend, sentinel = Backend([]), object()
    seen = []
    monkeypatch.setattr(builders, "_build_external_runtime",
                        lambda config, llm: seen.append((config, llm)) or sentinel)
    config = RunConfig("rewrite.query2doc.querygym_live",
                       runtime={"kind": "querygym_query2doc", "checkout": "/abs/source"})
    with pytest.raises(ValidationError, match="requires a configured backend"):
        config._configured_context(None)
    context = config._configured_context(InterventionContext(backend=backend))
    assert context.external_runtime is sentinel
    assert seen[0][1] is backend and seen[0][0]["checkout"] == Path("/abs/source")
    with pytest.raises(ValidationError, match="twice"):
        config._configured_context(InterventionContext(backend=backend, external_runtime=sentinel))


def test_qrag_uses_original_frozen_policy_constructor(monkeypatch):
    from agentinputlab.algorithms.acquire import qrag as module

    seen = []
    monkeypatch.setattr(module, "LocalQRAGFrozenPolicyRuntime", lambda **kw: seen.append(kw) or kw)
    RunConfig("acquire.qrag.common_support", runtime=qrag())._runtime(load=True)
    assert seen[0]["checkpoint_root"] == Path("/abs/policy")
    assert seen[0]["compatibility_manifest_sha256"] == "a" * 64
    assert seen[0]["embedder_manifest_sha256"] == "b" * 64
    assert seen[0]["device"] == "cpu"


def test_r2ag_file_paths_and_original_constructors(tmp_path, monkeypatch):
    from agentinputlab.algorithms.acquire import r2ag as module

    monkeypatch.delenv(module.R2AG_RUNTIME_SOCKET_ENV, raising=False)
    runtime = r2ag()
    keys = ("source_root", "artifact_root", "base_model_root", "feature_model_root")
    for key in keys:
        runtime[key] = key
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"release_id": "acquire.r2ag.common_support", "runtime": runtime}))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    config.validate()
    for key in keys:
        assert config.runtime[key] == str(tmp_path / key)
    seen = {}

    def capture(name):
        def construct(**kwargs):
            seen[name] = kwargs
            return kwargs
        return construct

    monkeypatch.setattr(module, "LocalR2AGCommonSupportReader", capture("reader"))
    monkeypatch.setattr(module, "LocalSentenceTransformerR2AGFeatureEncoder", capture("encoder"))
    monkeypatch.setattr(module, "LocalR2AGCommonSupportRuntime", capture("runtime"))
    result = config._runtime(load=True)
    assert seen["reader"]["artifact_root"] == tmp_path / "artifact_root"
    assert seen["reader"]["base_model_load_set_sha256"] == "b" * 64
    assert seen["reader"]["manifest_sha256"] == "a" * 64
    assert seen["reader"]["device"] == "cpu"
    assert seen["encoder"] == {"model_root": tmp_path / "feature_model_root",
                               "model_load_set_sha256": "c" * 64, "device": "cpu"}
    assert result == {"reader": seen["reader"], "feature_encoder": seen["encoder"]}
    with pytest.raises(ValidationError, match="twice"):
        config._configured_context(InterventionContext(external_runtime=object()))


@pytest.mark.parametrize("field,value", [("manifest_sha256", "bad"),
                                        ("base_model_root", "relative"),
                                        ("factory", "arbitrary")])
def test_r2ag_invalid_binding_rejected(field, value):
    with pytest.raises(ValidationError):
        RunConfig("acquire.r2ag.common_support", runtime={**r2ag(), field: value}).validate()


def test_recomp_relative_paths_and_original_driver(tmp_path, monkeypatch):
    import hashlib

    from workers.recomp import driver

    runtime = recomp()
    keys = ("checkout", "checkpoint_root", "checkpoint_manifest_path",
            "compatibility_manifest_path")
    for key in keys:
        runtime[key] = key
    for key in ("checkpoint_manifest", "compatibility_manifest"):
        raw = b'{"test": "constructor binding only"}'
        (tmp_path / runtime[key + "_path"]).write_bytes(raw)
        runtime[key + "_sha256"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"release_id": "curate.recomp.extractive", "runtime": runtime}))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    seen = []
    monkeypatch.setattr(driver, "RECOMPDriver", lambda **kw: seen.append(kw) or kw)
    config._runtime(load=True)
    assert seen[0]["checkout_path"] == tmp_path / "checkout"
    assert seen[0]["checkpoint_path"] == tmp_path / "checkpoint_root"
    assert seen[0]["checkpoint_manifest_path"] == tmp_path / "checkpoint_manifest_path"
    assert seen[0]["manifest_path"] == tmp_path / "compatibility_manifest_path"
    assert seen[0]["device"] == "cpu"
    assert callable(seen[0]["model_loader"])
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        RunConfig("curate.recomp.extractive", runtime={**config.runtime,
                  "checkpoint_manifest_sha256": "c" * 64})._runtime(load=True)


@pytest.mark.parametrize("release", ["curate.recomp.abstractive", "curate.recomp.selective"])
def test_recomp_extractive_runtime_cannot_bind_other_variants(release):
    with pytest.raises(ValidationError, match="supported only"):
        RunConfig(release, runtime=recomp())._runtime(load=False)


@pytest.mark.parametrize("field,value", [("device", "shell"),
                                        ("checkpoint_manifest_sha256", "bad"),
                                        ("factory", "arbitrary")])
def test_recomp_invalid_binding_rejected(field, value):
    with pytest.raises(ValidationError):
        RunConfig("curate.recomp.extractive", runtime={**recomp(), field: value})._runtime(load=False)


def test_querygym_evaluation_captures_both_calls_and_charges_once(monkeypatch):
    from agentinputlab.algorithms.reformulate.query2doc import QueryGymRawResult
    from agentinputlab.backends.llm import GenerationRequest
    from agentinputlab.experiments import production_builders as builders
    from tests.algorithms.unit.test_query2doc import RecordingQueryGymRuntime
    from tests.unit.test_benchmark_configuration import Backend, evaluate

    instances = []

    class BoundRuntime(RecordingQueryGymRuntime):
        def __init__(self, backend):
            self.backend = backend
            instances.append(self)

        def reformulate(self, request):
            response = self.backend.generate(GenerationRequest(prompt=request.query, max_tokens=4))
            return QueryGymRawResult(reformulated=response.text, usage={
                "model_calls": 1, "retrieval_calls": 0, "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens})

    monkeypatch.setattr(builders, "_build_external_runtime", lambda config, backend: BoundRuntime(backend))
    config = RunConfig("rewrite.query2doc.querygym_live",
                       runtime={"kind": "querygym_query2doc", "checkout": "/abs/source"})
    backend = Backend([("expanded", 2), ("answer", 3)] * 2)
    result = evaluate(config, backend)
    assert result["scored_count"] == 2, result
    assert len(backend.requests) == 4
    assert len(instances) == 3  # preflight plus a fresh binding for each case
    for row in result["records"]:
        assert row["usage"]["model_calls"] == 2
        assert row["usage"]["output_tokens"] == 5
        assert [call["response"]["text"] for call in row["model_responses"]] == ["expanded", "answer"]
