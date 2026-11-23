"""Public file bindings to existing process workers; no subprocess is executed."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from agentinputlab import InterventionContext, RunConfig
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from tests.unit.test_benchmark_configuration import Backend

PROFILES = [
    ("ask_or_assume_worker", "clarify.ask_or_assume.intent_monitor_component"),
    ("acon_observation_worker", "curate.acon.common_support_observation"),
    ("acon_observation_worker", "curate.acon.observation_runtime"),
    ("acon_history_worker", "curate.acon.history_runtime"),
    ("reformer_worker", "rewrite.reformer.repo_native_r_only"),
    ("reformer_worker", "rewrite.reformer.paper_spec_q_concat_r"),
    ("reformer_worker", "rewrite.reformer.controlled_r_only"),
    ("reformer_worker", "rewrite.reformer.controlled_q_concat_r"),
    ("infocqr_worker", "rewrite.infocqr.common_support_rw_zsl"),
]


def runtime(kind):
    value = {"kind": kind, "checkout": "/abs/source", "driver_path": "/abs/worker/driver.py",
             "driver_sha256": "a" * 64, "python_executable": "/abs/python",
             "python_executable_sha256": "b" * 64,
             "compatibility_manifest_path": "/abs/worker/compatibility_manifest.json",
             "compatibility_manifest_sha256": "c" * 64,
             "base_url": "http://localhost:9999/v1", "model_id": "synthetic"}
    if kind in {"acon_observation_worker", "acon_history_worker"}:
        value.update(optimizer_model_revision="fixture", target_model_revision="fixture",
                     tokenizer_revision="fixture")
    else:
        value["api_key_env"] = "TEST_WORKER_API_KEY"
    return value


@pytest.mark.parametrize("kind,release", PROFILES)
def test_process_config_shape_without_io_or_execution(kind, release, monkeypatch):
    from agentinputlab.experiments import production_builders as builders

    def forbidden(*args, **kwargs):
        pytest.fail("validation cannot construct a worker or access assets")

    monkeypatch.setattr(builders, "_build_external_runtime", forbidden)
    config = RunConfig(release, runtime=runtime(kind))
    assert config.validate()["runtime_ready"] is None
    with pytest.raises(ValidationError, match="requires a configured backend"):
        config._configured_context(None)
    with pytest.raises(ValidationError, match="supported only"):
        RunConfig("baseline.no_op", runtime=runtime(kind)).validate()


@pytest.mark.parametrize("kind,release", PROFILES)
def test_process_file_binds_original_worker_and_preserves_limits(kind, release, tmp_path, monkeypatch):
    from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

    value = runtime(kind)
    worker = tmp_path / "workers" / kind
    worker.mkdir(parents=True)
    for key, name, raw in [("driver", "driver.py", b"raise AssertionError('must not execute')"),
                           ("compatibility_manifest", "compatibility_manifest.json", b"{}")]:
        file = worker / name
        file.write_bytes(raw)
        value[key + "_path"] = str(file.relative_to(tmp_path))
        value[key + "_sha256"] = hashlib.sha256(raw).hexdigest()
    value["checkout"] = "source"
    value["python_executable"] = sys.executable
    value["python_executable_sha256"] = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    value["max_output_tokens"] = 123
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"release_id": release, "runtime": value}))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    context = config._configured_context(InterventionContext(backend=Backend([])))
    bound = context.external_runtime
    assert isinstance(bound, JsonLineSubprocessWorker)
    if kind in {"reformer_worker", "acon_observation_worker", "acon_history_worker"}:
        assert bound.fingerprint()["release_id"] == release
    if kind == "acon_history_worker":
        assert bound.fingerprint()["component_release_id"] == release
        assert bound.worker_id == "acon-history"
    assert bound.command[:2] == (sys.executable, str(worker / "driver.py"))
    assert config.runtime["checkout"] == str(tmp_path / "source")
    assert config.runtime["compatibility_manifest_path"] == str(worker / "compatibility_manifest.json")
    if kind in {"acon_observation_worker", "acon_history_worker"}:
        assert bound.environment["AGENTINPUTLAB_ACON_MAX_TOKENS"] == "123"
    else:
        options = json.loads(bound.command[bound.command.index("--backend-config-json") + 1])
        assert options["api_key_env"] == "TEST_WORKER_API_KEY"
        if kind == "ask_or_assume_worker":
            assert options["roles"]["intent"]["max_output_tokens"] == 123
        else:
            assert options["max_output_tokens"] == 123
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        RunConfig(release, runtime={**config.runtime, "driver_sha256": "f" * 64})._runtime(
            load=True, backend=Backend([]))
    with pytest.raises(ValidationError, match="twice"):
        config._configured_context(InterventionContext(backend=Backend([]), external_runtime=bound))


@pytest.mark.parametrize("kind,release", PROFILES)
@pytest.mark.parametrize("field,value", [("max_output_tokens", 0), ("driver_sha256", "bad"),
                                        ("shell_command", "arbitrary")])
def test_process_invalid_config_rejected(kind, release, field, value):
    with pytest.raises(ValidationError):
        RunConfig(release, runtime={**runtime(kind), field: value}).validate()


@pytest.mark.parametrize("release,semantics,mode", [
    ("rewrite.reformer.repo_native_r_only", "r_only", "automatic"),
    ("rewrite.reformer.paper_spec_q_concat_r", "q_concat_r", "automatic"),
    ("rewrite.reformer.controlled_r_only", "r_only", "manual"),
    ("rewrite.reformer.controlled_q_concat_r", "q_concat_r", "manual"),
])
def test_reformer_file_keeps_profile_semantics(release, semantics, mode):
    parameters = {"pattern_artifact_hash": "a" * 64, "source_license_acknowledged": True}
    if mode == "manual":
        with pytest.raises(ValidationError, match="require manual_pattern_id"):
            RunConfig(release, parameters=parameters).create_intervention()
        parameters["manual_pattern_id"] = "explicit-pattern"
    method = RunConfig(release, parameters=parameters).create_intervention()
    assert method.semantics == semantics
    assert method.pattern_mode == mode
    assert method.manual_pattern_id == ("explicit-pattern" if mode == "manual" else "")
    assert method.controls.max_model_calls == (1 if mode == "manual" else 2)


def acon_artifacts():
    return {"optimizer_model_revision": "fixture", "target_model_revision": "fixture",
            "tokenizer_revision": "fixture", "prompt_artifact_hash": "a" * 64,
            "guideline_artifact_hash": "b" * 64,
            "optimizer_deployment_fingerprint_hash": "c" * 64,
            "target_backend_fingerprint_hash": "d" * 64,
            "tokenizer_fingerprint_hash": "e" * 64}


def test_acon_file_parameters_construct_optional_artifact_identity(tmp_path):
    from agentinputlab.algorithms.curate.acon.common import ACONArtifactIdentity

    path = tmp_path / "acon.json"
    path.write_text(json.dumps({"release_id": "curate.acon.common_support_observation",
                                "parameters": {"artifacts": acon_artifacts()}}))
    method = RunConfig.from_file(path).create_intervention()
    assert isinstance(method.artifacts, ACONArtifactIdentity)
    assert method.artifacts.as_payload() == acon_artifacts()


@pytest.mark.parametrize("fault,match", [("unknown", "unknown parameters.artifacts fields"),
                                        ("missing", "invalid or missing fields"),
                                        ("hash", "SHA-256")])
def test_acon_optional_artifact_preserves_strict_field_validation(fault, match):
    artifacts = acon_artifacts()
    if fault == "unknown":
        artifacts["arbitrary_factory"] = "anything"
    elif fault == "missing":
        del artifacts["tokenizer_revision"]
    else:
        artifacts["prompt_artifact_hash"] = "not-a-hash"
    with pytest.raises(ValidationError, match=match):
        RunConfig("curate.acon.common_support_observation",
                  parameters={"artifacts": artifacts}).create_intervention()


def test_optional_conversion_does_not_invent_required_acon_artifacts():
    with pytest.raises(ValidationError, match="requires frozen artifact identities"):
        RunConfig("curate.acon.common_support_observation",
                  parameters={"artifacts": None}).create_intervention()


def test_optional_parameter_conversion_only_unwraps_one_type_plus_none():
    from typing import Optional

    from agentinputlab.algorithms.curate.acon.common import ACONArtifactIdentity
    from agentinputlab.configuration import _typed_parameter

    value = acon_artifacts()
    legacy_annotation = Optional[ACONArtifactIdentity]  # noqa: UP045 - verify Python 3.10 typing.Optional
    assert isinstance(_typed_parameter(value, legacy_annotation, "artifacts"), ACONArtifactIdentity)
    assert _typed_parameter(None, ACONArtifactIdentity | None, "artifacts") is None
    assert _typed_parameter(value, ACONArtifactIdentity | str | None, "artifacts") is value
