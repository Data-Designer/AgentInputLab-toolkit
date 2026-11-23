"""Synthetic file fixtures; original archives are acceptance data, not package data."""

import hashlib
import json

import pytest

from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.algorithms.acquire.ircot_replay import (
    IRCOT_REPLAY_RELEASE,
    IRCOT_REPLAY_SCHEMA,
    load_ircot_prediction_archive,
)
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab.backends.llm import CallableBackend, GenerationResponse
from agentinputlab.contracts import AlgorithmOutputError, ArtifactError
from agentinputlab.contracts.serialization import content_hash


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    state = FrozenAgentState(query="SYNTHETIC QUESTION", task_id="case")
    provider = CallableBackend(
        lambda request: GenerationResponse("MUST NOT GENERATE"),
        revision="synthetic-revision",
        callable_id="synthetic-model",
    )
    predictions = {"case": "SYNTHETIC PREDICTION", "unselected": "PRESERVED OUTSIDE SCOPE"}
    inputs = {
        "case": dict(
            query=state.query,
            input_state_hash=state.state_hash,
            historical_usage={
                "model_calls": 2,
                "retrieval_calls": 3,
                "input_tokens": 17,
                "output_tokens": 9,
            },
            origin_receipt_sha256="a" * 64,
        )
    }
    manifest = dict(
        schema=IRCOT_REPLAY_SCHEMA,
        origin_kind="controlled_capture",
        run_revision="b" * 40,
        benchmark="hotpotqa.distractor",
        split="validation",
        generator_fingerprint=provider.fingerprint(),
        dependency_fingerprint={"corpus": "synthetic-source-v1", "index": "synthetic-index-v1"},
        provenance=dict(
            source_uri="urn:synthetic-fixture",
            source_artifact_sha256="c" * 64,
            run_config_sha256="d" * 64,
            query_source_sha256="e" * 64,
        ),
        selection_rule="source IDs, not scores",
        predictions=dict(
            path="predictions.json", sha256=save(tmp_path / "predictions.json", predictions)
        ),
        inputs=dict(path="inputs.json", sha256=save(tmp_path / "inputs.json", inputs)),
    )

    def build():
        manifest["predictions"]["sha256"] = save(tmp_path / "predictions.json", predictions)
        manifest["inputs"]["sha256"] = save(tmp_path / "inputs.json", inputs)
        pin = save(tmp_path / "manifest.json", manifest)
        method = create_intervention(
            IRCOT_REPLAY_RELEASE,
            manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256=pin,
            expected_generator_fingerprint_sha256=content_hash(provider.fingerprint()),
            benchmark="hotpotqa.distractor",
            split="validation",
        )
        return method

    return tmp_path, state, provider, predictions, inputs, manifest, build


def test_real_factory_composite_and_zero_new_model_calls(bundle):
    _, state, provider, _, _, _, build = bundle
    method = build()
    value = run_composite_intervention(method, state, InterventionContext(backend=provider))
    assert value["final_answer"] == "SYNTHETIC PREDICTION" and value["scored"] is False
    result = value["record"]["result"]
    usage = value["record"]["usage"]
    assert result["evidence"] == [] and not provider.calls
    assert usage["model_calls"] == usage["input_tokens"] == usage["output_tokens"] == 0
    assert result["metadata"]["historical_usage"]["model_calls"] == 2
    assert not result["metadata"]["historical_cost_recharged"]
    assert result["metadata"]["origin_kind"] == "controlled_capture"
    assert result["profile_kind"] == "controlled_adapter"
    assert method.archive.source_prediction_count == 2 and len(method.archive.records) == 1


@pytest.mark.parametrize("relative", [True, False])
def test_public_config_replay_from_another_directory(bundle, monkeypatch, capsys, relative):
    from agentinputlab import RunConfig
    from agentinputlab.cli.main import main

    root, state, provider, _, _, _, build = bundle
    build()
    manifest = root / "manifest.json"
    parameters = {
        "manifest_path": "manifest.json" if relative else str(manifest),
        "expected_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_generator_fingerprint_sha256": content_hash(provider.fingerprint()),
        "benchmark": "hotpotqa.distractor", "split": "validation",
    }
    config = root / "run.json"
    config.write_text(json.dumps({"release_id": IRCOT_REPLAY_RELEASE,
                                  "parameters": parameters}))
    state_file = root / "state.json"
    state_file.write_text(json.dumps({"query": state.query, "task_id": state.task_id}))
    elsewhere = root / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = RunConfig.from_file(config)
    assert loaded.parameters["manifest_path"] == str(manifest)
    assert main(["config", "run", str(config), "--state", str(state_file)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["final_answer"] == "SYNTHETIC PREDICTION"
    assert result["record"]["usage"]["model_calls"] == 0
    assert result["record"]["result"]["metadata"]["historical_usage"]["model_calls"] == 2
    manifest.write_text("{}")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        loaded.run(state)


def test_author_release_unknown_usage_is_not_fabricated_zero(bundle):
    _, state, _, _, inputs, manifest, build = bundle
    manifest["origin_kind"] = "author_release"
    inputs["case"]["historical_usage"] = None
    inputs["case"]["input_state_hash"] = None
    inputs["case"]["origin_receipt_sha256"] = None
    result = build().run(state, InterventionContext())
    assert result.profile_kind == "official_native"
    assert result.metadata["historical_usage_unknown"]
    assert result.metadata["historical_usage"] is None
    assert not result.metadata["current_seed_or_budget_rerun_claimed"]


def test_source_question_whitespace_is_not_normalized(bundle):
    _, _, _, _, inputs, _, build = bundle
    inputs["case"]["query"] = " SYNTHETIC QUESTION "
    inputs["case"]["input_state_hash"] = None
    method = build()
    result = method.run(
        FrozenAgentState(query=" SYNTHETIC QUESTION ", task_id="case"),
        InterventionContext(),
    )
    assert result.final_answer == "SYNTHETIC PREDICTION"
    with pytest.raises(ArtifactError, match="query mismatch"):
        method.run(
            FrozenAgentState(query="SYNTHETIC QUESTION", task_id="case"),
            InterventionContext(),
        )


def test_original_empty_prediction_retained_without_fallback(bundle):
    _, state, provider, predictions, _, _, build = bundle
    predictions["case"] = ""
    method = build()
    assert method.archive.records["case"]["status"] == "archived_empty_prediction"
    with pytest.raises(AlgorithmOutputError, match="empty prediction preserved"):
        method.run(state, InterventionContext(backend=provider))
    assert not provider.calls


@pytest.mark.parametrize(
    "state",
    [
        FrozenAgentState(query="DIFFERENT", task_id="case"),
        FrozenAgentState(query="SYNTHETIC QUESTION", task_id="missing"),
        FrozenAgentState(query="SYNTHETIC QUESTION", task_id="case", instruction="CHANGED"),
    ],
)
def test_exact_id_query_and_captured_state_are_required(bundle, state):
    *_, build = bundle
    with pytest.raises(ArtifactError):
        build().run(state, InterventionContext())


def test_wrong_model_cannot_be_relabelled(bundle):
    _, state, _, _, _, _, build = bundle
    other = CallableBackend(
        lambda request: GenerationResponse("MUST NOT CALL"),
        revision="different",
        callable_id="different-model",
    )
    with pytest.raises(ArtifactError, match="another live backbone"):
        build().run(state, InterventionContext(backend=other))
    assert not other.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_revision", "main"),
        ("origin_kind", "gold_oracle"),
        ("selection_rule", ""),
        ("generator_fingerprint", {}),
        ("dependency_fingerprint", {}),
        ("split", ""),
    ],
)
def test_invalid_provenance_fails_before_replay(bundle, field, value):
    *_, manifest, build = bundle
    manifest[field] = value
    with pytest.raises(ArtifactError):
        build()


def test_input_labels_or_unknown_fields_are_not_accepted(bundle):
    _, _, _, _, inputs, _, build = bundle
    inputs["case"]["gold_answer"] = "PRIVATE REFERENCE"
    with pytest.raises(ArtifactError, match="labels are not allowed"):
        build()


def test_retrieval_predictions_are_not_coerced_into_qa_answers(bundle):
    _, _, _, predictions, _, _, build = bundle
    predictions["case"] = ["paragraph-id"]
    with pytest.raises(ArtifactError, match="ID-to-text"):
        build()


def test_capture_requires_original_receipt(bundle):
    _, _, _, _, inputs, _, build = bundle
    inputs["case"]["origin_receipt_sha256"] = None
    with pytest.raises(ArtifactError, match="original inference receipt"):
        build()


def test_mutated_source_bytes_are_rejected_at_run(bundle):
    root, state, _, _, _, _, build = bundle
    method = build()
    (root / "predictions.json").write_text("{}")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        method.run(state, InterventionContext())


def test_manifest_and_nested_records_are_frozen(bundle):
    *_, build = bundle
    method = build()
    with pytest.raises(TypeError):
        method.archive.records["case"]["historical_usage"]["model_calls"] = 0


def test_archive_path_cannot_escape_or_be_a_symlink(bundle):
    root, _, _, _, _, manifest, build = bundle
    build()
    outside = root.parent / (root.name + "-other.json")
    outside.write_text("{}")
    manifest["predictions"]["path"] = "../" + outside.name
    with pytest.raises(ArtifactError, match="manifest directory"):
        build()


def test_original_duplicate_ids_are_rejected(bundle):
    root, _, _, _, _, manifest, build = bundle
    build()
    p = root / "duplicate.json"
    p.write_text('{"case":"FIRST","case":"SECOND"}')
    manifest["predictions"] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    pin = save(root / "changed-manifest.json", manifest)
    with pytest.raises(ArtifactError, match="unique keys"):
        load_ircot_prediction_archive(root / "changed-manifest.json", expected_sha256=pin)
