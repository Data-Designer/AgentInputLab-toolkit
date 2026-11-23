"""Synthetic routing records exercise file binding, not new model inference."""
import json

import pytest

from agentinputlab import FrozenAgentState, InterventionContext, RunConfig
from agentinputlab.algorithms.acquire.adaptive_rag.replay import (
    AdaptiveRAGRecord,
    AdaptiveRAGSelectorReplay,
    AdaptiveRAGStrategy,
)
from agentinputlab.cli.main import main
from agentinputlab.contracts import ArtifactError, ContextBlock, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from tests.algorithms.unit.test_remaining_runtime import adaptive_prediction


def parameters(strategy=AdaptiveRAGStrategy.SINGLE_RETRIEVAL):
    record = AdaptiveRAGRecord(
        record_id="synthetic-case", split="dev", query="synthetic question",
        strategy=strategy, router_prediction=adaptive_prediction(strategy),
        dependency_hash="c" * 64, strategy_output_hash="d" * 64,
        contexts=() if strategy is AdaptiveRAGStrategy.NO_RETRIEVAL else (
            ContextBlock("synthetic fact", block_id="d1"),),
        selected_answer="archived answer; never inserted as evidence",
    )
    return dict(records=(record,), split="dev", dependency_hash="c" * 64,
                artifact_id="synthetic-v1", artifact_sha256="e" * 64)


@pytest.mark.parametrize("strategy", list(AdaptiveRAGStrategy))
def test_public_json_matches_native_typed_replay(strategy, tmp_path, capsys):
    values = parameters(strategy)
    state = FrozenAgentState(query="synthetic question", task_id="synthetic-case")
    expected = AdaptiveRAGSelectorReplay(**values).propose(state, InterventionContext())
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"release_id": "acquire.adaptive_rag.selector_replay",
                                  "parameters": to_jsonable(values)}))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(to_jsonable(state)))
    loaded = RunConfig.from_file(config)
    assert loaded.validate()["runtime_ready"] is None
    actual = loaded.create_intervention().propose(state, InterventionContext())
    assert to_jsonable(actual) == to_jsonable(expected)
    assert main(["config", "run", str(config), "--state", str(state_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == to_jsonable(expected.apply(state))
    assert output["record"]["usage"]["model_calls"] == 0


@pytest.mark.parametrize("fault,match", [
    ("labels", "cannot use gold"), ("route", "does not match"),
    ("split", "different split"), ("dependency", "mixes dependency"),
    ("duplicate", "duplicate IDs"), ("unknown", "unknown parameters.records"),
    ("context_unknown", "unknown parameters.records"),
    ("element", "must be an object"), ("container", "must be an array"),
])
def test_file_records_keep_native_guards(fault, match):
    values = to_jsonable(parameters())
    record = values["records"][0]
    if fault == "labels":
        record["router_prediction"]["evaluation_label_access"] = "gold"
    elif fault == "route":
        record["router_prediction"]["predicted_strategy"] = "ircot"
    elif fault == "split":
        record["split"] = "test"
    elif fault == "dependency":
        record["dependency_hash"] = "f" * 64
    elif fault == "duplicate":
        values["records"].append(dict(record))
    elif fault == "unknown":
        record["python_factory"] = "arbitrary.module"
    elif fault == "context_unknown":
        record["contexts"][0]["unexpected"] = True
    elif fault == "element":
        values["records"] = ["not a record"]
    else:
        values["records"] = {"not": "an array"}
    with pytest.raises((ValidationError, ArtifactError), match=match):
        RunConfig.from_dict({"release_id": "acquire.adaptive_rag.selector_replay",
                             "parameters": values}).create_intervention()


def file_config(tmp_path, mutate=None):
    import hashlib
    values = to_jsonable(parameters())
    payload = {"records": values.pop("records"), "dependency_hash": values["dependency_hash"]}
    if mutate:
        mutate(payload)
    raw = json.dumps(payload).encode()
    path = tmp_path / "selector.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    values["artifact_sha256"] = digest
    return RunConfig("acquire.adaptive_rag.selector_replay", parameters=values,
                     artifacts={"records": {"path": str(path), "sha256": digest}})


def test_pinned_file_is_equivalent_to_inline_records_and_never_retrains(tmp_path):
    config = file_config(tmp_path)
    assert config.validate()["runtime_ready"] is None
    state = FrozenAgentState(query="synthetic question", task_id="synthetic-case")
    method = config.create_intervention()
    values = parameters()
    values["artifact_sha256"] = config.parameters["artifact_sha256"]
    expected = AdaptiveRAGSelectorReplay(**values).propose(state, InterventionContext())
    assert method.propose(state, InterventionContext()) == expected
    assert config.run(state)["record"]["usage"]["model_calls"] == 0
    with pytest.raises(ArtifactError, match="no record"):
        config.run(FrozenAgentState(query=state.query, task_id="missing"))


@pytest.mark.parametrize("fault", ["gold", "unknown", "dependency", "missing", "empty", "duplicate"])
def test_artifact_file_cannot_bypass_native_guards(tmp_path, fault):
    def mutate(payload):
        if fault == "gold":
            payload["records"][0]["router_prediction"]["evaluation_label_access"] = "gold"
        elif fault == "unknown":
            payload["records"][0]["python_factory"] = "not.allowed"
        elif fault == "dependency":
            payload["dependency_hash"] = "a" * 64
        elif fault == "missing":
            del payload["records"]
        elif fault == "empty":
            payload["records"] = []
        else:
            payload["records"] *= 2
    config = file_config(tmp_path, mutate)
    with pytest.raises((ValidationError, ArtifactError)):
        config.create_intervention()


def test_file_digest_matches_declared_artifact_and_detects_mutation(tmp_path):
    from dataclasses import replace
    config = file_config(tmp_path)
    with pytest.raises(ValidationError, match="must match"):
        replace(config, parameters={**config.parameters, "artifact_sha256": "f" * 64}).validate()
    (tmp_path / "selector.json").write_text('{}')
    with pytest.raises(ArtifactError, match="checksum"):
        config.create_intervention()


def test_relative_selector_path_and_inspection_do_not_require_file(tmp_path):
    data = file_config(tmp_path).to_dict()
    data["artifacts"]["records"]["path"] = "selector.json"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))
    config = RunConfig.from_file(path)
    assert config.artifacts["records"]["path"] == str(tmp_path / "selector.json")
    config.create_intervention()
    (tmp_path / "selector.json").unlink()
    assert config.validate()["runtime_ready"] is None
    with pytest.raises(ArtifactError, match="unavailable"):
        config.create_intervention()
