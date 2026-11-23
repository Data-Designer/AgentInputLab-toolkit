"""Synthetic fixtures test loading, not new scientific/model acceptance."""

import hashlib
import json

import pytest

from agentinputlab import FrozenAgentState, InterventionContext, RunConfig
from agentinputlab.cli.main import main
from agentinputlab.configuration import load_run_state
from agentinputlab.contracts.errors import ArtifactError, ValidationError


def save(tmp_path, name, payload):
    raw = json.dumps(payload).encode()
    path = tmp_path / name
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize("release", ["recipe.ace.eval_only", "recipe.skillopt.eval_only",
                                    "recipe.mce.text_artifact_eval"])
def test_config_cli_loads_frozen_artifact_without_refitting(tmp_path, capsys, release):
    from tests.unit.test_ace_common_support import _artifact as ace_artifact
    from tests.unit.test_mce_text_only import _artifact as mce_artifact
    from tests.unit.test_skillopt_common_support import _artifact as skillopt_artifact

    artifact = {"recipe.ace.eval_only": ace_artifact, "recipe.skillopt.eval_only": skillopt_artifact,
                "recipe.mce.text_artifact_eval": mce_artifact}[release]()
    reference = save(tmp_path, "artifact.json", artifact.as_dict())
    config = RunConfig(release, artifacts={"artifact": reference})
    assert config.validate()["runtime_ready"] is None
    assert config.create_intervention().artifact.artifact_id == artifact.artifact_id
    output = config.run("Original question")
    assert output["state"]["query"] == "Original question"
    assert len(output["state"]["context"]) == 1
    assert output["configuration"]["artifacts"]["artifact"] == reference
    assert output["scored"] is False
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()))
    assert main(["config", "run", str(path), "--query", "New question"]) == 0
    assert json.loads(capsys.readouterr().out)["state"]["query"] == "New question"


def pab_config(tmp_path):
    reference = save(tmp_path, "plans.json", {"last_question_total": 0, "stats": [
        {"id": "case-1", "query": "Original question", "steps": ["Find", "Verify"],
         "credits": [40, 60], "level": 1}]})
    return RunConfig("plan.plan_and_budget.native_cached_soft_budget",
                     parameters={"total_word_budget": 200}, artifacts={"records": reference})


def test_cached_plan_cli_preserves_id_and_query_no_generation(tmp_path, capsys):
    config = pab_config(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config.to_dict()))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"task_id": "case-1", "query": "Original question",
                                      "context": [{"text": "visible evidence"}]}))
    assert main(["config", "run", str(config_path), "--state", str(state_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"]["task_id"] == "case-1"
    assert output["state"]["context"][0]["text"] == "visible evidence"
    assert output["scored"] is False
    with pytest.raises(ArtifactError, match="no cached"):
        config.run(FrozenAgentState(task_id="missing", query="Original question"))
    with pytest.raises(ArtifactError, match="query mismatch"):
        config.run(FrozenAgentState(task_id="case-1", query="Changed question"))


def hard_cap_config(tmp_path):
    reference = pab_config(tmp_path).artifacts["records"]
    return RunConfig("plan.plan_and_budget.hard_cap_extension",
                     parameters={"output_format": "Plain text"},
                     artifacts={"records": reference})


def test_hard_cap_file_cache_matches_existing_python_route(tmp_path):
    from agentinputlab.algorithms.decompose.plan_and_budget import (
        PlanAndBudgetCache,
        PlanAndBudgetHardCapExtension,
    )
    from agentinputlab.runner import CompositeInterventionRunner
    from tests.algorithms.unit.test_plan_and_budget_hard_cap import model

    config = hard_cap_config(tmp_path)
    assert config.validate()["runtime_ready"] is None
    ref = config.artifacts["records"]
    cache = PlanAndBudgetCache.from_native_json(ref["path"], expected_sha256=ref["sha256"])
    expected = PlanAndBudgetHardCapExtension(output_format="Plain text", records=cache)
    actual = config.create_intervention()
    backends = [model("A", "B", "Final") for _ in range(2)]
    state = FrozenAgentState(task_id="case-1", query="Original question")
    results = [CompositeInterventionRunner().run(policy, state,
               InterventionContext(backend=backend, seed=17))
               for policy, backend in zip((expected, actual), backends, strict=True)]
    assert results[0].result == results[1].result
    assert results[0].usage.output_tokens == results[1].usage.output_tokens
    assert backends[0].calls == backends[1].calls
    assert len(backends[1].calls) == 3  # No repeated plan-generation calls.


def test_hard_cap_file_cache_miss_does_not_generate(tmp_path):
    from agentinputlab.runner import CompositeInterventionRunner
    from tests.algorithms.unit.test_plan_and_budget_hard_cap import model

    policy = hard_cap_config(tmp_path).create_intervention()
    backend = model()
    with pytest.raises(ArtifactError, match="no live fallback"):
        CompositeInterventionRunner().run(policy, FrozenAgentState(task_id="missing", query="q"),
                                         InterventionContext(backend=backend))
    assert not backend.calls


def test_hard_cap_file_cache_changed_bytes_rejected(tmp_path):
    config = hard_cap_config(tmp_path)
    (tmp_path / "plans.json").write_text('{}')
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        config.create_intervention()


def test_config_relative_asset_path_is_independent_of_cwd(tmp_path, monkeypatch):
    config = pab_config(tmp_path).to_dict()
    config["artifacts"]["records"]["path"] = "plans.json"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = RunConfig.from_file(path)
    assert loaded.artifacts["records"]["path"] == str(tmp_path / "plans.json")
    loaded.run(FrozenAgentState(task_id="case-1", query="Original question"))


def test_validate_does_not_read_or_claim_ready_for_missing_asset(tmp_path):
    config = RunConfig("recipe.skillopt.eval_only",
                       artifacts={"artifact": {"path": str(tmp_path / "absent"), "sha256": "a" * 64}})
    assert config.validate()["runtime_ready"] is None
    with pytest.raises(ArtifactError):
        config.run("q")


def test_modified_artifact_fails_before_execution(tmp_path):
    config = pab_config(tmp_path)
    (tmp_path / "plans.json").write_text('{}')
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        config.run("q")


@pytest.mark.parametrize("reference", [
    {"path": "x", "sha256": "a" * 63}, {"path": "x", "sha256": "g" * 64},
    {"path": "", "sha256": "a" * 64}, {"path": "x", "sha256": "a" * 64, "factory": "os.system"},
])
def test_bad_asset_references_rejected(reference):
    with pytest.raises(ValidationError):
        RunConfig("recipe.ace.eval_only", artifacts={"artifact": reference}).validate()


def test_no_unknown_asset_factory_or_duplicate_parameter(tmp_path):
    ref = {"path": "absent", "sha256": "a" * 64}
    with pytest.raises(ValidationError, match="not supported"):
        RunConfig("baseline.no_op", artifacts={"artifact": ref}).validate()
    config = pab_config(tmp_path).to_dict()
    config["parameters"]["records"] = {}
    with pytest.raises(ValidationError, match="same field"):
        RunConfig.from_dict(config).validate()


def test_pab_native_runtime_explicit_and_relative(tmp_path):
    value = {"release_id": "plan.plan_and_budget.full", "parameters": {"output_format": "plain"},
             "runtime": {"kind": "plan_and_budget_native", "source_root": "upstream/P-and-B"}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    config = RunConfig.from_file(path)
    assert config.validate()["runtime_ready"] is None
    assert config._runtime(load=True).source_root == tmp_path / "upstream/P-and-B"
    with pytest.raises(ValidationError, match="twice"):
        config.run("q", context=InterventionContext(external_runtime=object()))
    with pytest.raises(ValidationError, match="supported only"):
        RunConfig("baseline.no_op", runtime=value["runtime"]).validate()


@pytest.mark.parametrize("raw", ['{"query":"q","query":"other"}', '{"query":"q","metadata":{"x":NaN}}',
                                  '[]', '{"query":"q","gold":"secret"}'])
def test_state_json_is_strict(tmp_path, raw):
    path = tmp_path / "state.json"
    path.write_text(raw)
    with pytest.raises(ValidationError):
        load_run_state(path)


def test_cli_state_shape_error_is_actionable(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(RunConfig("baseline.no_op").to_dict()))
    state = tmp_path / "state.json"
    state.write_text('{"query":null}')
    assert main(["config", "run", str(path), "--state", str(state)]) == 2
    assert "Traceback" not in capsys.readouterr().err
