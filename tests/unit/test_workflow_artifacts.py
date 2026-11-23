"""Synthetic artifact catalogs; these tests are not real-model acceptance."""
import json
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.workflows import (
    WorkflowConfig,
    inspect_workflow,
    launch_workflow,
    prepare_workflow,
)
from agentinputlab.workflows.artifacts import binding_identity, discover_artifacts
from agentinputlab.workflows.configuration import file_sha
from tests.unit.test_skillopt_common_support import _artifact
from tests.unit.test_workflows import Adapter

pytest_plugins = ("tests.unit.test_workflows",)


def catalog_workflow(tmp_path, backend):
    artifact = tmp_path / "skill.json"
    artifact.write_text(json.dumps(_artifact().as_dict()))
    workflow = WorkflowConfig(RunConfig("recipe.skillopt.eval_only", backend=backend),
        {"kind": "benchmark", "benchmark_id": "2wiki.multihop", "split": "dev", "case_ids": ["a"],
         "predictor_parameters": {"schema_version": 1, "qa": {"output_format": "answer_only"}}})
    payload = {"schema_version": 1, "bindings": [{"identity": binding_identity(workflow),
        "artifacts": {"artifact": {"path": "skill.json", "sha256": file_sha(artifact)}}}]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload))
    return replace(workflow, preparation={"artifact_catalog": str(path)}), path, artifact


def test_artifact_catalog_exact_model_discovery_and_real_executor(tmp_path, provider, monkeypatch):
    from agentinputlab.workflows import execution, planning
    backend, calls, answers = provider
    workflow, catalog, artifact = catalog_workflow(tmp_path, backend)
    monkeypatch.setattr(planning, "load_adapter", lambda *_a, **_k: Adapter())
    monkeypatch.setattr(execution, "load_adapter", lambda *_a, **_k: Adapter())
    report = inspect_workflow(workflow)
    assert report["can_attempt"] and report["workflow_id"] == workflow.identity and calls == [], report
    prepared = prepare_workflow(workflow, output_dir=tmp_path / "prepared")
    ready = WorkflowConfig.from_file(prepared["workflow_path"])
    assert ready.run.artifacts["artifact"]["path"] == str(artifact)
    assert ready.input_bindings[str(catalog)] == file_sha(catalog)
    answers.append("answer")
    result = launch_workflow(workflow, execute=True, output_dir=tmp_path / "launch")
    assert result["outcomes"]["scored"] == 1
    count = len(calls)
    assert launch_workflow(workflow, execute=True, resume=True, output_dir=tmp_path / "launch")["reused_existing_attempts"]
    assert len(calls) == count


def test_nested_model_identity_is_json_safe_without_changing_binding(tmp_path, provider, monkeypatch):
    from agentinputlab.contracts.serialization import content_hash
    from agentinputlab.workflows import planning
    backend, calls, _ = provider
    identity = {"revision": "fixture", "model_json_hashes": {"config.json": "a" * 64},
                "shards": [{"name": "weights", "revision": "frozen"}]}
    workflow, _, _ = catalog_workflow(tmp_path, {**backend, "deployment_identity": identity})
    monkeypatch.setattr(planning, "load_adapter", lambda *_a, **_k: Adapter())
    binding = binding_identity(workflow)
    assert json.loads(json.dumps(binding))["deployment_identity"] == identity
    assert content_hash(binding["deployment_identity"]) == content_hash(identity)
    report = json.loads(json.dumps(inspect_workflow(workflow)))
    assert report["can_attempt"] and calls == []
    details = next(c["details"] for c in report["checks"] if c["check"] == "artifact_discovery")
    assert details["identity"]["deployment_identity"] == identity
    changed = replace(workflow, run=replace(workflow.run,
        backend={**workflow.run.to_dict()["backend"], "deployment_identity": {
            **identity, "model_json_hashes": {"config.json": "b" * 64}}}))
    assert not inspect_workflow(changed)["can_attempt"]


@pytest.mark.parametrize("change", ["model", "revision", "benchmark", "duplicate", "missing_file"])
def test_catalog_does_not_guess_or_cross_models(tmp_path, provider, change):
    backend, calls, _ = provider
    workflow, path, _artifact_path = catalog_workflow(tmp_path, backend)
    payload = json.loads(path.read_text())
    identity = payload["bindings"][0]["identity"]
    if change == "model":
        identity["model_id"] = "another-model"
    elif change == "revision":
        identity["deployment_identity"] = {"revision": "another-revision"}
    elif change == "benchmark":
        identity["benchmark_id"] = "hotpotqa.distractor"
    elif change == "duplicate":
        payload["bindings"].append(payload["bindings"][0])
    else:
        path.unlink()
    if change != "missing_file":
        path.write_text(json.dumps(payload))
    report = inspect_workflow(workflow)
    assert not report["can_prepare"] and not report["can_attempt"]
    assert report["disposition"] == "missing_user_input"
    assert calls == []


def test_explicit_artifacts_win_and_malformed_contents_block_before_model(tmp_path, provider):
    backend, calls, _ = provider
    workflow, catalog, artifact = catalog_workflow(tmp_path, backend)
    resolved, _ = discover_artifacts(workflow)
    catalog.unlink()
    assert discover_artifacts(resolved)[0] == resolved
    artifact.write_text("{}")
    malformed = replace(resolved, run=replace(resolved.run, artifacts={"artifact": {
        "path": str(artifact), "sha256": file_sha(artifact)}}), preparation={})
    report = inspect_workflow(malformed)
    assert not report["can_attempt"]
    assert next(x for x in report["checks"] if x["check"] == "native_parameters")["status"] == "blocked"
    assert calls == []


def test_missing_private_artifact_is_user_input_not_automatic_training(tmp_path, provider):
    backend, calls, _ = provider
    workflow, _catalog, artifact = catalog_workflow(tmp_path, backend)
    resolved, _ = discover_artifacts(workflow)
    artifact.unlink()
    report = inspect_workflow(resolved)
    assert not report["can_prepare"] and report["disposition"] == "missing_user_input"
    assert calls == []
    with pytest.raises(ValidationError):
        prepare_workflow(resolved, output_dir=tmp_path / "unused", allow_network=True)
    assert not (tmp_path / "unused").exists()


def test_ir_scoring_dependency_is_an_automatic_execution_extra(tmp_path, monkeypatch):
    from agentinputlab.workflows import environment, workflow_template
    workflow = workflow_template(benchmark_id="beir.arguana", cache_dir=str(tmp_path / "cache"))
    assert environment.requirements(workflow) == (["evaluation"], ["ir_measures"])
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ValidationError, match="ir_measures"):
        environment.execution_python(workflow, allow_network=False)
    assert not (tmp_path / "cache").exists()


def test_native_descendants_use_the_same_code_without_caller_binary_path(tmp_path, monkeypatch):
    import subprocess
    import sys

    from agentinputlab import __file__ as package_file
    from agentinputlab.workflows import environment, workflow_template
    monkeypatch.setenv("AGENTINPUTLAB_CACHE", str(tmp_path / "cache"))
    workflow = workflow_template()
    env = environment._child_environment(workflow)
    code = "import agentinputlab; from importlib.resources import files; " \
        "assert files('workers').joinpath('reformer/compatibility_manifest.json').is_file(); print(agentinputlab.__file__)"
    child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert child.stdout.strip() == package_file
    assert "site-packages" not in env["PYTHONPATH"]
    bootstrap = __import__("pathlib").Path(env["PYTHONPATH"]) / "sitecustomize.py"
    bootstrap.write_text("changed")
    with pytest.raises(ValidationError, match="bootstrap changed"):
        environment._child_environment(workflow)


def test_native_source_checkout_is_checked_like_the_actual_executor(tmp_path):
    import sys

    from agentinputlab.workflows.readiness import interpreter_readiness
    source = tmp_path / "checkout"
    module = source / "caller_pinned_arena"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("raise AssertionError('availability must not import source')")
    result = interpreter_readiness(sys.executable, ("caller_pinned_arena",), source_paths=(source,))
    assert result["missing"] == [] and result["model_calls"] == 0


def test_cli_reuses_predictor_sources_and_preserves_budget(tmp_path, capsys):
    from agentinputlab.cli.main import main
    folder = tmp_path / "inputs"
    folder.mkdir()
    sources = folder / "longbench.json"
    sources.write_text(json.dumps({"data_zip": "data.zip", "scorer_path": "metrics.py", "prompts_path": "prompts.json"}))
    predictor = folder / "predictor.json"
    predictor.write_text(json.dumps({"schema_version": 1, "native_text": {"max_output_tokens": 256}}))
    budget = folder / "budget.json"
    budget.write_text(json.dumps({"max_model_calls": 4, "max_output_tokens": 2048}))
    assert main(["workflow", "init", "--benchmark", "longbench.govreport", "--longbench-config", str(sources),
                 "--predictor-config", str(predictor), "--budget-config", str(budget)]) == 0
    workflow = WorkflowConfig.from_dict(json.loads(capsys.readouterr().out))
    assert workflow.target["longbench_sources"]["data_zip"] == str(folder / "data.zip")
    assert workflow.target["predictor_parameters"]["native_text"]["max_output_tokens"] == 256
    assert workflow.run.budget.max_model_calls == 4 and workflow.run.budget.max_output_tokens == 2048


def test_invalid_predictor_or_missing_ir_corpus_blocks_before_acquisition(tmp_path, provider):
    from agentinputlab.workflows import workflow_template
    backend, calls, _ = provider
    workflow = workflow_template(benchmark_id="beir.arguana", backend=backend, cache_dir=str(tmp_path / "empty"))
    report = inspect_workflow(workflow)
    assert not report["can_prepare"] and report["disposition"] == "missing_user_input"
    assert "corpus retriever" in next(x for x in report["checks"] if x["check"] == "predictor_inputs")["error"]["message"]
    assert calls == [] and not (tmp_path / "empty").exists()
