"""Boundaries of the automatic preparation/execution entrypoint.

Fixture responses below are synthetic; real-response and cold-install receipts
are kept separately on the execution server.
"""
import json
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.workflows import (
    WorkflowConfig,
    execution,
    inspect_workflow,
    launch_workflow,
    list_recipes,
    planning,
    prepare_workflow,
    workflow_template,
)
from tests.unit.test_workflows import Adapter, benchmark

pytest_plugins = ("tests.unit.test_workflows",)


def test_missing_dataset_is_not_ready_and_no_model_is_contacted(tmp_path, provider):
    backend, calls, _ = provider
    workflow = replace(benchmark(backend), target={**benchmark(backend).target, "cache_dir": str(tmp_path / "empty")})
    report = inspect_workflow(workflow)
    assert report["can_attempt"] is False and report["ready_for_execution"] is False
    assert next(x for x in report["checks"] if x["check"] == "dataset")["status"] == "blocked"
    assert calls == [] and not (tmp_path / "empty").exists()


def test_missing_qrag_runtime_is_a_specific_input_requirement(provider):
    backend, calls, _ = provider
    workflow = workflow_template(run=RunConfig("acquire.qrag.common_support", backend=backend),
                                 benchmark_id="bfcl.v3.simple_python")
    report = inspect_workflow(workflow)
    missing = next(x for x in report["checks"] if x["check"] == "method_requirements")
    assert "qrag_common_support" in missing["error"]["message"]
    assert not report["can_attempt"] and calls == []


def test_all_profiles_have_explicit_routes_context_is_not_weight_training():
    rows = list_recipes()
    assert len(rows) == 88 and len({x["release_id"] for x in rows}) == 88
    routes = {x["release_id"]: x["route"] for x in rows}
    assert routes["recipe.ace.offline"] == routes["recipe.skillopt.offline"] == "context_preparation"
    assert routes["training.llmlingua2.distillation"] == "weight_training"
    assert routes["clarify.ask_or_assume.openhands_full"] == "unavailable"
    assert routes["acquire.r2ag.controlled_same_candidates"] == "native_python_api"
    assert routes["rewrite.infocqr.t5_student_external"] == "native_python_api"
    assert not any(x["configuration_is_acceptance_evidence"] for x in rows)


@pytest.mark.parametrize("release_id", [row["release_id"] for row in list_recipes()])
def test_each_catalog_profile_can_be_drafted_without_inventing_acceptance(release_id, tmp_path):
    run = RunConfig(release_id)
    workflow = workflow_template(run=run, benchmark_id="bfcl.v3.simple_python", cache_dir=str(tmp_path / "cache"))
    assert workflow.run.release_id == release_id
    assert workflow.run.seed == run.seed and workflow.run.budget == run.budget
    assert not (tmp_path / "cache").exists()


def test_python_only_profile_is_blocked_before_preparation_or_generation(tmp_path, provider):
    backend, calls, _ = provider
    workflow = workflow_template(release_id="acquire.r2ag.controlled_same_candidates", backend=backend)
    report = inspect_workflow(workflow)
    assert report["recipe"]["route"] == "native_python_api"
    assert report["can_attempt"] is False and report["can_prepare"] is False
    with pytest.raises(ValidationError, match="prerequisites"):
        prepare_workflow(workflow, output_dir=tmp_path / "unused", allow_network=True)
    assert not (tmp_path / "unused").exists() and calls == []


def test_template_completes_task_format_preserving_native_parameters(provider):
    backend, _calls, _ = provider
    run = RunConfig("plan.plan_and_budget.full", parameters={"temperature": 0.25}, backend=backend)
    workflow = workflow_template(run=run, benchmark_id="2wiki.multihop")
    assert "support_ids" in workflow.run.parameters["output_format"]
    assert workflow.run.parameters["temperature"] == 0.25
    assert workflow.run.seed == run.seed and workflow.run.budget == run.budget
    explicit = replace(run, parameters={"output_format": "my format"})
    assert workflow_template(run=explicit, benchmark_id="2wiki.multihop").run.parameters["output_format"] == "my format"


def test_one_launch_prepares_scores_zero_and_resumes_without_calls(tmp_path, monkeypatch, provider):
    backend, calls, answers = provider
    answers.extend(["answer", "wrong"])
    monkeypatch.setattr(planning, "load_adapter", lambda *_a, **_k: Adapter())
    monkeypatch.setattr(execution, "load_adapter", lambda *_a, **_k: Adapter())
    workflow = benchmark(backend)
    root = tmp_path / "launch"
    result = launch_workflow(workflow, output_dir=root, execute=True)
    assert result["outcomes"]["scored"] == 2 and result["preparation"]["model_calls"] == 0
    rows = [json.loads(x)["record"] for x in (root / "run/events.jsonl").read_text().splitlines()]
    assert [x["evaluation"]["metrics"]["answer_em"] for x in rows if x["status"] == "scored"] == [1.0, 0.0]
    observed = list(calls)
    assert launch_workflow(workflow, output_dir=root, execute=True, resume=True)["reused_existing_attempts"]
    assert calls == observed


def test_preparation_is_reused_but_modified_output_is_rejected(tmp_path):
    workflow = workflow_template()
    root = tmp_path / "prepared"
    first = prepare_workflow(workflow, output_dir=root)
    again = prepare_workflow(workflow, output_dir=root)
    assert again["reused_preparation"] and first["workflow_sha256"] == again["workflow_sha256"]
    (root / "workflow.json").write_text("{}")
    with pytest.raises(ValidationError):
        prepare_workflow(workflow, output_dir=root)


def test_launch_is_dry_and_existing_output_needs_explicit_resume(tmp_path):
    workflow = workflow_template()
    root = tmp_path / "launch"
    assert not launch_workflow(workflow, output_dir=root)["executed"] and not root.exists()
    launch_workflow(workflow, output_dir=root, execute=True)
    with pytest.raises(ValidationError, match="exists"):
        launch_workflow(workflow, output_dir=root, execute=True)


def test_api_failure_is_discovered_before_dispatch(tmp_path, monkeypatch):
    backend = {"model_id": "absent", "base_url": "http://127.0.0.1:1/v1",
               "deployment_identity": {"revision": "test-only"}}
    workflow = benchmark(backend)
    monkeypatch.setattr(planning, "load_adapter", lambda *_a, **_k: Adapter())
    monkeypatch.setattr(execution, "load_adapter", lambda *_a, **_k: Adapter())
    monkeypatch.setattr(RunConfig, "evaluate", lambda *_a, **_k: pytest.fail("model dispatch"))
    result = launch_workflow(workflow, output_dir=tmp_path / "launch", execute=True)
    assert result["execution_blocked"]
    assert not (tmp_path / "launch/run/events.jsonl").exists()


def test_arena_limit_is_frozen_without_running_agent(tmp_path, monkeypatch, provider):
    backend, calls, _ = provider
    source = tmp_path / "source"
    source.mkdir()
    native = tmp_path / "arena.json"
    native.write_text(json.dumps({"arena": "tooltalk", "checkout": str(source)}))
    workflow = WorkflowConfig(RunConfig("baseline.no_op", backend=backend),
                             {"kind": "arena", "arena_config": str(native), "limit": 2})
    monkeypatch.setattr(planning, "native_arena_readiness", lambda _a: {"fixture": True})
    def inventory(_run, _native, *, output_dir, inventory_only):
        assert inventory_only is True
        return {"available_case_ids": ["a", "b", "c"]}
    monkeypatch.setattr(RunConfig, "evaluate_arena", inventory)
    result = prepare_workflow(workflow, output_dir=tmp_path / "prepared")
    ready = WorkflowConfig.from_file(result["workflow_path"])
    assert ready.target["case_ids"] == ("a", "b") and "limit" not in ready.target
    assert calls == []


def test_cli_models_and_blocked_dry_run_have_useful_exit_codes(tmp_path, capsys):
    assert main(["workflow", "init", "--release-id", "plan.plan_and_budget.full",
                 "--benchmark", "2wiki.multihop", "--model-id", "q4", "--base-url", "http://127.0.0.1:1/v1",
                 "--deployment-revision", "pinned"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["run"]["backend"]["model_id"] == "q4" and value["run"]["parameters"]["output_format"]
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(value))
    assert main(["workflow", "run", str(path)]) == 2


def test_nested_source_paths_are_relative_to_config_not_cwd(tmp_path):
    value = workflow_template().to_dict()
    value["target"] = {"kind": "benchmark", "benchmark_id": "longbench.qasper", "split": "test", "limit": 1,
                       "longbench_sources": {"data_zip": "inputs/data.zip", "scorer_path": "source/metrics.py"}}
    workflow = WorkflowConfig.from_dict(value, base_dir=tmp_path)
    assert workflow.target["longbench_sources"]["data_zip"] == str(tmp_path / "inputs/data.zip")


def test_unknown_preparation_cannot_execute_arbitrary_commands():
    with pytest.raises(ValidationError, match="unknown method preparation"):
        replace(workflow_template(), preparation={"method": {"kind": "shell", "options": {"command": "anything"}}})


def test_native_arena_template_and_simulator_binding_are_explicit(tmp_path, provider):
    backend, calls, _ = provider
    tooltalk = workflow_template(benchmark_id="tooltalk.full", backend=backend, cache_dir=str(tmp_path))
    assert tooltalk.target["kind"] == "arena" and tooltalk.target["arena"]["arena"] == "tooltalk"
    state = workflow_template(benchmark_id="state_bench.common_support", backend=backend,
                              environment_backend=backend, cache_dir=str(tmp_path))
    assert state.target["arena"]["environment_backend"] == backend
    assert calls == [] and not (tmp_path / "runtime-sources").exists()


def test_worker_template_derives_preparation_but_does_not_claim_it_is_ready(tmp_path, provider):
    backend, calls, _ = provider
    workflow = workflow_template(release_id="rewrite.reformer.repo_native_r_only", benchmark_id="2wiki.multihop",
                                 backend=backend, cache_dir=str(tmp_path))
    assert workflow.run.runtime["kind"] == "reformer_worker" and workflow.preparation["sources"]
    assert workflow.run.retriever["kind"] == "visible_context_bm25"
    report = inspect_workflow(workflow)
    assert not report["can_attempt"] and calls == []
    assert any(x["operation"] == "method" and x["kind"] == "worker" for x in report["preparation_plan"])


def test_supplied_asset_reused_and_corruption_preserved(tmp_path):
    import hashlib

    from agentinputlab.workflows.preparation import prepare_assets
    path = tmp_path / "weights"
    path.write_bytes(b"fixture")
    asset = {"manifest": {"artifact_id": "fixture", "kind": "model", "revision": "v1",
                          "sha256": hashlib.sha256(b"fixture").hexdigest(), "size_bytes": 7,
                          "acquisition": "user_provided", "license_id": "MIT"}, "destination": str(path)}
    workflow = replace(workflow_template(), preparation={"assets": [asset]})
    assert prepare_assets(workflow, tmp_path / "output", allow_network=False)["downloaded_bytes"] == 0
    path.write_bytes(b"changed")
    with pytest.raises(ValidationError, match="differs"):
        prepare_assets(workflow, tmp_path / "output", allow_network=True)
    assert path.read_bytes() == b"changed"


def test_method_download_total_is_checked_before_network(tmp_path, monkeypatch):
    from agentinputlab.data import PinnedHTTPSDownloadTransport
    from agentinputlab.workflows.preparation import prepare_assets
    asset = {"manifest": {"artifact_id": "fixture", "kind": "model", "revision": "v1",
                          "sha256": "a" * 64, "size_bytes": 100,
                          "acquisition": "download", "source_uri": "https://example.org/fixture", "license_id": "MIT"},
             "destination": str(tmp_path / "missing")}
    workflow = replace(workflow_template(), preparation={"assets": [asset], "max_download_bytes": 50})
    monkeypatch.setattr(PinnedHTTPSDownloadTransport, "stream", lambda *_a, **_k: pytest.fail("download"))
    with pytest.raises(ValidationError, match="exceed"):
        prepare_assets(workflow, tmp_path / "output", allow_network=True)
    assert not (tmp_path / "missing").exists()


def test_bound_worker_is_reused_across_output_directories_without_install(tmp_path, monkeypatch):
    import sys

    from agentinputlab.workflows import preparation
    from agentinputlab.workflows.configuration import file_sha
    run = RunConfig("rewrite.infocqr.common_support_rw_zsl", runtime={
        "kind": "infocqr_worker", "python_executable": sys.executable,
        "python_executable_sha256": file_sha(sys.executable),
        "driver_path": str(tmp_path / "driver.py"), "driver_sha256": "a" * 64,
        "compatibility_manifest_path": str(tmp_path / "manifest.json")})
    workflow = replace(workflow_template(), run=run)
    monkeypatch.setattr(preparation, "interpreter_readiness", lambda *_a: {"ready": True})
    result = preparation.prepare_worker_environment(workflow, run, tmp_path / "other-output",
        python=sys.executable, environment=None, allow_network=False)
    assert result == run and not (tmp_path / "other-output").exists()


def test_infocqr_semantic_identities_are_derived_from_the_selected_benchmark(provider):
    backend, calls, _ = provider
    workflow = workflow_template(release_id="rewrite.infocqr.common_support_rw_zsl", backend=backend,
        parameters={"source_license_acknowledged": True}, benchmark_id="bfcl.v3.simple_python")
    method = workflow.run.create_intervention()
    assert method.dataset_id == "bfcl.v3.simple_python" and method.dataset_revision
    assert method.split == "test" and method.retrieval_corpus_id == "bfcl.v3.simple_python:visible-context"
    assert workflow.run.retriever["kind"] == "visible_context_bm25" and calls == []
    assert method.controls.max_tokens == 2 * (16384 + 512)
    assert workflow.run.budget.max_output_tokens == 4096


def test_explicit_worker_token_guard_is_preserved_and_conflict_is_reported(provider):
    from agentinputlab.workflows.readiness import budget_and_input_readiness
    backend, calls, _ = provider
    workflow = workflow_template(release_id="rewrite.infocqr.common_support_rw_zsl", backend=backend,
        parameters={"source_license_acknowledged": True, "controls": {"max_tokens": 100}},
        benchmark_id="bfcl.v3.simple_python")
    assert workflow.run.parameters["controls"]["max_tokens"] == 100
    with pytest.raises(ValidationError, match="input\\+output"):
        budget_and_input_readiness(workflow, None, [])
    assert workflow.run.budget.max_output_tokens == 4096 and calls == []


def test_invalid_native_semantic_values_are_reported_before_a_model_call(tmp_path, provider):
    backend, calls, _ = provider
    workflow = WorkflowConfig(RunConfig("rewrite.infocqr.common_support_rw_zsl",
        parameters={"source_license_acknowledged": True}, backend=backend),
        {"kind": "benchmark", "benchmark_id": "bfcl.v3.simple_python", "split": "test", "limit": 1})
    report = inspect_workflow(workflow)
    check = next(row for row in report["checks"] if row["check"] == "native_parameters")
    assert check["status"] == "blocked" and "dataset revision" in check["error"]["message"]
    assert report["disposition"] == "missing_user_input" and calls == []


def test_missing_cpu_execution_extra_needs_network_before_creating_environment(tmp_path, monkeypatch, provider):
    from agentinputlab.workflows import environment
    backend, calls, _ = provider
    workflow = workflow_template(release_id="plan.plan_and_budget.full", backend=backend,
                                 benchmark_id="2wiki.multihop", cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ValidationError, match="missing execution dependencies"):
        environment.execution_python(workflow, allow_network=False)
    assert not (tmp_path / "cache").exists() and calls == []


def test_warm_execution_dependencies_do_not_create_an_extra_environment(tmp_path, monkeypatch):
    from agentinputlab.workflows import environment
    workflow = workflow_template(benchmark_id="bfcl.v3.simple_python", cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: object())
    assert environment.execution_python(workflow, allow_network=False) is None
    assert not (tmp_path / "cache").exists()


def test_real_delegation_loads_the_workers_namespace_and_preserves_resume(tmp_path, monkeypatch):
    import sys

    from agentinputlab.workflows.environment import delegate
    monkeypatch.setenv("AGENTINPUTLAB_CACHE", str(tmp_path / "cache"))
    workflow = workflow_template(query="No-model delegated input patch")
    root = tmp_path / "delegated"
    result = delegate(workflow, sys.executable, output_dir=root, allow_network=False, resume=False)
    assert result["all_attempts_terminal"] and result["outcomes"]["succeeded"] == 1
    assert delegate(workflow, sys.executable, output_dir=root, allow_network=False, resume=True)["reused_existing_attempts"]


def test_isolated_worker_namespace_exposes_bundled_resources():
    import subprocess
    import sys

    from agentinputlab.workflows.environment import _package_bootstrap
    code = _package_bootstrap() + "from importlib.resources import files; " \
        "assert files('workers').joinpath('reformer/compatibility_manifest.json').is_file()"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)
