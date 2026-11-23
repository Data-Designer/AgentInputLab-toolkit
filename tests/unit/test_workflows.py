"""Workflow glue tests. All HTTP/model answers here are synthetic fixtures."""

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentinputlab import RunConfig
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.workflows import (
    WorkflowConfig,
    execution,
    inspect_workflow,
    planning,
    prepare_workflow,
    run_workflow,
    workflow_status,
    workflow_template,
)
from agentinputlab.workflows.configuration import file_sha


@pytest.fixture
def provider(monkeypatch):
    # Keep loopback fixture traffic out of the developer's external HTTP proxy.
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    calls = []
    answers = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            calls.append(("GET", self.path))
            raw = json.dumps({"data": [{"id": "fixture-model"}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", body))
            answer = answers.pop(0)
            raw = json.dumps({"model": "fixture-model", "choices": [{"message": {"content": answer},
                              "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = {"model_id": "fixture-model", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
               "deployment_identity": {"revision": "synthetic-fixture"}}
    try:
        yield backend, calls, answers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class Adapter:
    benchmark_id = "2wiki.multihop"
    pairing_mode = "stateless_score_twice"

    def examples(self, split):
        assert split == "dev"
        return tuple(BenchmarkExample(example_id=x, query="Fixture question " + x,
                                      evaluator_gold={"hidden": "DO_NOT_LEAK_GOLD"}) for x in ("a", "b"))

    def score(self, _example, prediction):
        return {"answer_em": float(prediction.answer == "answer")}


def benchmark(backend):
    return WorkflowConfig(RunConfig("baseline.no_op", backend=backend), {
        "kind": "benchmark", "benchmark_id": "2wiki.multihop", "split": "dev", "limit": 2,
        "predictor_parameters": {"schema_version": 1, "qa": {"output_format": "answer_only"}},
    })


def test_prompt_workflow_is_dry_by_default(tmp_path):
    config = workflow_template(query="Find a flight")
    report = run_workflow(config, output_dir=tmp_path / "unused")
    assert report["executed"] is False and report["inspection"]["can_attempt"]
    assert not (tmp_path / "unused").exists()
    assert report["inspection"]["model_calls"] == 0


def test_prompt_execution_and_resume_do_not_repeat(tmp_path, monkeypatch):
    workflow = workflow_template(query="Find a flight")
    root = tmp_path / "run"
    first = run_workflow(workflow, output_dir=root, execute=True)
    assert first["status"] == "completed" and first["all_attempts_terminal"]
    assert first["outcomes"]["succeeded"] == 1
    journal = (root / "events.jsonl").read_bytes()
    monkeypatch.setattr(RunConfig, "run", lambda *_args, **_kwargs: pytest.fail("repeat"))
    resumed = run_workflow(workflow, output_dir=root, execute=True, resume=True)
    assert resumed["executed"] is False and resumed["reused_existing_attempts"]
    assert (root / "events.jsonl").read_bytes() == journal


def test_pipeline_reuses_existing_scorer_and_keeps_zero_score(tmp_path, monkeypatch, provider):
    backend, calls, answers = provider
    answers.extend(["answer", "wrong"])
    adapter = Adapter()
    monkeypatch.setattr(planning, "load_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(execution, "load_adapter", lambda *_args, **_kwargs: adapter)
    config = benchmark(backend)
    prepared = prepare_workflow(config, output_dir=tmp_path / "prepared")
    assert prepared["model_calls"] == 0 and calls == []
    ready = WorkflowConfig.from_file(prepared["workflow_path"])
    assert ready.target["case_ids"] == ("a", "b") and set(ready.case_state_hashes) == {"a", "b"}
    root = tmp_path / "run"
    result = run_workflow(ready, output_dir=root, execute=True)
    assert result["status"] == "completed" and result["outcomes"]["scored"] == 2
    records = [json.loads(line)["record"] for line in (root / "events.jsonl").read_text().splitlines()]
    scored = [row for row in records if row["status"] == "scored"]
    assert [row["evaluation"]["metrics"]["answer_em"] for row in scored] == [1.0, 0.0]
    assert [row["usage"]["model_calls"] for row in scored] == [1, 1]
    posts = [body for verb, body in calls if verb == "POST"]
    assert len(posts) == 2 and all("DO_NOT_LEAK_GOLD" not in json.dumps(body) for body in posts)
    result = run_workflow(ready, output_dir=root, execute=True, resume=True)
    assert result["reused_existing_attempts"] and len([x for x in calls if x[0] == "POST"]) == 2


def test_interrupted_query_is_not_repeated_other_unstarted_query_continues(tmp_path, monkeypatch):
    workflow = WorkflowConfig(RunConfig("plan.plan_and_solve.ps_prompt"), {"kind": "query", "queries": ["one", "two"]})
    root = tmp_path / "run"
    original = RunConfig.run
    seen = []

    def interrupt(_config, query, **_kwargs):
        seen.append(query)
        raise KeyboardInterrupt

    monkeypatch.setattr(RunConfig, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_workflow(workflow, output_dir=root, execute=True)
    status = workflow_status(root)
    assert status["uncertain_case_ids"] == ["query-000000"]
    assert status["unstarted_case_ids"] == ["query-000001"]

    def complete(config, query, **kwargs):
        seen.append(query)
        return original(config, query, **kwargs)

    monkeypatch.setattr(RunConfig, "run", complete)
    resumed = run_workflow(workflow, output_dir=root, execute=True, resume=True)
    assert seen == ["one", "two"]
    assert resumed["status"] == "needs_reconciliation" and not resumed["all_attempts_terminal"]


def test_failure_is_terminal_and_secret_is_not_journaled(tmp_path, monkeypatch):
    workflow = workflow_template()
    monkeypatch.setenv("OPENAI_API_KEY", "opaque-sensitive-fixture-value")
    def fail(*_args, **_kwargs):
        raise RuntimeError("opaque-sensitive-fixture-value")
    monkeypatch.setattr(RunConfig, "run", fail)
    root = tmp_path / "failed"
    result = run_workflow(workflow, output_dir=root, execute=True)
    assert result["status"] == "completed_with_failures" and result["all_attempts_terminal"]
    assert "opaque-sensitive-fixture-value" not in (root / "events.jsonl").read_text()
    again = run_workflow(workflow, output_dir=root, execute=True, resume=True)
    assert again["reused_existing_attempts"]


def test_probe_only_uses_model_list_and_is_opt_in(provider):
    backend, calls, _answers = provider
    workflow = replace(workflow_template(), run=RunConfig("plan.plan_and_solve.ps_prompt", backend=backend))
    assert inspect_workflow(workflow)["can_attempt"] and calls == []
    report = inspect_workflow(workflow, probe_endpoint=True)
    assert report["can_attempt"] and calls == [("GET", "/v1/models")], report
    assert report["model_calls"] == 0 and report["real_acceptance_passed"] is None


@pytest.mark.parametrize("version", [True, 0, 2, "1", None])
def test_schema_version_is_exact(version):
    value = workflow_template().to_dict()
    value["schema_version"] = version
    with pytest.raises(ValidationError):
        WorkflowConfig.from_dict(value)


@pytest.mark.parametrize("target", [{"kind": []}, {"kind": "query", "queries": []},
    {"kind": "benchmark", "benchmark_id": "2wiki.multihop", "split": "dev", "limit": True},
    {"kind": "benchmark", "benchmark_id": "2wiki.multihop", "split": "dev", "limit": 2001},
    {"kind": "benchmark", "benchmark_id": "2wiki.multihop", "split": "dev", "case_ids": ["a", "a"]}])
def test_invalid_target_cannot_dispatch(target):
    with pytest.raises(ValidationError):
        WorkflowConfig(RunConfig("baseline.no_op"), target)


def test_missing_native_inputs_reported_before_download(tmp_path, monkeypatch):
    workflow = workflow_template(release_id="rewrite.rrr.common_support_prompt", benchmark_id="2wiki.multihop")
    report = inspect_workflow(workflow)
    assert not report["can_attempt"]
    monkeypatch.setattr(planning, "load_adapter", lambda *_args, **_kwargs: pytest.fail("download before config gate"))
    with pytest.raises(ValidationError):
        prepare_workflow(workflow, output_dir=tmp_path / "unused", allow_network=True)
    assert not (tmp_path / "unused").exists()


def test_changed_binding_or_truncated_journal_blocks_resume(tmp_path):
    source = tmp_path / "binding.json"
    source.write_text("{}")
    workflow = replace(workflow_template(), input_bindings={str(source): file_sha(source)})
    root = tmp_path / "run"
    run_workflow(workflow, output_dir=root, execute=True)
    source.write_text("[]")
    assert not inspect_workflow(workflow)["can_attempt"]
    assert run_workflow(workflow, output_dir=root, execute=True, resume=True)["execution_blocked"]
    source.write_text("{}")
    with (root / "events.jsonl").open("ab") as stream:
        stream.write(b'{"unfinished":')
    with pytest.raises(ValidationError, match="truncated"):
        run_workflow(workflow, output_dir=root, execute=True, resume=True)


def test_cli_uses_same_python_path_without_execution(tmp_path, capsys):
    assert main(["workflow", "init", "--query", "Find a flight"]) == 0
    value = json.loads(capsys.readouterr().out)
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(value))
    assert main(["workflow", "inspect", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
    assert main(["workflow", "run", str(path)]) == 0
    assert not json.loads(capsys.readouterr().out)["executed"]


def test_native_preflight_interruption_cannot_repeat_first_case(tmp_path, monkeypatch, provider):
    backend, calls, answers = provider
    workflow = benchmark(backend)
    adapter = Adapter()
    monkeypatch.setattr(execution, "load_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(planning, "load_adapter", lambda *_args, **_kwargs: adapter)
    original = RunConfig.evaluate
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(RunConfig, "evaluate", interrupt)
    root = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_workflow(workflow, output_dir=root, execute=True)
    assert workflow_status(root)["uncertain_case_ids"] == ["a"]
    assert not [x for x in calls if x[0] == "POST"]
    monkeypatch.setattr(RunConfig, "evaluate", original)
    answers.append("answer")
    result = run_workflow(workflow, output_dir=root, execute=True, resume=True)
    assert result["status"] == "needs_reconciliation"
    assert result["outcomes"]["scored"] == 1 and len([x for x in calls if x[0] == "POST"]) == 1
    rows = [json.loads(x)["record"] for x in (root / "events.jsonl").read_text().splitlines()]
    assert [x["workflow_case_id"] for x in rows if x["status"] == "started"] == ["a", "b"]


def test_relative_assets_follow_workflow_file_not_process_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    value = workflow_template().to_dict()
    value["target"] = {"kind": "arena", "arena_config": "arena.json", "case_ids": ["a"]}
    value["run"]["runtime"] = {"kind": "fixture", "checkout": "source", "python_executable": "venv/bin/python"}
    path = config_dir / "workflow.json"
    path.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path)
    workflow = WorkflowConfig.from_file(path)
    assert workflow.target["arena_config"] == str(config_dir / "arena.json")
    assert workflow.run.runtime["checkout"] == str(config_dir / "source")
    assert workflow.run.runtime["python_executable"] == str(config_dir / "venv/bin/python")


def test_changed_selection_and_existing_output_are_not_overwritten(tmp_path):
    workflow = workflow_template()
    root = tmp_path / "run"
    run_workflow(workflow, output_dir=root, execute=True)
    before = (root / "events.jsonl").read_bytes()
    with pytest.raises(ValidationError, match="exists"):
        run_workflow(workflow, output_dir=root, execute=True)
    with pytest.raises(ValidationError, match="changed"):
        run_workflow(workflow_template(query="different"), output_dir=root, execute=True, resume=True)
    assert (root / "events.jsonl").read_bytes() == before


def test_missing_entrypoint_and_training_recipe_are_not_promoted():
    missing = workflow_template(release_id="clarify.ask_or_assume.openhands_full")
    training = workflow_template(release_id="recipe.skillopt.offline")
    assert not inspect_workflow(missing)["can_attempt"]
    assert not inspect_workflow(training)["can_attempt"]


def test_concurrency_remains_with_existing_runner():
    config = replace(workflow_template(), max_workers=4)
    report = inspect_workflow(config)
    assert not report["can_attempt"]
    assert next(x for x in report["checks"] if x["check"] == "scheduling")["status"] == "blocked"


def test_missing_loader_blocks_asset_acquisition(tmp_path, monkeypatch, provider):
    backend, _calls, _answers = provider
    workflow = WorkflowConfig(RunConfig("plan.plan_and_solve.ps_prompt", backend=backend), {
        "kind": "benchmark", "benchmark_id": "2wiki.multihop", "split": "dev", "limit": 1,
    })
    metadata = planning.describe_benchmark("2wiki.multihop")
    monkeypatch.setattr(planning, "describe_benchmark", lambda _name: {**metadata, "automatic_loader": False})
    monkeypatch.setattr(planning, "load_adapter", lambda *_args, **_kwargs: pytest.fail("unnecessary acquisition"))
    with pytest.raises(ValidationError, match="prerequisites"):
        prepare_workflow(workflow, output_dir=tmp_path / "unused", allow_network=True)
    assert not (tmp_path / "unused").exists()


def test_coarse_family_tag_does_not_veto_native_configured_transfer(provider):
    backend, calls, _answers = provider
    workflow = WorkflowConfig(RunConfig("acquire.qrag.common_support", backend=backend), {
        "kind": "benchmark", "benchmark_id": "bfcl.v3.simple_python", "split": "test", "limit": 1,
    })
    report = inspect_workflow(workflow)
    assert not report["can_attempt"]  # The actual runtime/data prerequisites are still missing.
    details = next(x["details"] for x in report["checks"] if x["check"] == "benchmark")
    assert details["catalog_family_match"] is False
    assert details["native_compatibility_verified"] is False and calls == []


def test_active_guard_prevents_dispatch(tmp_path, monkeypatch):
    workflow = WorkflowConfig(RunConfig("plan.plan_and_solve.ps_prompt"), {"kind": "query", "queries": ["one", "two"]})
    original = RunConfig.run
    def interrupt(config, query, **kwargs):
        if query == "two":
            raise KeyboardInterrupt
        return original(config, query, **kwargs)
    monkeypatch.setattr(RunConfig, "run", interrupt)
    root = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_workflow(workflow, output_dir=root, execute=True)
    # Simulate a hard-kill guard on a run with never-started work remaining.
    fresh = WorkflowConfig(workflow.run, {"kind": "query", "queries": ["one", "two", "three"]})
    second = tmp_path / "second"
    with pytest.raises(KeyboardInterrupt):
        run_workflow(fresh, output_dir=second, execute=True)
    (second / ".running").write_text("test-fixture-owned-by-another-invocation")
    monkeypatch.setattr(RunConfig, "run", lambda *_args, **_kwargs: pytest.fail("duplicate dispatch"))
    with pytest.raises(ValidationError, match="guard"):
        run_workflow(fresh, output_dir=second, execute=True, resume=True)


def test_arena_delegation_keeps_clarification_and_receipt(tmp_path, monkeypatch, provider):
    backend, calls, _answers = provider
    checkout = tmp_path / "native-source"
    checkout.mkdir()
    native = tmp_path / "arena.json"
    native.write_text(json.dumps({"arena": "tooltalk", "checkout": str(checkout)}))
    workflow = WorkflowConfig(RunConfig("baseline.no_op", backend=backend), {
        "kind": "arena", "arena_config": str(native), "case_ids": ["native-case"],
    })
    seen = []
    def native_execute(_run, config_path, *, case_ids, output_dir):
        seen.append((config_path, case_ids, str(output_dir)))
        return {"cases": [{"case_id": case_ids[0], "status": "awaiting_clarification",
                           "receipt": "fixture-native-terminal.json"}]}
    monkeypatch.setattr(RunConfig, "evaluate_arena", native_execute)
    monkeypatch.setattr(planning, "native_arena_readiness", lambda _native: {"fixture": True})
    root = tmp_path / "run"
    result = run_workflow(workflow, output_dir=root, execute=True)
    assert result["status"] == "awaiting_user" and result["all_attempts_terminal"]
    assert not result["all_succeeded"] and not [x for x in calls if x[0] == "POST"] and len(seen) == 1
    journal = (root / "events.jsonl").read_text()
    assert "fixture-native-terminal.json" in journal
    again = run_workflow(workflow, output_dir=root, execute=True, resume=True)
    assert again["reused_existing_attempts"] and len(seen) == 1
