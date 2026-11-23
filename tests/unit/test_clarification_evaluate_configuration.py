"""Static scoring must not silently bypass an explicit clarification request."""
import json

from agentinputlab import InterventionContext, RunConfig
from agentinputlab.cli.main import main
from tests.unit.test_benchmark_configuration import Adapter, Backend

ASK = ("<think>Need a choice</think><u_request>0.9</u_request>"
       "<u_request_explanation>Ambiguous destination</u_request_explanation>"
       "<action>request_clarification: Which city?</action>"
       "<confidence>0.9</confidence><explanation>Ask the user</explanation>")
ACT = ASK.replace("request_clarification: Which city?", "answer")


def test_clarification_preserves_response_and_budget_without_reader_or_score():
    backend = Backend([(ASK, 31), ("answer", 8)])
    adapter = Adapter(ids=("a",))
    config = RunConfig("clarify.udcs.repo_native_prompt_route", parameters={"max_tokens": 512})
    result = config.evaluate("2wiki.multihop", split="dev", example_ids=["a"], adapter=adapter,
        context=InterventionContext(backend=backend),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}})
    row, = result["records"]
    assert row["status"] == "awaiting_clarification"
    assert row["clarification"]["questions"] == ["Which city?"]
    assert row["scored"] is False and "evaluation" not in row and "failure" not in row
    assert row["usage"]["model_calls"] == 1 and row["usage"]["output_tokens"] == 31
    assert len(backend.requests) == 1 and adapter.scored == []
    assert row["model_responses"][0]["response"]["text"] == ASK
    assert result["selected_count"] == 1 and result["scored_count"] == result["failed_count"] == 0
    assert result["awaiting_clarification_count"] == 1
    assert result["metrics_over_scored_cases"] == {}


def test_nonclarifying_route_keeps_original_reader_scoring():
    backend = Backend([(ACT, 31), ("answer", 8)])
    adapter = Adapter(ids=("a",))
    result = RunConfig("clarify.udcs.repo_native_prompt_route").evaluate(
        "2wiki.multihop", split="dev", example_ids=["a"], adapter=adapter,
        context=InterventionContext(backend=backend),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}})
    assert result["scored_count"] == 1 and result["awaiting_clarification_count"] == 0
    assert len(backend.requests) == 2 and adapter.scored == ["a"]


def test_pending_case_does_not_block_independent_scored_case():
    backend = Backend([(ASK, 31), (ACT, 27), ("answer", 8)])
    adapter = Adapter(ids=("a", "b"))
    result = RunConfig("clarify.udcs.repo_native_prompt_route").evaluate(
        "2wiki.multihop", split="dev", example_ids=["a", "b"], adapter=adapter,
        context=InterventionContext(backend=backend),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}})
    assert [row["status"] for row in result["records"]] == ["awaiting_clarification", "scored"]
    assert [row["usage"]["model_calls"] for row in result["records"]] == [1, 2]
    assert result["selected_count"] == 2
    assert result["awaiting_clarification_count"] == result["scored_count"] == 1
    assert result["failed_count"] == 0
    assert adapter.scored == ["b"] and len(backend.requests) == 3


def test_cli_reports_pending_instead_of_successful_scored_completion(tmp_path, monkeypatch, capsys):
    import agentinputlab.benchmarks

    backend = Backend([(ASK, 31), ("answer", 8)])
    monkeypatch.setattr(RunConfig, "_backend", lambda self: backend)
    monkeypatch.setattr(agentinputlab.benchmarks, "load_production_benchmark_adapter", lambda *a: Adapter(ids=("a",)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(RunConfig("clarify.udcs.repo_native_prompt_route", backend={"fixture": True}).to_dict()))
    output = tmp_path / "result.jsonl"
    assert main(["config", "evaluate", str(config), "--benchmark", "2wiki.multihop", "--split", "dev",
                 "--case-id", "a", "--prepared-root", str(tmp_path), "--output", str(output)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["awaiting_clarification_count"] == 1 and summary["failed_count"] == 0
    assert [json.loads(s)["status"] for s in output.read_text().splitlines()] == [
        "started", "awaiting_clarification", "batch_complete"]
    assert len(backend.requests) == 1
