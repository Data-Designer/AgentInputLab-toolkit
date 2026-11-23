"""Synthetic public pipeline tests; saved/live acceptance is recorded separately."""
import json

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ValidationError
from tests.unit.test_benchmark_configuration import Backend
from tests.unit.test_composite_ir_evaluate_configuration import IRAdapter


def setup(text='{"query":"savings"}', budget=None):
    config = RunConfig("plan.self_ask.planner_component", parameters={
        "config": {"demonstrations": "Frozen demonstrations"},
        "allow_no_decision_explanation": True}, budget=budget or ExecutionBudget())
    backend = Backend([("No.\nThis request needs no follow-up.", 6), (text, 4)])
    context = InterventionContext(backend=backend,
        retriever=InMemoryBM25Retriever([Document("d", "savings account")]))
    return config, backend, context


def options(**binding):
    return {"schema_version": 1, "retrieval": {"top_k": 1},
            "patch_retrieval": {"output_format": "json_query", **binding}}


def evaluate(config, context, parameters):
    return config.evaluate("beir.arguana", split="test", example_ids=["a"],
        adapter=IRAdapter(), context=context, predictor_parameters=parameters)


def test_paid_helper_reader_ranking_and_score_one_command():
    config, backend, context = setup()
    result = evaluate(config, context, options(context_prefill={"top_k": 1, "max_document_chars": 10000}))
    row = result["records"][0]
    assert row["status"] == "scored" and len(backend.requests) == 2
    assert row["usage"]["model_calls"] == 2 and row["usage"]["output_tokens"] == 10
    assert row["usage"]["retrieval_calls"] == 2
    assert row["intervention"]["patch"]["metadata"]["compatibility_applied"] is True
    assert row["downstream_metadata"]["patch_query_reader"]["paid_query_reader"] is True
    assert "NEVER_SEND_GOLD" not in str(backend.requests)
    reader = backend.requests[1]
    assert reader.max_tokens == 512 and reader.seed == 17
    assert reader.response_format == {"type": "json_object"}
    visible = json.loads(reader.prompt.split("\n\n", 1)[1])
    assert visible["visible_context"][0]["text"] == "savings account"
    assert "task_id" not in visible and "metadata" not in visible


@pytest.mark.parametrize("text", ['{"answer":"savings"}', '', '{"query":"a","query":"b"}'])
def test_reader_failure_preserves_response_cost_and_never_retries(text):
    config, backend, context = setup(text)
    row = evaluate(config, context, options())["records"][0]
    assert row["status"] == "failed" and row["failure"]["stage"] == "downstream"
    assert len(backend.requests) == 2 and row["usage"]["output_tokens"] == 10
    assert row["usage"]["retrieval_calls"] == 0
    assert row["model_responses"][1]["response"]["text"] == text


@pytest.mark.parametrize("allow,scored", [(False, 0), (True, 1)])
def test_fence_policy_is_explicit(allow, scored):
    config, backend, context = setup('```json\n{"query":"savings"}\n```')
    assert evaluate(config, context, options(allow_json_fence=allow))["scored_count"] == scored
    assert len(backend.requests) == 2


@pytest.mark.parametrize("budget", [ExecutionBudget(max_model_calls=1),
    ExecutionBudget(max_output_tokens=6), ExecutionBudget(max_retrieval_calls=0)])
def test_no_reader_after_remaining_budget_exhausted(budget):
    config, backend, context = setup(budget=budget)
    row = evaluate(config, context, options())["records"][0]
    assert row["status"] == "failed" and len(backend.requests) == 1
    assert row["usage"]["model_calls"] == 1


def test_reader_uses_only_remaining_token_allowance():
    config, backend, context = setup(budget=ExecutionBudget(max_output_tokens=10))
    assert evaluate(config, context, options())["scored_count"] == 1
    assert backend.requests[1].max_tokens == 4


@pytest.mark.parametrize("bad", [None, {}, {"output_format":"answer"},
    {"output_format":"json_query","max_tokens":True},
    {"output_format":"json_query","temperature":float('nan')},
    {"output_format":"json_query","extra":1}])
def test_bad_configuration_fails_before_generation(bad):
    config, backend, context = setup()
    with pytest.raises(ValidationError, match="patch_retrieval"):
        evaluate(config, context, {"schema_version":1,"patch_retrieval":bad})
    assert not backend.requests


def test_incompatible_query_source_fails_before_generation():
    config, backend, context = setup()
    params = options()
    params["retrieval"]["source_mode"] = "context_or_retriever"
    with pytest.raises(ValidationError, match="patch_retrieval"):
        evaluate(config, context, params)
    assert not backend.requests


def test_default_does_not_add_a_reader():
    config, backend, context = setup()
    row = evaluate(config, context, {"schema_version":1,"retrieval":{"top_k":1}})["records"][0]
    assert row["status"] == "scored" and len(backend.requests) == 1
    assert "patch_query_reader" not in row["downstream_metadata"]


@pytest.mark.parametrize("allow,scored", [(False, 0), (True, 1)])
def test_native_context_reader_uses_exact_original_request_and_preserves_fence_policy(allow, scored):
    from dataclasses import replace

    from agentinputlab.benchmarks import ContextIRQueryPredictor
    from agentinputlab.benchmarks.ir_query import IR_QUERY_ENDPOINT_ID, IR_QUERY_SYSTEM_PROMPT

    text = '```json\n{"query":"savings"}\n```'
    config, backend, context = setup(text)
    row = evaluate(config, context, options(protocol=IR_QUERY_ENDPOINT_ID, allow_json_fence=allow))["records"][0]
    assert (row["status"] == "scored") == bool(scored)
    assert row["usage"]["model_calls"] == 2 and row["usage"]["output_tokens"] == 10
    assert row["usage"]["retrieval_calls"] == scored
    assert len(backend.requests) == 2
    assert backend.requests[1].system == IR_QUERY_SYSTEM_PROMPT
    assert backend.requests[1].response_format is None
    native = Backend([(text, 4)])
    state = IRAdapter().examples('test')[0].visible_state(budget=replace(config.budget,
        max_model_calls=config.budget.max_model_calls-1, max_output_tokens=config.budget.max_output_tokens-6))
    predictor = ContextIRQueryPredictor(native, context.retriever, top_k=1, allow_json_fence=allow)
    if allow:
        predictor.predict(state)
        assert row['downstream_metadata']['endpoint_id'] == IR_QUERY_ENDPOINT_ID
    else:
        from agentinputlab.benchmarks import IRQueryOutputError
        with pytest.raises(IRQueryOutputError):
            predictor.predict(state)
    assert backend.requests[1] == native.requests[0]
    assert 'NEVER_SEND_GOLD' not in repr(backend.requests)


@pytest.mark.parametrize('extra', [dict(protocol='guessed'),
    dict(protocol='agentinputlab.context_ir_query_endpoint.v1', temperature=0.5)])
def test_native_reader_protocol_validation_before_any_call(extra):
    config, backend, context = setup()
    with pytest.raises(ValidationError):
        evaluate(config, context, options(**extra))
    assert not backend.requests


@pytest.mark.parametrize("cell", ["bright.earth_science", "bright.economics",
                                  "bright.robotics", "bright.stackoverflow"])
def test_all_registered_bright_domains_reach_native_scoring(cell):
    config, backend, context = setup()
    adapter = IRAdapter()
    adapter.benchmark_id = cell
    result = config.evaluate(cell, split="test", example_ids=["a"], adapter=adapter,
        context=context, predictor_parameters=options())
    assert result["scored_count"] == 1 and len(backend.requests) == 2


def test_cli_executes_explicit_patch_reader_and_persists_score(tmp_path, monkeypatch, capsys):
    import agentinputlab.benchmarks
    from agentinputlab.cli.main import main

    config, backend, context = setup()
    run = tmp_path / "run.json"
    predictor = tmp_path / "predictor.json"
    output = tmp_path / "out.jsonl"
    run.write_text(json.dumps(config.to_dict()))
    predictor.write_text(json.dumps(options()))
    monkeypatch.setattr(RunConfig, "_configured_context", lambda self, supplied: context)
    monkeypatch.setattr(agentinputlab.benchmarks, "load_production_benchmark_adapter", lambda *a: IRAdapter())
    assert main(["config", "evaluate", str(run), "--benchmark", "beir.arguana", "--split", "test",
        "--case-id", "a", "--prepared-root", str(tmp_path), "--predictor-config", str(predictor),
        "--output", str(output)]) == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["status"] for r in rows] == ["started", "scored", "batch_complete"]
    assert rows[1]["usage"]["model_calls"] == len(backend.requests) == 2
    assert rows[1]["downstream_metadata"]["patch_query_reader"]["paid_query_reader"]
