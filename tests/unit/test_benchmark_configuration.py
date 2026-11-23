"""Synthetic pipeline tests; these are not real-model acceptance claims."""

import json

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.backends.llm import GenerationResponse, ModelCapability, Usage
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ValidationError


class Backend:
    model_id = "synthetic"
    capabilities = frozenset(ModelCapability)

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def fingerprint(self):
        return {"model_id": self.model_id, "revision": "fixture"}

    def generate(self, request):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        text, tokens = value
        return GenerationResponse(text=text, model=self.model_id,
                                  usage=Usage(input_tokens=20, output_tokens=tokens))


class Adapter:
    benchmark_id = "2wiki.multihop"
    pairing_mode = "stateless_score_twice"

    def __init__(self, ids=("a", "b")):
        self.ids = ids
        self.scored = []

    def examples(self, split):
        assert split == "dev"
        return tuple(BenchmarkExample(example_id=i, query="Visible question " + i,
                                      evaluator_gold={"secret": "NEVER_SEND_GOLD"}) for i in self.ids)

    def score(self, example, prediction):
        self.scored.append(example.example_id)
        return {"answer_em": float(prediction.answer == "answer")}


def evaluate(config, backend, adapter=None, runtime=None, **options):
    return config.evaluate("2wiki.multihop", split="dev", example_ids=["a", "b"],
                           adapter=adapter or Adapter(), context=InterventionContext(backend=backend, external_runtime=runtime),
                           predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}},
                           **options)


def test_complete_pipeline_scores_and_preserves_gold_boundary():
    backend = Backend([("answer", 3), ("wrong", 2)])
    adapter = Adapter()
    events = []
    result = evaluate(RunConfig("baseline.no_op"), backend, adapter, on_record=events.append)
    assert result["scored_count"] == 2 and result["failed_count"] == 0
    assert result["metrics_over_scored_cases"] == {"answer_em": 0.5}
    assert [e["status"] for e in events] == ["started", "scored", "started", "scored"]
    assert len(backend.requests) == 2
    assert all("NEVER_SEND_GOLD" not in r.prompt for r in backend.requests)
    assert [r["usage"]["model_calls"] for r in result["records"]] == [1, 1]
    assert adapter.scored == ["a", "b"]


def test_provider_failure_keeps_usage_and_denominator_does_not_retry():
    backend = Backend([RuntimeError("token=sk-private123456789"), ("answer", 1)])
    result = evaluate(RunConfig("baseline.no_op"), backend)
    assert result["selected_count"] == 2 and result["failed_count"] == 1
    assert result["records"][0]["usage"]["model_calls"] == 1
    assert result["records"][0]["usage"]["unpriced_model_calls"] == 1
    assert "evaluation" not in result["records"][0]
    assert "sk-private123456789" not in json.dumps(result)
    assert len(backend.requests) == 2


def querygym_runtime(tokens):
    from agentinputlab.algorithms.reformulate.query2doc import QueryGymRawResult
    from tests.algorithms.unit.test_query2doc import RecordingQueryGymRuntime

    return RecordingQueryGymRuntime(QueryGymRawResult(reformulated="rewrite", usage={
        "model_calls": 1, "retrieval_calls": 0, "input_tokens": 1, "output_tokens": tokens}))


def test_intervention_and_reader_share_calls_budget():
    backend = Backend([])
    runtime = querygym_runtime(2)
    config = RunConfig("rewrite.query2doc.querygym_live", parameters={"max_tokens": 8},
                       budget=ExecutionBudget(max_model_calls=1))
    result = evaluate(config, backend, runtime=runtime)
    assert result["failed_count"] == 2 and len(runtime.requests) == 2
    assert not backend.requests
    assert all(row["usage"]["model_calls"] == 1 for row in result["records"])
    assert all(row["failure"]["stage"] == "downstream" for row in result["records"])


def test_remaining_tokens_limit_reader_request_without_increasing_total():
    backend = Backend([("answer", 3), ("answer", 3)])
    config = RunConfig("rewrite.query2doc.querygym_live", parameters={"max_tokens": 8},
                       budget=ExecutionBudget(max_output_tokens=10, max_model_calls=2))
    result = evaluate(config, backend, runtime=querygym_runtime(6))
    assert result["scored_count"] == 2
    assert [r.max_tokens for r in backend.requests] == [4, 4]
    assert all(row["usage"]["output_tokens"] == 9 for row in result["records"])


def test_invalid_case_selection_fails_before_any_model_call():
    backend = Backend([])
    with pytest.raises(ValidationError, match="unknown example IDs"):
        evaluate(RunConfig("baseline.no_op"), backend, Adapter(("a",)))
    assert not backend.requests


def test_parser_failure_keeps_reusable_response_and_cost():
    backend = Backend([("not valid JSON", 7)])
    result = RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", example_ids=["a"],
        adapter=Adapter(), context=InterventionContext(backend=backend))
    row = result["records"][0]
    assert row["status"] == "failed" and row["failure"]["stage"] == "downstream"
    assert row["usage"]["output_tokens"] == 7
    assert row["model_responses"][0]["response"]["text"] == "not valid JSON"
    assert row["model_responses"][0]["request"]["max_tokens"] == 512
    assert len(backend.requests) == 1


def test_provider_overrun_is_not_scored_and_actual_usage_is_retained():
    backend = Backend([("answer", 11), ("answer", 2)])
    result = evaluate(RunConfig("baseline.no_op", budget=ExecutionBudget(max_output_tokens=10)), backend)
    assert result["failed_count"] == 1
    assert result["records"][0]["usage"]["output_tokens"] == 11
    assert result["records"][0]["failure"]["type"] == "BudgetExceededError"
    assert len(backend.requests) == 2


def test_score_failure_preserves_prediction_attempt_and_other_cases():
    class BrokenScorer(Adapter):
        def score(self, example, prediction):
            raise RuntimeError("scorer dependency missing")

    result = evaluate(RunConfig("baseline.no_op"), Backend([("answer", 2), ("answer", 2)]), BrokenScorer())
    assert result["failed_count"] == 2 and result["scored_count"] == 0
    assert all(r["failure"]["stage"] == "score" for r in result["records"])
    assert result["metrics_over_scored_cases"] == {}


@pytest.mark.parametrize("ids", [[], ["a", "a"], "a", [""], [3]])
def test_invalid_ids_rejected(ids):
    with pytest.raises(ValidationError, match="example_ids"):
        RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", example_ids=ids)


def test_stateful_adapter_is_not_silently_scored_as_qa():
    adapter = Adapter()
    adapter.pairing_mode = "stateful"
    with pytest.raises(ValidationError, match="stateful"):
        evaluate(RunConfig("baseline.no_op"), Backend([]), adapter)


def test_limit_selects_stable_prefix_without_other_model_calls():
    backend = Backend([("answer", 1)])
    result = RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", limit=1,
        adapter=Adapter(), context=InterventionContext(backend=backend),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}})
    assert result["selected_example_ids"] == ["a"]
    assert result["scored_count"] == 1 and len(backend.requests) == 1


@pytest.mark.parametrize("selection", [{}, {"limit": 0}, {"limit": True}, {"limit": 1.5},
                                      {"limit": 1, "example_ids": ["a"]}])
def test_ambiguous_or_invalid_selection_never_executes(selection):
    with pytest.raises(ValidationError, match="choose"):
        RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", **selection)


def test_composite_final_answer_scores_without_extra_reader_call(tmp_path):
    import hashlib

    from tests.algorithms.unit.test_plan_and_budget_full import PromptFixture

    cache = tmp_path / "cache.json"
    raw = json.dumps({"last_question_total": 1, "stats": [
        {"id": i, "query": "Visible question " + i, "steps": ["Find", "Verify"],
         "credits": [40, 60], "level": 3} for i in ("a", "b")]}).encode()
    cache.write_bytes(raw)
    config = RunConfig("plan.plan_and_budget.full", parameters={"output_format": "plain"},
                       artifacts={"records": {"path": str(cache), "sha256": hashlib.sha256(raw).hexdigest()}})
    backend = Backend([("answer", 4), ("answer", 4)])
    result = evaluate(config, backend, runtime=PromptFixture())
    assert result["scored_count"] == 2
    assert len(backend.requests) == 2
    assert all(r["usage"]["model_calls"] == 1 for r in result["records"])
    assert all(r["downstream_metadata"]["composite_answer_no_extra_call"] for r in result["records"])


def test_cli_persists_records_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    import agentinputlab.benchmarks

    backend = Backend([("answer", 1)])
    monkeypatch.setattr(RunConfig, "_backend", lambda self: backend)
    monkeypatch.setattr(agentinputlab.benchmarks, "load_production_benchmark_adapter", lambda *a: Adapter())
    config = tmp_path / "config.json"
    config.write_text(json.dumps(RunConfig("baseline.no_op", backend={"fixture": True}).to_dict()))
    predictor = tmp_path / "reader.json"
    predictor.write_text(json.dumps({"schema_version": 1, "qa": {"output_format": "answer_only"}}))
    out = tmp_path / "out.jsonl"
    args = ["config", "evaluate", str(config), "--benchmark", "2wiki.multihop", "--split", "dev",
            "--case-id", "a", "--prepared-root", str(tmp_path), "--predictor-config", str(predictor),
            "--output", str(out)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["scored_count"] == 1
    before = out.read_bytes()
    assert [json.loads(x)["status"] for x in out.read_text().splitlines()] == ["started", "scored", "batch_complete"]
    assert main(args) == 2
    assert out.read_bytes() == before and len(backend.requests) == 1


@pytest.mark.parametrize("with_backend", [True, False])
def test_native_ir_scoring_uses_retriever_and_no_reader_model_call(with_backend):
    class IRAdapter(Adapter):
        benchmark_id = "beir.arguana"

        def examples(self, split):
            return (BenchmarkExample(example_id="a", query="Paris"),)

        def score(self, example, prediction):
            return {"native_metric_fixture": float(bool(prediction.scores))}

    backend = Backend([])
    retriever = InMemoryBM25Retriever([Document(doc_id="d", text="Paris")])
    result = RunConfig("baseline.no_op").evaluate("beir.arguana", split="test", example_ids=["a"],
        adapter=IRAdapter(), context=InterventionContext(backend=backend if with_backend else None, retriever=retriever))
    assert result["scored_count"] == 1
    assert result["records"][0]["usage"]["retrieval_calls"] == 1
    assert not backend.requests
    assert result["records"][0]["model_responses"] == []
    assert result["records"][0]["usage"]["model_calls"] == 0


def test_missing_qa_backend_rejected_before_dataset_loading():
    from agentinputlab.contracts.errors import CapabilityError

    retriever = InMemoryBM25Retriever([Document(doc_id="d", text="Paris")])
    with pytest.raises(CapabilityError, match="QA prediction requires an LLMBackend"):
        RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", limit=1,
            context=InterventionContext(retriever=retriever), offline=True)


def test_ir_method_requiring_llm_does_not_get_synthetic_backend():
    class IRAdapter(Adapter):
        benchmark_id = "beir.arguana"

        def examples(self, split):
            return (BenchmarkExample(example_id="a", query="Paris"),)

        def score(self, example, prediction):
            pytest.fail("missing method dependency must not produce a score")

    retriever = InMemoryBM25Retriever([Document(doc_id="d", text="Paris")])
    result = RunConfig("rewrite.query2doc.querygym_live").evaluate(
        "beir.arguana", split="test", limit=1, adapter=IRAdapter(),
        context=InterventionContext(retriever=retriever))
    assert result["failed_count"] == 1 and result["scored_count"] == 0
    assert result["records"][0]["failure"]["stage"] == "intervention"
    assert result["records"][0]["model_responses"] == []
