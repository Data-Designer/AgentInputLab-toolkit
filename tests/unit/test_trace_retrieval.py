import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.benchmarks.trace_retrieval import prediction_from_trace
from agentinputlab.contracts import ExecutionTrace, TraceEvent, TracePhase, ValidationError
from tests.unit.test_benchmark_configuration import Backend


@pytest.mark.parametrize(
    ("release", "event", "field"),
    [
        ("plan.decomp.common_support_odqa", "decomp_common_support_result", "retrieved_document_ids"),
        ("acquire.ircot.common_support", "ircot_exit", "active_document_ids"),
        ("acquire.qrag.common_support", "qrag_selected_real_context", "selected_document_ids"),
    ],
)
def test_final_rank_keeps_exact_order_and_rejects_unobserved_ids(release, event, field):
    record = TraceEvent(TracePhase.FINALIZE, event, {field: ("b", "a")})
    trace = ExecutionTrace(release.rsplit(".", 1)[0], release, (record,))
    assert list(prediction_from_trace(trace, observed_document_ids={"a", "b", "c"}).scores) == ["b", "a"]
    with pytest.raises(ValidationError, match="absent"):
        prediction_from_trace(trace, observed_document_ids={"b"})
    with pytest.raises(ValidationError, match="ambiguous"):
        prediction_from_trace(trace.append(record), observed_document_ids={"a", "b"})


def test_self_ask_uses_first_retrieval_appearance_only():
    def step(kind, ids):
        return TraceEvent(TracePhase.OBSERVE, "self_ask_step", {"answer_source": kind, "document_ids": ids})

    release = "plan.self_ask.full_loop"
    end = TraceEvent(TracePhase.FINALIZE, "self_ask_final", {"final_answer": "never rank this"})
    trace = ExecutionTrace("plan.self_ask", release, (
        step("retrieval", ("a", "b")), step("model_fallback", ()), step("retrieval", ("b", "c")), end,
    ))
    assert list(prediction_from_trace(trace, observed_document_ids={"a", "b", "c"}).scores) == ["a", "b", "c"]
    with pytest.raises(ValidationError, match="non-retrieval"):
        prediction_from_trace(trace.append(step("model_fallback", ("a",))), observed_document_ids={"a", "b", "c"})


class RetrievalAdapter:
    benchmark_id = "bright.earth_science"
    pairing_mode = "stateless_score_twice"

    def __init__(self):
        self.case = BenchmarkExample(example_id="a", query="apple?", evaluator_gold={"private": "secret-not-visible"})

    def examples(self, split):
        assert split == "test"
        return (self.case,)

    def score(self, example, prediction):
        assert example is self.case
        assert list(prediction.scores) == ["real-id"]
        return {"ndcg_at_10": 1.0}


def test_full_self_ask_uses_observed_rank_without_extra_reader_or_search():
    cfg = RunConfig("plan.self_ask.full_loop", seed=17,
        parameters={"config": {"demonstrations": "Frozen example", "max_tokens_per_call": 512}},
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
    backend = Backend([("Yes.\nFollow up: apple?", 5), ("So the final answer is: fruit", 7)])
    out = cfg.evaluate("bright.earth_science", split="test", limit=1, adapter=RetrievalAdapter(),
        context=InterventionContext(backend=backend,
            retriever=InMemoryBM25Retriever((Document("real-id", "apple fruit"),))),
        predictor_parameters={"schema_version": 1,
            "trace_ranking": {"protocol": "observed_rank.v1"},
            "retriever_capability": {"visible_name": "search"}})
    row, = out["records"]
    assert row["status"] == "scored"
    assert row["downstream_metadata"]["ranked_document_ids"] == ["real-id"]
    assert row["observed_retrieval_ids"] == [["real-id"]]
    assert row["usage"]["model_calls"] == len(backend.requests) == 2
    assert row["usage"]["retrieval_calls"] == 1
    assert "secret-not-visible" not in str(backend.requests)


def test_rank_protocol_rejects_other_releases_before_backend_call():
    backend = Backend([])
    with pytest.raises(ValidationError, match="trace_ranking"):
        RunConfig("baseline.no_op").evaluate("bright.earth_science", split="test", limit=1,
            adapter=RetrievalAdapter(), context=InterventionContext(backend=backend,
                retriever=InMemoryBM25Retriever((Document("real-id", "apple fruit"),))),
            predictor_parameters={"schema_version": 1, "trace_ranking": {"protocol": "observed_rank.v1"}})
    assert not backend.requests


@pytest.mark.parametrize("predictor_override", [{"task": "qa"}, {"qa": {"max_tokens": 4}},
                                                  {"retrieval": {"top_k": 1}}])
def test_rank_protocol_rejects_ignored_predictor_options_before_generation(predictor_override):
    cfg = RunConfig("plan.self_ask.full_loop", seed=17,
        parameters={"config": {"demonstrations": "Frozen example", "max_tokens_per_call": 512}})
    backend = Backend([])
    with pytest.raises(ValidationError, match="preserves the method's native ranking"):
        cfg.evaluate("bright.earth_science", split="test", limit=1, adapter=RetrievalAdapter(),
            context=InterventionContext(backend=backend,
                retriever=InMemoryBM25Retriever((Document("real-id", "apple fruit"),))),
            predictor_parameters={"schema_version": 1, "trace_ranking": {"protocol": "observed_rank.v1"},
                **predictor_override})
    assert not backend.requests


def test_workflow_preflight_uses_the_native_ranking_contract():
    from dataclasses import replace

    from agentinputlab.workflows import WorkflowConfig
    from agentinputlab.workflows.readiness import predictor_readiness

    run = RunConfig("plan.self_ask.full_loop", retriever={"kind": "beir_archive", "benchmark_id": "beir.fiqa_2018"},
        parameters={"config": {"demonstrations": "Frozen example", "max_tokens_per_call": 512}})
    params = {"schema_version": 1, "trace_ranking": {"protocol": "observed_rank.v1"},
              "retriever_capability": {"visible_name": "search"}}
    target = {"kind": "benchmark", "benchmark_id": "beir.fiqa_2018", "split": "dev", "limit": 1,
              "predictor_parameters": params}
    workflow = WorkflowConfig(run, target)
    assert predictor_readiness(workflow, dependencies=False)["model_calls"] == 0
    bad = replace(workflow, target={**target, "predictor_parameters": {**params, "retrieval": {"top_k": 1}}})
    with pytest.raises(ValidationError, match="preserves the method's native ranking"):
        predictor_readiness(bad, dependencies=False)


def test_frozen_qrag_ranking_preflight_does_not_require_unused_generation():
    from agentinputlab.workflows import WorkflowConfig, inspect_workflow
    from agentinputlab.workflows.readiness import budget_and_input_readiness, predictor_readiness

    run = RunConfig("acquire.qrag.common_support", budget=ExecutionBudget(max_model_calls=0, max_output_tokens=0),
                    retriever={"kind": "beir_archive", "benchmark_id": "beir.fiqa_2018"})
    workflow = WorkflowConfig(run, {"kind": "benchmark", "benchmark_id": "beir.fiqa_2018",
        "split": "dev", "limit": 1, "predictor_parameters": {"schema_version": 1,
            "trace_ranking": {"protocol": "observed_rank.v1"}}})
    assert predictor_readiness(workflow, dependencies=False)["model_calls"] == 0
    assert budget_and_input_readiness(workflow, None, [])["minimum_model_calls"] == 0
    report = inspect_workflow(workflow, _inspect_data=False)
    # Runtime assets may be missing; an unused remote generator is not a prerequisite.
    assert not any(check["check"] == "backend" and check["status"] == "blocked" for check in report["checks"])
