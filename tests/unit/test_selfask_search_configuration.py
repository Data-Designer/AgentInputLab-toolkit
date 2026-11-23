"""Synthetic capability-binding checks, not real-model acceptance."""
from dataclasses import replace

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts import ContextBlock
from agentinputlab.contracts.errors import ValidationError
from tests.unit.test_benchmark_configuration import Backend


class Adapter:
    benchmark_id = "2wiki.multihop"
    pairing_mode = "stateless_score_twice"

    def __init__(self, *, tools=(), empty=False):
        self.example = BenchmarkExample(example_id="a", query="apple?",
            visible_context=() if empty else (ContextBlock("apple fruit", block_id="a1"),),
            visible_tool_names=tools, evaluator_gold={"secret": "NEVER_SEND_GOLD"})

    def examples(self, split):
        return (self.example,)

    def score(self, example, prediction):
        assert example is self.example
        return {"answer_em": float(prediction.answer == "fruit")}


def config(**kwargs):
    return RunConfig("plan.self_ask.full_loop", seed=17,
        parameters={"config": {"demonstrations": "Frozen example", "max_tokens_per_call": 512}},
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096),
        retriever=kwargs.pop("retriever", {"kind": "visible_context_bm25"}), **kwargs)


def evaluate(cfg, backend, adapter, options):
    return cfg.evaluate("2wiki.multihop", split="dev", limit=1, adapter=adapter,
        context=InterventionContext(backend=backend), predictor_parameters={
            "schema_version": 1, "qa": {"output_format": "answer_only"}, **options})


@pytest.mark.parametrize("tools", [(), ("search",)])
def test_explicit_search_preserves_original_example_and_tracks_binding(tools):
    adapter = Adapter(tools=tools)
    backend = Backend([("So the final answer is: fruit", 7), ("fruit", 3)])
    cfg = config()
    original = adapter.example.visible_state(budget=cfg.budget)
    result = evaluate(cfg, backend, adapter, {"retriever_capability": {"visible_name": "search"}})
    row, = result["records"]
    assert row["status"] == "scored"
    assert row["source_input_state_hash"] == original.state_hash
    assert row["input_state_hash"] == replace(original, tool_names=("search",)).state_hash
    assert adapter.example.visible_tool_names == tools
    assert "Available tool names:\nsearch" in backend.requests[-1].prompt
    assert "NEVER_SEND_GOLD" not in str(backend.requests)
    assert row["retriever_capability"]["fingerprint"] == row["case_retriever"]["fingerprint"]
    assert row["retriever_capability"]["native_tool_executor_added"] is False
    assert row["usage"]["model_calls"] == 2 and row["usage"]["output_tokens"] == 10
    assert row["usage"]["retrieval_calls"] == 0


def test_default_does_not_add_visible_tools():
    backend = Backend([("So the final answer is: fruit", 7), ("fruit", 3)])
    adapter, cfg = Adapter(), config()
    row, = evaluate(cfg, backend, adapter, {})["records"]
    assert row["status"] == "scored" and "retriever_capability" not in row
    assert row["input_state_hash"] == adapter.example.visible_state(budget=cfg.budget).state_hash
    assert "Available tool names:\nsearch" not in backend.requests[-1].prompt


def test_declared_search_uses_real_bound_retriever_when_requested():
    backend = Backend([("Yes.\nFollow up: apple?", 5),
                       ("So the final answer is: fruit", 7), ("fruit", 3)])
    row, = evaluate(config(), backend, Adapter(),
        {"retriever_capability": {"visible_name": "search"}})["records"]
    assert row["status"] == "scored"
    assert row["usage"]["retrieval_calls"] == 1 and row["usage"]["model_calls"] == 3
    assert "Intermediate answer: apple fruit" in backend.requests[1].prompt
    trace = row["intervention"]["patch"]["trace"]
    assert trace["events"][1]["payload"]["answer_source"] == "retrieval"
    assert trace["events"][1]["payload"]["document_ids"] == ["a1"]


@pytest.mark.parametrize("options", [None, {}, {"visible_name": "python"},
    {"visible_name": "search", "extra": True}, True])
def test_invalid_declarations_rejected_before_model(options):
    backend = Backend([])
    with pytest.raises(ValidationError, match="retriever_capability"):
        evaluate(config(), backend, Adapter(), {"retriever_capability": options})
    assert not backend.requests


@pytest.mark.parametrize("cfg", [RunConfig("baseline.no_op"), config(retriever={})])
def test_requires_selfask_and_actual_retriever(cfg):
    backend = Backend([])
    with pytest.raises(ValidationError, match="retriever_capability"):
        evaluate(cfg, backend, Adapter(), {"retriever_capability": {"visible_name": "search"}})
    assert not backend.requests


@pytest.mark.parametrize("adapter", [Adapter(empty=True), Adapter(tools=("python",))])
def test_missing_corpus_or_conflicting_tools_never_call_model(adapter):
    backend = Backend([])
    row, = evaluate(config(), backend, adapter,
        {"retriever_capability": {"visible_name": "search"}})["records"]
    assert row["status"] == "failed" and not backend.requests
    assert "retriever_capability" not in row
