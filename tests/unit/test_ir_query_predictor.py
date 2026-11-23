from __future__ import annotations

import json

import pytest

from agentinputlab.backends import Document, InMemoryBM25Retriever, ReplayBackend
from agentinputlab.backends.llm import GenerationResponse, Usage
from agentinputlab.backends.retrieval import RetrievalResult
from agentinputlab.benchmarks import ContextIRQueryPredictor, IRQueryOutputError
from agentinputlab.benchmarks.ir_query import parse_ir_query
from agentinputlab.contracts import (
    AlgorithmOutputError,
    CapabilityError,
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    Message,
    ValidationError,
)


class Recording:
    def __init__(self, text='{"query":"alpha"}', *, tokens=8):
        self.delegate = ReplayBackend([
            GenerationResponse(text=text, model="fixture", usage=Usage(23, tokens))
        ])
        self.requests = []
        self.model_id = self.delegate.model_id
        self.capabilities = self.delegate.capabilities

    def fingerprint(self):
        return self.delegate.fingerprint()

    def generate(self, request):
        self.requests.append(request)
        return self.delegate.generate(request)


def retriever():
    return InMemoryBM25Retriever((Document("a", "alpha evidence"), Document("b", "beta")))


def state(**kwargs):
    return FrozenAgentState(query="find alpha", instruction="Find the counterargument.", **kwargs)


def test_exact_visible_query_endpoint_native_ranking_and_costs():
    backend = Recording()
    s = state(
        task_id="private-id", metadata={"private_label": "do-not-expose"},
        context=(ContextBlock("whole evidence", "doc1", metadata={"hidden": "secret"}),),
        messages=(Message("user", "visible conversation"),),
    )
    result = ContextIRQueryPredictor(backend, retriever(), seed=19, top_k=2).predict(s)
    request = backend.requests[0]
    payload = json.loads(request.prompt)
    assert payload["task_instruction"] == s.instruction
    assert payload["original_query"] == s.query
    assert payload["context"][0]["text"] == "whole evidence"
    assert payload["conversation"][0]["content"] == "visible conversation"
    assert not any(x in request.prompt for x in ("private-id", "do-not-expose", "secret"))
    assert request.seed == 19 and request.max_tokens == 512
    assert result.metadata["generated_query"] == "alpha"
    assert next(iter(result.prediction.scores)) == "a"
    assert result.usage.model_calls == result.usage.retrieval_calls == 1
    assert result.usage.input_tokens == 23 and result.usage.output_tokens == 8
    assert result.usage.unpriced_model_calls == 1 and len(backend.requests) == 1


@pytest.mark.parametrize("text", [
    "", "answer: alpha", '{"answer":"alpha"}', '{"query":"alpha","answer":"x"}',
    '{"query":"alpha","query":"beta"}', '{"query":[]}', '{"query":"  "}',
    '```json\n{"query":"alpha"}\n```', '{"query":null}', "[]",
])
def test_bad_final_query_never_falls_back_or_retries(text):
    backend = Recording(text)
    with pytest.raises(IRQueryOutputError) as failure:
        ContextIRQueryPredictor(backend, retriever()).predict(state())
    assert isinstance(failure.value, AlgorithmOutputError)
    assert failure.value.response.text == text
    assert failure.value.usage.model_calls == 1
    assert failure.value.usage.retrieval_calls == 0
    assert len(backend.requests) == 1


def test_parser_preserves_query_whitespace_and_unicode():
    assert parse_ir_query('{"query":"  \u03b1 evidence  "}') == "  \u03b1 evidence  "


@pytest.mark.parametrize("text", [
    '```json\n{"query":"  \u03b1 evidence  "}\n```',
    '```\n{"query":"  \u03b1 evidence  "}\n```',
    ' \n```json\r\n{"query":"  \u03b1 evidence  "}\r\n```\n ',
    '{"query":"  \u03b1 evidence  "}',
])
def test_opt_in_single_fence_keeps_exact_query_value(text):
    assert parse_ir_query(text, allow_json_fence=True) == "  \u03b1 evidence  "


@pytest.mark.parametrize("text", [
    'Answer: ```json\n{"query":"alpha"}\n```',
    '```json\n{"query":"alpha"}\n```\nExtra text',
    '```python\n{"query":"alpha"}\n```',
    '```json\n```json\n{"query":"alpha"}\n```\n```',
    '```json\n{"query":"alpha","query":"beta"}\n```',
    '```json\n{"answer":"alpha"}\n```',
    '```json\n{"query":" "}\n```',
    '```json\n{"query":"alpha"}\n```\n```json\n{"query":"beta"}\n```',
])
def test_opt_in_fence_does_not_extract_answers_or_repair_other_invalid_outputs(text):
    with pytest.raises(AlgorithmOutputError):
        parse_ir_query(text, allow_json_fence=True)


def test_opt_in_fence_preserves_raw_response_costs_and_generation_request():
    text = '```json\n{"query":"alpha"}\n```'
    strict_backend = Recording(text)
    with pytest.raises(IRQueryOutputError):
        ContextIRQueryPredictor(strict_backend, retriever()).predict(state())
    backend = Recording(text)
    p = ContextIRQueryPredictor(backend, retriever(), allow_json_fence=True)
    result = p.predict(state())
    assert backend.requests == strict_backend.requests
    assert result.metadata["generation_response"]["text"] == text
    assert result.metadata["decoder_policy"] == p.fingerprint()["decoder_policy"] == "single_json_fence.v1"
    assert result.usage.model_calls == result.usage.retrieval_calls == 1
    assert result.usage.output_tokens == 8
    assert "decoder_policy" not in ContextIRQueryPredictor(Recording(), retriever()).fingerprint()


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_fence_option_requires_an_explicit_boolean(value):
    with pytest.raises(ValidationError):
        parse_ir_query('{"query":"alpha"}', allow_json_fence=value)
    with pytest.raises(ValidationError):
        ContextIRQueryPredictor(Recording(), retriever(), allow_json_fence=value)


@pytest.mark.parametrize("budget", [
    ExecutionBudget(max_model_calls=0), ExecutionBudget(max_output_tokens=0),
    ExecutionBudget(max_retrieval_calls=0),
])
def test_joint_budget_gate_precedes_model_and_retrieval(budget):
    backend = Recording()
    with pytest.raises(CapabilityError):
        ContextIRQueryPredictor(backend, retriever()).predict(state(budget=budget))
    assert backend.requests == []


def test_configured_cap_is_clamped_to_remaining_not_replenished():
    backend = Recording()
    s = state(budget=ExecutionBudget(max_output_tokens=31, max_model_calls=7))
    ContextIRQueryPredictor(backend, retriever(), max_tokens=4000).predict(s)
    assert backend.requests[0].max_tokens == 31


def test_output_over_cap_is_a_consumed_failure():
    backend = Recording(tokens=33)
    with pytest.raises(IRQueryOutputError) as failure:
        ContextIRQueryPredictor(backend, retriever(), max_tokens=32).predict(state())
    assert failure.value.usage.output_tokens == 33 and len(backend.requests) == 1


def test_empty_native_ranking_is_not_replaced_with_an_oracle_or_original_query():
    class Empty:
        retriever_id = "empty"

        def fingerprint(self):
            return {"implementation": "empty_fixture"}

        def retrieve(self, query, *, top_k):
            assert query == "alpha" and top_k == 15
            return ()

    result = ContextIRQueryPredictor(Recording(), Empty()).predict(state())
    assert not result.prediction.scores and result.usage.retrieval_calls == 1


def test_retriever_exception_retains_the_original_failure_and_consumed_usage():
    class Broken:
        retriever_id = "broken"

        def fingerprint(self):
            return {"implementation": "broken_fixture"}

        def retrieve(self, query, *, top_k):
            raise OSError("original ranker failure")

    backend = Recording()
    with pytest.raises(IRQueryOutputError) as failure:
        ContextIRQueryPredictor(backend, Broken()).predict(state())
    assert isinstance(failure.value.__cause__, OSError)
    assert failure.value.usage.model_calls == failure.value.usage.retrieval_calls == 1
    assert len(backend.requests) == 1


@pytest.mark.parametrize("kind", ["rank", "duplicate", "oversized"])
def test_bad_retrieval_results_retain_consumed_costs(kind):
    class Bad:
        retriever_id = "bad"

        def fingerprint(self):
            return {"implementation": "bad_fixture"}

        def retrieve(self, query, *, top_k):
            if kind == "rank":
                return (RetrievalResult(Document("a", "x"), 1.0, 2),)
            if kind == "duplicate":
                return (RetrievalResult(Document("a", "x"), 1.0, 1),
                        RetrievalResult(Document("a", "x"), 1.0, 2))
            return tuple(RetrievalResult(Document(str(i), "x"), 1.0, i + 1) for i in range(3))

    with pytest.raises(IRQueryOutputError) as failure:
        ContextIRQueryPredictor(Recording(), Bad(), top_k=2).predict(state())
    assert failure.value.usage.model_calls == failure.value.usage.retrieval_calls == 1


@pytest.mark.parametrize("kwargs", [
    {"top_k": True}, {"top_k": 0}, {"top_k": 10_001},
    {"max_tokens": 0}, {"max_tokens": True}, {"seed": True}, {"seed": -1},
])
def test_bad_configuration_fails_before_generation(kwargs):
    with pytest.raises(ValidationError):
        ContextIRQueryPredictor(Recording(), retriever(), **kwargs)
