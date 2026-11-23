from __future__ import annotations

import json
import math
import unittest

from agentinputlab.algorithms.clarify import (
    ActiveDisambiguationControls,
    ActiveTaskDisambiguation,
    empirical_entropy,
)
from agentinputlab.backends.llm import (
    CallableBackend, ModelCapability, ReplayBackend, GenerationResponse, Usage,
)
from agentinputlab.algorithms.clarify.active_task_disambiguation import _atd_string_list
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)


def _response(value):
    return json.dumps(value)


_CAPABILITIES = frozenset(
    {
        ModelCapability.CHAT,
        ModelCapability.COMPLETION,
        ModelCapability.JSON_OBJECT,
        ModelCapability.SEED,
    }
)


def test_active_disambiguation_selects_maximum_entropy_question() -> None:
    backend = ReplayBackend(
        [
            _response({"hypotheses": ["intent a", "intent b", "intent c", "intent d"]}),
            _response({"questions": ["Question one?", "Question two?"]}),
            _response(
                {
                    "answers": [
                        {"hypothesis_id": index, "answer_category": "same"} for index in range(4)
                    ]
                }
            ),
            _response(
                {
                    "answers": [
                        {"hypothesis_id": 0, "answer_category": "yes"},
                        {"hypothesis_id": 1, "answer_category": "yes"},
                        {"hypothesis_id": 2, "answer_category": "no"},
                        {"hypothesis_id": 3, "answer_category": "no"},
                    ]
                }
            ),
        ],
        capabilities=_CAPABILITIES,
    )
    method = ActiveTaskDisambiguation(
        backend=backend,
        controls=ActiveDisambiguationControls(total_hypotheses=4, total_questions=2),
    )
    patch = method.propose(FrozenAgentState(query="Ambiguous task"), InterventionContext())
    assert patch.operations[0].value == "Question two?"
    assert patch.metadata["selected_question_index"] == 1
    assert patch.trace.counters["model_calls"] == 4
    assert math.isclose(patch.metadata["entropies"][1], math.log(2))


def test_active_disambiguation_entropy_matches_uniform_partition() -> None:
    assert math.isclose(empirical_entropy(("a", "a", "b", "b")), math.log(2))
    assert empirical_entropy(("a", "a")) == 0.0


def test_active_disambiguation_respects_visible_fair_per_call_output_cap() -> None:
    responses = iter(
        [
            _response({"hypotheses": ["intent a", "intent b"]}),
            _response({"questions": ["Which intent?"]}),
            _response(
                {
                    "answers": [
                        {"hypothesis_id": 0, "answer_category": "a"},
                        {"hypothesis_id": 1, "answer_category": "b"},
                    ]
                }
            ),
        ]
    )
    backend = CallableBackend(
        fn=lambda _request: next(responses),
        revision="fixture-v1",
        callable_id="active-task-output-cap-fixture",
        capabilities=_CAPABILITIES,
    )
    state = FrozenAgentState(
        query="Ambiguous task",
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096),
    )

    patch = ActiveTaskDisambiguation(
        backend=backend,
        controls=ActiveDisambiguationControls(
            total_hypotheses=2,
            total_questions=1,
            max_tokens=1536,
        ),
    ).propose(state, InterventionContext())

    assert [request.max_tokens for request in backend.calls] == [512, 512, 512]
    assert patch.metadata["effective_max_tokens_per_call"] == 512


def test_active_disambiguation_rejects_incomplete_partition() -> None:
    backend = ReplayBackend(
        [
            _response({"hypotheses": ["a", "b"]}),
            _response({"questions": ["Which?"]}),
            _response({"answers": [{"hypothesis_id": 0, "answer_category": "a"}]}),
        ],
        capabilities=_CAPABILITIES,
    )
    with unittest.TestCase().assertRaisesRegex(AlgorithmOutputError, "cover every hypothesis"):
        ActiveTaskDisambiguation(backend=backend).propose(
            FrozenAgentState(query="Ambiguous task"), InterventionContext()
        )


def test_atd_text_records_preserve_content_order_entropy_calls_and_output_cap() -> None:
    outputs = [
        {"hypotheses": [{"id": "H1", "description": "a"},
                        {"id": "H2", "description": "b"}]},
        {"questions": [{"text": "Which?"}]},
        {"answers": [{"hypothesis_id": 0, "answer_category": "a"},
                     {"hypothesis_id": 1, "answer_category": "b"}]},
    ]
    backend = ReplayBackend([_response(x) for x in outputs], capabilities=_CAPABILITIES)
    patch = ActiveTaskDisambiguation(backend=backend).propose(
        FrozenAgentState(query="Ambiguous task",
                         budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)),
        InterventionContext(),
    )
    assert patch.operations[0].value == "Which?"
    assert math.isclose(patch.metadata["entropies"][0], math.log(2))
    assert patch.trace.counters["model_calls"] == 3
    assert patch.metadata["effective_max_tokens_per_call"] == 512
    assert patch.metadata["response_adapter_revision"] == "unambiguous-text-records.v1"
    assert [x["text_records_unwrapped"] for x in patch.metadata["response_normalizations"]] == [2, 1]


def test_atd_rejects_ambiguous_records_without_dropping_fields_or_filling_content() -> None:
    invalid = [
        [{"text": "a", "description": "b"}, "b"],
        [{"text": "a", "probability": 0.8}, "b"],
        [{"text": ""}, "b"],
        [{"id": "H1", "text": "a"}, {"id": "H1", "text": "b"}],
        [{"id": True, "text": "a"}, "b"],
        ["1. a; 2. b"],
    ]
    for value in invalid:
        with unittest.TestCase().assertRaises(AlgorithmOutputError):
            _atd_string_list(value, label="hypotheses", minimum=2, maximum=8,
                             text_fields=("text", "description"), normalizations=[])


def test_atd_hypothesis_aliases_preserve_next_requests_and_decisions() -> None:
    def run(field):
        first = (["intent a", "intent b"] if field is None else
                 [{"id": "H1", field: "intent a"}, {"id": "H2", field: "intent b"}])
        outputs = iter([
            _response({"hypotheses": first}),
            _response({"questions": ["Which intent?"]}),
            _response({"answers": [{"hypothesis_id": 0, "answer_category": "a"},
                                   {"hypothesis_id": 1, "answer_category": "b"}]}),
        ])
        backend = CallableBackend(fn=lambda _: next(outputs), revision="alias-test-v1",
                                  callable_id="atd-alias-test", capabilities=_CAPABILITIES)
        patch = ActiveTaskDisambiguation(backend=backend).propose(
            FrozenAgentState(query="Ambiguous task",
                             budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)),
            InterventionContext(seed=17),
        )
        return backend.calls, patch

    reference_calls, reference_patch = run(None)
    for field in ("text", "description", "hypothesis", "statement"):
        calls, patch = run(field)
        assert calls == reference_calls
        assert patch.operations == reference_patch.operations
        assert patch.metadata["entropies"] == reference_patch.metadata["entropies"]
        assert patch.trace.counters["model_calls"] == 3
        assert [r.max_tokens for r in calls] == [512, 512, 512]


def test_atd_hypothesis_aliases_do_not_accept_conflicts_or_discard_fields() -> None:
    invalid = [
        [{"hypothesis": "a", "text": "a"}, "b"],
        [{"hypothesis": "a", "statement": "b"}, "b"],
        [{"statement": "a", "confidence": 0.9}, "b"],
        [{"hypothesis": ""}, "b"],
        [{"statement": ["a"]}, "b"],
        [{"id": "H1", "hypothesis": "a"}, {"id": "H1", "statement": "b"}],
    ]
    for value in invalid:
        with unittest.TestCase().assertRaises(AlgorithmOutputError):
            _atd_string_list(value, label="hypotheses", minimum=2, maximum=8,
                             text_fields=("text", "description", "hypothesis", "statement"),
                             normalizations=[])


def test_atd_empty_or_truncated_final_is_not_retried_or_replaced_by_reasoning() -> None:
    for final in (None, "", '{"hypotheses":["unfinished'):
        response = GenerationResponse(
            text=final, model="fixture", finish_reason="length",
            usage=Usage(input_tokens=10, output_tokens=512),
            raw={"choices": [{"message": {
                "reasoning_content": '{"hypotheses":["a","b"]}'}}]},
        )
        backend = CallableBackend(fn=lambda _: response, revision="v1", callable_id="atd-test",
                                  capabilities=_CAPABILITIES)
        with unittest.TestCase().assertRaisesRegex(
            AlgorithmOutputError, "finish_reason=length; output_tokens=512; max_tokens=512"
        ):
            ActiveTaskDisambiguation(backend=backend).propose(
                FrozenAgentState(query="Ambiguous task",
                                 budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)),
                InterventionContext(),
            )
        assert len(backend.calls) == 1
        assert backend.calls[0].max_tokens == 512
