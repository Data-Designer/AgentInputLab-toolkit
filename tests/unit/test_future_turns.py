from __future__ import annotations

import json
import unittest

from agentinputlab.algorithms.clarify import FutureTurnClarifier, FutureTurnControls
from agentinputlab.backends.llm import ModelCapability, ReplayBackend
from agentinputlab.contracts import FrozenAgentState, InterventionContext, PatchOp


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


def test_future_turns_selects_question_with_better_simulated_outcomes() -> None:
    backend = ReplayBackend(
        [
            _response(
                {
                    "interpretations": ["personal account", "business account"],
                    "direct_answer": "Use the default flow.",
                    "direct_supported_ids": [0],
                }
            ),
            _response({"questions": ["Which account type?", "Which browser?"]}),
            _response(
                {
                    "outcomes": [
                        {
                            "interpretation_id": 0,
                            "clarification_answer": "Personal.",
                            "tailored_response": "Use personal settings.",
                            "abstain": False,
                        },
                        {
                            "interpretation_id": 1,
                            "clarification_answer": "Business.",
                            "tailored_response": "Use business settings.",
                            "abstain": False,
                        },
                    ]
                }
            ),
            _response(
                {
                    "outcomes": [
                        {
                            "interpretation_id": 0,
                            "clarification_answer": "Unknown.",
                            "tailored_response": "",
                            "abstain": True,
                        },
                        {
                            "interpretation_id": 1,
                            "clarification_answer": "Unknown.",
                            "tailored_response": "",
                            "abstain": True,
                        },
                    ]
                }
            ),
        ],
        capabilities=_CAPABILITIES,
    )
    patch = FutureTurnClarifier(backend=backend).propose(
        FrozenAgentState(query="Configure my account"),
        InterventionContext(seed=17),
    )
    assert patch.operations[0].op is PatchOp.REQUEST_CLARIFICATION
    assert patch.operations[0].value == "Which account type?"
    assert patch.trace.counters["model_calls"] == 4
    assert patch.metadata["official_checkpoint_used"] is False


def test_future_turns_keeps_direct_response_when_it_covers_all_interpretations() -> None:
    backend = ReplayBackend(
        [
            _response(
                {
                    "interpretations": ["one clear intent"],
                    "direct_answer": "Proceed safely.",
                    "direct_supported_ids": [0],
                }
            )
        ],
        capabilities=_CAPABILITIES,
    )
    patch = FutureTurnClarifier(backend=backend).propose(
        FrozenAgentState(query="Do the clear task"), InterventionContext(seed=2)
    )
    assert patch.operations[0].op is PatchOp.NO_OP
    assert patch.trace.counters["model_calls"] == 1


def test_future_turn_controls_fail_closed() -> None:
    with unittest.TestCase().assertRaisesRegex(Exception, "max_candidates"):
        FutureTurnControls(max_candidates=0)


def test_future_turns_derives_abstention_instead_of_trusting_model_flag() -> None:
    backend = ReplayBackend(
        [
            _response(
                {
                    "interpretations": ["personal", "business"],
                    "direct_answer": "Need more information.",
                    "direct_supported_ids": [],
                }
            ),
            _response({"questions": ["Which account type?"]}),
            _response(
                {
                    "outcomes": [
                        {
                            "interpretation_id": 0,
                            "clarification_answer": "Personal.",
                            "tailored_response": "Use personal settings.",
                            "abstain": "not-a-trusted-protocol-field",
                        },
                        {
                            "interpretation_id": 1,
                            "clarification_answer": "Unknown.",
                            "tailored_response": "",
                            "abstain": False,
                        },
                    ]
                }
            ),
        ],
        capabilities=_CAPABILITIES,
    )
    patch = FutureTurnClarifier(
        backend=backend,
        controls=FutureTurnControls(max_candidates=1),
    ).propose(FrozenAgentState(query="Configure account"), InterventionContext())
    assert patch.operations[0].op is PatchOp.REQUEST_CLARIFICATION
