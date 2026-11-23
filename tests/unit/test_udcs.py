from __future__ import annotations

import unittest

from agentinputlab.algorithms.clarify.udcs import (
    UDCSPromptNativePolicy,
    parse_udcs_response,
)
from agentinputlab.backends.llm import ReplayBackend
from agentinputlab.contracts import (
    AlgorithmOutputError,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
    ValidationError,
)
from agentinputlab.runner import InterventionRunner


def response(*, u_request="0.75", action="request_clarification", confidence="0.8"):
    return (
        "<think>There are several valid choices.</think>"
        f"<u_request>{u_request}</u_request>"
        "<u_request_explanation>The user did not specify a preference.</u_request_explanation>"
        f"<action>{action}</action>"
        f"<confidence>{confidence}</confidence>"
        "<explanation>The next step depends on that preference.</explanation>"
    )


class UDCSParserTests(unittest.TestCase):
    def test_parses_all_six_fields(self):
        decision = parse_udcs_response(response())
        self.assertEqual(decision.think, "There are several valid choices.")
        self.assertEqual(decision.u_request, 0.75)
        self.assertEqual(decision.action, "request_clarification")
        self.assertEqual(decision.confidence, 0.8)
        self.assertTrue(decision.requests_clarification)

    def test_repository_probability_clamp_is_preserved(self):
        decision = parse_udcs_response(response(u_request="1.4", confidence="9.0"))
        self.assertEqual(decision.u_request, 1.0)
        self.assertEqual(decision.confidence, 1.0)
        self.assertEqual(decision.clamped_fields, ("u_request", "confidence"))

    def test_missing_required_probability_fails(self):
        with self.assertRaisesRegex(AlgorithmOutputError, "u_request"):
            parse_udcs_response(
                "<think>x</think><action>search[x]</action><confidence>0.2</confidence>"
            )

    def test_optional_explanations_match_upstream_empty_behavior(self):
        decision = parse_udcs_response(
            "<u_request>0.2</u_request><action>search[x]</action><confidence>0.7</confidence>"
        )
        self.assertEqual(decision.think, "")
        self.assertEqual(decision.explanation, "")


class UDCSPolicyTests(unittest.TestCase):
    def test_native_profile_does_not_enforce_threshold_in_python(self):
        policy = UDCSPromptNativePolicy(
            backend=ReplayBackend([response(u_request="0.9", action="search[red shoes]")]),
            clarification_threshold=0.5,
        )
        patch = policy.propose(
            FrozenAgentState(query="Buy shoes", tool_names=("search[keywords]",)),
            InterventionContext(),
        )
        self.assertIs(patch.operations[0].op, PatchOp.NO_OP)
        self.assertFalse(patch.metadata["threshold_enforced"])
        parsed = patch.trace.events[1].payload
        self.assertTrue(parsed["threshold_disagreement"])

    def test_model_selected_clarification_materializes_request(self):
        policy = UDCSPromptNativePolicy(backend=ReplayBackend([response()]))
        state = FrozenAgentState(query="Buy shoes")
        patch = policy.propose(state, InterventionContext())
        self.assertIs(patch.operations[0].op, PatchOp.REQUEST_CLARIFICATION)
        materialized = patch.apply(state)
        self.assertIn("preference", materialized.metadata["clarification_request"])

    def test_runner_requires_backend_via_observed_context(self):
        state = FrozenAgentState(query="Buy shoes")
        with self.assertRaisesRegex(ValidationError, "InterventionContext"):
            InterventionRunner().run(
                UDCSPromptNativePolicy(backend=ReplayBackend([response()])),
                state,
                InterventionContext(),
            )

        record = InterventionRunner().run(
            UDCSPromptNativePolicy(),
            state,
            InterventionContext(backend=ReplayBackend([response()])),
        )
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(record.usage.unpriced_model_calls, 1)


if __name__ == "__main__":
    unittest.main()
