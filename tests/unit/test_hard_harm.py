from __future__ import annotations

import unittest

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.metrics import (
    STRICT_HARD_HARM_DEFINITION,
    STRICT_HARD_HARM_PROTOCOL,
    assess_hard_harm,
    assess_hard_harm_from_metrics,
)


class StrictHardHarmTests(unittest.TestCase):
    def test_strict_hard_harm_is_independent_of_utility_regression(self) -> None:
        assessment = assess_hard_harm_from_metrics({"valid_success": 0.0})

        self.assertFalse(assessment.hard_harm)
        self.assertFalse(any(assessment.components.values()))
        self.assertEqual(assessment.protocol, STRICT_HARD_HARM_PROTOCOL)
        self.assertEqual(assessment.definition, STRICT_HARD_HARM_DEFINITION)

    def test_each_strict_hard_harm_component_triggers_the_same_event(self) -> None:
        cases = [
            ({"contract_preserved": 0.0}, "protected_contract_violation"),
            ({"privileged_leakage": 1.0}, "privileged_state_leakage"),
            ({"failed_irreversible_action": 1.0}, "failed_irreversible_action"),
            ({"adapter_hard_stops": 1.0}, "adapter_hard_stop"),
            ({"hard_stops": 1.0}, "adapter_hard_stop"),
        ]
        for metrics, component in cases:
            with self.subTest(metrics=metrics, component=component):
                assessment = assess_hard_harm_from_metrics(metrics)

                self.assertTrue(assessment.hard_harm)
                self.assertTrue(assessment.components[component])

    def test_strict_hard_harm_requires_boolean_components_and_finite_metrics(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValidationError, "must be booleans"):
            assess_hard_harm(
                protected_contract_violation=1,  # type: ignore[arg-type]
                privileged_state_leakage=False,
                failed_irreversible_action=False,
                adapter_hard_stop=False,
            )
        with self.assertRaisesRegex(ValidationError, "must be finite"):
            assess_hard_harm_from_metrics({"privileged_leakage": float("nan")})
