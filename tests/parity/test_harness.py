from __future__ import annotations

import unittest

from agentinputlab.parity import ParityCase, ParityMode, ParityRunner


class ParityHarnessTests(unittest.TestCase):
    def runner(self, mode: ParityMode) -> ParityRunner:
        return ParityRunner(
            method_id="fixture",
            release_id="fixture.v1",
            mode=mode,
            reference_identity="upstream@0123456789abcdef",
            candidate_identity="adapter@fedcba9876543210",
        )

    def test_exact_parity_uses_canonical_hashes(self) -> None:
        report = self.runner(ParityMode.FIXTURE_EXACT).run(
            (ParityCase("a", {"value": 1}, {"b": 2, "a": 1}),),
            lambda _payload: {"a": 1, "b": 2},
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.passed_count, 1)
        self.assertEqual(len(report.report_hash), 64)

    def test_numeric_tolerance_and_failure_gate(self) -> None:
        report = self.runner(ParityMode.NUMERIC_RESULT).run(
            (
                ParityCase(
                    "within",
                    {"actual": 0.805},
                    0.8,
                    absolute_tolerance=0.01,
                ),
                ParityCase(
                    "outside",
                    {"actual": 0.9},
                    0.8,
                    absolute_tolerance=0.01,
                ),
            ),
            lambda payload: payload["actual"],
        )
        self.assertFalse(report.passed)
        with self.assertRaisesRegex(AssertionError, "outside"):
            report.require_passed()

    def test_direction_parity(self) -> None:
        report = self.runner(ParityMode.DIRECTION).run(
            (
                ParityCase("positive", {"actual": 0.7}, expected=0.8, baseline=0.5),
                ParityCase("negative", {"actual": 0.4}, expected=0.3, baseline=0.5),
            ),
            lambda payload: payload["actual"],
        )
        self.assertTrue(report.passed)

    def test_direction_requires_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.runner(ParityMode.DIRECTION).run(
                (ParityCase("x", {}, expected=1.0),),
                lambda _payload: 1.0,
            )

    def test_expected_fixture_is_frozen_against_caller_mutation(self) -> None:
        expected = {"value": [1, 2]}
        case = ParityCase("frozen", {}, expected)
        expected["value"].append(3)
        report = self.runner(ParityMode.FIXTURE_EXACT).run(
            (case,),
            lambda _payload: {"value": [1, 2, 3]},
        )
        self.assertFalse(report.passed)

    def test_unbounded_tolerance_and_unpinned_identity_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            ParityCase("x", {}, 1.0, absolute_tolerance=float("inf"))
        with self.assertRaisesRegex(ValueError, "pinned"):
            ParityRunner(
                method_id="fixture",
                release_id="fixture.v1",
                mode=ParityMode.FIXTURE_EXACT,
                reference_identity="upstream@latest",
                candidate_identity="adapter@fedcba9876543210",
            )


if __name__ == "__main__":
    unittest.main()
