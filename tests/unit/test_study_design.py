from __future__ import annotations

import unittest

from agentinputlab.contracts import BalanceError, DependencyBinding, DependencyRole, ProfileKind
from agentinputlab.runner import ComparisonProfile
from agentinputlab.studies import StudyArmSpec, build_study_plan


def binding(
    slot: str,
    identity: str,
    role: DependencyRole = DependencyRole.NUISANCE,
) -> DependencyBinding:
    return DependencyBinding(slot, role, identity, "v1")


class StudyDesignTests(unittest.TestCase):
    def test_controlled_plan_is_deterministic_and_complete(self) -> None:
        profile = ComparisonProfile(
            "controlled",
            ProfileKind.CONTROLLED_ADAPTER,
            required_slots=("model",),
        )
        arms = (
            StudyArmSpec("noop", "baseline.no_op", (binding("model", "m"),)),
            StudyArmSpec("method", "method.v1", (binding("model", "m"),)),
        )
        first = build_study_plan(
            study_id="s",
            profile=profile,
            benchmark_id="b",
            example_ids=("a", "b"),
            seeds=(1, 2, 3),
            arms=arms,
        )
        second = build_study_plan(
            study_id="s",
            profile=profile,
            benchmark_id="b",
            example_ids=("a", "b"),
            seeds=(1, 2, 3),
            arms=arms,
        )
        self.assertEqual(len(first.cells), 6)
        self.assertEqual(first.plan_hash, second.plan_hash)

    def test_controlled_plan_accepts_explicit_treatment_varied_slots(self) -> None:
        profile = ComparisonProfile(
            "controlled-treatment",
            ProfileKind.CONTROLLED_ADAPTER,
            required_slots=("model",),
            varied_slots=("query_optimizer",),
        )
        plan = build_study_plan(
            study_id="controlled-treatment",
            profile=profile,
            benchmark_id="b",
            example_ids=("a",),
            seeds=(1,),
            arms=(
                StudyArmSpec(
                    "noop",
                    "baseline.no_op",
                    (
                        binding("model", "m"),
                        binding(
                            "query_optimizer",
                            "identity",
                            DependencyRole.TREATMENT,
                        ),
                    ),
                ),
                StudyArmSpec(
                    "method",
                    "method.v1",
                    (
                        binding("model", "m"),
                        binding(
                            "query_optimizer",
                            "method.v1",
                            DependencyRole.TREATMENT,
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(len(plan.cells), 1)
        self.assertEqual(len(plan.cells[0].arms), 2)

    def test_controlled_plan_rejects_missing_varied_slot_in_one_arm(self) -> None:
        profile = ComparisonProfile(
            "controlled-missing-treatment",
            ProfileKind.CONTROLLED_ADAPTER,
            required_slots=("model",),
            varied_slots=("query_optimizer",),
        )
        with self.assertRaisesRegex(BalanceError, "query_optimizer missing in noop"):
            build_study_plan(
                study_id="controlled-missing-treatment",
                profile=profile,
                benchmark_id="b",
                example_ids=("a",),
                seeds=(1,),
                arms=(
                    StudyArmSpec(
                        "noop",
                        "baseline.no_op",
                        (binding("model", "m"),),
                    ),
                    StudyArmSpec(
                        "method",
                        "method.v1",
                        (
                            binding("model", "m"),
                            binding(
                                "query_optimizer",
                                "method.v1",
                                DependencyRole.TREATMENT,
                            ),
                        ),
                    ),
                ),
            )

    def test_factorial_plan_crosses_every_declared_level(self) -> None:
        profile = ComparisonProfile(
            "factorial",
            ProfileKind.FACTORIAL_ADAPTER,
            required_slots=("corpus",),
            varied_slots=("retriever",),
        )
        plan = build_study_plan(
            study_id="factorial",
            profile=profile,
            benchmark_id="b",
            example_ids=("a",),
            seeds=(7,),
            arms=(
                StudyArmSpec(
                    "noop",
                    "baseline.no_op",
                    (binding("corpus", "c"),),
                ),
                StudyArmSpec(
                    "method",
                    "method.v1",
                    (binding("corpus", "c"),),
                ),
            ),
            factorial_levels={
                "retriever": (
                    binding("retriever", "bm25", DependencyRole.CROSSED),
                    binding("retriever", "dense", DependencyRole.CROSSED),
                )
            },
        )
        self.assertEqual(len(plan.cells), 1)
        self.assertEqual(len(plan.cells[0].arms), 4)

    def test_factorial_levels_cannot_be_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_study_plan(
                study_id="bad",
                profile=ComparisonProfile(
                    "factorial",
                    ProfileKind.FACTORIAL_ADAPTER,
                    required_slots=(),
                    varied_slots=("retriever",),
                ),
                benchmark_id="b",
                example_ids=("a",),
                seeds=(1,),
                arms=(
                    StudyArmSpec("a", "a", ()),
                    StudyArmSpec("b", "b", ()),
                ),
            )

    def test_factorial_levels_cannot_overwrite_native_arm_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "silently replace"):
            build_study_plan(
                study_id="bad-overwrite",
                profile=ComparisonProfile(
                    "factorial",
                    ProfileKind.FACTORIAL_ADAPTER,
                    required_slots=(),
                    varied_slots=("retriever",),
                ),
                benchmark_id="b",
                example_ids=("a",),
                seeds=(1,),
                arms=(
                    StudyArmSpec(
                        "a",
                        "a",
                        (binding("retriever", "native", DependencyRole.TREATMENT),),
                    ),
                    StudyArmSpec("b", "b", ()),
                ),
                factorial_levels={
                    "retriever": (binding("retriever", "bm25", DependencyRole.CROSSED),)
                },
            )


if __name__ == "__main__":
    unittest.main()
