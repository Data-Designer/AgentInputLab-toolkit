from __future__ import annotations

import unittest
from dataclasses import dataclass

from agentinputlab.algorithms.acquire.hyde import HyDECleanRoom, HyDEConfig
from agentinputlab.algorithms.clarify.udcs import UDCSPromptNativePolicy
from agentinputlab.backends.llm import ReplayBackend
from agentinputlab.backends.retrieval import FunctionRetriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    DependencyBinding,
    DependencyRole,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ProfileKind,
    SemanticFamily,
    ValidationError,
)
from agentinputlab.contracts.errors import BalanceError
from agentinputlab.observability import BudgetLedger, ObservedRetriever, RunMonitor
from agentinputlab.runner import ComparisonArm, ComparisonProfile
from agentinputlab.selectors import NoOpIntervention
from agentinputlab.studies import PairedArm, PairedStudyRunner


def shared_bindings() -> tuple[DependencyBinding, ...]:
    return (
        DependencyBinding("corpus", DependencyRole.NUISANCE, "tiny", "v1"),
        DependencyBinding("retriever", DependencyRole.NUISANCE, "bm25", "v1"),
    )


class PairedStudyTests(unittest.TestCase):
    def _context(self, retriever_revision: str = "v1") -> InterventionContext:
        retriever = FunctionRetriever(
            lambda _query, _top_k: (),
            identity={"implementation": "fixture", "revision": retriever_revision},
        )
        return InterventionContext(
            retriever=retriever,
            run_config={"dependency_fingerprints": {"corpus": {"id": "tiny", "revision": "v1"}}},
        )

    def test_same_frozen_state_is_used_for_each_arm(self) -> None:
        bindings = shared_bindings()
        arms = (
            PairedArm(
                ComparisonArm("noop", "baseline.no_op", bindings),
                NoOpIntervention(),
                self._context(),
            ),
            PairedArm(
                ComparisonArm("alt", "baseline.alternate_no_op", bindings),
                NoOpIntervention(
                    method_id="alternate_no_op",
                    release_id="baseline.alternate_no_op",
                ),
                self._context(),
            ),
        )
        run = PairedStudyRunner().run(
            FrozenAgentState(query="q"),
            ComparisonProfile(
                "paired",
                ProfileKind.CONTROLLED_ADAPTER,
                required_slots=("corpus", "retriever"),
            ),
            arms,
        )
        self.assertTrue(run.balance.eligible)
        self.assertEqual(len({record.input_state_hash for record in run.records}), 1)
        self.assertIn("retriever", run.observed_dependency_hashes["noop"])

    def test_declared_equal_retrievers_cannot_hide_different_runtime_objects(self) -> None:
        bindings = shared_bindings()
        arms = (
            PairedArm(
                ComparisonArm("left", "baseline.no_op", bindings),
                NoOpIntervention(),
                self._context("v1"),
            ),
            PairedArm(
                ComparisonArm("right", "baseline.alternate_no_op", bindings),
                NoOpIntervention(
                    method_id="alternate_no_op",
                    release_id="baseline.alternate_no_op",
                ),
                self._context("v2"),
            ),
        )
        with self.assertRaisesRegex(BalanceError, "retriever runtime differs"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q"),
                ComparisonProfile(
                    "paired",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("corpus", "retriever"),
                ),
                arms,
            )

    def test_shared_runtime_object_is_rejected_even_with_same_fingerprint(self) -> None:
        bindings = shared_bindings()
        shared_context = self._context()
        arms = (
            PairedArm(
                ComparisonArm("left", "baseline.no_op", bindings),
                NoOpIntervention(),
                shared_context,
            ),
            PairedArm(
                ComparisonArm("right", "baseline.alternate_no_op", bindings),
                NoOpIntervention(
                    method_id="alternate_no_op",
                    release_id="baseline.alternate_no_op",
                ),
                shared_context,
            ),
        )
        with self.assertRaisesRegex(BalanceError, "independently constructed"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q"),
                ComparisonProfile(
                    "paired",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("corpus", "retriever"),
                ),
                arms,
            )

    def test_distinct_observed_wrappers_cannot_hide_a_shared_inner_runtime(self) -> None:
        bindings = shared_bindings()
        inner = FunctionRetriever(
            lambda _query, _top_k: (),
            identity={"implementation": "fixture", "revision": "v1"},
        )

        def wrap_twice(label: str) -> ObservedRetriever:
            inner_monitor = RunMonitor()
            outer_monitor = RunMonitor()
            inner_wrapper = ObservedRetriever(
                inner,
                inner_monitor,
                BudgetLedger(ExecutionBudget()),
                f"{label}-inner",
            )
            return ObservedRetriever(
                inner_wrapper,  # type: ignore[arg-type]
                outer_monitor,
                BudgetLedger(ExecutionBudget()),
                f"{label}-outer",
            )

        arms = (
            PairedArm(
                ComparisonArm("left", "baseline.no_op", bindings),
                NoOpIntervention(),
                InterventionContext(retriever=wrap_twice("left")),
            ),
            PairedArm(
                ComparisonArm("right", "baseline.alternate_no_op", bindings),
                NoOpIntervention(
                    method_id="alternate_no_op",
                    release_id="baseline.alternate_no_op",
                ),
                InterventionContext(retriever=wrap_twice("right")),
            ),
        )
        with self.assertRaisesRegex(BalanceError, "independently constructed"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q"),
                ComparisonProfile(
                    "paired",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("corpus", "retriever"),
                ),
                arms,
            )

    def test_distinct_custom_proxies_cannot_hide_a_shared_inner_backend(self) -> None:
        class ProxyBackend:
            def __init__(self, inner) -> None:
                self.target = inner

            @property
            def model_id(self):
                return self.target.model_id

            @property
            def capabilities(self):
                return self.target.capabilities

            def generate(self, request):
                return self.target.generate(request)

            def fingerprint(self):
                return self.target.fingerprint()

        inner = ReplayBackend(("one", "two"))
        backend_binding = DependencyBinding("backend", DependencyRole.NUISANCE, "fixture", "v1")
        bindings = (backend_binding,)
        arms = (
            PairedArm(
                ComparisonArm("left", "baseline.no_op", bindings),
                NoOpIntervention(),
                InterventionContext(backend=ProxyBackend(inner)),
            ),
            PairedArm(
                ComparisonArm("right", "baseline.alternate_no_op", bindings),
                NoOpIntervention(
                    method_id="alternate_no_op",
                    release_id="baseline.alternate_no_op",
                ),
                InterventionContext(backend=ProxyBackend(inner)),
            ),
        )

        with self.assertRaisesRegex(BalanceError, "proxy inners"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q"),
                ComparisonProfile(
                    "paired-proxy",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("backend",),
                ),
                arms,
            )
        self.assertEqual(inner._calls, 0)

    def test_budget_ledger_cannot_be_shared_across_arms(self) -> None:
        bindings = shared_bindings()
        ledger = BudgetLedger(ExecutionBudget())
        arms = (
            PairedArm(
                ComparisonArm("left", "baseline.no_op", bindings),
                NoOpIntervention(),
                InterventionContext(
                    retriever=self._context().retriever,
                    budget_ledger=ledger,
                    run_config={
                        "dependency_fingerprints": {"corpus": {"id": "tiny", "revision": "v1"}}
                    },
                ),
            ),
            PairedArm(
                ComparisonArm("right", "baseline.alternate_no_op", bindings),
                NoOpIntervention(
                    method_id="alternate_no_op",
                    release_id="baseline.alternate_no_op",
                ),
                InterventionContext(
                    retriever=self._context().retriever,
                    budget_ledger=ledger,
                    run_config={
                        "dependency_fingerprints": {"corpus": {"id": "tiny", "revision": "v1"}}
                    },
                ),
            ),
        )
        with self.assertRaisesRegex(BalanceError, "BudgetLedger"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q"),
                ComparisonProfile(
                    "paired",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("corpus", "retriever"),
                ),
                arms,
            )

    def test_one_arm_failure_is_recorded_without_skipping_other_arms(self) -> None:
        @dataclass(frozen=True, slots=True)
        class FailingIntervention:
            method_id: str = "fixture.fail"
            release_id: str = "fixture.fail.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                raise RuntimeError("expected fixture failure")

        bindings = shared_bindings()
        run = PairedStudyRunner().run(
            FrozenAgentState(query="q"),
            ComparisonProfile(
                "paired",
                ProfileKind.CONTROLLED_ADAPTER,
                required_slots=("corpus", "retriever"),
            ),
            (
                PairedArm(
                    ComparisonArm("failed", "fixture.fail.v1", bindings),
                    FailingIntervention(),
                    self._context(),
                ),
                PairedArm(
                    ComparisonArm("healthy", "baseline.no_op", bindings),
                    NoOpIntervention(),
                    self._context(),
                ),
            ),
        )
        self.assertEqual([outcome.status for outcome in run.outcomes], ["failed", "succeeded"])
        self.assertEqual(len(run.records), 1)
        self.assertEqual(run.outcomes[0].failure.category, "RuntimeError")

    def test_malformed_intervention_return_fails_only_its_arm(self) -> None:
        @dataclass(frozen=True, slots=True)
        class MalformedOutputIntervention:
            output_kind: str
            method_id: str = "fixture.malformed_output"
            release_id: str = "fixture.malformed_output.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                if self.output_kind == "none":
                    return None
                if self.output_kind == "non_patch":
                    return {"query": state.query}
                if self.output_kind == "wrong_method":
                    return NoOpIntervention(
                        method_id="fixture.other_method",
                        release_id=self.release_id,
                    ).propose(state, context)
                if self.output_kind == "wrong_release":
                    return NoOpIntervention(
                        method_id=self.method_id,
                        release_id="fixture.other_release.v1",
                    ).propose(state, context)
                raise AssertionError(f"unknown output kind: {self.output_kind}")

        bindings = shared_bindings()
        for output_kind, expected_message in (
            ("none", "must return an InputPatch"),
            ("non_patch", "must return an InputPatch"),
            ("wrong_method", "another method"),
            ("wrong_release", "another release"),
        ):
            with self.subTest(output_kind=output_kind):
                run = PairedStudyRunner().run(
                    FrozenAgentState(query="q"),
                    ComparisonProfile(
                        "paired-malformed-output",
                        ProfileKind.CONTROLLED_ADAPTER,
                        required_slots=("corpus", "retriever"),
                    ),
                    (
                        PairedArm(
                            ComparisonArm(
                                "malformed",
                                "fixture.malformed_output.v1",
                                bindings,
                            ),
                            MalformedOutputIntervention(output_kind),
                            self._context(),
                        ),
                        PairedArm(
                            ComparisonArm("healthy", "baseline.no_op", bindings),
                            NoOpIntervention(),
                            self._context(),
                        ),
                    ),
                )

                self.assertEqual(
                    [outcome.status for outcome in run.outcomes],
                    ["failed", "succeeded"],
                )
                failure = run.outcomes[0].failure
                self.assertIsNotNone(failure)
                assert failure is not None
                self.assertEqual(failure.category, AlgorithmOutputError.__name__)
                self.assertIn(expected_message, failure.message)

    def test_algorithm_failure_is_redacted_and_linked_to_its_monitor_trace(self) -> None:
        @dataclass(frozen=True, slots=True)
        class SecretFailure:
            method_id: str = "fixture.secret_fail"
            release_id: str = "fixture.secret_fail.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                raise RuntimeError(
                    "Bearer abcdefghijk OPENAI_API_KEY=topsecretvalue "
                    "sk-anothersecret eyJabcdef.abcdefgh.ijklmnop"
                )

        monitor = RunMonitor()
        bindings = shared_bindings()
        run = PairedStudyRunner().run(
            FrozenAgentState(query="q"),
            ComparisonProfile(
                "paired",
                ProfileKind.CONTROLLED_ADAPTER,
                required_slots=("corpus", "retriever"),
            ),
            (
                PairedArm(
                    ComparisonArm("failed", "fixture.secret_fail.v1", bindings),
                    SecretFailure(),
                    InterventionContext(
                        retriever=self._context().retriever,
                        monitor=monitor,
                        run_config={
                            "dependency_fingerprints": {"corpus": {"id": "tiny", "revision": "v1"}}
                        },
                    ),
                ),
                PairedArm(
                    ComparisonArm("healthy", "baseline.no_op", bindings),
                    NoOpIntervention(),
                    self._context(),
                ),
            ),
        )
        failure = run.outcomes[0].failure
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertTrue(failure.monitor_run_id)
        self.assertEqual(monitor.snapshot(failure.monitor_run_id).run_id, failure.monitor_run_id)
        self.assertNotIn("topsecretvalue", failure.message)
        self.assertNotIn("anothersecret", failure.message)
        self.assertNotIn("abcdefghijk", failure.message)
        self.assertNotIn("eyJabcdef", failure.message)
        self.assertGreaterEqual(failure.message.count("<redacted>"), 4)

    def test_udcs_model_output_validation_fails_only_its_arm(self) -> None:
        malformed_response = (
            "<think>missing uncertainty</think>"
            "<action>search[x]</action>"
            "<confidence>0.2</confidence>"
        )
        bindings = (DependencyBinding("backend", DependencyRole.NUISANCE, "replay", "fixture-v1"),)

        def context() -> InterventionContext:
            return InterventionContext(backend=ReplayBackend((malformed_response,)))

        run = PairedStudyRunner().run(
            FrozenAgentState(query="q"),
            ComparisonProfile(
                "paired-udcs-output",
                ProfileKind.CONTROLLED_ADAPTER,
                required_slots=("backend",),
            ),
            (
                PairedArm(
                    ComparisonArm(
                        "malformed",
                        "clarify.udcs.repo_native_prompt_route",
                        bindings,
                    ),
                    UDCSPromptNativePolicy(),
                    context(),
                ),
                PairedArm(
                    ComparisonArm("healthy", "baseline.no_op", bindings),
                    NoOpIntervention(),
                    context(),
                ),
            ),
        )

        self.assertEqual([outcome.status for outcome in run.outcomes], ["failed", "succeeded"])
        failure = run.outcomes[0].failure
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure.category, AlgorithmOutputError.__name__)
        self.assertIn("u_request", failure.message)

    def test_hyde_embedding_output_validation_fails_only_its_arm(self) -> None:
        class InvalidEmbedder:
            embedder_id = "fixture-embedder"

            def embed(self, texts):
                return ((1.0, 2.0), (3.0,))

            def fingerprint(self):
                return {"embedder_id": self.embedder_id, "revision": "fixture-v1"}

        class FixtureSearcher:
            searcher_id = "fixture-vector-index"

            def search(self, vector, *, top_k):
                return ()

            def fingerprint(self):
                return {"searcher_id": self.searcher_id, "revision": "fixture-v1"}

        bindings = tuple(
            DependencyBinding(slot, DependencyRole.NUISANCE, identity, "fixture-v1")
            for slot, identity in (
                ("backend", "replay"),
                ("dense_embedder", "fixture-embedder"),
                ("vector_searcher", "fixture-vector-index"),
            )
        )

        def context() -> InterventionContext:
            return InterventionContext(
                backend=ReplayBackend(("hypothesis",)),
                dense_embedder=InvalidEmbedder(),
                vector_searcher=FixtureSearcher(),
            )

        run = PairedStudyRunner().run(
            FrozenAgentState(query="q"),
            ComparisonProfile(
                "paired-hyde-output",
                ProfileKind.CONTROLLED_ADAPTER,
                required_slots=("backend", "dense_embedder", "vector_searcher"),
            ),
            (
                PairedArm(
                    ComparisonArm(
                        "malformed",
                        "acquire.hyde.paper_reimplementation",
                        bindings,
                    ),
                    HyDECleanRoom(config=HyDEConfig(num_hypotheses=1)),
                    context(),
                ),
                PairedArm(
                    ComparisonArm("healthy", "baseline.no_op", bindings),
                    NoOpIntervention(),
                    context(),
                ),
            ),
        )

        self.assertEqual([outcome.status for outcome in run.outcomes], ["failed", "succeeded"])
        failure = run.outcomes[0].failure
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure.category, AlgorithmOutputError.__name__)
        self.assertIn("inconsistent dimensions", failure.message)

    def test_artifact_and_configuration_errors_invalidate_the_cell(self) -> None:
        @dataclass(frozen=True, slots=True)
        class InvalidCellIntervention:
            error: Exception
            method_id: str = "fixture.invalid"
            release_id: str = "fixture.invalid.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                raise self.error

        bindings = shared_bindings()
        for error in (
            ArtifactError("artifact checksum mismatch"),
            ValidationError("invalid frozen configuration"),
            ExternalDependencyError("upstream worker is unavailable"),
        ):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                PairedStudyRunner().run(
                    FrozenAgentState(query="q"),
                    ComparisonProfile(
                        "paired",
                        ProfileKind.CONTROLLED_ADAPTER,
                        required_slots=("corpus", "retriever"),
                    ),
                    (
                        PairedArm(
                            ComparisonArm("invalid", "fixture.invalid.v1", bindings),
                            InvalidCellIntervention(error),
                            self._context(),
                        ),
                        PairedArm(
                            ComparisonArm("healthy", "baseline.no_op", bindings),
                            NoOpIntervention(),
                            self._context(),
                        ),
                    ),
                )

    def test_telemetry_sink_failure_invalidates_the_cell(self) -> None:
        class FailingSink:
            def write(self, event) -> None:
                raise OSError("telemetry storage unavailable")

        bindings = shared_bindings()
        monitor = RunMonitor(sinks=(FailingSink(),))
        with self.assertRaisesRegex(OSError, "telemetry storage unavailable"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q"),
                ComparisonProfile(
                    "paired",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("corpus", "retriever"),
                ),
                (
                    PairedArm(
                        ComparisonArm("invalid", "baseline.no_op", bindings),
                        NoOpIntervention(),
                        InterventionContext(
                            retriever=self._context().retriever,
                            monitor=monitor,
                            run_config={
                                "dependency_fingerprints": {
                                    "corpus": {"id": "tiny", "revision": "v1"}
                                }
                            },
                        ),
                    ),
                    PairedArm(
                        ComparisonArm("healthy", "baseline.alternate_no_op", bindings),
                        NoOpIntervention(
                            method_id="alternate_no_op",
                            release_id="baseline.alternate_no_op",
                        ),
                        self._context(),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
