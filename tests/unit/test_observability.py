from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from agentinputlab.backends.llm import (
    CallableBackend,
    GenerationRequest,
    GenerationResponse,
    ReplayBackend,
    Usage,
)
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.connectors.tools import InMemoryToolConnector, ToolCall, ToolSpec
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    IntentContract,
    IntentViolationError,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    ValidationError,
)
from agentinputlab.contracts.errors import BudgetExceededError
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from agentinputlab.observability import (
    BudgetLedger,
    ComputePrice,
    EventKind,
    JsonlEventSink,
    ModelPrice,
    ObservedLLMBackend,
    ObservedRunStatus,
    PricingCatalog,
    ResourceUsage,
    RunMonitor,
    normalize_external_usage,
)
from agentinputlab.runner import InterventionRunner, RunStatus


@dataclass(frozen=True, slots=True)
class BackendRewrite:
    method_id: str = "fixture.rewrite"
    release_id: str = "fixture.rewrite.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        response = context.backend.generate(GenerationRequest(prompt=state.query))
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.REPLACE_QUERY, response.text),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="test-fixture",
            profile_kind="fixture_unverified",
        )


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    method_id: str = "fixture.dependencies"
    release_id: str = "fixture.dependencies.v1"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        context.retriever.retrieve(state.query, top_k=1)
        context.tools.call(ToolCall("echo", {"value": state.query}, call_id="one"))
        return InputPatch.no_op(
            state,
            method_id=self.method_id,
            release_id=self.release_id,
        )


@dataclass(frozen=True, slots=True)
class ThreadedBackendRewrite:
    method_id: str = "fixture.threaded_rewrite"
    release_id: str = "fixture.threaded_rewrite.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(
                context.backend.generate,
                GenerationRequest(prompt=state.query),
            ).result()
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.REPLACE_QUERY, response.text),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="threaded-test-fixture",
            profile_kind="fixture_unverified",
        )


@dataclass(frozen=True, slots=True)
class CatchBackendBudget:
    calls: int = 1
    method_id: str = "fixture.catch_budget"
    release_id: str = "fixture.catch_budget.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        for _ in range(self.calls):
            # Simulate an upstream algorithm that catches the immediate
            # wrapper exception and attempts to return a normal result.
            with contextlib.suppress(BudgetExceededError):
                context.backend.generate(GenerationRequest(prompt=state.query))
        return InputPatch.no_op(
            state,
            method_id=self.method_id,
            release_id=self.release_id,
        )


@dataclass(frozen=True, slots=True)
class TraceUsageProbe:
    counters: Mapping[str, int | float]
    reports_budget_exhaustion: bool = False
    method_id: str = "fixture.trace_usage"
    release_id: str = "fixture.trace_usage.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.NO_OP),),
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                counters=self.counters,
            ),
            implementation_identity="trace-test-fixture",
            profile_kind="fixture_unverified",
            metadata={"budget_exhausted": self.reports_budget_exhaustion},
        )


@dataclass(frozen=True, slots=True)
class ExternalFailureProbe:
    method_id: str = "fixture.external_failure"
    release_id: str = "fixture.external_failure.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit="fixture-commit",
            state_hash=state.state_hash,
            payload={},
            minimum_reservation={"model_calls": 1},
        )
        context.external_runtime.call_controlled(request)
        raise AssertionError("declared failure did not terminate the intervention")


class DeclaredFailureWorker:
    worker_id = "declared-failure-fixture"

    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = dict(response)
        self.calls = 0

    def call(self, _payload: Mapping[str, object]) -> Mapping[str, object]:
        self.calls += 1
        return self.response

    def fingerprint(self) -> Mapping[str, object]:
        return {"worker_id": self.worker_id, "revision": "fixture-v1"}


class ObservabilityTests(unittest.TestCase):
    def test_external_usage_cost_semantics_fail_closed(self) -> None:
        unpriced = normalize_external_usage(
            {
                "model_calls": 2,
                "retrieval_calls": 1,
                "input_tokens": 10,
                "output_tokens": 4,
            }
        )
        self.assertEqual(unpriced.unpriced_model_calls, 2)

        priced = normalize_external_usage(
            {
                "model_calls": 2,
                "retrieval_calls": 1,
                "input_tokens": 10,
                "output_tokens": 4,
                "cost_usd": 0.25,
            }
        )
        self.assertEqual(priced.unpriced_model_calls, 0)
        self.assertEqual(priced.cost_usd, 0.25)

        with self.assertRaisesRegex(AlgorithmOutputError, "every model call unpriced"):
            normalize_external_usage(
                {
                    "model_calls": 2,
                    "retrieval_calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "unpriced_model_calls": 0,
                }
            )

    def test_pricing_is_manifest_supplied_and_exact(self) -> None:
        catalog = PricingCatalog({"pinned-model": ModelPrice(2.0, 8.0)})
        self.assertAlmostEqual(
            catalog.estimate("pinned-model", input_tokens=250_000, output_tokens=125_000),
            1.5,
        )
        self.assertEqual(catalog.estimate("unknown", input_tokens=10, output_tokens=10), 0.0)
        compute = ComputePrice(cpu_per_hour=0.1, accelerator_per_hour=1.9)
        self.assertAlmostEqual(
            compute.estimate(cpu_seconds=3600, accelerator_seconds=3600),
            2.0,
        )

    def test_budget_ledger_rejects_overage_without_corrupting_usage(self) -> None:
        ledger = BudgetLedger(ExecutionBudget(max_model_calls=1))
        ledger.reserve(ResourceUsage(model_calls=1))
        with self.assertRaisesRegex(BudgetExceededError, "model_calls"):
            ledger.reserve(ResourceUsage(model_calls=1))
        self.assertEqual(ledger.usage.model_calls, 1)
        self.assertEqual(ledger.actual_violations, ())
        self.assertEqual(ledger.reservation_rejections, ("model_calls",))
        with self.assertRaisesRegex(BudgetExceededError, "model_calls"):
            ledger.require_within_budget()

    def test_pending_reservations_can_be_committed_or_cancelled_atomically(self) -> None:
        reservation = ResourceUsage(model_calls=1, retrieval_calls=1)

        cancelled = BudgetLedger(ExecutionBudget())
        cancelled.reserve_pending(reservation)
        self.assertEqual(cancelled.usage, reservation)
        cancelled.cancel_pending(reservation)
        self.assertEqual(cancelled.usage, ResourceUsage())

        committed = BudgetLedger(ExecutionBudget())
        committed.reserve_pending(reservation)
        committed.charge_actual(ResourceUsage(input_tokens=4, output_tokens=2))
        committed.commit_pending(reservation)
        self.assertEqual(
            committed.usage,
            ResourceUsage(
                model_calls=1,
                retrieval_calls=1,
                input_tokens=4,
                output_tokens=2,
            ),
        )
        with self.assertRaisesRegex(ValidationError, "release more"):
            committed.cancel_pending(reservation)

    def test_declared_worker_failure_without_valid_usage_retains_reservation(self) -> None:
        for response in (
            {"status": "failed", "failure_code": "AFTER_MODEL_CRASH"},
            {
                "status": "failed",
                "failure_code": "AFTER_MODEL_CRASH",
                "usage": "malformed",
            },
        ):
            with self.subTest(response=response):
                runner = InterventionRunner()
                worker = DeclaredFailureWorker(response)
                with self.assertRaises(ExternalDependencyError):
                    runner.run(
                        ExternalFailureProbe(),
                        FrozenAgentState(query="q"),
                        InterventionContext(external_runtime=worker),
                    )
                snapshot = runner.monitor.list_snapshots()[0]
                self.assertEqual(worker.calls, 1)
                self.assertEqual(snapshot.usage.model_calls, 1)
                self.assertEqual(snapshot.usage.unpriced_model_calls, 1)

    def test_declared_worker_failure_cancels_only_explicit_zero_consumption(self) -> None:
        runner = InterventionRunner()
        worker = DeclaredFailureWorker(
            {
                "status": "failed",
                "failure_code": "PRE_EXECUTION_REJECTION",
                "consumed": False,
            }
        )
        with self.assertRaises(ExternalDependencyError):
            runner.run(
                ExternalFailureProbe(),
                FrozenAgentState(query="q"),
                InterventionContext(external_runtime=worker),
            )
        usage = runner.monitor.list_snapshots()[0].usage
        self.assertEqual(usage.model_calls, 0)
        self.assertEqual(usage.unpriced_model_calls, 0)

    def test_declared_worker_failure_records_valid_actual_usage(self) -> None:
        runner = InterventionRunner()
        worker = DeclaredFailureWorker(
            {
                "status": "failed",
                "failure_code": "POST_EXECUTION_FAILURE",
                "usage": {
                    "model_calls": 2,
                    "retrieval_calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 3,
                },
            }
        )
        with self.assertRaises(ExternalDependencyError):
            runner.run(
                ExternalFailureProbe(),
                FrozenAgentState(query="q"),
                InterventionContext(external_runtime=worker),
            )
        usage = runner.monitor.list_snapshots()[0].usage
        self.assertEqual(usage.model_calls, 2)
        self.assertEqual(usage.retrieval_calls, 1)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 3)

    def test_actual_overage_is_recorded_and_sticky(self) -> None:
        ledger = BudgetLedger(ExecutionBudget(max_output_tokens=1))
        with self.assertRaisesRegex(BudgetExceededError, "actual.*output_tokens"):
            ledger.charge_actual(ResourceUsage(output_tokens=2))
        self.assertEqual(ledger.usage.output_tokens, 2)
        self.assertEqual(ledger.actual_violations, ("output_tokens",))
        with self.assertRaisesRegex(BudgetExceededError, "output_tokens"):
            ledger.require_within_budget()

    def test_declared_algorithm_output_failure_keeps_all_known_usage(self) -> None:
        runner = InterventionRunner()
        worker = DeclaredFailureWorker({"status": "error", "failure_code": "BACKEND_OUTPUT_INVALID",
            "consumed": True, "usage": {"model_calls": 3, "retrieval_calls": 1,
                "input_tokens": 581, "output_tokens": 512}})
        with self.assertRaisesRegex(AlgorithmOutputError, "BACKEND_OUTPUT_INVALID"):
            runner.run(ExternalFailureProbe(), FrozenAgentState(query="q"),
                       InterventionContext(external_runtime=worker))
        usage = runner.monitor.list_snapshots()[0].usage
        self.assertEqual(worker.calls, 1)
        self.assertEqual((usage.model_calls, usage.retrieval_calls, usage.input_tokens, usage.output_tokens),
                         (3, 1, 581, 512))
        self.assertEqual(usage.unpriced_model_calls, 3)

    def test_algorithm_output_failure_does_not_turn_explicit_zero_or_unknown_into_success(self) -> None:
        for telemetry, expected in (({"consumed": False}, 0), ({"usage": "malformed"}, 1)):
            with self.subTest(telemetry=telemetry):
                runner = InterventionRunner()
                worker = DeclaredFailureWorker({"status": "error", "failure_code": "BACKEND_OUTPUT_INVALID", **telemetry})
                with self.assertRaises(AlgorithmOutputError):
                    runner.run(ExternalFailureProbe(), FrozenAgentState(query="q"),
                               InterventionContext(external_runtime=worker))
                usage = runner.monitor.list_snapshots()[0].usage
                self.assertEqual(usage.model_calls, expected)
                self.assertEqual(usage.unpriced_model_calls, expected)

    def test_algorithm_output_failure_retains_irreversible_budget_overage(self) -> None:
        budget = ExecutionBudget(max_model_calls=1, max_output_tokens=2)
        ledger = BudgetLedger(budget)
        runner = InterventionRunner()
        worker = DeclaredFailureWorker({"status": "error", "failure_code": "BACKEND_OUTPUT_INVALID",
            "consumed": True, "usage": {"model_calls": 3, "retrieval_calls": 0,
                "input_tokens": 581, "output_tokens": 512}})
        with self.assertRaises(AlgorithmOutputError):
            runner.run(ExternalFailureProbe(), FrozenAgentState(query="q", budget=budget),
                       InterventionContext(external_runtime=worker, budget_ledger=ledger))
        usage = runner.monitor.list_snapshots()[0].usage
        self.assertEqual((usage.model_calls, usage.input_tokens, usage.output_tokens), (3, 581, 512))
        self.assertIn("model_calls", ledger.actual_violations)
        self.assertIn("output_tokens", ledger.actual_violations)

    def test_injected_budget_ledger_is_owned_by_exactly_one_run(self) -> None:
        ledger = BudgetLedger(ExecutionBudget())
        context = InterventionContext(budget_ledger=ledger)
        runner = InterventionRunner()

        first = runner.run(
            TraceUsageProbe({}),
            FrozenAgentState(query="q"),
            context,
        )

        self.assertEqual(ledger.owner_run_id, first.monitor_run_id)
        with self.assertRaisesRegex(ValidationError, "already owned.*fresh BudgetLedger"):
            runner.run(
                TraceUsageProbe({}),
                FrozenAgentState(query="q"),
                context,
            )
        self.assertEqual(ledger.owner_run_id, first.monitor_run_id)

    def test_injected_budget_ledger_must_be_pristine(self) -> None:
        ledger = BudgetLedger(ExecutionBudget())
        ledger.reserve(ResourceUsage(model_calls=1))

        with self.assertRaisesRegex(ValidationError, "must be pristine"):
            InterventionRunner().run(
                TraceUsageProbe({}),
                FrozenAgentState(query="q"),
                InterventionContext(budget_ledger=ledger),
            )
        self.assertIsNone(ledger.owner_run_id)
        self.assertEqual(ledger.usage.model_calls, 1)

    def test_fresh_injected_ledger_remains_observable_after_run(self) -> None:
        ledger = BudgetLedger(ExecutionBudget())
        record = InterventionRunner().run(
            BackendRewrite(),
            FrozenAgentState(query="q"),
            InterventionContext(
                backend=ReplayBackend(("rewritten",)),
                budget_ledger=ledger,
            ),
        )

        self.assertEqual(ledger.owner_run_id, record.monitor_run_id)
        self.assertEqual(ledger.usage.model_calls, 1)

    def test_runner_records_time_usage_cost_and_trace(self) -> None:
        response = GenerationResponse(
            text="find red book",
            model="fixture",
            finish_reason="stop",
            usage=Usage(input_tokens=7, output_tokens=3, cost_usd=0.012),
        )
        monitor = RunMonitor()
        runner = InterventionRunner(monitor=monitor)
        record = runner.run(
            BackendRewrite(),
            FrozenAgentState(query="find red book"),
            InterventionContext(backend=ReplayBackend((response,))),
        )
        snapshot = monitor.snapshot(record.monitor_run_id)
        self.assertEqual(record.status, RunStatus.SUCCESS)
        self.assertEqual(snapshot.status, ObservedRunStatus.SUCCEEDED)
        self.assertEqual(snapshot.usage.model_calls, 1)
        self.assertEqual(snapshot.usage.input_tokens, 7)
        self.assertEqual(snapshot.usage.output_tokens, 3)
        self.assertAlmostEqual(snapshot.usage.cost_usd, 0.012)
        self.assertGreater(snapshot.usage.cpu_seconds, 0.0)
        self.assertEqual(record.usage, snapshot.usage)
        self.assertIsNotNone(snapshot.duration_ms)
        self.assertGreaterEqual(len(snapshot.events), 7)

    def test_call_budget_blocks_backend_before_network_or_compute(self) -> None:
        backend = ReplayBackend(("should not be consumed",))
        state = FrozenAgentState(
            query="q",
            budget=ExecutionBudget(max_model_calls=0),
        )
        runner = InterventionRunner()
        with self.assertRaises(BudgetExceededError):
            runner.run(BackendRewrite(), state, InterventionContext(backend=backend))
        self.assertEqual(backend._calls, 0)
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.failure.category, "budget_exhausted")
        self.assertEqual(snapshot.usage.model_calls, 0)

    def test_caught_post_response_overage_still_exhausts_the_run(self) -> None:
        response = GenerationResponse(
            text="too long",
            model="fixture",
            usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.0),
        )
        runner = InterventionRunner()
        with self.assertRaisesRegex(BudgetExceededError, "output_tokens"):
            runner.run(
                CatchBackendBudget(),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_output_tokens=1),
                ),
                InterventionContext(backend=ReplayBackend((response,))),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.usage.output_tokens, 2)

    def test_caught_reservation_rejection_still_exhausts_without_consumption(self) -> None:
        backend = ReplayBackend(("must not be consumed",))
        runner = InterventionRunner()
        with self.assertRaisesRegex(BudgetExceededError, "model_calls"):
            runner.run(
                CatchBackendBudget(),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_model_calls=0),
                ),
                InterventionContext(backend=backend),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.usage.model_calls, 0)
        self.assertEqual(backend._calls, 0)

    def test_context_token_budget_is_cumulative_across_model_calls(self) -> None:
        responses = tuple(
            GenerationResponse(
                text=f"response-{index}",
                model="fixture",
                usage=Usage(input_tokens=3, output_tokens=0, cost_usd=0.0),
            )
            for index in range(2)
        )
        runner = InterventionRunner()
        with self.assertRaisesRegex(BudgetExceededError, "context_tokens"):
            runner.run(
                CatchBackendBudget(calls=2),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_context_tokens=5),
                ),
                InterventionContext(backend=ReplayBackend(responses)),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.usage.model_calls, 2)
        self.assertEqual(snapshot.usage.input_tokens, 6)

    def test_usage_and_prices_reject_nonfinite_or_fractional_counters(self) -> None:
        with self.assertRaisesRegex(ValueError, "integers"):
            ResourceUsage(model_calls=0.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "finite"):
            ResourceUsage(cost_usd=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            ModelPrice(float("inf"), 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ComputePrice(accelerator_per_hour=float("nan"))

    def test_trace_usage_rejects_malformed_telemetry_without_coercion(self) -> None:
        malformed = (
            ({"model_calls": -1}, "model_calls"),
            ({"model_calls": 1.5}, "model_calls"),
            ({"input_tokens": float("nan")}, "input_tokens"),
            ({"cpu_seconds": float("inf")}, "cpu_seconds"),
            ({"cost_usd": -0.01}, "cost_usd"),
        )
        for counters, field_name in malformed:
            with self.subTest(counters=counters):
                runner = InterventionRunner()
                with self.assertRaisesRegex(ValidationError, field_name):
                    runner.run(
                        TraceUsageProbe(counters),
                        FrozenAgentState(query="q"),
                        InterventionContext(),
                    )
                snapshot = runner.monitor.list_snapshots()[0]
                self.assertEqual(snapshot.status, ObservedRunStatus.FAILED)
                self.assertEqual(
                    snapshot.usage, ResourceUsage(cpu_seconds=snapshot.usage.cpu_seconds)
                )

    def test_budget_exhausted_record_has_matching_monitor_status(self) -> None:
        runner = InterventionRunner()
        record = runner.run(
            TraceUsageProbe({}, reports_budget_exhaustion=True),
            FrozenAgentState(query="q"),
            InterventionContext(),
        )
        snapshot = runner.monitor.snapshot(record.monitor_run_id)
        self.assertEqual(record.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.failure.category, "budget_exhausted")

    def test_finished_run_cannot_be_mutated_by_a_second_finish(self) -> None:
        monitor = RunMonitor()
        run_id = monitor.start_run(release_id="r", state_hash="h")
        first = monitor.finish_run(ObservedRunStatus.SUCCEEDED, run_id=run_id)
        with self.assertRaisesRegex(ValueError, "already finished"):
            monitor.finish_run(ObservedRunStatus.FAILED, run_id=run_id)
        second = monitor.snapshot(run_id)
        self.assertEqual(first.usage, second.usage)
        self.assertEqual(first.events, second.events)
        with self.assertRaisesRegex(RuntimeError, "finished run"):
            monitor.record_usage(ResourceUsage(model_calls=1), run_id=run_id)
        with self.assertRaisesRegex(RuntimeError, "finished run"):
            monitor.emit(EventKind.STAGE_STARTED, run_id=run_id)

    def test_unknown_model_cost_fails_closed_only_when_cost_budget_is_set(self) -> None:
        response = GenerationResponse(
            text="rewritten",
            model="unpriced",
            usage=Usage(input_tokens=5, output_tokens=2, cost_usd=None),
        )
        unrestricted = InterventionRunner().run(
            BackendRewrite(),
            FrozenAgentState(query="q"),
            InterventionContext(backend=ReplayBackend((response,), model_id="unpriced")),
        )
        self.assertEqual(unrestricted.usage.unpriced_model_calls, 1)

        runner = InterventionRunner()
        with self.assertRaisesRegex(BudgetExceededError, "unknown_cost"):
            runner.run(
                BackendRewrite(),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_cost_usd=0.0),
                ),
                InterventionContext(backend=ReplayBackend((response,), model_id="unpriced")),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.unpriced_model_calls, 1)
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)

    def test_failed_backend_keeps_one_attempt_and_unknown_cost(self) -> None:
        for cost_limit in (None, 1.0):
            with self.subTest(cost_limit=cost_limit):
                calls = []

                def fail(request, *, calls=calls):
                    calls.append(request)
                    raise RuntimeError("transport failed")

                inner = CallableBackend(
                    fail, model_id="priced", revision="test-v1", callable_id="tests.failed.v1",
                )
                monitor = RunMonitor()
                run_id = monitor.start_run(release_id="test", state_hash="state")
                ledger = BudgetLedger(ExecutionBudget(max_model_calls=1, max_cost_usd=cost_limit))
                backend = ObservedLLMBackend(
                    inner=inner, monitor=monitor, ledger=ledger, run_id=run_id,
                    pricing=PricingCatalog({"priced": ModelPrice(2.0, 8.0)}),
                )
                with self.assertRaisesRegex(RuntimeError, "transport failed"):
                    backend.generate(GenerationRequest(prompt="question"))
                self.assertEqual(len(calls), 1)
                self.assertEqual(ledger.usage.model_calls, 1)
                self.assertEqual(ledger.usage.unpriced_model_calls, 1)
                snapshot = monitor.snapshot(run_id)
                self.assertEqual(snapshot.usage, ledger.usage)
                events = [e for e in snapshot.events if e.kind == EventKind.STAGE_FINISHED]
                self.assertEqual(events[-1].payload["usage_status"], "unknown")
                self.assertEqual(events[-1].payload["token_counters"], "known_lower_bound_only")
                if cost_limit is not None:
                    with self.assertRaisesRegex(BudgetExceededError, "unknown_cost"):
                        ledger.require_within_budget()
                with self.assertRaises(BudgetExceededError):
                    backend.generate(GenerationRequest(prompt="must not dispatch"))
                self.assertEqual(len(calls), 1)
                self.assertEqual(ledger.usage.model_calls, 1)
                self.assertEqual(ledger.usage.unpriced_model_calls, 1)

    def test_manifest_pricing_is_injected_into_the_runner(self) -> None:
        response = GenerationResponse(
            text="rewritten",
            model="priced",
            usage=Usage(input_tokens=1_000_000, output_tokens=0, cost_usd=None),
        )
        record = InterventionRunner().run(
            BackendRewrite(),
            FrozenAgentState(query="q", budget=ExecutionBudget(max_cost_usd=3.0)),
            InterventionContext(
                backend=ReplayBackend((response,), model_id="priced"),
                pricing_catalog=PricingCatalog({"priced": ModelPrice(2.0, 8.0)}),
            ),
        )
        self.assertEqual(record.usage.unpriced_model_calls, 0)
        self.assertAlmostEqual(record.usage.cost_usd, 2.0)

    def test_explicit_run_identity_propagates_to_worker_threads(self) -> None:
        response = GenerationResponse(text="threaded rewrite", model="fixture")
        record = InterventionRunner().run(
            ThreadedBackendRewrite(),
            FrozenAgentState(query="q"),
            InterventionContext(backend=ReplayBackend((response,))),
        )
        self.assertEqual(record.usage.model_calls, 1)

    def test_preobserved_backend_is_unwrapped_and_rebound_to_current_run(self) -> None:
        old_monitor = RunMonitor()
        old_run_id = old_monitor.start_run(release_id="old", state_hash="old")
        inner = ReplayBackend(("current rewrite",))
        observed = ObservedLLMBackend(
            inner,
            old_monitor,
            BudgetLedger(ExecutionBudget()),
            old_run_id,
        )
        record = InterventionRunner().run(
            BackendRewrite(),
            FrozenAgentState(query="q"),
            InterventionContext(backend=observed),
        )
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(len(old_monitor.snapshot(old_run_id).events), 1)

    def test_component_failure_category_is_not_overwritten(self) -> None:
        def fail(_request: object) -> GenerationResponse:
            raise RuntimeError("backend unavailable")

        backend = CallableBackend(
            fail,
            revision="fixture-v1",
            callable_id="tests.fail",
        )
        runner = InterventionRunner()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            runner.run(
                BackendRewrite(),
                FrozenAgentState(query="q"),
                InterventionContext(backend=backend),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.FAILED)
        self.assertEqual(snapshot.failure.category, "backend_error")
        self.assertEqual(snapshot.failure.stage, "llm")

    def test_strict_intent_gate_rejects_entity_loss(self) -> None:
        runner = InterventionRunner()
        with self.assertRaises(IntentViolationError):
            runner.run(
                BackendRewrite(),
                FrozenAgentState(query="find red book"),
                InterventionContext(
                    backend=ReplayBackend(("find a book",)),
                    intent_contract=IntentContract(required_entities=("red",)),
                ),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.INTENT_REJECTED)
        self.assertTrue(any(event.kind is EventKind.INTENT_CHECKED for event in snapshot.events))

    def test_intent_gate_uses_token_phrases_not_substrings(self) -> None:
        runner = InterventionRunner()
        with self.assertRaises(IntentViolationError):
            runner.run(
                BackendRewrite(),
                FrozenAgentState(query="US policy"),
                InterventionContext(
                    backend=ReplayBackend(("Russia policy",)),
                    intent_contract=IntentContract(required_entities=("US",)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            IntentContract(required_entities=("",))

    def test_retriever_and_tool_calls_share_the_run_ledger(self) -> None:
        retriever = InMemoryBM25Retriever(
            (Document("d1", "red book", corpus_revision="fixture-v1"),)
        )
        tools = InMemoryToolConnector(
            (ToolSpec("echo", "Return input"),),
            {"echo": lambda value: value},
        )
        record = InterventionRunner().run(
            DependencyProbe(),
            FrozenAgentState(query="red book"),
            InterventionContext(retriever=retriever, tools=tools),
        )
        self.assertEqual(record.usage.retrieval_calls, 1)
        self.assertEqual(record.usage.tool_calls, 1)
        self.assertEqual(len(tools.calls), 1)

    def test_tool_budget_blocks_call_before_side_effect(self) -> None:
        tools = InMemoryToolConnector(
            (ToolSpec("echo", "Return input"),),
            {"echo": lambda value: value},
        )
        state = FrozenAgentState(
            query="red book",
            budget=ExecutionBudget(max_tool_calls=0),
        )
        with self.assertRaises(BudgetExceededError):
            InterventionRunner().run(
                DependencyProbe(),
                state,
                InterventionContext(
                    retriever=InMemoryBM25Retriever(
                        (Document("d1", "red book", corpus_revision="fixture-v1"),)
                    ),
                    tools=tools,
                ),
            )
        self.assertEqual(tools.calls, [])

    def test_jsonl_sink_redacts_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            monitor = RunMonitor(sinks=(JsonlEventSink(path),))
            run_id = monitor.start_run(release_id="r", state_hash="h")
            monitor.emit(
                EventKind.STAGE_STARTED,
                run_id=run_id,
                payload={
                    "api_key": "sk-private-value",
                    "safe": "visible",
                    "input_tokens": 17,
                    "message": "Authorization: Bearer abcdefghijklmnop",
                },
            )
            monitor.finish_run(ObservedRunStatus.SUCCEEDED, run_id=run_id)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        serialized = json.dumps(records)
        self.assertNotIn("sk-private", serialized)
        self.assertNotIn("abcdefghijklmnop", serialized)
        self.assertIn("<redacted>", serialized)
        self.assertIn("visible", serialized)
        self.assertIn('"input_tokens": 17', serialized)

    def test_failure_text_redacts_named_credentials_without_hiding_token_counts(self) -> None:
        from agentinputlab.contracts.serialization import to_jsonable

        for text in ('token=private-value', 'https://example.invalid/?access_token=private-value&x=1',
                     'password="private-value with spaces"', "api_key:'private-value'"):
            with self.subTest(text=text):
                monitor = RunMonitor()
                with self.assertRaises(RuntimeError), monitor.run_scope(release_id="r", state_hash="h"):
                    raise RuntimeError(text + " input_tokens=17 output_tokens=3")
                serialized = json.dumps(to_jsonable(monitor.list_snapshots()))
                self.assertNotIn("private-value", serialized)
                self.assertIn("<redacted>", serialized)
                self.assertIn("input_tokens=17 output_tokens=3", serialized)

    def test_plain_holder_cannot_hide_a_renamed_backend(self) -> None:
        class PlainHolder:
            def __init__(self, generator) -> None:
                self.generator = generator

        @dataclass(frozen=True, slots=True)
        class HiddenRewrite:
            holder: object
            method_id: str = "fixture.hidden"
            release_id: str = "fixture.hidden.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                del context
                response = self.holder.generator.generate(  # type: ignore[attr-defined]
                    GenerationRequest(prompt=state.query)
                )
                return InputPatch(
                    method_id=self.method_id,
                    release_id=self.release_id,
                    source_state_hash=state.state_hash,
                    operations=(PatchOperation(PatchOp.REPLACE_QUERY, response.text),),
                    trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
                    implementation_identity="hidden-test-fixture",
                    profile_kind="fixture_unverified",
                )

        backend = ReplayBackend(("must not run",))
        with self.assertRaisesRegex(ValidationError, "holder.generator"):
            InterventionRunner().run(
                HiddenRewrite(PlainHolder(backend)),
                FrozenAgentState(query="q"),
                InterventionContext(),
            )
        self.assertEqual(backend._calls, 0)

    def test_class_attributes_and_method_captures_cannot_hide_backends(self) -> None:
        def patch_from(response, state):
            return InputPatch(
                method_id="fixture.class_hidden",
                release_id="fixture.class_hidden.v1",
                source_state_hash=state.state_hash,
                operations=(PatchOperation(PatchOp.REPLACE_QUERY, response.text),),
                trace=ExecutionTrace(
                    method_id="fixture.class_hidden",
                    release_id="fixture.class_hidden.v1",
                ),
                implementation_identity="class-hidden-test-fixture",
                profile_kind="fixture_unverified",
            )

        class_backend = ReplayBackend(("must not run",))

        class ClassAttributeHidden:
            raw = class_backend
            method_id = "fixture.class_hidden"
            release_id = "fixture.class_hidden.v1"
            family = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                del context
                return patch_from(
                    self.raw.generate(GenerationRequest(prompt=state.query)),
                    state,
                )

        default_backend = ReplayBackend(("must not run",))

        class MethodDefaultHidden:
            method_id = "fixture.class_hidden"
            release_id = "fixture.class_hidden.v1"
            family = SemanticFamily.REFORMULATE

            def propose(self, state, context, backend=default_backend):
                del context
                return patch_from(
                    backend.generate(GenerationRequest(prompt=state.query)),
                    state,
                )

        closure_backend = ReplayBackend(("must not run",))

        def closure_intervention(captured):
            class MethodClosureHidden:
                method_id = "fixture.class_hidden"
                release_id = "fixture.class_hidden.v1"
                family = SemanticFamily.REFORMULATE

                def propose(self, state, context):
                    del context
                    return patch_from(
                        captured.generate(GenerationRequest(prompt=state.query)),
                        state,
                    )

            return MethodClosureHidden()

        property_backend = ReplayBackend(("must not run",))

        class PropertyHidden:
            method_id = "fixture.class_hidden"
            release_id = "fixture.class_hidden.v1"
            family = SemanticFamily.REFORMULATE

            @property
            def raw(self, backend=property_backend):
                return backend

            def propose(self, state, context):
                del context
                return patch_from(
                    self.raw.generate(GenerationRequest(prompt=state.query)),
                    state,
                )

        cases = (
            (ClassAttributeHidden(), class_backend),
            (MethodDefaultHidden(), default_backend),
            (closure_intervention(closure_backend), closure_backend),
            (PropertyHidden(), property_backend),
        )
        state = FrozenAgentState(
            query="q",
            budget=ExecutionBudget(max_model_calls=0),
        )
        for intervention, backend in cases:
            with self.subTest(intervention=type(intervention).__name__):
                with self.assertRaisesRegex(ValidationError, "hidden on intervention"):
                    InterventionRunner().run(
                        intervention,
                        state,
                        InterventionContext(),
                    )
                self.assertEqual(backend._calls, 0)

        class RootExternalExecutor:
            method_id = "fixture.class_hidden"
            release_id = "fixture.class_hidden.v1"
            family = SemanticFamily.REFORMULATE

            def __init__(self):
                self.calls = 0

            def fingerprint(self):
                return {"runtime": "root-executor", "revision": "v1"}

            def call(self, state):
                self.calls += 1
                return InputPatch.no_op(
                    state,
                    method_id=self.method_id,
                    release_id=self.release_id,
                )

            def propose(self, state, context):
                del context
                return self.call(state)

        root_executor = RootExternalExecutor()
        with self.assertRaisesRegex(ValidationError, "hidden on intervention"):
            InterventionRunner().run(root_executor, state, InterventionContext())
        self.assertEqual(root_executor.calls, 0)

    def test_mapping_key_bound_method_and_closure_cannot_hide_backend(self) -> None:
        @dataclass(frozen=True, slots=True)
        class HiddenCallable:
            payload: object
            mode: str
            method_id: str = "fixture.hidden_callable"
            release_id: str = "fixture.hidden_callable.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                del context
                request = GenerationRequest(prompt=state.query)
                if self.mode == "mapping_key":
                    response = next(iter(self.payload)).generate(request)  # type: ignore[arg-type]
                else:
                    response = self.payload(request)  # type: ignore[operator]
                return InputPatch(
                    method_id=self.method_id,
                    release_id=self.release_id,
                    source_state_hash=state.state_hash,
                    operations=(PatchOperation(PatchOp.REPLACE_QUERY, response.text),),
                    trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
                    implementation_identity="hidden-callable-test-fixture",
                    profile_kind="fixture_unverified",
                )

        for mode in ("mapping_key", "bound_method", "closure"):
            with self.subTest(mode=mode):

                class HashableReplayBackend(ReplayBackend):
                    __hash__ = object.__hash__

                backend = HashableReplayBackend(("must not run",))
                if mode == "mapping_key":
                    payload = {backend: "ignored"}
                elif mode == "bound_method":
                    payload = backend.generate
                else:

                    def make_payload(captured):
                        def invoke(request):
                            return captured.generate(request)

                        return invoke

                    payload = make_payload(backend)

                with self.assertRaisesRegex(ValidationError, "hidden on intervention"):
                    InterventionRunner().run(
                        HiddenCallable(payload, mode),
                        FrozenAgentState(query="q"),
                        InterventionContext(),
                    )
                self.assertEqual(backend._calls, 0)

    def test_runtime_dependency_is_terminal_for_hidden_dependency_inspection(self) -> None:
        class LargeBackend:
            model_id = "large-fixture"
            capabilities = frozenset()

            def __init__(self) -> None:
                self.payload = tuple(object() for _ in range(5000))
                self.calls = 0

            def generate(self, request):
                del request
                self.calls += 1
                return GenerationResponse(text="must not run", model=self.model_id)

            def fingerprint(self):
                return {"backend": "large-fixture", "revision": "v1"}

        @dataclass(frozen=True, slots=True)
        class HiddenLargeBackend:
            backend: object
            method_id: str = "fixture.hidden_large_backend"
            release_id: str = "fixture.hidden_large_backend.v1"
            family: SemanticFamily = SemanticFamily.REFORMULATE

            def propose(self, state, context):
                del context
                self.backend.generate(GenerationRequest(prompt=state.query))
                return InputPatch.no_op(
                    state,
                    method_id=self.method_id,
                    release_id=self.release_id,
                )

        backend = LargeBackend()
        with self.assertRaisesRegex(ValidationError, "hidden on intervention"):
            InterventionRunner().run(
                HiddenLargeBackend(backend),
                FrozenAgentState(query="q"),
                InterventionContext(),
            )
        self.assertEqual(backend.calls, 0)


if __name__ == "__main__":
    unittest.main()
