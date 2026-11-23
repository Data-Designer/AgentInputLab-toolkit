"""Observed runner for full agent/search architectures."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CompositeIntervention,
    CompositeResult,
    FrozenAgentState,
    InterventionContext,
)
from agentinputlab.observability import (
    BudgetLedger,
    EventKind,
    ObservedRunStatus,
    PricingCatalog,
    ResourceUsage,
    RunMonitor,
    instrument_context,
)

from .engine import InterventionRunner, _reject_hidden_runtime_dependencies


@dataclass(frozen=True, slots=True)
class CompositeRunRecord:
    """Observed terminal record for a composite architecture."""

    release_id: str
    input_state_hash: str
    result: CompositeResult
    monitor_run_id: str = ""
    duration_ms: float | None = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)


class CompositeInterventionRunner:
    """Run a composite profile with the same accounting guarantees as patches."""

    def __init__(self, *, monitor: RunMonitor | None = None) -> None:
        self.monitor = monitor or RunMonitor()

    def run(
        self,
        intervention: CompositeIntervention,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> CompositeRunRecord:
        _reject_hidden_runtime_dependencies(intervention)
        monitor = context.monitor or self.monitor
        if not isinstance(monitor, RunMonitor):
            raise TypeError("context.monitor must be a RunMonitor")
        with monitor.run_scope(
            release_id=intervention.release_id,
            state_hash=state.state_hash,
            tags={
                "method_id": intervention.method_id,
                "family": intervention.family.value,
                "execution_kind": "composite",
            },
        ) as run_id:
            pricing = context.pricing_catalog
            if pricing is not None and not isinstance(pricing, PricingCatalog):
                raise TypeError("context.pricing_catalog must be a PricingCatalog")
            runtime_context = instrument_context(
                context,
                state,
                monitor,
                run_id=run_id,
                pricing=pricing,
            )
            ledger = runtime_context.budget_ledger
            if not isinstance(ledger, BudgetLedger):  # pragma: no cover - helper invariant
                raise TypeError("instrumented context did not produce a BudgetLedger")
            monitor.emit(EventKind.STAGE_STARTED, component="composite_intervention")
            try:
                result = intervention.run(state, runtime_context)
                self._validate_result(intervention, state, result)
                InterventionRunner._reconcile_trace_usage(result, ledger, monitor)
                ledger.require_within_budget()
            except BudgetExceededError as exc:
                monitor.record_failure(
                    exc,
                    category="budget_exhausted",
                    stage="composite_intervention",
                )
                monitor.finish_run(ObservedRunStatus.BUDGET_EXHAUSTED, run_id=run_id)
                raise

            monitor.emit(
                EventKind.STAGE_FINISHED,
                component="composite_intervention",
                payload={"status": "success", "result_hash": result.result_hash},
            )
            snapshot = monitor.finish_run(
                ObservedRunStatus.SUCCEEDED,
                run_id=run_id,
                payload={"outcome": "success"},
            )
            return CompositeRunRecord(
                release_id=intervention.release_id,
                input_state_hash=state.state_hash,
                result=result,
                monitor_run_id=run_id,
                duration_ms=snapshot.duration_ms,
                usage=snapshot.usage,
            )

    @staticmethod
    def _validate_result(
        intervention: CompositeIntervention,
        state: FrozenAgentState,
        result: object,
    ) -> None:
        if not isinstance(result, CompositeResult):
            raise AlgorithmOutputError(
                f"composite intervention must return a CompositeResult; got {type(result).__name__}"
            )
        if result.method_id != intervention.method_id:
            raise AlgorithmOutputError("composite returned a result for another method")
        if result.release_id != intervention.release_id:
            raise AlgorithmOutputError("composite returned a result for another release")
        if result.source_state_hash != state.state_hash:
            raise AlgorithmOutputError("composite result source_state_hash mismatch")


__all__ = ["CompositeInterventionRunner", "CompositeRunRecord"]
