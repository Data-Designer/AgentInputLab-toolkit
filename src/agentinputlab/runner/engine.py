"""Minimal intervention runner preserving paired state and typed failures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from agentinputlab.contracts.enums import PatchOp, StrEnum
from agentinputlab.contracts.errors import (
    AlgorithmOutputError,
    BudgetExceededError,
    IntentViolationError,
    ValidationError,
)
from agentinputlab.contracts.intent import IntentCheck, check_intent_contract
from agentinputlab.contracts.lifecycle import CompositeResult, Intervention, InterventionContext
from agentinputlab.contracts.patch import InputPatch
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.observability import (
    BudgetLedger,
    EventKind,
    ObservedRunStatus,
    PricingCatalog,
    ResourceUsage,
    RunMonitor,
    instrument_context,
)

from .runtime_guard import runtime_dependency_nodes


class RunStatus(StrEnum):
    SUCCESS = "success"
    NO_OP = "no_op"
    FALLBACK = "fallback"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class RunRecord:
    release_id: str
    input_state_hash: str
    output_state_hash: str
    patch: InputPatch
    status: RunStatus
    monitor_run_id: str = ""
    duration_ms: float | None = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    intent_check: IntentCheck | None = None


class InterventionRunner:
    """Run a profile without swallowing unavailable/error conditions."""

    def __init__(self, *, monitor: RunMonitor | None = None) -> None:
        self.monitor = monitor or RunMonitor()

    def run(
        self,
        intervention: Intervention,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> RunRecord:
        _reject_hidden_runtime_dependencies(intervention)
        monitor = context.monitor or self.monitor
        if not isinstance(monitor, RunMonitor):
            raise TypeError("context.monitor must be a RunMonitor")
        intent_check: IntentCheck | None = None
        with monitor.run_scope(
            release_id=intervention.release_id,
            state_hash=state.state_hash,
            tags={"method_id": intervention.method_id, "family": intervention.family.value},
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
            monitor.emit(EventKind.STAGE_STARTED, component="intervention")
            try:
                patch = intervention.propose(state, runtime_context)
                if not isinstance(patch, InputPatch):
                    raise AlgorithmOutputError(
                        f"intervention must return an InputPatch; got {type(patch).__name__}"
                    )
                if patch.method_id != intervention.method_id:
                    raise AlgorithmOutputError("intervention returned a patch for another method")
                if patch.release_id != intervention.release_id:
                    raise AlgorithmOutputError("intervention returned a patch for another release")
                self._reconcile_trace_usage(patch, ledger, monitor)
                output = patch.apply(state)
                if runtime_context.intent_contract is not None:
                    intent_check = check_intent_contract(
                        runtime_context.intent_contract,
                        state,
                        patch,
                        output,
                    )
                    monitor.emit(
                        EventKind.INTENT_CHECKED,
                        component="intent_gate",
                        payload={
                            "preserved": intent_check.preserved,
                            "missing_entities": intent_check.missing_entities,
                            "missing_constraints": intent_check.missing_constraints,
                            "introduced_assumptions": intent_check.introduced_assumptions,
                        },
                    )
                    if runtime_context.strict_intent and not intent_check.preserved:
                        raise IntentViolationError("observable intent contract was not preserved")
                ledger.require_within_budget()
            except BudgetExceededError as exc:
                monitor.record_failure(
                    exc,
                    category="budget_exhausted",
                    stage="intervention",
                )
                monitor.finish_run(ObservedRunStatus.BUDGET_EXHAUSTED, run_id=run_id)
                raise
            except IntentViolationError as exc:
                monitor.record_failure(
                    exc,
                    category="intent_drift",
                    stage="intent_gate",
                )
                monitor.finish_run(ObservedRunStatus.INTENT_REJECTED, run_id=run_id)
                raise

            is_noop = all(operation.op is PatchOp.NO_OP for operation in patch.operations)
            if patch.metadata.get("fallback"):
                status = RunStatus.FALLBACK
            elif patch.metadata.get("budget_exhausted"):
                status = RunStatus.BUDGET_EXHAUSTED
            elif is_noop:
                status = RunStatus.NO_OP
            else:
                status = RunStatus.SUCCESS
            monitor.emit(
                EventKind.STAGE_FINISHED,
                component="intervention",
                payload={"status": status.value, "patch_hash": patch.patch_hash},
            )
            observed_status = ObservedRunStatus.SUCCEEDED
            if status is RunStatus.BUDGET_EXHAUSTED:
                error = BudgetExceededError("intervention reported budget exhaustion")
                monitor.record_failure(
                    error,
                    category="budget_exhausted",
                    stage="intervention",
                )
                observed_status = ObservedRunStatus.BUDGET_EXHAUSTED
            snapshot = monitor.finish_run(
                observed_status,
                run_id=run_id,
                payload={"outcome": status.value},
            )
            return RunRecord(
                release_id=intervention.release_id,
                input_state_hash=state.state_hash,
                output_state_hash=output.state_hash,
                patch=patch,
                status=status,
                monitor_run_id=run_id,
                duration_ms=snapshot.duration_ms,
                usage=snapshot.usage,
                intent_check=intent_check,
            )

    @staticmethod
    def _reconcile_trace_usage(
        patch: InputPatch | CompositeResult,
        ledger: BudgetLedger,
        monitor: RunMonitor,
    ) -> None:
        """Charge counters emitted by uninstrumented upstream workers once."""

        counters = patch.trace.counters
        current = ledger.usage
        declared = ResourceUsage(
            model_calls=_trace_integer_counter(counters, "model_calls"),
            retrieval_calls=_trace_integer_counter(counters, "retrieval_calls"),
            tool_calls=_trace_integer_counter(counters, "tool_calls"),
            embedding_calls=_trace_integer_counter(counters, "embedding_calls"),
            unpriced_model_calls=_trace_integer_counter(counters, "unpriced_model_calls"),
            input_tokens=_trace_integer_counter(counters, "input_tokens"),
            output_tokens=_trace_integer_counter(counters, "output_tokens"),
            cpu_seconds=_trace_float_counter(counters, "cpu_seconds"),
            accelerator_seconds=_trace_float_counter(counters, "accelerator_seconds"),
            cost_usd=_trace_float_counter(counters, "cost_usd"),
            compute_cost_usd=_trace_float_counter(counters, "compute_cost_usd"),
        )
        delta = ResourceUsage(
            model_calls=max(0, declared.model_calls - current.model_calls),
            retrieval_calls=max(0, declared.retrieval_calls - current.retrieval_calls),
            tool_calls=max(0, declared.tool_calls - current.tool_calls),
            embedding_calls=max(0, declared.embedding_calls - current.embedding_calls),
            unpriced_model_calls=max(
                0,
                declared.unpriced_model_calls - current.unpriced_model_calls,
            ),
            input_tokens=max(0, declared.input_tokens - current.input_tokens),
            output_tokens=max(0, declared.output_tokens - current.output_tokens),
            cpu_seconds=max(0.0, declared.cpu_seconds - current.cpu_seconds),
            accelerator_seconds=max(
                0.0,
                declared.accelerator_seconds - current.accelerator_seconds,
            ),
            cost_usd=max(0.0, declared.cost_usd - current.cost_usd),
            compute_cost_usd=max(
                0.0,
                declared.compute_cost_usd - current.compute_cost_usd,
            ),
        )
        if delta != ResourceUsage():
            try:
                ledger.charge_actual(delta)
            except BudgetExceededError:
                monitor.record_usage(delta, component="upstream_trace")
                monitor.emit(EventKind.BUDGET_EXCEEDED, component="upstream_trace")
                raise
            monitor.record_usage(delta, component="upstream_trace")


def _trace_integer_counter(counters: Mapping[str, object], name: str) -> int:
    value = counters.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(
            f"trace counter {name!r} must be a non-negative integer; got {value!r}"
        )
    return value


def _trace_float_counter(counters: Mapping[str, object], name: str) -> float:
    value = counters.get(name, 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or not math.isfinite(value)
    ):
        raise ValidationError(
            f"trace counter {name!r} must be finite and non-negative; got {value!r}"
        )
    return float(value)


def _reject_hidden_runtime_dependencies(intervention: Intervention) -> None:
    """Keep executable dependencies on the context, independent of field names.

    Upstream adapters may call an LLM field ``generator`` or nest a backend in a
    small configuration holder.  Looking only for attributes literally named
    ``backend`` would let those calls bypass monitoring, budgets, and balance
    fingerprints.  This bounded walk inspects the intervention's declared
    state and data containers, but never descends into arbitrary dependency
    implementations.
    """

    hidden = _hidden_runtime_dependency_paths(intervention)
    if hidden:
        names = ", ".join(hidden)
        raise ValidationError(
            "runtime dependencies must be injected through InterventionContext "
            f"for monitoring and balance audit; hidden on intervention: {names}"
        )


def _hidden_runtime_dependency_paths(intervention: Intervention) -> list[str]:
    return sorted(
        {
            node.path
            for node in runtime_dependency_nodes(
                intervention,
                root_path="intervention",
            )
        }
    )
