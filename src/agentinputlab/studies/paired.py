"""Paired study runner with mandatory pre-run balance audit."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType

from agentinputlab.contracts.errors import (
    AlgorithmUnavailableError,
    ArmExecutionError,
    ArtifactError,
    BalanceError,
    CapabilityError,
    RegistryError,
    ValidationError,
)
from agentinputlab.contracts.lifecycle import Intervention, InterventionContext
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.observability import (
    BudgetLedger,
    RunMonitor,
)
from agentinputlab.runner.balance import (
    BalanceAuditor,
    BalanceReport,
    ComparisonArm,
    ComparisonProfile,
    context_dependency_fingerprints,
)
from agentinputlab.runner.engine import InterventionRunner, RunRecord
from agentinputlab.runner.runtime_guard import (
    inspect_object_graph,
    is_runtime_dependency,
)

_PAIRED_RUNTIME_SLOTS = (
    "backend",
    "dense_embedder",
    "retriever",
    "vector_searcher",
    "tools",
    "external_runtime",
    "artifact_manager",
)
_CELL_INVALID_ERRORS = (
    AlgorithmUnavailableError,
    ArtifactError,
    BalanceError,
    CapabilityError,
    RegistryError,
    ValidationError,
)
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(
        r"(?i)(\b(?:[a-z0-9]+[_-])*(?:api[_-]?key|x[_-]?api[_-]?key|authorization)"
        r"\b\s*[:=]\s*)['\"]?(?:bearer\s+)?[^\s,;'\"]{4,}"
    ),
)


@dataclass(frozen=True, slots=True)
class PairedArm:
    comparison_arm: ComparisonArm
    intervention: Intervention
    context: InterventionContext

    def __post_init__(self) -> None:
        if self.comparison_arm.release_id != self.intervention.release_id:
            raise ValueError("comparison arm and intervention release IDs differ")


@dataclass(frozen=True, slots=True)
class PairedRun:
    state_hash: str
    balance: BalanceReport
    outcomes: tuple[ArmOutcome, ...]
    observed_dependency_hashes: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(
            self,
            "observed_dependency_hashes",
            freeze_mapping(self.observed_dependency_hashes),
        )

    @property
    def records(self) -> tuple[RunRecord, ...]:
        """Successful records retained for backward-compatible aggregation."""

        return tuple(outcome.record for outcome in self.outcomes if outcome.record is not None)


@dataclass(frozen=True, slots=True)
class ArmFailure:
    category: str
    message: str
    monitor_run_id: str = ""

    def __post_init__(self) -> None:
        if not self.category:
            raise ValueError("arm failure category must be non-empty")
        object.__setattr__(self, "message", _safe_failure_message(self.message))
        if not isinstance(self.monitor_run_id, str):
            raise TypeError("monitor_run_id must be a string")


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    arm_id: str
    status: str
    record: RunRecord | None = None
    failure: ArmFailure | None = None

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed"}:
            raise ValueError("arm outcome status must be succeeded or failed")
        if (self.record is None) == (self.failure is None):
            raise ValueError("arm outcome requires exactly one record or failure")


class PairedStudyRunner:
    def __init__(self) -> None:
        self._balance = BalanceAuditor()
        self._runner = InterventionRunner()

    def run(
        self,
        state: FrozenAgentState,
        profile: ComparisonProfile,
        arms: tuple[PairedArm, ...],
    ) -> PairedRun:
        _reject_shared_runtime_instances(arms)
        before = {
            arm.comparison_arm.arm_id: context_dependency_fingerprints(arm.context) for arm in arms
        }
        report = self._balance.audit(
            profile,
            tuple(arm.comparison_arm for arm in arms),
            runtime_fingerprints=before,
        )
        report.require_eligible()
        outcomes: list[ArmOutcome] = []
        for arm in arms:
            monitor = arm.context.monitor or self._runner.monitor
            if monitor is not None and not isinstance(monitor, RunMonitor):
                raise TypeError("context.monitor must be a RunMonitor")
            prior_run_ids = (
                frozenset(snapshot.run_id for snapshot in monitor.list_snapshots())
                if monitor is not None
                else frozenset()
            )
            try:
                record = self._runner.run(arm.intervention, state, arm.context)
            except Exception as exc:
                monitor_run_id = _find_new_monitor_run_id(
                    monitor,
                    prior_run_ids,
                    release_id=arm.intervention.release_id,
                    state_hash=state.state_hash,
                )
                if _invalidates_cell(exc):
                    raise
                outcomes.append(
                    ArmOutcome(
                        arm_id=arm.comparison_arm.arm_id,
                        status="failed",
                        failure=ArmFailure(
                            type(exc).__name__,
                            str(exc),
                            monitor_run_id=monitor_run_id,
                        ),
                    )
                )
            else:
                outcomes.append(
                    ArmOutcome(
                        arm_id=arm.comparison_arm.arm_id,
                        status="succeeded",
                        record=record,
                    )
                )
        records = tuple(outcome.record for outcome in outcomes if outcome.record is not None)
        if any(record.input_state_hash != state.state_hash for record in records):
            raise RuntimeError("paired runner produced a non-paired input identity")
        after = {
            arm.comparison_arm.arm_id: context_dependency_fingerprints(arm.context) for arm in arms
        }
        if before != after:
            raise BalanceError("runtime dependency identity changed while executing paired arms")
        post_report = self._balance.audit(
            profile,
            tuple(arm.comparison_arm for arm in arms),
            runtime_fingerprints=after,
        )
        post_report.require_eligible()
        return PairedRun(
            state_hash=state.state_hash,
            balance=post_report,
            outcomes=tuple(outcomes),
            observed_dependency_hashes=after,
        )


def _reject_shared_runtime_instances(arms: tuple[PairedArm, ...]) -> None:
    """Require equal-but-independent runtime dependencies across comparison arms."""

    runtime_owners: dict[int, set[str]] = {}
    ledger_owners: dict[int, set[str]] = {}
    for arm in arms:
        arm_id = arm.comparison_arm.arm_id
        if isinstance(arm.context.budget_ledger, BudgetLedger):
            ledger_owners.setdefault(id(arm.context.budget_ledger), set()).add(arm_id)
        for slot in _PAIRED_RUNTIME_SLOTS:
            dependency = getattr(arm.context, slot)
            if dependency is None:
                continue
            graph = inspect_object_graph(
                dependency,
                root_path=f"context.{slot}",
                terminal_types=(RunMonitor, BudgetLedger),
                traverse_runtime_delegates=True,
            )
            for node in graph:
                if is_runtime_dependency(node.value):
                    runtime_owners.setdefault(id(node.value), set()).add(arm_id)
                if isinstance(node.value, BudgetLedger):
                    ledger_owners.setdefault(id(node.value), set()).add(arm_id)

    shared_runtimes = [sorted(arm_ids) for arm_ids in runtime_owners.values() if len(arm_ids) > 1]
    if shared_runtimes:
        details = ", ".join("/".join(arm_ids) for arm_ids in shared_runtimes)
        raise BalanceError(
            "runtime dependency objects, including proxy inners, must be independently "
            f"constructed across arms: {details}"
        )
    shared_ledgers = [sorted(arm_ids) for arm_ids in ledger_owners.values() if len(arm_ids) > 1]
    if shared_ledgers:
        details = ", ".join("/".join(arm_ids) for arm_ids in shared_ledgers)
        raise BalanceError(
            "runtime BudgetLedger instances must be independently constructed across arms: "
            f"{details}"
        )


def _find_new_monitor_run_id(
    monitor: RunMonitor | None,
    prior_run_ids: frozenset[str],
    *,
    release_id: str,
    state_hash: str,
) -> str:
    if monitor is None:
        return ""
    matches = [
        snapshot.run_id
        for snapshot in monitor.list_snapshots()
        if snapshot.run_id not in prior_run_ids
        and snapshot.release_id == release_id
        and snapshot.state_hash == state_hash
    ]
    return matches[-1] if matches else ""


def _invalidates_cell(error: Exception) -> bool:
    """Return whether an exception invalidates the experiment rather than one arm."""

    # Arm-local subclasses take precedence because they intentionally remain
    # ValidationError subclasses for public API compatibility.
    if isinstance(error, ArmExecutionError):
        return False
    if isinstance(error, (*_CELL_INVALID_ERRORS, MemoryError)):
        return True
    frames = _traceback_frames(error.__traceback__)
    if any(
        module == "agentinputlab.observability.monitor" and function != "run_scope"
        for module, function in frames
    ):
        return True
    origin = frames[-1][0] if frames else ""
    if origin.startswith(("agentinputlab.artifacts", "agentinputlab.data")):
        return True
    if origin == "agentinputlab.observability.instrumentation" and isinstance(
        error, (TypeError, ValueError)
    ):
        return True
    if origin == "agentinputlab.runner.runtime_guard" and isinstance(
        error, (AttributeError, TypeError)
    ):
        return True
    return origin == "agentinputlab.runner.engine" and isinstance(
        error, (AttributeError, TypeError)
    )


def _traceback_frames(traceback: TracebackType | None) -> tuple[tuple[str, str], ...]:
    frames: list[tuple[str, str]] = []
    while traceback is not None:
        frames.append(
            (
                str(traceback.tb_frame.f_globals.get("__name__", "")),
                traceback.tb_frame.f_code.co_name,
            )
        )
        traceback = traceback.tb_next
    return tuple(frames)


def _safe_failure_message(message: object) -> str:
    safe = str(message)[:2048]
    for pattern in _SECRET_TEXT_PATTERNS:
        if pattern.groups:
            safe = pattern.sub(lambda match: f"{match.group(1)}<redacted>", safe)
        else:
            safe = pattern.sub("<redacted>", safe)
    return safe
