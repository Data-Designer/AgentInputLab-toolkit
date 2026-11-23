"""Controlled benchmark execution across interventions and downstream predictors.

The lower-level benchmark runner accepts an arbitrary prediction callback.  This
module owns the missing research boundary: it applies one intervention to an
evaluator-blind state, sends only the transformed state to a typed downstream
predictor, and lets the benchmark adapter see evaluator gold only while
scoring.  It provides both the historical two-arm split runner and a per-unit
multi-arm block runner used by production experiment cells.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.enums import PatchOp, SemanticFamily
from agentinputlab.contracts.errors import (
    ArmExecutionError,
    BalanceError,
    ValidationError,
)
from agentinputlab.contracts.lifecycle import Intervention, InterventionContext
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.contracts.state import ExecutionBudget, FrozenAgentState
from agentinputlab.observability import (
    BudgetLedger,
    ResourceReceipt,
    ResourceUsage,
    RunMonitor,
)
from agentinputlab.runner import (
    BalanceAuditor,
    BalanceReport,
    ComparisonArm,
    ComparisonProfile,
    InterventionRunner,
    RunRecord,
    context_dependency_fingerprints,
)
from agentinputlab.runner.runtime_guard import inspect_object_graph, is_runtime_dependency
from agentinputlab.studies.paired import ArmFailure

from .base import BenchmarkAdapter, BenchmarkExample
from .runtime import (
    BenchmarkExampleRecord,
    PairedBenchmarkExampleRecord,
    PairedBenchmarkRecord,
    build_benchmark_example_record,
)

_PREDICTOR_FINGERPRINT_SLOT = "downstream_predictor"
_CONTEXT_RUNTIME_SLOTS = (
    "backend",
    "dense_embedder",
    "retriever",
    "vector_searcher",
    "tools",
    "external_runtime",
    "artifact_manager",
)


@dataclass(frozen=True, slots=True)
class DownstreamPrediction:
    """Typed downstream output, usage, and per-call supervisor receipts."""

    prediction: Any = field(repr=False)
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    resource_receipts: tuple[ResourceReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.usage, ResourceUsage):
            raise TypeError("downstream prediction usage must be ResourceUsage")
        receipts = _validated_resource_receipts(
            self.resource_receipts,
            label="downstream prediction",
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "resource_receipts", receipts)


@runtime_checkable
class DownstreamPredictor(Protocol):
    """Agent/retriever boundary used after an input intervention is applied."""

    def predict(self, state: FrozenAgentState) -> DownstreamPrediction: ...

    def fingerprint(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkInterventionArm:
    """One intervention and an independently constructed downstream runtime."""

    role: str
    comparison_arm: ComparisonArm
    intervention: Intervention
    context: InterventionContext
    predictor: DownstreamPredictor

    def __post_init__(self) -> None:
        if self.role not in {"baseline", "treatment"}:
            raise ValidationError("benchmark intervention arm role must be baseline or treatment")
        if self.comparison_arm.release_id != self.intervention.release_id:
            raise ValidationError("comparison arm and intervention release IDs differ")
        if not isinstance(self.context, InterventionContext):
            raise TypeError("benchmark arm context must be InterventionContext")
        if not isinstance(self.predictor, DownstreamPredictor):
            raise TypeError("benchmark arm predictor must implement predict() and fingerprint()")


@dataclass(frozen=True, slots=True)
class BenchmarkArmExecutionRecord:
    """One arm's intervention, prediction, score, usage, or typed failure."""

    role: str
    arm_id: str
    status: str
    intervention_record: RunRecord | None = None
    benchmark_record: BenchmarkExampleRecord | None = None
    downstream_usage: ResourceUsage = field(default_factory=ResourceUsage)
    downstream_duration_ms: float = 0.0
    downstream_metadata: Mapping[str, Any] = field(default_factory=dict)
    downstream_resource_receipts: tuple[ResourceReceipt, ...] = ()
    failure: ArmFailure | None = None

    def __post_init__(self) -> None:
        if self.role not in {"baseline", "treatment"}:
            raise ValidationError("benchmark arm execution role must be baseline or treatment")
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise ValidationError("benchmark arm execution requires a non-empty arm_id")
        if self.status not in {"succeeded", "failed"}:
            raise ValidationError("benchmark arm status must be succeeded or failed")
        if not isinstance(self.downstream_usage, ResourceUsage):
            raise TypeError("benchmark arm downstream_usage must be ResourceUsage")
        receipts = _validated_resource_receipts(
            self.downstream_resource_receipts,
            label="benchmark arm downstream",
        )
        object.__setattr__(
            self,
            "downstream_metadata",
            freeze_mapping(self.downstream_metadata),
        )
        if (
            isinstance(self.downstream_duration_ms, bool)
            or not isinstance(self.downstream_duration_ms, (int, float))
            or not math.isfinite(self.downstream_duration_ms)
            or self.downstream_duration_ms < 0
        ):
            raise ValidationError(
                "benchmark arm downstream_duration_ms must be finite and non-negative"
            )
        object.__setattr__(self, "downstream_duration_ms", float(self.downstream_duration_ms))
        object.__setattr__(self, "downstream_resource_receipts", receipts)
        if self.status == "succeeded":
            if (
                self.intervention_record is None
                or self.benchmark_record is None
                or self.failure is not None
            ):
                raise ValidationError(
                    "successful benchmark arm requires intervention and benchmark records"
                )
        elif self.benchmark_record is not None or self.failure is None:
            raise ValidationError(
                "failed benchmark arm requires a failure and cannot contain a benchmark score"
            )

    @property
    def total_usage(self) -> ResourceUsage:
        intervention_usage = (
            self.intervention_record.usage
            if self.intervention_record is not None
            else ResourceUsage()
        )
        return intervention_usage + self.downstream_usage


@dataclass(frozen=True, slots=True)
class PairedBenchmarkExecutionExample:
    """Both arm attempts for one shared evaluator-blind input state."""

    example_id: str
    state_hash: str
    arms: tuple[BenchmarkArmExecutionRecord, ...]

    def __post_init__(self) -> None:
        arms = tuple(self.arms)
        if len(arms) != 2 or len({arm.arm_id for arm in arms}) != 2:
            raise ValidationError("paired benchmark execution requires two unique arms")
        if {arm.role for arm in arms} != {"baseline", "treatment"}:
            raise ValidationError(
                "paired benchmark execution requires baseline and treatment roles"
            )
        for arm in arms:
            record = arm.benchmark_record
            if record is not None and (
                record.example_id != self.example_id or record.state_hash != self.state_hash
            ):
                raise ValidationError(
                    "benchmark arm score does not match the paired example identity"
                )
        object.__setattr__(self, "arms", arms)

    @property
    def complete(self) -> bool:
        return all(arm.status == "succeeded" for arm in self.arms)


@dataclass(frozen=True, slots=True)
class PairedBenchmarkExecution:
    """Auditable high-level result; aggregates exist only for a complete run."""

    benchmark_id: str
    split: str
    profile_id: str
    balance: BalanceReport
    examples: tuple[PairedBenchmarkExecutionExample, ...]
    observed_dependency_hashes: Mapping[str, Mapping[str, str]]
    paired_record: PairedBenchmarkRecord | None = None

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if not examples:
            raise ValidationError("paired benchmark execution requires at least one example")
        if len({example.example_id for example in examples}) != len(examples):
            raise ValidationError("paired benchmark example IDs must be unique")
        complete = all(example.complete for example in examples)
        if complete != (self.paired_record is not None):
            raise ValidationError(
                "paired aggregate must exist exactly when every example has two successful arms"
            )
        if self.paired_record is not None and (
            self.paired_record.benchmark_id != self.benchmark_id
            or self.paired_record.split != self.split
        ):
            raise ValidationError("paired aggregate identity differs from execution identity")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(
            self,
            "observed_dependency_hashes",
            freeze_mapping(self.observed_dependency_hashes),
        )

    @property
    def complete(self) -> bool:
        return self.paired_record is not None


@dataclass(frozen=True, slots=True)
class BenchmarkBlockArm:
    """One planned arm in a per-example controlled comparison block."""

    comparison_arm: ComparisonArm
    intervention: Intervention
    context: InterventionContext
    predictor: DownstreamPredictor

    def __post_init__(self) -> None:
        if self.comparison_arm.release_id != self.intervention.release_id:
            raise ValidationError("comparison arm and intervention release IDs differ")
        if not isinstance(self.context, InterventionContext):
            raise TypeError("benchmark block arm context must be InterventionContext")
        if not isinstance(self.predictor, DownstreamPredictor):
            raise TypeError(
                "benchmark block arm predictor must implement predict() and fingerprint()"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkBlockArmExecutionRecord:
    """One exact planned arm outcome for a single benchmark example."""

    arm_id: str
    release_id: str
    status: str
    intervention_record: RunRecord | None = None
    benchmark_record: BenchmarkExampleRecord | None = None
    downstream_usage: ResourceUsage = field(default_factory=ResourceUsage)
    downstream_duration_ms: float = 0.0
    downstream_metadata: Mapping[str, Any] = field(default_factory=dict)
    downstream_resource_receipts: tuple[ResourceReceipt, ...] = ()
    failure: ArmFailure | None = None

    def __post_init__(self) -> None:
        for label, value in (("arm_id", self.arm_id), ("release_id", self.release_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"benchmark block arm execution requires a non-empty {label}")
        if self.status not in {"succeeded", "failed"}:
            raise ValidationError("benchmark block arm status must be succeeded or failed")
        if not isinstance(self.downstream_usage, ResourceUsage):
            raise TypeError("benchmark block downstream_usage must be ResourceUsage")
        receipts = _validated_resource_receipts(
            self.downstream_resource_receipts,
            label="benchmark block downstream",
        )
        object.__setattr__(
            self,
            "downstream_metadata",
            freeze_mapping(self.downstream_metadata),
        )
        if (
            isinstance(self.downstream_duration_ms, bool)
            or not isinstance(self.downstream_duration_ms, (int, float))
            or not math.isfinite(self.downstream_duration_ms)
            or self.downstream_duration_ms < 0
        ):
            raise ValidationError(
                "benchmark block downstream_duration_ms must be finite and non-negative"
            )
        object.__setattr__(self, "downstream_duration_ms", float(self.downstream_duration_ms))
        object.__setattr__(self, "downstream_resource_receipts", receipts)
        if self.status == "succeeded":
            if (
                self.intervention_record is None
                or self.benchmark_record is None
                or self.failure is not None
            ):
                raise ValidationError(
                    "successful benchmark block arm requires intervention and benchmark records"
                )
        elif self.benchmark_record is not None or self.failure is None:
            raise ValidationError(
                "failed benchmark block arm requires a failure and cannot contain a score"
            )

    @property
    def total_usage(self) -> ResourceUsage:
        intervention_usage = (
            self.intervention_record.usage
            if self.intervention_record is not None
            else ResourceUsage()
        )
        return intervention_usage + self.downstream_usage


@dataclass(frozen=True, slots=True)
class ControlledMultiArmBlockResult:
    """One complete attempt of the exact planned arm set on one shared state."""

    benchmark_id: str
    split: str
    example_id: str
    state_hash: str
    profile_id: str
    baseline_arm_id: str
    planned_arm_ids: tuple[str, ...]
    balance: BalanceReport
    arms: tuple[BenchmarkBlockArmExecutionRecord, ...]
    observed_dependency_hashes: Mapping[str, Mapping[str, str]]
    pre_action: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("benchmark_id", self.benchmark_id),
            ("split", self.split),
            ("example_id", self.example_id),
            ("state_hash", self.state_hash),
            ("profile_id", self.profile_id),
            ("baseline_arm_id", self.baseline_arm_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"controlled block {label} must be non-empty")
        planned = tuple(self.planned_arm_ids)
        arms = tuple(self.arms)
        if len(planned) < 2 or len(planned) != len(set(planned)):
            raise ValidationError("controlled block requires at least two unique planned arms")
        if self.baseline_arm_id not in planned:
            raise ValidationError("controlled block baseline must belong to the planned arm set")
        if tuple(arm.arm_id for arm in arms) != planned:
            raise ValidationError(
                "controlled block outcomes must exactly match planned arm IDs and order"
            )
        if not isinstance(self.balance, BalanceReport) or not self.balance.eligible:
            raise ValidationError("controlled block requires an eligible balance report")
        successful_metric_keys: tuple[str, ...] | None = None
        for arm in arms:
            record = arm.benchmark_record
            if record is not None:
                if record.example_id != self.example_id or record.state_hash != self.state_hash:
                    raise ValidationError(
                        "controlled block arm score differs from the shared example identity"
                    )
                keys = tuple(sorted(record.metrics))
                if successful_metric_keys is None:
                    successful_metric_keys = keys
                elif keys != successful_metric_keys:
                    raise ValidationError("controlled block successful arm metric schemas differ")
        object.__setattr__(self, "planned_arm_ids", planned)
        object.__setattr__(self, "arms", arms)
        object.__setattr__(
            self,
            "observed_dependency_hashes",
            freeze_mapping(self.observed_dependency_hashes),
        )
        object.__setattr__(self, "pre_action", freeze_mapping(self.pre_action))

    @property
    def all_arms_succeeded(self) -> bool:
        return all(arm.status == "succeeded" for arm in self.arms)

    @property
    def metric_deltas(self) -> Mapping[str, Mapping[str, float]]:
        """Return successful-arm effects against the one shared NO_OP outcome."""

        by_id = {arm.arm_id: arm for arm in self.arms}
        baseline = by_id[self.baseline_arm_id].benchmark_record
        if baseline is None:
            return freeze_mapping({})
        deltas: dict[str, Mapping[str, float]] = {}
        for arm in self.arms:
            if arm.arm_id == self.baseline_arm_id or arm.benchmark_record is None:
                continue
            deltas[arm.arm_id] = {
                key: arm.benchmark_record.metrics[key] - baseline.metrics[key]
                for key in baseline.metrics
            }
        return freeze_mapping(deltas)


class ControlledMultiArmBlockRunner:
    """Execute one exact multi-arm block without replaying the full split.

    The baseline is one explicitly declared ``SemanticFamily.NO_OP`` arm.  It is
    constructed and executed once, alongside each treatment on the same frozen
    pre-action state.  This prevents an N-arm experiment from silently becoming
    N-1 independently repeated two-arm studies.
    """

    def __init__(self, *, intervention_runner: InterventionRunner | None = None) -> None:
        self._intervention_runner = intervention_runner or InterventionRunner()
        self._balance = BalanceAuditor()

    def run_one(
        self,
        adapter: BenchmarkAdapter,
        split: str,
        example_id: str,
        profile: ComparisonProfile,
        arms: tuple[BenchmarkBlockArm, ...],
        *,
        planned_arm_ids: tuple[str, ...],
        baseline_arm_id: str,
        budget: ExecutionBudget | None = None,
        pre_action_observer: Callable[[FrozenAgentState], Mapping[str, Any]] | None = None,
    ) -> ControlledMultiArmBlockResult:
        normalized_arms = tuple(arms)
        planned = tuple(planned_arm_ids)
        _validate_block_inputs(
            adapter,
            split,
            example_id,
            profile,
            normalized_arms,
            planned,
            baseline_arm_id,
            budget,
        )
        _reject_shared_execution_instances(normalized_arms)
        before = _observed_fingerprints(normalized_arms)
        _require_equal_predictor_fingerprints(before)
        balance = self._audit(profile, normalized_arms, before)

        example = _resolve_example(adapter, split, example_id)
        state = example.visible_state(budget=budget)
        pre_action: Mapping[str, Any] = {}
        if pre_action_observer is not None:
            observed = pre_action_observer(state)
            if not isinstance(observed, Mapping) or not observed:
                raise ValidationError(
                    "pre_action_observer must return a non-empty canonical mapping"
                )
            try:
                pre_action = freeze_mapping(observed)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"pre_action_observer returned a non-canonical mapping: {exc}"
                ) from exc
        records = tuple(
            self._run_arm(
                adapter,
                example,
                state,
                arm,
                require_no_op=arm.comparison_arm.arm_id == baseline_arm_id,
            )
            for arm in normalized_arms
        )

        current = _observed_fingerprints(normalized_arms)
        if current != before:
            raise BalanceError(
                "runtime or downstream predictor identity changed during controlled block"
            )
        balance = self._audit(profile, normalized_arms, current)
        return ControlledMultiArmBlockResult(
            benchmark_id=str(adapter.benchmark_id),
            split=split,
            example_id=example.example_id,
            state_hash=state.state_hash,
            profile_id=profile.profile_id,
            baseline_arm_id=baseline_arm_id,
            planned_arm_ids=planned,
            balance=balance,
            arms=records,
            observed_dependency_hashes=current,
            pre_action=pre_action,
        )

    def _audit(
        self,
        profile: ComparisonProfile,
        arms: tuple[BenchmarkBlockArm, ...],
        fingerprints: Mapping[str, Mapping[str, str]],
    ) -> BalanceReport:
        report = self._balance.audit(
            profile,
            tuple(arm.comparison_arm for arm in arms),
            runtime_fingerprints=fingerprints,
        )
        report.require_eligible()
        return report

    def _run_arm(
        self,
        adapter: BenchmarkAdapter,
        example: BenchmarkExample,
        state: FrozenAgentState,
        arm: BenchmarkBlockArm,
        *,
        require_no_op: bool,
    ) -> BenchmarkBlockArmExecutionRecord:
        intervention_record: RunRecord | None = None
        downstream_usage = ResourceUsage()
        downstream_duration_ms = 0.0
        downstream_metadata: Mapping[str, Any] = {}
        downstream_resource_receipts: tuple[ResourceReceipt, ...] = ()
        monitor = arm.context.monitor or self._intervention_runner.monitor
        if monitor is not None and not isinstance(monitor, RunMonitor):
            raise TypeError("context.monitor must be a RunMonitor")
        prior_run_ids = (
            frozenset(snapshot.run_id for snapshot in monitor.list_snapshots())
            if monitor is not None
            else frozenset()
        )
        try:
            intervention_record = self._intervention_runner.run(
                arm.intervention,
                state,
                arm.context,
            )
            if require_no_op and any(
                operation.op is not PatchOp.NO_OP
                for operation in intervention_record.patch.operations
            ):
                raise BalanceError("declared controlled-block baseline did not produce NO_OP")
            transformed = intervention_record.patch.apply(state)
            started = time.perf_counter()
            downstream = arm.predictor.predict(transformed)
            downstream_duration_ms = (time.perf_counter() - started) * 1_000.0
            if not isinstance(downstream, DownstreamPrediction):
                raise TypeError(
                    "downstream predictor must return DownstreamPrediction, "
                    f"got {type(downstream).__name__}"
                )
            downstream_usage = downstream.usage
            downstream_metadata = downstream.metadata
            downstream_resource_receipts = downstream.resource_receipts
            _require_total_usage_within_budget(
                state,
                intervention_record=intervention_record,
                downstream_usage=downstream_usage,
            )
            benchmark_record = build_benchmark_example_record(
                adapter,
                example,
                state_hash=state.state_hash,
                prediction=downstream.prediction,
            )
        except ArmExecutionError as exc:
            downstream_resource_receipts = _failure_resource_receipts(
                exc,
                downstream_resource_receipts,
            )
            monitor_run_id = (
                intervention_record.monitor_run_id
                if intervention_record is not None
                else _find_new_monitor_run_id(
                    monitor,
                    prior_run_ids,
                    release_id=arm.intervention.release_id,
                    state_hash=state.state_hash,
                )
            )
            return BenchmarkBlockArmExecutionRecord(
                arm_id=arm.comparison_arm.arm_id,
                release_id=arm.intervention.release_id,
                status="failed",
                intervention_record=intervention_record,
                downstream_usage=downstream_usage,
                downstream_duration_ms=downstream_duration_ms,
                downstream_metadata=downstream_metadata,
                downstream_resource_receipts=downstream_resource_receipts,
                failure=ArmFailure(
                    category=type(exc).__name__,
                    message=str(exc),
                    monitor_run_id=monitor_run_id,
                ),
            )
        return BenchmarkBlockArmExecutionRecord(
            arm_id=arm.comparison_arm.arm_id,
            release_id=arm.intervention.release_id,
            status="succeeded",
            intervention_record=intervention_record,
            benchmark_record=benchmark_record,
            downstream_usage=downstream_usage,
            downstream_duration_ms=downstream_duration_ms,
            downstream_metadata=downstream_metadata,
            downstream_resource_receipts=downstream_resource_receipts,
        )


class PairedInterventionBenchmarkRunner:
    """Run two balanced intervention arms and score their downstream outcomes."""

    def __init__(self, *, intervention_runner: InterventionRunner | None = None) -> None:
        self._intervention_runner = intervention_runner or InterventionRunner()
        self._balance = BalanceAuditor()

    def run(
        self,
        adapter: BenchmarkAdapter,
        split: str,
        profile: ComparisonProfile,
        arms: tuple[BenchmarkInterventionArm, ...],
        *,
        budget: ExecutionBudget | None = None,
    ) -> PairedBenchmarkExecution:
        candidate_arms = tuple(arms)
        _validate_run_inputs(adapter, split, profile, candidate_arms, budget)
        by_role = {arm.role: arm for arm in candidate_arms}
        normalized_arms = (by_role["baseline"], by_role["treatment"])
        _reject_shared_execution_instances(normalized_arms)
        before = _observed_fingerprints(normalized_arms)
        _require_equal_predictor_fingerprints(before)
        balance = self._audit(profile, normalized_arms, before)

        examples = tuple(adapter.examples(split))
        _validate_examples(examples)
        execution_examples: list[PairedBenchmarkExecutionExample] = []
        paired_examples: list[PairedBenchmarkExampleRecord] = []

        for example in examples:
            state = example.visible_state(budget=budget)
            arm_records = tuple(
                self._run_arm(adapter, example, state, arm) for arm in normalized_arms
            )
            execution_example = PairedBenchmarkExecutionExample(
                example_id=example.example_id,
                state_hash=state.state_hash,
                arms=arm_records,
            )
            execution_examples.append(execution_example)
            if execution_example.complete:
                records_by_role = {record.role: record for record in arm_records}
                baseline = records_by_role["baseline"].benchmark_record
                treatment = records_by_role["treatment"].benchmark_record
                if baseline is None or treatment is None:  # pragma: no cover - dataclass invariant
                    raise RuntimeError("complete paired execution lost one benchmark record")
                paired_examples.append(
                    PairedBenchmarkExampleRecord(
                        example_id=example.example_id,
                        state_hash=state.state_hash,
                        baseline=baseline,
                        treatment=treatment,
                        metric_deltas={
                            key: treatment.metrics[key] - baseline.metrics[key]
                            for key in baseline.metrics
                        },
                    )
                )

            current = _observed_fingerprints(normalized_arms)
            if current != before:
                raise BalanceError(
                    "runtime or downstream predictor identity changed during benchmark execution"
                )
            balance = self._audit(profile, normalized_arms, current)

        paired_record = None
        if len(paired_examples) == len(execution_examples):
            baseline_metrics = _aggregate_metrics(
                tuple(record.baseline.metrics for record in paired_examples)
            )
            treatment_metrics = _aggregate_metrics(
                tuple(record.treatment.metrics for record in paired_examples)
            )
            paired_record = PairedBenchmarkRecord(
                benchmark_id=str(adapter.benchmark_id),
                split=split,
                examples=tuple(paired_examples),
                baseline_metrics=baseline_metrics,
                treatment_metrics=treatment_metrics,
                metric_deltas={
                    key: treatment_metrics[key] - baseline_metrics[key] for key in baseline_metrics
                },
            )

        return PairedBenchmarkExecution(
            benchmark_id=str(adapter.benchmark_id),
            split=split,
            profile_id=profile.profile_id,
            balance=balance,
            examples=tuple(execution_examples),
            observed_dependency_hashes=before,
            paired_record=paired_record,
        )

    def _audit(
        self,
        profile: ComparisonProfile,
        arms: tuple[BenchmarkInterventionArm, ...],
        fingerprints: Mapping[str, Mapping[str, str]],
    ) -> BalanceReport:
        report = self._balance.audit(
            profile,
            tuple(arm.comparison_arm for arm in arms),
            runtime_fingerprints=fingerprints,
        )
        report.require_eligible()
        return report

    def _run_arm(
        self,
        adapter: BenchmarkAdapter,
        example: BenchmarkExample,
        state: FrozenAgentState,
        arm: BenchmarkInterventionArm,
    ) -> BenchmarkArmExecutionRecord:
        intervention_record: RunRecord | None = None
        downstream_usage = ResourceUsage()
        downstream_duration_ms = 0.0
        downstream_metadata: Mapping[str, Any] = {}
        downstream_resource_receipts: tuple[ResourceReceipt, ...] = ()
        monitor = arm.context.monitor or self._intervention_runner.monitor
        if monitor is not None and not isinstance(monitor, RunMonitor):
            raise TypeError("context.monitor must be a RunMonitor")
        prior_run_ids = (
            frozenset(snapshot.run_id for snapshot in monitor.list_snapshots())
            if monitor is not None
            else frozenset()
        )
        try:
            intervention_record = self._intervention_runner.run(
                arm.intervention,
                state,
                arm.context,
            )
            transformed = intervention_record.patch.apply(state)
            started = time.perf_counter()
            downstream = arm.predictor.predict(transformed)
            downstream_duration_ms = (time.perf_counter() - started) * 1_000.0
            if not isinstance(downstream, DownstreamPrediction):
                raise TypeError(
                    "downstream predictor must return DownstreamPrediction, "
                    f"got {type(downstream).__name__}"
                )
            downstream_usage = downstream.usage
            downstream_metadata = downstream.metadata
            downstream_resource_receipts = downstream.resource_receipts
            _require_total_usage_within_budget(
                state,
                intervention_record=intervention_record,
                downstream_usage=downstream_usage,
            )
            benchmark_record = build_benchmark_example_record(
                adapter,
                example,
                state_hash=state.state_hash,
                prediction=downstream.prediction,
            )
        except ArmExecutionError as exc:
            downstream_resource_receipts = _failure_resource_receipts(
                exc,
                downstream_resource_receipts,
            )
            monitor_run_id = (
                intervention_record.monitor_run_id
                if intervention_record is not None
                else _find_new_monitor_run_id(
                    monitor,
                    prior_run_ids,
                    release_id=arm.intervention.release_id,
                    state_hash=state.state_hash,
                )
            )
            return BenchmarkArmExecutionRecord(
                role=arm.role,
                arm_id=arm.comparison_arm.arm_id,
                status="failed",
                intervention_record=intervention_record,
                downstream_usage=downstream_usage,
                downstream_duration_ms=downstream_duration_ms,
                downstream_metadata=downstream_metadata,
                downstream_resource_receipts=downstream_resource_receipts,
                failure=ArmFailure(
                    category=type(exc).__name__,
                    message=str(exc),
                    monitor_run_id=monitor_run_id,
                ),
            )
        return BenchmarkArmExecutionRecord(
            role=arm.role,
            arm_id=arm.comparison_arm.arm_id,
            status="succeeded",
            intervention_record=intervention_record,
            benchmark_record=benchmark_record,
            downstream_usage=downstream_usage,
            downstream_duration_ms=downstream_duration_ms,
            downstream_metadata=downstream_metadata,
            downstream_resource_receipts=downstream_resource_receipts,
        )


def _require_total_usage_within_budget(
    state: FrozenAgentState,
    *,
    intervention_record: RunRecord,
    downstream_usage: ResourceUsage,
) -> None:
    """Enforce one budget across intervention and downstream agent execution.

    The intervention runtime owns its own reservation ledger, while a
    downstream predictor reports usage only after its side effects have
    occurred.  Replaying both actual usage records through a fresh ledger makes
    the experiment-level budget cover the complete arm without double-charging
    either runtime.  Any overage remains an infrastructure-level cell failure,
    not an algorithm-quality outcome.
    """

    ledger = BudgetLedger(state.budget)
    ledger.charge_actual(intervention_record.usage + downstream_usage)


def _validated_resource_receipts(
    values: tuple[ResourceReceipt, ...],
    *,
    label: str,
) -> tuple[ResourceReceipt, ...]:
    receipts = tuple(values)
    if any(not isinstance(receipt, ResourceReceipt) for receipt in receipts):
        raise TypeError(f"{label} resource receipts must contain ResourceReceipt values")
    return receipts


def _failure_resource_receipts(
    error: ArmExecutionError,
    current: tuple[ResourceReceipt, ...],
) -> tuple[ResourceReceipt, ...]:
    receipt = getattr(error, "resource_receipt", None)
    if receipt is None:
        return current
    if not isinstance(receipt, ResourceReceipt):
        raise TypeError("arm failure resource_receipt must be a ResourceReceipt")
    return current if receipt in current else (*current, receipt)


def _validate_run_inputs(
    adapter: BenchmarkAdapter,
    split: str,
    profile: ComparisonProfile,
    arms: tuple[BenchmarkInterventionArm, ...],
    budget: ExecutionBudget | None,
) -> None:
    benchmark_id = getattr(adapter, "benchmark_id", None)
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValidationError("adapter.benchmark_id must be a non-empty string")
    if getattr(adapter, "pairing_mode", None) != "stateless_score_twice":
        raise ValidationError(
            "paired intervention benchmark execution requires pairing_mode="
            "'stateless_score_twice'; use the atomic external replay runner for "
            "stateful arenas"
        )
    if not isinstance(split, str) or not split.strip():
        raise ValidationError("benchmark split must be a non-empty string")
    if not isinstance(profile, ComparisonProfile):
        raise TypeError("profile must be a ComparisonProfile")
    if not all(isinstance(arm, BenchmarkInterventionArm) for arm in arms):
        raise TypeError("arms must be BenchmarkInterventionArm instances")
    if len(arms) != 2 or len({arm.comparison_arm.arm_id for arm in arms}) != 2:
        raise ValidationError("paired intervention benchmark runner requires two unique arms")
    if {arm.role for arm in arms} != {"baseline", "treatment"}:
        raise ValidationError(
            "paired intervention benchmark runner requires baseline and treatment roles"
        )
    if budget is not None and not isinstance(budget, ExecutionBudget):
        raise TypeError("budget must be an ExecutionBudget or None")


def _validate_block_inputs(
    adapter: BenchmarkAdapter,
    split: str,
    example_id: str,
    profile: ComparisonProfile,
    arms: tuple[BenchmarkBlockArm, ...],
    planned_arm_ids: tuple[str, ...],
    baseline_arm_id: str,
    budget: ExecutionBudget | None,
) -> None:
    benchmark_id = getattr(adapter, "benchmark_id", None)
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValidationError("adapter.benchmark_id must be a non-empty string")
    if getattr(adapter, "pairing_mode", None) != "stateless_score_twice":
        raise ValidationError(
            "controlled multi-arm blocks require pairing_mode='stateless_score_twice'; "
            "stateful arenas require an atomic external replay bridge"
        )
    if not isinstance(split, str) or not split.strip():
        raise ValidationError("controlled block split must be non-empty")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValidationError("controlled block example_id must be non-empty")
    if not isinstance(profile, ComparisonProfile):
        raise TypeError("profile must be a ComparisonProfile")
    if not all(isinstance(arm, BenchmarkBlockArm) for arm in arms):
        raise TypeError("controlled block arms must be BenchmarkBlockArm instances")
    if len(planned_arm_ids) < 2 or len(planned_arm_ids) != len(set(planned_arm_ids)):
        raise ValidationError("planned_arm_ids must contain at least two unique arms")
    observed_arm_ids = tuple(arm.comparison_arm.arm_id for arm in arms)
    if observed_arm_ids != planned_arm_ids:
        raise ValidationError(
            "controlled block arm IDs and order must exactly match planned_arm_ids"
        )
    if baseline_arm_id not in planned_arm_ids:
        raise ValidationError("controlled block baseline_arm_id is outside planned_arm_ids")
    no_op_ids = tuple(
        arm.comparison_arm.arm_id for arm in arms if arm.intervention.family is SemanticFamily.NO_OP
    )
    if no_op_ids != (baseline_arm_id,):
        raise ValidationError(
            "controlled block requires exactly one declared NO_OP arm matching baseline_arm_id"
        )
    if budget is not None and not isinstance(budget, ExecutionBudget):
        raise TypeError("budget must be an ExecutionBudget or None")


def _resolve_example(
    adapter: BenchmarkAdapter,
    split: str,
    example_id: str,
) -> BenchmarkExample:
    resolver = getattr(adapter, "example", None)
    if callable(resolver):
        example = resolver(split, example_id)
        if not isinstance(example, BenchmarkExample):
            raise ValidationError("adapter.example() must return a BenchmarkExample")
        if example.example_id != example_id:
            raise ValidationError("adapter.example() returned another example identity")
        return example
    examples = tuple(adapter.examples(split))
    _validate_examples(examples)
    matches = tuple(example for example in examples if example.example_id == example_id)
    if len(matches) != 1:
        raise ValidationError(
            f"benchmark split must contain exactly one example with ID {example_id!r}"
        )
    return matches[0]


def _validate_examples(examples: tuple[BenchmarkExample, ...]) -> None:
    if not examples:
        raise ValidationError("benchmark split must contain at least one example")
    if not all(isinstance(example, BenchmarkExample) for example in examples):
        raise ValidationError("adapter.examples() must return BenchmarkExample instances")
    identifiers = [example.example_id for example in examples]
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers):
        raise ValidationError("benchmark example IDs must be non-empty strings")
    if len(identifiers) != len(set(identifiers)):
        raise ValidationError("benchmark example IDs must be unique")


def _observed_fingerprints(
    arms: tuple[BenchmarkInterventionArm | BenchmarkBlockArm, ...],
) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for arm in arms:
        arm_id = arm.comparison_arm.arm_id
        fingerprints = context_dependency_fingerprints(arm.context)
        fingerprint = arm.predictor.fingerprint()
        if not isinstance(fingerprint, Mapping) or not fingerprint:
            raise ValidationError("downstream predictor fingerprint must be a non-empty mapping")
        try:
            fingerprints[_PREDICTOR_FINGERPRINT_SLOT] = content_hash(freeze_mapping(fingerprint))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"downstream predictor fingerprint must be canonical: {exc}"
            ) from exc
        observed[arm_id] = fingerprints
    return observed


def _require_equal_predictor_fingerprints(
    observed: Mapping[str, Mapping[str, str]],
) -> None:
    hashes = {fingerprints[_PREDICTOR_FINGERPRINT_SLOT] for fingerprints in observed.values()}
    if len(hashes) != 1:
        raise BalanceError("downstream predictor fingerprint differs across comparison arms")


def _reject_shared_execution_instances(
    arms: tuple[BenchmarkInterventionArm | BenchmarkBlockArm, ...],
) -> None:
    if len({id(arm.context) for arm in arms}) != len(arms):
        raise BalanceError("comparison arms require independently constructed contexts")
    if len({id(arm.predictor) for arm in arms}) != len(arms):
        raise BalanceError("comparison arms require independently constructed predictors")

    owners: dict[int, set[str]] = {}
    ledger_owners: dict[int, set[str]] = {}
    for arm in arms:
        arm_id = arm.comparison_arm.arm_id
        if isinstance(arm.context.budget_ledger, BudgetLedger):
            ledger_owners.setdefault(id(arm.context.budget_ledger), set()).add(arm_id)
        roots = [
            *(getattr(arm.context, slot) for slot in _CONTEXT_RUNTIME_SLOTS),
            arm.predictor,
        ]
        for index, root in enumerate(roots):
            if root is None:
                continue
            graph = inspect_object_graph(
                root,
                root_path=f"arm[{arm_id}].runtime[{index}]",
                terminal_types=(BudgetLedger,),
                traverse_runtime_delegates=True,
            )
            for node in graph:
                if is_runtime_dependency(node.value):
                    owners.setdefault(id(node.value), set()).add(arm_id)
                if isinstance(node.value, BudgetLedger):
                    ledger_owners.setdefault(id(node.value), set()).add(arm_id)

    shared = [sorted(arm_ids) for arm_ids in owners.values() if len(arm_ids) > 1]
    if shared:
        details = ", ".join("/".join(arm_ids) for arm_ids in shared)
        raise BalanceError(
            f"runtime dependency objects must be independently constructed across arms: {details}"
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


def _aggregate_metrics(
    rows: tuple[Mapping[str, float], ...],
) -> Mapping[str, float]:
    if not rows:
        raise ValidationError("cannot aggregate an empty metric collection")
    keys = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != keys for row in rows[1:]):
        raise ValidationError("metric keys differ across paired benchmark examples")
    count = len(rows)
    aggregate: dict[str, float] = {}
    for key in keys:
        value = math.fsum(float(row[key]) / count for row in rows)
        if not math.isfinite(value):
            raise ValidationError(f"aggregate metric {key!r} is not finite")
        aggregate[key] = value
    return freeze_mapping(aggregate)


__all__ = [
    "BenchmarkArmExecutionRecord",
    "BenchmarkBlockArm",
    "BenchmarkBlockArmExecutionRecord",
    "BenchmarkInterventionArm",
    "ControlledMultiArmBlockResult",
    "ControlledMultiArmBlockRunner",
    "DownstreamPrediction",
    "DownstreamPredictor",
    "PairedBenchmarkExecution",
    "PairedBenchmarkExecutionExample",
    "PairedInterventionBenchmarkRunner",
]
