"""MIT-source-compatible Plan-and-Budget allocation and prompt materialization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import (
    ArtifactError,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.enums import StrEnum

P_AND_B_UPSTREAM_REVISION = "2ca2d90c6d3d630ec2b11cc22e0f56751f7f2b7e"


class DecaySchedule(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"

    @classmethod
    def parse(cls, value: str | DecaySchedule) -> DecaySchedule:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            raise ValidationError(f"unsupported Plan-and-Budget decay: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class CachedPlanBudgetRecord:
    record_id: str
    query: str
    steps: tuple[str, ...]
    credits: tuple[float, ...]
    level: int
    audit_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "credits", tuple(float(item) for item in self.credits))
        object.__setattr__(self, "audit_warnings", tuple(self.audit_warnings))
        if not self.record_id:
            raise ArtifactError("Plan-and-Budget cached record requires id")
        if not self.query:
            raise ArtifactError("Plan-and-Budget cached record requires query")
        if not self.steps:
            raise ArtifactError("Plan-and-Budget cached record requires steps")
        if len(self.steps) != len(self.credits):
            raise ArtifactError("Plan-and-Budget steps/credits length mismatch")
        if any(not math.isfinite(item) or item < 0 for item in self.credits):
            raise ArtifactError("Plan-and-Budget credits must be finite and non-negative")
        if sum(self.credits) <= 0:
            raise ArtifactError("Plan-and-Budget credits must have positive mass")
        if not 1 <= self.level <= 5:
            raise ArtifactError("Plan-and-Budget level must be in [1, 5]")


def load_cached_record(row: Mapping[str, object]) -> CachedPlanBudgetRecord:
    """Project an upstream row to a solver-visible view, physically dropping gold."""

    record_id = str(row.get("id", row.get("query_id", "")))
    query = str(row.get("query", row.get("problem", "")))
    raw_steps = row.get("steps", ())
    raw_credits = row.get("credits", ())
    if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
        raise ArtifactError("Plan-and-Budget steps must be a sequence")
    if isinstance(raw_credits, (str, bytes)) or not isinstance(raw_credits, Sequence):
        raise ArtifactError("Plan-and-Budget credits must be a sequence")
    steps = tuple(str(item) for item in raw_steps)
    try:
        credits = tuple(float(item) for item in raw_credits)
        level = int(row.get("level", 0))
    except (TypeError, ValueError) as exc:
        raise ArtifactError("invalid Plan-and-Budget credit/level value") from exc
    warnings = []
    if not 2 <= len(steps) <= 5:
        warnings.append("cached_step_count_outside_prompt_intended_2_to_5")
    if not math.isclose(sum(credits), 100.0, rel_tol=0.0, abs_tol=1e-9):
        warnings.append("cached_credit_sum_not_100")
    if any(key in row for key in ("reference", "solution", "answer", "gold")):
        warnings.append("gold_fields_removed_from_solver_view")
    return CachedPlanBudgetRecord(
        record_id=record_id,
        query=query,
        steps=steps,
        credits=credits,
        level=level,
        audit_warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class PlanAndBudgetCache:
    """Solver-only records loaded from a caller-pinned native cache envelope.

    Byte verification does not establish authorship, licensing, the generating
    model, or generation cost. Keep that provenance in the trusted source receipt.
    """

    records: Mapping[str, CachedPlanBudgetRecord]
    artifact_sha256: str
    verification: str = field(default="caller_declared_digest", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_sha256, str) or (
            len(self.artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.artifact_sha256)
        ):
            raise ArtifactError("Plan-and-Budget cache requires a SHA-256 digest")
        copied = dict(self.records)
        if not copied or any(
            not isinstance(record, CachedPlanBudgetRecord) or key != record.record_id
            for key, record in copied.items()
        ):
            raise ArtifactError("Plan-and-Budget cache requires records keyed by exact id")
        object.__setattr__(self, "records", MappingProxyType(copied))

    @classmethod
    def from_native_json(
        cls, path: str | Path, *, expected_sha256: str, max_bytes: int = 8 * 1024 * 1024
    ) -> PlanAndBudgetCache:
        """Load the upstream ``{last_question_total, stats: [...]}`` format.

        References, solutions and answers are physically absent from the returned
        records. This loader does not download assets or run their generator.
        """
        payload = load_pinned_json_mapping(
            path, expected_sha256=expected_sha256, max_bytes=max_bytes
        )
        if set(payload) != {"last_question_total", "stats"}:
            raise ArtifactError("Plan-and-Budget native cache envelope differs")
        if type(payload["last_question_total"]) not in (int, float) or payload["last_question_total"] < 0:
            raise ArtifactError("Plan-and-Budget native cache timing field differs")
        rows = payload["stats"]
        if not isinstance(rows, list) or not rows:
            raise ArtifactError("Plan-and-Budget native cache stats must be a non-empty list")
        records = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ArtifactError("Plan-and-Budget native cache row must be an object")
            required = {"id", "query", "steps", "credits", "level"}
            if not required <= row.keys() or row.keys() - required - {
                "reference", "solution", "answer", "gold"
            }:
                raise ArtifactError("Plan-and-Budget native cache row fields differ")
            if not all(isinstance(row[key], str) and row[key].strip() for key in ("id", "query")):
                raise ArtifactError("Plan-and-Budget native cache id/query must be non-empty strings")
            if not isinstance(row["steps"], list) or any(
                not isinstance(step, str) or not step.strip() for step in row["steps"]
            ):
                raise ArtifactError("Plan-and-Budget native steps must be non-empty strings")
            if not isinstance(row["credits"], list) or any(
                type(credit) not in (int, float) for credit in row["credits"]
            ) or type(row["level"]) is not int:
                raise ArtifactError("Plan-and-Budget native credit/level types differ")
            record = load_cached_record(row)
            if record.record_id in records:
                raise ArtifactError("duplicate Plan-and-Budget native cache record id")
            records[record.record_id] = record
        result = cls(records, expected_sha256.lower())
        object.__setattr__(result, "verification", "caller_sha256_verified")
        return result


def _schedule_weights(schedule: DecaySchedule, count: int) -> tuple[float, ...]:
    values = []
    for index in range(count):
        if schedule is DecaySchedule.CONSTANT:
            weight = 1.0
        elif schedule is DecaySchedule.LINEAR:
            weight = count - index
        elif schedule is DecaySchedule.POLYNOMIAL:
            weight = float((count - index) ** 2)
        elif schedule is DecaySchedule.EXPONENTIAL:
            weight = 0.9**index
        elif schedule is DecaySchedule.COSINE:
            weight = (
                0.5 * (1.0 + math.cos(math.pi * index / (count - 1))) + 0.1 if count > 1 else 1.0
            )
        else:  # pragma: no cover - enum exhaustiveness
            raise ValidationError(f"unsupported schedule: {schedule}")
        values.append(float(weight))
    return tuple(values)


def allocate_tokens(
    credits: Sequence[float],
    total_tokens: int,
    schedule: str | DecaySchedule = DecaySchedule.CONSTANT,
) -> tuple[int, ...]:
    """Pure-Python port of upstream ``allocate_tokens``.

    The upstream name says tokens, although its runtime inserts the allocations
    as soft *word* budgets.  The caller must preserve that semantic distinction.
    """

    if total_tokens < 0:
        raise ValidationError("total_tokens must be non-negative")
    values = tuple(float(item) for item in credits)
    if not values:
        raise ValidationError("credits cannot be empty")
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise ValidationError("credits must be finite and non-negative")
    schedule_value = DecaySchedule.parse(schedule)
    weights = _schedule_weights(schedule_value, len(values))
    combined = tuple(credit * weight for credit, weight in zip(values, weights, strict=True))
    mass = sum(combined)
    if mass <= 0:
        raise ValidationError("weighted credits must have positive mass")
    normalized = tuple(item / mass for item in combined)
    allocations = [math.floor(item * total_tokens) for item in normalized]
    difference = total_tokens - sum(allocations)
    # NumPy argsort tie behavior was not an intentional method contribution.
    # Stable index ordering makes the port deterministic while preserving the
    # upstream descending-combined allocation rule.
    order = sorted(range(len(normalized)), key=lambda index: (-normalized[index], index))
    for index in order[:difference]:
        allocations[index] += 1
    return tuple(allocations)


def render_soft_budget_instruction(
    record: CachedPlanBudgetRecord,
    allocations: Sequence[int],
) -> str:
    if len(record.steps) != len(allocations):
        raise ValidationError("steps/allocations length mismatch")
    decomposed = "\n\n".join(
        f"{step} Please only think a little, and directly solve it using up to {budget} words."
        for step, budget in zip(record.steps, allocations, strict=True)
    )
    return (
        f"The problem has difficulty level {record.level} out of 5. "
        "You may follow these sub-questions or use another method that works best "
        "towards the final answer. All credit is awarded for the correct final "
        "answer.\n\n"
        f"{decomposed}"
    )


class PlanAndBudgetCachedSoftBudget:
    method_id = "plan_and_budget"
    release_id = "plan.plan_and_budget.native_cached_soft_budget"
    family = SemanticFamily.DECOMPOSE

    def __init__(
        self,
        records: Mapping[str, CachedPlanBudgetRecord] | PlanAndBudgetCache,
        *,
        total_word_budget: int,
        schedule: str | DecaySchedule = DecaySchedule.CONSTANT,
    ) -> None:
        if total_word_budget < 0:
            raise ValidationError("total_word_budget must be non-negative")
        self.cache_sha256 = records.artifact_sha256 if isinstance(records, PlanAndBudgetCache) else None
        self.cache_verification = (
            records.verification if isinstance(records, PlanAndBudgetCache) else "caller_mapping_unverified"
        )
        self.records = dict(records.records if isinstance(records, PlanAndBudgetCache) else records)
        self.total_word_budget = total_word_budget
        self.schedule = DecaySchedule.parse(schedule)

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        record_id = state.task_id or str(state.metadata.get("query_id", ""))
        try:
            record = self.records[record_id]
        except KeyError as exc:
            raise ArtifactError(f"no cached Plan-and-Budget record for {record_id!r}") from exc
        if record.query != state.query:
            raise ArtifactError(f"Plan-and-Budget cached query mismatch for {record_id!r}")
        allocations = allocate_tokens(
            record.credits,
            self.total_word_budget,
            self.schedule,
        )
        instruction = render_soft_budget_instruction(record, allocations)
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "cached_plan_budget_record",
                    {
                        "record_id": record.record_id,
                        "steps": record.steps,
                        "credits": record.credits,
                        "credit_sum": sum(record.credits),
                        "level": record.level,
                        "audit_warnings": record.audit_warnings,
                        "gold_visible_to_solver": False,
                        "cache_artifact_sha256": self.cache_sha256,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "soft_word_budget_allocation",
                    {
                        "schedule": self.schedule.value,
                        "total_budget": self.total_word_budget,
                        "budget_unit": "soft_words",
                        "allocation_tie_policy": "stable_source_index",
                        "allocations": allocations,
                        "decoder_hard_cap_applied": False,
                        "instruction": instruction,
                    },
                ),
            ),
            counters={
                "optimizer_model_calls": 0,
                "planned_steps": len(record.steps),
                "allocated_soft_words": sum(allocations),
            },
            warnings=record.audit_warnings,
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.APPEND_INSTRUCTION,
                    value=instruction,
                    target="instruction",
                    rationale="Plan-and-Budget cached weighted soft word budgets",
                ),
            ),
            trace=trace,
            implementation_identity=f"minimal_mit_port:junhongmit/P-and-B@{P_AND_B_UPSTREAM_REVISION}",
            profile_kind="official_native",
            metadata={
                "implementation_origin": "official_mit_minimal_port",
                "cached_decomposition": True,
                "budget_unit": "soft_words",
                "allocation_tie_policy": "stable_source_index",
                "numpy_tie_order_parity_claimed": False,
                "hard_decoder_cap": False,
                "gold_fields_removed": True,
                "cache_artifact_sha256": self.cache_sha256,
                "cache_verification": self.cache_verification,
                "historical_cache_generation_cost": "unknown_not_free",
                "paper_result_parity_claimed": False,
            },
        )
