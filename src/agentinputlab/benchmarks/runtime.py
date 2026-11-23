"""Fail-closed batch execution for unified benchmark adapters.

Prediction runners receive only the evaluator-blind :class:`FrozenAgentState`.
The original :class:`BenchmarkExample`, including evaluator gold, crosses the
boundary only when the adapter scores a completed prediction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, TypeAlias

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import (
    canonical_json,
    deep_freeze_jsonable,
    freeze_mapping,
)
from agentinputlab.contracts.state import ExecutionBudget, FrozenAgentState

from .base import BenchmarkAdapter, BenchmarkExample

PredictionRunner: TypeAlias = Callable[[FrozenAgentState], Any]


@dataclass(frozen=True, slots=True)
class BenchmarkExampleRecord:
    """One prediction, its evaluator-produced metrics, and scorer provenance."""

    example_id: str
    state_hash: str
    prediction: Any = field(repr=False)
    metrics: Mapping[str, float]
    evaluation_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_text(self.example_id, field_name="example_id")
        _require_state_hash(self.state_hash)
        try:
            frozen_prediction = deep_freeze_jsonable(self.prediction)
            canonical_json(frozen_prediction)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"prediction for example {self.example_id!r} must be canonically serializable: "
                f"{exc}"
            ) from exc
        object.__setattr__(self, "prediction", frozen_prediction)
        object.__setattr__(
            self,
            "metrics",
            _validated_metrics(self.metrics, example_id=self.example_id),
        )
        object.__setattr__(
            self,
            "evaluation_provenance",
            _validated_evaluation_provenance(
                self.evaluation_provenance,
                example_id=self.example_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkBatchRecord:
    """Complete batch result for one benchmark split and prediction runner."""

    benchmark_id: str
    split: str
    examples: tuple[BenchmarkExampleRecord, ...]
    aggregate_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.benchmark_id, field_name="benchmark_id")
        _require_nonempty_text(self.split, field_name="split")
        examples = tuple(self.examples)
        if not examples:
            raise ValidationError("benchmark batch must contain at least one example")
        if not all(isinstance(record, BenchmarkExampleRecord) for record in examples):
            raise ValidationError("batch examples must be BenchmarkExampleRecord instances")
        _require_unique_example_ids(record.example_id for record in examples)
        expected = _metric_keys(examples[0].metrics)
        for record in examples[1:]:
            _require_same_metric_keys(expected, record.metrics, example_id=record.example_id)
        validated_aggregate = _validated_metrics(
            self.aggregate_metrics,
            example_id="<aggregate>",
        )
        _require_same_metric_keys(expected, validated_aggregate, example_id="<aggregate>")
        computed_aggregate = _aggregate_metrics(tuple(record.metrics for record in examples))
        _require_same_metric_values(
            computed_aggregate,
            validated_aggregate,
            record_name="batch aggregate",
        )
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "aggregate_metrics", validated_aggregate)

    @property
    def example_count(self) -> int:
        return len(self.examples)


@dataclass(frozen=True, slots=True)
class PairedBenchmarkExampleRecord:
    """Baseline and treatment outcomes sharing one pre-action state identity."""

    example_id: str
    state_hash: str
    baseline: BenchmarkExampleRecord
    treatment: BenchmarkExampleRecord
    metric_deltas: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.example_id, field_name="example_id")
        _require_state_hash(self.state_hash)
        if not isinstance(self.baseline, BenchmarkExampleRecord) or not isinstance(
            self.treatment, BenchmarkExampleRecord
        ):
            raise ValidationError("paired outcomes must be BenchmarkExampleRecord instances")
        for arm_name, record in (
            ("baseline", self.baseline),
            ("treatment", self.treatment),
        ):
            if record.example_id != self.example_id:
                raise ValidationError(f"{arm_name} example ID differs from paired example ID")
            if record.state_hash != self.state_hash:
                raise ValidationError(f"{arm_name} state hash differs from paired state hash")
        expected = _metric_keys(self.baseline.metrics)
        _require_same_metric_keys(expected, self.treatment.metrics, example_id=self.example_id)
        deltas = _validated_metrics(self.metric_deltas, example_id=self.example_id)
        _require_same_metric_keys(expected, deltas, example_id=self.example_id)
        for key in expected:
            expected_delta = self.treatment.metrics[key] - self.baseline.metrics[key]
            if deltas[key] != expected_delta:
                raise ValidationError(
                    f"metric delta {key!r} for example {self.example_id!r} is inconsistent"
                )
        object.__setattr__(self, "metric_deltas", deltas)


@dataclass(frozen=True, slots=True)
class PairedBenchmarkRecord:
    """Aggregated baseline/treatment comparison over a benchmark split."""

    benchmark_id: str
    split: str
    examples: tuple[PairedBenchmarkExampleRecord, ...]
    baseline_metrics: Mapping[str, float]
    treatment_metrics: Mapping[str, float]
    metric_deltas: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.benchmark_id, field_name="benchmark_id")
        _require_nonempty_text(self.split, field_name="split")
        examples = tuple(self.examples)
        if not examples:
            raise ValidationError("paired benchmark must contain at least one example")
        if not all(isinstance(record, PairedBenchmarkExampleRecord) for record in examples):
            raise ValidationError("paired examples must be PairedBenchmarkExampleRecord instances")
        _require_unique_example_ids(record.example_id for record in examples)
        expected = _metric_keys(examples[0].metric_deltas)
        for record in examples[1:]:
            _require_same_metric_keys(expected, record.metric_deltas, example_id=record.example_id)
        baseline = _validated_metrics(self.baseline_metrics, example_id="<baseline aggregate>")
        treatment = _validated_metrics(
            self.treatment_metrics,
            example_id="<treatment aggregate>",
        )
        deltas = _validated_metrics(self.metric_deltas, example_id="<paired aggregate>")
        for name, metrics in (
            ("baseline aggregate", baseline),
            ("treatment aggregate", treatment),
            ("paired aggregate", deltas),
        ):
            _require_same_metric_keys(expected, metrics, example_id=f"<{name}>")
        computed_baseline = _aggregate_metrics(
            tuple(record.baseline.metrics for record in examples)
        )
        computed_treatment = _aggregate_metrics(
            tuple(record.treatment.metrics for record in examples)
        )
        _require_same_metric_values(
            computed_baseline,
            baseline,
            record_name="baseline aggregate",
        )
        _require_same_metric_values(
            computed_treatment,
            treatment,
            record_name="treatment aggregate",
        )
        for key in expected:
            expected_delta = treatment[key] - baseline[key]
            if deltas[key] != expected_delta:
                raise ValidationError(f"aggregate metric delta {key!r} is inconsistent")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "baseline_metrics", baseline)
        object.__setattr__(self, "treatment_metrics", treatment)
        object.__setattr__(self, "metric_deltas", deltas)

    @property
    def example_count(self) -> int:
        return len(self.examples)


class BenchmarkBatchRunner:
    """Execute an evaluator-blind prediction callback over one benchmark split."""

    def run(
        self,
        adapter: BenchmarkAdapter,
        split: str,
        predict: PredictionRunner,
        *,
        budget: ExecutionBudget | None = None,
    ) -> BenchmarkBatchRecord:
        benchmark_id, examples = _prepare_run(adapter, split, predict, budget)
        records = tuple(
            _run_example(adapter, example, predict, budget=budget) for example in examples
        )
        return BenchmarkBatchRecord(
            benchmark_id=benchmark_id,
            split=split,
            examples=records,
            aggregate_metrics=_aggregate_metrics(tuple(record.metrics for record in records)),
        )


class PairedBenchmarkRunner:
    """Run baseline and treatment callbacks on identical stateless examples.

    Stateful external arenas must expose an atomic restore/evaluate-pair seam
    and use a dedicated replay runner.  Scoring the two arms sequentially proves
    only that their optimizer-visible state hashes match; it cannot prove that
    hidden environment or user-simulator state was restored.
    """

    def run(
        self,
        adapter: BenchmarkAdapter,
        split: str,
        baseline: PredictionRunner,
        treatment: PredictionRunner,
        *,
        budget: ExecutionBudget | None = None,
    ) -> PairedBenchmarkRecord:
        if not callable(treatment):
            raise TypeError("treatment prediction runner must be callable")
        pairing_mode = getattr(adapter, "pairing_mode", None)
        if pairing_mode != "stateless_score_twice":
            raise ValidationError(
                "PairedBenchmarkRunner requires an explicit pairing_mode="
                "'stateless_score_twice'; "
                "stateful external arenas require an atomic restore/evaluate-pair bridge "
                "with replay receipts"
            )
        benchmark_id, examples = _prepare_run(adapter, split, baseline, budget)

        records: list[PairedBenchmarkExampleRecord] = []
        for example in examples:
            state = example.visible_state(budget=budget)
            baseline_record = _score_prediction(adapter, example, state, baseline(state))
            treatment_record = _score_prediction(adapter, example, state, treatment(state))
            if baseline_record.state_hash != treatment_record.state_hash:
                raise RuntimeError("paired callbacks did not share one input state identity")
            _require_same_metric_keys(
                _metric_keys(baseline_record.metrics),
                treatment_record.metrics,
                example_id=example.example_id,
            )
            records.append(
                PairedBenchmarkExampleRecord(
                    example_id=example.example_id,
                    state_hash=state.state_hash,
                    baseline=baseline_record,
                    treatment=treatment_record,
                    metric_deltas={
                        key: treatment_record.metrics[key] - baseline_record.metrics[key]
                        for key in baseline_record.metrics
                    },
                )
            )

        paired = tuple(records)
        baseline_metrics = _aggregate_metrics(tuple(record.baseline.metrics for record in paired))
        treatment_metrics = _aggregate_metrics(tuple(record.treatment.metrics for record in paired))
        return PairedBenchmarkRecord(
            benchmark_id=benchmark_id,
            split=split,
            examples=paired,
            baseline_metrics=baseline_metrics,
            treatment_metrics=treatment_metrics,
            metric_deltas={
                key: treatment_metrics[key] - baseline_metrics[key] for key in baseline_metrics
            },
        )


def _prepare_run(
    adapter: BenchmarkAdapter,
    split: str,
    predict: PredictionRunner,
    budget: ExecutionBudget | None,
) -> tuple[str, tuple[BenchmarkExample, ...]]:
    benchmark_id = getattr(adapter, "benchmark_id", None)
    _require_nonempty_text(benchmark_id, field_name="adapter.benchmark_id")
    _require_nonempty_text(split, field_name="split")
    if not callable(predict):
        raise TypeError("prediction runner must be callable")
    if budget is not None and not isinstance(budget, ExecutionBudget):
        raise TypeError("budget must be an ExecutionBudget or None")

    examples = tuple(adapter.examples(split))
    if not examples:
        raise ValidationError("benchmark split must contain at least one example")
    if not all(isinstance(example, BenchmarkExample) for example in examples):
        raise ValidationError("adapter.examples() must return BenchmarkExample instances")
    for example in examples:
        _require_nonempty_text(example.example_id, field_name="example.example_id")
    _require_unique_example_ids(example.example_id for example in examples)
    return benchmark_id, examples


def _run_example(
    adapter: BenchmarkAdapter,
    example: BenchmarkExample,
    predict: PredictionRunner,
    *,
    budget: ExecutionBudget | None,
) -> BenchmarkExampleRecord:
    state = example.visible_state(budget=budget)
    return _score_prediction(adapter, example, state, predict(state))


def _score_prediction(
    adapter: BenchmarkAdapter,
    example: BenchmarkExample,
    state: FrozenAgentState,
    prediction: Any,
) -> BenchmarkExampleRecord:
    return build_benchmark_example_record(
        adapter,
        example,
        state_hash=state.state_hash,
        prediction=prediction,
    )


def build_benchmark_example_record(
    adapter: BenchmarkAdapter,
    example: BenchmarkExample,
    *,
    state_hash: str,
    prediction: Any,
) -> BenchmarkExampleRecord:
    """Score one prediction and bind the adapter's exact evaluator provenance."""

    provenance = getattr(adapter, "evaluation_provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValidationError("adapter.evaluation_provenance must be a mapping when it is exposed")
    frozen_provenance = _validated_evaluation_provenance(
        provenance,
        example_id=example.example_id,
    )
    metrics = adapter.score(example, prediction)
    return BenchmarkExampleRecord(
        example_id=example.example_id,
        state_hash=state_hash,
        prediction=prediction,
        metrics=metrics,
        evaluation_provenance=frozen_provenance,
    )


def _validated_evaluation_provenance(
    value: Mapping[str, Any],
    *,
    example_id: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"evaluation provenance for {example_id!r} must be a mapping")
    try:
        frozen = freeze_mapping(value)
        canonical_json(frozen)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"evaluation provenance for {example_id!r} must be canonically serializable: {exc}"
        ) from exc
    return frozen


def _validated_metrics(
    metrics: Mapping[str, float],
    *,
    example_id: str,
) -> Mapping[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValidationError(f"metrics for {example_id!r} must be a mapping")
    if not metrics:
        raise ValidationError(f"metrics for {example_id!r} must not be empty")
    validated: dict[str, float] = {}
    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or not key
            or key != key.strip()
            or any(not character.isprintable() for character in key)
        ):
            raise ValidationError(
                f"metric key for {example_id!r} must be a non-empty, trimmed string "
                "without control characters"
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(f"metric {key!r} for {example_id!r} must be numeric")
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValidationError(f"metric {key!r} for {example_id!r} must be finite") from exc
        if not math.isfinite(numeric):
            raise ValidationError(f"metric {key!r} for {example_id!r} must be finite")
        validated[key] = numeric
    return freeze_mapping(validated)


def _aggregate_metrics(metrics: tuple[Mapping[str, float], ...]) -> Mapping[str, float]:
    if not metrics:
        raise ValidationError("cannot aggregate an empty metric collection")
    expected = _metric_keys(metrics[0])
    for index, item in enumerate(metrics[1:], start=1):
        _require_same_metric_keys(expected, item, example_id=f"<index {index}>")
    count = len(metrics)
    aggregated: dict[str, float] = {}
    for key in expected:
        try:
            value = math.fsum(item[key] / count for item in metrics)
        except (OverflowError, ValueError) as exc:
            raise ValidationError(f"aggregate metric {key!r} is not finite") from exc
        if not math.isfinite(value):
            raise ValidationError(f"aggregate metric {key!r} is not finite")
        aggregated[key] = value
    return freeze_mapping(aggregated)


def _metric_keys(metrics: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(sorted(metrics))


def _require_same_metric_keys(
    expected: tuple[str, ...],
    metrics: Mapping[str, float],
    *,
    example_id: str,
) -> None:
    actual = _metric_keys(metrics)
    if actual != expected:
        raise ValidationError(
            f"metric keys for {example_id!r} differ from the batch schema: "
            f"expected {expected!r}, got {actual!r}"
        )


def _require_same_metric_values(
    expected: Mapping[str, float],
    actual: Mapping[str, float],
    *,
    record_name: str,
) -> None:
    for key in expected:
        if actual[key] != expected[key]:
            raise ValidationError(f"{record_name} metric {key!r} is inconsistent")


def _require_nonempty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")


def _require_state_hash(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError("state_hash must be a lowercase SHA-256 digest")


def _require_unique_example_ids(example_ids: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example_id in example_ids:
        if example_id in seen:
            duplicates.add(example_id)
        seen.add(example_id)
    if duplicates:
        raise ValidationError(f"benchmark example IDs must be unique: {sorted(duplicates)!r}")


__all__ = [
    "BenchmarkBatchRecord",
    "BenchmarkBatchRunner",
    "BenchmarkExampleRecord",
    "PairedBenchmarkExampleRecord",
    "PairedBenchmarkRecord",
    "PairedBenchmarkRunner",
    "PredictionRunner",
]
