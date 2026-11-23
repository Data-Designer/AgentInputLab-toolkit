"""Build selector training rows from audited, full-information arm outcomes.

Feature observations are produced before interventions run and joined by the
frozen state hash.  Evaluator outcomes enter only this offline table-building
boundary; they are never passed to a selector at inference time.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from types import MappingProxyType

from agentinputlab.contracts.enums import SemanticFamily, StrEnum
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .learned import SelectorExample
from .runtime import FeatureObservation

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIONS = frozenset(
    {
        SemanticFamily.NO_OP,
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)


class EvaluatedArmStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ALGORITHM_FAILURE = "algorithm_failure"
    STRUCTURAL_INELIGIBLE = "structural_ineligible"
    INFRASTRUCTURE_MISSING = "infrastructure_missing"


@dataclass(frozen=True, slots=True)
class EvaluatedArmOutcome:
    """One evaluator result for a frozen state, action, seed, and runtime stratum."""

    example_id: str
    group_id: str
    state_hash: str
    split: str
    benchmark_id: str
    family: SemanticFamily
    release_id: str
    seed: int
    backend_hash: str
    comparison_profile_hash: str
    status: EvaluatedArmStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    cost: float = 0.0
    risk: float = 0.0

    def __post_init__(self) -> None:
        for label in ("example_id", "group_id", "split", "benchmark_id", "release_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"selector arm {label} must be trimmed and non-empty")
        for label in ("state_hash", "backend_hash", "comparison_profile_hash"):
            value = getattr(self, label)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValidationError(f"selector arm {label} must be a lowercase SHA-256")
        if self.family not in _ACTIONS:
            raise ValidationError("selector arm family must be NO_OP or a core intervention family")
        if not isinstance(self.status, EvaluatedArmStatus):
            raise ValidationError("selector arm status must be EvaluatedArmStatus")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("selector arm seed must be a non-negative integer")
        metrics = dict(self.metrics)
        if any(
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for name, value in metrics.items()
        ):
            raise ValidationError("selector arm metrics require trimmed names and finite numbers")
        if self.status is EvaluatedArmStatus.SUCCEEDED and not metrics:
            raise ValidationError("succeeded selector arms require evaluator metrics")
        if self.status is EvaluatedArmStatus.STRUCTURAL_INELIGIBLE:
            if self.family is SemanticFamily.NO_OP:
                raise ValidationError("NO_OP cannot be structurally ineligible")
            if metrics:
                raise ValidationError("structurally ineligible arms cannot carry outcome metrics")
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, (int, float))
            or not math.isfinite(self.cost)
            or self.cost < 0
        ):
            raise ValidationError("selector arm cost must be finite and non-negative")
        if (
            isinstance(self.risk, bool)
            or not isinstance(self.risk, (int, float))
            or not math.isfinite(self.risk)
            or not 0 <= self.risk <= 1
        ):
            raise ValidationError("selector arm risk must be finite and in [0, 1]")
        if self.family is SemanticFamily.NO_OP and self.release_id != "baseline.no_op":
            raise ValidationError("NO_OP selector arms must use baseline.no_op")
        object.__setattr__(self, "metrics", freeze_mapping(metrics))
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "risk", float(self.risk))


@dataclass(frozen=True, slots=True)
class UtilityDefinition:
    """Pre-registered conversion from evaluator records to selector utility."""

    utility_metric: str
    algorithm_failure_utility: float = 0.0
    aggregate: str = "mean"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.utility_metric, str)
            or not self.utility_metric.strip()
            or self.utility_metric != self.utility_metric.strip()
        ):
            raise ValidationError("utility_metric must be trimmed and non-empty")
        if (
            isinstance(self.algorithm_failure_utility, bool)
            or not isinstance(self.algorithm_failure_utility, (int, float))
            or not math.isfinite(self.algorithm_failure_utility)
        ):
            raise ValidationError("algorithm_failure_utility must be finite")
        if self.aggregate != "mean":
            raise ValidationError("selector utility aggregation currently supports mean only")
        object.__setattr__(
            self,
            "algorithm_failure_utility",
            float(self.algorithm_failure_utility),
        )

    @property
    def definition_hash(self) -> str:
        return content_hash(self)

    def value(self, outcome: EvaluatedArmOutcome) -> float:
        if outcome.status is EvaluatedArmStatus.ALGORITHM_FAILURE:
            return self.algorithm_failure_utility
        if outcome.status is not EvaluatedArmStatus.SUCCEEDED:
            raise ValidationError(f"cannot derive utility from arm status {outcome.status.value}")
        try:
            return float(outcome.metrics[self.utility_metric])
        except KeyError as exc:
            raise ValidationError(
                f"succeeded selector arm lacks utility metric {self.utility_metric!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class SelectorTrainingTable:
    """Immutable rows and provenance consumed by selector fit/calibration APIs."""

    examples: tuple[SelectorExample, ...]
    representative_by_family: Mapping[SemanticFamily, str]
    group_split_digests: Mapping[str, str]
    extractor_id: str
    extractor_revision: str
    feature_table_sha256: str
    outcome_table_sha256: str
    utility_definition_sha256: str

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if not examples or any(not isinstance(row, SelectorExample) for row in examples):
            raise ValidationError("selector training table requires SelectorExample rows")
        ids = [row.example_id for row in examples]
        if len(ids) != len(set(ids)):
            raise ValidationError("selector training table example IDs must be unique")
        representatives = _validate_representatives(self.representative_by_family)
        digests = dict(self.group_split_digests)
        if not digests or any(
            not isinstance(split, str)
            or not split.strip()
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            for split, digest in digests.items()
        ):
            raise ValidationError("selector group split digests must be keyed SHA-256 values")
        for label in ("extractor_id", "extractor_revision"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"selector training table {label} must be non-empty")
        for label in (
            "feature_table_sha256",
            "outcome_table_sha256",
            "utility_definition_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, label)):
                raise ValidationError(f"selector training table {label} must be SHA-256")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(
            self,
            "representative_by_family",
            MappingProxyType(representatives),
        )
        object.__setattr__(self, "group_split_digests", MappingProxyType(digests))

    @property
    def table_hash(self) -> str:
        return content_hash(self)


def build_selector_training_table(
    outcomes: Sequence[EvaluatedArmOutcome],
    observations: Mapping[str, FeatureObservation],
    *,
    utility_definition: UtilityDefinition,
    representative_by_family: Mapping[SemanticFamily, str],
) -> SelectorTrainingTable:
    """Validate, aggregate, and join pre-action features with paired outcomes."""

    rows = tuple(outcomes)
    if not rows or any(not isinstance(row, EvaluatedArmOutcome) for row in rows):
        raise ValidationError("selector outcome table requires EvaluatedArmOutcome rows")
    if not isinstance(utility_definition, UtilityDefinition):
        raise TypeError("utility_definition must be UtilityDefinition")
    representatives = _validate_representatives(representative_by_family)
    if any(row.status is EvaluatedArmStatus.INFRASTRUCTURE_MISSING for row in rows):
        raise ValidationError(
            "infrastructure-missing outcomes invalidate the selector table; "
            "they cannot be converted to utility"
        )

    feature_by_state = dict(observations)
    if not feature_by_state or any(
        not isinstance(state_hash, str)
        or not _SHA256.fullmatch(state_hash)
        or not isinstance(observation, FeatureObservation)
        or observation.state_hash != state_hash
        for state_hash, observation in feature_by_state.items()
    ):
        raise ValidationError(
            "selector observations must map state hashes to matching FeatureObservation values"
        )
    missing_features = sorted({row.state_hash for row in rows} - set(feature_by_state))
    if missing_features:
        raise ValidationError("selector outcomes are missing pre-action feature observations")
    extractor_identities = {
        (
            feature_by_state[row.state_hash].extractor_id,
            feature_by_state[row.state_hash].extractor_revision,
        )
        for row in rows
    }
    if len(extractor_identities) != 1:
        raise ValidationError("selector feature table must use one extractor identity")
    extractor_id, extractor_revision = next(iter(extractor_identities))

    group_owner: dict[str, str] = {}
    example_owner: dict[tuple[str, str], tuple[str, str, str]] = {}
    state_owner: dict[str, tuple[str, str, str, str]] = {}
    group_ids_by_split: dict[str, set[str]] = {}
    for row in rows:
        previous = group_owner.setdefault(row.group_id, row.split)
        if previous != row.split:
            raise ValidationError(
                f"selector group {row.group_id!r} crosses splits {previous!r} and {row.split!r}"
            )
        example_key = (row.benchmark_id, row.example_id)
        example_binding = (row.split, row.group_id, row.state_hash)
        previous_example_binding = example_owner.setdefault(
            example_key,
            example_binding,
        )
        if previous_example_binding != example_binding:
            raise ValidationError(
                "selector source example is bound to multiple states, groups, or splits"
            )
        state_binding = (
            row.split,
            row.group_id,
            row.benchmark_id,
            row.example_id,
        )
        previous_state_binding = state_owner.setdefault(row.state_hash, state_binding)
        if previous_state_binding != state_binding:
            raise ValidationError(
                "selector state hash is bound to multiple source examples, groups, or splits"
            )
        group_ids_by_split.setdefault(row.split, set()).add(row.group_id)

    cells: dict[
        tuple[str, str, str, str, str],
        list[EvaluatedArmOutcome],
    ] = {}
    for row in rows:
        key = (
            row.state_hash,
            row.group_id,
            row.split,
            row.backend_hash,
            row.comparison_profile_hash,
        )
        cells.setdefault(key, []).append(row)

    examples: list[SelectorExample] = []
    for key in sorted(cells):
        state_hash, group_id, split, backend_hash, profile_hash = key
        cell_rows = cells[key]
        source_example_ids = {row.example_id for row in cell_rows}
        benchmark_ids = {row.benchmark_id for row in cell_rows}
        if len(source_example_ids) != 1 or len(benchmark_ids) != 1:
            raise ValidationError(
                "one selector state/runtime cell must have one source example and benchmark"
            )
        by_family: dict[SemanticFamily, list[EvaluatedArmOutcome]] = {}
        for row in cell_rows:
            by_family.setdefault(row.family, []).append(row)
        if SemanticFamily.NO_OP not in by_family:
            raise ValidationError("every selector training cell requires an explicit NO_OP arm")

        baseline_rows = by_family[SemanticFamily.NO_OP]
        _validate_family_replicates(
            SemanticFamily.NO_OP,
            baseline_rows,
            expected_release="baseline.no_op",
        )
        baseline_seeds = {row.seed for row in baseline_rows}
        if any(row.status is EvaluatedArmStatus.STRUCTURAL_INELIGIBLE for row in baseline_rows):
            raise ValidationError("NO_OP cannot be structurally ineligible")

        utilities: dict[SemanticFamily, float] = {
            SemanticFamily.NO_OP: fmean(utility_definition.value(row) for row in baseline_rows)
        }
        costs: dict[SemanticFamily, float] = {}
        risks: dict[SemanticFamily, float] = {}
        for family in sorted(
            (item for item in by_family if item is not SemanticFamily.NO_OP),
            key=lambda item: item.value,
        ):
            family_rows = by_family[family]
            try:
                expected_release = representatives[family]
            except KeyError as exc:
                raise ValidationError(
                    f"selector family {family.value} has no frozen representative"
                ) from exc
            _validate_family_replicates(
                family,
                family_rows,
                expected_release=expected_release,
            )
            family_seeds = {row.seed for row in family_rows}
            if family_seeds != baseline_seeds:
                raise ValidationError(f"selector family {family.value} seed set differs from NO_OP")
            structural = [
                row.status is EvaluatedArmStatus.STRUCTURAL_INELIGIBLE for row in family_rows
            ]
            if any(structural):
                if not all(structural):
                    raise ValidationError(
                        f"selector family {family.value} mixes structural ineligibility "
                        "with evaluated replicates"
                    )
                continue
            utilities[family] = fmean(utility_definition.value(row) for row in family_rows)
            costs[family] = fmean(row.cost for row in family_rows)
            risks[family] = fmean(row.risk for row in family_rows)

        example_id = "selector_" + content_hash(
            {
                "state_hash": state_hash,
                "group_id": group_id,
                "split": split,
                "backend_hash": backend_hash,
                "comparison_profile_hash": profile_hash,
            }
        )
        examples.append(
            SelectorExample(
                example_id=example_id,
                features=feature_by_state[state_hash].features,
                utilities=utilities,
                split=split,
                costs=costs,
                risks=risks,
            )
        )

    ordered_rows = tuple(sorted(rows, key=_outcome_sort_key))
    feature_table_sha256 = content_hash(
        [
            {
                "state_hash": state_hash,
                "feature_hash": feature_by_state[state_hash].feature_hash,
            }
            for state_hash in sorted({row.state_hash for row in rows})
        ]
    )
    outcome_table_sha256 = content_hash(ordered_rows)
    group_split_digests = {
        split: content_hash({"split": split, "group_ids": sorted(group_ids)})
        for split, group_ids in sorted(group_ids_by_split.items())
    }
    return SelectorTrainingTable(
        examples=tuple(sorted(examples, key=lambda row: row.example_id)),
        representative_by_family=representatives,
        group_split_digests=group_split_digests,
        extractor_id=extractor_id,
        extractor_revision=extractor_revision,
        feature_table_sha256=feature_table_sha256,
        outcome_table_sha256=outcome_table_sha256,
        utility_definition_sha256=utility_definition.definition_hash,
    )


def _validate_family_replicates(
    family: SemanticFamily,
    rows: Sequence[EvaluatedArmOutcome],
    *,
    expected_release: str,
) -> None:
    releases = {row.release_id for row in rows}
    if releases != {expected_release}:
        raise ValidationError(
            f"selector family {family.value} release differs from its frozen representative"
        )
    seeds = [row.seed for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValidationError(f"selector family {family.value} contains duplicate seed outcomes")


def _validate_representatives(
    value: Mapping[SemanticFamily, str],
) -> dict[SemanticFamily, str]:
    copied = dict(value)
    if any(
        not isinstance(family, SemanticFamily)
        or family in {SemanticFamily.NO_OP, SemanticFamily.COMPOSITE}
        or not isinstance(release, str)
        or not release.strip()
        or release != release.strip()
        for family, release in copied.items()
    ):
        raise ValidationError(
            "selector representatives require core intervention families and release IDs"
        )
    return copied


def _outcome_sort_key(
    row: EvaluatedArmOutcome,
) -> tuple[str, str, str, str, str, str, int]:
    return (
        row.split,
        row.group_id,
        row.state_hash,
        row.backend_hash,
        row.comparison_profile_hash,
        row.family.value,
        row.seed,
    )


__all__ = [
    "EvaluatedArmOutcome",
    "EvaluatedArmStatus",
    "SelectorTrainingTable",
    "UtilityDefinition",
    "build_selector_training_table",
]
