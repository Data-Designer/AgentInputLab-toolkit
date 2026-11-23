"""Dependency-free selector training, abstention calibration, and evaluation.

The implementation is intentionally a transparent full-information ridge
baseline.  It consumes discovery-set paired utilities; it never trains on the
confirmatory or external-test split.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from statistics import fmean
from types import MappingProxyType

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import FrozenAgentState

from .base import SelectionDecision, SelectionFeatures, Selector

_ACTIONS = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)


@dataclass(frozen=True, slots=True)
class SelectorExample:
    example_id: str
    features: SelectionFeatures
    utilities: Mapping[SemanticFamily, float]
    split: str
    costs: Mapping[SemanticFamily, float] = field(default_factory=dict)
    risks: Mapping[SemanticFamily, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("selector example requires an ID")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("selector example requires a non-empty split")
        if any(not isinstance(family, SemanticFamily) for family in self.utilities):
            raise ValueError("selector utility keys must be SemanticFamily values")
        if SemanticFamily.NO_OP not in self.utilities:
            raise ValueError("selector utilities require an explicit NO_OP value")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.utilities.values()
        ):
            raise ValueError("selector utilities must be finite numbers")
        if any(not isinstance(family, SemanticFamily) for family in self.costs):
            raise ValueError("selector cost keys must be SemanticFamily values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in self.costs.values()
        ):
            raise ValueError("selector costs must be finite non-negative numbers")
        if any(not isinstance(family, SemanticFamily) for family in self.risks):
            raise ValueError("selector risk keys must be SemanticFamily values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in self.risks.values()
        ):
            raise ValueError("selector risks must be finite numbers in [0, 1]")
        object.__setattr__(self, "utilities", MappingProxyType(dict(self.utilities)))
        object.__setattr__(self, "costs", MappingProxyType(dict(self.costs)))
        object.__setattr__(self, "risks", MappingProxyType(dict(self.risks)))


@dataclass(frozen=True, slots=True)
class SelectorSplitProvenance:
    """Order-invariant identities for one disjoint selector data partition."""

    id_set_digests: Mapping[str, str]
    split_sizes: Mapping[str, int]

    def __post_init__(self) -> None:
        digests = dict(self.id_set_digests)
        sizes = dict(self.split_sizes)
        if not digests or set(digests) != set(sizes):
            raise ValueError("selector split provenance requires matching split identities")
        if any(
            not isinstance(split, str)
            or not split.strip()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for split, digest in digests.items()
        ):
            raise ValueError("selector split provenance requires SHA-256 split digests")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in sizes.values()
        ):
            raise ValueError("selector split provenance requires positive split sizes")
        object.__setattr__(self, "id_set_digests", MappingProxyType(digests))
        object.__setattr__(self, "split_sizes", MappingProxyType(sizes))

    @property
    def provenance_hash(self) -> str:
        return content_hash(
            {
                "id_set_digests": self.id_set_digests,
                "split_sizes": self.split_sizes,
            }
        )


def validate_selector_split_ids(
    split_examples: Mapping[str, Sequence[SelectorExample]],
) -> SelectorSplitProvenance:
    """Reject train/calibration/test ID leakage and bind each split's ID set.

    The held-out split may use a study-specific name such as
    ``confirmatory_test`` or ``external_test``.  At least one such split is
    required in addition to ``train`` and ``calibration``.
    """

    copied = dict(split_examples)
    if "train" not in copied or "calibration" not in copied or len(copied) < 3:
        raise ValueError(
            "selector split validation requires train, calibration, and a held-out test split"
        )
    owner_by_id: dict[str, str] = {}
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for split in sorted(copied):
        if not isinstance(split, str) or not split.strip():
            raise ValueError("selector split names must be non-empty strings")
        rows = tuple(copied[split])
        if not rows or any(not isinstance(row, SelectorExample) for row in rows):
            raise ValueError(f"selector split {split!r} requires SelectorExample rows")
        if any(row.split != split for row in rows):
            raise ValueError(f"selector split {split!r} contains a mismatched split label")
        ids = [row.example_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"selector split {split!r} requires unique example IDs")
        for example_id in ids:
            previous = owner_by_id.setdefault(example_id, split)
            if previous != split:
                raise ValueError(
                    f"selector example ID {example_id!r} overlaps splits {previous!r} and {split!r}"
                )
        digests[split] = _selector_id_set_digest(rows, required_split=split)
        sizes[split] = len(rows)
    return SelectorSplitProvenance(digests, sizes)


@dataclass(frozen=True, slots=True)
class RidgeUtilitySelector:
    """Predict per-family utility gains and abstain unless evidence is sufficient."""

    coefficients: Mapping[SemanticFamily, tuple[float, ...]]
    release_by_family: Mapping[SemanticFamily, str]
    min_gain: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        expected = len(SelectionFeatures().as_vector()) + 1
        if not self.coefficients:
            raise ValueError("selector requires at least one fitted family")
        if any(
            not isinstance(family, SemanticFamily) or family is SemanticFamily.NO_OP
            for family in self.coefficients
        ):
            raise ValueError("selector coefficient keys must be intervention families")
        frozen_coefficients = {
            family: tuple(values) for family, values in self.coefficients.items()
        }
        if any(len(values) != expected for values in frozen_coefficients.values()):
            raise ValueError(f"selector coefficient vectors must contain {expected} values")
        if any(
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
            for values in frozen_coefficients.values()
        ):
            raise ValueError("selector coefficients must be finite numbers")
        if (
            isinstance(self.min_gain, bool)
            or not isinstance(self.min_gain, (int, float))
            or not math.isfinite(self.min_gain)
        ):
            raise ValueError("selector min_gain must be a finite number")
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(self.margin)
            or self.margin < 0
        ):
            raise ValueError("selector margin must be finite and non-negative")
        release_mapping = dict(self.release_by_family)
        if any(
            not isinstance(family, SemanticFamily)
            or family is SemanticFamily.NO_OP
            or not isinstance(release, str)
            or not release.strip()
            for family, release in release_mapping.items()
        ):
            raise ValueError("release mappings require intervention-family keys and IDs")
        missing_release = set(frozen_coefficients) - set(release_mapping)
        if missing_release:
            names = ", ".join(sorted(family.value for family in missing_release))
            raise ValueError(f"missing release mapping for fitted families: {names}")
        object.__setattr__(self, "coefficients", MappingProxyType(frozen_coefficients))
        object.__setattr__(self, "release_by_family", MappingProxyType(release_mapping))

    @property
    def model_hash(self) -> str:
        return content_hash(
            {
                "coefficients": {
                    family.value: values for family, values in self.coefficients.items()
                },
                "release_by_family": {
                    family.value: release for family, release in self.release_by_family.items()
                },
                "min_gain": self.min_gain,
                "margin": self.margin,
            }
        )

    def predict_gains(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        vector = (1.0, *features.as_vector())
        return {
            family: sum(weight * value for weight, value in zip(weights, vector, strict=True))
            for family, weights in self.coefficients.items()
        }

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        del state
        gains = self.predict_gains(features)
        eligible = _normalize_eligible_families(eligible_families)
        ordered = sorted(
            ((family, gain) for family, gain in gains.items() if family in eligible),
            key=lambda item: (-item[1], item[0].value),
        )
        scores = {item.value: value for item, value in gains.items()}
        scores[SemanticFamily.NO_OP.value] = 0.0
        if not ordered:
            return SelectionDecision(
                family=SemanticFamily.NO_OP,
                release_id="baseline.no_op",
                selection_score=0.0,
                rationale="abstained: no fitted intervention family is eligible",
                scores=scores,
            )
        family, gain = ordered[0]
        runner_up = max(0.0, ordered[1][1] if len(ordered) > 1 else 0.0)
        if gain <= self.min_gain or gain - runner_up <= self.margin:
            return SelectionDecision(
                family=SemanticFamily.NO_OP,
                release_id="baseline.no_op",
                selection_score=0.0,
                rationale="abstained: calibrated gain or margin gate not met",
                scores=scores,
            )
        return SelectionDecision(
            family=family,
            release_id=self.release_by_family[family],
            selection_score=gain,
            rationale=f"selected calibrated predicted-utility winner: {family.value}",
            scores=scores,
        )


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    count: int
    coverage: float
    mean_policy_value: float
    mean_no_op_value: float
    mean_gain: float
    harm_rate: float
    oracle_regret: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    selector: RidgeUtilitySelector
    evaluation: PolicyEvaluation
    candidates_considered: int


def fit_ridge_selector(
    examples: Sequence[SelectorExample],
    *,
    release_by_family: Mapping[SemanticFamily, str],
    ridge: float = 1e-6,
) -> RidgeUtilitySelector:
    """Fit one utility-gain model per family on an explicitly named train split."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("selector training requires examples")
    if any(example.split != "train" for example in rows):
        raise ValueError("fit_ridge_selector only accepts split='train' examples")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    matrix = [(1.0, *example.features.as_vector()) for example in rows]
    coefficients: dict[SemanticFamily, tuple[float, ...]] = {}
    for family in _ACTIONS:
        if any(family not in example.utilities for example in rows):
            continue
        targets = [
            example.utilities[family] - example.utilities[SemanticFamily.NO_OP] for example in rows
        ]
        coefficients[family] = _ridge_solve(matrix, targets, ridge=ridge)
    if not coefficients:
        raise ValueError("no intervention family has complete training utilities")
    missing_release = set(coefficients) - set(release_by_family)
    if missing_release:
        names = ", ".join(sorted(family.value for family in missing_release))
        raise ValueError(f"missing release mapping for fitted families: {names}")
    return RidgeUtilitySelector(coefficients, release_by_family)


def calibrate_abstention(
    selector: RidgeUtilitySelector,
    examples: Sequence[SelectorExample],
    *,
    gain_grid: Sequence[float],
    margin_grid: Sequence[float],
    max_harm_rate: float,
) -> CalibrationResult:
    """Choose gates on calibration data only, maximizing value then coverage."""

    rows = tuple(examples)
    if not rows or any(example.split != "calibration" for example in rows):
        raise ValueError("abstention calibration requires split='calibration' examples")
    if not 0 <= max_harm_rate <= 1:
        raise ValueError("max_harm_rate must be in [0, 1]")
    if not gain_grid or not margin_grid:
        raise ValueError("abstention calibration grids must be non-empty")
    if any(not math.isfinite(float(value)) for value in gain_grid):
        raise ValueError("gain grid values must be finite")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in margin_grid):
        raise ValueError("margin grid values must be finite and non-negative")
    candidates: list[tuple[PolicyEvaluation, RidgeUtilitySelector]] = []
    considered = 0
    for gain in gain_grid:
        for margin in margin_grid:
            considered += 1
            candidate = replace(selector, min_gain=float(gain), margin=float(margin))
            evaluation = evaluate_selector(candidate, rows, required_split="calibration")
            if evaluation.harm_rate <= max_harm_rate:
                candidates.append((evaluation, candidate))
    if not candidates:
        raise ValueError("no abstention candidate satisfies the harm constraint")
    evaluation, candidate = max(
        candidates,
        key=lambda item: (
            item[0].mean_policy_value,
            item[0].coverage,
            -item[0].harm_rate,
            -item[1].min_gain,
            -item[1].margin,
        ),
    )
    return CalibrationResult(candidate, evaluation, considered)


def evaluate_selector(
    selector: Selector,
    examples: Sequence[SelectorExample],
    *,
    required_split: str,
) -> PolicyEvaluation:
    """Evaluate a frozen selector on one named, non-training split."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("selector evaluation requires examples")
    if required_split == "train" or any(example.split != required_split for example in rows):
        raise ValueError("evaluation split must be uniform and must not be train")
    chosen_values: list[float] = []
    no_op_values: list[float] = []
    regrets: list[float] = []
    selected = 0
    harmed = 0
    for example in rows:
        decision = selector.select(
            FrozenAgentState(query=example.example_id),
            example.features,
            eligible_families=tuple(example.utilities),
        )
        family = decision.family
        if family not in example.utilities:
            raise ValueError(
                f"selector chose ineligible family {family.value!r} for {example.example_id!r}"
            )
        chosen = example.utilities[family]
        no_op = example.utilities[SemanticFamily.NO_OP]
        chosen_values.append(chosen)
        no_op_values.append(no_op)
        selected += family is not SemanticFamily.NO_OP
        harmed += chosen < no_op
        regrets.append(max(example.utilities.values()) - chosen)
    return PolicyEvaluation(
        count=len(rows),
        coverage=selected / len(rows),
        mean_policy_value=fmean(chosen_values),
        mean_no_op_value=fmean(no_op_values),
        mean_gain=fmean(
            after - before for before, after in zip(no_op_values, chosen_values, strict=True)
        ),
        harm_rate=harmed / len(rows),
        oracle_regret=fmean(regrets),
    )


def _selector_id_set_digest(
    examples: Sequence[SelectorExample],
    *,
    required_split: str,
) -> str:
    rows = tuple(examples)
    if not rows or any(
        not isinstance(row, SelectorExample) or row.split != required_split for row in rows
    ):
        raise ValueError(f"selector ID digest requires uniform split={required_split!r} rows")
    ids = [row.example_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("selector ID digest requires unique example IDs")
    return content_hash(
        {
            "split": required_split,
            "example_ids": sorted(ids),
        }
    )


def _ridge_solve(
    matrix: Sequence[tuple[float, ...]],
    targets: Sequence[float],
    *,
    ridge: float,
) -> tuple[float, ...]:
    width = len(matrix[0])
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, target in zip(matrix, targets, strict=True):
        for left in range(width):
            rhs[left] += row[left] * target
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    for index in range(1, width):
        gram[index][index] += ridge
    gram[0][0] += ridge * 1e-3
    return tuple(_gaussian_elimination(gram, rhs))


def _gaussian_elimination(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [[*row, value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("selector design matrix is singular; increase ridge")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[row][-1] for row in range(size)]


def _normalize_eligible_families(
    eligible_families: Collection[SemanticFamily] | None,
) -> frozenset[SemanticFamily]:
    if eligible_families is None:
        return frozenset(_ACTIONS)
    eligible = tuple(eligible_families)
    if any(not isinstance(family, SemanticFamily) for family in eligible):
        raise ValueError("eligible_families must contain SemanticFamily values")
    return frozenset(family for family in eligible if family is not SemanticFamily.NO_OP)
