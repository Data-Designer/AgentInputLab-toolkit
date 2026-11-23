"""Auditable, dependency-free selector baselines for offline confirmation.

All fitting functions consume outcome-derived information only from an explicit
``train`` split.  Calibration functions accept only ``calibration`` rows.  At
runtime the selectors receive :class:`SelectionFeatures` and an eligibility
mask; evaluator gold and other-arm outcomes never cross the ``Selector``
protocol.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from statistics import fmean
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import FrozenAgentState

from .base import SelectionDecision, SelectionFeatures, Selector, UtilityGainProvider
from .learned import (
    PolicyEvaluation,
    SelectorExample,
    _selector_id_set_digest,
    evaluate_selector,
)

_ACTIONS = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)
_FEATURE_WIDTH = len(SelectionFeatures().as_vector())
_EMPTY_TRAIN_ID_SET_DIGEST = content_hash({"split": "train", "example_ids": []})


@dataclass(frozen=True, slots=True)
class KNNTrainingRow:
    """One identity-free feature/gain row retained by a KNN selector."""

    feature_vector: tuple[float, ...]
    gains: Mapping[SemanticFamily, float]

    def __post_init__(self) -> None:
        raw_vector = tuple(self.feature_vector)
        if len(raw_vector) != _FEATURE_WIDTH or any(not _finite(value) for value in raw_vector):
            raise ValueError(f"KNN feature vectors must contain {_FEATURE_WIDTH} finite values")
        raw_gains = dict(self.gains)
        if not raw_gains or any(family not in _ACTIONS for family in raw_gains):
            raise ValueError("KNN rows require intervention-family gains")
        if any(not _finite(value) for value in raw_gains.values()):
            raise ValueError("KNN gains must be finite numbers")
        vector = tuple(float(value) for value in raw_vector)
        copied = {family: float(value) for family, value in raw_gains.items()}
        object.__setattr__(self, "feature_vector", vector)
        object.__setattr__(self, "gains", MappingProxyType(copied))

    @property
    def row_hash(self) -> str:
        return content_hash(
            {
                "feature_vector": self.feature_vector,
                "gains": {family.value: value for family, value in self.gains.items()},
            }
        )


@dataclass(frozen=True, slots=True)
class KNNUtilitySelector:
    """Estimate family utility gains from nearby gold-free feature vectors."""

    training_rows: tuple[KNNTrainingRow, ...]
    release_by_family: Mapping[SemanticFamily, str]
    k: int = 5
    distance_power: float = 2.0
    min_gain: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        rows = tuple(self.training_rows)
        if not rows or any(not isinstance(row, KNNTrainingRow) for row in rows):
            raise ValueError("KNN selector requires frozen training rows")
        releases = _validated_release_mapping(self.release_by_family)
        observed = {family for row in rows for family in row.gains}
        missing = set(releases) - observed
        if missing:
            names = ", ".join(sorted(family.value for family in missing))
            raise ValueError(f"KNN training rows have no outcomes for: {names}")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("KNN k must be a positive integer")
        if not _finite(self.distance_power) or self.distance_power <= 0:
            raise ValueError("KNN distance_power must be finite and positive")
        _validate_abstention(self.min_gain, self.margin)
        object.__setattr__(self, "training_rows", rows)
        object.__setattr__(self, "release_by_family", MappingProxyType(releases))

    @property
    def model_hash(self) -> str:
        return content_hash(
            {
                "selector": "knn_utility",
                "training_rows": [
                    {
                        "row_hash": row.row_hash,
                        "feature_vector": row.feature_vector,
                        "gains": {family.value: value for family, value in row.gains.items()},
                    }
                    for row in self.training_rows
                ],
                "release_by_family": {
                    family.value: release for family, release in self.release_by_family.items()
                },
                "k": self.k,
                "distance_power": self.distance_power,
                "min_gain": self.min_gain,
                "margin": self.margin,
            }
        )

    def predict_gains(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        target = features.as_vector()
        predictions: dict[SemanticFamily, float] = {}
        for family in self.release_by_family:
            distances = sorted(
                (
                    (_euclidean(target, row.feature_vector), row.row_hash, row.gains[family])
                    for row in self.training_rows
                    if family in row.gains
                ),
                key=lambda item: (item[0], item[1]),
            )
            exact = [gain for distance, _, gain in distances if distance <= 1e-12]
            if exact:
                predictions[family] = fmean(exact)
                continue
            neighbors = distances[: min(self.k, len(distances))]
            weights = [1.0 / (distance**self.distance_power) for distance, _, _ in neighbors]
            predictions[family] = math.fsum(
                weight * item[2] for weight, item in zip(weights, neighbors, strict=True)
            ) / math.fsum(weights)
        return predictions

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        del state
        return _decision_from_gains(
            self.predict_gains(features),
            self.release_by_family,
            eligible_families=eligible_families,
            min_gain=self.min_gain,
            margin=self.margin,
            label="KNN utility",
        )


def fit_knn_selector(
    examples: Sequence[SelectorExample],
    *,
    release_by_family: Mapping[SemanticFamily, str],
    k: int = 5,
    distance_power: float = 2.0,
) -> KNNUtilitySelector:
    """Fit an identity-free KNN utility model on discovery ``train`` rows."""

    rows = _training_rows(examples, label="KNN selector")
    releases = _validated_release_mapping(release_by_family)
    frozen_rows: list[KNNTrainingRow] = []
    for example in rows:
        no_op = example.utilities[SemanticFamily.NO_OP]
        gains = {
            family: example.utilities[family] - no_op
            for family in releases
            if family in example.utilities
        }
        if gains:
            frozen_rows.append(KNNTrainingRow(example.features.as_vector(), gains))
    frozen_rows.sort(key=lambda row: row.row_hash)
    return KNNUtilitySelector(
        tuple(frozen_rows),
        releases,
        k=k,
        distance_power=distance_power,
    )


@dataclass(frozen=True, slots=True)
class LogisticTLearnerSelector:
    """Penalized logistic T-learner over full-information potential outcomes."""

    outcome_coefficients: Mapping[SemanticFamily, tuple[float, ...]]
    release_by_family: Mapping[SemanticFamily, str]
    l2: float
    learning_rate: float
    iterations: int
    min_gain: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        coefficients = {
            family: tuple(values) for family, values in self.outcome_coefficients.items()
        }
        if SemanticFamily.NO_OP not in coefficients:
            raise ValueError("logistic T-learner requires an explicit NO_OP outcome model")
        if any(family not in {*_ACTIONS, SemanticFamily.NO_OP} for family in coefficients):
            raise ValueError("logistic T-learner coefficient keys must be selector actions")
        expected = _FEATURE_WIDTH + 1
        if any(
            len(values) != expected or any(not _finite(value) for value in values)
            for values in coefficients.values()
        ):
            raise ValueError(
                f"logistic T-learner coefficient vectors must contain {expected} finite values"
            )
        releases = _validated_release_mapping(self.release_by_family)
        modeled = set(coefficients) - {SemanticFamily.NO_OP}
        if modeled != set(releases):
            raise ValueError("logistic T-learner models and release mappings must match")
        _validate_logistic_config(self.l2, self.learning_rate, self.iterations)
        _validate_abstention(self.min_gain, self.margin)
        object.__setattr__(self, "outcome_coefficients", MappingProxyType(coefficients))
        object.__setattr__(self, "release_by_family", MappingProxyType(releases))

    @property
    def model_hash(self) -> str:
        return content_hash(
            {
                "selector": "logistic_t_learner",
                "outcome_coefficients": {
                    family.value: values for family, values in self.outcome_coefficients.items()
                },
                "release_by_family": {
                    family.value: release for family, release in self.release_by_family.items()
                },
                "fit": {
                    "l2": self.l2,
                    "learning_rate": self.learning_rate,
                    "iterations": self.iterations,
                },
                "min_gain": self.min_gain,
                "margin": self.margin,
            }
        )

    def predict_outcomes(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        vector = (1.0, *features.as_vector())
        return {
            family: _sigmoid(
                math.fsum(weight * value for weight, value in zip(weights, vector, strict=True))
            )
            for family, weights in self.outcome_coefficients.items()
        }

    def predict_gains(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        outcomes = self.predict_outcomes(features)
        no_op = outcomes[SemanticFamily.NO_OP]
        return {family: outcomes[family] - no_op for family in self.release_by_family}

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        del state
        return _decision_from_gains(
            self.predict_gains(features),
            self.release_by_family,
            eligible_families=eligible_families,
            min_gain=self.min_gain,
            margin=self.margin,
            label="logistic T-learner",
        )


def fit_logistic_t_learner(
    examples: Sequence[SelectorExample],
    *,
    release_by_family: Mapping[SemanticFamily, str],
    l2: float = 1e-3,
    learning_rate: float = 0.2,
    iterations: int = 600,
) -> LogisticTLearnerSelector:
    """Fit one outcome model per action using discovery ``train`` rows only."""

    rows = _training_rows(examples, label="logistic T-learner")
    releases = _validated_release_mapping(release_by_family)
    _validate_logistic_config(l2, learning_rate, iterations)
    actions = (SemanticFamily.NO_OP, *releases)
    coefficients: dict[SemanticFamily, tuple[float, ...]] = {}
    for family in actions:
        supported = tuple(example for example in rows if family in example.utilities)
        if not supported:
            raise ValueError(f"logistic T-learner has no outcomes for {family.value}")
        labels = tuple(float(example.utilities[family]) for example in supported)
        if any(not 0 <= label <= 1 for label in labels):
            raise ValueError("logistic T-learner utilities must be probabilities in [0, 1]")
        matrix = tuple((1.0, *example.features.as_vector()) for example in supported)
        coefficients[family] = _fit_logistic(
            matrix,
            labels,
            l2=l2,
            learning_rate=learning_rate,
            iterations=iterations,
        )
    return LogisticTLearnerSelector(
        coefficients,
        releases,
        l2=l2,
        learning_rate=learning_rate,
        iterations=iterations,
    )


@dataclass(frozen=True, slots=True)
class RidgeTLearnerSelector:
    """Linear T-learner with one ridge potential-outcome model per action."""

    outcome_coefficients: Mapping[SemanticFamily, tuple[float, ...]]
    release_by_family: Mapping[SemanticFamily, str]
    ridge: float = 1e-3
    min_gain: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        coefficients = {
            family: tuple(values) for family, values in self.outcome_coefficients.items()
        }
        if SemanticFamily.NO_OP not in coefficients:
            raise ValueError("ridge T-learner requires an explicit NO_OP outcome model")
        if any(family not in {*_ACTIONS, SemanticFamily.NO_OP} for family in coefficients):
            raise ValueError("ridge T-learner coefficient keys must be selector actions")
        expected = _FEATURE_WIDTH + 1
        if any(
            len(values) != expected or any(not _finite(value) for value in values)
            for values in coefficients.values()
        ):
            raise ValueError(
                f"ridge T-learner coefficient vectors must contain {expected} finite values"
            )
        releases = _validated_release_mapping(self.release_by_family)
        if set(coefficients) - {SemanticFamily.NO_OP} != set(releases):
            raise ValueError("ridge T-learner models and release mappings must match")
        if not _finite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge T-learner penalty must be finite and non-negative")
        _validate_abstention(self.min_gain, self.margin)
        object.__setattr__(self, "outcome_coefficients", MappingProxyType(coefficients))
        object.__setattr__(self, "release_by_family", MappingProxyType(releases))

    @property
    def model_hash(self) -> str:
        return content_hash(
            {
                "selector": "ridge_t_learner",
                "outcome_coefficients": {
                    family.value: values for family, values in self.outcome_coefficients.items()
                },
                "release_by_family": {
                    family.value: release for family, release in self.release_by_family.items()
                },
                "ridge": self.ridge,
                "min_gain": self.min_gain,
                "margin": self.margin,
            }
        )

    def predict_outcomes(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        vector = (1.0, *features.as_vector())
        return {
            family: math.fsum(weight * value for weight, value in zip(weights, vector, strict=True))
            for family, weights in self.outcome_coefficients.items()
        }

    def predict_gains(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        outcomes = self.predict_outcomes(features)
        no_op = outcomes[SemanticFamily.NO_OP]
        return {family: outcomes[family] - no_op for family in self.release_by_family}

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        del state
        return _decision_from_gains(
            self.predict_gains(features),
            self.release_by_family,
            eligible_families=eligible_families,
            min_gain=self.min_gain,
            margin=self.margin,
            label="ridge T-learner",
        )


def fit_ridge_t_learner(
    examples: Sequence[SelectorExample],
    *,
    release_by_family: Mapping[SemanticFamily, str],
    ridge: float = 1e-3,
) -> RidgeTLearnerSelector:
    """Fit per-action ridge potential outcomes on discovery ``train`` rows."""

    rows = _training_rows(examples, label="ridge T-learner")
    releases = _validated_release_mapping(release_by_family)
    if not _finite(ridge) or ridge < 0:
        raise ValueError("ridge T-learner penalty must be finite and non-negative")
    coefficients: dict[SemanticFamily, tuple[float, ...]] = {}
    for family in (SemanticFamily.NO_OP, *releases):
        supported = tuple(example for example in rows if family in example.utilities)
        if not supported:
            raise ValueError(f"ridge T-learner has no outcomes for {family.value}")
        matrix = tuple((1.0, *example.features.as_vector()) for example in supported)
        labels = tuple(float(example.utilities[family]) for example in supported)
        coefficients[family] = _fit_ridge(matrix, labels, ridge=ridge)
    return RidgeTLearnerSelector(coefficients, releases, ridge=ridge)


@dataclass(frozen=True, slots=True)
class FrozenRegressionTree:
    """Minimal immutable regression tree extracted from a fitted sklearn tree."""

    children_left: tuple[int, ...]
    children_right: tuple[int, ...]
    features: tuple[int, ...]
    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        left = tuple(self.children_left)
        right = tuple(self.children_right)
        features = tuple(self.features)
        thresholds = tuple(self.thresholds)
        values = tuple(self.values)
        size = len(left)
        if size < 1 or any(len(items) != size for items in (right, features, thresholds, values)):
            raise ValueError("frozen regression tree arrays must have one non-zero length")
        if any(type(value) is not int for value in (*left, *right, *features)):
            raise ValueError("frozen regression tree indices must be integers")
        if any(not _finite(value) for value in (*thresholds, *values)):
            raise ValueError("frozen regression tree thresholds and values must be finite")
        parents = [0] * size
        for index, (left_child, right_child, feature) in enumerate(
            zip(left, right, features, strict=True)
        ):
            if left_child == -1 and right_child == -1:
                if feature != -2:
                    raise ValueError("frozen regression tree leaves require feature sentinel -2")
                continue
            if left_child == -1 or right_child == -1 or left_child == right_child:
                raise ValueError("frozen regression tree nodes require two distinct children")
            if not (index < left_child < size and index < right_child < size):
                raise ValueError("frozen regression tree child indices must form an acyclic tree")
            if not 0 <= feature < _FEATURE_WIDTH:
                raise ValueError("frozen regression tree feature index is out of range")
            parents[left_child] += 1
            parents[right_child] += 1
        if parents[0] != 0 or any(parent_count != 1 for parent_count in parents[1:]):
            raise ValueError("frozen regression tree nodes must form one connected tree")
        object.__setattr__(self, "children_left", left)
        object.__setattr__(self, "children_right", right)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "thresholds", tuple(float(value) for value in thresholds))
        object.__setattr__(self, "values", tuple(float(value) for value in values))

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != _FEATURE_WIDTH or any(not _finite(value) for value in features):
            raise ValueError("tree prediction requires one finite selector feature vector")
        node = 0
        while self.children_left[node] != -1:
            node = (
                self.children_left[node]
                if features[self.features[node]] <= self.thresholds[node]
                else self.children_right[node]
            )
        return self.values[node]

    def to_artifact(self) -> dict[str, Any]:
        return {
            "children_left": list(self.children_left),
            "children_right": list(self.children_right),
            "features": list(self.features),
            "thresholds": list(self.thresholds),
            "values": list(self.values),
        }

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> FrozenRegressionTree:
        try:
            artifact = _strict_artifact_mapping(
                value,
                expected_keys={
                    "children_left",
                    "children_right",
                    "features",
                    "thresholds",
                    "values",
                },
                label="frozen regression tree",
            )
            return cls(
                tuple(
                    _strict_json_integer(item, label="children_left item")
                    for item in _strict_json_array(
                        artifact["children_left"],
                        label="children_left",
                    )
                ),
                tuple(
                    _strict_json_integer(item, label="children_right item")
                    for item in _strict_json_array(
                        artifact["children_right"],
                        label="children_right",
                    )
                ),
                tuple(
                    _strict_json_integer(item, label="feature item")
                    for item in _strict_json_array(artifact["features"], label="features")
                ),
                tuple(
                    _strict_json_number(item, label="threshold item")
                    for item in _strict_json_array(
                        artifact["thresholds"],
                        label="thresholds",
                    )
                ),
                tuple(
                    _strict_json_number(item, label="tree value")
                    for item in _strict_json_array(artifact["values"], label="values")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid frozen regression tree artifact") from exc


@dataclass(frozen=True, slots=True)
class FrozenGBDTOutcomeModel:
    initial_value: float
    learning_rate: float
    trees: tuple[FrozenRegressionTree, ...]

    def __post_init__(self) -> None:
        trees = tuple(self.trees)
        if not _finite(self.initial_value):
            raise ValueError("GBDT initial value must be finite")
        if not _finite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("GBDT learning rate must be finite and positive")
        if not trees or any(not isinstance(tree, FrozenRegressionTree) for tree in trees):
            raise ValueError("GBDT outcome model requires frozen regression trees")
        object.__setattr__(self, "trees", trees)

    def predict(self, features: Sequence[float]) -> float:
        return self.initial_value + self.learning_rate * math.fsum(
            tree.predict(features) for tree in self.trees
        )

    def to_artifact(self) -> dict[str, Any]:
        return {
            "initial_value": self.initial_value,
            "learning_rate": self.learning_rate,
            "trees": [tree.to_artifact() for tree in self.trees],
        }

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> FrozenGBDTOutcomeModel:
        try:
            artifact = _strict_artifact_mapping(
                value,
                expected_keys={"initial_value", "learning_rate", "trees"},
                label="frozen GBDT outcome",
            )
            trees = tuple(
                FrozenRegressionTree.from_artifact(item)
                for item in _strict_json_array(artifact["trees"], label="trees")
            )
            return cls(
                _strict_json_number(artifact["initial_value"], label="initial_value"),
                _strict_json_number(artifact["learning_rate"], label="learning_rate"),
                trees,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid frozen GBDT outcome artifact") from exc


@dataclass(frozen=True, slots=True)
class GBDTTLearnerSelector:
    """Pure-Python inference for frozen sklearn GBDT T-learner artifacts."""

    outcome_models: Mapping[SemanticFamily, FrozenGBDTOutcomeModel]
    release_by_family: Mapping[SemanticFamily, str]
    sklearn_version: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    random_state: int
    min_gain: float = 0.0
    margin: float = 0.0
    train_id_set_digest: str = _EMPTY_TRAIN_ID_SET_DIGEST
    train_example_count: int = 0

    def __post_init__(self) -> None:
        models = dict(self.outcome_models)
        if SemanticFamily.NO_OP not in models:
            raise ValueError("GBDT T-learner requires an explicit NO_OP outcome model")
        if any(
            family not in {*_ACTIONS, SemanticFamily.NO_OP}
            or not isinstance(model, FrozenGBDTOutcomeModel)
            for family, model in models.items()
        ):
            raise ValueError("GBDT outcome models require typed selector actions")
        releases = _validated_release_mapping(self.release_by_family)
        if set(models) - {SemanticFamily.NO_OP} != set(releases):
            raise ValueError("GBDT T-learner models and release mappings must match")
        if not isinstance(self.sklearn_version, str) or not self.sklearn_version.strip():
            raise ValueError("GBDT artifact requires the sklearn fit version")
        for name, value in (
            ("n_estimators", self.n_estimators),
            ("max_depth", self.max_depth),
            ("min_samples_leaf", self.min_samples_leaf),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"GBDT {name} must be a positive integer")
        if any(len(model.trees) != self.n_estimators for model in models.values()):
            raise ValueError("GBDT outcome models must match the declared n_estimators")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("GBDT random_state must be an integer")
        if not _valid_sha256(self.train_id_set_digest):
            raise ValueError("GBDT train_id_set_digest must be a lowercase SHA-256 digest")
        if (
            isinstance(self.train_example_count, bool)
            or not isinstance(self.train_example_count, int)
            or self.train_example_count < 0
        ):
            raise ValueError("GBDT train_example_count must be a non-negative integer")
        _validate_abstention(self.min_gain, self.margin)
        object.__setattr__(self, "outcome_models", MappingProxyType(models))
        object.__setattr__(self, "release_by_family", MappingProxyType(releases))

    @property
    def model_hash(self) -> str:
        return content_hash(self.to_artifact())

    def predict_outcomes(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        vector = features.as_vector()
        return {family: model.predict(vector) for family, model in self.outcome_models.items()}

    def predict_gains(self, features: SelectionFeatures) -> dict[SemanticFamily, float]:
        outcomes = self.predict_outcomes(features)
        no_op = outcomes[SemanticFamily.NO_OP]
        return {family: outcomes[family] - no_op for family in self.release_by_family}

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        del state
        return _decision_from_gains(
            self.predict_gains(features),
            self.release_by_family,
            eligible_families=eligible_families,
            min_gain=self.min_gain,
            margin=self.margin,
            label="GBDT T-learner",
        )

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "selector_kind": "gbdt_t_learner",
            "outcome_models": {
                family.value: model.to_artifact() for family, model in self.outcome_models.items()
            },
            "release_by_family": {
                family.value: release for family, release in self.release_by_family.items()
            },
            "fit": {
                "sklearn_version": self.sklearn_version,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            },
            "abstention": {"min_gain": self.min_gain, "margin": self.margin},
            "provenance": {
                "train_id_set_digest": self.train_id_set_digest,
                "train_example_count": self.train_example_count,
            },
        }

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> GBDTTLearnerSelector:
        try:
            artifact = _strict_artifact_mapping(
                value,
                expected_keys={
                    "schema_version",
                    "selector_kind",
                    "outcome_models",
                    "release_by_family",
                    "fit",
                    "abstention",
                    "provenance",
                },
                label="GBDT selector",
            )
            schema_version = _strict_json_integer(
                artifact["schema_version"],
                label="schema_version",
            )
            if (
                schema_version != 2
                or artifact["selector_kind"] != "gbdt_t_learner"
                or not isinstance(artifact["selector_kind"], str)
            ):
                raise ValueError("unsupported GBDT selector artifact schema")
            raw_models = _strict_string_keyed_mapping(
                artifact["outcome_models"],
                label="outcome_models",
            )
            raw_releases = _strict_string_keyed_mapping(
                artifact["release_by_family"],
                label="release_by_family",
            )
            fit = _strict_artifact_mapping(
                artifact["fit"],
                expected_keys={
                    "sklearn_version",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "random_state",
                },
                label="GBDT fit",
            )
            abstention = _strict_artifact_mapping(
                artifact["abstention"],
                expected_keys={"min_gain", "margin"},
                label="GBDT abstention",
            )
            provenance = _strict_artifact_mapping(
                artifact["provenance"],
                expected_keys={"train_id_set_digest", "train_example_count"},
                label="GBDT provenance",
            )
            models: dict[SemanticFamily, FrozenGBDTOutcomeModel] = {}
            for family, model in raw_models.items():
                if not isinstance(model, Mapping):
                    raise TypeError("GBDT outcome model entries must be mappings")
                models[SemanticFamily(family)] = FrozenGBDTOutcomeModel.from_artifact(model)
            releases: dict[SemanticFamily, str] = {}
            for family, release in raw_releases.items():
                if not isinstance(release, str) or not release.strip():
                    raise TypeError("GBDT release IDs must be non-empty strings")
                releases[SemanticFamily(family)] = release
            sklearn_version = fit["sklearn_version"]
            if not isinstance(sklearn_version, str) or not sklearn_version.strip():
                raise TypeError("GBDT sklearn_version must be a non-empty string")
            train_id_set_digest = provenance["train_id_set_digest"]
            if not isinstance(train_id_set_digest, str):
                raise TypeError("GBDT train_id_set_digest must be a string")
            return cls(
                models,
                releases,
                sklearn_version=sklearn_version,
                n_estimators=_strict_json_integer(
                    fit["n_estimators"],
                    label="n_estimators",
                ),
                max_depth=_strict_json_integer(fit["max_depth"], label="max_depth"),
                min_samples_leaf=_strict_json_integer(
                    fit["min_samples_leaf"],
                    label="min_samples_leaf",
                ),
                random_state=_strict_json_integer(
                    fit["random_state"],
                    label="random_state",
                ),
                min_gain=_strict_json_number(
                    abstention["min_gain"],
                    label="min_gain",
                ),
                margin=_strict_json_number(abstention["margin"], label="margin"),
                train_id_set_digest=train_id_set_digest,
                train_example_count=_strict_json_integer(
                    provenance["train_example_count"],
                    label="train_example_count",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid GBDT selector artifact") from exc


def fit_gbdt_t_learner(
    examples: Sequence[SelectorExample],
    *,
    release_by_family: Mapping[SemanticFamily, str],
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_samples_leaf: int = 2,
    random_state: int = 0,
) -> GBDTTLearnerSelector:
    """Fit sklearn GBDTs lazily, then detach into safe pure-Python artifacts."""

    try:
        import sklearn
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - exercised in a clean optional-dep env
        raise ImportError(
            "GBDT selector fitting requires `pip install agentinputlab[selectors]`"
        ) from exc
    rows = _training_rows(examples, label="GBDT T-learner")
    releases = _validated_release_mapping(release_by_family)
    for name, value in (
        ("n_estimators", n_estimators),
        ("max_depth", max_depth),
        ("min_samples_leaf", min_samples_leaf),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"GBDT {name} must be a positive integer")
    if not _finite(learning_rate) or learning_rate <= 0:
        raise ValueError("GBDT learning_rate must be finite and positive")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("GBDT random_state must be an integer")
    models: dict[SemanticFamily, FrozenGBDTOutcomeModel] = {}
    for family in (SemanticFamily.NO_OP, *releases):
        supported = tuple(example for example in rows if family in example.utilities)
        if len(supported) < 2:
            raise ValueError(f"GBDT T-learner needs at least two outcomes for {family.value}")
        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            loss="squared_error",
        )
        estimator.fit(
            [example.features.as_vector() for example in supported],
            [float(example.utilities[family]) for example in supported],
        )
        models[family] = _freeze_sklearn_gbdt(estimator)
    return GBDTTLearnerSelector(
        models,
        releases,
        sklearn_version=str(sklearn.__version__),
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        train_id_set_digest=_selector_id_set_digest(rows, required_split="train"),
        train_example_count=len(rows),
    )


@dataclass(frozen=True, slots=True)
class CostRiskAwareSelector:
    """Two-stage wrapper that applies frozen cost and hard-risk guardrails."""

    base_selector: UtilityGainProvider
    release_by_family: Mapping[SemanticFamily, str]
    cost_by_family: Mapping[SemanticFamily, float]
    risk_by_family: Mapping[SemanticFamily, float]
    max_cost: float
    max_risk: float
    cost_weight: float = 0.0
    risk_weight: float = 0.0
    min_gain: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_selector, UtilityGainProvider):
            raise ValueError(
                "cost/risk wrapper requires an explicit UtilityGainProvider base model"
            )
        _selector_hash(self.base_selector)
        releases = _validated_release_mapping(self.release_by_family)
        declared_releases = getattr(self.base_selector, "release_by_family", None)
        if isinstance(declared_releases, Mapping) and set(declared_releases) != set(releases):
            raise ValueError("utility-gain provider and guardrail release families must match")
        costs = _validated_family_values(
            self.cost_by_family,
            families=set(releases),
            label="cost",
            upper=None,
        )
        risks = _validated_family_values(
            self.risk_by_family,
            families=set(releases),
            label="risk",
            upper=1.0,
        )
        if not _finite(self.max_cost) or self.max_cost < 0:
            raise ValueError("max_cost must be finite and non-negative")
        if not _finite(self.max_risk) or not 0 <= self.max_risk <= 1:
            raise ValueError("max_risk must be finite and in [0, 1]")
        if not _finite(self.cost_weight) or self.cost_weight < 0:
            raise ValueError("cost_weight must be finite and non-negative")
        if not _finite(self.risk_weight) or self.risk_weight < 0:
            raise ValueError("risk_weight must be finite and non-negative")
        _validate_abstention(self.min_gain, self.margin)
        object.__setattr__(self, "release_by_family", MappingProxyType(releases))
        object.__setattr__(self, "cost_by_family", MappingProxyType(costs))
        object.__setattr__(self, "risk_by_family", MappingProxyType(risks))

    @property
    def model_hash(self) -> str:
        return content_hash(
            {
                "selector": "cost_risk_aware",
                "base_selector_hash": _selector_hash(self.base_selector),
                "release_by_family": {
                    family.value: release for family, release in self.release_by_family.items()
                },
                "cost_by_family": {
                    family.value: value for family, value in self.cost_by_family.items()
                },
                "risk_by_family": {
                    family.value: value for family, value in self.risk_by_family.items()
                },
                "guards": {
                    "max_cost": self.max_cost,
                    "max_risk": self.max_risk,
                    "cost_weight": self.cost_weight,
                    "risk_weight": self.risk_weight,
                    "min_gain": self.min_gain,
                    "margin": self.margin,
                },
            }
        )

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        eligible = _eligible_actions(eligible_families) & set(self.release_by_family)
        base = self.base_selector.select(
            state,
            features,
            eligible_families=eligible,
        )
        gains = _validated_predicted_gains(
            self.base_selector.predict_gains(features),
            expected_families=set(self.release_by_family),
        )
        adjusted: dict[SemanticFamily, float] = {}
        safe: dict[SemanticFamily, float] = {}
        for family in self.release_by_family:
            raw = gains[family]
            value = (
                raw
                - self.cost_weight * self.cost_by_family[family]
                - self.risk_weight * self.risk_by_family[family]
            )
            adjusted[family] = value
            if (
                family in eligible
                and self.cost_by_family[family] <= self.max_cost
                and self.risk_by_family[family] <= self.max_risk
            ):
                safe[family] = value
        scores = {family.value: value for family, value in adjusted.items()}
        scores[SemanticFamily.NO_OP.value] = 0.0
        if base.family is SemanticFamily.NO_OP:
            return _no_op("base selector abstained before safety guards", scores)
        if not safe:
            return _no_op("no eligible family passed cost/risk guards", scores)
        ordered = sorted(safe.items(), key=lambda item: (-item[1], item[0].value))
        family, gain = ordered[0]
        runner_up = max(0.0, ordered[1][1] if len(ordered) > 1 else 0.0)
        if gain <= self.min_gain or gain - runner_up <= self.margin:
            return _no_op("adjusted gain or margin gate not met", scores)
        return SelectionDecision(
            family=family,
            release_id=self.release_by_family[family],
            selection_score=gain,
            rationale=f"selected cost/risk-adjusted utility winner: {family.value}",
            scores=scores,
        )


def fit_cost_risk_selector(
    base_selector: UtilityGainProvider,
    examples: Sequence[SelectorExample],
    *,
    release_by_family: Mapping[SemanticFamily, str],
    max_cost: float,
    max_risk: float,
    cost_weight: float = 0.0,
    risk_weight: float = 0.0,
) -> CostRiskAwareSelector:
    """Estimate static family cost/risk priors from discovery ``train`` rows."""

    if not isinstance(base_selector, UtilityGainProvider):
        raise ValueError("cost/risk wrapper requires an explicit UtilityGainProvider base model")
    rows = _training_rows(examples, label="cost/risk selector")
    releases = _validated_release_mapping(release_by_family)
    costs: dict[SemanticFamily, float] = {}
    risks: dict[SemanticFamily, float] = {}
    for family in releases:
        family_costs = [example.costs[family] for example in rows if family in example.costs]
        family_risks = [example.risks[family] for example in rows if family in example.risks]
        if not family_costs or not family_risks:
            raise ValueError(f"cost/risk selector has no train priors for {family.value}")
        costs[family] = fmean(family_costs)
        risks[family] = fmean(family_risks)
    return CostRiskAwareSelector(
        base_selector,
        releases,
        costs,
        risks,
        max_cost=max_cost,
        max_risk=max_risk,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )


@dataclass(frozen=True, slots=True)
class SelectorCalibrationResult:
    selector: Selector
    evaluation: PolicyEvaluation
    candidates_considered: int


def calibrate_selector_abstention(
    selector: Selector,
    examples: Sequence[SelectorExample],
    *,
    gain_grid: Sequence[float],
    margin_grid: Sequence[float],
    max_harm_rate: float,
) -> SelectorCalibrationResult:
    """Freeze gain/margin gates using calibration rows, never test outcomes."""

    rows = _calibration_rows(examples)
    if (
        not is_dataclass(selector)
        or not hasattr(selector, "min_gain")
        or not hasattr(selector, "margin")
    ):
        raise TypeError("selector does not expose calibratable min_gain and margin fields")
    if not 0 <= max_harm_rate <= 1:
        raise ValueError("max_harm_rate must be in [0, 1]")
    gains = _canonical_calibration_grid(gain_grid, label="gain")
    margins = _canonical_calibration_grid(
        margin_grid,
        label="margin",
        lower=0.0,
    )
    candidates: list[tuple[PolicyEvaluation, Selector]] = []
    considered = 0
    for gain in gains:
        for margin in margins:
            considered += 1
            candidate = replace(selector, min_gain=gain, margin=margin)
            if not isinstance(candidate, Selector):  # pragma: no cover - dataclass invariant
                raise TypeError("calibrated object no longer satisfies Selector")
            evaluation = evaluate_selector(candidate, rows, required_split="calibration")
            if evaluation.harm_rate <= max_harm_rate:
                candidates.append((evaluation, candidate))
    return _best_calibration(candidates, considered)


def calibrate_cost_risk_guardrails(
    selector: CostRiskAwareSelector,
    examples: Sequence[SelectorExample],
    *,
    max_cost_grid: Sequence[float],
    max_risk_grid: Sequence[float],
    gain_grid: Sequence[float],
    margin_grid: Sequence[float],
    max_harm_rate: float,
) -> SelectorCalibrationResult:
    """Freeze cost, risk, gain, and margin gates on calibration rows only."""

    rows = _calibration_rows(examples)
    costs = _canonical_calibration_grid(
        max_cost_grid,
        label="max_cost",
        lower=0.0,
    )
    risks = _canonical_calibration_grid(
        max_risk_grid,
        label="max_risk",
        lower=0.0,
        upper=1.0,
    )
    gains = _canonical_calibration_grid(gain_grid, label="gain")
    margins = _canonical_calibration_grid(
        margin_grid,
        label="margin",
        lower=0.0,
    )
    if not 0 <= max_harm_rate <= 1:
        raise ValueError("max_harm_rate must be in [0, 1]")
    candidates: list[tuple[PolicyEvaluation, Selector]] = []
    considered = 0
    for max_cost in costs:
        for max_risk in risks:
            for gain in gains:
                for margin in margins:
                    considered += 1
                    candidate = replace(
                        selector,
                        max_cost=max_cost,
                        max_risk=max_risk,
                        min_gain=gain,
                        margin=margin,
                    )
                    evaluation = evaluate_selector(
                        candidate,
                        rows,
                        required_split="calibration",
                    )
                    if evaluation.harm_rate <= max_harm_rate:
                        candidates.append((evaluation, candidate))
    return _best_calibration(candidates, considered)


def _best_calibration(
    candidates: Sequence[tuple[PolicyEvaluation, Selector]],
    considered: int,
) -> SelectorCalibrationResult:
    if not candidates:
        raise ValueError("no calibration candidate satisfies the harm constraint")
    evaluation, selector = max(
        candidates,
        key=lambda item: (
            item[0].mean_policy_value,
            item[0].coverage,
            -item[0].harm_rate,
            -float(item[1].max_cost) if isinstance(item[1], CostRiskAwareSelector) else 0.0,
            -float(item[1].max_risk) if isinstance(item[1], CostRiskAwareSelector) else 0.0,
            -float(item[1].min_gain),
            -float(item[1].margin),
        ),
    )
    return SelectorCalibrationResult(selector, evaluation, considered)


def _training_rows(
    examples: Sequence[SelectorExample],
    *,
    label: str,
) -> tuple[SelectorExample, ...]:
    rows = tuple(examples)
    if not rows:
        raise ValueError(f"{label} requires training examples")
    if any(not isinstance(example, SelectorExample) for example in rows):
        raise TypeError(f"{label} examples must be SelectorExample instances")
    if any(example.split != "train" for example in rows):
        raise ValueError(f"{label} only accepts split='train' examples")
    ids = [example.example_id for example in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} requires unique example IDs")
    return rows


def _calibration_rows(examples: Sequence[SelectorExample]) -> tuple[SelectorExample, ...]:
    rows = tuple(examples)
    if not rows or any(not isinstance(example, SelectorExample) for example in rows):
        raise ValueError("selector calibration requires SelectorExample rows")
    if any(example.split != "calibration" for example in rows):
        raise ValueError("selector calibration only accepts split='calibration' examples")
    ids = [example.example_id for example in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("selector calibration requires unique example IDs")
    return rows


def _canonical_calibration_grid(
    values: Sequence[float],
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError(f"{label} grid must be non-empty")
    normalized: list[float] = []
    for value in raw:
        if not _finite(value):
            raise ValueError(f"{label} grid must contain finite numeric values")
        number = float(value)
        if lower is not None and number < lower:
            raise ValueError(f"{label} grid values must be at least {lower:g}")
        if upper is not None and number > upper:
            raise ValueError(f"{label} grid values must be at most {upper:g}")
        normalized.append(number)
    return tuple(sorted(set(normalized)))


def _validated_release_mapping(
    value: Mapping[SemanticFamily, str],
) -> dict[SemanticFamily, str]:
    copied = dict(value)
    if not copied:
        raise ValueError("selector requires at least one release mapping")
    if any(
        family not in _ACTIONS or not isinstance(release, str) or not release.strip()
        for family, release in copied.items()
    ):
        raise ValueError("release mappings require intervention-family keys and IDs")
    return copied


def _validated_family_values(
    value: Mapping[SemanticFamily, float],
    *,
    families: set[SemanticFamily],
    label: str,
    upper: float | None,
) -> dict[SemanticFamily, float]:
    copied = dict(value)
    if set(copied) != families:
        raise ValueError(f"{label} priors must exactly match release families")
    if any(
        not _finite(item) or item < 0 or (upper is not None and item > upper)
        for item in copied.values()
    ):
        bound = f" in [0, {upper:g}]" if upper is not None else " and non-negative"
        raise ValueError(f"{label} priors must be finite{bound}")
    return {family: float(item) for family, item in copied.items()}


def _validated_predicted_gains(
    value: Mapping[SemanticFamily, float],
    *,
    expected_families: set[SemanticFamily],
) -> dict[SemanticFamily, float]:
    if not isinstance(value, Mapping):
        raise TypeError("utility-gain provider must return a mapping")
    copied = dict(value)
    if set(copied) != expected_families or any(
        not isinstance(family, SemanticFamily) or family is SemanticFamily.NO_OP
        for family in copied
    ):
        raise ValueError("utility-gain provider outputs must exactly match release families")
    if any(not _finite(gain) for gain in copied.values()):
        raise ValueError("utility-gain provider outputs must be finite numbers")
    return {family: float(gain) for family, gain in copied.items()}


def _validate_abstention(min_gain: float, margin: float) -> None:
    if not _finite(min_gain):
        raise ValueError("selector min_gain must be finite")
    if not _finite(margin) or margin < 0:
        raise ValueError("selector margin must be finite and non-negative")


def _validate_logistic_config(l2: float, learning_rate: float, iterations: int) -> None:
    if not _finite(l2) or l2 < 0:
        raise ValueError("logistic l2 must be finite and non-negative")
    if not _finite(learning_rate) or learning_rate <= 0:
        raise ValueError("logistic learning_rate must be finite and positive")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("logistic iterations must be a positive integer")


def _decision_from_gains(
    gains: Mapping[SemanticFamily, float],
    release_by_family: Mapping[SemanticFamily, str],
    *,
    eligible_families: Collection[SemanticFamily] | None,
    min_gain: float,
    margin: float,
    label: str,
) -> SelectionDecision:
    eligible = _eligible_actions(eligible_families)
    scores = {family.value: value for family, value in gains.items()}
    scores[SemanticFamily.NO_OP.value] = 0.0
    ordered = sorted(
        ((family, gain) for family, gain in gains.items() if family in eligible),
        key=lambda item: (-item[1], item[0].value),
    )
    if not ordered:
        return _no_op("no modeled intervention family is eligible", scores)
    family, gain = ordered[0]
    runner_up = max(0.0, ordered[1][1] if len(ordered) > 1 else 0.0)
    if gain <= min_gain or gain - runner_up <= margin:
        return _no_op(f"{label} gain or margin gate not met", scores)
    return SelectionDecision(
        family=family,
        release_id=release_by_family[family],
        selection_score=gain,
        rationale=f"selected {label} winner: {family.value}",
        scores=scores,
    )


def _no_op(rationale: str, scores: Mapping[str, float]) -> SelectionDecision:
    return SelectionDecision(
        family=SemanticFamily.NO_OP,
        release_id="baseline.no_op",
        selection_score=0.0,
        rationale=f"abstained: {rationale}",
        scores=scores,
    )


def _eligible_actions(
    eligible_families: Collection[SemanticFamily] | None,
) -> set[SemanticFamily]:
    if eligible_families is None:
        return set(_ACTIONS)
    eligible = tuple(eligible_families)
    if any(not isinstance(family, SemanticFamily) for family in eligible):
        raise ValueError("eligible_families must contain SemanticFamily values")
    return {family for family in eligible if family in _ACTIONS}


def _selector_hash(selector: Selector) -> str:
    declared = getattr(selector, "model_hash", None)
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    try:
        return content_hash(selector)
    except TypeError as exc:
        raise ValueError("base selector must be frozen and canonically hashable") from exc


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(math.fsum((one - two) ** 2 for one, two in zip(left, right, strict=True)))


def _fit_ridge(
    matrix: Sequence[tuple[float, ...]],
    labels: Sequence[float],
    *,
    ridge: float,
) -> tuple[float, ...]:
    width = len(matrix[0])
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, label in zip(matrix, labels, strict=True):
        for left in range(width):
            rhs[left] += row[left] * label
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    gram[0][0] += ridge * 1e-3
    for index in range(1, width):
        gram[index][index] += ridge
    return tuple(_solve_linear_system(gram, rhs))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [[*row, value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("T-learner design matrix is singular; increase ridge")
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


def _freeze_sklearn_gbdt(estimator: Any) -> FrozenGBDTOutcomeModel:
    try:
        initial_value = float(estimator.init_.constant_.ravel()[0])
        trees = []
        for row in estimator.estimators_:
            fitted = row[0].tree_
            trees.append(
                FrozenRegressionTree(
                    tuple(int(value) for value in fitted.children_left.tolist()),
                    tuple(int(value) for value in fitted.children_right.tolist()),
                    tuple(int(value) for value in fitted.feature.tolist()),
                    tuple(float(value) for value in fitted.threshold.tolist()),
                    tuple(float(value) for value in fitted.value[:, 0, 0].tolist()),
                )
            )
        return FrozenGBDTOutcomeModel(initial_value, float(estimator.learning_rate), tuple(trees))
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("unsupported sklearn GBDT fitted structure") from exc


def _fit_logistic(
    matrix: Sequence[tuple[float, ...]],
    labels: Sequence[float],
    *,
    l2: float,
    learning_rate: float,
    iterations: int,
) -> tuple[float, ...]:
    width = len(matrix[0])
    weights = [0.0] * width
    count = len(matrix)
    for _ in range(iterations):
        gradient = [0.0] * width
        for row, label in zip(matrix, labels, strict=True):
            error = _sigmoid(math.fsum(a * b for a, b in zip(weights, row, strict=True))) - label
            for index, value in enumerate(row):
                gradient[index] += error * value / count
        for index in range(1, width):
            gradient[index] += l2 * weights[index]
        for index in range(width):
            weights[index] -= learning_rate * gradient[index]
    return tuple(weights)


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _strict_artifact_mapping(
    value: object,
    *,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    mapping = _strict_string_keyed_mapping(value, label=label)
    if set(mapping) != expected_keys:
        raise ValueError(f"{label} artifact keys do not match the canonical schema")
    return mapping


def _strict_string_keyed_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} keys must be JSON strings")
    return value


def _strict_json_array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    return value


def _strict_json_integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a JSON integer")
    return value


def _strict_json_number(value: object, *, label: str) -> float:
    if not _finite(value):
        raise TypeError(f"{label} must be a finite JSON number")
    return float(value)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


__all__ = [
    "CostRiskAwareSelector",
    "FrozenGBDTOutcomeModel",
    "FrozenRegressionTree",
    "GBDTTLearnerSelector",
    "KNNTrainingRow",
    "KNNUtilitySelector",
    "LogisticTLearnerSelector",
    "RidgeTLearnerSelector",
    "SelectorCalibrationResult",
    "calibrate_cost_risk_guardrails",
    "calibrate_selector_abstention",
    "fit_cost_risk_selector",
    "fit_gbdt_t_learner",
    "fit_knn_selector",
    "fit_logistic_t_learner",
    "fit_ridge_t_learner",
]
