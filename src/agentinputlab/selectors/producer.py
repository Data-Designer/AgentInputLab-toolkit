"""Explicit selector fit, calibration, and freeze producers.

The functions here consume a previously frozen :class:`SelectorDiscoveryFreeze`.
They never execute benchmark cells or fetch external assets.  Every transition
returns an immutable selector bundle whose store enforces the lifecycle.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .advanced import (
    CostRiskAwareSelector,
    GBDTTLearnerSelector,
    KNNUtilitySelector,
    LogisticTLearnerSelector,
    RidgeTLearnerSelector,
    SelectorCalibrationResult,
    calibrate_cost_risk_guardrails,
    calibrate_selector_abstention,
    fit_cost_risk_selector,
    fit_gbdt_t_learner,
    fit_knn_selector,
    fit_logistic_t_learner,
    fit_ridge_t_learner,
)
from .artifacts import SelectorArtifactBundle, SelectorArtifactStage
from .base import Selector, UtilityGainProvider
from .discovery import SelectorDiscoveryFreeze
from .learned import (
    PolicyEvaluation,
    RidgeUtilitySelector,
    SelectorExample,
    evaluate_selector,
    fit_ridge_selector,
)
from .rules import RuleSelector

_SELECTOR_KINDS = frozenset(
    {
        "rule",
        "ridge_utility",
        "ridge_t_learner",
        "logistic_t_learner",
        "gbdt_t_learner",
        "knn_utility",
        "cost_risk_guardrail",
    }
)
_PRIVACY_SAFE_GROUP_ID = re.compile(r"^group_[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SelectorCalibrationSummary:
    selector: Selector
    evaluation: PolicyEvaluation
    candidates_considered: int
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.selector, Selector):
            raise ValidationError("calibration summary requires a Selector")
        if not isinstance(self.evaluation, PolicyEvaluation):
            raise ValidationError("calibration summary requires PolicyEvaluation")
        if (
            isinstance(self.candidates_considered, bool)
            or not isinstance(self.candidates_considered, int)
            or self.candidates_considered < 1
        ):
            raise ValidationError("calibration candidates_considered must be positive")
        object.__setattr__(self, "receipt", freeze_mapping(self.receipt))


def fit_selector(
    discovery: SelectorDiscoveryFreeze,
    *,
    selector_kind: str,
    fit_config: Mapping[str, Any],
    base_selector_bundle: SelectorArtifactBundle | None = None,
) -> Selector:
    """Fit one of the seven catalogued selectors from ``split='train'`` rows."""

    if not isinstance(discovery, SelectorDiscoveryFreeze):
        raise TypeError("selector fitting requires SelectorDiscoveryFreeze")
    if selector_kind not in _SELECTOR_KINDS:
        raise ValidationError(f"unsupported selector_kind: {selector_kind!r}")
    if selector_kind != "cost_risk_guardrail" and base_selector_bundle is not None:
        raise ValidationError("base_selector_bundle is valid only for cost_risk_guardrail")
    config = _mapping(fit_config, label="fit config")
    train = _examples_for_split(discovery, "train")
    releases = discovery.representative_by_family

    if selector_kind == "rule":
        _require_exact_keys(config, {"threshold", "margin"}, label="rule fit config")
        return RuleSelector(
            threshold=_number(config["threshold"], label="threshold"),
            margin=_number(config["margin"], label="margin"),
        )
    if selector_kind == "ridge_utility":
        _require_exact_keys(config, {"ridge"}, label="ridge utility fit config")
        return fit_ridge_selector(
            train,
            release_by_family=releases,
            ridge=_nonnegative(config["ridge"], label="ridge"),
        )
    if selector_kind == "ridge_t_learner":
        _require_exact_keys(config, {"ridge"}, label="ridge T-learner fit config")
        return fit_ridge_t_learner(
            train,
            release_by_family=releases,
            ridge=_nonnegative(config["ridge"], label="ridge"),
        )
    if selector_kind == "logistic_t_learner":
        _require_exact_keys(
            config,
            {"l2", "learning_rate", "iterations"},
            label="logistic T-learner fit config",
        )
        return fit_logistic_t_learner(
            train,
            release_by_family=releases,
            l2=_nonnegative(config["l2"], label="l2"),
            learning_rate=_positive(
                config["learning_rate"],
                label="learning_rate",
            ),
            iterations=_positive_integer(
                config["iterations"],
                label="iterations",
            ),
        )
    if selector_kind == "gbdt_t_learner":
        _require_exact_keys(
            config,
            {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_leaf",
                "random_state",
            },
            label="GBDT T-learner fit config",
        )
        return fit_gbdt_t_learner(
            train,
            release_by_family=releases,
            n_estimators=_positive_integer(
                config["n_estimators"],
                label="n_estimators",
            ),
            learning_rate=_positive(
                config["learning_rate"],
                label="learning_rate",
            ),
            max_depth=_positive_integer(config["max_depth"], label="max_depth"),
            min_samples_leaf=_positive_integer(
                config["min_samples_leaf"],
                label="min_samples_leaf",
            ),
            random_state=_integer(config["random_state"], label="random_state"),
        )
    if selector_kind == "knn_utility":
        _require_exact_keys(
            config,
            {"k", "distance_power"},
            label="KNN utility fit config",
        )
        return fit_knn_selector(
            train,
            release_by_family=releases,
            k=_positive_integer(config["k"], label="k"),
            distance_power=_positive(
                config["distance_power"],
                label="distance_power",
            ),
        )

    _require_exact_keys(
        config,
        {"max_cost", "max_risk", "cost_weight", "risk_weight"},
        label="cost/risk guardrail fit config",
    )
    base_selector = _guardrail_base_selector(
        discovery,
        base_selector_bundle,
    )
    declared_releases = getattr(base_selector, "release_by_family", None)
    if not isinstance(declared_releases, Mapping) or dict(declared_releases) != dict(releases):
        raise ValidationError(
            "cost/risk base selector release mapping differs from discovery freeze"
        )
    return fit_cost_risk_selector(
        base_selector,
        train,
        release_by_family=releases,
        max_cost=_nonnegative(config["max_cost"], label="max_cost"),
        max_risk=_bounded(config["max_risk"], label="max_risk"),
        cost_weight=_nonnegative(config["cost_weight"], label="cost_weight"),
        risk_weight=_nonnegative(config["risk_weight"], label="risk_weight"),
    )


def create_fitted_selector_bundle(
    discovery: SelectorDiscoveryFreeze,
    *,
    selector_kind: str,
    revision: str,
    fit_config: Mapping[str, Any],
    base_selector_bundle: SelectorArtifactBundle | None = None,
) -> SelectorArtifactBundle:
    """Fit and bind a selector to one exact discovery freeze."""

    selector = fit_selector(
        discovery,
        selector_kind=selector_kind,
        fit_config=fit_config,
        base_selector_bundle=base_selector_bundle,
    )
    fit_lineage: Mapping[str, Any] = _selector_fit_lineage(
        discovery,
        fit_config=fit_config,
    )
    if selector_kind == "cost_risk_guardrail":
        if not isinstance(base_selector_bundle, SelectorArtifactBundle):
            raise ValidationError("cost_risk_guardrail requires a SelectorArtifactBundle base")
        fit_lineage = _guardrail_fit_lineage(
            discovery,
            base_selector_bundle,
            fit_config=fit_config,
        )
    return SelectorArtifactBundle.create(
        selector,
        stage=SelectorArtifactStage.FITTED,
        revision=revision,
        extractor_id=discovery.extractor_id,
        extractor_revision=discovery.extractor_revision,
        extractor_hash=discovery.extractor_hash,
        feature_schema_id=discovery.feature_schema_id,
        feature_schema_hash=discovery.feature_schema_hash,
        representative_by_family=discovery.representative_by_family,
        training_table_sha256=discovery.training_table.table_hash,
        split_provenance_hash=discovery.split_provenance_hash,
        comparison_profile_hash=discovery.comparison_profile_hash,
        fit_lineage=fit_lineage,
    )


def calibrate_selector_bundle(
    fitted: SelectorArtifactBundle,
    discovery: SelectorDiscoveryFreeze,
    *,
    revision: str,
    calibration_config: Mapping[str, Any],
) -> tuple[SelectorArtifactBundle, SelectorCalibrationSummary]:
    """Calibrate only predeclared policy gates on discovery calibration rows."""

    if not isinstance(fitted, SelectorArtifactBundle):
        raise TypeError("selector calibration requires SelectorArtifactBundle")
    if fitted.stage is not SelectorArtifactStage.FITTED:
        raise ValidationError("selector calibration requires a fitted bundle")
    _validate_discovery_binding(fitted, discovery)
    config = _mapping(calibration_config, label="calibration config")
    calibration = _examples_for_split(discovery, "calibration")
    selector = fitted.selector()

    if isinstance(selector, RuleSelector):
        _require_exact_keys(
            config,
            {"threshold_grid", "margin_grid", "max_harm_rate"},
            label="rule calibration config",
        )
        result = _calibrate_rule(
            selector,
            calibration,
            threshold_grid=_number_array(
                config["threshold_grid"],
                label="threshold_grid",
                lower=0.0,
                upper=1.0,
            ),
            margin_grid=_number_array(
                config["margin_grid"],
                label="margin_grid",
                lower=0.0,
            ),
            max_harm_rate=_bounded(
                config["max_harm_rate"],
                label="max_harm_rate",
            ),
        )
    elif isinstance(selector, CostRiskAwareSelector):
        _require_exact_keys(
            config,
            {
                "max_cost_grid",
                "max_risk_grid",
                "gain_grid",
                "margin_grid",
                "max_harm_rate",
            },
            label="cost/risk calibration config",
        )
        advanced = calibrate_cost_risk_guardrails(
            selector,
            calibration,
            max_cost_grid=_number_array(
                config["max_cost_grid"],
                label="max_cost_grid",
                lower=0.0,
            ),
            max_risk_grid=_number_array(
                config["max_risk_grid"],
                label="max_risk_grid",
                lower=0.0,
                upper=1.0,
            ),
            gain_grid=_number_array(config["gain_grid"], label="gain_grid"),
            margin_grid=_number_array(
                config["margin_grid"],
                label="margin_grid",
                lower=0.0,
            ),
            max_harm_rate=_bounded(
                config["max_harm_rate"],
                label="max_harm_rate",
            ),
        )
        result = _from_advanced_calibration(advanced)
    elif isinstance(
        selector,
        (
            RidgeUtilitySelector,
            RidgeTLearnerSelector,
            LogisticTLearnerSelector,
            GBDTTLearnerSelector,
            KNNUtilitySelector,
        ),
    ):
        _require_exact_keys(
            config,
            {"gain_grid", "margin_grid", "max_harm_rate"},
            label="selector calibration config",
        )
        advanced = calibrate_selector_abstention(
            selector,
            calibration,
            gain_grid=_number_array(config["gain_grid"], label="gain_grid"),
            margin_grid=_number_array(
                config["margin_grid"],
                label="margin_grid",
                lower=0.0,
            ),
            max_harm_rate=_bounded(
                config["max_harm_rate"],
                label="max_harm_rate",
            ),
        )
        result = _from_advanced_calibration(advanced)
    else:  # pragma: no cover - bundle codec admits exactly the seven selectors
        raise ValidationError(f"unsupported selector calibration type: {type(selector).__name__}")

    receipt = {
        "schema_version": 1,
        "predecessor_bundle_hash": fitted.bundle_hash,
        "discovery_hash": discovery.discovery_hash,
        "calibration_config_hash": content_hash(config),
        "calibration_split_digest": discovery.training_table.group_split_digests["calibration"],
        "candidates_considered": result.candidates_considered,
        "evaluation": _evaluation_to_artifact(result.evaluation),
    }
    summary = SelectorCalibrationSummary(
        selector=result.selector,
        evaluation=result.evaluation,
        candidates_considered=result.candidates_considered,
        receipt=receipt,
    )
    calibrated = SelectorArtifactBundle.create(
        result.selector,
        stage=SelectorArtifactStage.CALIBRATED,
        revision=revision,
        extractor_id=fitted.extractor_id,
        extractor_revision=fitted.extractor_revision,
        extractor_hash=fitted.extractor_hash,
        feature_schema_id=fitted.feature_schema_id,
        feature_schema_hash=fitted.feature_schema_hash,
        representative_by_family=fitted.representative_by_family,
        training_table_sha256=fitted.training_table_sha256,
        split_provenance_hash=fitted.split_provenance_hash,
        comparison_profile_hash=fitted.comparison_profile_hash,
        calibration_receipt=receipt,
        fit_lineage=fitted.fit_lineage,
    )
    return calibrated, summary


def freeze_selector_bundle(
    calibrated: SelectorArtifactBundle,
    *,
    revision: str,
) -> SelectorArtifactBundle:
    """Create a byte-equivalent policy at the terminal frozen stage."""

    if not isinstance(calibrated, SelectorArtifactBundle):
        raise TypeError("selector freeze requires SelectorArtifactBundle")
    if calibrated.stage is not SelectorArtifactStage.CALIBRATED:
        raise ValidationError("selector freeze requires a calibrated bundle")
    return replace(
        calibrated,
        stage=SelectorArtifactStage.FROZEN,
        revision=revision,
    )


def build_selector_policy_manifest(
    frozen: SelectorArtifactBundle,
    discovery: SelectorDiscoveryFreeze,
) -> Mapping[str, Any]:
    """Derive the locked global comparator from the frozen calibration split.

    The returned object is the exact schema consumed by the external-arena
    driver.  It is intentionally derived from a content-addressed discovery
    freeze and a selector bundle already at the terminal ``frozen`` stage;
    callers never type a family or release ID into the policy manifest by
    hand.
    """

    if not isinstance(frozen, SelectorArtifactBundle):
        raise TypeError("policy manifest production requires SelectorArtifactBundle")
    if frozen.stage is not SelectorArtifactStage.FROZEN:
        raise ValidationError("policy manifest requires a frozen selector bundle")
    _validate_discovery_binding(frozen, discovery)
    lineage = frozen.fit_lineage
    lineage_discovery = lineage.get("discovery") if isinstance(lineage, Mapping) else None
    if (
        not isinstance(lineage_discovery, Mapping)
        or lineage_discovery.get("revision") != discovery.revision
        or lineage_discovery.get("sha256") != discovery.discovery_hash
    ):
        raise ValidationError(
            "frozen selector fit lineage differs from the supplied discovery freeze"
        )
    calibration_receipt = frozen.calibration_receipt
    if calibration_receipt.get(
        "discovery_hash"
    ) != discovery.discovery_hash or calibration_receipt.get(
        "calibration_split_digest"
    ) != discovery.training_table.group_split_digests.get("calibration"):
        raise ValidationError(
            "frozen selector calibration receipt differs from the supplied discovery freeze"
        )

    calibration = _examples_for_split(discovery, "calibration")
    families = tuple(sorted(discovery.representative_by_family, key=lambda item: item.value))
    if len(families) != 5 or any(
        set(row.utilities) != {SemanticFamily.NO_OP, *families} for row in calibration
    ):
        raise ValidationError(
            "locked best-eligible-global selection requires a full-five-family "
            "calibration block for every state"
        )
    mean_utility = {
        family: fmean(row.utilities[family] for row in calibration) for family in families
    }
    selected_family = min(
        families,
        key=lambda family: (-mean_utility[family], family.value),
    )
    selected_release = discovery.representative_by_family[selected_family]
    calibration_digest = discovery.training_table.group_split_digests["calibration"]
    members_by_split = {
        split: tuple(
            sorted({outcome.group_id for outcome in discovery.outcomes if outcome.split == split})
        )
        for split in ("train", "calibration")
    }
    if any(
        not members or any(_PRIVACY_SAFE_GROUP_ID.fullmatch(item) is None for item in members)
        for members in members_by_split.values()
    ):
        raise ValidationError(
            "policy manifest requires privacy-safe benchmark-scoped group hashes "
            "for train and calibration"
        )
    if set(members_by_split["train"]).intersection(members_by_split["calibration"]):
        raise ValidationError("policy manifest group membership crosses splits")
    partition_splits = {
        split: {
            "member_hashes": members,
            "commitment_sha256": content_hash({"split": split, "group_ids": list(members)}),
        }
        for split, members in members_by_split.items()
    }
    if {split: row["commitment_sha256"] for split, row in partition_splits.items()} != {
        split: discovery.training_table.group_split_digests[split]
        for split in ("train", "calibration")
    }:
        raise ValidationError("policy group membership differs from the frozen split provenance")
    return freeze_mapping(
        {
            "schema_version": 1,
            "artifact": "agentinputlab.selector_policy_manifest",
            "stage": "frozen",
            "discovery": {
                "revision": discovery.revision,
                "sha256": discovery.discovery_hash,
            },
            "training_table_sha256": discovery.training_table.table_hash,
            "split_provenance_hash": discovery.split_provenance_hash,
            "comparison_profile_hash": discovery.comparison_profile_hash,
            "representative_by_family": {
                family.value: release_id
                for family, release_id in sorted(
                    discovery.representative_by_family.items(),
                    key=lambda item: item[0].value,
                )
            },
            "group_partition": {
                "schema_version": 1,
                "hash_scheme": "benchmark_group.content_hash.v1",
                "splits": partition_splits,
            },
            "locked_best_eligible_global": {
                "selected_family": selected_family.value,
                "selected_release_id": selected_release,
                "selection_receipt": {
                    "schema_version": 1,
                    "rule_id": "best_eligible_global.v1",
                    "selection_split": "calibration",
                    "training_table_sha256": discovery.training_table.table_hash,
                    "calibration_split_digest": calibration_digest,
                },
            },
        }
    )


def load_selector_fit_config(path: str | Path) -> tuple[str, Mapping[str, Any]]:
    raw = _read_json(path, label="selector fit config")
    _require_exact_keys(
        raw,
        {"schema_version", "selector_kind", "fit"},
        label="selector fit config",
    )
    if _integer(raw["schema_version"], label="schema_version") != 1:
        raise ValidationError("unsupported selector fit config schema")
    kind = _text(raw["selector_kind"], label="selector_kind")
    if kind not in _SELECTOR_KINDS:
        raise ValidationError(f"unsupported selector_kind: {kind!r}")
    return kind, freeze_mapping(_mapping(raw["fit"], label="fit"))


def load_selector_calibration_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read_json(path, label="selector calibration config")
    _require_exact_keys(
        raw,
        {"schema_version", "calibration"},
        label="selector calibration config",
    )
    if _integer(raw["schema_version"], label="schema_version") != 1:
        raise ValidationError("unsupported selector calibration config schema")
    return freeze_mapping(_mapping(raw["calibration"], label="calibration"))


def _guardrail_base_selector(
    discovery: SelectorDiscoveryFreeze,
    base_bundle: SelectorArtifactBundle | None,
) -> UtilityGainProvider:
    if not isinstance(base_bundle, SelectorArtifactBundle):
        raise ValidationError(
            "cost_risk_guardrail requires a SelectorArtifactBundle base, not a naked selector"
        )
    _validate_discovery_binding(base_bundle, discovery)
    _validate_base_discovery_lineage(base_bundle, discovery)
    base_selector = base_bundle.selector()
    if not isinstance(base_selector, UtilityGainProvider):
        raise ValidationError("cost_risk_guardrail base bundle must contain a UtilityGainProvider")
    return base_selector


def _guardrail_fit_lineage(
    discovery: SelectorDiscoveryFreeze,
    base_bundle: SelectorArtifactBundle,
    *,
    fit_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    _validate_discovery_binding(base_bundle, discovery)
    return freeze_mapping(
        {
            "schema_version": 1,
            "kind": "cost_risk_guardrail_base_bundle",
            "discovery": {
                "revision": discovery.revision,
                "sha256": discovery.discovery_hash,
            },
            "fit_config_sha256": content_hash(_mapping(fit_config, label="fit config")),
            "base_selector_bundle": {
                "selector_id": base_bundle.selector_card_id,
                "revision": base_bundle.revision,
                "sha256": base_bundle.bundle_hash,
                "stage": base_bundle.stage.value,
                "model_hash": base_bundle.model_hash,
            },
        }
    )


def _selector_fit_lineage(
    discovery: SelectorDiscoveryFreeze,
    *,
    fit_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    return freeze_mapping(
        {
            "schema_version": 1,
            "kind": "selector_fit",
            "discovery": {
                "revision": discovery.revision,
                "sha256": discovery.discovery_hash,
            },
            "fit_config_sha256": content_hash(_mapping(fit_config, label="fit config")),
        }
    )


def _validate_base_discovery_lineage(
    base_bundle: SelectorArtifactBundle,
    discovery: SelectorDiscoveryFreeze,
) -> None:
    lineage = base_bundle.fit_lineage
    if not lineage:
        raise ValidationError("cost_risk_guardrail base bundle lacks exact discovery fit lineage")
    source = lineage.get("discovery")
    if not isinstance(source, Mapping):
        raise ValidationError("cost_risk_guardrail base bundle has invalid discovery fit lineage")
    if (
        source.get("revision") != discovery.revision
        or source.get("sha256") != discovery.discovery_hash
    ):
        raise ValidationError(
            "cost/risk base selector bundle was fitted from an unrelated SelectorDiscoveryFreeze"
        )


def _validate_discovery_binding(
    bundle: SelectorArtifactBundle,
    discovery: SelectorDiscoveryFreeze,
) -> None:
    if not isinstance(discovery, SelectorDiscoveryFreeze):
        raise TypeError("selector calibration requires SelectorDiscoveryFreeze")
    expected = {
        "extractor_id": discovery.extractor_id,
        "extractor_revision": discovery.extractor_revision,
        "extractor_hash": discovery.extractor_hash,
        "feature_schema_id": discovery.feature_schema_id,
        "feature_schema_hash": discovery.feature_schema_hash,
        "representative_by_family": discovery.representative_by_family,
        "training_table_sha256": discovery.training_table.table_hash,
        "split_provenance_hash": discovery.split_provenance_hash,
        "comparison_profile_hash": discovery.comparison_profile_hash,
    }
    if any(getattr(bundle, name) != value for name, value in expected.items()):
        raise ValidationError("fitted selector bundle differs from the supplied discovery freeze")
    if "calibration" not in discovery.training_table.group_split_digests:
        raise ValidationError("discovery freeze contains no calibration split")


def _examples_for_split(
    discovery: SelectorDiscoveryFreeze,
    split: str,
) -> tuple[SelectorExample, ...]:
    rows = tuple(item for item in discovery.training_table.examples if item.split == split)
    if not rows:
        raise ValidationError(f"selector discovery freeze has no {split!r} examples")
    return rows


def _calibrate_rule(
    selector: RuleSelector,
    examples: Sequence[SelectorExample],
    *,
    threshold_grid: Sequence[float],
    margin_grid: Sequence[float],
    max_harm_rate: float,
) -> SelectorCalibrationResult:
    candidates: list[tuple[PolicyEvaluation, RuleSelector]] = []
    considered = 0
    for threshold in threshold_grid:
        for margin in margin_grid:
            considered += 1
            candidate = replace(
                selector,
                threshold=float(threshold),
                margin=float(margin),
            )
            evaluation = evaluate_selector(
                candidate,
                examples,
                required_split="calibration",
            )
            if evaluation.harm_rate <= max_harm_rate:
                candidates.append((evaluation, candidate))
    if not candidates:
        raise ValidationError("no rule calibration candidate satisfies the harm constraint")
    evaluation, selected = max(
        candidates,
        key=lambda item: (
            item[0].mean_policy_value,
            item[0].coverage,
            -item[0].harm_rate,
            -item[1].threshold,
            -item[1].margin,
        ),
    )
    return SelectorCalibrationResult(selected, evaluation, considered)


def _from_advanced_calibration(
    result: SelectorCalibrationResult,
) -> SelectorCalibrationResult:
    if not isinstance(result, SelectorCalibrationResult):  # pragma: no cover
        raise TypeError("calibration API returned an invalid result")
    return result


def _evaluation_to_artifact(value: PolicyEvaluation) -> dict[str, Any]:
    return {
        "count": value.count,
        "coverage": value.coverage,
        "mean_policy_value": value.mean_policy_value,
        "mean_no_op_value": value.mean_no_op_value,
        "mean_gain": value.mean_gain,
        "harm_rate": value.harm_rate,
        "oracle_regret": value.oracle_regret,
    }


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSON: {source}") from exc
    return _mapping(raw, label=label)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValidationError(
            f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}"
        )


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object with string keys")
    return dict(value)


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{label} must be an array")
    return list(value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    result = _integer(value, label=label)
    if result < 1:
        raise ValidationError(f"{label} must be positive")
    return result


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{label} must be a finite number")
    return float(value)


def _positive(value: object, *, label: str) -> float:
    result = _number(value, label=label)
    if result <= 0:
        raise ValidationError(f"{label} must be positive")
    return result


def _nonnegative(value: object, *, label: str) -> float:
    result = _number(value, label=label)
    if result < 0:
        raise ValidationError(f"{label} must be non-negative")
    return result


def _bounded(value: object, *, label: str) -> float:
    result = _number(value, label=label)
    if not 0 <= result <= 1:
        raise ValidationError(f"{label} must be in [0, 1]")
    return result


def _number_array(
    value: object,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, ...]:
    raw = _array(value, label=label)
    if not raw:
        raise ValidationError(f"{label} must not be empty")
    numbers = tuple(_number(item, label=f"{label} item") for item in raw)
    if lower is not None and any(item < lower for item in numbers):
        raise ValidationError(f"{label} values must be at least {lower:g}")
    if upper is not None and any(item > upper for item in numbers):
        raise ValidationError(f"{label} values must be at most {upper:g}")
    return tuple(sorted(set(numbers)))


__all__ = [
    "SelectorCalibrationSummary",
    "calibrate_selector_bundle",
    "create_fitted_selector_bundle",
    "fit_selector",
    "freeze_selector_bundle",
    "load_selector_calibration_config",
    "load_selector_fit_config",
]
