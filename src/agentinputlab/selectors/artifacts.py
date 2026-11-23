"""Safe JSON codecs and append-only storage for frozen selector policies."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily, StrEnum
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, freeze_mapping

from .advanced import (
    CostRiskAwareSelector,
    GBDTTLearnerSelector,
    KNNTrainingRow,
    KNNUtilitySelector,
    LogisticTLearnerSelector,
    RidgeTLearnerSelector,
)
from .base import Selector
from .catalog import get_selector_card
from .learned import RidgeUtilitySelector
from .rules import RuleSelector
from .runtime import SelectorArtifactIdentity

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CORE_FAMILIES = frozenset(
    {
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)
_SELECTOR_KIND_BY_TYPE = {
    RuleSelector: "rule",
    RidgeUtilitySelector: "ridge_utility",
    RidgeTLearnerSelector: "ridge_t_learner",
    LogisticTLearnerSelector: "logistic_t_learner",
    GBDTTLearnerSelector: "gbdt_t_learner",
    KNNUtilitySelector: "knn_utility",
    CostRiskAwareSelector: "cost_risk_guardrail",
}
_CARD_BY_KIND = {
    "rule": "rule.v1",
    "ridge_utility": "ridge_utility.v1",
    "ridge_t_learner": "ridge_t_learner.v1",
    "logistic_t_learner": "logistic_t_learner.v1",
    "gbdt_t_learner": "gbdt_t_learner.v1",
    "knn_utility": "knn_utility.v1",
    "cost_risk_guardrail": "cost_risk_guardrail.v1",
}


class SelectorArtifactStage(StrEnum):
    FITTED = "fitted"
    CALIBRATED = "calibrated"
    FROZEN = "frozen"


def selector_model_hash(selector: Selector) -> str:
    declared = getattr(selector, "model_hash", None)
    if not isinstance(declared, str) or not _SHA256.fullmatch(declared):
        raise ValidationError("selector must expose a lowercase SHA-256 model_hash")
    return declared


def selector_card_id(selector: Selector) -> str:
    for selector_type, kind in _SELECTOR_KIND_BY_TYPE.items():
        if type(selector) is selector_type:
            return _CARD_BY_KIND[kind]
    raise ValidationError(f"unsupported selector artifact type: {type(selector).__name__}")


def selector_to_artifact(selector: Selector) -> dict[str, Any]:
    """Encode one built-in selector without pickle or executable objects."""

    kind = _kind_for_selector(selector)
    payload: dict[str, Any]
    if isinstance(selector, RuleSelector):
        payload = {
            "threshold": selector.threshold,
            "margin": selector.margin,
        }
    elif isinstance(selector, RidgeUtilitySelector):
        payload = {
            "coefficients": _encode_family_vectors(selector.coefficients),
            "release_by_family": _encode_releases(selector.release_by_family),
            "min_gain": selector.min_gain,
            "margin": selector.margin,
        }
    elif isinstance(selector, RidgeTLearnerSelector):
        payload = {
            "outcome_coefficients": _encode_family_vectors(selector.outcome_coefficients),
            "release_by_family": _encode_releases(selector.release_by_family),
            "ridge": selector.ridge,
            "min_gain": selector.min_gain,
            "margin": selector.margin,
        }
    elif isinstance(selector, LogisticTLearnerSelector):
        payload = {
            "outcome_coefficients": _encode_family_vectors(selector.outcome_coefficients),
            "release_by_family": _encode_releases(selector.release_by_family),
            "l2": selector.l2,
            "learning_rate": selector.learning_rate,
            "iterations": selector.iterations,
            "min_gain": selector.min_gain,
            "margin": selector.margin,
        }
    elif isinstance(selector, GBDTTLearnerSelector):
        payload = selector.to_artifact()
    elif isinstance(selector, KNNUtilitySelector):
        payload = {
            "training_rows": [
                {
                    "feature_vector": list(row.feature_vector),
                    "gains": {
                        family.value: value
                        for family, value in sorted(
                            row.gains.items(),
                            key=lambda item: item[0].value,
                        )
                    },
                }
                for row in selector.training_rows
            ],
            "release_by_family": _encode_releases(selector.release_by_family),
            "k": selector.k,
            "distance_power": selector.distance_power,
            "min_gain": selector.min_gain,
            "margin": selector.margin,
        }
    elif isinstance(selector, CostRiskAwareSelector):
        payload = {
            "base_selector": selector_to_artifact(selector.base_selector),
            "release_by_family": _encode_releases(selector.release_by_family),
            "cost_by_family": _encode_family_scalars(selector.cost_by_family),
            "risk_by_family": _encode_family_scalars(selector.risk_by_family),
            "max_cost": selector.max_cost,
            "max_risk": selector.max_risk,
            "cost_weight": selector.cost_weight,
            "risk_weight": selector.risk_weight,
            "min_gain": selector.min_gain,
            "margin": selector.margin,
        }
    else:  # pragma: no cover - guarded by _kind_for_selector
        raise ValidationError(f"unsupported selector artifact type: {type(selector).__name__}")
    return {
        "schema_version": 1,
        "selector_kind": kind,
        "payload": payload,
    }


def selector_from_artifact(value: Mapping[str, Any]) -> Selector:
    """Strictly decode a built-in selector from canonical JSON-shaped data."""

    artifact = _strict_mapping(
        value,
        expected={"schema_version", "selector_kind", "payload"},
        label="selector artifact",
    )
    if _integer(artifact["schema_version"], label="schema_version") != 1:
        raise ValidationError("unsupported selector artifact schema")
    kind = _text(artifact["selector_kind"], label="selector_kind")
    if kind not in _CARD_BY_KIND:
        raise ValidationError(f"unsupported selector artifact kind: {kind}")
    payload_value = artifact["payload"]
    if not isinstance(payload_value, Mapping):
        raise ValidationError("selector artifact payload must be an object")

    try:
        if kind == "rule":
            payload = _strict_mapping(
                payload_value,
                expected={"threshold", "margin"},
                label="rule selector payload",
            )
            selector: Selector = RuleSelector(
                threshold=_number(payload["threshold"], label="threshold"),
                margin=_number(payload["margin"], label="margin"),
            )
        elif kind == "ridge_utility":
            payload = _strict_mapping(
                payload_value,
                expected={
                    "coefficients",
                    "release_by_family",
                    "min_gain",
                    "margin",
                },
                label="ridge utility payload",
            )
            selector = RidgeUtilitySelector(
                _decode_family_vectors(payload["coefficients"], include_no_op=False),
                _decode_releases(payload["release_by_family"]),
                min_gain=_number(payload["min_gain"], label="min_gain"),
                margin=_number(payload["margin"], label="margin"),
            )
        elif kind == "ridge_t_learner":
            payload = _strict_mapping(
                payload_value,
                expected={
                    "outcome_coefficients",
                    "release_by_family",
                    "ridge",
                    "min_gain",
                    "margin",
                },
                label="ridge T-learner payload",
            )
            selector = RidgeTLearnerSelector(
                _decode_family_vectors(
                    payload["outcome_coefficients"],
                    include_no_op=True,
                ),
                _decode_releases(payload["release_by_family"]),
                ridge=_number(payload["ridge"], label="ridge"),
                min_gain=_number(payload["min_gain"], label="min_gain"),
                margin=_number(payload["margin"], label="margin"),
            )
        elif kind == "logistic_t_learner":
            payload = _strict_mapping(
                payload_value,
                expected={
                    "outcome_coefficients",
                    "release_by_family",
                    "l2",
                    "learning_rate",
                    "iterations",
                    "min_gain",
                    "margin",
                },
                label="logistic T-learner payload",
            )
            selector = LogisticTLearnerSelector(
                _decode_family_vectors(
                    payload["outcome_coefficients"],
                    include_no_op=True,
                ),
                _decode_releases(payload["release_by_family"]),
                l2=_number(payload["l2"], label="l2"),
                learning_rate=_number(
                    payload["learning_rate"],
                    label="learning_rate",
                ),
                iterations=_integer(payload["iterations"], label="iterations"),
                min_gain=_number(payload["min_gain"], label="min_gain"),
                margin=_number(payload["margin"], label="margin"),
            )
        elif kind == "gbdt_t_learner":
            selector = GBDTTLearnerSelector.from_artifact(payload_value)
        elif kind == "knn_utility":
            payload = _strict_mapping(
                payload_value,
                expected={
                    "training_rows",
                    "release_by_family",
                    "k",
                    "distance_power",
                    "min_gain",
                    "margin",
                },
                label="KNN selector payload",
            )
            raw_rows = _array(payload["training_rows"], label="training_rows")
            rows: list[KNNTrainingRow] = []
            for raw_row in raw_rows:
                row = _strict_mapping(
                    raw_row,
                    expected={"feature_vector", "gains"},
                    label="KNN training row",
                )
                rows.append(
                    KNNTrainingRow(
                        tuple(
                            _number(item, label="feature_vector item")
                            for item in _array(
                                row["feature_vector"],
                                label="feature_vector",
                            )
                        ),
                        _decode_family_scalars(
                            row["gains"],
                            include_no_op=False,
                            label="KNN gains",
                        ),
                    )
                )
            selector = KNNUtilitySelector(
                tuple(rows),
                _decode_releases(payload["release_by_family"]),
                k=_integer(payload["k"], label="k"),
                distance_power=_number(
                    payload["distance_power"],
                    label="distance_power",
                ),
                min_gain=_number(payload["min_gain"], label="min_gain"),
                margin=_number(payload["margin"], label="margin"),
            )
        else:
            payload = _strict_mapping(
                payload_value,
                expected={
                    "base_selector",
                    "release_by_family",
                    "cost_by_family",
                    "risk_by_family",
                    "max_cost",
                    "max_risk",
                    "cost_weight",
                    "risk_weight",
                    "min_gain",
                    "margin",
                },
                label="cost/risk selector payload",
            )
            raw_base = payload["base_selector"]
            if not isinstance(raw_base, Mapping):
                raise ValidationError("base_selector must be an object")
            selector = CostRiskAwareSelector(
                selector_from_artifact(raw_base),
                _decode_releases(payload["release_by_family"]),
                _decode_family_scalars(
                    payload["cost_by_family"],
                    include_no_op=False,
                    label="cost_by_family",
                ),
                _decode_family_scalars(
                    payload["risk_by_family"],
                    include_no_op=False,
                    label="risk_by_family",
                ),
                max_cost=_number(payload["max_cost"], label="max_cost"),
                max_risk=_number(payload["max_risk"], label="max_risk"),
                cost_weight=_number(payload["cost_weight"], label="cost_weight"),
                risk_weight=_number(payload["risk_weight"], label="risk_weight"),
                min_gain=_number(payload["min_gain"], label="min_gain"),
                margin=_number(payload["margin"], label="margin"),
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"invalid {kind} selector artifact") from exc

    if selector_card_id(selector) != _CARD_BY_KIND[kind]:
        raise ValidationError("decoded selector type differs from its artifact kind")
    return selector


@dataclass(frozen=True, slots=True)
class SelectorArtifactBundle:
    """One content-addressed selector plus all inference-critical bindings."""

    selector_card_id: str
    stage: SelectorArtifactStage
    revision: str
    model_payload: Mapping[str, Any]
    model_hash: str
    extractor_id: str
    extractor_revision: str
    extractor_hash: str
    feature_schema_id: str
    feature_schema_hash: str
    representative_by_family: Mapping[SemanticFamily, str]
    training_table_sha256: str
    split_provenance_hash: str
    comparison_profile_hash: str
    calibration_receipt: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    fit_lineage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValidationError("unsupported selector bundle schema")
        try:
            get_selector_card(self.selector_card_id)
        except KeyError as exc:
            raise ValidationError("selector bundle references an unknown selector card") from exc
        if not isinstance(self.stage, SelectorArtifactStage):
            raise ValidationError("selector bundle stage must be SelectorArtifactStage")
        if (
            not isinstance(self.revision, str)
            or not _SAFE_SEGMENT.fullmatch(self.revision)
            or ".." in self.revision
        ):
            raise ValidationError("selector bundle revision must be a safe flat identifier")
        for label in (
            "extractor_id",
            "extractor_revision",
            "feature_schema_id",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"selector bundle {label} must be non-empty")
        for label in (
            "model_hash",
            "extractor_hash",
            "feature_schema_hash",
            "training_table_sha256",
            "split_provenance_hash",
            "comparison_profile_hash",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValidationError(f"selector bundle {label} must be SHA-256")
        try:
            normalized_payload = json.loads(canonical_json(self.model_payload))
            normalized_receipt = json.loads(canonical_json(self.calibration_receipt))
            normalized_lineage = json.loads(canonical_json(self.fit_lineage))
            frozen_payload = freeze_mapping(normalized_payload)
            receipt = freeze_mapping(normalized_receipt)
            lineage = freeze_mapping(normalized_lineage)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationError("selector bundle values must be canonical JSON") from exc
        representatives = _validated_representatives(self.representative_by_family)
        selector = selector_from_artifact(normalized_payload)
        if selector_card_id(selector) != self.selector_card_id:
            raise ValidationError("selector bundle card and model payload differ")
        if selector_model_hash(selector) != self.model_hash:
            raise ValidationError("selector bundle model hash does not match its payload")
        declared_releases = getattr(selector, "release_by_family", None)
        if isinstance(declared_releases, Mapping) and dict(declared_releases) != representatives:
            raise ValidationError(
                "selector bundle representatives differ from model release mapping"
            )
        if self.stage is SelectorArtifactStage.FITTED and receipt:
            raise ValidationError("fitted selector bundles cannot contain calibration receipts")
        if self.stage is not SelectorArtifactStage.FITTED and not receipt:
            raise ValidationError(
                "calibrated/frozen selector bundles require a calibration receipt"
            )
        _validate_fit_lineage(
            lineage,
            selector=selector,
            selector_card=self.selector_card_id,
        )
        object.__setattr__(self, "model_payload", frozen_payload)
        object.__setattr__(self, "calibration_receipt", receipt)
        object.__setattr__(self, "fit_lineage", lineage)
        object.__setattr__(
            self,
            "representative_by_family",
            MappingProxyType(representatives),
        )

    @classmethod
    def create(
        cls,
        selector: Selector,
        *,
        stage: SelectorArtifactStage,
        revision: str,
        extractor_id: str,
        extractor_revision: str,
        extractor_hash: str,
        feature_schema_id: str,
        feature_schema_hash: str,
        representative_by_family: Mapping[SemanticFamily, str],
        training_table_sha256: str,
        split_provenance_hash: str,
        comparison_profile_hash: str,
        calibration_receipt: Mapping[str, Any] | None = None,
        fit_lineage: Mapping[str, Any] | None = None,
    ) -> SelectorArtifactBundle:
        return cls(
            selector_card_id=selector_card_id(selector),
            stage=stage,
            revision=revision,
            model_payload=selector_to_artifact(selector),
            model_hash=selector_model_hash(selector),
            extractor_id=extractor_id,
            extractor_revision=extractor_revision,
            extractor_hash=extractor_hash,
            feature_schema_id=feature_schema_id,
            feature_schema_hash=feature_schema_hash,
            representative_by_family=representative_by_family,
            training_table_sha256=training_table_sha256,
            split_provenance_hash=split_provenance_hash,
            comparison_profile_hash=comparison_profile_hash,
            calibration_receipt=calibration_receipt or {},
            fit_lineage=fit_lineage or {},
        )

    @property
    def bundle_hash(self) -> str:
        return content_hash(self.to_artifact())

    def selector(self) -> Selector:
        normalized = json.loads(canonical_json(self.model_payload))
        return selector_from_artifact(normalized)

    def to_artifact(self) -> dict[str, Any]:
        artifact = {
            "schema_version": self.schema_version,
            "selector_card_id": self.selector_card_id,
            "stage": self.stage.value,
            "revision": self.revision,
            "model_payload": self.model_payload,
            "model_hash": self.model_hash,
            "extractor": {
                "id": self.extractor_id,
                "revision": self.extractor_revision,
                "sha256": self.extractor_hash,
            },
            "feature_schema": {
                "id": self.feature_schema_id,
                "sha256": self.feature_schema_hash,
            },
            "representative_by_family": _encode_releases(self.representative_by_family),
            "training_table_sha256": self.training_table_sha256,
            "split_provenance_hash": self.split_provenance_hash,
            "comparison_profile_hash": self.comparison_profile_hash,
            "calibration_receipt": self.calibration_receipt,
        }
        if self.fit_lineage:
            artifact["fit_lineage"] = self.fit_lineage
        return artifact

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> SelectorArtifactBundle:
        artifact = _string_mapping(value, label="selector bundle")
        required_fields = {
            "schema_version",
            "selector_card_id",
            "stage",
            "revision",
            "model_payload",
            "model_hash",
            "extractor",
            "feature_schema",
            "representative_by_family",
            "training_table_sha256",
            "split_provenance_hash",
            "comparison_profile_hash",
            "calibration_receipt",
        }
        if set(artifact) not in (required_fields, required_fields | {"fit_lineage"}):
            raise ValidationError(
                "selector bundle fields differ: expected the legacy fields with "
                "optional fit_lineage"
            )
        extractor = _strict_mapping(
            artifact["extractor"],
            expected={"id", "revision", "sha256"},
            label="selector bundle extractor",
        )
        schema = _strict_mapping(
            artifact["feature_schema"],
            expected={"id", "sha256"},
            label="selector bundle feature schema",
        )
        model_payload = artifact["model_payload"]
        calibration_receipt = artifact["calibration_receipt"]
        fit_lineage = artifact.get("fit_lineage", {})
        if (
            not isinstance(model_payload, Mapping)
            or not isinstance(calibration_receipt, Mapping)
            or not isinstance(fit_lineage, Mapping)
        ):
            raise ValidationError("selector bundle model/receipt values must be objects")
        try:
            stage = SelectorArtifactStage(_text(artifact["stage"], label="stage"))
        except ValueError as exc:
            raise ValidationError("selector bundle stage is unsupported") from exc
        return cls(
            schema_version=_integer(
                artifact["schema_version"],
                label="schema_version",
            ),
            selector_card_id=_text(
                artifact["selector_card_id"],
                label="selector_card_id",
            ),
            stage=stage,
            revision=_text(artifact["revision"], label="revision"),
            model_payload=model_payload,
            model_hash=_text(artifact["model_hash"], label="model_hash"),
            extractor_id=_text(extractor["id"], label="extractor.id"),
            extractor_revision=_text(
                extractor["revision"],
                label="extractor.revision",
            ),
            extractor_hash=_text(extractor["sha256"], label="extractor.sha256"),
            feature_schema_id=_text(schema["id"], label="feature_schema.id"),
            feature_schema_hash=_text(
                schema["sha256"],
                label="feature_schema.sha256",
            ),
            representative_by_family=_decode_releases(artifact["representative_by_family"]),
            training_table_sha256=_text(
                artifact["training_table_sha256"],
                label="training_table_sha256",
            ),
            split_provenance_hash=_text(
                artifact["split_provenance_hash"],
                label="split_provenance_hash",
            ),
            comparison_profile_hash=_text(
                artifact["comparison_profile_hash"],
                label="comparison_profile_hash",
            ),
            calibration_receipt=calibration_receipt,
            fit_lineage=fit_lineage,
        )


def _validate_fit_lineage(
    value: Mapping[str, Any],
    *,
    selector: Selector,
    selector_card: str,
) -> None:
    if not value:
        return
    lineage = _string_mapping(value, label="selector fit_lineage")
    common_fields = {
        "schema_version",
        "kind",
        "discovery",
        "fit_config_sha256",
    }
    kind = _text(lineage.get("kind"), label="fit_lineage.kind")
    if kind == "selector_fit":
        if set(lineage) != common_fields:
            raise ValidationError("selector_fit lineage fields differ")
        if selector_card == _CARD_BY_KIND["cost_risk_guardrail"]:
            raise ValidationError("cost/risk guardrail lineage must bind its exact base bundle")
    elif kind == "cost_risk_guardrail_base_bundle":
        if set(lineage) != common_fields | {"base_selector_bundle"}:
            raise ValidationError("cost/risk selector fit_lineage fields differ")
        if selector_card != _CARD_BY_KIND["cost_risk_guardrail"] or not isinstance(
            selector,
            CostRiskAwareSelector,
        ):
            raise ValidationError("guardrail base lineage is valid only for a cost/risk selector")
    else:
        raise ValidationError("selector fit_lineage kind is unsupported")
    if _integer(lineage["schema_version"], label="fit_lineage.schema_version") != 1:
        raise ValidationError("unsupported selector fit_lineage schema")
    discovery = _strict_mapping(
        lineage["discovery"],
        expected={"revision", "sha256"},
        label="selector fit_lineage discovery",
    )
    for label, value_to_check in (
        ("fit_lineage.fit_config_sha256", lineage["fit_config_sha256"]),
        ("fit_lineage.discovery.sha256", discovery["sha256"]),
    ):
        if not isinstance(value_to_check, str) or not _SHA256.fullmatch(value_to_check):
            raise ValidationError(f"{label} must be SHA-256")
    value_to_check = discovery["revision"]
    if (
        not isinstance(value_to_check, str)
        or not _SAFE_SEGMENT.fullmatch(value_to_check)
        or ".." in value_to_check
    ):
        raise ValidationError("fit_lineage.discovery.revision must be a safe flat identifier")
    if kind == "selector_fit":
        return

    base = _strict_mapping(
        lineage["base_selector_bundle"],
        expected={
            "selector_id",
            "revision",
            "sha256",
            "stage",
            "model_hash",
        },
        label="selector fit_lineage base_selector_bundle",
    )
    for label, value_to_check in (
        ("fit_lineage.base_selector_bundle.sha256", base["sha256"]),
        ("fit_lineage.base_selector_bundle.model_hash", base["model_hash"]),
    ):
        if not isinstance(value_to_check, str) or not _SHA256.fullmatch(value_to_check):
            raise ValidationError(f"{label} must be SHA-256")
    for label, value_to_check in (
        ("fit_lineage.base_selector_bundle.selector_id", base["selector_id"]),
        ("fit_lineage.base_selector_bundle.revision", base["revision"]),
    ):
        if (
            not isinstance(value_to_check, str)
            or not _SAFE_SEGMENT.fullmatch(value_to_check)
            or ".." in value_to_check
        ):
            raise ValidationError(f"{label} must be a safe flat identifier")
    try:
        base_stage = SelectorArtifactStage(
            _text(base["stage"], label="fit_lineage.base_selector_bundle.stage")
        )
    except ValueError as exc:
        raise ValidationError("fit_lineage base selector stage is unsupported") from exc
    if base_stage not in {
        SelectorArtifactStage.FITTED,
        SelectorArtifactStage.CALIBRATED,
        SelectorArtifactStage.FROZEN,
    }:  # pragma: no cover - exhaustive enum guard
        raise ValidationError("fit_lineage base selector stage is unsupported")
    embedded_base = selector.base_selector
    if base["selector_id"] != selector_card_id(embedded_base):
        raise ValidationError("fit_lineage base selector card differs from the embedded base model")
    if base["model_hash"] != selector_model_hash(embedded_base):
        raise ValidationError("fit_lineage base model hash differs from the embedded base model")


@dataclass(slots=True)
class SelectorArtifactStore:
    """Content-addressed selector storage without mutable aliases or ``latest``."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, identity: SelectorArtifactIdentity) -> Path:
        for label, value in (
            ("selector_id", identity.selector_id),
            ("revision", identity.revision),
        ):
            if not _SAFE_SEGMENT.fullmatch(value) or ".." in value:
                raise ArtifactError(f"selector artifact {label} is not a safe path segment")
        candidate = (
            self.root / identity.selector_id / identity.revision / identity.sha256 / "bundle.json"
        ).resolve()
        if self.root not in candidate.parents:
            raise ArtifactError("selector artifact path escaped store root")
        return candidate

    def save(
        self,
        bundle: SelectorArtifactBundle,
        *,
        previous_identity: SelectorArtifactIdentity | None = None,
    ) -> SelectorArtifactIdentity:
        if not isinstance(bundle, SelectorArtifactBundle):
            raise TypeError("selector artifact store requires SelectorArtifactBundle")
        self._validate_transition(bundle, previous_identity)
        identity = SelectorArtifactIdentity(
            selector_id=bundle.selector_card_id,
            revision=bundle.revision,
            sha256=bundle.bundle_hash,
        )
        destination = self.path_for(identity)
        revision_root = destination.parent.parent
        if revision_root.exists():
            existing_hashes = {
                path.name
                for path in revision_root.iterdir()
                if path.is_dir() and _SHA256.fullmatch(path.name)
            }
            if existing_hashes - {identity.sha256}:
                raise ArtifactError(
                    "selector artifact revision is append-only and already has different content"
                )
        payload = (canonical_json(bundle.to_artifact()) + "\n").encode("utf-8")
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ArtifactError("stored selector artifact bytes differ from bundle identity")
            self.load(identity)
            return identity
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".partial-",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, destination)
            except FileExistsError:
                if destination.read_bytes() != payload:
                    raise ArtifactError(
                        "concurrent selector artifact materialization differs"
                    ) from None
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        self.load(identity)
        return identity

    def load(
        self,
        identity: SelectorArtifactIdentity,
        *,
        require_stage: SelectorArtifactStage | None = None,
    ) -> tuple[Selector, SelectorArtifactBundle]:
        if not isinstance(identity, SelectorArtifactIdentity):
            raise TypeError("selector artifact load requires SelectorArtifactIdentity")
        if require_stage is not None and not isinstance(require_stage, SelectorArtifactStage):
            raise TypeError("require_stage must be SelectorArtifactStage or None")
        path = self.path_for(identity)
        if not path.is_file():
            raise ArtifactError(
                f"selector artifact missing: {identity.selector_id}@{identity.revision}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("selector artifact is not valid UTF-8 JSON") from exc
        if not isinstance(raw, Mapping):
            raise ArtifactError("selector artifact root must be an object")
        try:
            bundle = SelectorArtifactBundle.from_artifact(raw)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("selector artifact bundle failed validation") from exc
        if (
            bundle.selector_card_id != identity.selector_id
            or bundle.revision != identity.revision
            or bundle.bundle_hash != identity.sha256
        ):
            raise ArtifactError("selector artifact identity does not match stored content")
        if require_stage is not None and bundle.stage is not require_stage:
            raise ArtifactError(
                f"selector artifact stage is {bundle.stage.value}, "
                f"not required {require_stage.value}"
            )
        return bundle.selector(), bundle

    def _validate_transition(
        self,
        bundle: SelectorArtifactBundle,
        previous_identity: SelectorArtifactIdentity | None,
    ) -> None:
        if bundle.stage is SelectorArtifactStage.FITTED:
            if previous_identity is not None:
                raise ArtifactError("fitted selector artifact cannot have a predecessor")
            return
        if previous_identity is None:
            raise ArtifactError(
                f"{bundle.stage.value} selector artifact requires its exact predecessor"
            )
        _, previous = self.load(previous_identity)
        expected_previous = (
            SelectorArtifactStage.FITTED
            if bundle.stage is SelectorArtifactStage.CALIBRATED
            else SelectorArtifactStage.CALIBRATED
        )
        if previous.stage is not expected_previous:
            raise ArtifactError(
                f"selector transition requires {expected_previous.value} predecessor"
            )
        if previous.selector_card_id != bundle.selector_card_id:
            raise ArtifactError("selector transition cannot change selector card")
        if previous.revision == bundle.revision:
            raise ArtifactError("selector lifecycle stages require distinct revisions")
        binding_fields = (
            "extractor_id",
            "extractor_revision",
            "extractor_hash",
            "feature_schema_id",
            "feature_schema_hash",
            "representative_by_family",
            "training_table_sha256",
            "split_provenance_hash",
            "comparison_profile_hash",
            "fit_lineage",
        )
        if any(getattr(previous, name) != getattr(bundle, name) for name in binding_fields):
            raise ArtifactError("selector transition changed a frozen data/runtime binding")
        if bundle.stage is SelectorArtifactStage.CALIBRATED and _calibration_invariant_payload(
            previous.model_payload
        ) != _calibration_invariant_payload(bundle.model_payload):
            raise ArtifactError(
                "selector calibration changed fitted model parameters; only the "
                "predefined calibration/guardrail fields may change"
            )
        if bundle.stage is SelectorArtifactStage.FROZEN and (
            previous.model_hash != bundle.model_hash
            or previous.model_payload != bundle.model_payload
            or previous.calibration_receipt != bundle.calibration_receipt
        ):
            raise ArtifactError("freezing cannot change a calibrated selector policy")


def _calibration_invariant_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portion of a selector artifact that calibration cannot change."""

    normalized = json.loads(canonical_json(value))
    if not isinstance(normalized, dict):  # pragma: no cover - bundle validation invariant
        raise ArtifactError("selector model payload must be an object")
    kind = normalized.get("selector_kind")
    payload = normalized.get("payload")
    if not isinstance(kind, str) or not isinstance(payload, dict):
        raise ArtifactError("selector model payload has an invalid calibration schema")
    direct_fields = {
        "rule": {"threshold", "margin"},
        "ridge_utility": {"min_gain", "margin"},
        "ridge_t_learner": {"min_gain", "margin"},
        "logistic_t_learner": {"min_gain", "margin"},
        "knn_utility": {"min_gain", "margin"},
        "cost_risk_guardrail": {
            "max_cost",
            "max_risk",
            "min_gain",
            "margin",
        },
    }
    if kind == "gbdt_t_learner":
        abstention = payload.get("abstention")
        if not isinstance(abstention, dict):
            raise ArtifactError("GBDT selector payload lacks abstention fields")
        for field in ("min_gain", "margin"):
            abstention.pop(field, None)
        return normalized
    try:
        allowed = direct_fields[kind]
    except KeyError as exc:  # pragma: no cover - selector codec rejects unknown kinds
        raise ArtifactError(f"unsupported selector calibration kind: {kind}") from exc
    for field in allowed:
        payload.pop(field, None)
    return normalized


def _kind_for_selector(selector: Selector) -> str:
    try:
        return _SELECTOR_KIND_BY_TYPE[type(selector)]
    except KeyError as exc:
        raise ValidationError(
            f"unsupported selector artifact type: {type(selector).__name__}"
        ) from exc


def _encode_family_vectors(
    values: Mapping[SemanticFamily, tuple[float, ...]],
) -> dict[str, list[float]]:
    return {
        family.value: list(vector)
        for family, vector in sorted(values.items(), key=lambda item: item[0].value)
    }


def _encode_releases(values: Mapping[SemanticFamily, str]) -> dict[str, str]:
    return {
        family.value: release
        for family, release in sorted(values.items(), key=lambda item: item[0].value)
    }


def _encode_family_scalars(
    values: Mapping[SemanticFamily, float],
) -> dict[str, float]:
    return {
        family.value: value
        for family, value in sorted(values.items(), key=lambda item: item[0].value)
    }


def _decode_family_vectors(
    value: object,
    *,
    include_no_op: bool,
) -> dict[SemanticFamily, tuple[float, ...]]:
    mapping = _string_mapping(value, label="family vectors")
    return {
        _family(name, include_no_op=include_no_op): tuple(
            _number(item, label=f"{name} coefficient")
            for item in _array(vector, label=f"{name} coefficients")
        )
        for name, vector in mapping.items()
    }


def _decode_releases(value: object) -> dict[SemanticFamily, str]:
    mapping = _string_mapping(value, label="release_by_family")
    releases = {
        _family(name, include_no_op=False): _text(
            release,
            label=f"{name} release",
        )
        for name, release in mapping.items()
    }
    return _validated_representatives(releases)


def _decode_family_scalars(
    value: object,
    *,
    include_no_op: bool,
    label: str,
) -> dict[SemanticFamily, float]:
    mapping = _string_mapping(value, label=label)
    return {
        _family(name, include_no_op=include_no_op): _number(
            scalar,
            label=f"{label}.{name}",
        )
        for name, scalar in mapping.items()
    }


def _validated_representatives(
    values: Mapping[SemanticFamily, str],
) -> dict[SemanticFamily, str]:
    copied = dict(values)
    if any(
        family not in _CORE_FAMILIES
        or not isinstance(release, str)
        or not release.strip()
        or release != release.strip()
        for family, release in copied.items()
    ):
        raise ValidationError(
            "selector representatives require core families and trimmed release IDs"
        )
    return copied


def _family(value: str, *, include_no_op: bool) -> SemanticFamily:
    try:
        family = SemanticFamily(value)
    except ValueError as exc:
        raise ValidationError(f"unknown selector family: {value!r}") from exc
    allowed = _CORE_FAMILIES | ({SemanticFamily.NO_OP} if include_no_op else set())
    if family not in allowed:
        raise ValidationError(f"family {value!r} is not valid in this selector artifact")
    return family


def _strict_mapping(
    value: object,
    *,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    mapping = _string_mapping(value, label=label)
    if set(mapping) != expected:
        raise ValidationError(
            f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(mapping)!r}"
        )
    return mapping


def _string_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object with string keys")
    return dict(value)


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a JSON array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be a JSON integer")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{label} must be a finite JSON number")
    return float(value)


__all__ = [
    "SelectorArtifactBundle",
    "SelectorArtifactStage",
    "SelectorArtifactStore",
    "selector_card_id",
    "selector_from_artifact",
    "selector_model_hash",
    "selector_to_artifact",
]
