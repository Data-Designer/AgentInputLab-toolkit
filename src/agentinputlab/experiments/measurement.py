"""Blinded, outcome-free human measurement preparation and agreement checks."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import (
    content_hash,
    freeze_mapping,
    to_jsonable,
)

MEASUREMENT_ASSIGNMENT_PROTOCOL = "agentinputlab.measurement-assignment.v1"
MEASUREMENT_ANALYSIS_PROTOCOL = "agentinputlab.measurement-analysis.v1"
AUTOMATIC_HUMAN_VALIDATION_PROTOCOL = "agentinputlab.automatic-human-validation.v1"
MEASUREMENT_KINDS = ("contract_anchor", "defect_eligibility")
DEFAULT_MEASUREMENT_COUNTS = {
    "contract_anchor": 200,
    "defect_eligibility": 100,
}

_FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "answer",
        "arm",
        "arm_id",
        "gold",
        "gold_answer",
        "intervention",
        "outcome",
        "post_action",
        "reference_answer",
        "release_id",
        "reward",
        "score",
        "success",
        "task_success",
        "treatment",
        "valid_success",
    }
)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _validate_outcome_free(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValidationError(f"{path} keys must be non-empty strings")
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_VISIBLE_KEYS:
                raise ValidationError(f"{path}.{key} exposes a forbidden outcome/treatment field")
            _validate_outcome_free(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_outcome_free(item, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class MeasurementItem:
    item_id: str
    group_id: str
    kind: str
    payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.item_id, label="measurement item_id")
        _text(self.group_id, label="measurement group_id")
        if self.kind not in MEASUREMENT_KINDS:
            raise ValidationError("measurement kind must be contract_anchor or defect_eligibility")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValidationError("measurement payload must be a non-empty object")
        _validate_outcome_free(self.payload)
        try:
            payload = freeze_mapping(self.payload)
        except (TypeError, ValueError) as exc:
            raise ValidationError("measurement payload must be canonical JSON") from exc
        object.__setattr__(self, "payload", payload)

    @property
    def blind_id(self) -> str:
        return "blind_" + content_hash(
            {
                "item_id": self.item_id,
                "group_id": self.group_id,
                "kind": self.kind,
                "payload": self.payload,
            }
        )

    def blinded_record(self) -> dict[str, Any]:
        return {
            "blind_id": self.blind_id,
            "group_hash": "group_" + content_hash({"group_id": self.group_id}),
            "kind": self.kind,
            "payload": to_jsonable(self.payload),
        }


def _annotators(values: Sequence[str]) -> tuple[str, ...]:
    annotators = tuple(_text(value, label="annotator ID") for value in values)
    if len(annotators) < 3:
        raise ValidationError(
            "measurement requires at least three annotators for two-pass review and adjudication"
        )
    if len(annotators) != len(set(annotators)):
        raise ValidationError("annotator IDs must be unique")
    return annotators


def build_measurement_assignment(
    items: Sequence[MeasurementItem],
    *,
    annotator_ids: Sequence[str],
    seed: int,
    expected_counts: Mapping[str, int] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Create deterministic, balanced, double-blind first-pass packets."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationError("measurement seed must be a non-negative integer")
    if not isinstance(allow_partial, bool):
        raise ValidationError("allow_partial must be boolean")
    values = tuple(items)
    if not values or not all(isinstance(item, MeasurementItem) for item in values):
        raise ValidationError("measurement assignment requires typed items")
    if len({item.item_id for item in values}) != len(values):
        raise ValidationError("measurement item IDs must be unique")
    if len({item.blind_id for item in values}) != len(values):
        raise ValidationError("measurement blind IDs must be unique")
    annotators = _annotators(annotator_ids)
    counts = Counter(item.kind for item in values)
    expected = dict(DEFAULT_MEASUREMENT_COUNTS if expected_counts is None else expected_counts)
    if set(expected) != set(MEASUREMENT_KINDS) or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in expected.values()
    ):
        raise ValidationError("measurement expected_counts are invalid")
    if not allow_partial and dict(counts) != expected:
        raise ValidationError(
            f"measurement item counts differ from the frozen census: "
            f"observed={dict(counts)}, expected={expected}"
        )
    if allow_partial and any(counts.get(kind, 0) < 1 for kind in MEASUREMENT_KINDS):
        raise ValidationError("partial measurement smoke still requires both measurement kinds")

    packets: dict[str, list[dict[str, Any]]] = {annotator: [] for annotator in annotators}
    assignment_rows: list[dict[str, Any]] = []
    for kind in MEASUREMENT_KINDS:
        kind_items = [item for item in values if item.kind == kind]
        ordered = sorted(
            kind_items,
            key=lambda item: hashlib.sha256(f"{seed}:{kind}:{item.blind_id}".encode()).hexdigest(),
        )
        offset = int(
            hashlib.sha256(f"{seed}:{kind}:offset".encode()).hexdigest()[:8],
            16,
        ) % len(annotators)
        for position, item in enumerate(ordered):
            first = annotators[(offset + position) % len(annotators)]
            second = annotators[(offset + position + 1) % len(annotators)]
            record = item.blinded_record()
            assignment_rows.append(
                {
                    "blind_id": item.blind_id,
                    "kind": kind,
                    "assigned_annotators": [first, second],
                }
            )
            for annotator in (first, second):
                packets[annotator].append(
                    {
                        **record,
                        "annotator_id": annotator,
                        "annotation_stage": "first_pass",
                    }
                )
    for packet in packets.values():
        packet.sort(key=lambda row: (row["kind"], row["blind_id"]))
    assignment_rows.sort(key=lambda row: (row["kind"], row["blind_id"]))
    packet_payload = {annotator: rows for annotator, rows in sorted(packets.items())}
    manifest = {
        "schema_version": 1,
        "protocol": MEASUREMENT_ASSIGNMENT_PROTOCOL,
        "record_type": "MEASUREMENT_PLAN",
        "observed": False,
        "outcome_blinded": True,
        "seed": seed,
        "expected_counts": expected,
        "observed_counts": dict(sorted(counts.items())),
        "allow_partial": allow_partial,
        "annotator_ids": list(annotators),
        "double_annotation": True,
        "adjudication_policy": "third_annotator_required_on_any_field_disagreement",
        "assignments": assignment_rows,
        "packet_hashes": {
            annotator: content_hash(rows) for annotator, rows in packet_payload.items()
        },
    }
    manifest["assignment_hash"] = content_hash(manifest)
    return {
        "manifest": manifest,
        "packets": packet_payload,
    }


def _label_scalar(value: Any, *, path: str) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip() and value == value.strip():
        return value
    raise ValidationError(f"{path} must be a boolean, integer, or trimmed non-empty string")


def _assignment_manifest(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("measurement assignment manifest must be an object")
    required = {
        "schema_version",
        "protocol",
        "record_type",
        "observed",
        "outcome_blinded",
        "seed",
        "expected_counts",
        "observed_counts",
        "allow_partial",
        "annotator_ids",
        "double_annotation",
        "adjudication_policy",
        "assignments",
        "packet_hashes",
        "assignment_hash",
    }
    if set(value) != required:
        raise ValidationError("measurement assignment manifest fields are invalid")
    if (
        value["schema_version"] != 1
        or value["protocol"] != MEASUREMENT_ASSIGNMENT_PROTOCOL
        or value["record_type"] != "MEASUREMENT_PLAN"
        or value["observed"] is not False
        or value["outcome_blinded"] is not True
        or value["double_annotation"] is not True
    ):
        raise ValidationError("measurement assignment manifest identity is invalid")
    unhashed = dict(value)
    observed_hash = unhashed.pop("assignment_hash")
    if observed_hash != content_hash(unhashed):
        raise ValidationError("measurement assignment manifest hash mismatch")
    return freeze_mapping(value)


def analyze_measurement_annotations(
    assignment_manifest: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate assigned first-pass records and compute field-level agreement."""

    manifest = _assignment_manifest(assignment_manifest)
    assignments: dict[str, tuple[str, tuple[str, str]]] = {}
    for row in manifest["assignments"]:
        if not isinstance(row, Mapping) or set(row) != {
            "blind_id",
            "kind",
            "assigned_annotators",
        }:
            raise ValidationError("measurement assignment row is invalid")
        blind_id = _text(row["blind_id"], label="assignment blind_id")
        kind = row["kind"]
        raw_annotators = row["assigned_annotators"]
        if (
            kind not in MEASUREMENT_KINDS
            or not isinstance(raw_annotators, Sequence)
            or isinstance(raw_annotators, (str, bytes))
            or len(raw_annotators) != 2
        ):
            raise ValidationError("measurement assignment row values are invalid")
        assigned = tuple(_text(value, label="assigned annotator") for value in raw_annotators)
        if assigned[0] == assigned[1]:
            raise ValidationError("measurement assignment requires two distinct annotators")
        if blind_id in assignments:
            raise ValidationError("measurement assignment contains duplicate blind IDs")
        assignments[blind_id] = (kind, assigned)

    first_pass: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    adjudications: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for index, raw in enumerate(annotations):
        if not isinstance(raw, Mapping) or set(raw) != {
            "blind_id",
            "annotator_id",
            "annotation_stage",
            "labels",
        }:
            raise ValidationError(f"annotation[{index}] fields are invalid")
        blind_id = _text(raw["blind_id"], label=f"annotation[{index}].blind_id")
        annotator = _text(raw["annotator_id"], label=f"annotation[{index}].annotator_id")
        try:
            _kind, assigned = assignments[blind_id]
        except KeyError as exc:
            raise ValidationError("annotation references an unknown blind ID") from exc
        labels = raw["labels"]
        if not isinstance(labels, Mapping) or not labels:
            raise ValidationError("annotation labels must be a non-empty object")
        normalized = {
            _text(key, label="annotation label field"): _label_scalar(
                value,
                path=f"annotation[{index}].labels.{key}",
            )
            for key, value in labels.items()
        }
        stage = raw["annotation_stage"]
        if stage == "first_pass":
            if annotator not in assigned:
                raise ValidationError("first-pass annotation came from an unassigned annotator")
            if annotator in first_pass[blind_id]:
                raise ValidationError("duplicate first-pass annotation")
            first_pass[blind_id][annotator] = freeze_mapping(normalized)
        elif stage == "adjudication":
            if annotator in assigned:
                raise ValidationError("adjudication must come from a third annotator")
            if blind_id in adjudications:
                raise ValidationError("duplicate adjudication")
            adjudications[blind_id] = (annotator, freeze_mapping(normalized))
        else:
            raise ValidationError("annotation_stage must be first_pass or adjudication")

    missing = sorted(
        blind_id
        for blind_id, (_kind, assigned) in assignments.items()
        if set(first_pass.get(blind_id, {})) != set(assigned)
    )
    if missing:
        raise ValidationError(f"measurement annotations omit assigned first passes: {missing[:5]}")

    field_pairs: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
    disagreement_ids: list[str] = []
    final_labels: dict[str, Mapping[str, Any]] = {}
    for blind_id, (kind, assigned) in assignments.items():
        left = first_pass[blind_id][assigned[0]]
        right = first_pass[blind_id][assigned[1]]
        if set(left) != set(right):
            raise ValidationError("paired first-pass annotations use different label fields")
        disagreed = left != right
        if disagreed:
            disagreement_ids.append(blind_id)
            adjudication = adjudications.get(blind_id)
            if adjudication is not None:
                if set(adjudication[1]) != set(left):
                    raise ValidationError("adjudication label fields differ from first pass")
                final_labels[blind_id] = adjudication[1]
        else:
            if blind_id in adjudications:
                raise ValidationError("adjudication is forbidden when first-pass labels agree")
            final_labels[blind_id] = left
        for field_name in sorted(left):
            field_pairs[(kind, field_name)].append((left[field_name], right[field_name]))

    agreement: list[dict[str, Any]] = []
    for (kind, field_name), pairs in sorted(field_pairs.items()):
        observed = sum(left == right for left, right in pairs) / len(pairs)
        left_counts = Counter(left for left, _right in pairs)
        right_counts = Counter(right for _left, right in pairs)
        categories = set(left_counts) | set(right_counts)
        expected = sum(
            (left_counts[category] / len(pairs)) * (right_counts[category] / len(pairs))
            for category in categories
        )
        kappa = (
            1.0
            if math.isclose(expected, 1.0) and math.isclose(observed, 1.0)
            else (
                (observed - expected) / (1.0 - expected)
                if not math.isclose(expected, 1.0)
                else float("nan")
            )
        )
        if not math.isfinite(kappa):
            raise ValidationError("measurement agreement produced a non-finite kappa")
        pooled_counts = Counter(label for left, right in pairs for label in (left, right))
        pooled_total = 2 * len(pairs)
        expected_disagreement = (
            1.0
            - sum(count * (count - 1) for count in pooled_counts.values())
            / (pooled_total * (pooled_total - 1))
            if pooled_total > 1
            else 0.0
        )
        observed_disagreement = 1.0 - observed
        alpha = (
            1.0
            if math.isclose(expected_disagreement, 0.0) and math.isclose(observed_disagreement, 0.0)
            else (
                1.0 - observed_disagreement / expected_disagreement
                if not math.isclose(expected_disagreement, 0.0)
                else float("nan")
            )
        )
        if not math.isfinite(alpha):
            raise ValidationError("measurement agreement produced a non-finite Krippendorff alpha")
        agreement.append(
            {
                "kind": kind,
                "field": field_name,
                "n": len(pairs),
                "percent_agreement": observed,
                "cohen_kappa": kappa,
                "krippendorff_alpha_nominal": alpha,
            }
        )

    unresolved = sorted(set(disagreement_ids) - set(adjudications))
    result = {
        "schema_version": 1,
        "protocol": MEASUREMENT_ANALYSIS_PROTOCOL,
        "assignment_hash": manifest["assignment_hash"],
        "item_count": len(assignments),
        "first_pass_annotation_count": sum(len(rows) for rows in first_pass.values()),
        "disagreement_count": len(disagreement_ids),
        "adjudicated_count": len(adjudications),
        "unresolved_disagreement_count": len(unresolved),
        "unresolved_blind_ids": unresolved,
        "agreement": agreement,
        "final_label_count": len(final_labels),
        # Preserve the adjudicated item-level evidence.  Aggregate counts alone
        # cannot validate an automatic labeler or produce a confusion matrix.
        "final_labels": [
            {
                "blind_id": blind_id,
                "kind": assignments[blind_id][0],
                "labels": to_jsonable(labels),
            }
            for blind_id, labels in sorted(final_labels.items())
        ],
        "analysis_complete": not unresolved,
    }
    result["analysis_hash"] = content_hash(result)
    return result


def compare_automatic_human_audits(
    *,
    automatic_input_labels: Sequence[Mapping[str, Any]],
    input_audit_items: Sequence[Mapping[str, Any]],
    input_final_annotations: Sequence[Mapping[str, Any]],
    automatic_trajectory_labels: Sequence[Mapping[str, Any]],
    trajectory_audit_items: Sequence[Mapping[str, Any]],
    trajectory_final_annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare frozen automatic labels with final human adjudications.

    The function produces the six rows used by the paper's human-audit table,
    plus a full micro confusion matrix for every row.  ``uncertain`` human
    labels are retained in the denominator metadata but excluded from binary
    precision/recall rather than silently coerced.
    """

    input_auto = {
        f"defect:{row.get('benchmark_id')}:{row.get('checkpoint_id')}": row
        for row in automatic_input_labels
    }
    trajectory_auto = {
        f"trajectory:{row.get('slot_id')}": row for row in automatic_trajectory_labels
    }
    input_items = _audit_item_map(input_audit_items)
    trajectory_items = _audit_item_map(trajectory_audit_items)
    input_human = _annotation_map(input_final_annotations)
    trajectory_human = _annotation_map(trajectory_final_annotations)
    _require_exact_annotation_coverage(input_items, input_human, label="input audit")
    _require_exact_annotation_coverage(
        trajectory_items,
        trajectory_human,
        label="trajectory audit",
    )

    comparisons: dict[str, list[tuple[bool, bool | None]]] = defaultdict(list)
    defect_fields = (
        "unclear_or_misaligned_request",
        "decomposition_need",
        "missing_external_evidence",
        "clarification_need",
        "context_overload_or_noise",
    )
    eligibility_fields = tuple(
        f"{family}_eligible"
        for family in ("reformulate", "decompose", "acquire", "clarify", "curate")
    )
    for item_id in sorted(input_items):
        try:
            auto = input_auto[item_id]
        except KeyError as exc:
            raise ValidationError(f"input audit item has no automatic label: {item_id}") from exc
        defect_labels = auto.get("defect_labels")
        eligibility_labels = auto.get("eligibility_labels")
        if not isinstance(defect_labels, Mapping) or not isinstance(eligibility_labels, Mapping):
            raise ValidationError("automatic input label fields are invalid")
        labels = input_human[item_id]
        for field_name in defect_fields:
            comparisons["Defect labeling"].append(
                (bool(defect_labels.get(field_name)), _human_binary(labels.get(field_name)))
            )
        for field_name in eligibility_fields:
            comparisons["Eligibility"].append(
                (bool(eligibility_labels.get(field_name)), _human_binary(labels.get(field_name)))
            )

    trajectory_mapping = {
        "Contract preservation": "overall_contract_valid",
        "Entity preservation": "entity_preserved",
        "Numeric preservation": "numeric_preserved",
        "Constraint preservation": "constraint_preserved",
    }
    for item_id in sorted(trajectory_items):
        try:
            auto = trajectory_auto[item_id]
        except KeyError as exc:
            raise ValidationError(
                f"trajectory audit item has no automatic label: {item_id}"
            ) from exc
        checkpoints = auto.get("checkpoint_results")
        if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
            raise ValidationError("automatic trajectory checkpoint results are invalid")
        checks = []
        for row in checkpoints:
            if not isinstance(row, Mapping) or not isinstance(row.get("checks"), Mapping):
                raise ValidationError("automatic trajectory checkpoint result is invalid")
            checks.append(row["checks"])
        auto_values = {
            "Contract preservation": bool(auto.get("auto_contract_valid")),
            "Entity preservation": all(bool(row.get("entity_preservation")) for row in checks),
            "Numeric preservation": all(bool(row.get("numeric_preservation")) for row in checks),
            "Constraint preservation": all(
                bool(row.get("constraint_preservation")) for row in checks
            ),
        }
        human = trajectory_human[item_id]
        for measurement, field_name in trajectory_mapping.items():
            comparisons[measurement].append(
                (auto_values[measurement], _human_binary(human.get(field_name)))
            )

    ordered_names = (
        "Defect labeling",
        "Eligibility",
        "Contract preservation",
        "Entity preservation",
        "Numeric preservation",
        "Constraint preservation",
    )
    rows = [_confusion_row(name, comparisons[name]) for name in ordered_names]
    result = {
        "schema_version": 1,
        "protocol": AUTOMATIC_HUMAN_VALIDATION_PROTOCOL,
        "positive_class": (
            "defect present for Defect labeling; eligible/preserved for all other rows"
        ),
        "uncertain_policy": "excluded_from_binary_metrics_and_reported",
        "input_audit_item_count": len(input_items),
        "trajectory_audit_item_count": len(trajectory_items),
        "rows": rows,
    }
    result["validation_hash"] = content_hash(result)
    return result


def _audit_item_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValidationError(f"audit item {index} must be an object")
        raw_id = row.get("audit_item_id", row.get("item_id"))
        item_id = _text(raw_id, label=f"audit item {index} ID")
        if item_id in values:
            raise ValidationError("audit item IDs must be unique")
        values[item_id] = row
    if not values:
        raise ValidationError("audit items must be non-empty")
    return values


def _annotation_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValidationError(f"final annotation {index} must be an object")
        item_id = _text(row.get("item_id"), label=f"final annotation {index} item_id")
        labels = row.get("labels")
        if not isinstance(labels, Mapping) or not labels:
            raise ValidationError("final annotation labels must be a non-empty object")
        if item_id in values:
            raise ValidationError("final annotations contain duplicate item IDs")
        values[item_id] = labels
    return values


def _require_exact_annotation_coverage(
    items: Mapping[str, Any], annotations: Mapping[str, Any], *, label: str
) -> None:
    if set(items) != set(annotations):
        missing = sorted(set(items) - set(annotations))
        extra = sorted(set(annotations) - set(items))
        raise ValidationError(
            f"{label} final annotation coverage differs: missing={missing[:5]}, extra={extra[:5]}"
        )


def _human_binary(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "yes":
            return True
        if normalized == "no":
            return False
        if normalized == "uncertain":
            return None
    raise ValidationError("human audit binary label must be yes, no, uncertain, or boolean")


def _confusion_row(name: str, pairs: Sequence[tuple[bool, bool | None]]) -> dict[str, Any]:
    certain = [(automatic, human) for automatic, human in pairs if human is not None]
    tp = sum(automatic and bool(human) for automatic, human in certain)
    fp = sum(automatic and not bool(human) for automatic, human in certain)
    tn = sum(not automatic and not bool(human) for automatic, human in certain)
    fn = sum(not automatic and bool(human) for automatic, human in certain)
    n = len(certain)
    agreement = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "measurement": name,
        "n_total": len(pairs),
        "n_evaluable": n,
        "n_uncertain": len(pairs) - n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "agreement": agreement,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


__all__ = [
    "AUTOMATIC_HUMAN_VALIDATION_PROTOCOL",
    "DEFAULT_MEASUREMENT_COUNTS",
    "MEASUREMENT_ANALYSIS_PROTOCOL",
    "MEASUREMENT_ASSIGNMENT_PROTOCOL",
    "MEASUREMENT_KINDS",
    "MeasurementItem",
    "analyze_measurement_annotations",
    "build_measurement_assignment",
    "compare_automatic_human_audits",
]
