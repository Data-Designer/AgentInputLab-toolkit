"""Small, schema-checked analyses over completed experiment cell artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.metrics.paired import summarize_pairs

from .artifacts import verify_experiment_run_directory
from .schema import ExperimentManifest


def load_verified_cell_results(run_dir: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load a complete plan whose comparisons expose their full arm payloads.

    A terminal failed cell is complete, but cannot supply an unobserved paired
    arm value. Use ``load_verified_terminal_cell_results`` for lifecycle
    accounting; do not rerun failed cells merely to satisfy this loader.
    """

    return _load_verified_results(run_dir, require_all_succeeded=True)


def load_verified_terminal_cell_results(
    run_dir: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Load every verified terminal result, including explicit failed cells.

    Checksum, frozen-plan, manifest, complete-union and shard guards are the
    same as the paired loader. Missing results are not silently made failures.
    """

    return _load_verified_results(run_dir, require_all_succeeded=False)


def _load_verified_results(
    run_dir: str | Path, *, require_all_succeeded: bool
) -> tuple[Mapping[str, Any], ...]:

    root = Path(run_dir).expanduser().resolve()
    verification = verify_experiment_run_directory(root)
    if verification["is_shard"]:
        raise ArtifactError(
            "analysis refuses a single experiment shard; rerun the frozen full plan "
            "or merge all shards with a separately verified exact-union workflow"
        )
    if not verification["complete"]:
        raise ArtifactError("experiment run is incomplete; analysis refuses partial-plan results")
    if require_all_succeeded and not verification["all_succeeded"]:
        raise ArtifactError(
            "experiment run contains terminal failed cells; they already have "
            "terminal outcomes, but lack paired arm measurements. Use terminal "
            "accounting or an applicable explicit failure estimand; do not rerun "
            "them merely to obtain all-succeeded artifacts"
        )
    index = _read_object(root / "result_index.json", label="result index")
    if (
        set(index) != {"schema_version", "results"}
        or index["schema_version"] != 1
        or not isinstance(index["results"], Mapping)
    ):
        raise ArtifactError("experiment result index schema is invalid")
    records: list[Mapping[str, Any]] = []
    for cell_id, expected in sorted(index["results"].items()):
        if not isinstance(cell_id, str) or not isinstance(expected, str):
            raise ArtifactError("experiment result index entry is invalid")
        path = root / "results" / f"{cell_id}.json"
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"missing result for cell {cell_id}") from exc
        if hashlib.sha256(content).hexdigest() != expected:
            raise ArtifactError(f"result checksum mismatch for cell {cell_id}")
        record = _read_object(path, label=f"result {cell_id}")
        if record.get("manifest_hash") != _manifest_hash(root):
            raise ArtifactError(f"result manifest identity mismatch for cell {cell_id}")
        cell = record.get("cell")
        result = record.get("result")
        if (
            not isinstance(cell, Mapping)
            or cell.get("cell_id") != cell_id
            or not isinstance(result, Mapping)
            or result.get("cell_id") != cell_id
        ):
            raise ArtifactError(f"result cell identity is invalid for {cell_id}")
        records.append(record)
    if not records:
        raise ArtifactError("experiment run has no completed cell results")
    return tuple(records)


def summarize_terminal_execution_outcomes(run_dir: str | Path) -> Mapping[str, Any]:
    """Describe a complete plan without deleting or repairing its failures.

    ``succeeded`` here means the driver completed the comparison, not that the
    algorithm solved the benchmark. This summary must not be presented as a
    task-success rate or used to impute missing baseline/arm scores.
    """

    records = load_verified_terminal_cell_results(run_dir)
    failures: dict[str, int] = defaultdict(int)
    cells = []
    completed_comparisons = 0
    for record in records:
        result = record["result"]
        status = result["status"]
        if status == "succeeded":
            completed_comparisons += 1
        elif status == "failed":
            failure = result["failure"]
            failures[str(failure.get("category", "unspecified"))] += 1
        else:  # Also guarded by the artifact verifier; never treat missing as zero.
            raise ArtifactError("execution outcome must be an explicit terminal result")
        cells.append(
            {
                "cell_id": result["cell_id"],
                "status": status,
                "terminal": True,
                "retry_required": False,
                "included_in_execution_denominator": True,
                "task_score_imputed": False,
            }
        )
    count = len(records)
    return {
        "schema_version": 1,
        "artifact": "agentinputlab.terminal_execution_accounting.v1",
        "planned_terminal_cells": count,
        "terminal_cells": count,
        "completed_comparisons": completed_comparisons,
        "terminal_failed_cells": count - completed_comparisons,
        "terminal_complete": True,
        "all_comparisons_completed": completed_comparisons == count,
        "execution_completion_fraction": completed_comparisons / count,
        "task_success_rate": None,
        "failure_categories": dict(sorted(failures.items())),
        "cells": cells,
        "inference_performed": False,
        "raw_results_modified": False,
        "paper_claim_authorized": False,
        "scope_note": (
            "Terminal accounting, not benchmark task-success or paired inference. "
            "Fixed-budget failures require their source-bound failure estimand; "
            "repair attempts remain separate and never replace these outcomes."
        ),
    }


def summarize_paired_arm_metric(
    run_dir: str | Path,
    *,
    metric: str,
    baseline_arm_id: str = "no_op",
    algorithm_failure_value: float | None = None,
) -> Mapping[str, Any]:
    """Summarize matched arm values supplied by a production experiment driver.

    Drivers must put a list named ``arms`` in ``result.payload``.  Each item
    contains ``arm_id``, ``status``, and a finite numeric ``metrics`` mapping.
    A planned arm may report a typed algorithm failure.  Such failures are
    never silently excluded: callers must provide a preregistered finite
    ``algorithm_failure_value`` or analysis stops.
    """

    if not isinstance(metric, str) or not metric.strip():
        raise ValidationError("analysis metric must be non-empty")
    if algorithm_failure_value is not None and (
        isinstance(algorithm_failure_value, bool)
        or not isinstance(algorithm_failure_value, (int, float))
        or not math.isfinite(algorithm_failure_value)
    ):
        raise ValidationError("algorithm_failure_value must be finite when provided")
    pairs: dict[str, tuple[list[float], list[float]]] = {}
    arm_values: dict[str, list[float]] = defaultdict(list)
    complete_blocks = 0
    excluded_blocks = 0
    records = load_verified_cell_results(run_dir)
    for record in records:
        normalized = _arm_metrics(
            record,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
        )
        if normalized is None or baseline_arm_id not in normalized:
            excluded_blocks += 1
            continue
        complete_blocks += 1
        baseline = normalized[baseline_arm_id]
        for arm_id, value in normalized.items():
            arm_values[arm_id].append(value)
            if arm_id == baseline_arm_id:
                continue
            before, after = pairs.setdefault(arm_id, ([], []))
            before.append(baseline)
            after.append(value)
    if complete_blocks == 0:
        raise ArtifactError("no complete matched blocks expose the requested arm metric")
    summaries: dict[str, Any] = {}
    for arm_id in sorted(arm_values):
        values = arm_values[arm_id]
        payload: dict[str, Any] = {
            "count": len(values),
            "mean": fmean(values),
        }
        if arm_id != baseline_arm_id and arm_id in pairs:
            before, after = pairs[arm_id]
            paired = summarize_pairs(tuple(before), tuple(after))
            payload.update(
                {
                    "mean_effect_vs_no_op": paired.mean_effect,
                    "win_rate_vs_no_op": paired.win_rate,
                    "tie_rate_vs_no_op": paired.tie_rate,
                    "harm_rate_vs_no_op": paired.harm_rate,
                }
            )
        summaries[arm_id] = payload
    return {
        "schema_version": 1,
        "metric": metric,
        "baseline_arm_id": baseline_arm_id,
        "complete_blocks": complete_blocks,
        "excluded_incomplete_blocks": excluded_blocks,
        "algorithm_failure_value": algorithm_failure_value,
        "arms": summaries,
        "inference_performed": False,
        "scope_note": (
            "Descriptive matched summaries only; preregistered confidence intervals, "
            "interaction contrasts, and multiplicity correction run in the paper analysis."
        ),
    }


def analyze_confirmatory_interactions(
    run_dir: str | Path,
    *,
    metric: str,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Run the five preregistered family-by-visible-defect contrasts.

    The unit of resampling is the benchmark/unit group. Replicate seeds,
    model slots, and the two frozen implementations within a family are
    averaged inside that group before uncertainty is estimated.
    """

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    root = Path(run_dir).expanduser().resolve()
    manifest = _load_manifest(root)
    records = load_verified_cell_results(root)
    baseline_arm_id, arms_by_family = _confirmatory_arm_map(manifest)
    observations: dict[str, list[tuple[str, str, float, float]]] = {
        family.value: [] for family in _primary_families()
    }
    model_effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        values = _arm_metrics(
            record,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
        )
        if values is None:  # pragma: no cover - all-succeeded invariant
            raise ArtifactError("confirmatory record is not a completed cell")
        cell = record["cell"]
        cluster_id = _cluster_id(cell)
        model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
        defect_scores = _pre_action_defect_scores(record)
        baseline = values[baseline_arm_id]
        for family in _primary_families():
            family_name = family.value
            effect = fmean(values[arm_id] - baseline for arm_id in arms_by_family[family])
            observations[family_name].append(
                (cluster_id, model_slot, defect_scores[family_name], effect)
            )
            model_effects[model_slot][family_name].append(effect)

    rng = random.Random(random_seed)
    contrasts: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for family in _primary_families():
        family_name = family.value
        collapsed = _collapse_confirmatory_clusters(observations[family_name])
        effects = [row[2] for row in collapsed]
        defects = [row[1] for row in collapsed]
        slope = _linear_slope(defects, effects)
        effect_interval = _cluster_bootstrap_interval(
            effects,
            samples=bootstrap_samples,
            rng=rng,
        )
        slope_interval = _paired_cluster_bootstrap_slope_interval(
            defects,
            effects,
            samples=bootstrap_samples,
            rng=rng,
        )
        p_value = _permutation_slope_p_value(
            defects,
            effects,
            observed=slope,
            samples=permutation_samples,
            rng=rng,
        )
        raw_p_values[family_name] = p_value
        contrasts[family_name] = {
            "cluster_count": len(collapsed),
            "family_effect_mean": fmean(effects),
            "family_effect_cluster_bootstrap_95ci": list(effect_interval),
            "matched_defect_slope": slope,
            "matched_defect_slope_cluster_bootstrap_95ci": list(slope_interval),
            "permutation_p_value_raw": p_value,
        }
    adjusted = _holm_adjust(raw_p_values)
    for family_name, value in adjusted.items():
        contrasts[family_name]["holm_adjusted_p_value"] = value

    return {
        "schema_version": 1,
        "analysis": "five_family_matched_visible_defect_confirmation",
        "metric": metric,
        "baseline_arm_id": baseline_arm_id,
        "cluster_definition": "benchmark_id/split/unit_id",
        "replicate_handling": (
            "equal-weight mean over frozen implementations, model slots, and seeds "
            "within each benchmark/unit cluster"
        ),
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "random_seed": random_seed,
        "algorithm_failure_value": algorithm_failure_value,
        "contrasts": contrasts,
        "model_strata": {
            model_slot: {
                family: {
                    "count": len(values),
                    "mean_effect_vs_no_op": fmean(values),
                }
                for family, values in sorted(families.items())
            }
            for model_slot, families in sorted(model_effects.items())
        },
        "multiplicity": "Holm family-wise correction across the five matched contrasts",
        "paper_claim_authorized": False,
    }


def analyze_selector_policy(
    run_dir: str | Path,
    *,
    metric: str,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
    calibration_bins: int = 10,
) -> Mapping[str, Any]:
    """Evaluate frozen Selector value, risk-coverage, and gain calibration."""

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=1,
        random_seed=random_seed,
    )
    if (
        isinstance(calibration_bins, bool)
        or not isinstance(calibration_bins, int)
        or calibration_bins < 2
    ):
        raise ValidationError("calibration_bins must be an integer >= 2")
    records = load_verified_cell_results(run_dir)
    required_arms = {
        "no_op",
        "locked_best_eligible_global",
        "eligibility_aware_selector",
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        values = _arm_metrics(
            record,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
        )
        if values is None:  # pragma: no cover - all-succeeded invariant
            raise ArtifactError("selector record is not a completed cell")
        if set(values) != required_arms:
            raise ArtifactError(
                "selector confirmation requires exactly NO_OP, locked global, and selector arms"
            )
        decision = _selector_decision(record)
        baseline = values["no_op"]
        selector_value = values["eligibility_aware_selector"]
        rows.append(
            {
                "cluster_id": _cluster_id(record["cell"]),
                "model_slot": _required_record_text(
                    record["cell"],
                    "model_slot",
                    label="experiment cell",
                ),
                "baseline": baseline,
                "global": values["locked_best_eligible_global"],
                "selector": selector_value,
                "selector_gain": selector_value - baseline,
                "predicted_gain": decision["predicted_gain"],
                "confidence": decision["confidence"],
                "selected_family": decision["selected_family"],
                "selected_release_id": decision["selected_release_id"],
                "abstained": decision["abstained"],
                "selector_bundle_hash": decision["selector_bundle_hash"],
            }
        )
    collapsed_rows = _collapse_selector_clusters(rows)
    rng = random.Random(random_seed)
    selector_vs_no_op = _cluster_effect_interval(
        collapsed_rows,
        treatment_key="selector",
        baseline_key="baseline",
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    selector_vs_global = _cluster_effect_interval(
        collapsed_rows,
        treatment_key="selector",
        baseline_key="global",
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    coverage = fmean(
        0.0 if row["abstained"] or row["selected_family"] == SemanticFamily.NO_OP.value else 1.0
        for row in collapsed_rows
    )
    family_distribution: dict[str, int] = defaultdict(int)
    for row in collapsed_rows:
        family_distribution[row["selected_family"]] += 1
    return {
        "schema_version": 1,
        "analysis": "frozen_selector_confirmation",
        "metric": metric,
        "record_count": len(rows),
        "cluster_count": len(collapsed_rows),
        "analysis_unit": (
            "frozen benchmark unit; model-slot and seed outcomes are averaged "
            "within unit after requiring one invariant selector decision"
        ),
        "coverage": coverage,
        "no_op_rate": 1.0 - coverage,
        "family_distribution": dict(sorted(family_distribution.items())),
        "policy_value": {
            "no_op_mean": fmean(row["baseline"] for row in collapsed_rows),
            "locked_best_eligible_global_mean": fmean(row["global"] for row in collapsed_rows),
            "selector_mean": fmean(row["selector"] for row in collapsed_rows),
            "selector_vs_no_op": selector_vs_no_op,
            "selector_vs_locked_global": selector_vs_global,
            "selector_harm_rate_vs_no_op": fmean(
                1.0 if row["selector"] < row["baseline"] else 0.0 for row in collapsed_rows
            ),
        },
        "risk_coverage": _risk_coverage_curve(collapsed_rows),
        "gain_calibration": _gain_calibration(collapsed_rows, bins=calibration_bins),
        "algorithm_failure_value": algorithm_failure_value,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "paper_claim_authorized": False,
    }


def _arm_metrics(
    record: Mapping[str, Any],
    *,
    metric: str,
    algorithm_failure_value: float | None,
) -> Mapping[str, float] | None:
    result = record["result"]
    if result.get("status") != "succeeded":
        return None
    payload = result.get("payload")
    if not isinstance(payload, Mapping):
        raise ArtifactError("successful experiment result payload must be an object")
    arms = payload.get("arms")
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        raise ArtifactError("successful experiment result payload requires an arms array")
    normalized: dict[str, float] = {}
    for item in arms:
        if not isinstance(item, Mapping):
            raise ArtifactError("experiment arm outcome must be an object")
        arm_id = item.get("arm_id")
        status = item.get("status")
        metrics = item.get("metrics")
        if not isinstance(arm_id, str) or not arm_id.strip():
            raise ArtifactError("experiment arm outcome requires arm_id")
        if arm_id in normalized:
            raise ArtifactError("experiment result contains duplicate arm outcomes")
        if status == "failed":
            failure = item.get("failure")
            if not isinstance(failure, Mapping) or failure.get("scope") != "algorithm":
                raise ArtifactError(
                    f"experiment arm {arm_id} has a non-algorithm or invalid failure"
                )
            if algorithm_failure_value is None:
                raise ArtifactError(
                    f"experiment arm {arm_id} failed; provide the preregistered "
                    "algorithm_failure_value instead of complete-case deletion"
                )
            normalized[arm_id] = float(algorithm_failure_value)
            continue
        if status != "succeeded":
            raise ArtifactError(f"experiment arm {arm_id} status must be succeeded or failed")
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise ArtifactError(f"experiment arm {arm_id} lacks metric {metric!r}")
        value = metrics[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArtifactError(f"experiment arm metric {metric!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ArtifactError(f"experiment arm metric {metric!r} must be finite")
        normalized[arm_id] = numeric
    cell = record["cell"]
    expected = cell.get("arm_ids")
    if (
        not isinstance(expected, Sequence)
        or isinstance(expected, (str, bytes))
        or any(not isinstance(item, str) for item in expected)
    ):
        raise ArtifactError("experiment cell lacks a valid frozen arm inventory")
    if set(normalized) != set(expected) or len(normalized) != len(expected):
        raise ArtifactError("experiment result arm inventory differs from the frozen plan")
    return normalized


def _load_manifest(root: Path) -> ExperimentManifest:
    raw = _read_object(root / "manifest.json", label="experiment manifest")
    return ExperimentManifest.from_mapping(raw)


def _primary_families() -> tuple[SemanticFamily, ...]:
    return (
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    )


def _confirmatory_arm_map(
    manifest: ExperimentManifest,
) -> tuple[str, Mapping[SemanticFamily, tuple[str, ...]]]:
    baseline = tuple(arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.NO_OP)
    if len(baseline) != 1:
        raise ArtifactError("confirmatory manifest requires exactly one NO_OP arm")
    by_family: dict[SemanticFamily, tuple[str, ...]] = {}
    for family in _primary_families():
        arm_ids = tuple(arm.arm_id for arm in manifest.arms if arm.family is family)
        if len(arm_ids) != 2:
            raise ArtifactError(
                f"confirmatory manifest requires two frozen implementations for {family.value}"
            )
        by_family[family] = arm_ids
    return baseline[0], by_family


def _pre_action_defect_scores(record: Mapping[str, Any]) -> Mapping[str, float]:
    result = record["result"]
    payload = result.get("payload")
    pre_action = payload.get("pre_action") if isinstance(payload, Mapping) else None
    if not isinstance(pre_action, Mapping):
        raise ArtifactError("confirmatory payload requires a pre_action feature receipt")
    for key in ("feature_schema_id", "feature_schema_hash", "provenance_hash"):
        _required_record_text(pre_action, key, label="pre_action feature receipt")
    scores = pre_action.get("defect_scores")
    expected = {family.value for family in _primary_families()}
    if not isinstance(scores, Mapping) or set(scores) != expected:
        raise ArtifactError("pre_action defect_scores must contain exactly five families")
    normalized: dict[str, float] = {}
    for family, value in scores.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ArtifactError("pre_action defect scores must be finite in [0, 1]")
        normalized[family] = float(value)
    return normalized


def _selector_decision(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record["result"].get("payload")
    arms = payload.get("arms") if isinstance(payload, Mapping) else None
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        raise ArtifactError("selector result payload requires arms")
    matches = [
        item
        for item in arms
        if isinstance(item, Mapping) and item.get("arm_id") == "eligibility_aware_selector"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("decision"), Mapping):
        raise ArtifactError("selector arm requires exactly one frozen decision receipt")
    decision = matches[0]["decision"]
    required = {
        "selected_family",
        "selected_release_id",
        "abstained",
        "predicted_gain",
        "confidence",
        "selector_bundle_hash",
    }
    if set(decision) != required:
        raise ArtifactError("selector decision receipt fields are invalid")
    selected_family = decision["selected_family"]
    try:
        family = SemanticFamily(selected_family)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("selector decision selected_family is invalid") from exc
    selected_release_id = decision["selected_release_id"]
    bundle_hash = decision["selector_bundle_hash"]
    if not isinstance(selected_release_id, str) or not selected_release_id:
        raise ArtifactError("selector decision selected_release_id must be non-empty")
    if not isinstance(bundle_hash, str) or not _is_sha256(bundle_hash):
        raise ArtifactError("selector decision selector_bundle_hash must be SHA-256")
    abstained = decision["abstained"]
    if not isinstance(abstained, bool):
        raise ArtifactError("selector decision abstained must be boolean")
    predicted_gain = _finite_record_number(
        decision["predicted_gain"],
        label="selector predicted_gain",
    )
    confidence = _finite_record_number(
        decision["confidence"],
        label="selector confidence",
    )
    if not 0 <= confidence <= 1:
        raise ArtifactError("selector confidence must be in [0, 1]")
    if abstained != (family is SemanticFamily.NO_OP):
        raise ArtifactError("selector abstention and selected_family are inconsistent")
    return {
        "selected_family": family.value,
        "selected_release_id": selected_release_id,
        "abstained": abstained,
        "predicted_gain": predicted_gain,
        "confidence": confidence,
        "selector_bundle_hash": bundle_hash,
    }


def _cluster_id(cell: Mapping[str, Any]) -> str:
    benchmark = _required_record_text(cell, "benchmark_id", label="experiment cell")
    split = _required_record_text(cell, "split", label="experiment cell")
    unit = _required_record_text(cell, "unit_id", label="experiment cell")
    return hashlib.sha256(f"{benchmark}\0{split}\0{unit}".encode()).hexdigest()


def _collapse_selector_clusters(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Collapse repeated model/seed observations to the frozen-unit estimand."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    collapsed: list[Mapping[str, Any]] = []
    decision_fields = (
        "selected_family",
        "selected_release_id",
        "abstained",
        "predicted_gain",
        "confidence",
        "selector_bundle_hash",
    )
    for cluster_id, members in sorted(grouped.items()):
        reference = {field: members[0][field] for field in decision_fields}
        for member in members[1:]:
            if any(member[field] != value for field, value in reference.items()):
                raise ArtifactError(
                    "selector decision changed across model/seed replicates of one "
                    "frozen benchmark unit"
                )
        baseline = fmean(float(item["baseline"]) for item in members)
        selector = fmean(float(item["selector"]) for item in members)
        collapsed.append(
            {
                "cluster_id": cluster_id,
                "replicate_count": len(members),
                "baseline": baseline,
                "global": fmean(float(item["global"]) for item in members),
                "selector": selector,
                "selector_gain": selector - baseline,
                **reference,
            }
        )
    if len(collapsed) < 2:
        raise ArtifactError("selector analysis requires at least two independent clusters")
    return tuple(collapsed)


def _collapse_confirmatory_clusters(
    rows: Sequence[tuple[str, str, float, float]],
) -> tuple[tuple[str, float, float], ...]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cluster_id, _model_slot, defect, effect in rows:
        grouped[cluster_id].append((defect, effect))
    collapsed: list[tuple[str, float, float]] = []
    for cluster_id, values in sorted(grouped.items()):
        defects = [item[0] for item in values]
        if max(defects) - min(defects) > 1e-12:
            raise ArtifactError("pre-action defect score changed across model/seed replicates")
        collapsed.append((cluster_id, fmean(defects), fmean(item[1] for item in values)))
    if len(collapsed) < 2:
        raise ArtifactError("confirmatory analysis requires at least two independent clusters")
    return tuple(collapsed)


def _linear_slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ArtifactError("matched interaction slope requires paired observations")
    x_mean = fmean(x)
    y_mean = fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        raise ArtifactError("matched defect has no between-cluster variation")
    return (
        sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y, strict=True))
        / denominator
    )


def _cluster_bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if len(values) < 2:
        raise ArtifactError("cluster bootstrap requires at least two clusters")
    estimates = [fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return _quantile_interval(estimates)


def _paired_cluster_bootstrap_slope_interval(
    defects: Sequence[float],
    effects: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    estimates: list[float] = []
    attempts = 0
    max_attempts = samples * 20
    while len(estimates) < samples and attempts < max_attempts:
        attempts += 1
        indices = [rng.randrange(len(defects)) for _ in defects]
        sampled_defects = [defects[index] for index in indices]
        if max(sampled_defects) == min(sampled_defects):
            continue
        estimates.append(
            _linear_slope(
                sampled_defects,
                [effects[index] for index in indices],
            )
        )
    if len(estimates) < samples:
        raise ArtifactError("cluster bootstrap could not preserve defect variation")
    return _quantile_interval(estimates)


def _permutation_slope_p_value(
    defects: Sequence[float],
    effects: Sequence[float],
    *,
    observed: float,
    samples: int,
    rng: random.Random,
) -> float:
    exceedances = 0
    shuffled = list(defects)
    for _ in range(samples):
        rng.shuffle(shuffled)
        permuted = _linear_slope(shuffled, effects)
        if abs(permuted) >= abs(observed) - 1e-15:
            exceedances += 1
    return (exceedances + 1.0) / (samples + 1.0)


def _holm_adjust(values: Mapping[str, float]) -> Mapping[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def _cluster_effect_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment_key: str,
    baseline_key: str,
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(float(row[treatment_key]) - float(row[baseline_key]))
    cluster_effects = [fmean(values) for _, values in sorted(grouped.items())]
    return {
        "cluster_count": len(cluster_effects),
        "mean_effect": fmean(cluster_effects),
        "cluster_bootstrap_95ci": list(
            _cluster_bootstrap_interval(
                cluster_effects,
                samples=bootstrap_samples,
                rng=rng,
            )
        ),
    }


def _risk_coverage_curve(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    confidences = sorted({float(row["confidence"]) for row in rows}, reverse=True)
    if not confidences:
        raise ArtifactError("selector risk-coverage requires confidence values")
    indices = {
        min(len(confidences) - 1, round(position * (len(confidences) - 1) / 10))
        for position in range(11)
    }
    thresholds = [confidences[index] for index in sorted(indices)]
    curve: list[Mapping[str, Any]] = []
    for threshold in thresholds:
        accepted = [
            (not bool(row["abstained"])) and float(row["confidence"]) >= threshold for row in rows
        ]
        deployed = [
            float(row["selector"]) if use_selector else float(row["baseline"])
            for row, use_selector in zip(rows, accepted, strict=True)
        ]
        baseline = [float(row["baseline"]) for row in rows]
        curve.append(
            {
                "confidence_threshold": threshold,
                "coverage": fmean(1.0 if value else 0.0 for value in accepted),
                "policy_value": fmean(deployed),
                "mean_effect_vs_no_op": fmean(
                    value - reference for value, reference in zip(deployed, baseline, strict=True)
                ),
                "harm_rate_vs_no_op": fmean(
                    1.0 if value < reference else 0.0
                    for value, reference in zip(deployed, baseline, strict=True)
                ),
            }
        )
    return curve


def _gain_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> Mapping[str, Any]:
    ordered = sorted(
        (
            float(row["predicted_gain"]),
            float(row["selector_gain"]),
        )
        for row in rows
    )
    bin_count = min(bins, len(ordered))
    if bin_count < 2:
        raise ArtifactError("selector gain calibration requires at least two records")
    groups: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for index, item in enumerate(ordered):
        groups[min(bin_count - 1, index * bin_count // len(ordered))].append(item)
    payload: list[Mapping[str, Any]] = []
    weighted_error = 0.0
    for index, group in enumerate(groups):
        predicted = fmean(item[0] for item in group)
        realized = fmean(item[1] for item in group)
        error = abs(predicted - realized)
        weighted_error += len(group) * error
        payload.append(
            {
                "bin": index,
                "count": len(group),
                "predicted_gain_mean": predicted,
                "realized_gain_mean": realized,
                "absolute_error": error,
            }
        )
    return {
        "bins": payload,
        "expected_absolute_calibration_error": weighted_error / len(ordered),
        "median_absolute_record_error": median(
            abs(predicted - realized) for predicted, realized in ordered
        ),
    }


def _quantile_interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    lower = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return lower, upper


def _validate_resample_arguments(
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    random_seed: int,
) -> None:
    for label, value, minimum in (
        ("bootstrap_samples", bootstrap_samples, 100),
        ("permutation_samples", permutation_samples, 1),
        ("random_seed", random_seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValidationError(f"{label} must be an integer >= {minimum}")


def _required_record_text(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ArtifactError(f"{label} requires non-empty {key}")
    return result


def _finite_record_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArtifactError(f"{label} must be finite numeric")
    return float(value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _manifest_hash(root: Path) -> str:
    identity = _read_object(root / "run_identity.json", label="run identity")
    raw = identity.get("identity")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("manifest_hash"), str):
        raise ArtifactError("run identity lacks manifest_hash")
    return raw["manifest_hash"]


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is missing or invalid") from exc
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be a JSON object")
    return value


__all__ = [
    "analyze_confirmatory_interactions",
    "analyze_selector_policy",
    "load_verified_cell_results",
    "summarize_paired_arm_metric",
]
