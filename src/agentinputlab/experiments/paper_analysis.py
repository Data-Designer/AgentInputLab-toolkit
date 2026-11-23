"""Protocol-checked analyses for the frozen KDD paper experiment blocks.

The general analysis helpers intentionally accept many matched-arm layouts.
This module adds the stricter paper contracts: five separate three-arm
family-native runs, one six-arm common-support run, four matched five-arm
retriever-factorial runs, and the 526-group/14-policy Selector campaign.

All entry points require complete, checksum-verified run directories.  They
never execute experiments, mutate artifacts, or authorize paper claims.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .analysis import (
    _arm_metrics,
    _cluster_bootstrap_interval,
    _cluster_id,
    _collapse_confirmatory_clusters,
    _holm_adjust,
    _linear_slope,
    _load_manifest,
    _paired_cluster_bootstrap_slope_interval,
    _permutation_slope_p_value,
    _pre_action_defect_scores,
    _primary_families,
    _required_record_text,
    _validate_resample_arguments,
    analyze_confirmatory_interactions,
    load_verified_cell_results,
)
from .schema import ExperimentManifest, ExperimentStage

PAPER_FAMILY_NATIVE_PROTOCOL = "paper-family-native-v1"
PAPER_COMMON_SUPPORT_PROTOCOL = "paper-common-support-v1"
PAPER_RETRIEVER_FACTORIAL_PROTOCOL = "paper-retriever-factorial-v1"
PAPER_SELECTOR_PROTOCOL = "paper-selector-14-policy-v1"
LEGACY_CONFIRMATORY_PROTOCOL = "legacy-11-arm-v1"

_CENSUS_PROTOCOL = "agentinputlab.experiment_census.v1"
_FAMILY_NATIVE_KIND = "family_native"
_COMMON_SUPPORT_KIND = "common_support"
_RETRIEVER_FACTORIAL_KIND = "retriever_factorial_cell"
_RETRIEVER_FACTORIAL_PRIMARY_KIND = "retriever_factorial_primary_cell"
_RETRIEVER_FACTORIAL_AUXILIARY_KIND = "retriever_factorial_auxiliary_cell"
_PAPER_SELECTOR_POLICIES = frozenset(
    {
        "no_op",
        "random_eligible",
        "fixed_reformulate",
        "fixed_decompose",
        "fixed_acquire",
        "fixed_clarify",
        "fixed_curate",
        "best_fixed_global",
        "defect_rule",
        "zero_shot_router",
        "penalized_t_learner",
        "gbdt_t_learner",
        "eligibility_aware_selector",
        "observed_action_oracle",
    }
)
_PAPER_RETRIEVERS = ("BAAI/bge-m3", "intfloat/e5-large-v2")
_FORMAL_PAPER_RETRIEVERS = (
    "bm25",
    "intfloat/e5-large-v2",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
)
_PAPER_TOP_K = (5, 20)
_SELECTOR_GATE_ARTIFACT = "agentinputlab.paper_selector_claim_gates"


@dataclass(frozen=True, slots=True)
class _VerifiedPaperRun:
    root: Path
    manifest: ExperimentManifest
    records: tuple[Mapping[str, Any], ...]
    campaign_kind: str
    independent_groups: int
    expected_plan_cells: int
    expected_agent_episodes: int
    cluster_ids: frozenset[str]


def infer_paper_run_protocol(run_dir: str | Path) -> str:
    """Infer a paper protocol only from a frozen manifest census declaration."""

    manifest = _load_manifest(Path(run_dir).expanduser().resolve())
    census = manifest.metadata.get("experiment_census")
    if not isinstance(census, Mapping):
        return LEGACY_CONFIRMATORY_PROTOCOL
    kind = census.get("campaign_kind")
    if kind == _FAMILY_NATIVE_KIND:
        return PAPER_FAMILY_NATIVE_PROTOCOL
    if kind == _COMMON_SUPPORT_KIND:
        return PAPER_COMMON_SUPPORT_PROTOCOL
    if kind in {
        _RETRIEVER_FACTORIAL_KIND,
        _RETRIEVER_FACTORIAL_PRIMARY_KIND,
        _RETRIEVER_FACTORIAL_AUXILIARY_KIND,
    }:
        return PAPER_RETRIEVER_FACTORIAL_PROTOCOL
    return LEGACY_CONFIRMATORY_PROTOCOL


def analyze_confirmatory_protocol(
    run_dirs: Sequence[str | Path],
    *,
    protocol: str = "auto",
    metric: str = "valid_success",
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Route old and paper-v1 confirmatory layouts without weakening either."""

    roots = _run_roots(run_dirs)
    selected = protocol
    if selected == "auto":
        inferred = {infer_paper_run_protocol(root) for root in roots}
        if len(inferred) != 1:
            raise ValidationError("confirmatory run directories mix incompatible protocols")
        selected = next(iter(inferred))
    if selected == LEGACY_CONFIRMATORY_PROTOCOL:
        if len(roots) != 1:
            raise ValidationError("legacy 11-arm confirmation accepts exactly one run directory")
        return analyze_confirmatory_interactions(
            roots[0],
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            random_seed=random_seed,
        )
    if selected == PAPER_FAMILY_NATIVE_PROTOCOL:
        return analyze_paper_family_confirmation(
            roots,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            random_seed=random_seed,
        )
    if selected == PAPER_COMMON_SUPPORT_PROTOCOL:
        if len(roots) != 1:
            raise ValidationError("paper common-support analysis accepts exactly one run directory")
        return analyze_paper_common_support(
            roots[0],
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
    if selected == PAPER_RETRIEVER_FACTORIAL_PROTOCOL:
        analyzer = (
            analyze_paper_dual_arena_retriever_factorial
            if len(roots) == 16
            else analyze_paper_retriever_factorial
        )
        return analyzer(
            roots,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
    raise ValidationError(f"unsupported confirmatory analysis protocol: {selected!r}")


def analyze_paper_family_confirmation(
    run_dirs: Sequence[str | Path],
    *,
    metric: str,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Analyze the exact five-run, three-arm family-native paper protocol."""

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    roots = _run_roots(run_dirs)
    if len(roots) != 5:
        raise ValidationError("paper family confirmation requires exactly five run directories")
    runs = tuple(
        _load_and_validate_paper_run(root, expected_kind=_FAMILY_NATIVE_KIND) for root in roots
    )
    by_family: dict[SemanticFamily, _VerifiedPaperRun] = {}
    all_clusters: set[str] = set()
    for run in runs:
        family = _family_native_arm_contract(run.manifest)
        if family in by_family:
            raise ArtifactError(f"duplicate family-native run for {family.value}")
        overlap = all_clusters.intersection(run.cluster_ids)
        if overlap:
            raise ArtifactError(
                "paper family-native runs reuse benchmark/unit groups across families"
            )
        all_clusters.update(run.cluster_ids)
        by_family[family] = run
    expected_families = set(_primary_families())
    if set(by_family) != expected_families:
        missing = sorted(family.value for family in expected_families - set(by_family))
        raise ArtifactError(f"paper family confirmation is missing families: {missing}")

    observations: dict[str, list[tuple[str, str, float, float]]] = {
        family.value: [] for family in _primary_families()
    }
    model_effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for family in _primary_families():
        run = by_family[family]
        baseline_arm_id, treatment_arm_ids = _family_native_arm_ids(run.manifest, family)
        for record in run.records:
            values = _arm_metrics(
                record,
                metric=metric,
                algorithm_failure_value=algorithm_failure_value,
            )
            if values is None:  # pragma: no cover - verified all-succeeded invariant
                raise ArtifactError("family-native record is not a completed cell")
            cell = record["cell"]
            cluster_id = _cluster_id(cell)
            model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
            defect = _pre_action_defect_scores(record)[family.value]
            baseline = values[baseline_arm_id]
            effect = fmean(values[arm_id] - baseline for arm_id in treatment_arm_ids)
            observations[family.value].append((cluster_id, model_slot, defect, effect))
            model_effects[model_slot][family.value].append(effect)

    rng = random.Random(random_seed)
    contrasts: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for family in _primary_families():
        family_name = family.value
        collapsed = _collapse_confirmatory_clusters(observations[family_name])
        effects = [row[2] for row in collapsed]
        defects = [row[1] for row in collapsed]
        slope = _linear_slope(defects, effects)
        raw_p = _permutation_slope_p_value(
            defects,
            effects,
            observed=slope,
            samples=permutation_samples,
            rng=rng,
        )
        raw_p_values[family_name] = raw_p
        contrasts[family_name] = {
            "cluster_count": len(collapsed),
            "family_effect_mean": fmean(effects),
            "family_effect_cluster_bootstrap_95ci": list(
                _cluster_bootstrap_interval(effects, samples=bootstrap_samples, rng=rng)
            ),
            "matched_defect_slope": slope,
            "matched_defect_slope_cluster_bootstrap_95ci": list(
                _paired_cluster_bootstrap_slope_interval(
                    defects,
                    effects,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            ),
            "permutation_p_value_raw": raw_p,
        }
    for family_name, adjusted in _holm_adjust(raw_p_values).items():
        contrasts[family_name]["holm_adjusted_p_value"] = adjusted

    return {
        "schema_version": 1,
        "analysis": "paper_five_family_native_confirmation",
        "analysis_protocol": PAPER_FAMILY_NATIVE_PROTOCOL,
        "metric": metric,
        "run_count": 5,
        "independent_groups": 3_000,
        "verified_plan_cells": 6_000,
        "verified_agent_episodes": 18_000,
        "arm_layout": "five disjoint runs: NO_OP + two frozen representatives",
        "cluster_definition": "benchmark_id/split/unit_id",
        "replicate_handling": (
            "equal-weight mean over two frozen implementations, model slots, and "
            "seeds within each benchmark/unit cluster"
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


def analyze_paper_common_support(
    run_dir: str | Path,
    *,
    metric: str,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Analyze the exact 526-group, six-arm horizontal-comparison block."""

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=1,
        random_seed=random_seed,
    )
    run = _load_and_validate_paper_run(run_dir, expected_kind=_COMMON_SUPPORT_KIND)
    baseline, arms = _common_support_arm_contract(run.manifest)
    rows_by_cluster: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    model_effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in run.records:
        values = _arm_metrics(
            record,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
        )
        if values is None:  # pragma: no cover
            raise ArtifactError("common-support record is not a completed cell")
        cluster_id = _cluster_id(record["cell"])
        model_slot = _required_record_text(
            record["cell"],
            "model_slot",
            label="experiment cell",
        )
        row = {"no_op": values[baseline]}
        for family, arm_id in arms.items():
            row[family.value] = values[arm_id]
            model_effects[model_slot][family.value].append(values[arm_id] - values[baseline])
        rows_by_cluster[cluster_id].append(row)
    collapsed = {
        cluster_id: {
            key: fmean(float(row[key]) for row in rows)
            for key in ("no_op", *(family.value for family in _primary_families()))
        }
        for cluster_id, rows in sorted(rows_by_cluster.items())
    }
    if len(collapsed) != 526:
        raise ArtifactError("paper common-support analysis requires exactly 526 clusters")

    rng = random.Random(random_seed)
    family_summaries: dict[str, Any] = {}
    for family in _primary_families():
        effects = [row[family.value] - row["no_op"] for row in collapsed.values()]
        family_summaries[family.value] = {
            "cluster_count": len(effects),
            "mean_value": fmean(row[family.value] for row in collapsed.values()),
            "mean_effect_vs_no_op": fmean(effects),
            "cluster_bootstrap_95ci": list(
                _cluster_bootstrap_interval(effects, samples=bootstrap_samples, rng=rng)
            ),
        }
    pairwise: dict[str, Any] = {}
    families = _primary_families()
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            effects = [row[left.value] - row[right.value] for row in collapsed.values()]
            pairwise[f"{left.value}_vs_{right.value}"] = {
                "mean_difference": fmean(effects),
                "cluster_bootstrap_95ci": list(
                    _cluster_bootstrap_interval(effects, samples=bootstrap_samples, rng=rng)
                ),
            }
    ranking = sorted(
        (
            {
                "family": family.value,
                "mean_value": family_summaries[family.value]["mean_value"],
            }
            for family in families
        ),
        key=lambda row: (-float(row["mean_value"]), str(row["family"])),
    )
    return {
        "schema_version": 1,
        "analysis": "paper_common_support_horizontal_comparison",
        "analysis_protocol": PAPER_COMMON_SUPPORT_PROTOCOL,
        "metric": metric,
        "run_count": 1,
        "independent_groups": 526,
        "verified_plan_cells": 1_052,
        "verified_agent_episodes": 6_312,
        "arm_layout": "NO_OP + one frozen representative per family",
        "minimum_groups_for_ranking": 400,
        "ranking_allowed": len(collapsed) >= 400,
        "family_summaries": family_summaries,
        "pairwise_family_differences": pairwise,
        "descriptive_ranking": ranking,
        "model_strata": {
            model_slot: {
                family: {
                    "count": len(values),
                    "mean_effect_vs_no_op": fmean(values),
                }
                for family, values in sorted(by_family.items())
            }
            for model_slot, by_family in sorted(model_effects.items())
        },
        "scope_note": (
            "Bounded common-support population; family-native effects remain primary. "
            "No multiplicity-adjusted superiority claim is authorized here."
        ),
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "algorithm_failure_value": algorithm_failure_value,
        "paper_claim_authorized": False,
    }


def analyze_paper_retriever_factorial(
    run_dirs: Sequence[str | Path],
    *,
    metric: str,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Analyze four matched retriever-by-top-k cells with shared unit IDs."""

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=1,
        random_seed=random_seed,
    )
    roots = _run_roots(run_dirs)
    if len(roots) != 4:
        raise ValidationError("paper retriever factorial requires exactly four run directories")
    runs = tuple(
        _load_and_validate_paper_run(root, expected_kind=_RETRIEVER_FACTORIAL_KIND)
        for root in roots
    )
    by_cell: dict[tuple[str, int], _VerifiedPaperRun] = {}
    reference_arms: tuple[tuple[str, str, str], ...] | None = None
    reference_clusters: frozenset[str] | None = None
    reference_targets: tuple[tuple[str, str], ...] | None = None
    reference_models: tuple[str, ...] | None = None
    reference_seeds: tuple[int, ...] | None = None
    for run in runs:
        key = _retriever_dependency_cell(run.manifest)
        if key in by_cell:
            raise ArtifactError(f"duplicate retriever-factorial dependency cell: {key}")
        _retriever_arm_contract(run.manifest)
        arm_identity = tuple(
            (arm.arm_id, arm.release_id, arm.family.value) for arm in run.manifest.arms
        )
        targets = tuple((target.cell_id, target.split) for target in run.manifest.benchmarks)
        if reference_arms is None:
            reference_arms = arm_identity
            reference_clusters = run.cluster_ids
            reference_targets = targets
            reference_models = run.manifest.model_slots
            reference_seeds = run.manifest.seeds
        elif (
            arm_identity != reference_arms
            or run.cluster_ids != reference_clusters
            or targets != reference_targets
            or run.manifest.model_slots != reference_models
            or run.manifest.seeds != reference_seeds
        ):
            raise ArtifactError(
                "retriever-factorial cells must share arms, groups, targets, model slots, and seeds"
            )
        by_cell[key] = run
    expected_cells = {
        (retriever, top_k) for retriever in _PAPER_RETRIEVERS for top_k in _PAPER_TOP_K
    }
    if set(by_cell) != expected_cells:
        raise ArtifactError("retriever-factorial runs do not form the frozen 2x2 dependency grid")
    dependency_receipts = {key: _retriever_runtime_receipt(run) for key, run in by_cell.items()}
    dependency_balance = _validate_retriever_factorial_balance(dependency_receipts)

    effects: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    cell_summaries: dict[str, Any] = {}
    rng = random.Random(random_seed)
    for key in sorted(by_cell):
        run = by_cell[key]
        baseline, treatment_arms = _retriever_arm_contract(run.manifest)
        by_cluster: dict[str, dict[str, float]] = {}
        for record in run.records:
            values = _arm_metrics(
                record,
                metric=metric,
                algorithm_failure_value=algorithm_failure_value,
            )
            if values is None:  # pragma: no cover
                raise ArtifactError("retriever-factorial record is not a completed cell")
            cluster_id = _cluster_id(record["cell"])
            by_cluster[cluster_id] = {
                arm_id: values[arm_id] - values[baseline] for arm_id in treatment_arms
            }
        effects[key] = by_cluster
        label = f"{key[0]}::k{key[1]}"
        cell_summaries[label] = {
            "retriever_id": key[0],
            "top_k": key[1],
            "independent_groups": len(by_cluster),
            "arms": {
                arm_id: {
                    "mean_effect_vs_no_op": fmean(row[arm_id] for row in by_cluster.values()),
                    "cluster_bootstrap_95ci": list(
                        _cluster_bootstrap_interval(
                            [row[arm_id] for row in by_cluster.values()],
                            samples=bootstrap_samples,
                            rng=rng,
                        )
                    ),
                }
                for arm_id in treatment_arms
            },
        }

    assert reference_clusters is not None  # established by four non-empty runs
    assert reference_arms is not None
    treatment_arm_ids = tuple(item[0] for item in reference_arms if item[2] != "no_op")
    interactions: dict[str, Any] = {}
    bge, e5 = _PAPER_RETRIEVERS
    k5, k20 = _PAPER_TOP_K
    for arm_id in treatment_arm_ids:
        retriever_contrast = []
        top_k_contrast = []
        joint_contrast = []
        for cluster_id in sorted(reference_clusters):
            bge_k5 = effects[(bge, k5)][cluster_id][arm_id]
            bge_k20 = effects[(bge, k20)][cluster_id][arm_id]
            e5_k5 = effects[(e5, k5)][cluster_id][arm_id]
            e5_k20 = effects[(e5, k20)][cluster_id][arm_id]
            retriever_contrast.append((bge_k5 + bge_k20 - e5_k5 - e5_k20) / 2.0)
            top_k_contrast.append((bge_k20 + e5_k20 - bge_k5 - e5_k5) / 2.0)
            joint_contrast.append((bge_k20 - bge_k5) - (e5_k20 - e5_k5))
        interactions[arm_id] = {
            "retriever_bge_minus_e5": _effect_summary(
                retriever_contrast,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "top_k_20_minus_5": _effect_summary(
                top_k_contrast,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "retriever_by_top_k": _effect_summary(
                joint_contrast,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
        }
    return {
        "schema_version": 1,
        "analysis": "paper_retriever_by_top_k_factorial",
        "analysis_protocol": PAPER_RETRIEVER_FACTORIAL_PROTOCOL,
        "metric": metric,
        "run_count": 4,
        "independent_groups": 240,
        "verified_plan_cells": 960,
        "verified_agent_episodes": 4_800,
        "dependency_cells": cell_summaries,
        "dependency_balance": dependency_balance,
        "interactions": interactions,
        "cluster_definition": "same benchmark_id/split/unit_id replayed in all four cells",
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "algorithm_failure_value": algorithm_failure_value,
        "paper_claim_authorized": False,
    }


def analyze_paper_dual_arena_retriever_factorial(
    run_dirs: Sequence[str | Path],
    *,
    metric: str,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Analyze the formal 4-retriever x 2-top-k design in τ³ and BRIGHT.

    τ³ is the primary estimand. BRIGHT is reported separately as a directional
    auxiliary validation; its observations are never pooled with τ³.
    """

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=1,
        random_seed=random_seed,
    )
    roots = _run_roots(run_dirs)
    if len(roots) != 16:
        raise ValidationError("formal retriever factorial requires exactly sixteen run directories")
    by_role: dict[str, list[_VerifiedPaperRun]] = {"primary": [], "auxiliary": []}
    for root in roots:
        manifest = _load_manifest(root)
        dependency = manifest.metadata.get("dependency_cell")
        if not isinstance(dependency, Mapping):
            raise ArtifactError("formal retriever-factorial run lacks dependency metadata")
        role = dependency.get("arena_role")
        if role not in by_role:
            raise ArtifactError("formal retriever-factorial arena role is invalid")
        expected_kind = (
            _RETRIEVER_FACTORIAL_PRIMARY_KIND
            if role == "primary"
            else _RETRIEVER_FACTORIAL_AUXILIARY_KIND
        )
        by_role[str(role)].append(_load_and_validate_paper_run(root, expected_kind=expected_kind))
    if any(len(runs) != 8 for runs in by_role.values()):
        raise ArtifactError("formal retriever factorial requires eight cells in each arena")

    rng = random.Random(random_seed)
    analyses = {
        role: _analyze_formal_retriever_arena(
            runs,
            role=role,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for role, runs in by_role.items()
    }
    return {
        "schema_version": 2,
        "analysis": "paper_dual_arena_retriever_by_top_k_factorial",
        "analysis_protocol": PAPER_RETRIEVER_FACTORIAL_PROTOCOL,
        "metric": metric,
        "run_count": 16,
        "primary": analyses["primary"],
        "auxiliary": analyses["auxiliary"],
        "claim_order": [
            "tau3_primary_retriever_factorial",
            "bright_auxiliary_directional_validation",
        ],
        "cross_arena_pooling": False,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "algorithm_failure_value": algorithm_failure_value,
        "paper_claim_authorized": False,
    }


def _analyze_formal_retriever_arena(
    runs: Sequence[_VerifiedPaperRun],
    *,
    role: str,
    metric: str,
    algorithm_failure_value: float | None,
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    by_cell: dict[tuple[str, int], _VerifiedPaperRun] = {}
    reference: _VerifiedPaperRun | None = None
    for run in runs:
        key = _retriever_dependency_cell(run.manifest, formal=True)
        if key in by_cell:
            raise ArtifactError(f"duplicate formal retriever-factorial cell: {key}")
        _retriever_arm_contract(run.manifest)
        if reference is None:
            reference = run
        elif (
            tuple((arm.arm_id, arm.release_id, arm.family.value) for arm in run.manifest.arms)
            != tuple(
                (arm.arm_id, arm.release_id, arm.family.value) for arm in reference.manifest.arms
            )
            or run.cluster_ids != reference.cluster_ids
            or run.manifest.model_slots != reference.manifest.model_slots
            or run.manifest.seeds != reference.manifest.seeds
            or tuple((cell.cell_id, cell.split) for cell in run.manifest.benchmarks)
            != tuple((cell.cell_id, cell.split) for cell in reference.manifest.benchmarks)
        ):
            raise ArtifactError(
                "formal retriever-factorial cells must share arms, groups, targets, models, and seeds"
            )
        by_cell[key] = run
    expected = {
        (retriever, top_k) for retriever in _FORMAL_PAPER_RETRIEVERS for top_k in _PAPER_TOP_K
    }
    if set(by_cell) != expected:
        raise ArtifactError("formal retriever-factorial cells do not form the frozen 4x2 grid")
    dependency_balance = _validate_formal_retriever_factorial_balance(
        {key: _formal_retriever_runtime_receipt(run) for key, run in by_cell.items()}
    )
    assert reference is not None
    baseline, treatment_arms = _retriever_arm_contract(reference.manifest)
    effects: dict[tuple[str, int], dict[str, dict[str, dict[str, float]]]] = {}
    cell_summaries: dict[str, Any] = {}
    for key, run in sorted(by_cell.items()):
        by_group: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for record in run.records:
            values = _arm_metrics(
                record,
                metric=metric,
                algorithm_failure_value=algorithm_failure_value,
            )
            if values is None:  # pragma: no cover
                raise ArtifactError("formal retriever-factorial record is incomplete")
            cell = record["cell"]
            group_id = _cluster_id(cell)
            model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
            by_group[group_id][model_slot] = {
                arm_id: values[arm_id] - values[baseline] for arm_id in treatment_arms
            }
        if any(set(models) != set(run.manifest.model_slots) for models in by_group.values()):
            raise ArtifactError("formal retriever cell lacks the exact two-model group grid")
        effects[key] = by_group
        cell_summaries[f"{key[0]}::k{key[1]}"] = {
            "retriever_id": key[0],
            "top_k": key[1],
            "independent_groups": len(by_group),
            "model_slots": list(run.manifest.model_slots),
            "arms": {
                arm_id: _effect_summary(
                    [
                        fmean(model_values[arm_id] for model_values in models.values())
                        for models in by_group.values()
                    ],
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                )
                for arm_id in treatment_arms
            },
        }

    interactions: dict[str, Any] = {}
    reference_retriever = "BAAI/bge-m3"
    for arm_id in treatment_arms:
        top_k_contrast: list[float] = []
        retriever_contrasts: dict[str, list[float]] = {
            retriever: []
            for retriever in _FORMAL_PAPER_RETRIEVERS
            if retriever != reference_retriever
        }
        for group_id in sorted(reference.cluster_ids):
            group_top_k_contrasts: list[float] = []
            reference_mean: float | None = None
            for retriever in _FORMAL_PAPER_RETRIEVERS:
                low = fmean(values[arm_id] for values in effects[(retriever, 5)][group_id].values())
                high = fmean(
                    values[arm_id] for values in effects[(retriever, 20)][group_id].values()
                )
                if retriever == reference_retriever:
                    reference_mean = (low + high) / 2.0
                group_top_k_contrasts.append(high - low)
            top_k_contrast.append(fmean(group_top_k_contrasts))
            if reference_mean is None:  # pragma: no cover - frozen grid contains BGE-M3
                raise ArtifactError("formal retriever grid lacks its BGE-M3 reference")
            for retriever in retriever_contrasts:
                candidate = fmean(
                    fmean(values[arm_id] for values in effects[(retriever, k)][group_id].values())
                    for k in _PAPER_TOP_K
                )
                retriever_contrasts[retriever].append(candidate - reference_mean)
        interactions[arm_id] = {
            "top_k_20_minus_5": _effect_summary(
                top_k_contrast,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "retriever_minus_bge_m3": {
                retriever: _effect_summary(
                    values,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                )
                for retriever, values in retriever_contrasts.items()
            },
        }
    return {
        "arena_role": role,
        "benchmark_id": reference.manifest.benchmarks[0].cell_id,
        "run_count": 8,
        "independent_groups": reference.independent_groups,
        "verified_plan_cells": sum(run.expected_plan_cells for run in runs),
        "verified_agent_episodes": sum(run.expected_agent_episodes for run in runs),
        "dependency_cells": cell_summaries,
        "dependency_balance": dependency_balance,
        "interactions": interactions,
        "cluster_definition": "same benchmark/split/unit replayed across all eight cells",
    }


def _formal_retriever_runtime_receipt(run: _VerifiedPaperRun) -> Mapping[str, Any]:
    expected_arm_ids = {arm.arm_id for arm in run.manifest.arms}
    required_slots = {"backend", "retriever", "downstream_predictor", "seed"}
    by_model: dict[str, dict[str, str]] = {}
    dataset_identity: str | None = None
    for record in run.records:
        cell = record.get("cell")
        result = record.get("result")
        payload = result.get("payload") if isinstance(result, Mapping) else None
        if not isinstance(cell, Mapping) or not isinstance(payload, Mapping):
            raise ArtifactError("formal retriever result lacks cell or payload")
        model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
        observed = payload.get("observed_dependency_hashes")
        if not isinstance(observed, Mapping) or set(observed) != expected_arm_ids:
            raise ArtifactError("formal retriever result lacks exact arm dependency receipts")
        parsed_by_arm: list[dict[str, str]] = []
        for raw in observed.values():
            if not isinstance(raw, Mapping):
                raise ArtifactError("formal retriever arm dependency receipt is invalid")
            parsed = {
                str(slot): str(digest)
                for slot, digest in raw.items()
                if isinstance(slot, str) and isinstance(digest, str)
            }
            if (
                set(parsed) != set(raw)
                or not required_slots <= set(parsed)
                or any(
                    len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    for digest in parsed.values()
                )
            ):
                raise ArtifactError("formal retriever dependency fingerprints are invalid")
            parsed_by_arm.append(parsed)
        first = parsed_by_arm[0]
        if any(value != first for value in parsed_by_arm[1:]):
            raise ArtifactError("formal retriever nuisance dependencies differ across arms")
        previous = by_model.setdefault(model_slot, first)
        if previous != first:
            raise ArtifactError("formal retriever dependencies drift across groups")
        current_dataset = payload.get("dataset_identity_sha256")
        if (
            not isinstance(current_dataset, str)
            or len(current_dataset) != 64
            or any(character not in "0123456789abcdef" for character in current_dataset)
        ):
            raise ArtifactError("formal retriever result lacks dataset identity SHA-256")
        if dataset_identity is None:
            dataset_identity = current_dataset
        elif dataset_identity != current_dataset:
            raise ArtifactError("formal retriever dataset identity drifts across groups/models")
    if set(by_model) != set(run.manifest.model_slots) or dataset_identity is None:
        raise ArtifactError("formal retriever receipt lacks the exact model grid")
    return {"dependencies_by_model": by_model, "dataset_identity_sha256": dataset_identity}


def _validate_formal_retriever_factorial_balance(
    receipts: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    dataset_hashes = {str(value["dataset_identity_sha256"]) for value in receipts.values()}
    if len(dataset_hashes) != 1:
        raise ArtifactError("formal retriever cells use different dataset/corpus identities")
    models = set.intersection(
        *(
            set(value["dependencies_by_model"])
            for value in receipts.values()
            if isinstance(value["dependencies_by_model"], Mapping)
        )
    )
    if not models or any(
        set(value["dependencies_by_model"]) != models for value in receipts.values()
    ):
        raise ArtifactError("formal retriever cells expose different model slots")
    varied_slots = {"retriever", "downstream_predictor"}
    summaries: dict[str, Any] = {}
    for model_slot in sorted(models):
        dependency_maps = {
            key: dict(value["dependencies_by_model"][model_slot]) for key, value in receipts.items()
        }
        slot_sets = {frozenset(value) for value in dependency_maps.values()}
        if len(slot_sets) != 1:
            raise ArtifactError("formal retriever cells expose different dependency slots")
        slots = next(iter(slot_sets))
        for slot in slots - varied_slots:
            if len({value[slot] for value in dependency_maps.values()}) != 1:
                raise ArtifactError(
                    f"formal retriever non-varied dependency {slot!r} differs across cells"
                )
        retriever_hashes = {}
        for retriever in _FORMAL_PAPER_RETRIEVERS:
            values = {dependency_maps[(retriever, top_k)]["retriever"] for top_k in _PAPER_TOP_K}
            if len(values) != 1:
                raise ArtifactError("formal retriever identity changes across top-k")
            retriever_hashes[retriever] = next(iter(values))
        if len(set(retriever_hashes.values())) != len(_FORMAL_PAPER_RETRIEVERS):
            raise ArtifactError("formal retriever identities are not distinct")
        predictor_hashes = {}
        for top_k in _PAPER_TOP_K:
            values = {
                dependency_maps[(retriever, top_k)]["downstream_predictor"]
                for retriever in _FORMAL_PAPER_RETRIEVERS
            }
            if len(values) != 1:
                raise ArtifactError("formal downstream predictor changes across retrievers")
            predictor_hashes[str(top_k)] = next(iter(values))
        if len(set(predictor_hashes.values())) != len(_PAPER_TOP_K):
            raise ArtifactError("formal top-k identities are not distinct")
        summaries[model_slot] = {
            "shared_dependency_slots": sorted(slots - varied_slots),
            "retriever_slot_hashes": retriever_hashes,
            "predictor_slot_hashes_by_top_k": predictor_hashes,
        }
    return {
        "dataset_identity_sha256": next(iter(dataset_hashes)),
        "models": summaries,
        "balance_verified": True,
    }


def validate_paper_selector_analysis_contract(
    campaign: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require the exact 14-policy, 526-group three-arena Selector analysis."""

    _validate_paper_selector_campaign(campaign)
    metadata = campaign["metadata"]
    summaries = analysis.get("policy_summaries")
    if (
        analysis.get("analysis") != "selector_multi_policy_campaign"
        or analysis.get("campaign_hash") != campaign.get("campaign_hash")
        or analysis.get("primary_metric") != "valid_success"
        or analysis.get("primary_metric") != campaign.get("primary_metric")
        or analysis.get("evaluation_split") != campaign.get("evaluation_split")
        or analysis.get("independent_groups") != 526
        or analysis.get("common_support_groups") != 526
        or not isinstance(summaries, Mapping)
        or set(summaries) != _PAPER_SELECTOR_POLICIES
        or any(
            not isinstance(summary, Mapping) or summary.get("independent_groups") != 526
            for summary in summaries.values()
        )
        or analysis.get("observed_action_oracle_used_for_training") is not False
        or analysis.get("observed_action_oracle_used_for_inference") is not False
        or analysis.get("paper_claim_authorized") is not False
    ):
        raise ValidationError("Selector analysis output differs from the frozen paper protocol")
    return {
        "analysis_protocol": PAPER_SELECTOR_PROTOCOL,
        "independent_groups": metadata["required_independent_groups"],
        "common_support_groups": 526,
        "partial_support_groups": 0,
        "policy_count": 14,
        "policy_rows": 7_364,
        "paper_claim_authorized": False,
    }


def build_paper_selector_claim_gates(
    campaign: Mapping[str, Any],
    *,
    alpha: float,
    valid_success_superiority_margin: float,
    best_fixed_noninferiority_margin: float,
    hard_harm_noninferiority_margin: float,
    safety_comparator_policy_id: str,
    minimum_intervention_coverage: float,
    maximum_intervention_coverage: float,
    maximum_mean_cost: float,
    bootstrap_samples: int,
    permutation_samples: int,
    random_seed: int,
) -> Mapping[str, Any]:
    """Freeze scientific claim gates before untouched Selector outcomes exist."""

    _validate_paper_selector_campaign(campaign)
    campaign_hash = campaign.get("campaign_hash")
    if not isinstance(campaign_hash, str) or len(campaign_hash) != 64:
        raise ValidationError("claim gates require an exact Selector campaign hash")
    base = {
        "schema_version": 1,
        "artifact": _SELECTOR_GATE_ARTIFACT,
        "stage": "frozen_before_test_outcomes",
        "campaign_manifest_hash": campaign_hash,
        "primary_policy_id": "eligibility_aware_selector",
        "no_op_policy_id": "no_op",
        "best_fixed_policy_id": "best_fixed_global",
        "safety_comparator_policy_id": safety_comparator_policy_id,
        "alpha": alpha,
        "valid_success_superiority_margin": valid_success_superiority_margin,
        "best_fixed_noninferiority_margin": best_fixed_noninferiority_margin,
        "hard_harm_noninferiority_margin": hard_harm_noninferiority_margin,
        "minimum_intervention_coverage": minimum_intervention_coverage,
        "maximum_intervention_coverage": maximum_intervention_coverage,
        "maximum_mean_cost": maximum_mean_cost,
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "random_seed": random_seed,
        "experiments_executed": False,
    }
    candidate = {**base, "gate_hash": content_hash(base)}
    return freeze_mapping(validate_paper_selector_claim_gates(campaign, candidate))


def validate_paper_selector_claim_gates(
    campaign: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate an exact campaign-bound Selector claim hierarchy."""

    _validate_paper_selector_campaign(campaign)
    campaign_hash = campaign.get("campaign_hash")
    if not isinstance(campaign_hash, str) or len(campaign_hash) != 64:
        raise ValidationError("claim gates require an exact Selector campaign hash")
    expected_keys = {
        "schema_version",
        "artifact",
        "stage",
        "campaign_manifest_hash",
        "primary_policy_id",
        "no_op_policy_id",
        "best_fixed_policy_id",
        "safety_comparator_policy_id",
        "alpha",
        "valid_success_superiority_margin",
        "best_fixed_noninferiority_margin",
        "hard_harm_noninferiority_margin",
        "minimum_intervention_coverage",
        "maximum_intervention_coverage",
        "maximum_mean_cost",
        "bootstrap_samples",
        "permutation_samples",
        "random_seed",
        "experiments_executed",
        "gate_hash",
    }
    if set(gates) != expected_keys:
        raise ValidationError("paper Selector claim-gate schema is invalid")
    if (
        gates["schema_version"] != 1
        or gates["artifact"] != _SELECTOR_GATE_ARTIFACT
        or gates["stage"] != "frozen_before_test_outcomes"
        or gates["experiments_executed"] is not False
        or gates["campaign_manifest_hash"] != campaign_hash
        or gates["primary_policy_id"] != "eligibility_aware_selector"
        or gates["no_op_policy_id"] != "no_op"
        or gates["best_fixed_policy_id"] != "best_fixed_global"
        or gates["safety_comparator_policy_id"] not in {"no_op", "best_fixed_global"}
    ):
        raise ValidationError("paper Selector claim gates differ from the frozen campaign")
    alpha = _bounded_protocol_number(gates["alpha"], lower=0.0, upper=0.5, label="alpha")
    if not math.isclose(alpha, 0.05, abs_tol=1e-12):
        raise ValidationError("paper Selector protocol fixes alpha at 0.05")
    for key in (
        "valid_success_superiority_margin",
        "best_fixed_noninferiority_margin",
        "hard_harm_noninferiority_margin",
        "minimum_intervention_coverage",
        "maximum_intervention_coverage",
    ):
        _bounded_protocol_number(gates[key], lower=0.0, upper=1.0, label=key)
    minimum_coverage = float(gates["minimum_intervention_coverage"])
    maximum_coverage = float(gates["maximum_intervention_coverage"])
    if minimum_coverage > maximum_coverage:
        raise ValidationError("paper Selector minimum coverage exceeds maximum coverage")
    _bounded_protocol_number(
        gates["maximum_mean_cost"],
        lower=0.0,
        upper=None,
        label="maximum_mean_cost",
    )
    for key, minimum in (
        ("bootstrap_samples", 100),
        ("permutation_samples", 100),
        ("random_seed", 0),
    ):
        value = gates[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValidationError(f"paper Selector {key} must be an integer >= {minimum}")
    gate_hash = gates["gate_hash"]
    if (
        not isinstance(gate_hash, str)
        or len(gate_hash) != 64
        or content_hash({key: value for key, value in gates.items() if key != "gate_hash"})
        != gate_hash
    ):
        raise ValidationError("paper Selector claim-gate hash differs from its content")
    return freeze_mapping(gates)


def analyze_paper_selector_claim_hierarchy(
    campaign: Mapping[str, Any],
    verified_results: Mapping[str, Any],
    base_analysis: Mapping[str, Any],
    claim_gates: Mapping[str, Any],
    *,
    bootstrap_samples: int | None = None,
) -> Mapping[str, Any]:
    """Run the preregistered paired Selector claim hierarchy."""

    protocol_receipt = validate_paper_selector_analysis_contract(campaign, base_analysis)
    gates = validate_paper_selector_claim_gates(campaign, claim_gates)
    if (
        verified_results.get("artifact") != "agentinputlab.selector_campaign_results"
        or verified_results.get("stage") != "verified"
        or verified_results.get("data_status") != "VERIFIED_REAL_RESULTS"
        or verified_results.get("is_synthetic") is not False
        or verified_results.get("campaign_manifest_hash") != campaign.get("campaign_hash")
    ):
        raise ValidationError(
            "paper Selector claim hierarchy requires campaign-bound verified real results"
        )
    frozen_bootstrap_samples = int(gates["bootstrap_samples"])
    if bootstrap_samples is not None and bootstrap_samples != frozen_bootstrap_samples:
        raise ValidationError(
            "paper Selector bootstrap_samples differs from the frozen claim gates"
        )
    if base_analysis.get("bootstrap_samples") != frozen_bootstrap_samples:
        raise ValidationError(
            "Selector campaign analysis bootstrap count differs from the claim gates"
        )
    if base_analysis.get("random_seed") != gates["random_seed"]:
        raise ValidationError("Selector campaign analysis seed differs from the claim gates")
    _validate_resample_arguments(
        bootstrap_samples=frozen_bootstrap_samples,
        permutation_samples=int(gates["permutation_samples"]),
        random_seed=int(gates["random_seed"]),
    )
    records = verified_results.get("records")
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValidationError("verified Selector results require records")
    by_group: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        if not isinstance(row, Mapping):
            raise ValidationError("verified Selector result row must be an object")
        group_id = row.get("group_id")
        policy_id = row.get("policy_id")
        if not isinstance(group_id, str) or not isinstance(policy_id, str):
            raise ValidationError("verified Selector result row lacks group/policy identity")
        if policy_id in by_group[group_id]:
            raise ValidationError("verified Selector results duplicate a group-policy row")
        by_group[group_id][policy_id] = row
    if len(by_group) != 526 or any(
        set(rows) != _PAPER_SELECTOR_POLICIES for rows in by_group.values()
    ):
        raise ValidationError("paper Selector claim hierarchy requires 526 complete policy blocks")

    primary_id = str(gates["primary_policy_id"])
    no_op_id = str(gates["no_op_policy_id"])
    best_id = str(gates["best_fixed_policy_id"])
    safety_id = str(gates["safety_comparator_policy_id"])
    superiority = []
    best_difference = []
    harm_difference = []
    primary_costs = []
    intervened = []
    for group_id in sorted(by_group):
        rows = by_group[group_id]
        primary = rows[primary_id]
        no_op = rows[no_op_id]
        best = rows[best_id]
        safety = rows[safety_id]
        superiority.append(float(primary["value"]) - float(no_op["value"]))
        best_difference.append(float(primary["value"]) - float(best["value"]))
        harm_difference.append(float(bool(primary["hard_harm"])) - float(bool(safety["hard_harm"])))
        primary_costs.append(float(primary["cost"]))
        intervened.append(1.0 if primary["intervened"] else 0.0)

    rng = random.Random(int(gates["random_seed"]))
    alpha = float(gates["alpha"])
    superiority_margin = float(gates["valid_success_superiority_margin"])
    best_margin = float(gates["best_fixed_noninferiority_margin"])
    harm_margin = float(gates["hard_harm_noninferiority_margin"])
    superiority_p = _one_sided_sign_flip_p_value(
        [value - superiority_margin for value in superiority],
        samples=int(gates["permutation_samples"]),
        rng=rng,
    )
    best_p = _one_sided_sign_flip_p_value(
        [value + best_margin for value in best_difference],
        samples=int(gates["permutation_samples"]),
        rng=rng,
    )
    harm_p = _one_sided_sign_flip_p_value(
        [harm_margin - value for value in harm_difference],
        samples=int(gates["permutation_samples"]),
        rng=rng,
    )
    superiority_interval = _cluster_bootstrap_interval(
        superiority,
        samples=frozen_bootstrap_samples,
        rng=rng,
    )
    best_interval = _cluster_bootstrap_interval(
        best_difference,
        samples=frozen_bootstrap_samples,
        rng=rng,
    )
    harm_interval = _cluster_bootstrap_interval(
        harm_difference,
        samples=frozen_bootstrap_samples,
        rng=rng,
    )
    coverage = fmean(intervened)
    mean_cost = fmean(primary_costs)
    raw_passes = (
        superiority_p <= alpha and superiority_interval[0] > superiority_margin,
        best_p <= alpha and best_interval[0] > -best_margin,
        harm_p <= alpha and harm_interval[1] < harm_margin,
        (
            float(gates["minimum_intervention_coverage"])
            <= coverage
            <= float(gates["maximum_intervention_coverage"])
            and mean_cost <= float(gates["maximum_mean_cost"])
        ),
    )
    reached = True
    formal_passes: list[bool] = []
    formally_tested: list[bool] = []
    for passed in raw_passes:
        formally_tested.append(reached)
        formal_passes.append(reached and passed)
        reached = reached and passed
    steps = (
        {
            "step": 1,
            "claim": "valid_success_superiority_to_no_op",
            "estimate": fmean(superiority),
            "cluster_bootstrap_95ci": list(superiority_interval),
            "margin": superiority_margin,
            "one_sided_sign_flip_p_value": superiority_p,
        },
        {
            "step": 2,
            "claim": "noninferiority_to_calibration_frozen_best_fixed",
            "estimate": fmean(best_difference),
            "cluster_bootstrap_95ci": list(best_interval),
            "margin": best_margin,
            "one_sided_sign_flip_p_value": best_p,
        },
        {
            "step": 3,
            "claim": f"hard_harm_noninferiority_to_{safety_id}",
            "estimate": fmean(harm_difference),
            "cluster_bootstrap_95ci": list(harm_interval),
            "margin": harm_margin,
            "one_sided_sign_flip_p_value": harm_p,
        },
        {
            "step": 4,
            "claim": "frozen_cost_and_coverage_constraint",
            "intervention_coverage": coverage,
            "minimum_intervention_coverage": gates["minimum_intervention_coverage"],
            "maximum_intervention_coverage": gates["maximum_intervention_coverage"],
            "mean_cost": mean_cost,
            "maximum_mean_cost": gates["maximum_mean_cost"],
        },
    )
    return freeze_mapping(
        {
            "schema_version": 1,
            "analysis": "paper_selector_hierarchical_claim_gate",
            "analysis_protocol": PAPER_SELECTOR_PROTOCOL,
            "campaign_hash": campaign.get("campaign_hash"),
            "claim_gate_hash": gates["gate_hash"],
            "independent_groups": protocol_receipt["independent_groups"],
            "alpha": alpha,
            "bootstrap_samples": frozen_bootstrap_samples,
            "permutation_samples": gates["permutation_samples"],
            "random_seed": gates["random_seed"],
            "steps": tuple(
                {
                    **step,
                    "formally_tested": formally_tested[index],
                    "passed_if_reached": raw_passes[index],
                    "hierarchical_pass": formal_passes[index],
                }
                for index, step in enumerate(steps)
            ),
            "positive_selector_claim": all(formal_passes),
            "observed_action_oracle_used_for_inference": False,
            "paper_claim_authorized": False,
        }
    )


def _validate_paper_selector_campaign(campaign: Mapping[str, Any]) -> None:
    if campaign.get("campaign_kind") != "baseline":
        raise ValidationError("paper Selector protocol requires the baseline campaign")
    if campaign.get("evaluation_split") != "untouched_test":
        raise ValidationError("paper Selector protocol requires the untouched test split")
    policies = campaign.get("policies")
    if not isinstance(policies, Sequence) or isinstance(policies, str | bytes):
        raise ValidationError("paper Selector campaign requires a policies array")
    policy_ids = {item.get("policy_id") for item in policies if isinstance(item, Mapping)}
    if len(policies) != 14 or policy_ids != _PAPER_SELECTOR_POLICIES:
        raise ValidationError("paper Selector protocol requires the exact 14-policy roster")
    metadata = campaign.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("required_independent_groups") != 526
        or metadata.get("minimum_common_support_groups") != 526
    ):
        raise ValidationError(
            "paper Selector protocol requires exactly 526 three-arena common-support groups"
        )
    if campaign.get("campaign_hash") is not None:
        campaign_hash = campaign["campaign_hash"]
        if not isinstance(campaign_hash, str) or len(campaign_hash) != 64:
            raise ValidationError("paper Selector campaign hash must be SHA-256")


def validate_paper_run_contract(
    run_dir: str | Path,
    *,
    protocol: str,
) -> Mapping[str, Any]:
    """Validate one paper run and return its census receipt without analyzing it."""

    expected = {
        PAPER_FAMILY_NATIVE_PROTOCOL: _FAMILY_NATIVE_KIND,
        PAPER_COMMON_SUPPORT_PROTOCOL: _COMMON_SUPPORT_KIND,
        PAPER_RETRIEVER_FACTORIAL_PROTOCOL: _RETRIEVER_FACTORIAL_KIND,
    }.get(protocol)
    if expected is None:
        raise ValidationError(f"unsupported paper run protocol: {protocol!r}")
    if expected == _RETRIEVER_FACTORIAL_KIND:
        manifest = _load_manifest(Path(run_dir).expanduser().resolve())
        census = manifest.metadata.get("experiment_census")
        actual_kind = census.get("campaign_kind") if isinstance(census, Mapping) else None
        if actual_kind in {
            _RETRIEVER_FACTORIAL_PRIMARY_KIND,
            _RETRIEVER_FACTORIAL_AUXILIARY_KIND,
        }:
            expected = str(actual_kind)
    run = _load_and_validate_paper_run(run_dir, expected_kind=expected)
    if expected == _FAMILY_NATIVE_KIND:
        _family_native_arm_contract(run.manifest)
    elif expected == _COMMON_SUPPORT_KIND:
        _common_support_arm_contract(run.manifest)
    else:
        _retriever_arm_contract(run.manifest)
        _retriever_dependency_cell(
            run.manifest,
            formal=expected
            in {_RETRIEVER_FACTORIAL_PRIMARY_KIND, _RETRIEVER_FACTORIAL_AUXILIARY_KIND},
        )
    return {
        "analysis_protocol": protocol,
        "study_id": run.manifest.study_id,
        "campaign_kind": run.campaign_kind,
        "independent_groups": run.independent_groups,
        "verified_plan_cells": run.expected_plan_cells,
        "verified_agent_episodes": run.expected_agent_episodes,
        "paper_claim_authorized": False,
    }


def _load_and_validate_paper_run(
    run_dir: str | Path,
    *,
    expected_kind: str,
) -> _VerifiedPaperRun:
    root = Path(run_dir).expanduser().resolve()
    manifest = _load_manifest(root)
    records = load_verified_cell_results(root)
    census = manifest.metadata.get("experiment_census")
    if not isinstance(census, Mapping) or census.get("protocol") != _CENSUS_PROTOCOL:
        raise ArtifactError("paper run lacks the frozen experiment census protocol")
    if census.get("campaign_kind") != expected_kind:
        raise ArtifactError(
            f"paper run campaign kind is {census.get('campaign_kind')!r}, "
            f"expected {expected_kind!r}"
        )
    expected = _paper_census_expectation(expected_kind)
    for field, value in expected.items():
        if census.get(field) != value:
            raise ArtifactError(
                f"paper {expected_kind} census field {field} is "
                f"{census.get(field)!r}, expected {value!r}"
            )
    if manifest.stage is not ExperimentStage.CONFIRMATORY:
        raise ArtifactError("paper primary/factorial analyses require confirmatory-stage runs")
    retriever_kinds = {
        _RETRIEVER_FACTORIAL_KIND,
        _RETRIEVER_FACTORIAL_PRIMARY_KIND,
        _RETRIEVER_FACTORIAL_AUXILIARY_KIND,
    }
    profile = (
        ProfileKind.FACTORIAL_ADAPTER
        if expected_kind in retriever_kinds
        else ProfileKind.CONTROLLED_ADAPTER
    )
    if manifest.estimand_profile_kind is not profile:
        raise ArtifactError(f"paper {expected_kind} run has the wrong estimand profile")
    expected_arm_ids = census.get("expected_arm_ids")
    expected_models = census.get("expected_model_slots")
    expected_seeds = census.get("expected_seeds")
    if (
        not isinstance(expected_arm_ids, Sequence)
        or isinstance(expected_arm_ids, str | bytes)
        or tuple(expected_arm_ids) != tuple(arm.arm_id for arm in manifest.arms)
        or not isinstance(expected_models, Sequence)
        or isinstance(expected_models, str | bytes)
        or tuple(expected_models) != manifest.model_slots
        or not isinstance(expected_seeds, Sequence)
        or isinstance(expected_seeds, str | bytes)
        or tuple(expected_seeds) != manifest.seeds
    ):
        raise ArtifactError("paper run census arm/model/seed identities differ from its manifest")

    expected_plan_cells = int(expected["expected_plan_cells"])
    expected_agent_episodes = int(expected["expected_agent_episodes"])
    independent_groups = int(expected["independent_group_target"])
    if len(records) != expected_plan_cells:
        raise ArtifactError(
            f"paper run has {len(records)} verified cells, expected {expected_plan_cells}"
        )
    clusters: dict[str, set[tuple[str, int]]] = defaultdict(set)
    episodes = 0
    for record in records:
        cell = record.get("cell")
        if not isinstance(cell, Mapping):
            raise ArtifactError("paper run record lacks an experiment cell")
        cluster_id = _cluster_id(cell)
        model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
        seed = cell.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ArtifactError("paper experiment cell seed must be an integer")
        replicate = (model_slot, seed)
        if replicate in clusters[cluster_id]:
            raise ArtifactError("paper run duplicates a group/model/seed cell")
        clusters[cluster_id].add(replicate)
        arm_ids = cell.get("arm_ids")
        if not isinstance(arm_ids, Sequence) or isinstance(arm_ids, str | bytes):
            raise ArtifactError("paper experiment cell lacks an arm inventory")
        episodes += len(arm_ids)
    expected_replicates = {
        (model_slot, seed) for model_slot in manifest.model_slots for seed in manifest.seeds
    }
    if len(clusters) != independent_groups:
        raise ArtifactError(
            f"paper run has {len(clusters)} independent groups, expected {independent_groups}"
        )
    if any(replicates != expected_replicates for replicates in clusters.values()):
        raise ArtifactError("paper run does not contain the exact model/seed grid per group")
    if episodes != expected_agent_episodes:
        raise ArtifactError(
            f"paper run has {episodes} arm episodes, expected {expected_agent_episodes}"
        )
    return _VerifiedPaperRun(
        root=root,
        manifest=manifest,
        records=records,
        campaign_kind=expected_kind,
        independent_groups=independent_groups,
        expected_plan_cells=expected_plan_cells,
        expected_agent_episodes=expected_agent_episodes,
        cluster_ids=frozenset(clusters),
    )


def _paper_census_expectation(kind: str) -> Mapping[str, int]:
    if kind == _FAMILY_NATIVE_KIND:
        return {
            "independent_group_target": 600,
            "expected_plan_cells": 1_200,
            "expected_agent_episodes": 3_600,
        }
    if kind == _COMMON_SUPPORT_KIND:
        return {
            "independent_group_target": 526,
            "expected_plan_cells": 1_052,
            "expected_agent_episodes": 6_312,
        }
    if kind == _RETRIEVER_FACTORIAL_KIND:
        return {
            "independent_group_target": 240,
            "expected_plan_cells": 240,
            "expected_agent_episodes": 1_200,
        }
    if kind == _RETRIEVER_FACTORIAL_PRIMARY_KIND:
        return {
            "independent_group_target": 97,
            "expected_plan_cells": 194,
            "expected_agent_episodes": 970,
        }
    if kind == _RETRIEVER_FACTORIAL_AUXILIARY_KIND:
        return {
            "independent_group_target": 240,
            "expected_plan_cells": 480,
            "expected_agent_episodes": 2_400,
        }
    raise ValidationError(f"unknown paper census kind: {kind!r}")


def _family_native_arm_contract(manifest: ExperimentManifest) -> SemanticFamily:
    census = manifest.metadata["experiment_census"]
    raw_family = census.get("family")
    try:
        family = SemanticFamily(raw_family)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("family-native census requires one core family") from exc
    if family not in _primary_families():
        raise ArtifactError("family-native census family is not one of the five core families")
    _family_native_arm_ids(manifest, family)
    if len(manifest.model_slots) != 2 or len(manifest.seeds) != 1:
        raise ArtifactError("family-native paper run requires two models and one seed")
    return family


def _family_native_arm_ids(
    manifest: ExperimentManifest,
    family: SemanticFamily,
) -> tuple[str, tuple[str, str]]:
    baselines = tuple(arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.NO_OP)
    treatments = tuple(arm.arm_id for arm in manifest.arms if arm.family is family)
    if len(manifest.arms) != 3 or len(baselines) != 1 or len(treatments) != 2:
        raise ArtifactError(
            "family-native paper run requires exactly NO_OP plus two same-family arms"
        )
    return baselines[0], (treatments[0], treatments[1])


def _common_support_arm_contract(
    manifest: ExperimentManifest,
) -> tuple[str, Mapping[SemanticFamily, str]]:
    baselines = tuple(arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.NO_OP)
    by_family = {
        family: tuple(arm.arm_id for arm in manifest.arms if arm.family is family)
        for family in _primary_families()
    }
    if (
        len(manifest.arms) != 6
        or len(baselines) != 1
        or any(len(arm_ids) != 1 for arm_ids in by_family.values())
        or len(manifest.model_slots) != 2
        or len(manifest.seeds) != 1
    ):
        raise ArtifactError(
            "common-support paper run requires NO_OP, one arm per family, two models, and one seed"
        )
    return baselines[0], {family: arm_ids[0] for family, arm_ids in by_family.items()}


def _retriever_arm_contract(
    manifest: ExperimentManifest,
) -> tuple[str, tuple[str, str, str, str]]:
    baselines = tuple(arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.NO_OP)
    reformulate = tuple(
        arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.REFORMULATE
    )
    acquire = tuple(arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.ACQUIRE)
    campaign_kind = manifest.metadata.get("experiment_census", {}).get("campaign_kind")
    expected_model_count = (
        2
        if campaign_kind in {_RETRIEVER_FACTORIAL_PRIMARY_KIND, _RETRIEVER_FACTORIAL_AUXILIARY_KIND}
        else 1
    )
    if (
        len(manifest.arms) != 5
        or len(baselines) != 1
        or len(reformulate) != 2
        or len(acquire) != 2
        or len(manifest.model_slots) != expected_model_count
        or len(manifest.seeds) != 1
    ):
        raise ArtifactError(
            "retriever-factorial paper cell requires NO_OP, two reformulation arms, "
            f"two acquisition arms, {expected_model_count} model(s), and one seed"
        )
    return baselines[0], (reformulate[0], reformulate[1], acquire[0], acquire[1])


def _retriever_dependency_cell(
    manifest: ExperimentManifest,
    *,
    formal: bool = False,
) -> tuple[str, int]:
    dependency = manifest.metadata.get("dependency_cell")
    if not isinstance(dependency, Mapping):
        raise ArtifactError("retriever-factorial run lacks dependency_cell metadata")
    retriever = dependency.get("retriever_id")
    top_k = dependency.get("top_k")
    allowed = _FORMAL_PAPER_RETRIEVERS if formal else _PAPER_RETRIEVERS
    if retriever not in allowed or top_k not in _PAPER_TOP_K:
        grid = "4x2" if formal else "2x2"
        raise ArtifactError(
            f"retriever-factorial dependency cell is outside the frozen {grid} grid"
        )
    slugs = {
        "bm25": "bm25",
        "intfloat/e5-large-v2": "e5_large_v2",
        "BAAI/bge-m3": "bge_m3",
        "Qwen/Qwen3-Embedding-0.6B": "qwen3_embedding",
    }
    expected_binding = slugs[str(retriever)] + f":k{top_k}"
    if any(arm.parameters.get("dependency_cell") != expected_binding for arm in manifest.arms):
        raise ArtifactError(
            "retriever-factorial arm dependency binding differs from its declared cell"
        )
    return str(retriever), int(top_k)


def _retriever_runtime_receipt(run: _VerifiedPaperRun) -> Mapping[str, Any]:
    """Collapse one cell's observed runtime identities after strict equality checks."""

    expected_arm_ids = {arm.arm_id for arm in run.manifest.arms}
    reference_dependencies: dict[str, str] | None = None
    dataset_identity: str | None = None
    required_slots = {"backend", "retriever", "downstream_predictor", "seed"}
    for record in run.records:
        result = record.get("result")
        payload = result.get("payload") if isinstance(result, Mapping) else None
        if not isinstance(payload, Mapping):
            raise ArtifactError("retriever-factorial result lacks a payload")
        observed = payload.get("observed_dependency_hashes")
        if not isinstance(observed, Mapping) or set(observed) != expected_arm_ids:
            raise ArtifactError(
                "retriever-factorial dependency receipt differs from the arm inventory"
            )
        by_arm: dict[str, dict[str, str]] = {}
        for arm_id, raw_dependencies in observed.items():
            if not isinstance(arm_id, str) or not isinstance(raw_dependencies, Mapping):
                raise ArtifactError("retriever-factorial dependency receipt is malformed")
            parsed: dict[str, str] = {}
            for slot, digest in raw_dependencies.items():
                if (
                    not isinstance(slot, str)
                    or not slot
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ArtifactError(
                        "retriever-factorial dependency fingerprints must be SHA-256 mappings"
                    )
                parsed[slot] = digest
            if not parsed:
                raise ArtifactError("retriever-factorial dependency receipt is empty")
            by_arm[arm_id] = parsed
        first = next(iter(by_arm.values()))
        if any(dependencies != first for dependencies in by_arm.values()):
            raise ArtifactError(
                "retriever-factorial nuisance dependencies differ across arms within a cell"
            )
        if not required_slots.issubset(first):
            raise ArtifactError(
                "retriever-factorial dependency receipt lacks backend, retriever, "
                "predictor, or seed identity"
            )
        if reference_dependencies is None:
            reference_dependencies = first
        elif first != reference_dependencies:
            raise ArtifactError(
                "retriever-factorial runtime dependencies drift across groups within a cell"
            )
        current_dataset = payload.get("dataset_identity_sha256")
        if (
            not isinstance(current_dataset, str)
            or len(current_dataset) != 64
            or any(character not in "0123456789abcdef" for character in current_dataset)
        ):
            raise ArtifactError("retriever-factorial result lacks dataset identity SHA-256")
        if dataset_identity is None:
            dataset_identity = current_dataset
        elif current_dataset != dataset_identity:
            raise ArtifactError(
                "retriever-factorial dataset identity drifts across groups within a cell"
            )
    if reference_dependencies is None or dataset_identity is None:  # pragma: no cover
        raise ArtifactError("retriever-factorial dependency receipt has no records")
    return {
        "dependencies": reference_dependencies,
        "dataset_identity_sha256": dataset_identity,
    }


def _validate_retriever_factorial_balance(
    receipts: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Verify all non-factorial runtime identities across the four dependency cells."""

    dependency_maps = {key: dict(receipt["dependencies"]) for key, receipt in receipts.items()}
    key_sets = {frozenset(dependencies) for dependencies in dependency_maps.values()}
    if len(key_sets) != 1:
        raise ArtifactError("retriever-factorial cells expose different dependency slots")
    slots = next(iter(key_sets))
    dataset_hashes = {str(receipt["dataset_identity_sha256"]) for receipt in receipts.values()}
    if len(dataset_hashes) != 1:
        raise ArtifactError("retriever-factorial cells use different dataset/corpus identities")
    varied_slots = {"retriever", "downstream_predictor"}
    for slot in slots - varied_slots:
        if len({dependencies[slot] for dependencies in dependency_maps.values()}) != 1:
            raise ArtifactError(
                f"retriever-factorial non-varied dependency {slot!r} differs across cells"
            )
    retriever_hashes: dict[str, str] = {}
    for retriever in _PAPER_RETRIEVERS:
        values = {dependency_maps[(retriever, top_k)]["retriever"] for top_k in _PAPER_TOP_K}
        if len(values) != 1:
            raise ArtifactError(
                "retriever identity changes across top-k cells for the same retriever"
            )
        retriever_hashes[retriever] = next(iter(values))
    if len(set(retriever_hashes.values())) != len(_PAPER_RETRIEVERS):
        raise ArtifactError("the two declared retrievers resolve to the same runtime identity")
    predictor_hashes: dict[int, str] = {}
    for top_k in _PAPER_TOP_K:
        values = {
            dependency_maps[(retriever, top_k)]["downstream_predictor"]
            for retriever in _PAPER_RETRIEVERS
        }
        if len(values) != 1:
            raise ArtifactError(
                "downstream predictor identity changes across retrievers at fixed top-k"
            )
        predictor_hashes[top_k] = next(iter(values))
    if len(set(predictor_hashes.values())) != len(_PAPER_TOP_K):
        raise ArtifactError("the two declared top-k cells resolve to the same predictor identity")
    return {
        "dataset_identity_sha256": next(iter(dataset_hashes)),
        "shared_dependency_slots": sorted(slots - varied_slots),
        "retriever_slot_hashes": retriever_hashes,
        "predictor_slot_hashes_by_top_k": {
            str(top_k): digest for top_k, digest in predictor_hashes.items()
        },
        "balance_verified": True,
    }


def _effect_summary(
    values: Sequence[float],
    *,
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    return {
        "cluster_count": len(values),
        "mean_difference": fmean(values),
        "cluster_bootstrap_95ci": list(
            _cluster_bootstrap_interval(values, samples=bootstrap_samples, rng=rng)
        ),
    }


def _one_sided_sign_flip_p_value(
    shifted_effects: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> float:
    if len(shifted_effects) < 2:
        raise ArtifactError("paired sign-flip test requires at least two groups")
    observed = fmean(shifted_effects)
    exceedances = 0
    for _ in range(samples):
        permuted = fmean(value if rng.random() < 0.5 else -value for value in shifted_effects)
        if permuted >= observed - 1e-15:
            exceedances += 1
    return (exceedances + 1.0) / (samples + 1.0)


def _bounded_protocol_number(
    value: Any,
    *,
    lower: float,
    upper: float | None,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < lower
        or (upper is not None and value > upper)
    ):
        suffix = f" and <= {upper}" if upper is not None else ""
        raise ValidationError(f"paper Selector {label} must be finite, >= {lower}{suffix}")
    return float(value)


def _run_roots(run_dirs: Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(run_dirs, str | bytes):
        raise ValidationError("run_dirs must be a sequence of paths")
    roots = tuple(Path(path).expanduser().resolve() for path in run_dirs)
    if not roots:
        raise ValidationError("at least one run directory is required")
    if len(roots) != len(set(roots)):
        raise ValidationError("run directories must be unique")
    return roots


__all__ = [
    "LEGACY_CONFIRMATORY_PROTOCOL",
    "PAPER_COMMON_SUPPORT_PROTOCOL",
    "PAPER_FAMILY_NATIVE_PROTOCOL",
    "PAPER_RETRIEVER_FACTORIAL_PROTOCOL",
    "PAPER_SELECTOR_PROTOCOL",
    "analyze_confirmatory_protocol",
    "analyze_paper_common_support",
    "analyze_paper_dual_arena_retriever_factorial",
    "analyze_paper_family_confirmation",
    "analyze_paper_retriever_factorial",
    "analyze_paper_selector_claim_hierarchy",
    "build_paper_selector_claim_gates",
    "infer_paper_run_protocol",
    "validate_paper_run_contract",
    "validate_paper_selector_analysis_contract",
    "validate_paper_selector_claim_gates",
]
