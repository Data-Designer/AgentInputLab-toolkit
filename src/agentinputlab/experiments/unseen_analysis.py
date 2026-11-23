"""Fail-closed analysis for the stratified unseen-:math:`\\tau^3` transfer.

The unseen transfer is not an all-actions benchmark.  Each frozen stratum
contains ``NO_OP`` plus only the families that were declared eligible before
outcomes existed.  This module therefore reports family effects only on their
observed legal support.  It never fabricates missing counterfactuals or pools
them into an all-family ranking.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import MappingProxyType
from typing import Any

from agentinputlab.benchmarks import ATOMIC_EXTERNAL_PAIR_CONTRACT
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .analysis import (
    _arm_metrics,
    _cluster_bootstrap_interval,
    _validate_resample_arguments,
    load_verified_cell_results,
)
from .artifacts import verify_experiment_run_directory
from .loader import load_experiment_manifest
from .schema import ExperimentManifest, ExperimentStage
from .stateful_analysis import validate_external_arena_record_receipts

PAPER_UNSEEN_TAU3_PROTOCOL = "paper-unseen-tau3-transfer-v1"
UNSEEN_TAU3_ELIGIBILITY_PROTOCOL = "agentinputlab.unseen-tau3-eligibility.v1"
UNSEEN_TAU3_SELECTOR_DECISIONS_ARTIFACT = "agentinputlab.unseen_tau3_selector_decisions"

_TAU3_REVISION = "cf71a8070269883e38a365ffa85f78f46844c1f4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GROUP_SHA256 = re.compile(r"^group_[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_CORE_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)
_VALID_SUCCESS = "valid_success"


@dataclass(frozen=True, slots=True)
class _UnseenRun:
    root: Path
    manifest: ExperimentManifest
    records: tuple[Mapping[str, Any], ...]
    stratum_id: str
    eligible_families: tuple[SemanticFamily, ...]
    group_count: int
    expected_episodes: int
    run_identity_hash: str
    manifest_hash: str
    plan_hash: str


def validate_unseen_tau3_analysis_plan(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact outcome-blind 200-group eligibility plan."""

    if not isinstance(value, Mapping):
        raise ValidationError("unseen tau3 eligibility plan must be an object")
    raw = dict(value)
    supplied_hash = raw.pop("plan_hash", None)
    expected_keys = {
        "schema_version",
        "protocol",
        "source_revision",
        "total_groups",
        "model_slot",
        "seed",
        "strata",
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != 1
        or raw.get("protocol") != UNSEEN_TAU3_ELIGIBILITY_PROTOCOL
        or raw.get("source_revision") != _TAU3_REVISION
        or raw.get("total_groups") != 200
    ):
        raise ValidationError("unseen tau3 eligibility plan identity or census is invalid")
    model_slot = _identifier(raw["model_slot"], label="unseen tau3 model slot")
    seed = _nonnegative_integer(raw["seed"], label="unseen tau3 seed")
    raw_strata = _sequence(raw["strata"], label="unseen tau3 strata")
    if not raw_strata:
        raise ValidationError("unseen tau3 eligibility plan requires strata")
    family_order = {family.value: index for index, family in enumerate(_CORE_FAMILIES)}
    strata: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_groups = 0
    total_episodes = 0
    for index, item in enumerate(raw_strata):
        if not isinstance(item, Mapping) or set(item) != {
            "stratum_id",
            "eligible_families",
            "group_count",
            "unit_inventory_env",
            "unit_inventory_sha256_env",
        }:
            raise ValidationError(f"unseen tau3 stratum {index} fields are invalid")
        stratum_id = _identifier(
            item["stratum_id"],
            label=f"unseen tau3 stratum {index} ID",
        )
        if stratum_id in seen:
            raise ValidationError("unseen tau3 stratum IDs must be unique")
        seen.add(stratum_id)
        families_raw = _sequence(
            item["eligible_families"],
            label=f"unseen tau3 stratum {stratum_id} families",
        )
        try:
            families = tuple(SemanticFamily(family) for family in families_raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError("unseen tau3 eligibility contains an unknown family") from exc
        if (
            not families
            or len(families) != len(set(families))
            or any(family not in _CORE_FAMILIES for family in families)
            or list(families_raw) != sorted(families_raw, key=lambda family: family_order[family])
        ):
            raise ValidationError(
                "unseen tau3 eligible families must be canonical unique core families"
            )
        group_count = _positive_integer(
            item["group_count"],
            label=f"unseen tau3 stratum {stratum_id} group count",
        )
        if group_count < 2:
            raise ValidationError(
                "unseen tau3 strata require at least two independent groups "
                "for a stratum-level uncertainty interval"
            )
        for field in ("unit_inventory_env", "unit_inventory_sha256_env"):
            text = item[field]
            if not isinstance(text, str) or _ENVIRONMENT_NAME.fullmatch(text) is None:
                raise ValidationError(f"unseen tau3 {field} is invalid")
        total_groups += group_count
        total_episodes += group_count * (1 + len(families))
        strata.append(
            {
                "stratum_id": stratum_id,
                "eligible_families": [family.value for family in families],
                "group_count": group_count,
                "unit_inventory_env": item["unit_inventory_env"],
                "unit_inventory_sha256_env": item["unit_inventory_sha256_env"],
            }
        )
    if total_groups != 200:
        raise ValidationError("unseen tau3 strata must contain exactly 200 groups")
    if total_episodes > 1_200:
        raise ValidationError("unseen tau3 plan exceeds the 1,200-episode ceiling")
    base = {
        "schema_version": 1,
        "protocol": UNSEEN_TAU3_ELIGIBILITY_PROTOCOL,
        "source_revision": _TAU3_REVISION,
        "total_groups": 200,
        "model_slot": model_slot,
        "seed": seed,
        "strata": strata,
    }
    plan_hash = content_hash(base)
    if supplied_hash is not None and supplied_hash != plan_hash:
        raise ValidationError("unseen tau3 eligibility plan hash differs from its content")
    return freeze_mapping(
        {
            **base,
            "planned_agent_episodes": total_episodes,
            "maximum_agent_episodes": 1_200,
            "plan_hash": plan_hash,
        }
    )


def validate_unseen_tau3_selector_decisions(
    value: Mapping[str, Any],
    *,
    eligibility_plan_hash: str,
    representative_manifest_hash: str,
) -> Mapping[str, Any]:
    """Validate an optional outcome-blind one-action-per-group policy artifact."""

    expected_keys = {
        "schema_version",
        "artifact",
        "stage",
        "selector_id",
        "selector_revision",
        "selector_artifact_sha256",
        "eligibility_plan_hash",
        "representative_manifest_hash",
        "records",
        "outcomes_accessed",
        "other_arm_outcomes_accessed",
        "experiments_executed",
        "decision_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValidationError("unseen tau3 selector decision schema is invalid")
    data = dict(value)
    if (
        data["schema_version"] != 1
        or data["artifact"] != UNSEEN_TAU3_SELECTOR_DECISIONS_ARTIFACT
        or data["stage"] != "frozen_before_outcomes"
        or data["outcomes_accessed"] is not False
        or data["other_arm_outcomes_accessed"] is not False
        or data["experiments_executed"] is not False
        or data["eligibility_plan_hash"] != eligibility_plan_hash
        or data["representative_manifest_hash"] != representative_manifest_hash
    ):
        raise ValidationError(
            "unseen tau3 selector decisions violate their frozen outcome-blind binding"
        )
    _identifier(data["selector_id"], label="unseen tau3 selector ID")
    _identifier(data["selector_revision"], label="unseen tau3 selector revision")
    _sha256(
        data["selector_artifact_sha256"],
        label="unseen tau3 selector artifact",
    )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(_sequence(data["records"], label="unseen tau3 selector records")):
        if not isinstance(item, Mapping) or set(item) != {
            "group_sha256",
            "selected_family",
            "selected_release_id",
        }:
            raise ValidationError(f"unseen tau3 selector decision {index} fields are invalid")
        group_id = item["group_sha256"]
        if (
            not isinstance(group_id, str)
            or _GROUP_SHA256.fullmatch(group_id) is None
            or group_id in seen
        ):
            raise ValidationError("unseen tau3 selector decisions require unique anonymized groups")
        seen.add(group_id)
        try:
            family = SemanticFamily(item["selected_family"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("unseen tau3 selector selected an unknown family") from exc
        if family not in {*_CORE_FAMILIES, SemanticFamily.NO_OP}:
            raise ValidationError("unseen tau3 selector selected an unsupported family")
        release_id = _identifier(
            item["selected_release_id"],
            label="unseen tau3 selector release",
        )
        records.append(
            {
                "group_sha256": group_id,
                "selected_family": family.value,
                "selected_release_id": release_id,
            }
        )
    if len(records) != 200:
        raise ValidationError("unseen tau3 selector decisions require exactly 200 groups")
    _sha256(data["decision_hash"], label="unseen tau3 selector decision hash")
    if (
        content_hash({key: item for key, item in data.items() if key != "decision_hash"})
        != data["decision_hash"]
    ):
        raise ValidationError("unseen tau3 selector decision hash differs from content")
    return freeze_mapping({**data, "records": tuple(records)})


def analyze_unseen_tau3_transfer(
    run_dirs: Sequence[str | Path],
    *,
    eligibility_plan: Mapping[str, Any],
    representatives: Mapping[str, Any],
    selector_decisions: Mapping[str, Any] | None = None,
    metric: str = _VALID_SUCCESS,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Analyze exact legal support without imputing unavailable family outcomes."""

    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=1,
        random_seed=random_seed,
    )
    if metric != _VALID_SUCCESS:
        raise ValidationError(
            "paper unseen tau3 transfer fixes the comparable metric to valid_success"
        )
    plan = validate_unseen_tau3_analysis_plan(eligibility_plan)
    representative_hash, releases = _validated_representatives(representatives)
    roots = tuple(Path(path).expanduser().resolve() for path in run_dirs)
    if len(roots) != len(plan["strata"]) or len(set(roots)) != len(roots):
        raise ValidationError(
            "unseen tau3 transfer requires exactly one unique run per frozen stratum"
        )
    runs = tuple(
        _load_unseen_run(
            root,
            plan=plan,
            representative_manifest_hash=representative_hash,
            releases=releases,
        )
        for root in roots
    )
    by_stratum: dict[str, _UnseenRun] = {}
    for run in runs:
        if run.stratum_id in by_stratum:
            raise ArtifactError("unseen tau3 transfer contains a duplicate stratum run")
        by_stratum[run.stratum_id] = run
    expected_strata = {stratum["stratum_id"] for stratum in plan["strata"]}
    if set(by_stratum) != expected_strata:
        raise ArtifactError("unseen tau3 transfer does not cover the frozen strata")

    group_rows: dict[str, dict[str, Any]] = {}
    dependency_identity: Mapping[str, str] | None = None
    scorer_identity: tuple[str, str, str] | None = None
    total_episodes = 0
    for stratum in plan["strata"]:
        run = by_stratum[stratum["stratum_id"]]
        baseline_id, family_arms = _unseen_arm_contract(
            run.manifest,
            expected_families=run.eligible_families,
            representative_manifest_hash=representative_hash,
            releases=releases,
        )
        for record in run.records:
            receipt = _validate_unseen_external_receipts(
                run.manifest,
                record,
                baseline_arm_id=baseline_id,
                family_arms=family_arms,
                metric=metric,
            )
            if dependency_identity is None:
                dependency_identity = receipt["dependencies"]
            elif receipt["dependencies"] != dependency_identity:
                raise ArtifactError("unseen tau3 dependencies differ across groups or strata")
            current_scorer = (
                receipt["scorer_id"],
                receipt["scorer_revision"],
                receipt["outcome_contract_sha256"],
            )
            if scorer_identity is None:
                scorer_identity = current_scorer
            elif current_scorer != scorer_identity:
                raise ArtifactError(
                    "unseen tau3 scorer or valid-success contract differs across strata"
                )
            group_id = receipt["group_sha256"]
            if group_id in group_rows:
                raise ArtifactError("unseen tau3 groups overlap across frozen eligibility strata")
            values = _arm_metrics(
                record,
                metric=metric,
                algorithm_failure_value=algorithm_failure_value,
            )
            if values is None:  # pragma: no cover - verified run invariant
                raise ArtifactError("unseen tau3 result cell is incomplete")
            baseline = values[baseline_id]
            group_rows[group_id] = {
                "group_sha256": group_id,
                "stratum_id": run.stratum_id,
                "eligible_families": tuple(family.value for family in run.eligible_families),
                "values": {
                    SemanticFamily.NO_OP.value: baseline,
                    **{family.value: values[arm_id] for family, arm_id in family_arms.items()},
                },
                "release_by_family": {
                    SemanticFamily.NO_OP.value: "baseline.no_op",
                    **{family.value: releases[family] for family in run.eligible_families},
                },
            }
            total_episodes += len(run.manifest.arms)
    if len(group_rows) != 200:
        raise ArtifactError("unseen tau3 transfer requires exactly 200 independent groups")
    if total_episodes != plan["planned_agent_episodes"] or total_episodes > 1_200:
        raise ArtifactError("unseen tau3 realized episode census differs from the frozen plan")

    rng = random.Random(random_seed)
    family_outputs: dict[str, Any] = {}
    for family in _CORE_FAMILIES:
        supported = [row for row in group_rows.values() if family.value in row["eligible_families"]]
        if not supported:
            continue
        effects = [
            row["values"][family.value] - row["values"][SemanticFamily.NO_OP.value]
            for row in supported
        ]
        strata_outputs: dict[str, Any] = {}
        for stratum_id in sorted({row["stratum_id"] for row in supported}):
            values = [
                row["values"][family.value] - row["values"][SemanticFamily.NO_OP.value]
                for row in supported
                if row["stratum_id"] == stratum_id
            ]
            strata_outputs[stratum_id] = {
                "support_groups": len(values),
                "mean_effect_vs_no_op": fmean(values),
                "cluster_bootstrap_95ci": list(
                    _cluster_bootstrap_interval(
                        values,
                        samples=bootstrap_samples,
                        rng=rng,
                    )
                ),
            }
        worst = min(
            strata_outputs.items(),
            key=lambda item: (
                float(item[1]["mean_effect_vs_no_op"]),
                item[0],
            ),
        )
        family_outputs[family.value] = {
            "support_groups": len(supported),
            "eligibility_coverage": len(supported) / 200.0,
            "mean_effect_vs_no_op_on_legal_support": fmean(effects),
            "cluster_bootstrap_95ci": list(
                _cluster_bootstrap_interval(
                    effects,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            ),
            "strata": strata_outputs,
            "worst_supported_stratum": {
                "stratum_id": worst[0],
                **worst[1],
            },
            "missing_counterfactuals_imputed": False,
        }

    selector_output: Mapping[str, Any]
    if selector_decisions is None:
        selector_output = {
            "evaluated": False,
            "analysis_scope": "family_transfer_only",
            "reason": (
                "No frozen unseen-group selector decision artifact was supplied; "
                "family effects are not selector-transfer evidence."
            ),
        }
    else:
        frozen_decisions = validate_unseen_tau3_selector_decisions(
            selector_decisions,
            eligibility_plan_hash=plan["plan_hash"],
            representative_manifest_hash=representative_hash,
        )
        selector_output = _evaluate_unseen_selector(
            frozen_decisions,
            group_rows=group_rows,
            releases=releases,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    assert dependency_identity is not None
    assert scorer_identity is not None
    return freeze_mapping(
        {
            "schema_version": 1,
            "analysis": "paper_unseen_tau3_stratified_transfer",
            "analysis_protocol": PAPER_UNSEEN_TAU3_PROTOCOL,
            "metric": metric,
            "eligibility_plan_hash": plan["plan_hash"],
            "representative_manifest_hash": representative_hash,
            "run_count": len(runs),
            "independent_groups": 200,
            "verified_agent_episodes": total_episodes,
            "maximum_agent_episodes": 1_200,
            "source_revision": _TAU3_REVISION,
            "model_slot": plan["model_slot"],
            "seed": plan["seed"],
            "family_effects": family_outputs,
            "selector_transfer": selector_output,
            "dependency_identity_sha256": content_hash(dependency_identity),
            "scorer_identity": {
                "scorer_id": scorer_identity[0],
                "scorer_revision": scorer_identity[1],
                "outcome_contract_sha256": scorer_identity[2],
            },
            "support_rule": (
                "Each family is estimated only on groups where that family was "
                "frozen as eligible; unavailable actions are never imputed."
            ),
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "algorithm_failure_value": algorithm_failure_value,
            "source_runs": tuple(
                {
                    "study_id": run.manifest.study_id,
                    "stratum_id": run.stratum_id,
                    "run_identity_hash": run.run_identity_hash,
                    "manifest_hash": run.manifest_hash,
                    "plan_hash": run.plan_hash,
                }
                for run in sorted(runs, key=lambda item: item.stratum_id)
            ),
            "experiments_executed_by_analysis": False,
            "paper_claim_authorized": False,
        }
    )


def _load_unseen_run(
    root: Path,
    *,
    plan: Mapping[str, Any],
    representative_manifest_hash: str,
    releases: Mapping[SemanticFamily, str],
) -> _UnseenRun:
    verification = verify_experiment_run_directory(root)
    if not verification["complete"] or not verification["all_succeeded"]:
        raise ArtifactError("unseen tau3 analysis requires complete all-succeeded verified runs")
    manifest = load_experiment_manifest(root / "manifest.json")
    if (
        manifest.stage is not ExperimentStage.TRANSFER
        or manifest.estimand_profile_kind is not ProfileKind.CONTROLLED_ADAPTER
        or not manifest.study_id.startswith("robustness.unseen_tau3.")
        or manifest.driver_entrypoint != "agentinputlab.experiments.drivers.external_arena:factory"
        or len(manifest.benchmarks) != 1
        or manifest.benchmarks[0].cell_id != "tau3.banking_knowledge"
        or manifest.benchmarks[0].source_revision != _TAU3_REVISION
        or manifest.metadata.get("requires_atomic_external_replay") is not True
        or manifest.metadata.get("fit_or_calibrate_allowed") is not False
        or manifest.metadata.get("eligibility_plan_hash") != plan["plan_hash"]
        or manifest.metadata.get("eligibility_plan_total_groups") != 200
        or manifest.metadata.get("campaign_maximum_agent_episodes") != 1_200
        or manifest.metadata.get("campaign_expected_agent_episodes")
        != plan["planned_agent_episodes"]
        or manifest.metadata.get("representative_manifest_hash") != representative_manifest_hash
        or manifest.model_slots != (plan["model_slot"],)
        or manifest.seeds != (plan["seed"],)
    ):
        raise ArtifactError("unseen tau3 run differs from the frozen transfer protocol")
    stratum_id = manifest.metadata.get("eligibility_stratum")
    candidates = [stratum for stratum in plan["strata"] if stratum["stratum_id"] == stratum_id]
    if len(candidates) != 1:
        raise ArtifactError("unseen tau3 run references an unknown eligibility stratum")
    stratum = candidates[0]
    families = tuple(SemanticFamily(value) for value in stratum["eligible_families"])
    if (
        manifest.metadata.get("eligible_families") != list(stratum["eligible_families"])
        or manifest.benchmarks[0].split != f"unseen_transfer.{stratum_id}"
    ):
        raise ArtifactError("unseen tau3 run eligibility or split differs from its stratum")
    _unseen_arm_contract(
        manifest,
        expected_families=families,
        representative_manifest_hash=representative_manifest_hash,
        releases=releases,
    )
    census = manifest.metadata.get("experiment_census")
    group_count = int(stratum["group_count"])
    expected_episodes = group_count * len(manifest.arms)
    if (
        not isinstance(census, Mapping)
        or census.get("protocol") != "agentinputlab.experiment_census.v1"
        or census.get("campaign_kind") != "unseen_tau3_transfer_stratum"
        or census.get("independent_group_target") != group_count
        or census.get("expected_plan_cells") != group_count
        or census.get("expected_agent_episodes") != expected_episodes
        or tuple(census.get("expected_arm_ids", ())) != tuple(arm.arm_id for arm in manifest.arms)
        or tuple(census.get("expected_model_slots", ())) != manifest.model_slots
        or tuple(census.get("expected_seeds", ())) != manifest.seeds
    ):
        raise ArtifactError("unseen tau3 run census differs from the frozen stratum")
    records = load_verified_cell_results(root)
    if len(records) != group_count:
        raise ArtifactError("unseen tau3 run record count differs from its group census")
    for record in records:
        cell = record.get("cell")
        if (
            not isinstance(cell, Mapping)
            or cell.get("benchmark_id") != "tau3.banking_knowledge"
            or cell.get("split") != f"unseen_transfer.{stratum_id}"
            or cell.get("model_slot") != plan["model_slot"]
            or cell.get("seed") != plan["seed"]
            or tuple(cell.get("arm_ids", ())) != tuple(arm.arm_id for arm in manifest.arms)
        ):
            raise ArtifactError("unseen tau3 result cell differs from its frozen manifest")
    return _UnseenRun(
        root=root,
        manifest=manifest,
        records=records,
        stratum_id=str(stratum_id),
        eligible_families=families,
        group_count=group_count,
        expected_episodes=expected_episodes,
        run_identity_hash=_sha256(
            verification["identity_hash"],
            label="unseen tau3 run identity",
        ),
        manifest_hash=_sha256(
            verification["manifest_hash"],
            label="unseen tau3 manifest hash",
        ),
        plan_hash=_sha256(
            verification["plan_hash"],
            label="unseen tau3 run plan hash",
        ),
    )


def _unseen_arm_contract(
    manifest: ExperimentManifest,
    *,
    expected_families: Sequence[SemanticFamily],
    representative_manifest_hash: str,
    releases: Mapping[SemanticFamily, str],
) -> tuple[str, Mapping[SemanticFamily, str]]:
    baselines = [arm for arm in manifest.arms if arm.family is SemanticFamily.NO_OP]
    interventions = [arm for arm in manifest.arms if arm.family is not SemanticFamily.NO_OP]
    if (
        len(baselines) != 1
        or baselines[0].release_id != "baseline.no_op"
        or tuple(arm.family for arm in interventions) != tuple(expected_families)
        or len({arm.family for arm in interventions}) != len(interventions)
    ):
        raise ArtifactError("unseen tau3 arm inventory differs from legal family support")
    family_arms: dict[SemanticFamily, str] = {}
    for arm in interventions:
        if (
            arm.release_id != releases.get(arm.family)
            or arm.parameters.get("representative_rank") != 1
            or arm.parameters.get("representative_manifest_hash") != representative_manifest_hash
        ):
            raise ArtifactError("unseen tau3 arm differs from the frozen rank-1 representative")
        family_arms[arm.family] = arm.arm_id
    return baselines[0].arm_id, MappingProxyType(family_arms)


def _validate_unseen_external_receipts(
    manifest: ExperimentManifest,
    record: Mapping[str, Any],
    *,
    baseline_arm_id: str,
    family_arms: Mapping[SemanticFamily, str],
    metric: str,
) -> Mapping[str, Any]:
    normalized = validate_external_arena_record_receipts(
        manifest,
        record,
        metric=metric,
    )
    expected_arm_ids = {baseline_arm_id, *family_arms.values()}
    if (
        normalized["suite_id"] != "tau3.banking_knowledge"
        or normalized["source_revision"] != _TAU3_REVISION
        or normalized["baseline_arm_id"] != baseline_arm_id
        or set(normalized["arms"]) != expected_arm_ids
        or any(
            normalized["arms"][arm_id]["family"] != family.value
            for family, arm_id in family_arms.items()
        )
    ):
        raise ArtifactError(
            "unseen tau3 normalized atomic receipt differs from legal family support"
        )
    cell = record.get("cell")
    result = record.get("result")
    payload = result.get("payload") if isinstance(result, Mapping) else None
    if not isinstance(cell, Mapping) or not isinstance(payload, Mapping):
        raise ArtifactError("unseen tau3 result lacks cell or payload")
    if (
        payload.get("atomic_replay_contract") != ATOMIC_EXTERNAL_PAIR_CONTRACT
        or payload.get("benchmark_id") != cell.get("benchmark_id")
        or payload.get("split") != cell.get("split")
        or payload.get("unit_id") != cell.get("unit_id")
        or payload.get("model_slot") != cell.get("model_slot")
        or payload.get("seed") != cell.get("seed")
        or payload.get("baseline_arm_id") != baseline_arm_id
        or tuple(payload.get("planned_arm_ids", ())) != tuple(cell.get("arm_ids", ()))
        or payload.get("target_source_revision") != _TAU3_REVISION
    ):
        raise ArtifactError("unseen tau3 atomic payload identity is invalid")
    checkpoint = payload.get("checkpoint_identity")
    if not isinstance(checkpoint, Mapping):
        raise ArtifactError("unseen tau3 payload lacks checkpoint identity")
    group_id = checkpoint.get("group_sha256")
    if (
        not isinstance(group_id, str)
        or _GROUP_SHA256.fullmatch(group_id) is None
        or checkpoint.get("source_revision") != _TAU3_REVISION
    ):
        raise ArtifactError("unseen tau3 checkpoint group or source identity is invalid")
    for field in (
        "identity_sha256",
        "visible_state_sha256",
        "origin_state_sha256",
        "replay_identity_sha256",
    ):
        _sha256(checkpoint.get(field), label=f"unseen tau3 checkpoint {field}")
    dependencies = _dependency_hashes(
        payload.get("observed_dependency_hashes"),
        label="unseen tau3 observed dependencies",
    )
    baseline_consistency = _sha256(
        payload.get("baseline_consistency_sha256"),
        label="unseen tau3 baseline consistency",
    )
    pairs = _sequence(payload.get("atomic_pairs"), label="unseen tau3 atomic pairs")
    expected_treatments = set(family_arms.values())
    if len(pairs) != len(expected_treatments):
        raise ArtifactError("unseen tau3 atomic pair inventory is incomplete")
    receipt_hash_by_arm: dict[str, str] = {}
    scorer: tuple[str, str] | None = None
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ArtifactError("unseen tau3 atomic pair receipt must be an object")
        treatment_id = pair.get("treatment_arm_id")
        if (
            treatment_id not in expected_treatments
            or treatment_id in receipt_hash_by_arm
            or pair.get("baseline_consistency_sha256") != baseline_consistency
        ):
            raise ArtifactError("unseen tau3 atomic pair treatment identity is invalid")
        _sha256(pair.get("request_sha256"), label="unseen tau3 atomic request")
        receipt = pair.get("receipt")
        if not isinstance(receipt, Mapping):
            raise ArtifactError("unseen tau3 atomic pair lacks a typed receipt")
        receipt_hash = _sha256(
            pair.get("receipt_sha256"),
            label="unseen tau3 atomic receipt",
        )
        if content_hash(receipt) != receipt_hash:
            raise ArtifactError("unseen tau3 atomic receipt hash differs from content")
        if (
            receipt.get("benchmark_id") != "tau3.banking_knowledge"
            or receipt.get("checkpoint_id") != cell.get("unit_id")
            or receipt.get("source_revision") != _TAU3_REVISION
            or receipt.get("origin_state_hash") != checkpoint["origin_state_sha256"]
            or receipt.get("replay_identity") != checkpoint["replay_identity_sha256"]
            or receipt.get("seed") != cell.get("seed")
            or receipt.get("locked_dependency_hashes") != dependencies
        ):
            raise ArtifactError("unseen tau3 atomic receipt differs from frozen replay identity")
        scorer_id = _nonempty_text(
            receipt.get("scorer_id"),
            label="unseen tau3 scorer ID",
        )
        scorer_revision = _sha256(
            receipt.get("scorer_revision"),
            label="unseen tau3 scorer revision",
        )
        current_scorer = (scorer_id, scorer_revision)
        if scorer is None:
            scorer = current_scorer
        elif current_scorer != scorer:
            raise ArtifactError("unseen tau3 atomic pairs use different official scorers")
        arms = _sequence(
            receipt.get("arms"),
            label="unseen tau3 atomic receipt arms",
        )
        by_role: dict[str, Mapping[str, Any]] = {}
        for arm in arms:
            if not isinstance(arm, Mapping) or arm.get("role") not in {
                "baseline",
                "treatment",
            }:
                raise ArtifactError("unseen tau3 atomic arm receipt is invalid")
            role = str(arm["role"])
            if role in by_role:
                raise ArtifactError("unseen tau3 atomic receipt duplicates an arm role")
            by_role[role] = arm
        if set(by_role) != {"baseline", "treatment"}:
            raise ArtifactError("unseen tau3 atomic receipt lacks both arm roles")
        if (
            by_role["baseline"].get("arm_id") != baseline_arm_id
            or by_role["treatment"].get("arm_id") != treatment_id
            or len(
                {
                    by_role["baseline"].get("runtime_instance_id"),
                    by_role["treatment"].get("runtime_instance_id"),
                }
            )
            != 2
        ):
            raise ArtifactError("unseen tau3 atomic arm roles or runtime instances are invalid")
        for arm in by_role.values():
            if (
                arm.get("completed") is not True
                or arm.get("origin_state_hash") != checkpoint["origin_state_sha256"]
                or arm.get("replay_identity") != checkpoint["replay_identity_sha256"]
                or arm.get("seed") != cell.get("seed")
                or arm.get("locked_dependency_hashes") != dependencies
            ):
                raise ArtifactError(
                    "unseen tau3 atomic arm receipt differs from shared replay identity"
                )
            _sha256(
                arm.get("runtime_instance_id"),
                label="unseen tau3 runtime instance",
            )
        receipt_hash_by_arm[str(treatment_id)] = receipt_hash
    if set(receipt_hash_by_arm) != expected_treatments or scorer is None:
        raise ArtifactError("unseen tau3 atomic pair receipt coverage is incomplete")

    raw_arms = _sequence(payload.get("arms"), label="unseen tau3 arm outcomes")
    outcomes: dict[str, Mapping[str, Any]] = {}
    for arm in raw_arms:
        if not isinstance(arm, Mapping) or not isinstance(arm.get("arm_id"), str):
            raise ArtifactError("unseen tau3 arm outcome is invalid")
        arm_id = str(arm["arm_id"])
        if arm_id in outcomes:
            raise ArtifactError("unseen tau3 arm outcomes contain duplicates")
        outcomes[arm_id] = arm
    expected_arm_ids = {baseline_arm_id, *expected_treatments}
    if set(outcomes) != expected_arm_ids:
        raise ArtifactError("unseen tau3 arm outcomes differ from the legal arm inventory")
    baseline_hashes = tuple(outcomes[baseline_arm_id].get("atomic_pair_receipt_sha256s", ()))
    if set(baseline_hashes) != set(receipt_hash_by_arm.values()):
        raise ArtifactError("unseen tau3 NO_OP outcome is not bound to every atomic pair")
    contract_hash: str | None = None
    for arm_id, outcome in outcomes.items():
        if outcome.get("status") != "succeeded":
            raise ArtifactError("unseen tau3 paper analysis requires succeeded arm outcomes")
        if arm_id != baseline_arm_id and (
            outcome.get("atomic_pair_receipt_sha256") != receipt_hash_by_arm[arm_id]
        ):
            raise ArtifactError("unseen tau3 treatment outcome is bound to another atomic pair")
        composition = outcome.get("valid_success_composition")
        metrics = outcome.get("metrics")
        if not isinstance(composition, Mapping) or not isinstance(metrics, Mapping):
            raise ArtifactError("unseen tau3 outcome lacks official score composition")
        if (
            composition.get("benchmark_id") != "tau3.banking_knowledge"
            or composition.get("arm_id") != arm_id
            or composition.get("scorer_id") != scorer[0]
            or composition.get("scorer_revision") != scorer[1]
            or composition.get("trace_hash") != outcome.get("trace_hash")
            or composition.get(metric) != metrics.get(metric)
        ):
            raise ArtifactError("unseen tau3 valid-success composition is inconsistent")
        composition_hash = _sha256(
            composition.get("composition_receipt_sha256"),
            label="unseen tau3 valid-success composition",
        )
        if (
            content_hash(
                {
                    key: item
                    for key, item in composition.items()
                    if key != "composition_receipt_sha256"
                }
            )
            != composition_hash
        ):
            raise ArtifactError("unseen tau3 valid-success receipt hash differs from content")
        current_contract = _sha256(
            composition.get("outcome_contract_sha256"),
            label="unseen tau3 outcome contract",
        )
        if contract_hash is None:
            contract_hash = current_contract
        elif current_contract != contract_hash:
            raise ArtifactError("unseen tau3 arms use different valid-success contracts")
    assert contract_hash is not None
    if (
        normalized["group_id"] != group_id
        or normalized["dependency_identity"] != dependencies
        or normalized["scorer_identity"]
        != {
            "scorer_id": scorer[0],
            "scorer_revision": scorer[1],
        }
        or normalized["outcome_contract_sha256"] != contract_hash
    ):
        raise ArtifactError("unseen tau3 specialized receipts differ from shared arena validation")
    return freeze_mapping(
        {
            "group_sha256": group_id,
            "dependencies": dependencies,
            "scorer_id": scorer[0],
            "scorer_revision": scorer[1],
            "outcome_contract_sha256": contract_hash,
        }
    )


def _evaluate_unseen_selector(
    decisions: Mapping[str, Any],
    *,
    group_rows: Mapping[str, Mapping[str, Any]],
    releases: Mapping[SemanticFamily, str],
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    decision_rows = {row["group_sha256"]: row for row in decisions["records"]}
    if set(decision_rows) != set(group_rows):
        raise ValidationError(
            "unseen tau3 selector decisions differ from the verified transfer groups"
        )
    values: list[float] = []
    baseline_values: list[float] = []
    effects: list[float] = []
    intervention_coverage: list[float] = []
    for group_id in sorted(group_rows):
        group = group_rows[group_id]
        decision = decision_rows[group_id]
        family = SemanticFamily(decision["selected_family"])
        if family is SemanticFamily.NO_OP:
            expected_release = "baseline.no_op"
        else:
            if family.value not in group["eligible_families"]:
                raise ValidationError(
                    "unseen tau3 selector chose a family outside frozen eligibility"
                )
            expected_release = releases[family]
        if decision["selected_release_id"] != expected_release:
            raise ValidationError("unseen tau3 selector changed the frozen representative release")
        baseline = float(group["values"][SemanticFamily.NO_OP.value])
        selected = float(group["values"][family.value])
        values.append(selected)
        baseline_values.append(baseline)
        effects.append(selected - baseline)
        intervention_coverage.append(0.0 if family is SemanticFamily.NO_OP else 1.0)
    return {
        "evaluated": True,
        "analysis_scope": "frozen_unseen_group_selector_transfer",
        "selector_id": decisions["selector_id"],
        "selector_revision": decisions["selector_revision"],
        "selector_artifact_sha256": decisions["selector_artifact_sha256"],
        "decision_hash": decisions["decision_hash"],
        "independent_groups": 200,
        "mean_policy_value": fmean(values),
        "mean_no_op_value": fmean(baseline_values),
        "mean_effect_vs_no_op": fmean(effects),
        "cluster_bootstrap_95ci": list(
            _cluster_bootstrap_interval(
                effects,
                samples=bootstrap_samples,
                rng=rng,
            )
        ),
        "intervention_coverage": fmean(intervention_coverage),
        "missing_counterfactuals_imputed": False,
        "paper_claim_authorized": False,
    }


def _validated_representatives(
    value: Mapping[str, Any],
) -> tuple[str, Mapping[SemanticFamily, str]]:
    # Local import avoids a package-import cycle: selector campaigns use the
    # experiment schema, while this analysis is exported by experiments.
    from agentinputlab.selectors.campaigns import validate_representative_manifest

    frozen = validate_representative_manifest(value)
    releases = {
        SemanticFamily(row["family"]): row["release_id"]
        for row in frozen["selections"]
        if row["rank"] == 1
    }
    if set(releases) != set(_CORE_FAMILIES):
        raise ValidationError("unseen tau3 transfer requires one frozen rank-1 release per family")
    return frozen["manifest_hash"], MappingProxyType(releases)


def _dependency_hashes(value: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactError(f"{label} must be a non-empty object")
    parsed: dict[str, str] = {}
    for slot, digest in value.items():
        if not isinstance(slot, str) or not slot:
            raise ArtifactError(f"{label} contains an invalid slot")
        parsed[slot] = _sha256(digest, label=f"{label}.{slot}")
    return freeze_mapping(parsed)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase identifier")
    return value


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty text")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    return tuple(value)


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result < 1:
        raise ValidationError(f"{label} must be positive")
    return result


__all__ = [
    "PAPER_UNSEEN_TAU3_PROTOCOL",
    "UNSEEN_TAU3_SELECTOR_DECISIONS_ARTIFACT",
    "analyze_unseen_tau3_transfer",
    "validate_unseen_tau3_analysis_plan",
    "validate_unseen_tau3_selector_decisions",
]
