"""Fail-closed analysis for gated stateful-arena replications.

Stateful arenas do not share an official raw outcome scale.  This module
therefore validates the complete atomic-replay evidence chain and estimates
effects inside each arena relative to the same-origin ``NO_OP`` result.  It
never pools raw metrics across arenas.  The only cross-arena summaries are
scale-free direction and win-minus-harm diagnostics over arena-local paired
effects.

The public record validator is deliberately reusable by transfer analyses.
It validates record-local atomic, scorer, source, dependency, and balance
evidence, but leaves the decision to require an approved external gate to the
calling protocol.  The stateful-replication entry point always requires that
gate and additionally cross-checks it against the immutable driver
fingerprint stored in the verified run directory.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

from agentinputlab.benchmarks import (
    ATOMIC_EXTERNAL_PAIR_CONTRACT,
    AtomicReplayArmReceipt,
    AtomicReplayReceipt,
)
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .analysis import _load_manifest, load_verified_cell_results
from .artifacts import verify_experiment_run_directory
from .schema import ExperimentManifest, ExperimentStage

STATEFUL_ARENA_REPLICATION_PROTOCOL = "stateful-arena-replication-v1"
STATEFUL_ARENA_REPLICATION_ARTIFACT = "agentinputlab.stateful_arena_replication_analysis"

_EXTERNAL_ARENA_DRIVER = "agentinputlab.experiments.drivers.external_arena:factory"
_EXTERNAL_ARENA_IMPLEMENTATION = "agentinputlab.external_arena.atomic_multi_arm"
_VALID_SUCCESS_FORMULA = "agentinputlab.valid_success.v1"
_VALID_SUCCESS_METRIC = "valid_success"
_SUPPORTED_STATEFUL_SUITES = frozenset(
    {
        "tau3.banking_knowledge",
        "state_bench.common_support",
        "toolsandbox.common_support",
        "appworld_ul.common_support",
    }
)
_DEPENDENCY_SLOTS = frozenset(
    {
        "agent_model",
        "official_runtime",
        "corpus",
        "preprocessing",
        "index",
        "embedder",
        "retriever",
        "reranker",
        "reader",
        "scorer",
    }
)
_NON_MODEL_DEPENDENCY_SLOTS = _DEPENDENCY_SLOTS - {"agent_model"}
_ALLOWED_STAGES = frozenset(
    {
        ExperimentStage.CONFIRMATORY,
        ExperimentStage.SELECTOR_CONFIRMATION,
        ExperimentStage.REPLICATION,
    }
)


@dataclass(frozen=True, slots=True)
class _VerifiedStatefulRun:
    root: Path
    manifest: ExperimentManifest
    records: tuple[Mapping[str, Any], ...]
    verification: Mapping[str, Any]
    driver_fingerprint: Mapping[str, Any]


def validate_external_arena_record_receipts(
    manifest: ExperimentManifest,
    record: Mapping[str, Any],
    *,
    metric: str,
) -> Mapping[str, Any]:
    """Validate one external-arena result and return a normalized paired row.

    The returned mapping contains no unpaired raw outcome summary.  It carries
    the requested value for each arm only so a caller can form arena-local,
    same-checkpoint effects.  An ``external_gate_receipt`` is validated when
    present and returned as ``gate_receipt``; protocols such as unseen-suite
    transfer may apply a different gate rule, whereas
    :func:`analyze_stateful_arena_replication` requires it unconditionally.
    """

    if not isinstance(manifest, ExperimentManifest):
        raise TypeError("external arena record validation requires ExperimentManifest")
    metric_name = _metric_name(metric)
    if not isinstance(record, Mapping):
        raise ArtifactError("external arena result record must be an object")

    cell = _object(record.get("cell"), label="external arena result cell")
    result = _object(record.get("result"), label="external arena result")
    if result.get("status") != "succeeded":
        raise ArtifactError("external arena receipt analysis requires a succeeded cell")
    payload = _object(
        result.get("payload"),
        label="external arena result payload",
    )

    suite_id = _text(cell.get("benchmark_id"), label="external arena cell benchmark_id")
    split = _text(cell.get("split"), label="external arena cell split")
    unit_id = _text(cell.get("unit_id"), label="external arena cell unit_id")
    model_slot = _text(cell.get("model_slot"), label="external arena cell model_slot")
    seed = _seed(cell.get("seed"), label="external arena cell seed")
    target = _manifest_target(manifest, suite_id=suite_id, split=split)
    if target.external is not True or target.source_revision is None:
        raise ArtifactError("stateful arena analysis requires an external source-locked target")
    if manifest.metadata.get("requires_atomic_external_replay") is not True:
        raise ArtifactError("external arena manifest lacks requires_atomic_external_replay=true")
    if model_slot not in manifest.model_slots or seed not in manifest.seeds:
        raise ArtifactError("external arena cell model slot or seed is outside the manifest")

    planned_arm_ids = _text_sequence(
        cell.get("arm_ids"),
        label="external arena cell arm_ids",
    )
    expected_arm_ids = target.arm_ids or tuple(arm.arm_id for arm in manifest.arms)
    if planned_arm_ids != expected_arm_ids:
        raise ArtifactError("external arena cell arms differ from the manifest target")
    _require_payload_identity(
        payload,
        suite_id=suite_id,
        split=split,
        unit_id=unit_id,
        model_slot=model_slot,
        seed=seed,
        planned_arm_ids=planned_arm_ids,
    )

    baseline_arm_id = _text(
        payload.get("baseline_arm_id"),
        label="external arena baseline_arm_id",
    )
    arm_specs = {arm.arm_id: arm for arm in manifest.arms}
    if baseline_arm_id not in planned_arm_ids:
        raise ArtifactError("external arena baseline is outside the planned arm inventory")
    baseline_spec = arm_specs.get(baseline_arm_id)
    if baseline_spec is None or baseline_spec.family is not SemanticFamily.NO_OP:
        raise ArtifactError("external arena baseline must be the manifest NO_OP arm")

    source_revision = target.source_revision
    checkpoint = _checkpoint_identity(
        payload.get("checkpoint_identity"),
        source_revision=source_revision,
    )
    if payload.get("target_source_revision") != source_revision:
        raise ArtifactError("external arena target source receipt differs from the manifest")
    if payload.get("atomic_replay_contract") != ATOMIC_EXTERNAL_PAIR_CONTRACT:
        raise ArtifactError("external arena result lacks the atomic replay contract")

    dependencies = _dependency_hashes(
        payload.get("observed_dependency_hashes"),
        label="external arena observed dependency receipt",
        exact_slots=True,
    )
    _validate_not_applicable_slots(payload, dependencies=dependencies)

    baseline_consistency = _sha256(
        payload.get("baseline_consistency_sha256"),
        label="external arena baseline consistency receipt",
    )
    raw_outcomes = _sequence(
        payload.get("arms"),
        label="external arena arm outcomes",
    )
    if len(raw_outcomes) != len(planned_arm_ids):
        raise ArtifactError("external arena arm outcome inventory is partial")
    outcomes: dict[str, Mapping[str, Any]] = {}
    metric_schema: tuple[str, ...] | None = None
    scorer_identity: tuple[str, str] | None = None
    official_success_metric: str | None = None
    outcome_contract_sha256: str | None = None
    normalized_arms: dict[str, Mapping[str, Any]] = {}
    for index, raw_outcome in enumerate(raw_outcomes):
        outcome = _object(raw_outcome, label=f"external arena arm outcome {index}")
        arm_id = _text(
            outcome.get("arm_id"),
            label=f"external arena arm outcome {index} arm_id",
        )
        if arm_id in outcomes:
            raise ArtifactError("external arena result contains duplicate arm outcomes")
        if arm_id != planned_arm_ids[index]:
            raise ArtifactError("external arena arm outcomes differ from planned order")
        spec = arm_specs.get(arm_id)
        if spec is None:
            raise ArtifactError("external arena result references an unknown manifest arm")
        if outcome.get("status") != "succeeded":
            raise ArtifactError("stateful arena analysis refuses failed or imputed arm outcomes")
        if (
            outcome.get("release_id") != spec.release_id
            or outcome.get("family") != spec.family.value
            or outcome.get("profile_kind") != spec.profile_kind.value
        ):
            raise ArtifactError(f"external arena arm {arm_id!r} identity differs from the manifest")
        metrics = _metrics(
            outcome.get("metrics"),
            label=f"external arena arm {arm_id!r} metrics",
        )
        current_schema = tuple(sorted(metrics))
        if metric_schema is None:
            metric_schema = current_schema
        elif current_schema != metric_schema:
            raise ArtifactError("external arena arms expose heterogeneous metric schemas")
        if metric_name not in metrics:
            raise ArtifactError(
                f"external arena arm {arm_id!r} lacks requested metric {metric_name!r}"
            )
        _sha256(
            outcome.get("payload_sha256"),
            label=f"external arena arm {arm_id!r} payload receipt",
        )

        composition = _validate_valid_success_composition(
            outcome,
            metrics=metrics,
            suite_id=suite_id,
            arm_id=arm_id,
        )
        current_scorer = (
            str(composition["scorer_id"]),
            str(composition["scorer_revision"]),
        )
        current_official_metric = str(composition["official_success_metric"])
        current_outcome_contract = str(composition["outcome_contract_sha256"])
        if scorer_identity is None:
            scorer_identity = current_scorer
            official_success_metric = current_official_metric
            outcome_contract_sha256 = current_outcome_contract
        elif (
            current_scorer != scorer_identity
            or current_official_metric != official_success_metric
            or current_outcome_contract != outcome_contract_sha256
        ):
            raise ArtifactError(
                "external arena official scorer, metric, or outcome contract "
                "identity differs across arms"
            )
        outcomes[arm_id] = outcome
        normalized_arms[arm_id] = {
            "arm_id": arm_id,
            "release_id": spec.release_id,
            "family": spec.family.value,
            "profile_kind": spec.profile_kind.value,
            "arm_kind": spec.arm_kind.value,
            "value": metrics[metric_name],
        }

    assert metric_schema is not None
    assert scorer_identity is not None
    assert official_success_metric is not None
    assert outcome_contract_sha256 is not None
    atomic_pairs = _validate_atomic_pairs(
        payload,
        outcomes=outcomes,
        planned_arm_ids=planned_arm_ids,
        baseline_arm_id=baseline_arm_id,
        suite_id=suite_id,
        unit_id=unit_id,
        seed=seed,
        source_revision=source_revision,
        checkpoint=checkpoint,
        dependencies=dependencies,
        scorer_identity=scorer_identity,
        baseline_consistency=baseline_consistency,
    )
    gate_receipt = _optional_gate_receipt(
        payload.get("external_gate_receipt"),
        manifest=manifest,
        suite_id=suite_id,
        split=split,
        source_revision=source_revision,
        scorer_identity=scorer_identity,
        atomic_replay_contract=ATOMIC_EXTERNAL_PAIR_CONTRACT,
    )

    source_receipt = {
        "suite_id": suite_id,
        "split": split,
        "source_revision": source_revision,
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "origin_state_sha256": checkpoint["origin_state_sha256"],
        "replay_identity_sha256": checkpoint["replay_identity_sha256"],
        "atomic_pair_receipt_sha256s": tuple(pair["receipt_sha256"] for pair in atomic_pairs),
    }
    dependency_receipt = {
        "model_slot": model_slot,
        "locked_dependency_hashes": dependencies,
        "atomic_pair_count": len(atomic_pairs),
    }
    balance_receipt = {
        "baseline_arm_id": baseline_arm_id,
        "planned_arm_ids": planned_arm_ids,
        "complete_atomic_pair_inventory": True,
        "baseline_consistency_sha256": baseline_consistency,
        "metric_schema": metric_schema,
        "dependency_receipt_sha256": content_hash(dependency_receipt),
    }
    observation_id = content_hash(
        {
            "suite_id": suite_id,
            "split": split,
            "unit_id": unit_id,
            "model_slot": model_slot,
            "seed": seed,
        }
    )
    return {
        "suite_id": suite_id,
        "split": split,
        "unit_id": unit_id,
        "group_id": checkpoint["group_sha256"],
        "observation_id": observation_id,
        "model_slot": model_slot,
        "seed": seed,
        "metric": metric_name,
        "metric_schema": metric_schema,
        "official_success_metric": official_success_metric,
        "outcome_contract_sha256": outcome_contract_sha256,
        "source_revision": source_revision,
        "scorer_identity": {
            "scorer_id": scorer_identity[0],
            "scorer_revision": scorer_identity[1],
        },
        "bridge_identity": {
            "bridge_id": atomic_pairs[0]["bridge_id"],
            "bridge_revision": atomic_pairs[0]["bridge_revision"],
        },
        "dependency_identity": dependencies,
        "baseline_arm_id": baseline_arm_id,
        "baseline_identity": {
            key: normalized_arms[baseline_arm_id][key]
            for key in ("release_id", "family", "profile_kind", "arm_kind")
        },
        "arms": normalized_arms,
        "gate_receipt": gate_receipt,
        "source_receipt": {
            **source_receipt,
            "receipt_sha256": content_hash(source_receipt),
        },
        "dependency_receipt": {
            **dependency_receipt,
            "receipt_sha256": content_hash(dependency_receipt),
        },
        "balance_receipt": {
            **balance_receipt,
            "receipt_sha256": content_hash(balance_receipt),
        },
    }


def analyze_stateful_arena_replication(
    run_dirs: Sequence[str | Path],
    *,
    metric: str = _VALID_SUCCESS_METRIC,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
    pool_raw_metrics: bool = False,
) -> Mapping[str, Any]:
    """Analyze one or more verified, gate-approved external arena runs.

    Every arena retains its own estimand.  ``pool_raw_metrics`` exists only as
    an explicit fail-closed API boundary: any value other than ``False`` is
    rejected because official reward, task completion, similarity, and task
    success are not a common raw scale.
    """

    metric_name = _metric_name(metric)
    if pool_raw_metrics is not False:
        raise ValidationError("stateful arena analysis forbids pooling heterogeneous raw metrics")
    _resample_arguments(
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    roots = _run_roots(run_dirs)
    runs = tuple(_load_verified_stateful_run(root) for root in roots)

    rows_by_suite: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_receipts_by_suite: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed_ids: set[str] = set()
    for run in runs:
        _validate_stateful_manifest(run.manifest)
        normalized = tuple(
            validate_external_arena_record_receipts(
                run.manifest,
                record,
                metric=metric_name,
            )
            for record in run.records
        )
        if not normalized:
            raise ArtifactError("verified stateful arena run contains no result records")
        _validate_run_fingerprint(run, normalized)
        for row in normalized:
            if row["gate_receipt"] is None:
                raise ArtifactError(
                    "stateful arena replication requires an approved external gate receipt"
                )
            observation_id = str(row["observation_id"])
            if observation_id in observed_ids:
                raise ArtifactError(
                    "stateful arena runs contain a duplicate checkpoint/model/seed observation"
                )
            observed_ids.add(observation_id)
            suite_id = str(row["suite_id"])
            rows_by_suite[suite_id].append(row)
        for suite_id in sorted({str(row["suite_id"]) for row in normalized}):
            run_receipts_by_suite[suite_id].append(
                {
                    "run_dir": str(run.root),
                    "study_id": run.manifest.study_id,
                    "manifest_hash": run.verification["manifest_hash"],
                    "plan_hash": run.verification["plan_hash"],
                    "driver_hash": run.verification["driver_hash"],
                    "comparison_profile_hash": run.verification["comparison_profile_hash"],
                }
            )

    rng = random.Random(random_seed)
    suites: dict[str, Mapping[str, Any]] = {}
    for suite_id, rows in sorted(rows_by_suite.items()):
        suites[suite_id] = _summarize_suite(
            suite_id,
            rows,
            run_receipts=run_receipts_by_suite[suite_id],
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )

    return {
        "schema_version": 1,
        "artifact": STATEFUL_ARENA_REPLICATION_ARTIFACT,
        "analysis": "stateful_arena_replication",
        "analysis_protocol": STATEFUL_ARENA_REPLICATION_PROTOCOL,
        "metric": metric_name,
        "run_count": len(runs),
        "suite_count": len(suites),
        "estimand": (
            "arena-local protected-checkpoint-group mean paired difference "
            "relative to same-origin NO_OP"
        ),
        "analysis_unit": (
            "protected checkpoint group; unit variants, model slots, and seeds "
            "are repeated observations collapsed within group"
        ),
        "receipt_policy": {
            "complete_verified_runs_required": True,
            "approved_gate_required": True,
            "atomic_replay_required": True,
            "official_scorer_receipt_required": True,
            "source_receipt_required": True,
            "dependency_balance_required": True,
        },
        "suites": suites,
        "cross_suite_robustness": _cross_suite_robustness(suites),
        "raw_metric_pooling_performed": False,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "experiments_executed_by_analysis": False,
        "paper_claim_authorized": False,
    }


def _load_verified_stateful_run(root: Path) -> _VerifiedStatefulRun:
    verification = verify_experiment_run_directory(root)
    if verification["is_shard"]:
        raise ArtifactError(
            "stateful arena analysis refuses a shard without an exact verified merge"
        )
    if not verification["complete"] or not verification["all_succeeded"]:
        raise ArtifactError(
            "stateful arena analysis requires one complete all-succeeded verified run"
        )
    manifest = _load_manifest(root)
    records = load_verified_cell_results(root)
    identity = _read_json_object(
        root / "run_identity.json",
        label="stateful arena run identity",
    )
    fingerprint = identity.get("driver_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ArtifactError("stateful arena run lacks a driver fingerprint")
    return _VerifiedStatefulRun(
        root=root,
        manifest=manifest,
        records=records,
        verification=verification,
        driver_fingerprint=fingerprint,
    )


def _validate_stateful_manifest(manifest: ExperimentManifest) -> None:
    if manifest.stage not in _ALLOWED_STAGES:
        raise ArtifactError(
            "stateful arena replication requires a confirmatory, selector-confirmation, "
            "or replication manifest"
        )
    if manifest.driver_entrypoint != _EXTERNAL_ARENA_DRIVER:
        raise ArtifactError("stateful arena replication requires the external atomic-arena driver")
    if manifest.metadata.get("requires_atomic_external_replay") is not True:
        raise ArtifactError(
            "stateful arena replication manifest lacks the atomic replay requirement"
        )
    unsupported = sorted(
        {
            target.cell_id
            for target in manifest.benchmarks
            if target.cell_id not in _SUPPORTED_STATEFUL_SUITES
        }
    )
    if unsupported:
        raise ArtifactError(
            "stateful arena replication contains unsupported suites: " + ", ".join(unsupported)
        )
    if any(not target.external or target.source_revision is None for target in manifest.benchmarks):
        raise ArtifactError("stateful arena replication targets must be external and source locked")


def _validate_run_fingerprint(
    run: _VerifiedStatefulRun,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fingerprint = run.driver_fingerprint
    if (
        fingerprint.get("implementation_id") != _EXTERNAL_ARENA_IMPLEMENTATION
        or fingerprint.get("schema_version") != 3
    ):
        raise ArtifactError(
            "stateful arena run was not produced by the locked atomic multi-arm driver"
        )
    targets = _object(
        fingerprint.get("benchmark_targets"),
        label="stateful arena driver benchmark targets",
    )
    arms = _object(
        fingerprint.get("arms"),
        label="stateful arena driver arm bindings",
    )
    gates = _object(
        fingerprint.get("resolved_gate_bindings"),
        label="stateful arena driver gate bindings",
    )
    baseline_arm_id = _text(
        fingerprint.get("baseline_arm_id"),
        label="stateful arena driver baseline_arm_id",
    )
    manifest_arms = {arm.arm_id: arm for arm in run.manifest.arms}
    for arm_id, spec in manifest_arms.items():
        binding = _object(
            arms.get(arm_id),
            label=f"stateful arena driver arm binding {arm_id!r}",
        )
        if (
            binding.get("release_id") != spec.release_id
            or binding.get("family") != spec.family.value
            or binding.get("profile_kind") != spec.profile_kind.value
            or binding.get("arm_kind") != spec.arm_kind.value
        ):
            raise ArtifactError(
                f"stateful arena driver binding for arm {arm_id!r} has mixed identity"
            )

    for row in rows:
        if row["baseline_arm_id"] != baseline_arm_id:
            raise ArtifactError(
                "stateful arena result baseline differs from the driver fingerprint"
            )
        key = f"{row['suite_id']}:{row['split']}"
        target = _object(
            targets.get(key),
            label=f"stateful arena driver target {key!r}",
        )
        bridge_lock = _object(
            target.get("bridge_lock"),
            label=f"stateful arena driver bridge lock {key!r}",
        )
        outcome_contract = _object(
            target.get("outcome_contract"),
            label=f"stateful arena driver outcome contract {key!r}",
        )
        expected_scorer = {
            "scorer_id": bridge_lock.get("scorer_id"),
            "scorer_revision": bridge_lock.get("scorer_revision"),
        }
        expected_bridge = {
            "bridge_id": bridge_lock.get("bridge_id"),
            "bridge_revision": bridge_lock.get("bridge_revision"),
        }
        dependency_by_model = _object(
            bridge_lock.get("dependency_hashes_by_model_slot"),
            label=f"stateful arena driver dependency locks {key!r}",
        )
        expected_dependencies = _dependency_hashes(
            dependency_by_model.get(row["model_slot"]),
            label=(f"stateful arena driver dependencies for model {row['model_slot']!r}"),
            exact_slots=True,
        )
        if (
            target.get("benchmark_id") != row["suite_id"]
            or target.get("split") != row["split"]
            or target.get("source_revision") != row["source_revision"]
            or expected_scorer != row["scorer_identity"]
            or expected_bridge != row["bridge_identity"]
            or expected_dependencies != row["dependency_identity"]
            or outcome_contract.get("formula_id") != _VALID_SUCCESS_FORMULA
            or outcome_contract.get("official_success_metric") != row["official_success_metric"]
            or content_hash(outcome_contract) != row["outcome_contract_sha256"]
        ):
            raise ArtifactError(
                "stateful arena result identity differs from its driver fingerprint"
            )
        gate = row["gate_receipt"]
        resolved_gate = gates.get(key)
        if (
            gate is None
            or not isinstance(resolved_gate, Mapping)
            or dict(gate) != dict(resolved_gate)
        ):
            raise ArtifactError(
                "stateful arena gate receipt differs from the resolved driver binding"
            )


def _summarize_suite(
    suite_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    run_receipts: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    reference = rows[0]
    reference_fields = (
        "split",
        "source_revision",
        "metric",
        "metric_schema",
        "official_success_metric",
        "outcome_contract_sha256",
        "scorer_identity",
        "bridge_identity",
        "baseline_identity",
    )
    for row in rows[1:]:
        for field in reference_fields:
            if row[field] != reference[field]:
                raise ArtifactError(f"stateful suite {suite_id!r} mixes {field} identities")

    dependencies_by_model: dict[str, Mapping[str, str]] = {}
    arm_identity_by_release: dict[str, Mapping[str, Any]] = {}
    arm_ids_by_release: dict[str, set[str]] = defaultdict(set)
    effects_by_release_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    effects_by_release_model_group: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    replicate_design_by_release_group: dict[str, dict[str, set[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    source_receipt_hashes: set[str] = set()
    dependency_receipt_hashes: set[str] = set()
    balance_receipt_hashes: set[str] = set()
    gate_receipt_hashes: set[str] = set()

    for row in rows:
        model_slot = str(row["model_slot"])
        dependencies = row["dependency_identity"]
        prior_dependencies = dependencies_by_model.get(model_slot)
        if prior_dependencies is None:
            dependencies_by_model[model_slot] = dependencies
        elif prior_dependencies != dependencies:
            raise ArtifactError(
                f"stateful suite {suite_id!r} dependency identity drifts within "
                f"model slot {model_slot!r}"
            )
        baseline = float(row["arms"][row["baseline_arm_id"]]["value"])
        for arm_id, arm in row["arms"].items():
            if arm_id == row["baseline_arm_id"]:
                continue
            release_id = str(arm["release_id"])
            identity = {
                key: arm[key] for key in ("release_id", "family", "profile_kind", "arm_kind")
            }
            prior_identity = arm_identity_by_release.get(release_id)
            if prior_identity is None:
                arm_identity_by_release[release_id] = identity
            elif prior_identity != identity:
                raise ArtifactError(
                    f"stateful suite {suite_id!r} mixes treatment identity "
                    f"for release {release_id!r}"
                )
            arm_ids_by_release[release_id].add(str(arm_id))
            effect = float(arm["value"]) - baseline
            group_id = str(row["group_id"])
            effects_by_release_group[release_id][group_id].append(effect)
            effects_by_release_model_group[release_id][model_slot][group_id].append(effect)
            replicate_design_by_release_group[release_id][group_id].add(
                (model_slot, int(row["seed"]))
            )
        source_receipt_hashes.add(str(row["source_receipt"]["receipt_sha256"]))
        dependency_receipt_hashes.add(str(row["dependency_receipt"]["receipt_sha256"]))
        balance_receipt_hashes.add(str(row["balance_receipt"]["receipt_sha256"]))
        gate_receipt_hashes.add(str(row["gate_receipt"]["receipt_sha256"]))

    _validate_cross_model_dependency_balance(
        suite_id,
        dependencies_by_model=dependencies_by_model,
    )
    treatments: dict[str, Mapping[str, Any]] = {}
    for release_id in sorted(effects_by_release_group):
        by_group = effects_by_release_group[release_id]
        replicate_designs = {
            tuple(sorted(design))
            for design in replicate_design_by_release_group[release_id].values()
        }
        if len(replicate_designs) != 1:
            raise ArtifactError(
                f"stateful suite {suite_id!r} treatment {release_id!r} has "
                "unbalanced model/seed coverage across checkpoint groups"
            )
        cluster_effects = tuple(fmean(values) for _group_id, values in sorted(by_group.items()))
        if len(cluster_effects) < 2:
            raise ArtifactError(
                f"stateful suite {suite_id!r} treatment {release_id!r} "
                "requires at least two independent checkpoint groups"
            )
        win_rate = sum(value > 0 for value in cluster_effects) / len(cluster_effects)
        tie_rate = sum(value == 0 for value in cluster_effects) / len(cluster_effects)
        harm_rate = sum(value < 0 for value in cluster_effects) / len(cluster_effects)
        model_strata: dict[str, Mapping[str, Any]] = {}
        for model_slot, group_values in sorted(effects_by_release_model_group[release_id].items()):
            collapsed = tuple(fmean(values) for _group_id, values in sorted(group_values.items()))
            model_strata[model_slot] = {
                "checkpoint_group_count": len(collapsed),
                "mean_effect_vs_no_op": fmean(collapsed),
                "win_rate_vs_no_op": (sum(value > 0 for value in collapsed) / len(collapsed)),
                "harm_rate_vs_no_op": (sum(value < 0 for value in collapsed) / len(collapsed)),
            }
        treatments[release_id] = {
            **arm_identity_by_release[release_id],
            "arm_ids": sorted(arm_ids_by_release[release_id]),
            "checkpoint_group_count": len(cluster_effects),
            "paired_record_count": sum(len(values) for values in by_group.values()),
            "model_seed_replicate_design": [
                {"model_slot": model_slot, "seed": seed}
                for model_slot, seed in next(iter(replicate_designs))
            ],
            "mean_effect_vs_no_op": fmean(cluster_effects),
            "median_effect_vs_no_op": median(cluster_effects),
            "cluster_bootstrap_95ci": list(
                _bootstrap_mean_interval(
                    cluster_effects,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            ),
            "win_rate_vs_no_op": win_rate,
            "tie_rate_vs_no_op": tie_rate,
            "harm_rate_vs_no_op": harm_rate,
            "normalized_win_minus_harm": win_rate - harm_rate,
            "model_strata": model_strata,
        }

    group_ids = {str(row["group_id"]) for row in rows}
    return {
        "suite_id": suite_id,
        "split": reference["split"],
        "source_revision": reference["source_revision"],
        "metric": reference["metric"],
        "metric_schema": list(reference["metric_schema"]),
        "official_success_metric": reference["official_success_metric"],
        "outcome_contract_sha256": reference["outcome_contract_sha256"],
        "scorer_identity": reference["scorer_identity"],
        "bridge_identity": reference["bridge_identity"],
        "baseline_identity": reference["baseline_identity"],
        "run_count": len(run_receipts),
        "record_count": len(rows),
        "checkpoint_group_count": len(group_ids),
        "run_receipts": list(run_receipts),
        "gate_receipts": {
            "count": len(gate_receipt_hashes),
            "receipt_sha256s": sorted(gate_receipt_hashes),
            "approved": True,
        },
        "source_receipts": {
            "count": len(source_receipt_hashes),
            "receipt_sha256s": sorted(source_receipt_hashes),
            "verified": True,
        },
        "dependency_receipts": {
            "count": len(dependency_receipt_hashes),
            "receipt_sha256s": sorted(dependency_receipt_hashes),
            "by_model_slot": {
                model_slot: dict(dependencies)
                for model_slot, dependencies in sorted(dependencies_by_model.items())
            },
            "balanced": True,
        },
        "balance_receipts": {
            "count": len(balance_receipt_hashes),
            "receipt_sha256s": sorted(balance_receipt_hashes),
            "atomic_pair_inventory_complete": True,
            "same_origin_no_op_verified": True,
            "model_seed_replicate_balance_verified": True,
        },
        "treatments": treatments,
        "raw_metric_pooled_across_suites": False,
        "estimand_scope": "this arena and split only",
    }


def _cross_suite_robustness(
    suites: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    scorer_identities = {
        (
            str(suite["scorer_identity"]["scorer_id"]),
            str(suite["scorer_identity"]["scorer_revision"]),
        )
        for suite in suites.values()
    }
    schemas = {tuple(str(item) for item in suite["metric_schema"]) for suite in suites.values()}
    identities: dict[str, Mapping[str, Any]] = {}
    normalized: dict[str, dict[str, float]] = defaultdict(dict)
    directions: dict[str, dict[str, str]] = defaultdict(dict)
    for suite_id, suite in suites.items():
        for release_id, treatment in suite["treatments"].items():
            identity = {
                key: treatment[key] for key in ("release_id", "family", "profile_kind", "arm_kind")
            }
            prior = identities.get(release_id)
            if prior is None:
                identities[release_id] = identity
            elif prior != identity:
                raise ArtifactError(
                    f"cross-suite release {release_id!r} has mixed treatment identity"
                )
            normalized[release_id][suite_id] = float(treatment["normalized_win_minus_harm"])
            effect = float(treatment["mean_effect_vs_no_op"])
            directions[release_id][suite_id] = (
                "positive" if effect > 0 else "negative" if effect < 0 else "tie"
            )

    by_treatment: dict[str, Mapping[str, Any]] = {}
    for release_id in sorted(normalized):
        values = tuple(normalized[release_id].values())
        direction_values = tuple(directions[release_id].values())
        by_treatment[release_id] = {
            **identities[release_id],
            "suite_count": len(values),
            "direction_counts": {
                direction: sum(value == direction for value in direction_values)
                for direction in ("positive", "tie", "negative")
            },
            "direction_by_suite": dict(sorted(directions[release_id].items())),
            "normalized_win_minus_harm_by_suite": dict(sorted(normalized[release_id].items())),
            "normalized_heterogeneity_range": (
                max(values) - min(values) if len(values) > 1 else None
            ),
            "direction_consistent_across_suites": (
                len({value for value in direction_values if value != "tie"}) <= 1
            ),
        }
    return {
        "summary_scope": (
            "arena-level direction and win-minus-harm diagnostics only; "
            "no official raw metric is pooled"
        ),
        "suite_estimands_retained": True,
        "raw_metric_pooling_performed": False,
        "metric_schemas_identical": len(schemas) == 1,
        "official_scorer_identities_identical": len(scorer_identities) == 1,
        "incompatible_scorers_or_schemas_are_stratified": True,
        "by_treatment": by_treatment,
    }


def _validate_atomic_pairs(
    payload: Mapping[str, Any],
    *,
    outcomes: Mapping[str, Mapping[str, Any]],
    planned_arm_ids: tuple[str, ...],
    baseline_arm_id: str,
    suite_id: str,
    unit_id: str,
    seed: int,
    source_revision: str,
    checkpoint: Mapping[str, str],
    dependencies: Mapping[str, str],
    scorer_identity: tuple[str, str],
    baseline_consistency: str,
) -> tuple[Mapping[str, Any], ...]:
    raw_pairs = _sequence(
        payload.get("atomic_pairs"),
        label="external arena atomic pair receipts",
    )
    expected_treatments = tuple(arm_id for arm_id in planned_arm_ids if arm_id != baseline_arm_id)
    if len(raw_pairs) != len(expected_treatments):
        raise ArtifactError("external arena atomic pair receipt inventory is partial")

    parsed: list[Mapping[str, Any]] = []
    runtime_instances: set[str] = set()
    receipt_hashes: list[str] = []
    baseline_outcome = outcomes[baseline_arm_id]
    baseline_metrics = _metrics(
        baseline_outcome.get("metrics"),
        label="external arena NO_OP metrics",
    )
    raw_metric_schema = tuple(
        key for key in sorted(baseline_metrics) if key != _VALID_SUCCESS_METRIC
    )
    for index, raw_pair in enumerate(raw_pairs):
        pair = _strict_object(
            raw_pair,
            label=f"external arena atomic pair {index}",
            keys={
                "treatment_arm_id",
                "request_sha256",
                "receipt",
                "receipt_sha256",
                "baseline_consistency_sha256",
                "metric_deltas",
                "resource_receipt_sha256s",
            },
        )
        treatment_id = _text(
            pair["treatment_arm_id"],
            label=f"external arena atomic pair {index} treatment_arm_id",
        )
        if treatment_id != expected_treatments[index]:
            raise ArtifactError(
                "external arena atomic pairs differ from the planned treatment order"
            )
        _sha256(
            pair["request_sha256"],
            label=f"external arena atomic pair {index} request receipt",
        )
        receipt = _atomic_replay_receipt(
            pair["receipt"],
            label=f"external arena atomic pair {index} receipt",
        )
        receipt_sha256 = _sha256(
            pair["receipt_sha256"],
            label=f"external arena atomic pair {index} receipt_sha256",
        )
        if content_hash(pair["receipt"]) != receipt_sha256:
            raise ArtifactError(f"external arena atomic pair {index} receipt checksum mismatch")
        if pair["baseline_consistency_sha256"] != baseline_consistency:
            raise ArtifactError("external arena atomic pair baseline consistency receipt changed")
        if (
            receipt.benchmark_id != suite_id
            or receipt.checkpoint_id != unit_id
            or receipt.source_revision != source_revision
            or receipt.origin_state_hash != checkpoint["origin_state_sha256"]
            or receipt.replay_identity != checkpoint["replay_identity_sha256"]
            or receipt.seed != seed
            or dict(receipt.locked_dependency_hashes) != dict(dependencies)
            or (receipt.scorer_id, receipt.scorer_revision) != scorer_identity
        ):
            raise ArtifactError(
                "external arena atomic receipt source, replay, dependency, or scorer "
                "identity differs from the result"
            )
        by_role = {arm.role: arm for arm in receipt.arms}
        if (
            by_role["baseline"].arm_id != baseline_arm_id
            or by_role["treatment"].arm_id != treatment_id
        ):
            raise ArtifactError("external arena atomic receipt arms differ from the plan")
        for arm_receipt in receipt.arms:
            if arm_receipt.runtime_instance_id in runtime_instances:
                raise ArtifactError("external arena atomic pairs reuse a runtime instance identity")
            runtime_instances.add(arm_receipt.runtime_instance_id)

        deltas = _metrics(
            pair["metric_deltas"],
            label=f"external arena atomic pair {index} metric deltas",
        )
        if tuple(sorted(deltas)) != raw_metric_schema:
            raise ArtifactError(
                "external arena atomic delta schema differs from official arm metrics"
            )
        treatment_metrics = _metrics(
            outcomes[treatment_id].get("metrics"),
            label=f"external arena treatment {treatment_id!r} metrics",
        )
        for name, delta in deltas.items():
            if not math.isclose(
                delta,
                treatment_metrics[name] - baseline_metrics[name],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ArtifactError(f"external arena atomic metric delta {name!r} is inconsistent")
        resource_hashes = _text_sequence(
            pair["resource_receipt_sha256s"],
            label=f"external arena atomic pair {index} resource receipt hashes",
            allow_empty=True,
        )
        for digest in resource_hashes:
            _sha256(
                digest,
                label=f"external arena atomic pair {index} resource receipt",
            )
        expected_consistency = _baseline_consistency_hash(
            baseline_outcome,
            receipt=receipt,
        )
        if expected_consistency != baseline_consistency:
            raise ArtifactError(
                "external arena NO_OP metrics, trace, usage, or logical receipt "
                "differs from its baseline consistency hash"
            )
        receipt_hashes.append(receipt_sha256)
        parsed.append(
            {
                "treatment_arm_id": treatment_id,
                "receipt_sha256": receipt_sha256,
                "bridge_id": receipt.bridge_id,
                "bridge_revision": receipt.bridge_revision,
            }
        )

    baseline_hashes = _text_sequence(
        outcomes[baseline_arm_id].get("atomic_pair_receipt_sha256s"),
        label="external arena NO_OP atomic receipt inventory",
    )
    if baseline_hashes != tuple(receipt_hashes):
        raise ArtifactError("external arena NO_OP receipt inventory differs from atomic pairs")
    for treatment_id, receipt_sha256 in zip(
        expected_treatments,
        receipt_hashes,
        strict=True,
    ):
        if outcomes[treatment_id].get("atomic_pair_receipt_sha256") != receipt_sha256:
            raise ArtifactError(
                f"external arena treatment {treatment_id!r} receipt binding differs "
                "from its atomic pair"
            )
    bridge_identities = {(str(item["bridge_id"]), str(item["bridge_revision"])) for item in parsed}
    if len(bridge_identities) != 1:
        raise ArtifactError("external arena bridge identity changes across atomic pairs")
    return tuple(parsed)


def _validate_valid_success_composition(
    outcome: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
    suite_id: str,
    arm_id: str,
) -> Mapping[str, Any]:
    composition = _strict_object(
        outcome.get("valid_success_composition"),
        label=f"external arena arm {arm_id!r} valid-success composition",
        keys={
            "schema_version",
            "formula_id",
            "benchmark_id",
            "arm_id",
            "official_success_metric",
            "components",
            "valid_success",
            "outcome_contract_sha256",
            "scorer_id",
            "scorer_revision",
            "trace_hash",
            "composition_receipt_sha256",
        },
    )
    if (
        composition["schema_version"] != 1
        or composition["formula_id"] != _VALID_SUCCESS_FORMULA
        or composition["benchmark_id"] != suite_id
        or composition["arm_id"] != arm_id
    ):
        raise ArtifactError(
            f"external arena arm {arm_id!r} valid-success receipt identity is invalid"
        )
    official_metric = _text(
        composition["official_success_metric"],
        label=f"external arena arm {arm_id!r} official metric",
    )
    components = _strict_object(
        composition["components"],
        label=f"external arena arm {arm_id!r} valid-success components",
        keys={
            "official_success",
            "contract_preserved",
            "privileged_leakage",
        },
    )
    official = _unit_interval(
        components["official_success"],
        label=f"external arena arm {arm_id!r} official success",
    )
    preserved = _unit_interval(
        components["contract_preserved"],
        label=f"external arena arm {arm_id!r} contract preservation",
    )
    leakage = _unit_interval(
        components["privileged_leakage"],
        label=f"external arena arm {arm_id!r} privileged leakage",
    )
    valid_success = _unit_interval(
        composition["valid_success"],
        label=f"external arena arm {arm_id!r} valid success",
    )
    if (
        official_metric not in metrics
        or not math.isclose(metrics[official_metric], official, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            metrics.get("contract_preserved", math.nan),
            preserved,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            metrics.get("privileged_leakage", math.nan),
            leakage,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            metrics.get(_VALID_SUCCESS_METRIC, math.nan),
            valid_success,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            official * preserved * (1.0 - leakage),
            valid_success,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ArtifactError(
            f"external arena arm {arm_id!r} valid-success composition is inconsistent"
        )
    _sha256(
        composition["outcome_contract_sha256"],
        label=f"external arena arm {arm_id!r} outcome contract receipt",
    )
    _text(
        composition["scorer_id"],
        label=f"external arena arm {arm_id!r} scorer_id",
    )
    _sha256(
        composition["scorer_revision"],
        label=f"external arena arm {arm_id!r} scorer_revision",
    )
    trace_hash = _sha256(
        composition["trace_hash"],
        label=f"external arena arm {arm_id!r} trace hash",
    )
    if outcome.get("trace_hash") != trace_hash:
        raise ArtifactError(f"external arena arm {arm_id!r} trace differs from its scorer receipt")
    usage = _object(
        outcome.get("usage"),
        label=f"external arena arm {arm_id!r} usage",
    )
    for name, value in usage.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ArtifactError(f"external arena arm {arm_id!r} usage receipt is invalid")
    expected_hash = _sha256(
        composition["composition_receipt_sha256"],
        label=f"external arena arm {arm_id!r} composition receipt",
    )
    body = {key: value for key, value in composition.items() if key != "composition_receipt_sha256"}
    if content_hash(body) != expected_hash:
        raise ArtifactError(f"external arena arm {arm_id!r} composition receipt checksum mismatch")
    return composition


def _optional_gate_receipt(
    value: Any,
    *,
    manifest: ExperimentManifest,
    suite_id: str,
    split: str,
    source_revision: str,
    scorer_identity: tuple[str, str],
    atomic_replay_contract: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    receipt = _strict_object(
        value,
        label="external arena gate receipt",
        keys={
            "file_sha256",
            "receipt_sha256",
            "scope_sha256",
            "benchmark_id",
            "split",
            "source_revision",
            "manifest_hash",
            "driver_config_sha256",
            "bridge",
            "scorer",
            "approval",
            "verification",
            "unit_inventory_sha256",
        },
    )
    for name in (
        "file_sha256",
        "receipt_sha256",
        "scope_sha256",
        "manifest_hash",
        "driver_config_sha256",
        "unit_inventory_sha256",
    ):
        _sha256(receipt[name], label=f"external arena gate {name}")
    if (
        receipt["benchmark_id"] != suite_id
        or receipt["split"] != split
        or receipt["source_revision"] != source_revision
        or receipt["manifest_hash"] != manifest.manifest_hash
    ):
        raise ArtifactError("external arena gate receipt is bound to another source or manifest")
    bridge = _strict_object(
        receipt["bridge"],
        label="external arena gate bridge",
        keys={"bridge_id", "bridge_revision"},
    )
    _text(bridge["bridge_id"], label="external arena gate bridge_id")
    _sha256(bridge["bridge_revision"], label="external arena gate bridge_revision")
    scorer = _strict_object(
        receipt["scorer"],
        label="external arena gate scorer",
        keys={"scorer_id", "scorer_revision"},
    )
    if (
        scorer.get("scorer_id"),
        scorer.get("scorer_revision"),
    ) != scorer_identity:
        raise ArtifactError(
            "external arena gate official scorer differs from arm composition receipts"
        )
    _text(scorer["scorer_id"], label="external arena gate scorer_id")
    _sha256(scorer["scorer_revision"], label="external arena gate scorer_revision")
    verification = _strict_object(
        receipt["verification"],
        label="external arena gate verification",
        keys={
            "atomic_replay_contract",
            "reset_mode",
            "reset_verified",
            "reset_receipt_sha256",
            "no_op_parity_verified",
            "no_op_parity_receipt_sha256",
        },
    )
    if (
        verification["atomic_replay_contract"] != atomic_replay_contract
        or verification["reset_verified"] is not True
        or verification["no_op_parity_verified"] is not True
    ):
        raise ArtifactError("external arena gate does not approve atomic reset and NO_OP parity")
    _text(verification["reset_mode"], label="external arena gate reset_mode")
    _sha256(
        verification["reset_receipt_sha256"],
        label="external arena gate reset receipt",
    )
    _sha256(
        verification["no_op_parity_receipt_sha256"],
        label="external arena gate NO_OP parity receipt",
    )
    approval = _strict_object(
        receipt["approval"],
        label="external arena gate approval",
        keys={
            "approved",
            "approver_id",
            "approval_revision",
            "approved_at",
            "scope_sha256",
        },
    )
    if approval["approved"] is not True or approval["scope_sha256"] != receipt["scope_sha256"]:
        raise ArtifactError("external arena gate receipt is not approved for this scope")
    _text(approval["approver_id"], label="external arena gate approver_id")
    _sha256(
        approval["approval_revision"],
        label="external arena gate approval revision",
    )
    _text(approval["approved_at"], label="external arena gate approval timestamp")
    return receipt


def _checkpoint_identity(value: Any, *, source_revision: str) -> Mapping[str, str]:
    checkpoint = _strict_object(
        value,
        label="external arena checkpoint identity",
        keys={
            "identity_sha256",
            "group_sha256",
            "visible_state_sha256",
            "origin_state_sha256",
            "replay_identity_sha256",
            "source_revision",
        },
    )
    if checkpoint["source_revision"] != source_revision:
        raise ArtifactError("external arena checkpoint source differs from the manifest")
    for name in (
        "identity_sha256",
        "visible_state_sha256",
        "origin_state_sha256",
        "replay_identity_sha256",
    ):
        _sha256(checkpoint[name], label=f"external arena checkpoint {name}")
    group_id = checkpoint["group_sha256"]
    if (
        not isinstance(group_id, str)
        or not group_id.startswith("group_")
        or len(group_id) != 70
        or not _is_sha256(group_id[6:])
    ):
        raise ArtifactError("external arena checkpoint group receipt must be group_<sha256>")
    return checkpoint


def _atomic_replay_receipt(value: Any, *, label: str) -> AtomicReplayReceipt:
    raw = _strict_object(
        value,
        label=label,
        keys={
            "benchmark_id",
            "checkpoint_id",
            "source_revision",
            "bridge_id",
            "bridge_revision",
            "scorer_id",
            "scorer_revision",
            "origin_state_hash",
            "replay_identity",
            "seed",
            "locked_dependency_hashes",
            "arms",
        },
    )
    raw_arms = _sequence(raw["arms"], label=f"{label} arms")
    if len(raw_arms) != 2:
        raise ArtifactError(f"{label} requires exactly two arm receipts")
    arms: list[AtomicReplayArmReceipt] = []
    for index, value_arm in enumerate(raw_arms):
        arm = _strict_object(
            value_arm,
            label=f"{label} arm {index}",
            keys={
                "role",
                "arm_id",
                "origin_state_hash",
                "replay_identity",
                "seed",
                "locked_dependency_hashes",
                "runtime_instance_id",
                "completed",
            },
        )
        try:
            arms.append(
                AtomicReplayArmReceipt(
                    role=arm["role"],
                    arm_id=arm["arm_id"],
                    origin_state_hash=arm["origin_state_hash"],
                    replay_identity=arm["replay_identity"],
                    seed=arm["seed"],
                    locked_dependency_hashes=arm["locked_dependency_hashes"],
                    runtime_instance_id=arm["runtime_instance_id"],
                    completed=arm["completed"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"{label} arm {index} is invalid: {exc}") from exc
    try:
        return AtomicReplayReceipt(
            benchmark_id=raw["benchmark_id"],
            checkpoint_id=raw["checkpoint_id"],
            source_revision=raw["source_revision"],
            bridge_id=raw["bridge_id"],
            bridge_revision=raw["bridge_revision"],
            scorer_id=raw["scorer_id"],
            scorer_revision=raw["scorer_revision"],
            origin_state_hash=raw["origin_state_hash"],
            replay_identity=raw["replay_identity"],
            seed=raw["seed"],
            locked_dependency_hashes=raw["locked_dependency_hashes"],
            arms=tuple(arms),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} is invalid: {exc}") from exc


def _baseline_consistency_hash(
    baseline_outcome: Mapping[str, Any],
    *,
    receipt: AtomicReplayReceipt,
) -> str:
    metrics = {
        key: value
        for key, value in _metrics(
            baseline_outcome.get("metrics"),
            label="external arena NO_OP metrics",
        ).items()
        if key != _VALID_SUCCESS_METRIC
    }
    baseline_receipts = tuple(arm for arm in receipt.arms if arm.role == "baseline")
    if len(baseline_receipts) != 1:
        raise ArtifactError("external arena atomic pair lacks one NO_OP receipt")
    arm = baseline_receipts[0]
    # ``ResourceUsage.as_dict()`` adds the derived ``total_cost_usd`` property,
    # while the driver's logical baseline hash is computed from the dataclass
    # fields.  Remove only that deterministic projection before recomputing the
    # receipt; every primitive usage field remains covered.
    usage = dict(
        _object(
            baseline_outcome.get("usage"),
            label="external arena NO_OP usage",
        )
    )
    usage.pop("total_cost_usd", None)
    return content_hash(
        {
            "metrics": metrics,
            "trace_hash": baseline_outcome.get("trace_hash"),
            "usage": usage,
            "receipt": {
                "benchmark_id": receipt.benchmark_id,
                "checkpoint_id": receipt.checkpoint_id,
                "source_revision": receipt.source_revision,
                "bridge_id": receipt.bridge_id,
                "bridge_revision": receipt.bridge_revision,
                "scorer_id": receipt.scorer_id,
                "scorer_revision": receipt.scorer_revision,
                "origin_state_hash": receipt.origin_state_hash,
                "replay_identity": receipt.replay_identity,
                "seed": receipt.seed,
                "locked_dependency_hashes": receipt.locked_dependency_hashes,
                "baseline_arm": {
                    "role": arm.role,
                    "arm_id": arm.arm_id,
                    "origin_state_hash": arm.origin_state_hash,
                    "replay_identity": arm.replay_identity,
                    "seed": arm.seed,
                    "locked_dependency_hashes": arm.locked_dependency_hashes,
                    "completed": arm.completed,
                },
            },
        }
    )


def _validate_cross_model_dependency_balance(
    suite_id: str,
    *,
    dependencies_by_model: Mapping[str, Mapping[str, str]],
) -> None:
    if not dependencies_by_model:
        raise ArtifactError(f"stateful suite {suite_id!r} has no dependency receipts")
    for slot in sorted(_NON_MODEL_DEPENDENCY_SLOTS):
        values = {str(dependencies[slot]) for dependencies in dependencies_by_model.values()}
        if len(values) != 1:
            raise ArtifactError(f"stateful suite {suite_id!r} mixes non-model dependency {slot!r}")


def _require_payload_identity(
    payload: Mapping[str, Any],
    *,
    suite_id: str,
    split: str,
    unit_id: str,
    model_slot: str,
    seed: int,
    planned_arm_ids: tuple[str, ...],
) -> None:
    expected = {
        "benchmark_id": suite_id,
        "split": split,
        "unit_id": unit_id,
        "model_slot": model_slot,
        "seed": seed,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ArtifactError(f"external arena payload {name} differs from its frozen cell")
    if (
        _text_sequence(
            payload.get("planned_arm_ids"),
            label="external arena payload planned_arm_ids",
        )
        != planned_arm_ids
    ):
        raise ArtifactError("external arena payload arm inventory differs from its cell")


def _validate_not_applicable_slots(
    payload: Mapping[str, Any],
    *,
    dependencies: Mapping[str, str],
) -> None:
    slots = _text_sequence(
        payload.get("not_applicable_dependency_slots"),
        label="external arena not-applicable dependency slots",
        allow_empty=True,
    )
    if tuple(sorted(slots)) != slots or len(slots) != len(set(slots)):
        raise ArtifactError(
            "external arena not-applicable dependency slots must be sorted and unique"
        )
    for slot in slots:
        if slot not in _DEPENDENCY_SLOTS:
            raise ArtifactError(
                "external arena not-applicable dependency receipt contains an unknown slot"
            )
        expected = content_hash(
            {
                "agentinputlab_dependency": "not_applicable",
                "schema_version": 1,
                "slot": slot,
            }
        )
        if dependencies[slot] != expected:
            raise ArtifactError(
                f"external arena not-applicable dependency {slot!r} has another hash"
            )


def _manifest_target(
    manifest: ExperimentManifest,
    *,
    suite_id: str,
    split: str,
) -> Any:
    targets = tuple(
        target
        for target in manifest.benchmarks
        if target.cell_id == suite_id and target.split == split
    )
    if len(targets) != 1:
        raise ArtifactError("external arena result has no unique source-locked manifest target")
    return targets[0]


def _dependency_hashes(
    value: Any,
    *,
    label: str,
    exact_slots: bool,
) -> Mapping[str, str]:
    dependencies = _object(value, label=label)
    if exact_slots and set(dependencies) != _DEPENDENCY_SLOTS:
        raise ArtifactError(f"{label} must contain the exact shared dependency slots")
    parsed: dict[str, str] = {}
    for name, digest in dependencies.items():
        if not isinstance(name, str) or not name:
            raise ArtifactError(f"{label} contains an invalid slot")
        parsed[name] = _sha256(digest, label=f"{label} slot {name!r}")
    if not parsed:
        raise ArtifactError(f"{label} must not be empty")
    return parsed


def _metrics(value: Any, *, label: str) -> Mapping[str, float]:
    raw = _object(value, label=label)
    if not raw:
        raise ArtifactError(f"{label} must not be empty")
    metrics: dict[str, float] = {}
    for name, value_item in raw.items():
        if not isinstance(name, str) or not name:
            raise ArtifactError(f"{label} contains an invalid metric name")
        if (
            isinstance(value_item, bool)
            or not isinstance(value_item, int | float)
            or not math.isfinite(float(value_item))
        ):
            raise ArtifactError(f"{label} metric {name!r} must be finite numeric")
        metrics[name] = float(value_item)
    return metrics


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    estimates = [fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    ordered = sorted(estimates)
    lower = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return lower, upper


def _run_roots(run_dirs: Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(run_dirs, str | bytes) or not isinstance(run_dirs, Sequence):
        raise ValidationError("run_dirs must be a sequence of paths")
    roots = tuple(Path(path).expanduser().resolve() for path in run_dirs)
    if not roots:
        raise ValidationError("stateful arena analysis requires at least one run directory")
    if len(roots) != len(set(roots)):
        raise ValidationError("stateful arena run directories must be unique")
    return roots


def _resample_arguments(*, bootstrap_samples: int, random_seed: int) -> None:
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 100
    ):
        raise ValidationError("bootstrap_samples must be an integer >= 100")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValidationError("random_seed must be a non-negative integer")


def _metric_name(value: Any) -> str:
    return _text(value, label="stateful arena analysis metric")


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is missing or invalid") from exc
    return _object(value, label=label)


def _strict_object(value: Any, *, label: str, keys: set[str]) -> Mapping[str, Any]:
    result = _object(value, label=label)
    if set(result) != keys:
        missing = sorted(keys - set(result))
        unknown = sorted(set(result) - keys)
        raise ArtifactError(f"{label} fields are invalid; missing={missing}, unknown={unknown}")
    return result


def _object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ArtifactError(f"{label} keys must be strings")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{label} must be an array")
    return tuple(value)


def _text_sequence(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    items = _sequence(value, label=label)
    if not items and not allow_empty:
        raise ArtifactError(f"{label} must not be empty")
    return tuple(_text(item, label=f"{label}[{index}]") for index, item in enumerate(items))


def _text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(not character.isprintable() for character in value)
    ):
        raise ValidationError(f"{label} must be trimmed non-empty text")
    return value


def _seed(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{label} must be a non-negative integer")
    return value


def _unit_interval(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ArtifactError(f"{label} must be finite and in [0, 1]")
    return float(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _is_sha256(value):
        raise ArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "STATEFUL_ARENA_REPLICATION_ARTIFACT",
    "STATEFUL_ARENA_REPLICATION_PROTOCOL",
    "analyze_stateful_arena_replication",
    "validate_external_arena_record_receipts",
]
