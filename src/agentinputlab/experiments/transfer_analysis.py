"""Fail-closed analysis for the frozen held-out-model transfer block.

This module deliberately accepts only the generated robustness protocol:
200 STATE-Bench groups, six rank-1 representative arms, four frozen transfer
models, and one continuation seed.  A checksum-complete run is necessary but not
sufficient.  The scientific boundary also rechecks the persisted prerequisite
binding receipt, driver arm/source identity, and per-cell dependency and
atomic-replay receipts before reading outcomes.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from agentinputlab.benchmarks import (
    ATOMIC_EXTERNAL_PAIR_CONTRACT,
    AtomicReplayArmReceipt,
    AtomicReplayReceipt,
)
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable
from agentinputlab.metrics.paired import summarize_pairs

from .analysis import (
    _arm_metrics,
    _cluster_bootstrap_interval,
    _cluster_id,
    _load_manifest,
    _primary_families,
    _required_record_text,
    _validate_resample_arguments,
    load_verified_cell_results,
)
from .bindings import FrozenBindingEntry, FrozenBindingReceipt
from .paper_analysis import (
    _COMMON_SUPPORT_KIND,
    _load_and_validate_paper_run,
)
from .schema import ExperimentManifest, ExperimentStage, PrerequisiteKind

HELDOUT_MODEL_TRANSFER_PROTOCOL = "paper-heldout-model-transfer-v1"

_CENSUS_PROTOCOL = "agentinputlab.experiment_census.v1"
_CAMPAIGN_KIND = "heldout_model_transfer"
_STUDY_ID = "robustness.heldout_model_transfer"
_BENCHMARK_ID = "state_bench.common_support"
_TRANSFER_SPLIT = "heldout_model_transfer"
_TRANSFER_MODEL_SLOTS = (
    "qwen3_4b",
    "deepseek_r1_distill_qwen_7b",
    "claude_sonnet_4_6",
    "deepseek_v4_flash",
)
_PRIMARY_MODEL_SLOTS = ("anchor_a", "anchor_b")
_SEED = 17
_STATEBENCH_REVISION = "4efcbf2d4fe60df04878859b692d9391f3d5b33a"
_EXTERNAL_DRIVER = "agentinputlab.experiments.drivers.external_arena:factory"
_EXTERNAL_DRIVER_IMPLEMENTATION = "agentinputlab.external_arena.atomic_multi_arm"
_GUARD_ID = "agentinputlab.frozen_binding_driver_guard"
_CORE_FAMILIES = _primary_families()
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
_HELDOUT_PREREQUISITES = {
    "AILAB_STATEBENCH_CHECKOUT",
    "AILAB_REPRESENTATIVE_MANIFEST",
    "AILAB_EXTERNAL_ARENA_DRIVER_CONFIG",
    "AILAB_MODEL_CONFIG",
    "AILAB_RETRIEVER_CONFIG",
    "AILAB_HELDOUT_MODEL_FREEZE_RECEIPT",
}
_PRIMARY_PREREQUISITES = {
    "AILAB_TAU3_CHECKOUT",
    "AILAB_STATEBENCH_CHECKOUT",
    "AILAB_TOOLSANDBOX_CHECKOUT",
    "AILAB_TOOLSANDBOX_GATE_RECEIPT",
    "AILAB_COMMON_SUPPORT_FREEZE_RECEIPT",
    "AILAB_EXTERNAL_ARENA_DRIVER_CONFIG",
    "AILAB_MODEL_CONFIG",
    "AILAB_RETRIEVER_CONFIG",
    "AILAB_REPRESENTATIVE_MANIFEST",
}
_PRIMARY_TARGET_COUNTS = {
    "tau3.banking_knowledge:common_support_lockbox": 97,
    "state_bench.common_support:common_support_lockbox": 300,
    "toolsandbox.common_support:common_support_lockbox": 129,
}
_PRIMARY_SOURCE_REVISIONS = {
    "tau3.banking_knowledge": "cf71a8070269883e38a365ffa85f78f46844c1f4",
    "state_bench.common_support": _STATEBENCH_REVISION,
    "toolsandbox.common_support": "165848b9a78cead7ca7fe7c89c688b58e6501219",
}


@dataclass(frozen=True, slots=True)
class _VerifiedTransferRun:
    root: Path
    manifest: ExperimentManifest
    records: tuple[Mapping[str, Any], ...]
    clusters: frozenset[str]
    representative_hash: str
    baseline_arm_id: str
    arms_by_family: Mapping[SemanticFamily, str]
    binding_receipt: Mapping[str, Any]
    runtime_receipts: Mapping[str, Any]


def validate_heldout_model_transfer_run(run_dir: str | Path) -> Mapping[str, Any]:
    """Validate the exact frozen transfer run without computing outcome effects."""

    run = _load_and_validate_heldout_run(run_dir)
    return {
        "analysis_protocol": HELDOUT_MODEL_TRANSFER_PROTOCOL,
        "study_id": run.manifest.study_id,
        "manifest_hash": run.manifest.manifest_hash,
        "campaign_kind": _CAMPAIGN_KIND,
        "verified_complete_run": True,
        "independent_groups": 200,
        "verified_plan_cells": 800,
        "verified_agent_episodes": 4_800,
        "model_slots": list(_TRANSFER_MODEL_SLOTS),
        "representative_binding": dict(run.binding_receipt["representative_binding"]),
        "runtime_receipts": dict(run.runtime_receipts),
        "experiments_executed_by_analysis": False,
        "paper_claim_authorized": False,
    }


def analyze_heldout_model_transfer(
    run_dir: str | Path,
    *,
    metric: str,
    primary_common_support_run_dir: str | Path | None = None,
    algorithm_failure_value: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Estimate each rank-1 family representative versus matched NO_OP.

    When a primary common-support run is supplied, the additional boundary is
    a matched difference-in-differences on the 200 underlying STATE-Bench
    units.  It is not a pooled common-population effect.
    """

    _validate_analysis_arguments(
        metric=metric,
        algorithm_failure_value=algorithm_failure_value,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    transfer = _load_and_validate_heldout_run(run_dir)
    transfer_rows = _collapsed_family_rows(
        transfer.records,
        manifest=transfer.manifest,
        baseline_arm_id=transfer.baseline_arm_id,
        arms_by_family=transfer.arms_by_family,
        metric=metric,
        algorithm_failure_value=algorithm_failure_value,
    )
    if len(transfer_rows) != 200:
        raise ArtifactError("heldout-model transfer analysis requires exactly 200 clusters")

    rng = random.Random(random_seed)
    family_summaries = _family_summaries(
        transfer_rows,
        arms_by_family=transfer.arms_by_family,
        manifest=transfer.manifest,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    model_summaries = {
        model_slot: _family_summaries(
            _collapsed_family_rows(
                tuple(
                    record
                    for record in transfer.records
                    if record["cell"]["model_slot"] == model_slot
                ),
                manifest=transfer.manifest,
                baseline_arm_id=transfer.baseline_arm_id,
                arms_by_family=transfer.arms_by_family,
                metric=metric,
                algorithm_failure_value=algorithm_failure_value,
            ),
            arms_by_family=transfer.arms_by_family,
            manifest=transfer.manifest,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for model_slot in _TRANSFER_MODEL_SLOTS
    }
    comparison: Mapping[str, Any] | None = None
    if primary_common_support_run_dir is not None:
        primary = _load_and_validate_primary_run(primary_common_support_run_dir)
        if primary.root == transfer.root:
            raise ArtifactError("transfer and primary comparison require different run directories")
        _require_same_representatives(transfer, primary)
        primary_rows = _collapsed_family_rows(
            primary.records,
            manifest=primary.manifest,
            baseline_arm_id=primary.baseline_arm_id,
            arms_by_family=primary.arms_by_family,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
        )
        comparison = _transfer_primary_boundary(
            transfer_rows,
            primary_rows,
            transfer=transfer,
            primary=primary,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "heldout_model_transfer_family_vs_no_op",
        "analysis_protocol": HELDOUT_MODEL_TRANSFER_PROTOCOL,
        "metric": metric,
        "run_count": 1 + int(primary_common_support_run_dir is not None),
        "study_id": transfer.manifest.study_id,
        "manifest_hash": transfer.manifest.manifest_hash,
        "verified_complete_run": True,
        "independent_groups": 200,
        "verified_plan_cells": 800,
        "verified_agent_episodes": 4_800,
        "arm_layout": "NO_OP + one frozen rank-1 representative per core family",
        "model_slots": list(_TRANSFER_MODEL_SLOTS),
        "seed": _SEED,
        "cluster_definition": "benchmark_id/split/unit_id",
        "family_summaries": family_summaries,
        "family_summaries_by_model": model_summaries,
        "representative_binding": dict(transfer.binding_receipt["representative_binding"]),
        "runtime_receipts": dict(transfer.runtime_receipts),
        "primary_common_support_boundary": comparison,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "algorithm_failure_value": algorithm_failure_value,
        "scope_note": (
            "Transfer estimates are matched family-versus-NO_OP effects on the "
            "prespecified 200-group four-model block. The aggregate gives every model "
            "equal weight. Any optional primary comparison "
            "is a matched transfer-minus-primary interaction boundary, not a pooled "
            "effect or evidence that the populations are exchangeable."
        ),
        "experiments_executed_by_analysis": False,
        "paper_claim_authorized": False,
    }
    return result


def _load_and_validate_heldout_run(run_dir: str | Path) -> _VerifiedTransferRun:
    root = Path(run_dir).expanduser().resolve()
    manifest = _load_manifest(root)
    records = load_verified_cell_results(root)
    baseline, arms_by_family, representative_hash = _heldout_manifest_contract(manifest)
    clusters = _validate_exact_grid(
        records,
        manifest=manifest,
        expected_groups=200,
        expected_cells=800,
        expected_episodes=4_800,
    )
    binding_receipt, operator = _validate_persisted_run_identity(root, manifest)
    _validate_operator_identity(operator, manifest)
    runtime_receipts = _validate_external_runtime_receipts(records, manifest)
    return _VerifiedTransferRun(
        root=root,
        manifest=manifest,
        records=records,
        clusters=clusters,
        representative_hash=representative_hash,
        baseline_arm_id=baseline,
        arms_by_family=arms_by_family,
        binding_receipt=binding_receipt,
        runtime_receipts=runtime_receipts,
    )


def _load_and_validate_primary_run(run_dir: str | Path) -> _VerifiedTransferRun:
    root = Path(run_dir).expanduser().resolve()
    paper_run = _load_and_validate_paper_run(root, expected_kind=_COMMON_SUPPORT_KIND)
    _primary_common_support_contract(paper_run.manifest)
    baseline, arms_by_family, representative_hash = _rank_one_arm_contract(paper_run.manifest)
    _validate_primary_grid(paper_run.records, manifest=paper_run.manifest)
    binding_receipt, operator = _validate_persisted_run_identity(root, paper_run.manifest)
    _validate_operator_identity(operator, paper_run.manifest)
    runtime_receipts = _validate_external_runtime_receipts(
        paper_run.records,
        paper_run.manifest,
    )
    return _VerifiedTransferRun(
        root=root,
        manifest=paper_run.manifest,
        records=paper_run.records,
        clusters=paper_run.cluster_ids,
        representative_hash=representative_hash,
        baseline_arm_id=baseline,
        arms_by_family=arms_by_family,
        binding_receipt=binding_receipt,
        runtime_receipts=runtime_receipts,
    )


def _heldout_manifest_contract(
    manifest: ExperimentManifest,
) -> tuple[str, Mapping[SemanticFamily, str], str]:
    if (
        manifest.study_id != _STUDY_ID
        or manifest.stage is not ExperimentStage.TRANSFER
        or manifest.estimand_profile_kind is not ProfileKind.CONTROLLED_ADAPTER
        or manifest.driver_entrypoint != _EXTERNAL_DRIVER
        or manifest.model_slots != _TRANSFER_MODEL_SLOTS
        or manifest.seeds != (_SEED,)
    ):
        raise ArtifactError(
            "heldout-model transfer manifest differs from the frozen study/stage/model/seed"
        )
    if len(manifest.benchmarks) != 1:
        raise ArtifactError("heldout-model transfer requires exactly one benchmark target")
    target = manifest.benchmarks[0]
    if (
        target.cell_id != _BENCHMARK_ID
        or target.split != _TRANSFER_SPLIT
        or target.external is not True
        or target.source_revision != _STATEBENCH_REVISION
        or target.unit_inventory_env != "AILAB_HELDOUT_MODEL_UNITS"
        or target.unit_inventory_sha256_env != "AILAB_HELDOUT_MODEL_UNITS_SHA256"
        or target.arm_ids
    ):
        raise ArtifactError("heldout-model transfer benchmark/source identity is invalid")
    metadata = manifest.metadata
    if (
        metadata.get("model_used_for_tuning") is not False
        or metadata.get("group_relation") != "prespecified_subset_of_common_support_lockbox"
        or metadata.get("requires_atomic_external_replay") is not True
        or metadata.get("confirmatory_claimed_before_execution") is not False
    ):
        raise ArtifactError("heldout-model transfer freeze metadata is invalid")

    census = metadata.get("experiment_census")
    expected_census = {
        "protocol": _CENSUS_PROTOCOL,
        "campaign_kind": _CAMPAIGN_KIND,
        "independent_group_unit": "benchmark_cell:unit_id",
        "independent_group_target": 200,
        "target_group_counts": {
            f"{_BENCHMARK_ID}:{_TRANSFER_SPLIT}": 200,
        },
        "expected_arm_ids": [
            "no_op",
            *(f"{family.value}_representative" for family in _CORE_FAMILIES),
        ],
        "expected_model_slots": list(_TRANSFER_MODEL_SLOTS),
        "expected_seeds": [_SEED],
        "expected_plan_cells": 800,
        "expected_agent_episodes": 4_800,
        "episode_formula": ("sum(target_groups * target_arms * models * continuation_seeds)"),
        "seed_semantics": "one_primary_continuation_seed_per_group_model_arm",
    }
    if not isinstance(census, Mapping) or canonical_json(census) != canonical_json(expected_census):
        raise ArtifactError("heldout-model transfer census must be exactly 200x6x4x1=4800")
    prerequisite_names = {item.name for item in manifest.prerequisites}
    if prerequisite_names != _HELDOUT_PREREQUISITES:
        raise ArtifactError("heldout-model transfer prerequisite inventory is invalid")
    return _rank_one_arm_contract(manifest)


def _primary_common_support_contract(manifest: ExperimentManifest) -> None:
    if (
        manifest.study_id != "primary.common_support"
        or manifest.stage is not ExperimentStage.CONFIRMATORY
        or manifest.estimand_profile_kind is not ProfileKind.CONTROLLED_ADAPTER
        or manifest.driver_entrypoint != _EXTERNAL_DRIVER
        or manifest.model_slots != _PRIMARY_MODEL_SLOTS
        or manifest.seeds != (_SEED,)
    ):
        raise ArtifactError("primary comparison is not the frozen paper common-support run")
    targets = {f"{target.cell_id}:{target.split}": target for target in manifest.benchmarks}
    if set(targets) != set(_PRIMARY_TARGET_COUNTS):
        raise ArtifactError("primary common-support benchmark inventory is invalid")
    for key, target in targets.items():
        if (
            target.external is not True
            or target.source_revision != _PRIMARY_SOURCE_REVISIONS[target.cell_id]
            or target.unit_inventory_env is None
            or target.unit_inventory_sha256_env is None
            or target.arm_ids
        ):
            raise ArtifactError(f"primary common-support source identity is invalid for {key}")
    census = manifest.metadata.get("experiment_census")
    if (
        not isinstance(census, Mapping)
        or census.get("protocol") != _CENSUS_PROTOCOL
        or census.get("campaign_kind") != _COMMON_SUPPORT_KIND
        or census.get("independent_group_target") != 526
        or census.get("target_group_counts") != _PRIMARY_TARGET_COUNTS
        or tuple(census.get("expected_model_slots", ())) != _PRIMARY_MODEL_SLOTS
        or tuple(census.get("expected_seeds", ())) != (_SEED,)
        or census.get("expected_plan_cells") != 1_052
        or census.get("expected_agent_episodes") != 6_312
    ):
        raise ArtifactError("primary comparison differs from the strict paper census")
    if {item.name for item in manifest.prerequisites} != _PRIMARY_PREREQUISITES:
        raise ArtifactError("primary common-support prerequisite inventory is invalid")


def _rank_one_arm_contract(
    manifest: ExperimentManifest,
) -> tuple[str, Mapping[SemanticFamily, str], str]:
    expected_families = (SemanticFamily.NO_OP, *_CORE_FAMILIES)
    if (
        len(manifest.arms) != 6
        or tuple(arm.family for arm in manifest.arms) != expected_families
        or any(arm.profile_kind is not ProfileKind.CONTROLLED_ADAPTER for arm in manifest.arms)
    ):
        raise ArtifactError(
            "transfer/common-support arm roster requires NO_OP plus five core families"
        )
    baseline = manifest.arms[0]
    if (
        baseline.arm_id != "no_op"
        or baseline.release_id != "baseline.no_op"
        or baseline.parameters.get("role") != "paired_baseline"
    ):
        raise ArtifactError("transfer/common-support NO_OP arm identity is invalid")
    representative_hash = manifest.metadata.get("representative_manifest_hash")
    if not _is_sha256(representative_hash):
        raise ArtifactError("run lacks a valid representative manifest hash")
    arms_by_family: dict[SemanticFamily, str] = {}
    releases: set[str] = set()
    for arm, family in zip(manifest.arms[1:], _CORE_FAMILIES, strict=True):
        if (
            arm.arm_id != f"{family.value}_representative"
            or arm.family is not family
            or arm.release_id == "baseline.no_op"
            or arm.parameters.get("representative_rank") != 1
            or arm.parameters.get("representative_manifest_hash") != representative_hash
        ):
            raise ArtifactError(f"{family.value} arm differs from its frozen rank-1 representative")
        if arm.release_id in releases:
            raise ArtifactError("rank-1 representative release identities are not unique")
        releases.add(arm.release_id)
        arms_by_family[family] = arm.arm_id
    census = manifest.metadata.get("experiment_census")
    if not isinstance(census, Mapping) or tuple(census.get("expected_arm_ids", ())) != tuple(
        arm.arm_id for arm in manifest.arms
    ):
        raise ArtifactError("census arm identities differ from the representative roster")
    return baseline.arm_id, arms_by_family, representative_hash


def _validate_exact_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: ExperimentManifest,
    expected_groups: int,
    expected_cells: int,
    expected_episodes: int,
) -> frozenset[str]:
    if len(records) != expected_cells:
        raise ArtifactError(
            f"heldout-model transfer has {len(records)} verified cells, expected {expected_cells}"
        )
    expected_arms = tuple(arm.arm_id for arm in manifest.arms)
    clusters: set[str] = set()
    replicates_by_group: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    episodes = 0
    for record in records:
        cell = record.get("cell")
        if not isinstance(cell, Mapping):
            raise ArtifactError("heldout-model transfer record lacks an experiment cell")
        if (
            _required_record_text(cell, "benchmark_id", label="experiment cell") != _BENCHMARK_ID
            or _required_record_text(cell, "split", label="experiment cell") != _TRANSFER_SPLIT
            or _required_record_text(cell, "model_slot", label="experiment cell")
            not in _TRANSFER_MODEL_SLOTS
            or cell.get("seed") != _SEED
            or tuple(cell.get("arm_ids", ())) != expected_arms
        ):
            raise ArtifactError(
                "heldout-model transfer cell differs from its benchmark/model/seed/arm grid"
            )
        model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
        cluster = f"{_cluster_id(cell)}::{model_slot}::{_SEED}"
        if cluster in clusters:
            raise ArtifactError("heldout-model transfer duplicates a group/model replicate")
        clusters.add(cluster)
        group = (
            _required_record_text(cell, "benchmark_id", label="experiment cell"),
            _required_record_text(cell, "unit_id", label="experiment cell"),
        )
        replicate = (
            _required_record_text(cell, "model_slot", label="experiment cell"),
            int(cell["seed"]),
        )
        if replicate in replicates_by_group[group]:
            raise ArtifactError("heldout-model transfer duplicates a group/model replicate")
        replicates_by_group[group].add(replicate)
        episodes += len(expected_arms)
    expected_replicates = {(model_slot, _SEED) for model_slot in _TRANSFER_MODEL_SLOTS}
    if (
        len(replicates_by_group) != expected_groups
        or any(replicates != expected_replicates for replicates in replicates_by_group.values())
        or episodes != expected_episodes
    ):
        raise ArtifactError(
            "heldout-model transfer must contain 200 groups, four model replicates, "
            "and 4800 arm episodes"
        )
    return frozenset(clusters)


def _validate_primary_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: ExperimentManifest,
) -> None:
    if len(records) != 1_052:
        raise ArtifactError("primary common-support comparison requires exactly 1052 cells")
    expected_arms = tuple(arm.arm_id for arm in manifest.arms)
    groups_by_target: dict[str, dict[str, set[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        cell = record.get("cell")
        if not isinstance(cell, Mapping):
            raise ArtifactError("primary common-support record lacks an experiment cell")
        benchmark = _required_record_text(
            cell,
            "benchmark_id",
            label="experiment cell",
        )
        split = _required_record_text(cell, "split", label="experiment cell")
        unit_id = _required_record_text(cell, "unit_id", label="experiment cell")
        model_slot = _required_record_text(
            cell,
            "model_slot",
            label="experiment cell",
        )
        seed = cell.get("seed")
        key = f"{benchmark}:{split}"
        if (
            key not in _PRIMARY_TARGET_COUNTS
            or model_slot not in _PRIMARY_MODEL_SLOTS
            or seed != _SEED
            or tuple(cell.get("arm_ids", ())) != expected_arms
        ):
            raise ArtifactError(
                "primary common-support cell differs from its target/model/seed/arm grid"
            )
        replicate = (model_slot, int(seed))
        if replicate in groups_by_target[key][unit_id]:
            raise ArtifactError("primary common-support duplicates a group replicate")
        groups_by_target[key][unit_id].add(replicate)
    expected_replicates = {(model_slot, _SEED) for model_slot in _PRIMARY_MODEL_SLOTS}
    if set(groups_by_target) != set(_PRIMARY_TARGET_COUNTS):
        raise ArtifactError("primary common-support records omit a frozen target")
    for key, expected_groups in _PRIMARY_TARGET_COUNTS.items():
        groups = groups_by_target[key]
        if len(groups) != expected_groups or any(
            replicates != expected_replicates for replicates in groups.values()
        ):
            raise ArtifactError(f"primary common-support record census is invalid for {key}")


def _validate_persisted_run_identity(
    root: Path,
    manifest: ExperimentManifest,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    identity = _load_run_identity(root)
    fingerprint = identity.get("driver_fingerprint")
    if (
        not isinstance(fingerprint, Mapping)
        or set(fingerprint)
        != {
            "schema_version",
            "guard",
            "operator_driver_fingerprint",
            "frozen_binding_receipt",
            "frozen_binding_receipt_hash",
            "parity_contract",
        }
        or fingerprint.get("schema_version") != 1
        or fingerprint.get("guard") != _GUARD_ID
        or fingerprint.get("parity_contract") != {}
    ):
        raise ArtifactError("run lacks the frozen binding driver-guard identity")
    raw_receipt = fingerprint.get("frozen_binding_receipt")
    receipt = _parse_binding_receipt(raw_receipt)
    if (
        receipt.manifest_hash != manifest.manifest_hash
        or fingerprint.get("frozen_binding_receipt_hash") != receipt.receipt_hash
    ):
        raise ArtifactError("frozen binding receipt belongs to another manifest")
    declared = {(item.kind.value, item.name) for item in manifest.prerequisites}
    observed = {(item.kind, item.name) for item in receipt.bindings}
    if observed != declared:
        raise ArtifactError("frozen binding receipt differs from the manifest prerequisites")
    binding_by_name = {item.name: item for item in receipt.bindings}
    for prerequisite in manifest.prerequisites:
        if prerequisite.kind is not PrerequisiteKind.FILE_ENV:
            continue
        entry = binding_by_name[prerequisite.name]
        _file_binding_sha(entry, label=prerequisite.name)
        if entry.identity.get("checksum_environment") != prerequisite.sha256_env:
            raise ArtifactError(
                f"{prerequisite.name} binding differs from its checksum prerequisite"
            )
    representative = binding_by_name.get("AILAB_REPRESENTATIVE_MANIFEST")
    if representative is None or representative.kind != PrerequisiteKind.FILE_ENV.value:
        raise ArtifactError("run lacks the frozen representative file binding")
    representative_file_sha = _file_binding_sha(
        representative,
        label="representative manifest",
    )
    freeze_name = (
        "AILAB_HELDOUT_MODEL_FREEZE_RECEIPT"
        if manifest.stage is ExperimentStage.TRANSFER
        else "AILAB_COMMON_SUPPORT_FREEZE_RECEIPT"
    )
    freeze = binding_by_name.get(freeze_name)
    if freeze is None:
        raise ArtifactError(f"run lacks required source/freeze receipt {freeze_name}")
    freeze_file_sha = _file_binding_sha(freeze, label=freeze_name)
    driver_config = binding_by_name.get("AILAB_EXTERNAL_ARENA_DRIVER_CONFIG")
    if driver_config is None:
        raise ArtifactError("run lacks the frozen external driver configuration")
    driver_config_sha = _file_binding_sha(
        driver_config,
        label="external driver configuration",
    )
    _validate_checkout_bindings(binding_by_name, manifest)
    operator = fingerprint.get("operator_driver_fingerprint")
    if not isinstance(operator, Mapping):
        raise ArtifactError("run lacks an operator driver fingerprint")
    if operator.get("config_sha256") != driver_config_sha:
        raise ArtifactError(
            "operator driver identity differs from its frozen configuration binding"
        )
    return (
        {
            "frozen_binding_receipt_hash": receipt.receipt_hash,
            "representative_binding": {
                "representative_manifest_hash": manifest.metadata.get(
                    "representative_manifest_hash"
                ),
                "representative_file_sha256": representative_file_sha,
                "frozen_binding_receipt_hash": receipt.receipt_hash,
                "external_driver_config_sha256": driver_config_sha,
                "source_freeze_receipt_name": freeze_name,
                "source_freeze_receipt_sha256": freeze_file_sha,
                "binding_verified": True,
            },
        },
        operator,
    )


def _parse_binding_receipt(value: Any) -> FrozenBindingReceipt:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "manifest_hash",
            "bindings",
            "receipt_hash",
        }
        or not isinstance(value.get("bindings"), Sequence)
        or isinstance(value.get("bindings"), str | bytes)
    ):
        raise ArtifactError("frozen binding receipt schema is invalid")
    entries: list[FrozenBindingEntry] = []
    try:
        for raw in value["bindings"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "schema_version",
                "name",
                "kind",
                "identity",
                "binding_hash",
            }:
                raise ArtifactError("frozen binding entry schema is invalid")
            entry = FrozenBindingEntry(
                schema_version=raw["schema_version"],
                name=raw["name"],
                kind=raw["kind"],
                identity=raw["identity"],
            )
            if canonical_json(raw) != canonical_json(entry.to_dict()):
                raise ArtifactError("frozen binding entry hash differs from its content")
            entries.append(entry)
        receipt = FrozenBindingReceipt(
            schema_version=value["schema_version"],
            manifest_hash=value["manifest_hash"],
            bindings=tuple(entries),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ArtifactError("frozen binding receipt is malformed") from exc
    if canonical_json(value) != canonical_json(receipt.to_dict()):
        raise ArtifactError("frozen binding receipt hash differs from its content")
    return receipt


def _file_binding_sha(entry: FrozenBindingEntry, *, label: str) -> str:
    identity = entry.identity
    digest = identity.get("content_sha256")
    if identity.get("identity_mode") != "file_sha256" or not _is_sha256(digest):
        raise ArtifactError(f"{label} binding lacks a file SHA-256 receipt")
    return str(digest)


def _validate_checkout_bindings(
    bindings: Mapping[str, FrozenBindingEntry],
    manifest: ExperimentManifest,
) -> None:
    expected = manifest.metadata.get("checkout_expected_revisions")
    if not isinstance(expected, Mapping):
        raise ArtifactError("manifest checkout source locks are invalid")
    for name, revision in expected.items():
        entry = bindings.get(str(name))
        identity = entry.identity if entry is not None else None
        if (
            entry is None
            or entry.kind != PrerequisiteKind.DIRECTORY_ENV.value
            or not isinstance(identity, Mapping)
            or identity.get("identity_mode") != "clean_git_checkout"
            or identity.get("git_head") != revision
            or identity.get("expected_revision") != revision
            or identity.get("git_clean") is not True
        ):
            raise ArtifactError(f"source checkout receipt is invalid for {name}")


def _validate_operator_identity(
    operator: Mapping[str, Any],
    manifest: ExperimentManifest,
) -> None:
    if (
        operator.get("implementation_id") != _EXTERNAL_DRIVER_IMPLEMENTATION
        or operator.get("schema_version") != 3
        or operator.get("baseline_arm_id") != "no_op"
        or not _is_sha256(operator.get("driver_revision_sha256"))
        or not _is_sha256(operator.get("config_sha256"))
    ):
        raise ArtifactError("run operator is not the frozen atomic external-arena driver")
    raw_arms = operator.get("arms")
    if not isinstance(raw_arms, Mapping) or set(raw_arms) != {arm.arm_id for arm in manifest.arms}:
        raise ArtifactError("operator arm identities differ from the manifest")
    for arm in manifest.arms:
        identity = raw_arms.get(arm.arm_id)
        if (
            not isinstance(identity, Mapping)
            or identity.get("release_id") != arm.release_id
            or identity.get("family") != arm.family.value
            or identity.get("profile_kind") != arm.profile_kind.value
            or identity.get("arm_kind") != arm.arm_kind.value
        ):
            raise ArtifactError(f"operator arm identity is mixed or invalid for {arm.arm_id}")
    raw_targets = operator.get("benchmark_targets")
    expected_targets = {
        f"{target.cell_id}:{target.split}": target for target in manifest.benchmarks
    }
    if not isinstance(raw_targets, Mapping) or set(raw_targets) != set(expected_targets):
        raise ArtifactError("operator source targets differ from the manifest")
    for key, target in expected_targets.items():
        identity = raw_targets.get(key)
        if (
            not isinstance(identity, Mapping)
            or identity.get("benchmark_id") != target.cell_id
            or identity.get("split") != target.split
            or identity.get("source_revision") != target.source_revision
        ):
            raise ArtifactError(f"operator source identity is invalid for {key}")
    policy_bindings = operator.get("resolved_policy_bindings")
    if not isinstance(policy_bindings, Mapping) or policy_bindings:
        raise ArtifactError("rank-1 release run unexpectedly contains policy bindings")


def _validate_external_runtime_receipts(
    records: Sequence[Mapping[str, Any]],
    manifest: ExperimentManifest,
) -> Mapping[str, Any]:
    expected_arms = tuple(arm.arm_id for arm in manifest.arms)
    baseline = expected_arms[0]
    treatment_arms = expected_arms[1:]
    targets = {(target.cell_id, target.split): target for target in manifest.benchmarks}
    dependency_by_stratum: dict[tuple[str, str], Mapping[str, str]] = {}
    source_revisions: dict[str, str] = {}
    atomic_pair_count = 0
    checkpoint_count = 0
    for record in records:
        cell = record.get("cell")
        result = record.get("result")
        payload = result.get("payload") if isinstance(result, Mapping) else None
        if not isinstance(cell, Mapping) or not isinstance(payload, Mapping):
            raise ArtifactError("external result lacks a cell payload")
        benchmark = _required_record_text(cell, "benchmark_id", label="experiment cell")
        split = _required_record_text(cell, "split", label="experiment cell")
        unit_id = _required_record_text(cell, "unit_id", label="experiment cell")
        model_slot = _required_record_text(cell, "model_slot", label="experiment cell")
        target = targets.get((benchmark, split))
        if target is None:
            raise ArtifactError("external result references another benchmark source")
        if (
            payload.get("benchmark_id") != benchmark
            or payload.get("split") != split
            or payload.get("unit_id") != unit_id
            or payload.get("model_slot") != model_slot
            or payload.get("seed") != cell.get("seed")
            or payload.get("baseline_arm_id") != baseline
            or tuple(payload.get("planned_arm_ids", ())) != expected_arms
            or payload.get("target_source_revision") != target.source_revision
            or payload.get("atomic_replay_contract") != ATOMIC_EXTERNAL_PAIR_CONTRACT
        ):
            raise ArtifactError("external result identity differs from its frozen cell/source")
        dependencies = _dependency_receipt(payload)
        stratum = (benchmark, model_slot)
        previous = dependency_by_stratum.get(stratum)
        if previous is None:
            dependency_by_stratum[stratum] = dependencies
        elif previous != dependencies:
            raise ArtifactError(
                "runtime dependency identity drifts across groups in one model/source stratum"
            )
        checkpoint = _checkpoint_source_receipt(
            payload,
            source_revision=str(target.source_revision),
        )
        checkpoint_count += 1
        atomic_pair_count += _validate_atomic_pair_receipts(
            payload,
            benchmark=benchmark,
            seed=int(cell["seed"]),
            source_revision=str(target.source_revision),
            baseline_arm_id=baseline,
            treatment_arm_ids=treatment_arms,
            dependencies=dependencies,
            checkpoint=checkpoint,
        )
        source_revisions[benchmark] = str(target.source_revision)
    return {
        "checkpoint_source_receipts": checkpoint_count,
        "atomic_pair_receipts": atomic_pair_count,
        "dependency_receipt_strata": {
            f"{benchmark}::{model}": dict(dependencies)
            for (benchmark, model), dependencies in sorted(dependency_by_stratum.items())
        },
        "source_revisions": dict(sorted(source_revisions.items())),
        "dependency_and_source_receipts_verified": True,
    }


def _dependency_receipt(payload: Mapping[str, Any]) -> Mapping[str, str]:
    raw = payload.get("observed_dependency_hashes")
    if not isinstance(raw, Mapping) or set(raw) != _DEPENDENCY_SLOTS:
        raise ArtifactError("external result lacks the exact shared dependency receipt slots")
    parsed: dict[str, str] = {}
    for slot, digest in raw.items():
        if not isinstance(slot, str) or not _is_sha256(digest):
            raise ArtifactError("external dependency receipts must be SHA-256 mappings")
        parsed[slot] = str(digest)
    not_applicable = payload.get("not_applicable_dependency_slots")
    if (
        not isinstance(not_applicable, Sequence)
        or isinstance(not_applicable, str | bytes)
        or len(not_applicable) != len(set(not_applicable))
        or any(slot not in _DEPENDENCY_SLOTS for slot in not_applicable)
    ):
        raise ArtifactError("not-applicable dependency receipt is malformed")
    expected_not_applicable = tuple(
        sorted(
            slot
            for slot, digest in parsed.items()
            if digest
            == content_hash(
                {
                    "agentinputlab_dependency": "not_applicable",
                    "schema_version": 1,
                    "slot": slot,
                }
            )
        )
    )
    if tuple(not_applicable) != expected_not_applicable:
        raise ArtifactError("not-applicable dependency receipt differs from observed hashes")
    return parsed


def _checkpoint_source_receipt(
    payload: Mapping[str, Any],
    *,
    source_revision: str,
) -> Mapping[str, Any]:
    checkpoint = payload.get("checkpoint_identity")
    required = {
        "identity_sha256",
        "group_sha256",
        "visible_state_sha256",
        "origin_state_sha256",
        "replay_identity_sha256",
        "source_revision",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != required:
        raise ArtifactError("external result lacks a checkpoint source receipt")
    if checkpoint.get("source_revision") != source_revision or any(
        not _is_sha256(checkpoint.get(key)) for key in required if key != "source_revision"
    ):
        raise ArtifactError("checkpoint source receipt is malformed or source-mixed")
    return checkpoint


def _validate_atomic_pair_receipts(
    payload: Mapping[str, Any],
    *,
    benchmark: str,
    seed: int,
    source_revision: str,
    baseline_arm_id: str,
    treatment_arm_ids: Sequence[str],
    dependencies: Mapping[str, str],
    checkpoint: Mapping[str, Any],
) -> int:
    raw_pairs = payload.get("atomic_pairs")
    if (
        not isinstance(raw_pairs, Sequence)
        or isinstance(raw_pairs, str | bytes)
        or len(raw_pairs) != len(treatment_arm_ids)
    ):
        raise ArtifactError("external result lacks the complete atomic replay receipts")
    baseline_consistency = payload.get("baseline_consistency_sha256")
    if not _is_sha256(baseline_consistency):
        raise ArtifactError("external result lacks the NO_OP consistency receipt")
    observed_treatments: set[str] = set()
    checkpoint_ids: set[str] = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, Mapping):
            raise ArtifactError("atomic replay receipt row must be an object")
        treatment_id = raw_pair.get("treatment_arm_id")
        if (
            treatment_id not in treatment_arm_ids
            or treatment_id in observed_treatments
            or not _is_sha256(raw_pair.get("request_sha256"))
            or raw_pair.get("baseline_consistency_sha256") != baseline_consistency
        ):
            raise ArtifactError("atomic replay receipt arm/request identity is invalid")
        observed_treatments.add(str(treatment_id))
        raw_receipt = raw_pair.get("receipt")
        receipt = _parse_atomic_receipt(raw_receipt)
        if (
            receipt.benchmark_id != benchmark
            or receipt.source_revision != source_revision
            or receipt.seed != seed
            or receipt.locked_dependency_hashes != dependencies
            or receipt.origin_state_hash != checkpoint["origin_state_sha256"]
            or receipt.replay_identity != checkpoint["replay_identity_sha256"]
            or raw_pair.get("receipt_sha256") != content_hash(receipt)
        ):
            raise ArtifactError("atomic replay source/dependency receipt is mixed")
        arm_roles = {(arm.role, arm.arm_id) for arm in receipt.arms}
        if arm_roles != {
            ("baseline", baseline_arm_id),
            ("treatment", treatment_id),
        }:
            raise ArtifactError("atomic replay receipt covers another arm pair")
        checkpoint_ids.add(receipt.checkpoint_id)
        resource_hashes = raw_pair.get("resource_receipt_sha256s")
        if (
            not isinstance(resource_hashes, Sequence)
            or isinstance(resource_hashes, str | bytes)
            or any(not _is_sha256(value) for value in resource_hashes)
        ):
            raise ArtifactError("atomic replay resource-receipt hashes are invalid")
    if observed_treatments != set(treatment_arm_ids) or len(checkpoint_ids) != 1:
        raise ArtifactError("atomic replay receipts do not form one complete checkpoint block")
    return len(raw_pairs)


def _parse_atomic_receipt(value: Any) -> AtomicReplayReceipt:
    if not isinstance(value, Mapping):
        raise ArtifactError("atomic replay receipt is missing")
    raw_arms = value.get("arms")
    if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, str | bytes):
        raise ArtifactError("atomic replay receipt lacks arm receipts")
    try:
        arms = tuple(
            AtomicReplayArmReceipt(
                role=raw["role"],
                arm_id=raw["arm_id"],
                origin_state_hash=raw["origin_state_hash"],
                replay_identity=raw["replay_identity"],
                seed=raw["seed"],
                locked_dependency_hashes=raw["locked_dependency_hashes"],
                runtime_instance_id=raw["runtime_instance_id"],
                completed=raw["completed"],
            )
            for raw in raw_arms
            if isinstance(raw, Mapping)
        )
        if len(arms) != len(raw_arms):
            raise ArtifactError("atomic replay arm receipt is malformed")
        receipt = AtomicReplayReceipt(
            benchmark_id=value["benchmark_id"],
            checkpoint_id=value["checkpoint_id"],
            source_revision=value["source_revision"],
            bridge_id=value["bridge_id"],
            bridge_revision=value["bridge_revision"],
            scorer_id=value["scorer_id"],
            scorer_revision=value["scorer_revision"],
            origin_state_hash=value["origin_state_hash"],
            replay_identity=value["replay_identity"],
            seed=value["seed"],
            locked_dependency_hashes=value["locked_dependency_hashes"],
            arms=arms,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ArtifactError("atomic replay receipt is malformed") from exc
    if canonical_json(value) != canonical_json(to_jsonable(receipt)):
        raise ArtifactError("atomic replay receipt schema contains missing or extra fields")
    return receipt


def _collapsed_family_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: ExperimentManifest,
    baseline_arm_id: str,
    arms_by_family: Mapping[SemanticFamily, str],
    metric: str,
    algorithm_failure_value: float | None,
) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        cell = record["cell"]
        values = _arm_metrics(
            record,
            metric=metric,
            algorithm_failure_value=algorithm_failure_value,
        )
        if values is None:  # pragma: no cover - verified all-succeeded invariant
            raise ArtifactError("transfer record is not a completed cell")
        benchmark = _required_record_text(cell, "benchmark_id", label="experiment cell")
        unit_id = _required_record_text(cell, "unit_id", label="experiment cell")
        grouped[(benchmark, unit_id)].append(
            {
                "cluster_id": _cluster_id(cell),
                "model_slot": _required_record_text(
                    cell,
                    "model_slot",
                    label="experiment cell",
                ),
                "baseline": values[baseline_arm_id],
                **{family.value: values[arm_id] for family, arm_id in arms_by_family.items()},
            }
        )
    collapsed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        model_slots = tuple(sorted(str(row["model_slot"]) for row in rows))
        if len(model_slots) != len(set(model_slots)):
            raise ArtifactError("family analysis duplicates a model replicate within a group")
        collapsed[key] = {
            "underlying_group": f"{key[0]}::{key[1]}",
            "cluster_ids": tuple(str(row["cluster_id"]) for row in rows),
            "model_slots": model_slots,
            "baseline": fmean(float(row["baseline"]) for row in rows),
            **{
                family.value: fmean(float(row[family.value]) for row in rows)
                for family in _CORE_FAMILIES
            },
        }
    return collapsed


def _family_summaries(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    arms_by_family: Mapping[SemanticFamily, str],
    manifest: ExperimentManifest,
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    release_by_arm = {arm.arm_id: arm.release_id for arm in manifest.arms}
    summaries: dict[str, Any] = {}
    for family in _CORE_FAMILIES:
        arm_id = arms_by_family[family]
        baseline = tuple(float(row["baseline"]) for row in rows.values())
        treatment = tuple(float(row[family.value]) for row in rows.values())
        paired = summarize_pairs(baseline, treatment)
        effects = tuple(after - before for before, after in zip(baseline, treatment, strict=True))
        summaries[family.value] = {
            "arm_id": arm_id,
            "release_id": release_by_arm[arm_id],
            "cluster_count": len(effects),
            "no_op_mean": fmean(baseline),
            "family_mean": fmean(treatment),
            "mean_effect_vs_no_op": paired.mean_effect,
            "cluster_bootstrap_95ci": list(
                _cluster_bootstrap_interval(
                    effects,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            ),
            "win_rate_vs_no_op": paired.win_rate,
            "tie_rate_vs_no_op": paired.tie_rate,
            "harm_rate_vs_no_op": paired.harm_rate,
        }
    return summaries


def _require_same_representatives(
    transfer: _VerifiedTransferRun,
    primary: _VerifiedTransferRun,
) -> None:
    if transfer.representative_hash != primary.representative_hash:
        raise ArtifactError("transfer and primary runs bind different representative manifests")
    transfer_file = transfer.binding_receipt["representative_binding"].get(
        "representative_file_sha256"
    )
    primary_file = primary.binding_receipt["representative_binding"].get(
        "representative_file_sha256"
    )
    if transfer_file != primary_file:
        raise ArtifactError("transfer and primary runs bind different representative files")
    transfer_releases = {arm.family: arm.release_id for arm in transfer.manifest.arms[1:]}
    primary_releases = {arm.family: arm.release_id for arm in primary.manifest.arms[1:]}
    if transfer_releases != primary_releases:
        raise ArtifactError("transfer and primary runs use different rank-1 releases")


def _transfer_primary_boundary(
    transfer_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    primary_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    transfer: _VerifiedTransferRun,
    primary: _VerifiedTransferRun,
    bootstrap_samples: int,
    rng: random.Random,
) -> Mapping[str, Any]:
    transfer_keys = set(transfer_rows)
    if (
        len(transfer_keys) != 200
        or any(key[0] != _BENCHMARK_ID for key in transfer_keys)
        or not transfer_keys.issubset(primary_rows)
    ):
        raise ArtifactError(
            "heldout-model units are not an exact 200-group subset of primary "
            "STATE-Bench common support"
        )
    dependency_boundary = _validate_dependency_interaction_boundary(
        transfer.runtime_receipts,
        primary.runtime_receipts,
    )
    summaries: dict[str, Any] = {}
    for family in _CORE_FAMILIES:
        transfer_effects: list[float] = []
        primary_effects: list[float] = []
        interaction: list[float] = []
        for key in sorted(transfer_keys):
            transfer_row = transfer_rows[key]
            primary_row = primary_rows[key]
            transfer_effect = float(transfer_row[family.value]) - float(transfer_row["baseline"])
            primary_effect = float(primary_row[family.value]) - float(primary_row["baseline"])
            transfer_effects.append(transfer_effect)
            primary_effects.append(primary_effect)
            interaction.append(transfer_effect - primary_effect)
        summaries[family.value] = {
            "matched_group_count": len(interaction),
            "heldout_model_effect_vs_no_op": fmean(transfer_effects),
            "primary_anchor_mean_effect_vs_no_op": fmean(primary_effects),
            "transfer_minus_primary_effect": fmean(interaction),
            "matched_cluster_bootstrap_95ci": list(
                _cluster_bootstrap_interval(
                    interaction,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            ),
        }
    return {
        "comparison_kind": "matched_transfer_minus_primary_interaction_boundary",
        "primary_study_id": primary.manifest.study_id,
        "primary_manifest_hash": primary.manifest.manifest_hash,
        "matched_underlying_groups": 200,
        "primary_effect_replicate_handling": (
            "equal-weight mean over anchor_a and anchor_b within each underlying group"
        ),
        "dependency_boundary": dependency_boundary,
        "families": summaries,
        "scope_note": (
            "Difference-in-differences on the same 200 STATE-Bench unit IDs. "
            "This boundary is not a pooled population effect and does not assert "
            "exchangeability with the remaining primary common-support groups."
        ),
        "paper_claim_authorized": False,
    }


def _validate_dependency_interaction_boundary(
    transfer_receipts: Mapping[str, Any],
    primary_receipts: Mapping[str, Any],
) -> Mapping[str, Any]:
    transfer_strata = transfer_receipts.get("dependency_receipt_strata")
    primary_strata = primary_receipts.get("dependency_receipt_strata")
    transfer_keys = tuple(f"{_BENCHMARK_ID}::{model_slot}" for model_slot in _TRANSFER_MODEL_SLOTS)
    primary_keys = tuple(f"{_BENCHMARK_ID}::{model_slot}" for model_slot in _PRIMARY_MODEL_SLOTS)
    if (
        not isinstance(transfer_strata, Mapping)
        or not isinstance(primary_strata, Mapping)
        or any(key not in transfer_strata for key in transfer_keys)
        or any(key not in primary_strata for key in primary_keys)
    ):
        raise ArtifactError("model-transfer dependency strata are incomplete")
    by_model = {
        **{
            model_slot: transfer_strata[f"{_BENCHMARK_ID}::{model_slot}"]
            for model_slot in _TRANSFER_MODEL_SLOTS
        },
        **{
            model_slot: primary_strata[f"{_BENCHMARK_ID}::{model_slot}"]
            for model_slot in _PRIMARY_MODEL_SLOTS
        },
    }
    if any(
        not isinstance(dependencies, Mapping) or set(dependencies) != _DEPENDENCY_SLOTS
        for dependencies in by_model.values()
    ):
        raise ArtifactError("model-transfer dependency boundary has invalid slots")
    nuisance_slots = _DEPENDENCY_SLOTS - {"agent_model"}
    for slot in nuisance_slots:
        if len({dependencies[slot] for dependencies in by_model.values()}) != 1:
            raise ArtifactError(f"model-transfer nuisance dependency {slot!r} differs across runs")
    model_hashes = {
        model_slot: str(dependencies["agent_model"])
        for model_slot, dependencies in by_model.items()
    }
    if len(set(model_hashes.values())) != len(_TRANSFER_MODEL_SLOTS) + len(_PRIMARY_MODEL_SLOTS):
        raise ArtifactError(
            "transfer and primary model slots do not resolve to six distinct identities"
        )
    return {
        "varied_dependency_slot": "agent_model",
        "model_identity_hashes": model_hashes,
        "shared_nuisance_dependency_slots": sorted(nuisance_slots),
        "source_revision": _STATEBENCH_REVISION,
        "boundary_verified": True,
    }


def _validate_analysis_arguments(
    *,
    metric: str,
    algorithm_failure_value: float | None,
    bootstrap_samples: int,
    random_seed: int,
) -> None:
    if not isinstance(metric, str) or not metric.strip():
        raise ValidationError("analysis metric must be non-empty")
    if algorithm_failure_value is not None and (
        isinstance(algorithm_failure_value, bool)
        or not isinstance(algorithm_failure_value, (int, float))
        or not math.isfinite(float(algorithm_failure_value))
    ):
        raise ValidationError("algorithm_failure_value must be finite when provided")
    _validate_resample_arguments(
        bootstrap_samples=bootstrap_samples,
        permutation_samples=1,
        random_seed=random_seed,
    )


def _load_run_identity(root: Path) -> Mapping[str, Any]:
    path = root / "run_identity.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("run identity is missing or invalid") from exc
    if not isinstance(value, Mapping):
        raise ArtifactError("run identity must be a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "HELDOUT_MODEL_TRANSFER_PROTOCOL",
    "analyze_heldout_model_transfer",
    "validate_heldout_model_transfer_run",
]
