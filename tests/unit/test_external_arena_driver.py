from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.benchmarks import (
    ATOMIC_EXTERNAL_PAIR_CONTRACT,
    AtomicReplayArmReceipt,
    AtomicReplayArmResult,
    AtomicReplayPairResult,
    AtomicReplayReceipt,
    AtomicReplayRequest,
)
from agentinputlab.contracts import (
    FrozenAgentState,
    Message,
    ProfileKind,
    SemanticFamily,
)
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import (
    canonical_json,
    content_hash,
    to_jsonable,
)
from agentinputlab.experiments import (
    ExperimentArmKind,
    ExperimentArmSpec,
    ExperimentArtifactStore,
    ExperimentBenchmarkTarget,
    ExperimentCell,
    ExperimentManifest,
    ExperimentPrerequisite,
    ExperimentStage,
    PrerequisiteKind,
    build_experiment_plan,
)
from agentinputlab.experiments.drivers.external_arena import (
    ExternalArenaCheckpoint,
    ExternalArenaExperimentDriver,
    ExternalArmPayload,
    ExternalArmPayloadBuildRequest,
    ExternalBridgeBuildRequest,
    ExternalCheckpointBuildRequest,
    ExternalUnitPreflightRequest,
    external_dependency_not_applicable_hash,
)
from agentinputlab.observability import (
    ResourceLimits,
    ResourceReceipt,
    ResourceUsage,
)
from agentinputlab.selectors import (
    RuleSelector,
    SelectorArtifactBundle,
    SelectorArtifactStage,
    UtilityDefinition,
    VisibleStateFeatureExtractor,
    build_selector_discovery_freeze,
)

_SOURCE_REVISION = "cf71a8070269883e38a365ffa85f78f46844c1f4"
_BRIDGE_REVISION = "8" * 64
_SCORER_REVISION = "9" * 64
_DEPENDENCIES = {
    "agent_model": "1" * 64,
    "official_runtime": "2" * 64,
    "corpus": "3" * 64,
    "preprocessing": "4" * 64,
    "index": "5" * 64,
    "embedder": "6" * 64,
    "retriever": "7" * 64,
    "reranker": "8" * 64,
    "reader": "9" * 64,
    "scorer": "a" * 64,
}
PAIR_REQUESTS: list[AtomicReplayRequest] = []
DRIFT_BASELINE = False
BRIDGE_ID_OVERRIDE: str | None = None
BUILDER_BLOCKER: str | None = None
SELECTOR_INPUT_DRIFT = False
INVALID_OUTCOME_MODE: str | None = None


def test_external_checkpoint_accepts_composite_release_family():
    checkpoint = ExternalArenaCheckpoint(
        benchmark_id="tau3.banking_knowledge",
        checkpoint_id="composite-checkpoint",
        group_id="composite-group",
        source_revision=_SOURCE_REVISION,
        origin_state_hash="4" * 64,
        replay_identity="5" * 64,
        locked_dependency_hashes=_DEPENDENCIES,
        visible_state=FrozenAgentState(
            task_id="opaque-composite-task",
            query="Resolve the request with the visible resources.",
            instruction="Use only visible state.",
        ),
        eligible_families=(SemanticFamily.COMPOSITE,),
    )
    assert checkpoint.eligible_families == (SemanticFamily.COMPOSITE,)


def _outcome_metrics(official_reward: float, *, cost: float) -> dict[str, float]:
    metrics = {
        "official_reward": official_reward,
        "cost": cost,
        "contract_preserved": 1.0,
        "privileged_leakage": 0.0,
    }
    if INVALID_OUTCOME_MODE == "missing_component":
        del metrics["contract_preserved"]
    elif INVALID_OUTCOME_MODE == "precomputed":
        metrics["valid_success"] = official_reward
    elif INVALID_OUTCOME_MODE == "out_of_range":
        metrics["privileged_leakage"] = 1.25
    return metrics


def _resource_receipt(call_index: int) -> ResourceReceipt:
    limits = ResourceLimits(wall_timeout_seconds=10)
    return ResourceReceipt(
        limit_hash=limits.limit_hash,
        wall_seconds=0.1 * call_index,
        cpu_seconds=0.05 * call_index,
        peak_rss_bytes=4096 * call_index,
        supervisor_identity={"kind": "fixture", "call_index": call_index},
        image_identity={"kind": "fixture", "digest": "sha256:" + "d" * 64},
    )


class CheckpointBuilder:
    builder_kind = "arena_checkpoint"

    def fingerprint(self) -> dict[str, str]:
        return {"builder": "fixture-checkpoint", "revision": "v1"}

    def preflight(self, parameters):
        assert parameters == {"capture_contract": "fixture-v1"}
        return () if BUILDER_BLOCKER is None else (BUILDER_BLOCKER,)

    def build(self, request: ExternalCheckpointBuildRequest) -> ExternalArenaCheckpoint:
        return ExternalArenaCheckpoint(
            benchmark_id=request.benchmark_id,
            checkpoint_id=request.checkpoint_id,
            group_id=f"fixture-group-{request.checkpoint_id}",
            source_revision=request.source_revision,
            origin_state_hash="4" * 64,
            replay_identity="5" * 64,
            locked_dependency_hashes=_DEPENDENCIES,
            visible_state=FrozenAgentState(
                task_id=f"opaque-task-{request.checkpoint_id}",
                query=(
                    "Please compare this and then verify the latest evidence for "
                    f"{request.checkpoint_id}."
                ),
                instruction="Use the visible tools.",
                messages=(
                    Message(
                        role="user",
                        content=(
                            "Please compare this and then verify the latest evidence for "
                            f"{request.checkpoint_id}."
                        ),
                    ),
                ),
                tool_names=("search",),
            ),
            eligible_families=(
                SemanticFamily.REFORMULATE,
                SemanticFamily.DECOMPOSE,
                SemanticFamily.ACQUIRE,
                SemanticFamily.CLARIFY,
                SemanticFamily.CURATE,
            ),
        )

    def preflight_units(
        self,
        request: ExternalUnitPreflightRequest,
    ) -> tuple[str, ...]:
        known = {
            "checkpoint-1",
            "train-checkpoint",
            "calibration-checkpoint",
        }
        unknown = sorted(set(request.unit_ids) - known)
        return tuple(f"unknown checkpoint ID {item}" for item in unknown)

    def preflight_group_ids(
        self,
        request: ExternalUnitPreflightRequest,
    ) -> dict[str, str]:
        return {unit_id: f"fixture-group-{unit_id}" for unit_id in request.unit_ids}


class FixtureAtomicBridge:
    atomic_replay_contract = ATOMIC_EXTERNAL_PAIR_CONTRACT
    benchmark_id = "tau3.banking_knowledge"
    source_revision = _SOURCE_REVISION
    bridge_id = "fixture.atomic-bridge"
    bridge_revision = _BRIDGE_REVISION
    scorer_id = "fixture.official-scorer"
    scorer_revision = _SCORER_REVISION
    locked_dependency_hashes: ClassVar[dict[str, str]] = _DEPENDENCIES

    def __init__(self) -> None:
        if BRIDGE_ID_OVERRIDE is not None:
            self.bridge_id = BRIDGE_ID_OVERRIDE

    def evaluate_pair(self, request: AtomicReplayRequest) -> AtomicReplayPairResult:
        PAIR_REQUESTS.append(request)
        call_index = len(PAIR_REQUESTS)
        baseline_value = 0.25
        if DRIFT_BASELINE and call_index > 1:
            baseline_value = 0.5
        treatment_id = next(arm.arm_id for arm in request.arms if arm.role == "treatment")
        receipt = AtomicReplayReceipt(
            benchmark_id=request.benchmark_id,
            checkpoint_id=request.checkpoint_id,
            source_revision=request.source_revision,
            bridge_id=self.bridge_id,
            bridge_revision=self.bridge_revision,
            scorer_id=self.scorer_id,
            scorer_revision=self.scorer_revision,
            origin_state_hash=request.origin_state_hash,
            replay_identity=request.replay_identity,
            seed=request.seed,
            locked_dependency_hashes=request.locked_dependency_hashes,
            arms=(
                AtomicReplayArmReceipt(
                    role="baseline",
                    arm_id="no_op",
                    origin_state_hash=request.origin_state_hash,
                    replay_identity=request.replay_identity,
                    seed=request.seed,
                    locked_dependency_hashes=request.locked_dependency_hashes,
                    runtime_instance_id=hashlib.sha256(
                        f"baseline-{call_index}".encode()
                    ).hexdigest(),
                    completed=True,
                ),
                AtomicReplayArmReceipt(
                    role="treatment",
                    arm_id=treatment_id,
                    origin_state_hash=request.origin_state_hash,
                    replay_identity=request.replay_identity,
                    seed=request.seed,
                    locked_dependency_hashes=request.locked_dependency_hashes,
                    runtime_instance_id=hashlib.sha256(
                        f"treatment-{call_index}".encode()
                    ).hexdigest(),
                    completed=True,
                ),
            ),
        )
        return AtomicReplayPairResult(
            receipt=receipt,
            arms=(
                AtomicReplayArmResult(
                    role="baseline",
                    arm_id="no_op",
                    metrics=_outcome_metrics(baseline_value, cost=1.0),
                    trace_hash="a" * 64,
                    usage=ResourceUsage(model_calls=1, input_tokens=3, output_tokens=1),
                ),
                AtomicReplayArmResult(
                    role="treatment",
                    arm_id=treatment_id,
                    metrics=_outcome_metrics(
                        0.75 if treatment_id == "rewrite" else 0.5,
                        cost=1.5,
                    ),
                    trace_hash=hashlib.sha256(treatment_id.encode()).hexdigest(),
                    usage=ResourceUsage(model_calls=1, input_tokens=4, output_tokens=2),
                ),
            ),
            resource_receipts=(_resource_receipt(call_index),),
        )


class BridgeBuilder:
    builder_kind = "atomic_bridge"

    def fingerprint(self) -> dict[str, str]:
        return {"builder": "fixture-bridge", "revision": "v1"}

    def preflight(self, parameters):
        assert parameters == {"restore_contract": "atomic-pair-v1"}
        return ()

    def build(self, request: ExternalBridgeBuildRequest) -> FixtureAtomicBridge:
        assert request.checkpoint.locked_dependency_hashes == _DEPENDENCIES
        return FixtureAtomicBridge()


class PayloadBuilder:
    builder_kind = "arm_payload"

    def fingerprint(self) -> dict[str, str]:
        return {"builder": "fixture-payload", "revision": "v1"}

    def preflight(self, parameters):
        assert parameters == {"wire_schema": "fixture-v1"}
        return ()

    def build(self, request: ExternalArmPayloadBuildRequest) -> ExternalArmPayload:
        decision = None
        if request.arm_kind is ExperimentArmKind.POLICY:
            identity_field: dict[str, str]
            if request.release_id == "policy.locked_best_eligible_global":
                identity_field = {
                    "policy_manifest_sha256": os.environ["AILAB_SELECTOR_POLICY_MANIFEST_SHA256"]
                }
            else:
                identity_field = {
                    "selector_bundle_hash": json.loads(os.environ["AILAB_SELECTOR_IDENTITY"])[
                        "sha256"
                    ]
                }
            decision = {
                "selected_family": SemanticFamily.REFORMULATE.value,
                "selected_release_id": "fixture.rewrite.v1",
                "abstained": False,
                "predicted_gain": 0.2,
                "confidence": 0.8,
                **identity_field,
            }
        return ExternalArmPayload(
            arm_id=request.arm_id,
            release_id=request.release_id,
            family=request.family,
            profile_kind=request.profile_kind,
            payload={
                "wire_schema": "fixture-v1",
                "release_id": request.release_id,
                "query": request.checkpoint.visible_state.query,
            },
            selector_decision=decision,
            selector_input_state_hash=(
                request.pre_action["state_hash"] if decision is not None else None
            ),
            selector_feature_hash=(
                ("f" * 64 if SELECTOR_INPUT_DRIFT else request.pre_action["feature_hash"])
                if decision is not None
                else None
            ),
        )


def checkpoint_builder_factory() -> CheckpointBuilder:
    return CheckpointBuilder()


def bridge_builder_factory() -> BridgeBuilder:
    return BridgeBuilder()


def payload_builder_factory() -> PayloadBuilder:
    return PayloadBuilder()


def _builder_spec(factory_name: str, fingerprint: dict[str, str], parameters: dict) -> dict:
    return {
        "entrypoint": f"{__name__}:{factory_name}",
        "fingerprint_sha256": content_hash(fingerprint),
        "parameters": parameters,
    }


def _arm_binding(
    *,
    release_id: str,
    family: SemanticFamily,
    arm_kind: ExperimentArmKind = ExperimentArmKind.RELEASE,
) -> dict[str, Any]:
    payload = {
        "release_id": release_id,
        "family": family.value,
        "profile_kind": ProfileKind.CONTROLLED_ADAPTER.value,
        "arm_kind": arm_kind.value,
        "payload_builder": _builder_spec(
            "payload_builder_factory",
            PayloadBuilder().fingerprint(),
            {"wire_schema": "fixture-v1"},
        ),
    }
    if arm_kind is ExperimentArmKind.POLICY:
        payload["policy_kind"] = {
            "policy.locked_best_eligible_global": "locked_best_eligible_global",
            "policy.eligibility_aware_selector": "frozen_selector",
        }[release_id]
    return payload


def _target_binding(
    benchmark_id: str,
    split: str,
    source_revision: str,
) -> dict[str, Any]:
    success_metric = {
        "tau3.banking_knowledge": "official_reward",
        "tau3.retail": "terminal_success",
        "state_bench.common_support": "task_completion_pass",
        "toolsandbox.common_support": "official_similarity",
        "appworld_ul.common_support": "task_success",
        "webarena.external": "task_success",
    }[benchmark_id]
    return {
        "benchmark_id": benchmark_id,
        "split": split,
        "source_revision": source_revision,
        "outcome_contract": {
            "formula_id": "agentinputlab.valid_success.v1",
            "official_success_metric": success_metric,
        },
        "checkpoint_builder": _builder_spec(
            "checkpoint_builder_factory",
            CheckpointBuilder().fingerprint(),
            {"capture_contract": "fixture-v1"},
        ),
        "bridge_builder": _builder_spec(
            "bridge_builder_factory",
            BridgeBuilder().fingerprint(),
            {"restore_contract": "atomic-pair-v1"},
        ),
        "bridge_lock": {
            "bridge_id": "fixture.atomic-bridge",
            "bridge_revision": _BRIDGE_REVISION,
            "scorer_id": "fixture.official-scorer",
            "scorer_revision": _SCORER_REVISION,
            "dependency_hashes_by_model_slot": {
                "anchor_a": {
                    **_DEPENDENCIES,
                    **({"user_simulator": "d" * 64, "nl_assertion_judge": "e" * 64}
                       if benchmark_id == "tau3.retail" else {}),
                },
            },
        },
    }


def _config() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "driver_id": "fixture.external-arena",
        "driver_revision_sha256": "d" * 64,
        "baseline_arm_id": "no_op",
        "benchmark_targets": {
            "tau3.banking_knowledge:base": _target_binding(
                "tau3.banking_knowledge",
                "base",
                _SOURCE_REVISION,
            ),
            "tau3.banking_knowledge:discovery_train": _target_binding(
                "tau3.banking_knowledge",
                "discovery_train",
                _SOURCE_REVISION,
            ),
            "tau3.banking_knowledge:discovery_calibration": _target_binding(
                "tau3.banking_knowledge",
                "discovery_calibration",
                _SOURCE_REVISION,
            ),
        },
        "arms": {
            "no_op": _arm_binding(
                release_id="baseline.no_op",
                family=SemanticFamily.NO_OP,
            ),
            "rewrite": _arm_binding(
                release_id="fixture.rewrite.v1",
                family=SemanticFamily.REFORMULATE,
            ),
            "acquire": _arm_binding(
                release_id="fixture.acquire.v1",
                family=SemanticFamily.ACQUIRE,
            ),
            "selector": _arm_binding(
                release_id="policy.eligibility_aware_selector",
                family=SemanticFamily.COMPOSITE,
                arm_kind=ExperimentArmKind.POLICY,
            ),
            "global": _arm_binding(
                release_id="policy.locked_best_eligible_global",
                family=SemanticFamily.COMPOSITE,
                arm_kind=ExperimentArmKind.POLICY,
            ),
        },
    }


def _write_config(tmp_path: Path, config: dict[str, Any]) -> tuple[Path, str]:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "external-driver.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _install_policy_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_checkpoint: str = "train-checkpoint",
    calibration_checkpoint: str = "calibration-checkpoint",
) -> dict[str, str]:
    discovery_revision = "fixture-discovery-v1"
    discovery_sha256 = "6" * 64
    training_table_sha256 = "7" * 64
    group_members = {
        split: (
            "group_"
            + content_hash(
                {
                    "benchmark_id": "tau3.banking_knowledge",
                    "group_id": f"fixture-group-{checkpoint}",
                }
            ),
        )
        for split, checkpoint in (
            ("train", train_checkpoint),
            ("calibration", calibration_checkpoint),
        )
    }
    group_commitments = {
        split: content_hash({"split": split, "group_ids": list(members)})
        for split, members in group_members.items()
    }
    split_provenance_hash = content_hash({"group_split_digests": group_commitments})
    comparison_profile_hash = "a" * 64
    calibration_split_digest = group_commitments["calibration"]
    representatives = {
        SemanticFamily.REFORMULATE: "fixture.rewrite.v1",
    }
    policy_manifest = {
        "schema_version": 1,
        "artifact": "agentinputlab.selector_policy_manifest",
        "stage": "frozen",
        "discovery": {
            "revision": discovery_revision,
            "sha256": discovery_sha256,
        },
        "training_table_sha256": training_table_sha256,
        "split_provenance_hash": split_provenance_hash,
        "comparison_profile_hash": comparison_profile_hash,
        "representative_by_family": {
            "reformulate": "fixture.rewrite.v1",
        },
        "group_partition": {
            "schema_version": 1,
            "hash_scheme": "benchmark_group.content_hash.v1",
            "splits": {
                split: {
                    "member_hashes": members,
                    "commitment_sha256": group_commitments[split],
                }
                for split, members in group_members.items()
            },
        },
        "locked_best_eligible_global": {
            "selected_family": "reformulate",
            "selected_release_id": "fixture.rewrite.v1",
            "selection_receipt": {
                "schema_version": 1,
                "rule_id": "best_eligible_global.v1",
                "selection_split": "calibration",
                "training_table_sha256": training_table_sha256,
                "calibration_split_digest": calibration_split_digest,
            },
        },
    }
    policy_bytes = (canonical_json(policy_manifest) + "\n").encode()
    policy_path = tmp_path / "selector-policy-manifest.json"
    policy_path.write_bytes(policy_bytes)
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()

    bundle = SelectorArtifactBundle.create(
        RuleSelector(),
        stage=SelectorArtifactStage.FROZEN,
        revision="fixture-selector-frozen-v1",
        extractor_id="visible_state_heuristic",
        extractor_revision="v1",
        extractor_hash="c" * 64,
        feature_schema_id="family_defect_features.v1",
        feature_schema_hash="d" * 64,
        representative_by_family=representatives,
        training_table_sha256=training_table_sha256,
        split_provenance_hash=split_provenance_hash,
        comparison_profile_hash=comparison_profile_hash,
        calibration_receipt={"discovery_hash": discovery_sha256},
        fit_lineage={
            "schema_version": 1,
            "kind": "selector_fit",
            "discovery": {
                "revision": discovery_revision,
                "sha256": discovery_sha256,
            },
            "fit_config_sha256": "e" * 64,
        },
    )
    bundle_bytes = (canonical_json(bundle.to_artifact()) + "\n").encode()
    bundle_path = tmp_path / "selector-bundle.json"
    bundle_path.write_bytes(bundle_bytes)
    bundle_file_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    monkeypatch.setenv("AILAB_SELECTOR_POLICY_MANIFEST", str(policy_path))
    monkeypatch.setenv(
        "AILAB_SELECTOR_POLICY_MANIFEST_SHA256",
        policy_sha256,
    )
    monkeypatch.setenv("AILAB_SELECTOR_BUNDLE", str(bundle_path))
    monkeypatch.setenv("AILAB_SELECTOR_BUNDLE_SHA256", bundle_file_sha256)
    monkeypatch.setenv(
        "AILAB_SELECTOR_IDENTITY",
        canonical_json(
            {
                "selector_id": bundle.selector_card_id,
                "revision": bundle.revision,
                "sha256": bundle.bundle_hash,
            }
        ),
    )
    return {
        "policy_file_sha256": policy_sha256,
        "selector_bundle_hash": bundle.bundle_hash,
        "calibration_split_digest": calibration_split_digest,
    }


def _manifest(*, policy: str | None = None) -> ExperimentManifest:
    arms = [
        ExperimentArmSpec(
            arm_id="no_op",
            release_id="baseline.no_op",
            family=SemanticFamily.NO_OP,
            profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        ),
        ExperimentArmSpec(
            arm_id="rewrite",
            release_id="fixture.rewrite.v1",
            family=SemanticFamily.REFORMULATE,
            profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        ),
    ]
    if policy is not None:
        if policy not in {"selector", "global"}:
            raise ValueError("unknown fixture policy")
        selector_parameters: dict[str, str] = {
            "policy_kind": (
                "frozen_selector" if policy == "selector" else "locked_best_eligible_global"
            ),
            "policy_manifest_env": "AILAB_SELECTOR_POLICY_MANIFEST",
            "policy_manifest_sha256_env": ("AILAB_SELECTOR_POLICY_MANIFEST_SHA256"),
        }
        if policy == "selector":
            selector_parameters.update(
                {
                    "selector_identity_env": "AILAB_SELECTOR_IDENTITY",
                    "selector_bundle_env": "AILAB_SELECTOR_BUNDLE",
                    "selector_bundle_sha256_env": ("AILAB_SELECTOR_BUNDLE_SHA256"),
                }
            )
        arms.append(
            ExperimentArmSpec(
                arm_id=policy,
                release_id=(
                    "policy.eligibility_aware_selector"
                    if policy == "selector"
                    else "policy.locked_best_eligible_global"
                ),
                family=SemanticFamily.COMPOSITE,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
                arm_kind=ExperimentArmKind.POLICY,
                parameters=selector_parameters,
            )
        )
    else:
        arms.append(
            ExperimentArmSpec(
                arm_id="acquire",
                release_id="fixture.acquire.v1",
                family=SemanticFamily.ACQUIRE,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            )
        )
    return ExperimentManifest(
        schema_version=1,
        study_id="fixture.external-arena",
        stage=(ExperimentStage.SELECTOR_CONFIRMATION if policy else ExperimentStage.CONFIRMATORY),
        estimand_profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id="tau3.banking_knowledge",
                split="base",
                unit_ids=("checkpoint-1",),
            ),
        ),
        arms=tuple(arms),
        seeds=(17,),
        model_slots=("anchor_a",),
        prerequisites=(
            (
                ExperimentPrerequisite(
                    kind=PrerequisiteKind.FILE_ENV,
                    name="AILAB_SELECTOR_POLICY_MANIFEST",
                    sha256_env="AILAB_SELECTOR_POLICY_MANIFEST_SHA256",
                    description="Fixture selector policy manifest",
                ),
            )
            + (
                (
                    ExperimentPrerequisite(
                        kind=PrerequisiteKind.ENV,
                        name="AILAB_SELECTOR_IDENTITY",
                        description="Fixture selector identity",
                    ),
                    ExperimentPrerequisite(
                        kind=PrerequisiteKind.FILE_ENV,
                        name="AILAB_SELECTOR_BUNDLE",
                        sha256_env="AILAB_SELECTOR_BUNDLE_SHA256",
                        description="Fixture selector bundle",
                    ),
                )
                if policy == "selector"
                else ()
            )
            if policy is not None
            else ()
        ),
        metadata={
            "requires_atomic_external_replay": True,
            "require_driver_unit_preflight": True,
        },
    )


def _cell(manifest: ExperimentManifest) -> ExperimentCell:
    return ExperimentCell(
        study_id=manifest.study_id,
        benchmark_id="tau3.banking_knowledge",
        split="base",
        unit_id="checkpoint-1",
        model_slot="anchor_a",
        seed=17,
        arm_ids=tuple(arm.arm_id for arm in manifest.arms),
        manifest_hash=manifest.manifest_hash,
    )


_GATED_TARGETS = {
    "toolsandbox.common_support": {
        "split": "external_checkpoints",
        "source_revision": "165848b9a78cead7ca7fe7c89c688b58e6501219",
        "gate_env": "AILAB_TOOLSANDBOX_GATE_RECEIPT",
        "gate_sha_env": "AILAB_TOOLSANDBOX_GATE_RECEIPT_SHA256",
        "inventory_env": "AILAB_TOOLSANDBOX_REPLICATION_CHECKPOINTS",
        "inventory_sha_env": "AILAB_TOOLSANDBOX_REPLICATION_CHECKPOINTS_SHA256",
        "reset_mode": "atomic_restore",
    },
    "appworld_ul.common_support": {
        "split": "external_checkpoints",
        "source_revision": "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
        "gate_env": "AILAB_APPWORLD_UL_GATE_RECEIPT",
        "gate_sha_env": "AILAB_APPWORLD_UL_GATE_RECEIPT_SHA256",
        "inventory_env": "AILAB_APPWORLD_UL_REPLICATION_CHECKPOINTS",
        "inventory_sha_env": "AILAB_APPWORLD_UL_REPLICATION_CHECKPOINTS_SHA256",
        "reset_mode": "atomic_restore",
    },
    "webarena.external": {
        "split": "locked_transfer",
        "source_revision": "fixture-webarena-revision",
        "gate_env": "AILAB_WEBARENA_STACK_RECEIPT",
        "gate_sha_env": "AILAB_WEBARENA_STACK_RECEIPT_SHA256",
        "inventory_env": "AILAB_WEBARENA_TRANSFER_UNITS",
        "inventory_sha_env": "AILAB_WEBARENA_TRANSFER_UNITS_SHA256",
        "reset_mode": "independent_stack_or_verified_reset",
    },
}


def _gated_manifest(benchmark_id: str) -> ExperimentManifest:
    binding = _GATED_TARGETS[benchmark_id]
    base = _manifest()
    return replace(
        base,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id=benchmark_id,
                split=binding["split"],
                unit_inventory_env=binding["inventory_env"],
                unit_inventory_sha256_env=binding["inventory_sha_env"],
                external=True,
                source_revision=binding["source_revision"],
            ),
        ),
        prerequisites=(
            ExperimentPrerequisite(
                kind=PrerequisiteKind.FILE_ENV,
                name=binding["gate_env"],
                sha256_env=binding["gate_sha_env"],
                description="Fixture external arena gate receipt",
            ),
        ),
    )


def _gated_config(benchmark_id: str) -> dict[str, Any]:
    binding = _GATED_TARGETS[benchmark_id]
    config = _config()
    config["benchmark_targets"] = {
        f"{benchmark_id}:{binding['split']}": _target_binding(
            benchmark_id,
            binding["split"],
            binding["source_revision"],
        )
    }
    return config


def _install_gate_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    benchmark_id: str,
    manifest: ExperimentManifest,
    driver_config_sha256: str,
    mutation: str | None = None,
) -> None:
    binding = _GATED_TARGETS[benchmark_id]
    inventory_bytes = b'["checkpoint-1"]\n'
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    inventory_path = tmp_path / f"{benchmark_id}.units.json"
    inventory_path.write_bytes(inventory_bytes)
    monkeypatch.setenv(binding["inventory_env"], str(inventory_path))
    monkeypatch.setenv(binding["inventory_sha_env"], inventory_sha256)

    identity = {
        "benchmark_id": benchmark_id,
        "split": binding["split"],
        "source_revision": binding["source_revision"],
        "manifest_hash": manifest.manifest_hash,
        "driver_config_sha256": driver_config_sha256,
        "unit_inventory_sha256": inventory_sha256,
    }
    bridge = {
        "bridge_id": "fixture.atomic-bridge",
        "bridge_revision": _BRIDGE_REVISION,
    }
    scorer = {
        "scorer_id": "fixture.official-scorer",
        "scorer_revision": _SCORER_REVISION,
    }
    verification = {
        "atomic_replay_contract": ATOMIC_EXTERNAL_PAIR_CONTRACT,
        "reset_mode": binding["reset_mode"],
        "reset_verified": True,
        "reset_receipt_sha256": "b" * 64,
        "no_op_parity_verified": True,
        "no_op_parity_receipt_sha256": "c" * 64,
    }
    if mutation in identity:
        identity[mutation] = "f" * 64
    elif mutation == "source_revision":
        identity["source_revision"] = "unexpected-source-revision"
    elif mutation == "bridge_revision":
        bridge["bridge_revision"] = "f" * 64
    elif mutation == "scorer_revision":
        scorer["scorer_revision"] = "f" * 64
    elif mutation == "reset_mode":
        verification["reset_mode"] = "unverified_reset"
    elif mutation == "no_op_parity":
        verification["no_op_parity_verified"] = False

    scope = {
        **identity,
        "bridge": bridge,
        "scorer": scorer,
        "verification": verification,
    }
    receipt = {
        "schema_version": 1,
        "artifact": "agentinputlab.external_arena_gate_receipt",
        "status": "approved",
        **identity,
        "bridge": bridge,
        "scorer": scorer,
        "verification": verification,
        "approval": {
            "approved": mutation != "approval",
            "approver_id": "fixture-review-board",
            "approval_revision": "d" * 64,
            "approved_at": "2026-07-24T00:00:00Z",
            "scope_sha256": content_hash(scope),
        },
    }
    raw = (canonical_json(receipt) + "\n").encode()
    path = tmp_path / f"{benchmark_id}.gate.json"
    path.write_bytes(raw)
    monkeypatch.setenv(binding["gate_env"], str(path))
    monkeypatch.setenv(binding["gate_sha_env"], hashlib.sha256(raw).hexdigest())


def _discovery_manifest() -> ExperimentManifest:
    return ExperimentManifest(
        schema_version=1,
        study_id="fixture.external-arena.discovery",
        stage=ExperimentStage.DISCOVERY,
        estimand_profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id="tau3.banking_knowledge",
                split="discovery_train",
                unit_ids=("train-checkpoint",),
                external=True,
                source_revision=_SOURCE_REVISION,
            ),
            ExperimentBenchmarkTarget(
                cell_id="tau3.banking_knowledge",
                split="discovery_calibration",
                unit_ids=("calibration-checkpoint",),
                external=True,
                source_revision=_SOURCE_REVISION,
            ),
        ),
        arms=(
            ExperimentArmSpec(
                arm_id="no_op",
                release_id="baseline.no_op",
                family=SemanticFamily.NO_OP,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            ),
            ExperimentArmSpec(
                arm_id="rewrite",
                release_id="fixture.rewrite.v1",
                family=SemanticFamily.REFORMULATE,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            ),
        ),
        seeds=(17,),
        model_slots=("anchor_a",),
        metadata={
            "requires_atomic_external_replay": True,
            "selector_training_allowed": True,
            "require_driver_unit_preflight": True,
        },
    )


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    global BRIDGE_ID_OVERRIDE, BUILDER_BLOCKER, DRIFT_BASELINE
    global INVALID_OUTCOME_MODE, SELECTOR_INPUT_DRIFT
    PAIR_REQUESTS.clear()
    DRIFT_BASELINE = False
    BRIDGE_ID_OVERRIDE = None
    BUILDER_BLOCKER = None
    SELECTOR_INPUT_DRIFT = False
    INVALID_OUTCOME_MODE = None


def test_multi_arm_cell_runs_exact_atomic_pairs_and_emits_audit_receipts(
    tmp_path: Path,
) -> None:
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()

    assert driver.fingerprint()["studio_campaign_attestation"] == {
        "schema_version": 1,
        "shared_dependency_slots": True,
        "selector_group_exclusion": True,
        "typed_gate_receipts": True,
    }
    assert driver.preflight(manifest) == ()
    result = driver.execute_cell(manifest, _cell(manifest))

    assert result.status == "succeeded"
    assert [item["arm_id"] for item in result.payload["arms"]] == [
        "no_op",
        "rewrite",
        "acquire",
    ]
    assert len(PAIR_REQUESTS) == 2
    assert {
        (request.origin_state_hash, request.replay_identity, request.seed)
        for request in PAIR_REQUESTS
    } == {("4" * 64, "5" * 64, 17)}
    assert all(request.locked_dependency_hashes == _DEPENDENCIES for request in PAIR_REQUESTS)
    assert len(result.payload["atomic_pairs"]) == 2
    assert len(result.resource_receipts) == 2
    assert tuple(
        item["resource_receipt_sha256s"] for item in result.payload["atomic_pairs"]
    ) == tuple((content_hash(receipt),) for receipt in result.resource_receipts)
    assert set(result.payload["pre_action"]) == {
        "state_hash",
        "feature_hash",
        "feature_schema_id",
        "feature_schema_hash",
        "provenance_hash",
        "defect_scores",
    }
    assert (
        result.payload["pre_action"]["state_hash"]
        == result.payload["checkpoint_identity"]["visible_state_sha256"]
    )
    assert len(result.payload["pre_action"]["feature_hash"]) == 64
    assert set(result.payload["pre_action"]["defect_scores"]) == {
        "reformulate",
        "decompose",
        "acquire",
        "clarify",
        "curate",
    }
    assert result.payload["arms"][0]["metrics"] == {
        "contract_preserved": 1.0,
        "cost": 1.0,
        "official_reward": 0.25,
        "privileged_leakage": 0.0,
        "valid_success": 0.25,
    }
    composition = result.payload["arms"][0]["valid_success_composition"]
    assert composition["formula_id"] == "agentinputlab.valid_success.v1"
    assert composition["components"] == {
        "official_success": 0.25,
        "contract_preserved": 1.0,
        "privileged_leakage": 0.0,
    }
    assert composition["scorer_id"] == "fixture.official-scorer"
    assert composition["scorer_revision"] == _SCORER_REVISION
    assert composition["trace_hash"] == "a" * 64
    assert len(composition["composition_receipt_sha256"]) == 64
    assert result.payload["observed_dependency_hashes"] == _DEPENDENCIES
    assert result.payload["not_applicable_dependency_slots"] == ()
    assert "selector_record" not in result.payload


def test_checkpoint_group_is_required_and_bound_into_identity() -> None:
    checkpoint = CheckpointBuilder().build(
        ExternalCheckpointBuildRequest(
            benchmark_id="tau3.banking_knowledge",
            split="discovery_train",
            checkpoint_id="checkpoint-1",
            source_revision=_SOURCE_REVISION,
            model_slot="anchor_a",
            seed=17,
            builder_parameters={"capture_contract": "fixture-v1"},
        )
    )

    assert (
        checkpoint.identity_hash
        != replace(
            checkpoint,
            group_id="another-fixture-group",
        ).identity_hash
    )
    with pytest.raises(ValidationError, match="group_id"):
        replace(checkpoint, group_id="")


def test_external_discovery_run_builds_selector_freeze_without_manual_record(
    tmp_path: Path,
) -> None:
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _discovery_manifest()
    plan = build_experiment_plan(
        manifest,
        {
            "tau3.banking_knowledge:discovery_train": ("train-checkpoint",),
            "tau3.banking_knowledge:discovery_calibration": ("calibration-checkpoint",),
        },
    )
    assert driver.preflight(manifest) == ()
    store = ExperimentArtifactStore(
        tmp_path / "runs",
        manifest=manifest,
        plan=plan,
        driver_fingerprint=driver.fingerprint(),
    )
    store.initialize(resume=False)

    for cell in plan.cells:
        result = driver.execute_cell(manifest, cell)
        assert set(result.payload["selector_record"]) == {
            "schema_version",
            "state",
            "group_id",
            "eligible_families",
        }
        assert result.payload["selector_record"]["group_id"] == (f"fixture-group-{cell.unit_id}")
        assert result.payload["selector_record"]["eligible_families"] == ("reformulate",)
        store.write_cell_result(
            cell.cell_id,
            {
                "schema_version": 1,
                "study_id": cell.study_id,
                "manifest_hash": manifest.manifest_hash,
                "plan_hash": plan.plan_hash,
                "cell": {**to_jsonable(cell), "cell_id": cell.cell_id},
                "result": result.to_dict(),
                "finished_at_utc": "2026-01-01T00:00:00Z",
                "claims": {
                    "official_parity_verified": False,
                    "confirmatory_candidate_record": False,
                    "paper_claim_authorized": False,
                    "resource_receipt_count": len(result.resource_receipts),
                },
            },
        )

    freeze = build_selector_discovery_freeze(
        (store.run_dir,),
        revision="external-arena-fixture-v1",
        extractor=VisibleStateFeatureExtractor(),
        utility_definition=UtilityDefinition("valid_success"),
        representative_by_family={
            SemanticFamily.REFORMULATE: "fixture.rewrite.v1",
        },
    )

    assert {row.split for row in freeze.training_table.examples} == {
        "train",
        "calibration",
    }
    assert len(freeze.outcomes) == 4


def test_cross_pair_baseline_drift_aborts_the_whole_cell(tmp_path: Path) -> None:
    global DRIFT_BASELINE
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    DRIFT_BASELINE = True

    with pytest.raises(ValidationError, match="NO_OP metrics, trace, usage"):
        driver.execute_cell(manifest, _cell(manifest))
    assert len(PAIR_REQUESTS) == 2


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("missing_component", "missing valid-success components"),
        ("precomputed", "must not precompute valid_success"),
        ("out_of_range", "must be in \\[0, 1\\]"),
    ),
)
def test_host_valid_success_composition_is_fail_closed(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    global INVALID_OUTCOME_MODE
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    INVALID_OUTCOME_MODE = mode

    with pytest.raises(ValidationError, match=message):
        driver.execute_cell(manifest, _cell(manifest))


def test_constructed_bridge_identity_mismatch_fails_before_evaluation(
    tmp_path: Path,
) -> None:
    global BRIDGE_ID_OVERRIDE
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    BRIDGE_ID_OVERRIDE = "unexpected.bridge"

    with pytest.raises(ValidationError, match="checksum-locked identity"):
        driver.execute_cell(manifest, _cell(manifest))
    assert PAIR_REQUESTS == []


def test_policy_arm_requires_and_exports_exact_selector_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = _install_policy_artifacts(tmp_path, monkeypatch)
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest(policy="selector")
    assert driver.preflight(manifest) == ()

    result = driver.execute_cell(manifest, _cell(manifest))
    selector = next(item for item in result.payload["arms"] if item["arm_id"] == "selector")
    assert selector["decision"] == {
        "selected_family": "reformulate",
        "selected_release_id": "fixture.rewrite.v1",
        "abstained": False,
        "predicted_gain": 0.2,
        "confidence": 0.8,
        "selector_bundle_hash": identities["selector_bundle_hash"],
    }
    assert selector["selector_input_receipt"] == {
        "state_hash": result.payload["pre_action"]["state_hash"],
        "feature_hash": result.payload["pre_action"]["feature_hash"],
    }
    assert len(selector["selector_decision_sha256"]) == 64
    policy_fingerprint = driver.fingerprint()["resolved_policy_bindings"]["selector"]
    assert (
        policy_fingerprint["selector_bundle"]["bundle_hash"] == identities["selector_bundle_hash"]
    )
    assert str(tmp_path) not in canonical_json(policy_fingerprint)


def test_locked_global_policy_binds_manifest_not_selector_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = _install_policy_artifacts(tmp_path, monkeypatch)
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest(policy="global")
    assert driver.preflight(manifest) == ()

    result = driver.execute_cell(manifest, _cell(manifest))
    global_arm = next(item for item in result.payload["arms"] if item["arm_id"] == "global")

    assert global_arm["decision"]["policy_manifest_sha256"] == identities["policy_file_sha256"]
    assert "selector_bundle_hash" not in global_arm["decision"]
    binding = driver.fingerprint()["resolved_policy_bindings"]["global"]
    assert binding["selected_representative"] == {
        "family": "reformulate",
        "release_id": "fixture.rewrite.v1",
        "calibration_split_digest": identities["calibration_split_digest"],
    }
    assert binding["group_partition"]["excluded_member_count"] == 2
    assert len(binding["group_partition"]["excluded_member_hashes"]) == 2


def test_policy_bundle_is_strictly_parsed_and_rechecked_before_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_policy_artifacts(tmp_path, monkeypatch)
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest(policy="selector")

    monkeypatch.setenv("AILAB_SELECTOR_IDENTITY", "{}")
    blockers = driver.preflight(manifest)
    assert any("identity" in item for item in blockers)
    _install_policy_artifacts(tmp_path, monkeypatch)
    assert driver.preflight(manifest) == ()

    bundle_path = Path(os.environ["AILAB_SELECTOR_BUNDLE"])
    bundle_path.write_bytes(bundle_path.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="checksum differs"):
        driver.execute_cell(manifest, _cell(manifest))
    assert PAIR_REQUESTS == []


def test_policy_decision_input_must_match_pre_action_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global SELECTOR_INPUT_DRIFT
    _install_policy_artifacts(tmp_path, monkeypatch)
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest(policy="selector")
    assert driver.preflight(manifest) == ()
    SELECTOR_INPUT_DRIFT = True

    with pytest.raises(ValidationError, match="selector input identity differs"):
        driver.execute_cell(manifest, _cell(manifest))
    assert PAIR_REQUESTS == []


def test_config_and_builder_hashes_are_fail_closed(tmp_path: Path) -> None:
    config = _config()
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        ExternalArenaExperimentDriver.from_file(path, expected_sha256="0" * 64)

    config["arms"]["rewrite"]["payload_builder"]["fingerprint_sha256"] = "f" * 64
    path, checksum = _write_config(tmp_path, config)
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    blockers = driver.preflight(_manifest())
    assert any("fingerprint checksum mismatch" in blocker for blocker in blockers)

    config = _config()
    config["benchmark_targets"]["tau3.banking_knowledge:base"]["outcome_contract"][
        "official_success_metric"
    ] = "cost"
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match="registered adapter metric"):
        ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)


def test_external_dependency_lock_requires_every_shared_slot(tmp_path: Path) -> None:
    config = _config()
    dependencies = config["benchmark_targets"]["tau3.banking_knowledge:base"]["bridge_lock"][
        "dependency_hashes_by_model_slot"
    ]["anchor_a"]
    del dependencies["retriever"]
    path, checksum = _write_config(tmp_path, config)

    with pytest.raises(ValidationError, match="exact shared dependency slots"):
        ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    assert external_dependency_not_applicable_hash("reader") == content_hash(
        {
            "agentinputlab_dependency": "not_applicable",
            "schema_version": 1,
            "slot": "reader",
        }
    )
    with pytest.raises(ValidationError, match="unknown external dependency slot"):
        external_dependency_not_applicable_hash("not-a-slot")


@pytest.mark.parametrize("benchmark_id", tuple(_GATED_TARGETS))
def test_gated_arena_requires_typed_cross_bound_approval_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_id: str,
) -> None:
    config_path, config_sha256 = _write_config(
        tmp_path,
        _gated_config(benchmark_id),
    )
    driver = ExternalArenaExperimentDriver.from_file(
        config_path,
        expected_sha256=config_sha256,
    )
    manifest = _gated_manifest(benchmark_id)
    _install_gate_receipt(
        tmp_path,
        monkeypatch,
        benchmark_id=benchmark_id,
        manifest=manifest,
        driver_config_sha256=config_sha256,
    )
    binding = _GATED_TARGETS[benchmark_id]

    arbitrary_path = tmp_path / f"{benchmark_id}.arbitrary.json"
    arbitrary_raw = b'{"approved":true}\n'
    arbitrary_path.write_bytes(arbitrary_raw)
    monkeypatch.setenv(binding["gate_env"], str(arbitrary_path))
    monkeypatch.setenv(
        binding["gate_sha_env"],
        hashlib.sha256(arbitrary_raw).hexdigest(),
    )
    blockers = driver.preflight(manifest)
    assert any("external gate binding" in item for item in blockers)

    _install_gate_receipt(
        tmp_path,
        monkeypatch,
        benchmark_id=benchmark_id,
        manifest=manifest,
        driver_config_sha256=config_sha256,
    )
    assert driver.preflight(manifest) == ()
    receipt = driver.fingerprint()["resolved_gate_bindings"][
        f"{benchmark_id}:{_GATED_TARGETS[benchmark_id]['split']}"
    ]
    assert receipt["manifest_hash"] == manifest.manifest_hash
    assert receipt["driver_config_sha256"] == config_sha256
    assert receipt["verification"]["reset_verified"] is True
    assert receipt["verification"]["no_op_parity_verified"] is True
    assert str(tmp_path) not in canonical_json(receipt)


def test_gate_receipt_is_rechecked_before_the_first_external_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_id = "toolsandbox.common_support"
    binding = _GATED_TARGETS[benchmark_id]
    config_path, config_sha256 = _write_config(
        tmp_path,
        _gated_config(benchmark_id),
    )
    driver = ExternalArenaExperimentDriver.from_file(
        config_path,
        expected_sha256=config_sha256,
    )
    manifest = _gated_manifest(benchmark_id)
    _install_gate_receipt(
        tmp_path,
        monkeypatch,
        benchmark_id=benchmark_id,
        manifest=manifest,
        driver_config_sha256=config_sha256,
    )
    assert driver.preflight(manifest) == ()
    gate_path = Path(os.environ[binding["gate_env"]])
    gate_path.write_bytes(gate_path.read_bytes() + b" ")
    cell = ExperimentCell(
        study_id=manifest.study_id,
        benchmark_id=benchmark_id,
        split=binding["split"],
        unit_id="checkpoint-1",
        model_slot="anchor_a",
        seed=17,
        arm_ids=tuple(arm.arm_id for arm in manifest.arms),
        manifest_hash=manifest.manifest_hash,
    )

    with pytest.raises(ArtifactError, match="checksum differs"):
        driver.execute_cell(manifest, cell)
    assert PAIR_REQUESTS == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("benchmark_id", "benchmark_id differs"),
        ("split", "split differs"),
        ("source_revision", "source_revision differs"),
        ("manifest_hash", "manifest_hash differs"),
        ("driver_config_sha256", "driver_config_sha256 differs"),
        ("unit_inventory_sha256", "unit_inventory_sha256 differs"),
        ("bridge_revision", "bridge differs"),
        ("scorer_revision", "scorer differs"),
        ("reset_mode", "does not approve atomic reset"),
        ("no_op_parity", "does not approve atomic reset"),
        ("approval", "approval.approved must be true"),
    ),
)
def test_gate_receipt_rejects_every_cross_binding_or_approval_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    benchmark_id = "toolsandbox.common_support"
    config_path, config_sha256 = _write_config(
        tmp_path,
        _gated_config(benchmark_id),
    )
    driver = ExternalArenaExperimentDriver.from_file(
        config_path,
        expected_sha256=config_sha256,
    )
    manifest = _gated_manifest(benchmark_id)
    _install_gate_receipt(
        tmp_path,
        monkeypatch,
        benchmark_id=benchmark_id,
        manifest=manifest,
        driver_config_sha256=config_sha256,
        mutation=mutation,
    )

    blockers = driver.preflight(manifest)

    assert any(message in item for item in blockers)


def test_preflight_builder_blockers_redact_secrets(tmp_path: Path) -> None:
    global BUILDER_BLOCKER
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    BUILDER_BLOCKER = "cannot connect with bearer super-secret-token"

    rendered = "\n".join(driver.preflight(_manifest()))

    assert "super-secret-token" not in rendered
    assert "<redacted>" in rendered


def test_required_unit_preflight_checks_every_resolved_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    key = "tau3.banking_knowledge:base"

    assert driver.preflight_units(manifest, {key: ("checkpoint-1",)}) == ()
    blockers = driver.preflight_units(
        manifest,
        {key: ("checkpoint-1", "not-in-fixture-inventory")},
    )
    assert any("not-in-fixture-inventory" in item for item in blockers)

    monkeypatch.delattr(CheckpointBuilder, "preflight_units")
    blockers = driver.preflight_units(manifest, {key: ("checkpoint-1",)})
    assert any("lacks required read-only preflight_units" in item for item in blockers)


def test_selector_policy_excludes_discovery_groups_before_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_policy_artifacts(
        tmp_path,
        monkeypatch,
        train_checkpoint="checkpoint-1",
    )
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest(policy="selector")
    key = "tau3.banking_knowledge:base"
    assert driver.preflight(manifest) == ()

    blockers = driver.preflight_units(manifest, {key: ("checkpoint-1",)})
    assert any("discovery train/calibration group" in item for item in blockers)
    with pytest.raises(ValidationError, match="discovery train/calibration group"):
        driver.execute_cell(manifest, _cell(manifest))
    assert PAIR_REQUESTS == []


def test_selector_policy_requires_read_only_group_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_policy_artifacts(tmp_path, monkeypatch)
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest(policy="selector")
    assert driver.preflight(manifest) == ()
    monkeypatch.delattr(CheckpointBuilder, "preflight_group_ids")

    blockers = driver.preflight_units(
        manifest,
        {"tau3.banking_knowledge:base": ("checkpoint-1",)},
    )

    assert any("lacks required read-only preflight_group_ids" in item for item in blockers)


def test_partial_cell_or_checkpoint_identity_drift_is_rejected(tmp_path: Path) -> None:
    path, checksum = _write_config(tmp_path, _config())
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    partial = replace(_cell(manifest), arm_ids=("no_op", "rewrite"))
    with pytest.raises(ValidationError, match="arms differ"):
        driver.execute_cell(manifest, partial)


@pytest.mark.parametrize(
    ("benchmark_id", "split", "source_revision"),
    (
        (
            "tau3.banking_knowledge",
            "base",
            "cf71a8070269883e38a365ffa85f78f46844c1f4",
        ),
        ("tau3.retail", "test", "cf71a8070269883e38a365ffa85f78f46844c1f4"),
        (
            "state_bench.common_support",
            "external_checkpoints",
            "4efcbf2d4fe60df04878859b692d9391f3d5b33a",
        ),
        (
            "toolsandbox.common_support",
            "external_checkpoints",
            "165848b9a78cead7ca7fe7c89c688b58e6501219",
        ),
        (
            "appworld_ul.common_support",
            "external_checkpoints",
            "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
        ),
        ("webarena.external", "locked_transfer", "reviewed-webarena-revision"),
    ),
)
def test_config_accepts_the_supported_arena_locks(
    tmp_path: Path,
    benchmark_id: str,
    split: str,
    source_revision: str,
) -> None:
    config = _config()
    config["benchmark_targets"] = {
        f"{benchmark_id}:{split}": _target_binding(
            benchmark_id,
            split,
            source_revision,
        )
    }
    path, checksum = _write_config(tmp_path, config)
    driver = ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)

    assert (
        driver.fingerprint()["benchmark_targets"][f"{benchmark_id}:{split}"]["source_revision"]
        == source_revision
    )


@pytest.mark.parametrize("source,metric", [
    ("different-source", "terminal_success"),
    (_SOURCE_REVISION, "official_reward"),
    (_SOURCE_REVISION, "cost"),
])
def test_retail_target_rejects_wrong_source_and_raw_reward_as_primary(
    tmp_path: Path, source: str, metric: str,
) -> None:
    config = _config()
    target = _target_binding("tau3.retail", "test", source)
    target["outcome_contract"]["official_success_metric"] = metric
    config["benchmark_targets"] = {"tau3.retail:test": target}
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ValidationError):
        ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)


@pytest.mark.parametrize("missing", ["user_simulator", "nl_assertion_judge", "scorer"])
def test_retail_bridge_lock_requires_both_environment_models(tmp_path: Path, missing: str) -> None:
    config = _config()
    target = _target_binding("tau3.retail", "test", _SOURCE_REVISION)
    del target["bridge_lock"]["dependency_hashes_by_model_slot"]["anchor_a"][missing]
    config["benchmark_targets"] = {"tau3.retail:test": target}
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match="exact shared"):
        ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)


def test_retail_environment_slots_do_not_loosen_banking_identity(tmp_path: Path) -> None:
    config = _config()
    target = _target_binding("tau3.banking_knowledge", "base", _SOURCE_REVISION)
    target["bridge_lock"]["dependency_hashes_by_model_slot"]["anchor_a"]["user_simulator"] = "d" * 64
    config["benchmark_targets"] = {"tau3.banking_knowledge:base": target}
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match="exact shared"):
        ExternalArenaExperimentDriver.from_file(path, expected_sha256=checksum)
