from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest import SkipTest
from unittest.mock import patch

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.contracts.enums import (
    ProfileKind,
    SemanticFamily,
)
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.contracts.state import ExecutionBudget, FrozenAgentState
from agentinputlab.experiments.builders import (
    ATOMIC_PROCESS_RESPONSE_PROTOCOL,
    CHECKPOINT_INVENTORY_PROTOCOL,
    RELEASE_PAYLOAD_PROTOCOL,
    AtomicProcessBridgeBuilder,
    FrozenCheckpointInventoryBuilder,
    ReleaseArmPayloadBuilder,
)
from agentinputlab.experiments.builders.external_process_arena import (
    _AtomicProcessBridge,
    _parse_atomic_response,
)
from agentinputlab.experiments.drivers.external_arena import (
    ExternalArmPayloadBuildRequest,
    ExternalBridgeBuildRequest,
    ExternalCheckpointBuildRequest,
    ExternalUnitPreflightRequest,
)
from agentinputlab.experiments.schema import ExperimentArmKind
from agentinputlab.observability import ResourceLimits, ResourceUsage

_H = "a" * 64
_D = {
    name: content_hash({"dependency": name})
    for name in (
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
    )
}


def _state():
    return FrozenAgentState(
        query="Please compare the visible options.",
        task_id="opaque-visible-id",
        instruction="Use only the displayed tools.",
        tool_names=("search", "respond"),
        metadata={"turn_index": 1, "benchmark_cell": "tau3.banking_knowledge"},
        budget=ExecutionBudget(),
    )


def _inventory(tmp_path, monkeypatch, *, privileged=False):
    state = _state()
    visible_state = to_jsonable(state)
    if privileged:
        visible_state["metadata"]["gold_answer"] = "leak"
    row = {
        "benchmark_id": "tau3.banking_knowledge",
        "split": "base",
        "checkpoint_id": "checkpoint-1",
        "group_id": "group_" + "b" * 64,
        "source_revision": "c" * 40,
        "origin_state_hash": _H,
        "replay_identity": "d" * 64,
        "locked_dependency_hashes": _D,
        "visible_state": visible_state,
        "visible_state_hash": state.state_hash,
        "eligible_families": [
            item.value
            for item in (
                SemanticFamily.REFORMULATE,
                SemanticFamily.DECOMPOSE,
                SemanticFamily.ACQUIRE,
                SemanticFamily.CLARIFY,
                SemanticFamily.CURATE,
            )
        ],
    }
    payload = {
        "schema_version": 1,
        "protocol": CHECKPOINT_INVENTORY_PROTOCOL,
        "checkpoint_count": 1,
        "checkpoints": [row],
    }
    path = tmp_path / "inventory.json"
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    monkeypatch.setenv("AILAB_TEST_INVENTORY", str(path))
    monkeypatch.setenv("AILAB_TEST_INVENTORY_SHA256", expected_sha256)
    return {
        "schema_version": 1,
        "protocol": CHECKPOINT_INVENTORY_PROTOCOL,
        "path_environment": "AILAB_TEST_INVENTORY",
        "sha256_environment": "AILAB_TEST_INVENTORY_SHA256",
        "expected_sha256": expected_sha256,
    }


def test_checkpoint_inventory_builds_and_preflights_units(tmp_path, monkeypatch):
    parameters = _inventory(tmp_path, monkeypatch)
    builder = FrozenCheckpointInventoryBuilder()
    assert builder.preflight(parameters) == ()
    unit_request = ExternalUnitPreflightRequest(
        benchmark_id="tau3.banking_knowledge",
        split="base",
        source_revision="c" * 40,
        unit_ids=("checkpoint-1",),
        builder_parameters=parameters,
    )
    assert builder.preflight_units(unit_request) == ()
    assert builder.preflight_group_ids(unit_request) == {"checkpoint-1": "group_" + "b" * 64}
    checkpoint = builder.build(
        ExternalCheckpointBuildRequest(
            benchmark_id="tau3.banking_knowledge",
            split="base",
            checkpoint_id="checkpoint-1",
            source_revision="c" * 40,
            model_slot="anchor",
            seed=17,
            builder_parameters=parameters,
        )
    )
    assert checkpoint.visible_state.query == "Please compare the visible options."
    assert checkpoint.locked_dependency_hashes == _D


def test_checkpoint_inventory_selects_dependencies_by_model_slot(
    tmp_path,
    monkeypatch,
):
    parameters = _inventory(tmp_path, monkeypatch)
    path = tmp_path / "inventory.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    row = value["checkpoints"][0]
    row.pop("locked_dependency_hashes")
    anchor_a = dict(_D)
    anchor_b = {**_D, "agent_model": content_hash({"agent_model": "anchor_b"})}
    row["locked_dependency_hashes_by_model_slot"] = {
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
    }
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    monkeypatch.setenv("AILAB_TEST_INVENTORY_SHA256", digest)
    parameters = {**parameters, "expected_sha256": digest}
    builder = FrozenCheckpointInventoryBuilder()
    assert builder.preflight(parameters) == ()

    def build(model_slot):
        return builder.build(
            ExternalCheckpointBuildRequest(
                benchmark_id="tau3.banking_knowledge",
                split="base",
                checkpoint_id="checkpoint-1",
                source_revision="c" * 40,
                model_slot=model_slot,
                seed=17,
                builder_parameters=parameters,
            )
        )

    assert build("anchor_a").locked_dependency_hashes == anchor_a
    assert build("anchor_b").locked_dependency_hashes == anchor_b
    with pytest.raises(ArtifactError, match="no dependency binding"):
        build("unbound")


def test_checkpoint_inventory_rejects_privileged_visible_fields(tmp_path, monkeypatch):
    parameters = _inventory(tmp_path, monkeypatch, privileged=True)
    blockers = FrozenCheckpointInventoryBuilder().preflight(parameters)
    assert len(blockers) == 1
    assert "privileged field" in blockers[0]


def test_checkpoint_inventory_rejects_digest_environment_drift_after_preflight(
    tmp_path,
    monkeypatch,
):
    parameters = _inventory(tmp_path, monkeypatch)
    builder = FrozenCheckpointInventoryBuilder()
    assert builder.preflight(parameters) == ()

    replacement = tmp_path / "replacement-inventory.json"
    replacement.write_text('{"different":true}\n', encoding="utf-8")
    monkeypatch.setenv("AILAB_TEST_INVENTORY", str(replacement))
    monkeypatch.setenv(
        "AILAB_TEST_INVENTORY_SHA256",
        hashlib.sha256(replacement.read_bytes()).hexdigest(),
    )

    request = ExternalCheckpointBuildRequest(
        benchmark_id="tau3.banking_knowledge",
        split="base",
        checkpoint_id="checkpoint-1",
        source_revision="c" * 40,
        model_slot="anchor",
        seed=17,
        builder_parameters=parameters,
    )
    with pytest.raises(ArtifactError, match="differs from the config lock"):
        builder.build(request)


def test_atomic_bridge_snapshots_only_allowlisted_environment_and_hides_repr(
    tmp_path,
    monkeypatch,
):
    parameters = _inventory(tmp_path, monkeypatch)
    checkpoint = FrozenCheckpointInventoryBuilder().build(
        ExternalCheckpointBuildRequest(
            benchmark_id="tau3.banking_knowledge",
            split="base",
            checkpoint_id="checkpoint-1",
            source_revision="c" * 40,
            model_slot="anchor",
            seed=17,
            builder_parameters=parameters,
        )
    )
    monkeypatch.setenv("MODEL_ENDPOINT", "https://example.invalid/v1")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-captured")
    worker = SimpleNamespace(pass_environment=("MODEL_ENDPOINT",))
    parsed = {
        "benchmark_id": checkpoint.benchmark_id,
        "source_revision": checkpoint.source_revision,
        "bridge_id": "test.atomic",
        "bridge_revision": "1" * 64,
        "scorer_id": "test.scorer",
        "scorer_revision": "2" * 64,
        "resource_limits": ResourceLimits(),
        "pipe_limits": {
            "max_input_bytes": 1024,
            "max_stdout_bytes": 1024,
            "max_stderr_bytes": 1024,
        },
    }
    request = ExternalBridgeBuildRequest(
        checkpoint=checkpoint,
        model_slot="anchor",
        seed=17,
        builder_parameters={"schema_version": 1},
    )
    with (
        patch(
            "agentinputlab.experiments.builders.external_process_arena._parse_bridge_parameters",
            return_value=parsed,
        ),
        patch(
            "agentinputlab.experiments.builders.external_process_arena._load_atomic_worker",
            return_value=worker,
        ),
        patch(
            "agentinputlab.experiments.builders.external_process_arena."
            "attest_locked_process_worker",
            return_value={},
        ),
    ):
        bridge = AtomicProcessBridgeBuilder().build(request)

    assert isinstance(bridge, _AtomicProcessBridge)
    assert bridge.environment == {"MODEL_ENDPOINT": "https://example.invalid/v1"}
    assert _AtomicProcessBridge.__dataclass_fields__["environment"].repr is False
    assert "https://example.invalid/v1" not in repr(bridge)
    assert "UNRELATED_SECRET" not in repr(bridge)
    assert "must-not-be-captured" not in repr(bridge)


def test_release_payload_keeps_identity_and_settings(tmp_path, monkeypatch):
    parameters = _inventory(tmp_path, monkeypatch)
    checkpoint = FrozenCheckpointInventoryBuilder().build(
        ExternalCheckpointBuildRequest(
            benchmark_id="tau3.banking_knowledge",
            split="base",
            checkpoint_id="checkpoint-1",
            source_revision="c" * 40,
            model_slot="anchor",
            seed=17,
            builder_parameters=parameters,
        )
    )
    builder_parameters = {
        "schema_version": 1,
        "protocol": RELEASE_PAYLOAD_PROTOCOL,
        "allowed_release_ids": [
            "baseline.no_op",
            "plan.plan_and_solve.ps_prompt",
        ],
        "release_parameters": {"prompt_revision": "locked"},
    }
    request = ExternalArmPayloadBuildRequest(
        checkpoint=checkpoint,
        arm_id="decompose",
        release_id="plan.plan_and_solve.ps_prompt",
        family=SemanticFamily.DECOMPOSE,
        profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        arm_kind=ExperimentArmKind.RELEASE,
        model_slot="anchor",
        seed=17,
        builder_parameters=builder_parameters,
        arm_parameters={"max_steps": 4},
        pre_action={"state_hash": checkpoint.visible_state.state_hash, "feature_hash": _H},
    )
    payload = ReleaseArmPayloadBuilder().build(request)
    assert payload.payload["release_id"] == request.release_id
    assert payload.payload["arm_parameters"] == {"max_steps": 4}

    with pytest.raises(ValidationError, match="selector policy"):
        ReleaseArmPayloadBuilder().build(
            ExternalArmPayloadBuildRequest(
                checkpoint=checkpoint,
                arm_id="selector",
                release_id="policy.eligibility_aware_selector",
                family=SemanticFamily.COMPOSITE,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
                arm_kind=ExperimentArmKind.POLICY,
                model_slot="anchor",
                seed=17,
                builder_parameters=builder_parameters,
                arm_parameters={},
                pre_action={
                    "state_hash": checkpoint.visible_state.state_hash,
                    "feature_hash": _H,
                },
            )
        )


def test_atomic_process_response_is_strict_and_typed():
    usage = to_jsonable(ResourceUsage(model_calls=1, input_tokens=3, output_tokens=2))
    arm_receipts = [
        {
            "role": role,
            "arm_id": arm_id,
            "origin_state_hash": _H,
            "replay_identity": "d" * 64,
            "seed": 17,
            "locked_dependency_hashes": _D,
            "runtime_instance_id": runtime_id,
            "completed": True,
        }
        for role, arm_id, runtime_id in (
            ("baseline", "no_op", "e" * 64),
            ("treatment", "candidate", "f" * 64),
        )
    ]
    results = [
        {
            "role": role,
            "arm_id": arm_id,
            "metrics": {
                "official_reward": reward,
                "contract_preserved": 1.0,
                "privileged_leakage": 0.0,
            },
            "trace_hash": trace,
            "usage": usage,
        }
        for role, arm_id, reward, trace in (
            ("baseline", "no_op", 0.0, "1" * 64),
            ("treatment", "candidate", 1.0, "2" * 64),
        )
    ]
    response = {
        "schema_version": 1,
        "protocol": ATOMIC_PROCESS_RESPONSE_PROTOCOL,
        "request_hash": "3" * 64,
        "receipt": {
            "benchmark_id": "tau3.banking_knowledge",
            "checkpoint_id": "checkpoint-1",
            "source_revision": "c" * 40,
            "bridge_id": "test.atomic",
            "bridge_revision": "4" * 64,
            "scorer_id": "tau3.official",
            "scorer_revision": "5" * 64,
            "origin_state_hash": _H,
            "replay_identity": "d" * 64,
            "seed": 17,
            "locked_dependency_hashes": _D,
            "arms": arm_receipts,
        },
        "arms": results,
    }
    parsed = _parse_atomic_response(
        json.dumps(response),
        expected_request_hash="3" * 64,
    )
    assert parsed.receipt.bridge_id == "test.atomic"
    assert parsed.arms[1].metrics["official_reward"] == 1.0

    opaque_secret = "opaqueValue987654321"
    response["arms"][0]["trace_hash"] = opaque_secret
    with pytest.raises(ArtifactError, match="copied from the worker environment") as caught:
        _parse_atomic_response(
            json.dumps(response),
            expected_request_hash="3" * 64,
            secret_values=(opaque_secret,),
        )
    assert opaque_secret not in str(caught.value)
    response["arms"][0]["trace_hash"] = "1" * 64

    response["extra"] = "not accepted"
    with pytest.raises(ValidationError, match="fields"):
        _parse_atomic_response(
            json.dumps(response),
            expected_request_hash="3" * 64,
        )
