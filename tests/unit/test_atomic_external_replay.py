from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.benchmarks import (
    ATOMIC_EXTERNAL_PAIR_CONTRACT,
    AtomicExternalReplayRunner,
    AtomicReplayArmReceipt,
    AtomicReplayArmRequest,
    AtomicReplayArmResult,
    AtomicReplayPairResult,
    AtomicReplayReceipt,
    AtomicReplayRequest,
)
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.observability import ResourceLimits, ResourceReceipt, ResourceUsage

_DEPENDENCIES = {
    "agent_model": "1" * 64,
    "tool_catalog": "2" * 64,
    "user_simulator": "3" * 64,
}


def _resource_receipt(*, wall_seconds: float = 0.1) -> ResourceReceipt:
    limits = ResourceLimits(wall_timeout_seconds=1)
    return ResourceReceipt(
        limit_hash=limits.limit_hash,
        wall_seconds=wall_seconds,
        cpu_seconds=0.05,
        peak_rss_bytes=4096,
        supervisor_identity={"kind": "fixture"},
        image_identity={"kind": "fixture"},
    )


def _request() -> AtomicReplayRequest:
    return AtomicReplayRequest(
        benchmark_id="fixture.interactive",
        checkpoint_id="checkpoint-1",
        source_revision="upstream-commit",
        origin_state_hash="4" * 64,
        replay_identity="5" * 64,
        seed=17,
        locked_dependency_hashes=_DEPENDENCIES,
        arms=(
            AtomicReplayArmRequest("baseline", "no-op", {"patch": "noop"}),
            AtomicReplayArmRequest("treatment", "rewrite", {"patch": "rewrite"}),
        ),
    )


def _arm_receipts(request: AtomicReplayRequest) -> tuple[AtomicReplayArmReceipt, ...]:
    return (
        AtomicReplayArmReceipt(
            role="baseline",
            arm_id="no-op",
            origin_state_hash=request.origin_state_hash,
            replay_identity=request.replay_identity,
            seed=request.seed,
            locked_dependency_hashes=request.locked_dependency_hashes,
            runtime_instance_id="6" * 64,
            completed=True,
        ),
        AtomicReplayArmReceipt(
            role="treatment",
            arm_id="rewrite",
            origin_state_hash=request.origin_state_hash,
            replay_identity=request.replay_identity,
            seed=request.seed,
            locked_dependency_hashes=request.locked_dependency_hashes,
            runtime_instance_id="7" * 64,
            completed=True,
        ),
    )


def _receipt(
    request: AtomicReplayRequest,
    *,
    bridge_id: str = "fixture.atomic-bridge",
    bridge_revision: str = "8" * 64,
) -> AtomicReplayReceipt:
    return AtomicReplayReceipt(
        benchmark_id=request.benchmark_id,
        checkpoint_id=request.checkpoint_id,
        source_revision=request.source_revision,
        bridge_id=bridge_id,
        bridge_revision=bridge_revision,
        scorer_id="fixture.official-scorer",
        scorer_revision="9" * 64,
        origin_state_hash=request.origin_state_hash,
        replay_identity=request.replay_identity,
        seed=request.seed,
        locked_dependency_hashes=request.locked_dependency_hashes,
        arms=_arm_receipts(request),
    )


def _pair_result(
    request: AtomicReplayRequest,
    *,
    receipt: AtomicReplayReceipt | None = None,
) -> AtomicReplayPairResult:
    return AtomicReplayPairResult(
        receipt=receipt or _receipt(request),
        arms=(
            AtomicReplayArmResult(
                role="baseline",
                arm_id="no-op",
                metrics={"reward": 0.25, "cost": 1.0},
                trace_hash="a" * 64,
                usage=ResourceUsage(model_calls=1),
            ),
            AtomicReplayArmResult(
                role="treatment",
                arm_id="rewrite",
                metrics={"reward": 0.75, "cost": 1.5},
                trace_hash="b" * 64,
                usage=ResourceUsage(model_calls=1),
            ),
        ),
    )


class FixtureAtomicBridge:
    atomic_replay_contract = ATOMIC_EXTERNAL_PAIR_CONTRACT
    benchmark_id = "fixture.interactive"
    source_revision = "upstream-commit"
    bridge_id = "fixture.atomic-bridge"
    bridge_revision = "8" * 64
    scorer_id = "fixture.official-scorer"
    scorer_revision = "9" * 64
    locked_dependency_hashes = _DEPENDENCIES

    def __init__(self) -> None:
        self.calls = 0
        self.result_override: object | None = None

    def evaluate_pair(self, request: AtomicReplayRequest) -> AtomicReplayPairResult:
        self.calls += 1
        if self.result_override is not None:
            return self.result_override  # type: ignore[return-value]
        return _pair_result(request)


class LegacySingleArmBridge:
    def evaluate(self, checkpoint: object, prediction: object) -> object:
        del checkpoint, prediction
        return object()


def test_atomic_bridge_is_called_once_and_receipt_drives_deltas() -> None:
    bridge = FixtureAtomicBridge()
    result = AtomicExternalReplayRunner().run(bridge, _request())

    assert bridge.calls == 1
    assert result.baseline.metrics == {"cost": 1.0, "reward": 0.25}
    assert result.treatment.metrics == {"cost": 1.5, "reward": 0.75}
    assert result.metric_deltas == {"cost": 0.5, "reward": 0.5}
    assert result.receipt.arms[0].runtime_instance_id != result.receipt.arms[1].runtime_instance_id
    with pytest.raises(TypeError):
        result.metric_deltas["reward"] = 0.0  # type: ignore[index]


def test_typed_resource_receipts_propagate_from_pair_and_bridge() -> None:
    request = _request()
    pair_receipt = _resource_receipt()
    bridge_receipt = _resource_receipt(wall_seconds=0.2)
    bridge = FixtureAtomicBridge()
    bridge.result_override = replace(
        _pair_result(request),
        resource_receipts=(pair_receipt,),
    )
    bridge.last_resource_receipt = bridge_receipt

    result = AtomicExternalReplayRunner().run(bridge, request)

    assert result.resource_receipts == (pair_receipt, bridge_receipt)
    with pytest.raises(TypeError, match="ResourceReceipt"):
        replace(
            _pair_result(request),
            resource_receipts=({"wall_seconds": 0.1},),  # type: ignore[arg-type]
        )


def test_untyped_bridge_last_resource_receipt_fails_closed() -> None:
    bridge = FixtureAtomicBridge()
    bridge.last_resource_receipt = {"wall_seconds": 0.1}

    with pytest.raises(TypeError, match="last_resource_receipt"):
        AtomicExternalReplayRunner().run(bridge, _request())


def test_single_arm_bridge_cannot_satisfy_atomic_protocol() -> None:
    with pytest.raises(TypeError, match="single-arm evaluation bridges are not accepted"):
        AtomicExternalReplayRunner().run(
            LegacySingleArmBridge(),  # type: ignore[arg-type]
            _request(),
        )


def test_bridge_identity_and_dependencies_are_checked_before_call() -> None:
    bridge = FixtureAtomicBridge()
    bridge.locked_dependency_hashes = {**_DEPENDENCIES, "agent_model": "f" * 64}

    with pytest.raises(ValidationError, match="locked dependencies"):
        AtomicExternalReplayRunner().run(bridge, _request())
    assert bridge.calls == 0


def test_receipt_must_match_request_and_bridge() -> None:
    bridge = FixtureAtomicBridge()
    request = _request()
    bridge.result_override = _pair_result(
        request,
        receipt=_receipt(request, bridge_id="different.bridge"),
    )

    with pytest.raises(ValidationError, match="receipt bridge_id"):
        AtomicExternalReplayRunner().run(bridge, request)
    assert bridge.calls == 1


def test_receipt_rejects_shared_runtime_seed_or_dependency_drift() -> None:
    request = _request()
    arms = _arm_receipts(request)
    with pytest.raises(ValidationError, match="different runtime instance"):
        replace(
            _receipt(request),
            arms=(arms[0], replace(arms[1], runtime_instance_id=arms[0].runtime_instance_id)),
        )
    with pytest.raises(ValidationError, match="shared origin, seed"):
        replace(
            _receipt(request),
            arms=(arms[0], replace(arms[1], seed=request.seed + 1)),
        )
    with pytest.raises(ValidationError, match="shared origin, seed"):
        replace(
            _receipt(request),
            arms=(
                arms[0],
                replace(
                    arms[1],
                    locked_dependency_hashes={
                        **_DEPENDENCIES,
                        "agent_model": "e" * 64,
                    },
                ),
            ),
        )


def test_partial_or_metric_incompatible_results_fail_closed() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="exactly two arms"):
        AtomicReplayPairResult(
            receipt=_receipt(request),
            arms=(
                AtomicReplayArmResult(
                    role="baseline",
                    arm_id="no-op",
                    metrics={"reward": 0.0},
                    trace_hash="a" * 64,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="metric schemas differ"):
        AtomicReplayPairResult(
            receipt=_receipt(request),
            arms=(
                AtomicReplayArmResult(
                    role="baseline",
                    arm_id="no-op",
                    metrics={"reward": 0.0},
                    trace_hash="a" * 64,
                ),
                AtomicReplayArmResult(
                    role="treatment",
                    arm_id="rewrite",
                    metrics={"other": 1.0},
                    trace_hash="b" * 64,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda request: replace(
                request,
                arms=(
                    SimpleNamespace(role="baseline", arm_id="no-op"),
                    request.arms[1],
                ),
            ),
            "AtomicReplayArmRequest",
        ),
        (
            lambda request: replace(
                _receipt(request),
                arms=(
                    SimpleNamespace(role="baseline", arm_id="no-op"),
                    _receipt(request).arms[1],
                ),
            ),
            "AtomicReplayArmReceipt",
        ),
        (
            lambda request: AtomicReplayPairResult(
                receipt=_receipt(request),
                arms=(
                    SimpleNamespace(
                        role="baseline",
                        arm_id="no-op",
                        metrics={"reward": 0.0},
                    ),
                    _pair_result(request).arms[1],
                ),
            ),
            "AtomicReplayArmResult",
        ),
    ),
)
def test_atomic_contract_rejects_untyped_arm_forgery(factory: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        factory(_request())  # type: ignore[operator]
