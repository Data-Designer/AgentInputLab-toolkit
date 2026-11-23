from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.benchmarks import (
    BenchmarkBlockArm,
    BenchmarkExample,
    ControlledMultiArmBlockRunner,
    DownstreamPrediction,
)
from agentinputlab.contracts import (
    DependencyBinding,
    DependencyRole,
    ExecutionBudget,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    ProfileKind,
    SemanticFamily,
)
from agentinputlab.contracts.errors import (
    AlgorithmOutputError,
    ArmExecutionError,
    BalanceError,
    BudgetExceededError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.observability import ResourceLimits, ResourceReceipt, ResourceUsage
from agentinputlab.runner import ComparisonArm, ComparisonProfile
from agentinputlab.selectors import NoOpIntervention


@dataclass(frozen=True)
class AppendQuery:
    suffix: str
    release_id: str
    method_id: str = "fixture.append"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.APPEND_QUERY, self.suffix),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="fixture",
            profile_kind=ProfileKind.CONTROLLED_ADAPTER.value,
        )


@dataclass(frozen=True)
class FalseNoOp:
    method_id: str = "fixture.false_no_op"
    release_id: str = "fixture.false_no_op.v1"
    family: SemanticFamily = SemanticFamily.NO_OP

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.APPEND_QUERY, "not-a-no-op"),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="fixture",
            profile_kind=ProfileKind.CONTROLLED_ADAPTER.value,
        )


class Predictor:
    def __init__(
        self,
        *,
        failure_type: type[Exception] | None = None,
        usage: ResourceUsage | None = None,
        resource_receipt: ResourceReceipt | None = None,
    ) -> None:
        self.failure_type = failure_type
        self.usage = usage or ResourceUsage(
            model_calls=1,
            input_tokens=2,
            output_tokens=1,
        )
        self.resource_receipt = resource_receipt
        self.states: list[FrozenAgentState] = []

    def fingerprint(self) -> dict[str, str]:
        return {"implementation": "fixture-predictor", "revision": "v1"}

    def predict(self, state: FrozenAgentState) -> DownstreamPrediction:
        self.states.append(state)
        if self.failure_type is not None:
            raise self.failure_type("fixture downstream failure")
        return DownstreamPrediction(
            prediction=state.query,
            usage=self.usage,
            resource_receipts=(
                (self.resource_receipt,) if self.resource_receipt is not None else ()
            ),
        )


class OneExampleAdapter:
    benchmark_id = "fixture.block"
    pairing_mode = "stateless_score_twice"

    def __init__(self) -> None:
        self.examples_calls = 0
        self.score_calls = 0
        self.example = BenchmarkExample(
            example_id="unit-one",
            query="base",
            visible_metadata={"public": True},
            evaluator_gold={"secret": "never-visible"},
        )

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        assert split == "test"
        self.examples_calls += 1
        return (self.example,)

    def score(self, example: BenchmarkExample, prediction: Any) -> dict[str, float]:
        assert example is self.example
        assert example.evaluator_gold == {"secret": "never-visible"}
        self.score_calls += 1
        return {"length": float(len(str(prediction)))}


def _binding(
    slot: str,
    role: DependencyRole,
    implementation_id: str,
    revision: str,
) -> DependencyBinding:
    return DependencyBinding(
        slot=slot,
        role=role,
        implementation_id=implementation_id,
        revision=revision,
    )


def _block_arm(arm_id: str, intervention: Any, predictor: Predictor) -> BenchmarkBlockArm:
    return BenchmarkBlockArm(
        comparison_arm=ComparisonArm(
            arm_id=arm_id,
            release_id=intervention.release_id,
            bindings=(
                _binding("dataset", DependencyRole.NUISANCE, "fixture.block", "v1"),
                _binding("model", DependencyRole.NUISANCE, "fixture-model", "v1"),
                _binding(
                    "downstream_predictor",
                    DependencyRole.NUISANCE,
                    "fixture-predictor",
                    "v1",
                ),
                _binding(
                    "intervention",
                    DependencyRole.TREATMENT,
                    intervention.method_id,
                    intervention.release_id,
                ),
            ),
        ),
        intervention=intervention,
        context=InterventionContext(
            seed=17,
            run_config={
                "dependency_fingerprints": {
                    "dataset": {"id": "fixture.block", "revision": "v1"},
                    "model": {"id": "fixture-model", "revision": "v1"},
                }
            },
        ),
        predictor=predictor,
    )


def _profile() -> ComparisonProfile:
    return ComparisonProfile(
        profile_id="fixture.multi_arm.v1",
        kind=ProfileKind.CONTROLLED_ADAPTER,
        required_slots=("dataset", "model", "downstream_predictor"),
        varied_slots=("intervention",),
    )


def _resource_receipt() -> ResourceReceipt:
    limits = ResourceLimits(wall_timeout_seconds=1)
    return ResourceReceipt(
        limit_hash=limits.limit_hash,
        wall_seconds=0.1,
        supervisor_identity={"kind": "fixture"},
        image_identity={"kind": "fixture"},
    )


def test_run_one_executes_one_shared_three_arm_block() -> None:
    adapter = OneExampleAdapter()
    receipt = _resource_receipt()
    predictors = (
        Predictor(resource_receipt=receipt),
        Predictor(),
        Predictor(),
    )
    observations: list[str] = []
    result = ControlledMultiArmBlockRunner().run_one(
        adapter,
        "test",
        "unit-one",
        _profile(),
        (
            _block_arm("no_op", NoOpIntervention(), predictors[0]),
            _block_arm(
                "append_x",
                AppendQuery("x", "fixture.append.x.v1"),
                predictors[1],
            ),
            _block_arm(
                "append_long",
                AppendQuery("long", "fixture.append.long.v1"),
                predictors[2],
            ),
        ),
        planned_arm_ids=("no_op", "append_x", "append_long"),
        baseline_arm_id="no_op",
        pre_action_observer=lambda state: (
            observations.append(state.state_hash)
            or {
                "feature_schema_id": "fixture.v1",
                "feature_schema_hash": "a" * 64,
                "provenance_hash": "b" * 64,
                "defect_scores": {
                    "reformulate": 0.1,
                    "decompose": 0.2,
                    "acquire": 0.3,
                    "clarify": 0.4,
                    "curate": 0.5,
                },
            }
        ),
    )

    assert adapter.examples_calls == 1
    assert adapter.score_calls == 3
    assert observations == [result.state_hash]
    assert result.planned_arm_ids == ("no_op", "append_x", "append_long")
    assert [arm.arm_id for arm in result.arms] == list(result.planned_arm_ids)
    assert result.all_arms_succeeded
    assert result.arms[0].downstream_resource_receipts == (receipt,)
    assert result.arms[1].downstream_resource_receipts == ()
    assert [predictor.states[0].query for predictor in predictors] == [
        "base",
        "base\n\nx",
        "base\n\nlong",
    ]
    assert result.metric_deltas == {
        "append_x": {"length": 3.0},
        "append_long": {"length": 6.0},
    }
    assert set(result.pre_action["defect_scores"]) == {
        "reformulate",
        "decompose",
        "acquire",
        "clarify",
        "curate",
    }


def test_exact_arm_set_is_checked_before_loading_the_example() -> None:
    adapter = OneExampleAdapter()
    arms = (
        _block_arm("no_op", NoOpIntervention(), Predictor()),
        _block_arm("append", AppendQuery("x", "fixture.append.v1"), Predictor()),
    )
    with pytest.raises(ValidationError, match="exactly match"):
        ControlledMultiArmBlockRunner().run_one(
            adapter,
            "test",
            "unit-one",
            _profile(),
            arms,
            planned_arm_ids=("append", "no_op"),
            baseline_arm_id="no_op",
        )
    assert adapter.examples_calls == 0


def test_downstream_usage_is_included_in_the_complete_block_arm_budget() -> None:
    adapter = OneExampleAdapter()
    with pytest.raises(BudgetExceededError, match="model_calls"):
        ControlledMultiArmBlockRunner().run_one(
            adapter,
            "test",
            "unit-one",
            _profile(),
            (
                _block_arm(
                    "no_op",
                    NoOpIntervention(),
                    Predictor(usage=ResourceUsage(model_calls=2)),
                ),
                _block_arm(
                    "append",
                    AppendQuery("x", "fixture.append.v1"),
                    Predictor(),
                ),
            ),
            planned_arm_ids=("no_op", "append"),
            baseline_arm_id="no_op",
            budget=ExecutionBudget(max_model_calls=1),
        )
    assert adapter.score_calls == 0


def test_shared_predictor_and_multiple_no_op_fail_before_execution() -> None:
    adapter = OneExampleAdapter()
    shared = Predictor()
    with pytest.raises(BalanceError, match="independently constructed predictors"):
        ControlledMultiArmBlockRunner().run_one(
            adapter,
            "test",
            "unit-one",
            _profile(),
            (
                _block_arm("no_op", NoOpIntervention(), shared),
                _block_arm("append", AppendQuery("x", "fixture.append.v1"), shared),
            ),
            planned_arm_ids=("no_op", "append"),
            baseline_arm_id="no_op",
        )
    assert adapter.examples_calls == 0

    with pytest.raises(ValidationError, match="exactly one declared NO_OP"):
        ControlledMultiArmBlockRunner().run_one(
            adapter,
            "test",
            "unit-one",
            _profile(),
            (
                _block_arm("no_op", NoOpIntervention(), Predictor()),
                _block_arm(
                    "another_no_op",
                    NoOpIntervention(release_id="baseline.no_op.other"),
                    Predictor(),
                ),
            ),
            planned_arm_ids=("no_op", "another_no_op"),
            baseline_arm_id="no_op",
        )
    assert adapter.examples_calls == 0


def test_declared_no_op_must_materialize_only_no_op_operations() -> None:
    adapter = OneExampleAdapter()
    with pytest.raises(BalanceError, match="did not produce NO_OP"):
        ControlledMultiArmBlockRunner().run_one(
            adapter,
            "test",
            "unit-one",
            _profile(),
            (
                _block_arm("no_op", FalseNoOp(), Predictor()),
                _block_arm("append", AppendQuery("x", "fixture.append.v1"), Predictor()),
            ),
            planned_arm_ids=("no_op", "append"),
            baseline_arm_id="no_op",
        )
    assert adapter.score_calls == 0


@pytest.mark.parametrize("failure_type", [ArmExecutionError, AlgorithmOutputError])
def test_typed_arm_failure_is_recorded_in_multi_arm_block(
    failure_type: type[Exception],
) -> None:
    result = ControlledMultiArmBlockRunner().run_one(
        OneExampleAdapter(),
        "test",
        "unit-one",
        _profile(),
        (
            _block_arm("no_op", NoOpIntervention(), Predictor()),
            _block_arm(
                "append",
                AppendQuery("x", "fixture.append.v1"),
                Predictor(failure_type=failure_type),
            ),
        ),
        planned_arm_ids=("no_op", "append"),
        baseline_arm_id="no_op",
    )

    failure = result.arms[1]
    assert failure.status == "failed"
    assert failure.failure is not None
    assert failure.failure.category == failure_type.__name__
    assert failure.intervention_record is not None
    assert failure.benchmark_record is None


@pytest.mark.parametrize(
    "failure_type",
    [TimeoutError, ConnectionError, RuntimeError, ExternalDependencyError],
)
def test_non_arm_failure_aborts_multi_arm_cell(
    failure_type: type[Exception],
) -> None:
    with pytest.raises(failure_type, match="fixture downstream failure"):
        ControlledMultiArmBlockRunner().run_one(
            OneExampleAdapter(),
            "test",
            "unit-one",
            _profile(),
            (
                _block_arm("no_op", NoOpIntervention(), Predictor()),
                _block_arm(
                    "append",
                    AppendQuery("x", "fixture.append.v1"),
                    Predictor(failure_type=failure_type),
                ),
            ),
            planned_arm_ids=("no_op", "append"),
            baseline_arm_id="no_op",
        )
