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
    BenchmarkExample,
    BenchmarkInterventionArm,
    DownstreamPrediction,
    PairedInterventionBenchmarkRunner,
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
)
from agentinputlab.observability import ResourceLimits, ResourceReceipt, ResourceUsage
from agentinputlab.runner import ComparisonArm, ComparisonProfile
from agentinputlab.selectors import NoOpIntervention


@dataclass(frozen=True)
class AppendQuery:
    method_id: str = "fixture.append"
    release_id: str = "fixture.append.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.APPEND_QUERY, "extra"),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="fixture",
            profile_kind="controlled_adapter",
        )


@dataclass(frozen=True)
class BrokenIntervention:
    method_id: str = "fixture.broken"
    release_id: str = "fixture.broken.v1"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del state, context
        raise AlgorithmOutputError("fixture intervention failure")


class FixturePredictor:
    def __init__(
        self,
        *,
        revision: str = "v1",
        failure_type: type[Exception] | None = None,
        usage: ResourceUsage | None = None,
        resource_receipt: ResourceReceipt | None = None,
    ) -> None:
        self.revision = revision
        self.failure_type = failure_type
        self.usage = usage or ResourceUsage(
            model_calls=1,
            input_tokens=2,
            output_tokens=1,
        )
        self.resource_receipt = resource_receipt
        self.states: list[FrozenAgentState] = []

    def fingerprint(self) -> dict[str, str]:
        return {"implementation": "fixture-predictor", "revision": self.revision}

    def predict(self, state: FrozenAgentState) -> DownstreamPrediction:
        self.states.append(state)
        if self.failure_type is not None:
            raise self.failure_type("fixture downstream failure")
        return DownstreamPrediction(
            prediction=state.query,
            usage=self.usage,
            metadata={"runtime": "fixture"},
            resource_receipts=(
                (self.resource_receipt,) if self.resource_receipt is not None else ()
            ),
        )


class LengthAdapter:
    benchmark_id = "fixture.length"
    pairing_mode = "stateless_score_twice"

    def __init__(self) -> None:
        self.examples_calls = 0
        self._examples = (
            BenchmarkExample(
                example_id="one",
                query="base",
                visible_metadata={"visible": "yes"},
                evaluator_gold={"secret": "evaluator-only"},
            ),
            BenchmarkExample(
                example_id="two",
                query="pair",
                evaluator_gold={"secret": "evaluator-only-two"},
            ),
        )

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        assert split == "test"
        self.examples_calls += 1
        return self._examples

    def score(self, example: BenchmarkExample, prediction: Any) -> dict[str, float]:
        assert example.evaluator_gold["secret"].startswith("evaluator-only")
        return {"length": float(len(str(prediction)))}

    @property
    def evaluation_provenance(self) -> dict[str, Any]:
        return {
            "official_runtime_verified": False,
            "scorer": {"identity": "fixture-length", "revision": "v1"},
        }


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


def _arm(
    arm_id: str,
    intervention: Any,
    predictor: FixturePredictor,
) -> BenchmarkInterventionArm:
    release_id = intervention.release_id
    return BenchmarkInterventionArm(
        role=arm_id,
        comparison_arm=ComparisonArm(
            arm_id=arm_id,
            release_id=release_id,
            bindings=(
                _binding("dataset", DependencyRole.NUISANCE, "fixture.length", "v1"),
                _binding(
                    "downstream_predictor",
                    DependencyRole.NUISANCE,
                    "fixture-predictor",
                    "v1",
                ),
                _binding("intervention", DependencyRole.TREATMENT, release_id, "v1"),
            ),
        ),
        intervention=intervention,
        context=InterventionContext(
            seed=11,
            run_config={
                "dependency_fingerprints": {"dataset": {"id": "fixture.length", "revision": "v1"}}
            },
        ),
        predictor=predictor,
    )


def _profile() -> ComparisonProfile:
    return ComparisonProfile(
        profile_id="fixture.paired.v1",
        kind=ProfileKind.CONTROLLED_ADAPTER,
        required_slots=("dataset", "downstream_predictor"),
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


def test_intervention_prediction_scoring_chain_is_paired_and_gold_isolated() -> None:
    receipt = _resource_receipt()
    baseline_predictor = FixturePredictor(resource_receipt=receipt)
    treatment_predictor = FixturePredictor()
    result = PairedInterventionBenchmarkRunner().run(
        LengthAdapter(),
        "test",
        _profile(),
        (
            _arm("baseline", NoOpIntervention(), baseline_predictor),
            _arm("treatment", AppendQuery(), treatment_predictor),
        ),
    )

    assert result.complete
    assert result.paired_record is not None
    assert result.paired_record.baseline_metrics == {"length": 4.0}
    assert result.paired_record.treatment_metrics == {"length": 11.0}
    assert result.paired_record.metric_deltas == {"length": 7.0}
    assert [state.query for state in baseline_predictor.states] == ["base", "pair"]
    assert [state.query for state in treatment_predictor.states] == [
        "base\n\nextra",
        "pair\n\nextra",
    ]
    for predictor in (baseline_predictor, treatment_predictor):
        assert all(
            "evaluator-only" not in repr(state.canonical_dict()) for state in predictor.states
        )
    for example in result.examples:
        assert example.complete
        assert [arm.role for arm in example.arms] == ["baseline", "treatment"]
        for arm in example.arms:
            assert arm.benchmark_record is not None
            assert arm.benchmark_record.evaluation_provenance == {
                "official_runtime_verified": False,
                "scorer": {"identity": "fixture-length", "revision": "v1"},
            }
            with pytest.raises(TypeError):
                arm.benchmark_record.evaluation_provenance["scorer"]["identity"] = "changed"
        assert example.arms[0].total_usage.model_calls == 1
        assert example.arms[1].total_usage.output_tokens == 1
        assert example.arms[0].downstream_metadata == {"runtime": "fixture"}
        assert example.arms[0].downstream_resource_receipts == (receipt,)
        assert example.arms[1].downstream_resource_receipts == ()
        assert example.arms[0].intervention_record is not None
        assert example.arms[1].intervention_record is not None
        assert (
            example.arms[0].intervention_record.input_state_hash
            == example.arms[1].intervention_record.input_state_hash
            == example.state_hash
        )


def test_downstream_prediction_rejects_untyped_resource_receipts() -> None:
    with pytest.raises(TypeError, match="ResourceReceipt"):
        DownstreamPrediction(
            prediction="invalid",
            resource_receipts=({"wall_seconds": 0.1},),  # type: ignore[arg-type]
        )


def test_explicit_roles_make_input_tuple_order_irrelevant() -> None:
    baseline_predictor = FixturePredictor()
    treatment_predictor = FixturePredictor()
    result = PairedInterventionBenchmarkRunner().run(
        LengthAdapter(),
        "test",
        _profile(),
        (
            _arm("treatment", AppendQuery(), treatment_predictor),
            _arm("baseline", NoOpIntervention(), baseline_predictor),
        ),
    )

    assert result.paired_record is not None
    assert result.paired_record.baseline_metrics == {"length": 4.0}
    assert result.paired_record.treatment_metrics == {"length": 11.0}
    assert [arm.role for arm in result.examples[0].arms] == ["baseline", "treatment"]


@pytest.mark.parametrize("failure_type", [ArmExecutionError, AlgorithmOutputError])
def test_typed_arm_failure_is_recorded_without_fabricating_paired_metrics(
    failure_type: type[Exception],
) -> None:
    result = PairedInterventionBenchmarkRunner().run(
        LengthAdapter(),
        "test",
        _profile(),
        (
            _arm("baseline", NoOpIntervention(), FixturePredictor()),
            _arm(
                "treatment",
                AppendQuery(),
                FixturePredictor(failure_type=failure_type),
            ),
        ),
    )

    assert not result.complete
    assert result.paired_record is None
    assert all(not example.complete for example in result.examples)
    for example in result.examples:
        failure = example.arms[1]
        assert failure.status == "failed"
        assert failure.failure is not None
        assert failure.failure.category == failure_type.__name__
        assert failure.intervention_record is not None
        assert failure.benchmark_record is None


@pytest.mark.parametrize(
    "failure_type",
    [TimeoutError, ConnectionError, RuntimeError, ExternalDependencyError],
)
def test_non_arm_failure_aborts_paired_cell(
    failure_type: type[Exception],
) -> None:
    with pytest.raises(failure_type, match="fixture downstream failure"):
        PairedInterventionBenchmarkRunner().run(
            LengthAdapter(),
            "test",
            _profile(),
            (
                _arm("baseline", NoOpIntervention(), FixturePredictor()),
                _arm(
                    "treatment",
                    AppendQuery(),
                    FixturePredictor(failure_type=failure_type),
                ),
            ),
        )


def test_downstream_usage_is_included_in_the_complete_arm_budget() -> None:
    with pytest.raises(BudgetExceededError, match="output_tokens"):
        PairedInterventionBenchmarkRunner().run(
            LengthAdapter(),
            "test",
            _profile(),
            (
                _arm(
                    "baseline",
                    NoOpIntervention(),
                    FixturePredictor(
                        usage=ResourceUsage(
                            model_calls=1,
                            input_tokens=2,
                            output_tokens=2,
                        )
                    ),
                ),
                _arm("treatment", AppendQuery(), FixturePredictor()),
            ),
            budget=ExecutionBudget(max_output_tokens=1),
        )


def test_intervention_failure_retains_monitor_trace_identity() -> None:
    result = PairedInterventionBenchmarkRunner().run(
        LengthAdapter(),
        "test",
        _profile(),
        (
            _arm("baseline", NoOpIntervention(), FixturePredictor()),
            _arm("treatment", BrokenIntervention(), FixturePredictor()),
        ),
    )

    for example in result.examples:
        failure = example.arms[1]
        assert failure.failure is not None
        assert failure.failure.category == "AlgorithmOutputError"
        assert failure.failure.monitor_run_id
        assert failure.intervention_record is None


def test_shared_predictor_and_fingerprint_drift_fail_before_dataset_or_prediction() -> None:
    adapter = LengthAdapter()
    shared = FixturePredictor()
    with pytest.raises(BalanceError, match="independently constructed predictors"):
        PairedInterventionBenchmarkRunner().run(
            adapter,
            "test",
            _profile(),
            (
                _arm("baseline", NoOpIntervention(), shared),
                _arm("treatment", AppendQuery(), shared),
            ),
        )
    assert adapter.examples_calls == 0
    assert shared.states == []

    first = FixturePredictor(revision="v1")
    second = FixturePredictor(revision="v2")
    with pytest.raises(BalanceError, match="fingerprint differs"):
        PairedInterventionBenchmarkRunner().run(
            adapter,
            "test",
            _profile(),
            (
                _arm("baseline", NoOpIntervention(), first),
                _arm("treatment", AppendQuery(), second),
            ),
        )
    assert adapter.examples_calls == 0
    assert first.states == []
    assert second.states == []


def test_stateful_external_pairing_mode_is_rejected_before_execution() -> None:
    adapter = LengthAdapter()
    adapter.pairing_mode = "unsupported_external_replay"
    with pytest.raises(ValueError, match="atomic external replay runner"):
        PairedInterventionBenchmarkRunner().run(
            adapter,
            "test",
            _profile(),
            (
                _arm("baseline", NoOpIntervention(), FixturePredictor()),
                _arm("treatment", AppendQuery(), FixturePredictor()),
            ),
        )
    assert adapter.examples_calls == 0
