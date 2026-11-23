from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

import agentinputlab.experiments.drivers.static_production as static_driver
from agentinputlab.benchmarks import BenchmarkExample, DownstreamPrediction
from agentinputlab.contracts import (
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    IntentContract,
    InterventionContext,
    PatchOp,
    PatchOperation,
    ProfileKind,
    SemanticFamily,
)
from agentinputlab.contracts.errors import (
    AlgorithmOutputError,
    ArtifactError,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.experiments import (
    ExperimentArmSpec,
    ExperimentBenchmarkTarget,
    ExperimentCell,
    ExperimentManifest,
    ExperimentStage,
)
from agentinputlab.experiments.drivers.static_production import (
    StaticArmBuildRequest,
    StaticBackendBuildRequest,
    StaticPredictorBuildRequest,
    StaticProductionExperimentDriver,
    StaticRuntimeComponents,
    static_runtime_product_sha256,
)
from agentinputlab.observability import (
    ModelPrice,
    PricingCatalog,
    ResourceLimits,
    ResourceReceipt,
    ResourceUsage,
)

BACKEND_IDS: list[int] = []
PREDICTOR_STATES: list[FrozenAgentState] = []


class FixtureBackend:
    revision = "v1"

    def fingerprint(self) -> dict[str, str]:
        return {
            "implementation": "fixture-backend",
            "model": "fixture-model",
            "revision": self.revision,
        }


@dataclass(frozen=True)
class FingerprintedDependency:
    identity: str

    def fingerprint(self) -> dict[str, str]:
        return {"identity": self.identity}


class BackendBuilder:
    builder_kind = "backend"

    def fingerprint(self) -> dict[str, str]:
        return {"builder": "fixture-backend-builder", "revision": "v1"}

    def preflight(self, parameters):
        assert parameters == {"deployment": "locked-fixture"}
        return ()

    def build(self, request: StaticBackendBuildRequest) -> StaticRuntimeComponents:
        assert request.builder_parameters == {"deployment": "locked-fixture"}
        backend = FixtureBackend()
        BACKEND_IDS.append(id(backend))
        return StaticRuntimeComponents(
            backend=backend,
            run_config={"generation": {"temperature": 0.0}},
        )


class FixturePredictor:
    fail = False
    revision = "v1"
    resource_receipt: ResourceReceipt | None = None

    def __init__(self, request: StaticPredictorBuildRequest) -> None:
        self.runtime = request.runtime
        self.benchmark_id = request.benchmark_id

    def fingerprint(self) -> dict[str, str]:
        return {
            "implementation": "fixture-predictor",
            "benchmark_id": self.benchmark_id,
            "revision": self.revision,
        }

    def predict(self, state: FrozenAgentState) -> DownstreamPrediction:
        PREDICTOR_STATES.append(state)
        if self.fail:
            raise AlgorithmOutputError("predictor failed with bearer super-secret-token")
        return DownstreamPrediction(
            prediction=state.query,
            usage=ResourceUsage(model_calls=1, input_tokens=3, output_tokens=1),
            resource_receipts=(
                (self.resource_receipt,) if self.resource_receipt is not None else ()
            ),
        )


class PredictorBuilder:
    builder_kind = "predictor"

    def fingerprint(self) -> dict[str, str]:
        return {"builder": "fixture-predictor-builder", "revision": "v1"}

    def preflight(self, parameters):
        assert parameters == {"prediction_contract": "query-text"}
        return ()

    def build(self, request: StaticPredictorBuildRequest) -> FixturePredictor:
        return FixturePredictor(request)


@dataclass(frozen=True)
class AppendIntervention:
    release_id: str
    method_id: str = "fixture.append"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.APPEND_QUERY, "expanded"),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="fixture",
            profile_kind=ProfileKind.CONTROLLED_ADAPTER.value,
        )


class InterventionBuilder:
    builder_kind = "intervention"

    def fingerprint(self) -> dict[str, str]:
        return {"builder": "fixture-intervention-builder", "revision": "v1"}

    def preflight(self, parameters):
        assert parameters == {"source_lock": "fixture"}
        return ()

    def build(self, request: StaticArmBuildRequest):
        if request.family is SemanticFamily.NO_OP:
            from agentinputlab.selectors import NoOpIntervention

            return NoOpIntervention(release_id=request.release_id)
        return AppendIntervention(release_id=request.release_id)


def backend_builder_factory() -> BackendBuilder:
    return BackendBuilder()


def predictor_builder_factory() -> PredictorBuilder:
    return PredictorBuilder()


def intervention_builder_factory() -> InterventionBuilder:
    return InterventionBuilder()


class FakePrepared:
    asset_id = "fixture.asset"
    source_revision = "fixture-source-v1"
    source_sha256 = "1" * 64
    processor = "fixture_processor"
    processor_revision = "1"
    counts: ClassVar[dict[str, int]] = {"examples": 1}
    file_sha256: ClassVar[dict[str, str]] = {
        "inputs.jsonl": "2" * 64,
        "gold.jsonl": "3" * 64,
    }

    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id


class FakeStaticAdapter:
    pairing_mode = "stateless_score_twice"

    def __init__(self, benchmark_id: str = "beir.arguana", split: str = "test") -> None:
        self.benchmark_id = benchmark_id
        self.split = split
        self.prepared = FakePrepared(benchmark_id)
        self._example = BenchmarkExample(
            example_id="unit-a",
            query="base query",
            visible_metadata={"safe": "visible"},
            evaluator_gold={"secret_answer": "must-not-cross"},
        )
        self.examples_calls = 0
        self.example_calls = 0

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        assert split == self.split
        self.examples_calls += 1
        return (self._example,)

    def example(self, split: str, example_id: str) -> BenchmarkExample:
        assert split == self.split
        assert example_id == self._example.example_id
        self.example_calls += 1
        return self._example

    def score(self, example: BenchmarkExample, prediction: Any) -> dict[str, float]:
        assert example is self._example
        assert example.evaluator_gold == {"secret_answer": "must-not-cross"}
        return {"length": float(len(str(prediction)))}

    @property
    def evaluation_provenance(self) -> dict[str, Any]:
        return {
            "official_runtime_verified": False,
            "scorer": {"identity": "fixture-static", "revision": "v1"},
        }


def _builder_spec(
    factory_name: str,
    fingerprint: dict[str, str],
    parameters: dict,
    *,
    expected_product_sha256: str | None = None,
) -> dict:
    spec = {
        "entrypoint": f"{__name__}:{factory_name}",
        "fingerprint_sha256": content_hash(fingerprint),
        "parameters": parameters,
    }
    if expected_product_sha256 is not None:
        spec["expected_product_sha256"] = expected_product_sha256
    return spec


def _config(tmp_path: Path) -> dict:
    runtime_product_sha256 = static_runtime_product_sha256(
        StaticRuntimeComponents(
            backend=FixtureBackend(),
            run_config={"generation": {"temperature": 0.0}},
        )
    )
    backend = _builder_spec(
        "backend_builder_factory",
        BackendBuilder().fingerprint(),
        {"deployment": "locked-fixture"},
        expected_product_sha256=runtime_product_sha256,
    )
    intervention = _builder_spec(
        "intervention_builder_factory",
        InterventionBuilder().fingerprint(),
        {"source_lock": "fixture"},
    )
    cells = {
        "beir.arguana": "test",
        "bright.earth_science": "test",
        "bright.economics": "test",
        "musique.answerable": "dev",
        "2wiki.multihop": "dev",
        "hotpotqa.distractor": "validation",
        "qasper.v0_3_dev": "dev",
    }
    return {
        "schema_version": 2,
        "driver_id": "fixture.static-production",
        "driver_revision_sha256": "d" * 64,
        "prepared_data_root": str(tmp_path.resolve()),
        "baseline_arm_id": "no_op",
        "execution_budget": {
            "max_model_calls": 4,
            "max_retrieval_calls": 4,
            "max_tool_calls": 4,
            "max_output_tokens": 256,
        },
        "model_slots": {"fixture_model": backend},
        "benchmark_cells": {
            cell_id: {
                "split": split,
                "predictor_builder": _builder_spec(
                    "predictor_builder_factory",
                    PredictorBuilder().fingerprint(),
                    {"prediction_contract": "query-text"},
                    expected_product_sha256=content_hash(
                        {
                            "implementation": "fixture-predictor",
                            "benchmark_id": cell_id,
                            "revision": "v1",
                        }
                    ),
                ),
            }
            for cell_id, split in cells.items()
        },
        "arms": {
            "no_op": {
                "release_id": "baseline.no_op",
                "intervention_builder": intervention,
            },
            "append": {
                "release_id": "fixture.append.v1",
                "intervention_builder": intervention,
            },
        },
    }


def _write_config(tmp_path: Path, config: dict) -> tuple[Path, str]:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "static-driver.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _manifest(
    benchmark_id: str = "beir.arguana",
    split: str = "test",
) -> ExperimentManifest:
    return ExperimentManifest(
        schema_version=1,
        study_id="fixture.static-study",
        stage=ExperimentStage.DISCOVERY,
        estimand_profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id=benchmark_id,
                split=split,
                unit_ids=("unit-a",),
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
                arm_id="append",
                release_id="fixture.append.v1",
                family=SemanticFamily.REFORMULATE,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            ),
        ),
        seeds=(13,),
        model_slots=("fixture_model",),
    )


def _cell(manifest: ExperimentManifest) -> ExperimentCell:
    target = manifest.benchmarks[0]
    return ExperimentCell(
        study_id=manifest.study_id,
        benchmark_id=target.cell_id,
        split=target.split,
        unit_id=target.unit_ids[0],
        model_slot=manifest.model_slots[0],
        seed=manifest.seeds[0],
        arm_ids=tuple(arm.arm_id for arm in manifest.arms),
        manifest_hash=manifest.manifest_hash,
    )


def _resource_receipt() -> ResourceReceipt:
    limits = ResourceLimits(wall_timeout_seconds=1)
    return ResourceReceipt(
        limit_hash=limits.limit_hash,
        wall_seconds=0.1,
        cpu_seconds=0.05,
        supervisor_identity={"kind": "fixture"},
        image_identity={"kind": "fixture"},
    )


@pytest.fixture(autouse=True)
def _clear_observations() -> None:
    BACKEND_IDS.clear()
    PREDICTOR_STATES.clear()
    FixtureBackend.revision = "v1"
    FixturePredictor.fail = False
    FixturePredictor.revision = "v1"
    FixturePredictor.resource_receipt = None


def test_driver_preflight_and_exact_multi_arm_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeStaticAdapter()
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: adapter,
    )
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(
        path,
        expected_sha256=checksum,
    )
    manifest = _manifest()

    assert driver.preflight(manifest) == ()
    assert adapter.examples_calls == 1
    cell = ExperimentCell(
        study_id=manifest.study_id,
        benchmark_id="beir.arguana",
        split="test",
        unit_id="unit-a",
        model_slot="fixture_model",
        seed=13,
        arm_ids=("no_op", "append"),
        manifest_hash=manifest.manifest_hash,
    )
    receipt = _resource_receipt()
    FixturePredictor.resource_receipt = receipt
    result = driver.execute_cell(manifest, cell)

    assert result.status == "succeeded"
    assert [item["arm_id"] for item in result.payload["arms"]] == ["no_op", "append"]
    assert [item["status"] for item in result.payload["arms"]] == [
        "succeeded",
        "succeeded",
    ]
    assert result.payload["arms"][0]["metrics"] == {"length": 10.0}
    assert result.payload["arms"][1]["metrics"] == {"length": 20.0}
    assert all(
        arm["evaluation_provenance"]
        == {
            "official_runtime_verified": False,
            "scorer": {"identity": "fixture-static", "revision": "v1"},
        }
        for arm in result.payload["arms"]
    )
    assert result.payload["metric_deltas"] == {"append": {"length": 10.0}}
    assert result.resource_receipts == (receipt, receipt)
    assert all(
        arm["downstream_resource_receipt_hashes"] == (content_hash(receipt),)
        for arm in result.payload["arms"]
    )
    assert result.payload["driver_config_sha256"] == checksum
    assert driver.fingerprint()["config_sha256"] == checksum
    assert set(result.payload["pre_action"]) == {
        "feature_schema_id",
        "feature_schema_hash",
        "state_hash",
        "feature_hash",
        "provenance_hash",
        "defect_scores",
    }
    assert result.payload["pre_action"]["state_hash"] == result.payload["state_hash"]
    assert len(result.payload["pre_action"]["feature_hash"]) == 64
    assert set(result.payload["pre_action"]["defect_scores"]) == {
        "reformulate",
        "decompose",
        "acquire",
        "clarify",
        "curate",
    }
    assert len(BACKEND_IDS) == 2
    assert len(set(BACKEND_IDS)) == 2
    assert len(PREDICTOR_STATES) == 2
    assert all("must-not-cross" not in repr(state.canonical_dict()) for state in PREDICTOR_STATES)
    assert adapter.examples_calls == 1
    assert adapter.example_calls == 1


@pytest.mark.parametrize(
    ("benchmark_id", "split"),
    (
        ("beir.arguana", "test"),
        ("bright.earth_science", "test"),
        ("bright.economics", "test"),
        ("musique.answerable", "dev"),
        ("2wiki.multihop", "dev"),
        ("hotpotqa.distractor", "validation"),
        ("qasper.v0_3_dev", "dev"),
    ),
)
def test_each_static_production_cell_has_a_preflighted_adapter_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_id: str,
    split: str,
) -> None:
    adapter = FakeStaticAdapter(benchmark_id, split)
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: adapter,
    )
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)

    assert driver.preflight(_manifest(benchmark_id, split)) == ()
    assert adapter.examples_calls == 1


def test_preflight_units_proves_exact_inventory_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeStaticAdapter()
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: adapter,
    )
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()

    assert driver.preflight(manifest) == ()
    inventory_key = "beir.arguana:test"
    assert driver.preflight_units(manifest, {inventory_key: ("unit-a",)}) == ()
    blockers = driver.preflight_units(manifest, {inventory_key: ("missing-unit",)})
    assert blockers
    assert "missing-unit" in blockers[0]


def test_driver_config_and_builder_fingerprints_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        StaticProductionExperimentDriver.from_file(
            path,
            expected_sha256="0" * 64,
        )

    config["model_slots"]["fixture_model"]["fingerprint_sha256"] = "9" * 64
    path, checksum = _write_config(tmp_path, config)
    driver = StaticProductionExperimentDriver.from_file(
        path,
        expected_sha256=checksum,
    )
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: FakeStaticAdapter(),
    )
    blockers = driver.preflight(_manifest())
    assert any("builder fingerprint checksum mismatch" in blocker for blocker in blockers)


def test_runtime_product_fingerprint_drift_aborts_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeStaticAdapter()
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: adapter,
    )
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    FixtureBackend.revision = "drifted-after-preflight"

    with pytest.raises(ArtifactError, match="runtime product checksum mismatch"):
        driver.execute_cell(manifest, _cell(manifest))


def test_predictor_product_fingerprint_drift_aborts_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeStaticAdapter()
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: adapter,
    )
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    FixturePredictor.revision = "drifted-after-preflight"

    with pytest.raises(ArtifactError, match="predictor product checksum mismatch"):
        driver.execute_cell(manifest, _cell(manifest))


@pytest.mark.parametrize(
    "changed",
    (
        {"backend": FingerprintedDependency("another-backend")},
        {"dense_embedder": FingerprintedDependency("embedder")},
        {"retriever": FingerprintedDependency("retriever")},
        {"vector_searcher": FingerprintedDependency("vector-searcher")},
        {"tools": FingerprintedDependency("tools")},
        {"external_runtime": FingerprintedDependency("external-runtime")},
        {"artifact_manager": FingerprintedDependency("artifact-manager")},
        {"run_config": {"generation": {"temperature": 0.25}}},
        {"intent_contract": IntentContract(required_entities=("entity",))},
        {"pricing_catalog": PricingCatalog(prices={"fixture-model": ModelPrice(0.1, 0.2)})},
        {"strict_intent": False},
    ),
)
def test_runtime_product_hash_covers_every_runtime_input(changed: dict[str, Any]) -> None:
    base = StaticRuntimeComponents(backend=FixtureBackend())
    inputs: dict[str, Any] = {"backend": FixtureBackend()}
    inputs.update(changed)
    variant = StaticRuntimeComponents(**inputs)

    assert static_runtime_product_sha256(variant) != static_runtime_product_sha256(base)


def test_runtime_product_rejects_unfingerprinted_non_null_components() -> None:
    runtime = StaticRuntimeComponents(backend=FixtureBackend(), tools=object())

    with pytest.raises(ValidationError, match="runtime tools must expose fingerprint"):
        static_runtime_product_sha256(runtime)


def test_runtime_and_predictor_product_locks_are_required(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config["model_slots"]["fixture_model"]["expected_product_sha256"]
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match=r"model_slots.*expected_product_sha256"):
        StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)

    config = _config(tmp_path)
    del config["benchmark_cells"]["beir.arguana"]["predictor_builder"]["expected_product_sha256"]
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(
        ValidationError,
        match=r"benchmark_cells.*predictor_builder requires expected_product_sha256",
    ):
        StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)


def test_algorithm_failures_keep_exact_arms_and_redact_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeStaticAdapter()
    monkeypatch.setattr(static_driver, "PreparedQABenchmarkAdapter", FakeStaticAdapter)
    monkeypatch.setattr(
        static_driver,
        "load_production_benchmark_adapter",
        lambda cell_id, prepared_root: adapter,
    )
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(path, expected_sha256=checksum)
    manifest = _manifest()
    assert driver.preflight(manifest) == ()
    FixturePredictor.fail = True
    cell = ExperimentCell(
        study_id=manifest.study_id,
        benchmark_id="beir.arguana",
        split="test",
        unit_id="unit-a",
        model_slot="fixture_model",
        seed=13,
        arm_ids=("no_op", "append"),
        manifest_hash=manifest.manifest_hash,
    )

    result = driver.execute_cell(manifest, cell)

    assert [arm["arm_id"] for arm in result.payload["arms"]] == ["no_op", "append"]
    assert all(arm["status"] == "failed" for arm in result.payload["arms"])
    assert all(arm["failure"]["scope"] == "algorithm" for arm in result.payload["arms"])
    serialized = json.dumps(result.to_dict())
    assert "super-secret-token" not in serialized
    assert "<redacted>" in serialized


def test_config_requires_all_seven_static_production_cells(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config["benchmark_cells"]["qasper.v0_3_dev"]
    path, checksum = _write_config(tmp_path, config)
    with pytest.raises(ValidationError, match="exactly the seven"):
        StaticProductionExperimentDriver.from_file(
            path,
            expected_sha256=checksum,
        )


def test_execute_requires_successful_preflight(tmp_path: Path) -> None:
    path, checksum = _write_config(tmp_path, _config(tmp_path))
    driver = StaticProductionExperimentDriver.from_file(
        path,
        expected_sha256=checksum,
    )
    manifest = _manifest()
    cell = ExperimentCell(
        study_id=manifest.study_id,
        benchmark_id="beir.arguana",
        split="test",
        unit_id="unit-a",
        model_slot="fixture_model",
        seed=13,
        arm_ids=("no_op", "append"),
        manifest_hash=manifest.manifest_hash,
    )
    with pytest.raises(ArtifactError, match="successful preflight"):
        driver.execute_cell(manifest, cell)
