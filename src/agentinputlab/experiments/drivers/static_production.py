"""Checksum-locked driver for the seven static production benchmark adapters.

This module is deliberately an assembly layer, not a model registry.  Every
backend, downstream predictor, and intervention comes from an operator-owned
zero-argument builder named in one checksum-locked JSON file.  The driver never
downloads an asset and never guesses a default deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.benchmarks import (
    BenchmarkAdapter,
    BenchmarkBlockArm,
    BenchmarkExample,
    ControlledMultiArmBlockRunner,
    DownstreamPredictor,
    PreparedQABenchmarkAdapter,
    PreparedRetrievalBenchmarkAdapter,
    load_production_benchmark_adapter,
    production_benchmark_specs,
)
from agentinputlab.contracts.dependency import DependencyBinding, DependencyRole
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.intent import IntentContract
from agentinputlab.contracts.lifecycle import Intervention, InterventionContext
from agentinputlab.contracts.serialization import (
    content_hash,
    freeze_mapping,
    to_jsonable,
)
from agentinputlab.contracts.state import ContextBlock, ExecutionBudget, FrozenAgentState
from agentinputlab.experiments.executor import ExperimentCellResult
from agentinputlab.experiments.loader import resolve_driver_entrypoint
from agentinputlab.experiments.plan import ExperimentCell
from agentinputlab.experiments.schema import (
    ExperimentArmSpec,
    ExperimentBenchmarkTarget,
    ExperimentManifest,
)
from agentinputlab.observability import PricingCatalog, ResourceReceipt, RunMonitor
from agentinputlab.runner import ComparisonArm, ComparisonProfile
from agentinputlab.selectors import VisibleStateFeatureExtractor

CONFIG_PATH_ENV = "AILAB_STATIC_PRODUCTION_DRIVER_CONFIG"
CONFIG_SHA256_ENV = "AILAB_STATIC_PRODUCTION_DRIVER_CONFIG_SHA256"

_CONFIG_SCHEMA_VERSION = 2
_DRIVER_IMPLEMENTATION_ID = "agentinputlab.static_production.multi_arm"
_PROFILE_ID = "static_production.multi_arm.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_BUILDER_KINDS = frozenset({"backend", "predictor", "intervention"})
# Schema v2 is a frozen seven-cell campaign contract. Adding a standalone
# source-locked loader must not expand an existing campaign or dispatch new data.
_STATIC_V2_CELLS = frozenset(
    {
        "beir.arguana",
        "bright.earth_science",
        "bright.economics",
        "musique.answerable",
        "2wiki.multihop",
        "hotpotqa.distractor",
        "qasper.v0_3_dev",
    }
)
_STATIC_PRODUCTION_SPECS = tuple(
    spec for spec in production_benchmark_specs() if spec.cell_id in _STATIC_V2_CELLS
)
_STATIC_SPLITS = MappingProxyType({spec.cell_id: spec.split for spec in _STATIC_PRODUCTION_SPECS})
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(
        r"(?i)(\b(?:[a-z0-9]+[_-])*(?:api[_-]?key|x[_-]?api[_-]?key|authorization)"
        r"\b\s*[:=]\s*)['\"]?(?:bearer\s+)?[^\s,;'\"]{4,}"
    ),
)


@runtime_checkable
class LockedComponentBuilder(Protocol):
    """Operator builder contract resolved only during explicit preflight/run."""

    builder_kind: str

    def fingerprint(self) -> Mapping[str, Any]: ...

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]: ...

    def build(self, request: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class LockedBuilderSpec:
    """One builder entrypoint plus its construction and optional product locks."""

    entrypoint: str
    fingerprint_sha256: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    expected_product_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entrypoint, str) or _ENTRYPOINT.fullmatch(self.entrypoint) is None:
            raise ValidationError("builder entrypoint must use module:factory syntax")
        _require_sha256(self.fingerprint_sha256, label="builder fingerprint_sha256")
        if self.expected_product_sha256 is not None:
            _require_sha256(
                self.expected_product_sha256,
                label="builder expected_product_sha256",
            )
        try:
            object.__setattr__(self, "parameters", freeze_mapping(self.parameters))
        except (TypeError, ValueError) as exc:
            raise ValidationError("builder parameters must be canonical JSON") from exc

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> LockedBuilderSpec:
        data = _strict_mapping(
            value,
            path=path,
            keys={
                "entrypoint",
                "fingerprint_sha256",
                "expected_product_sha256",
                "parameters",
            },
            required={"entrypoint", "fingerprint_sha256"},
        )
        parameters = data.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValidationError(f"{path}.parameters must be an object")
        return cls(
            entrypoint=_text(data["entrypoint"], f"{path}.entrypoint"),
            fingerprint_sha256=_text(
                data["fingerprint_sha256"],
                f"{path}.fingerprint_sha256",
            ),
            expected_product_sha256=(
                _text(
                    data["expected_product_sha256"],
                    f"{path}.expected_product_sha256",
                )
                if "expected_product_sha256" in data
                else None
            ),
            parameters=parameters,
        )

    @property
    def binding_identity(self) -> Mapping[str, Any]:
        """Public identity without copying raw operator parameters into results."""

        return freeze_mapping(
            {
                "entrypoint": self.entrypoint,
                "fingerprint_sha256": self.fingerprint_sha256,
                "expected_product_sha256": self.expected_product_sha256,
                "parameters_sha256": content_hash(self.parameters),
            }
        )


@dataclass(frozen=True, slots=True)
class StaticRuntimeComponents:
    """Fresh runtime graph returned by a backend builder for one arm."""

    backend: Any
    dense_embedder: Any = None
    retriever: Any = None
    vector_searcher: Any = None
    tools: Any = None
    external_runtime: Any = None
    artifact_manager: Any = None
    run_config: Mapping[str, Any] = field(default_factory=dict)
    intent_contract: IntentContract | None = None
    pricing_catalog: PricingCatalog | None = None
    strict_intent: bool = True
    score_only: bool = False

    def __post_init__(self) -> None:
        if type(self.score_only) is not bool:
            raise ValidationError("static production score_only must be boolean")
        if self.backend is None and self.retriever is None and not self.score_only:
            raise ValidationError("static production runtime requires a backend or retriever")
        fingerprint = getattr(self.backend, "fingerprint", None)
        if self.backend is not None and not callable(fingerprint):
            raise ValidationError("static production backend must expose fingerprint()")
        if not isinstance(self.strict_intent, bool):
            raise ValidationError("static production strict_intent must be boolean")
        try:
            object.__setattr__(self, "run_config", freeze_mapping(self.run_config))
        except (TypeError, ValueError) as exc:
            raise ValidationError("static production run_config must be canonical JSON") from exc
        if self.intent_contract is not None and not isinstance(
            self.intent_contract,
            IntentContract,
        ):
            raise TypeError("static production intent_contract must be IntentContract")
        if self.pricing_catalog is not None and not isinstance(
            self.pricing_catalog,
            PricingCatalog,
        ):
            raise TypeError("static production pricing_catalog must be PricingCatalog")


@dataclass(frozen=True, slots=True)
class StaticBackendBuildRequest:
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty(self.model_slot, label="model_slot")
        _require_seed(self.seed)
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )


@dataclass(frozen=True, slots=True)
class StaticPredictorBuildRequest:
    benchmark_id: str
    split: str
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]
    runtime: StaticRuntimeComponents

    def __post_init__(self) -> None:
        for label, value in (
            ("benchmark_id", self.benchmark_id),
            ("split", self.split),
            ("model_slot", self.model_slot),
        ):
            _require_nonempty(value, label=label)
        _require_seed(self.seed)
        if not isinstance(self.runtime, StaticRuntimeComponents):
            raise TypeError("predictor request runtime must be StaticRuntimeComponents")
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )


@dataclass(frozen=True, slots=True)
class StaticArmBuildRequest:
    arm_id: str
    release_id: str
    family: SemanticFamily
    profile_kind: ProfileKind
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]
    arm_parameters: Mapping[str, Any]
    runtime: StaticRuntimeComponents

    def __post_init__(self) -> None:
        for label, value in (
            ("arm_id", self.arm_id),
            ("release_id", self.release_id),
            ("model_slot", self.model_slot),
        ):
            _require_nonempty(value, label=label)
        if not isinstance(self.family, SemanticFamily):
            raise TypeError("arm request family must be SemanticFamily")
        if not isinstance(self.profile_kind, ProfileKind):
            raise TypeError("arm request profile_kind must be ProfileKind")
        _require_seed(self.seed)
        if not isinstance(self.runtime, StaticRuntimeComponents):
            raise TypeError("arm request runtime must be StaticRuntimeComponents")
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )
        object.__setattr__(self, "arm_parameters", freeze_mapping(self.arm_parameters))


@dataclass(frozen=True, slots=True)
class _BenchmarkBinding:
    split: str
    predictor_builder: LockedBuilderSpec

    def __post_init__(self) -> None:
        _require_nonempty(self.split, label="benchmark split")
        if not isinstance(self.predictor_builder, LockedBuilderSpec):
            raise TypeError("benchmark predictor_builder must be LockedBuilderSpec")


@dataclass(frozen=True, slots=True)
class _ArmBinding:
    release_id: str
    intervention_builder: LockedBuilderSpec

    def __post_init__(self) -> None:
        _require_nonempty(self.release_id, label="arm release_id")
        if not isinstance(self.intervention_builder, LockedBuilderSpec):
            raise TypeError("arm intervention_builder must be LockedBuilderSpec")


@dataclass(frozen=True, slots=True)
class _DriverConfig:
    schema_version: int
    driver_id: str
    driver_revision_sha256: str
    prepared_data_root: Path
    baseline_arm_id: str
    execution_budget: ExecutionBudget
    model_slots: Mapping[str, LockedBuilderSpec]
    benchmark_cells: Mapping[str, _BenchmarkBinding]
    arms: Mapping[str, _ArmBinding]

    def __post_init__(self) -> None:
        if self.schema_version != _CONFIG_SCHEMA_VERSION:
            raise ValidationError("unsupported static production config schema_version")
        _require_nonempty(self.driver_id, label="driver_id")
        _require_sha256(
            self.driver_revision_sha256,
            label="driver_revision_sha256",
        )
        root = self.prepared_data_root
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValidationError("prepared_data_root must be an absolute path")
        _require_nonempty(self.baseline_arm_id, label="baseline_arm_id")
        if not isinstance(self.execution_budget, ExecutionBudget):
            raise TypeError("execution_budget must be ExecutionBudget")
        model_slots = dict(self.model_slots)
        benchmark_cells = dict(self.benchmark_cells)
        arms = dict(self.arms)
        if not model_slots or not all(
            isinstance(name, str) and name.strip() and isinstance(spec, LockedBuilderSpec)
            for name, spec in model_slots.items()
        ):
            raise ValidationError("model_slots requires named LockedBuilderSpec values")
        for name, spec in model_slots.items():
            if spec.expected_product_sha256 is None:
                raise ValidationError(f"model_slots.{name} requires expected_product_sha256")
        if set(benchmark_cells) != set(_STATIC_SPLITS):
            missing = sorted(set(_STATIC_SPLITS) - set(benchmark_cells))
            extra = sorted(set(benchmark_cells) - set(_STATIC_SPLITS))
            raise ValidationError(
                "benchmark_cells must bind exactly the seven static production cells; "
                f"missing={missing}, extra={extra}"
            )
        for cell_id, binding in benchmark_cells.items():
            if not isinstance(binding, _BenchmarkBinding):
                raise TypeError("benchmark_cells values must be benchmark bindings")
            if binding.split != _STATIC_SPLITS[cell_id]:
                raise ValidationError(
                    f"benchmark_cells.{cell_id}.split differs from the production lock"
                )
            if binding.predictor_builder.expected_product_sha256 is None:
                raise ValidationError(
                    f"benchmark_cells.{cell_id}.predictor_builder requires expected_product_sha256"
                )
        if len(arms) < 2 or not all(
            isinstance(arm_id, str) and arm_id.strip() and isinstance(binding, _ArmBinding)
            for arm_id, binding in arms.items()
        ):
            raise ValidationError("arms requires at least two named arm bindings")
        if self.baseline_arm_id not in arms:
            raise ValidationError("baseline_arm_id is not present in configured arms")
        for arm_id, binding in arms.items():
            if binding.intervention_builder.expected_product_sha256 is not None:
                raise ValidationError(
                    f"arms.{arm_id}.intervention_builder must not declare expected_product_sha256"
                )
        object.__setattr__(self, "model_slots", MappingProxyType(model_slots))
        object.__setattr__(self, "benchmark_cells", MappingProxyType(benchmark_cells))
        object.__setattr__(self, "arms", MappingProxyType(arms))

    @classmethod
    def from_mapping(cls, value: Any) -> _DriverConfig:
        data = _strict_mapping(
            value,
            path="config",
            keys={
                "schema_version",
                "driver_id",
                "driver_revision_sha256",
                "prepared_data_root",
                "baseline_arm_id",
                "execution_budget",
                "model_slots",
                "benchmark_cells",
                "arms",
            },
            required={
                "schema_version",
                "driver_id",
                "driver_revision_sha256",
                "prepared_data_root",
                "baseline_arm_id",
                "model_slots",
                "benchmark_cells",
                "arms",
            },
        )
        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValidationError("config.schema_version must be an integer")
        root_text = _text(data["prepared_data_root"], "config.prepared_data_root")
        root = Path(root_text).expanduser()
        if not root.is_absolute():
            raise ValidationError("config.prepared_data_root must be an absolute path")
        models = _strict_object(data["model_slots"], path="config.model_slots")
        benchmarks = _strict_object(
            data["benchmark_cells"],
            path="config.benchmark_cells",
        )
        arms = _strict_object(data["arms"], path="config.arms")
        return cls(
            schema_version=schema_version,
            driver_id=_text(data["driver_id"], "config.driver_id"),
            driver_revision_sha256=_text(
                data["driver_revision_sha256"],
                "config.driver_revision_sha256",
            ),
            prepared_data_root=root,
            baseline_arm_id=_text(
                data["baseline_arm_id"],
                "config.baseline_arm_id",
            ),
            execution_budget=_execution_budget(
                data.get("execution_budget", {}),
                path="config.execution_budget",
            ),
            model_slots={
                model_slot: LockedBuilderSpec.from_mapping(
                    raw,
                    path=f"config.model_slots.{model_slot}",
                )
                for model_slot, raw in models.items()
            },
            benchmark_cells={
                cell_id: _benchmark_binding(
                    raw,
                    path=f"config.benchmark_cells.{cell_id}",
                )
                for cell_id, raw in benchmarks.items()
            },
            arms={
                arm_id: _arm_binding(raw, path=f"config.arms.{arm_id}")
                for arm_id, raw in arms.items()
            },
        )


class StaticProductionExperimentDriver:
    """Execute exact per-unit multi-arm blocks over verified static adapters."""

    def __init__(self, config: _DriverConfig, *, config_sha256: str) -> None:
        if not isinstance(config, _DriverConfig):
            raise TypeError("config must be a static production driver config")
        _require_sha256(config_sha256, label="config_sha256")
        self._config = config
        self._config_sha256 = config_sha256
        self._adapters: dict[str, BenchmarkAdapter] = {}
        self._preflight_manifest_hash: str | None = None
        self._feature_extractor = VisibleStateFeatureExtractor()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> StaticProductionExperimentDriver:
        """Load a strict UTF-8 JSON config after verifying its exact bytes."""

        _require_sha256(expected_sha256, label="expected config SHA-256")
        source = Path(path).expanduser()
        if not source.is_absolute():
            raise ArtifactError("static production config path must be absolute")
        try:
            size = source.stat().st_size
            if size > _MAX_CONFIG_BYTES:
                raise ArtifactError("static production config exceeds the size limit")
            raw_bytes = source.read_bytes()
        except OSError as exc:
            raise ArtifactError("cannot read static production driver config") from exc
        observed = hashlib.sha256(raw_bytes).hexdigest()
        if observed != expected_sha256:
            raise ArtifactError("static production driver config checksum mismatch")
        raw = _strict_json(raw_bytes)
        return cls(_DriverConfig.from_mapping(raw), config_sha256=observed)

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "implementation_id": _DRIVER_IMPLEMENTATION_ID,
                "schema_version": _CONFIG_SCHEMA_VERSION,
                "driver_id": self._config.driver_id,
                "driver_revision_sha256": self._config.driver_revision_sha256,
                "config_sha256": self._config_sha256,
                "baseline_arm_id": self._config.baseline_arm_id,
                "execution_budget_sha256": content_hash(self._config.execution_budget),
                "model_slots": {
                    name: spec.binding_identity for name, spec in self._config.model_slots.items()
                },
                "benchmark_cells": {
                    name: {
                        "split": binding.split,
                        "predictor_builder": binding.predictor_builder.binding_identity,
                    }
                    for name, binding in self._config.benchmark_cells.items()
                },
                "arms": {
                    name: {
                        "release_id": binding.release_id,
                        "intervention_builder": binding.intervention_builder.binding_identity,
                    }
                    for name, binding in self._config.arms.items()
                },
                "feature_schema_id": self._feature_extractor.feature_schema_id,
                "feature_schema_hash": self._feature_extractor.feature_schema_hash,
            }
        )

    def preflight(self, manifest: ExperimentManifest) -> Sequence[str]:
        """Verify every selected binding and prepared cache without executing a cell."""

        self._adapters.clear()
        self._preflight_manifest_hash = None
        blockers = self._manifest_blockers(manifest)
        if blockers:
            return tuple(blockers)

        used_cells = tuple(target.cell_id for target in manifest.benchmarks)
        used_arms = tuple(arm.arm_id for arm in manifest.arms)
        builder_checks: list[tuple[str, LockedBuilderSpec, str]] = []
        builder_checks.extend(
            (
                f"model slot {model_slot}",
                self._config.model_slots[model_slot],
                "backend",
            )
            for model_slot in manifest.model_slots
        )
        builder_checks.extend(
            (
                f"benchmark {cell_id}",
                self._config.benchmark_cells[cell_id].predictor_builder,
                "predictor",
            )
            for cell_id in used_cells
        )
        builder_checks.extend(
            (
                f"arm {arm_id}",
                self._config.arms[arm_id].intervention_builder,
                "intervention",
            )
            for arm_id in used_arms
        )
        for label, spec, kind in builder_checks:
            try:
                builder = _resolve_locked_builder(spec, expected_kind=kind)
                messages = _builder_preflight(builder, spec.parameters)
            except Exception as exc:
                blockers.append(
                    f"{label}: "
                    + _safe_blocker_message(
                        exc,
                        prepared_root=self._config.prepared_data_root,
                    )
                )
                continue
            blockers.extend(
                f"{label}: "
                + _safe_operator_message(
                    message,
                    prepared_root=self._config.prepared_data_root,
                )
                for message in messages
            )

        for target in manifest.benchmarks:
            try:
                adapter = load_production_benchmark_adapter(
                    target.cell_id,
                    self._config.prepared_data_root,
                )
                if not isinstance(
                    adapter,
                    (PreparedQABenchmarkAdapter, PreparedRetrievalBenchmarkAdapter),
                ):
                    raise ArtifactError("driver loaded a non-static production adapter")
                examples = tuple(adapter.examples(target.split))
                identifiers = {example.example_id for example in examples}
                if len(identifiers) != len(examples):
                    raise ArtifactError("production adapter contains duplicate unit IDs")
                missing = sorted(set(target.unit_ids) - identifiers)
                if missing:
                    raise ArtifactError(
                        "frozen manifest units are absent from the prepared adapter: "
                        + ", ".join(missing[:5])
                    )
                self._adapters[target.cell_id] = adapter
            except Exception as exc:
                blockers.append(
                    f"benchmark {target.cell_id}: "
                    + _safe_blocker_message(
                        exc,
                        prepared_root=self._config.prepared_data_root,
                    )
                )

        if blockers:
            self._adapters.clear()
            return tuple(blockers)
        self._preflight_manifest_hash = manifest.manifest_hash
        return ()

    def preflight_units(
        self,
        manifest: ExperimentManifest,
        resolved_units: Mapping[str, tuple[str, ...]],
    ) -> Sequence[str]:
        """Prove every planned unit exists before the plan is accepted."""

        if self._preflight_manifest_hash != manifest.manifest_hash:
            return ("unit inventory validation requires a successful driver preflight",)
        expected_keys = {f"{target.cell_id}:{target.split}" for target in manifest.benchmarks}
        if set(resolved_units) != expected_keys:
            return ("resolved unit inventories differ from manifest benchmark targets",)
        blockers: list[str] = []
        for target in manifest.benchmarks:
            key = f"{target.cell_id}:{target.split}"
            try:
                adapter = self._adapters[target.cell_id]
                identifiers = {example.example_id for example in adapter.examples(target.split)}
            except Exception as exc:
                blockers.append(
                    f"benchmark {target.cell_id}: "
                    + _safe_blocker_message(
                        exc,
                        prepared_root=self._config.prepared_data_root,
                    )
                )
                continue
            missing = sorted(set(resolved_units[key]) - identifiers)
            if missing:
                blockers.append(
                    f"benchmark {target.cell_id}: resolved units are absent from the "
                    "prepared adapter: " + ", ".join(missing[:5])
                )
        return tuple(blockers)

    def execute_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentCellResult:
        if self._preflight_manifest_hash != manifest.manifest_hash:
            raise ArtifactError(
                "static production driver requires successful preflight for this manifest"
            )
        target = self._validate_cell(manifest, cell)
        try:
            adapter = self._adapters[cell.benchmark_id]
        except KeyError as exc:  # pragma: no cover - preflight invariant
            raise ArtifactError("preflight did not retain the production adapter") from exc

        arm_specs = {arm.arm_id: arm for arm in manifest.arms}
        dataset_identity = _dataset_identity(adapter, split=cell.split)
        benchmark_binding = self._config.benchmark_cells[cell.benchmark_id]
        built_arms: list[BenchmarkBlockArm] = []
        contexts: list[InterventionContext] = []
        for arm_id in cell.arm_ids:
            arm_spec = arm_specs[arm_id]
            runtime = self._build_runtime(cell)
            predictor = self._build_predictor(
                cell,
                runtime=runtime,
                binding=benchmark_binding,
            )
            intervention = self._build_intervention(
                cell,
                arm_spec=arm_spec,
                runtime=runtime,
            )
            runtime_product_sha256 = _require_product_sha256(
                static_runtime_product_sha256(runtime),
                expected=self._config.model_slots[cell.model_slot].expected_product_sha256,
                label="static runtime product",
            )
            predictor_product_sha256 = _require_product_sha256(
                static_predictor_product_sha256(predictor),
                expected=benchmark_binding.predictor_builder.expected_product_sha256,
                label="static predictor product",
            )
            model_identity = freeze_mapping({"runtime_product_sha256": runtime_product_sha256})
            predictor_identity = freeze_mapping(
                {"predictor_product_sha256": predictor_product_sha256}
            )
            bindings = _comparison_bindings(
                dataset_identity=dataset_identity,
                model_identity=model_identity,
                predictor_identity=predictor_identity,
                arm_spec=arm_spec,
                builder_spec=self._config.arms[arm_id].intervention_builder,
                intervention=intervention,
                driver_config_sha256=self._config_sha256,
            )
            run_config = _context_run_config(
                runtime.run_config,
                dataset_identity=dataset_identity,
                model_identity=model_identity,
            )
            context = InterventionContext(
                backend=runtime.backend,
                dense_embedder=runtime.dense_embedder,
                retriever=runtime.retriever,
                vector_searcher=runtime.vector_searcher,
                tools=runtime.tools,
                external_runtime=runtime.external_runtime,
                artifact_manager=runtime.artifact_manager,
                seed=cell.seed,
                run_config=run_config,
                intent_contract=runtime.intent_contract,
                monitor=RunMonitor(),
                pricing_catalog=runtime.pricing_catalog,
                strict_intent=runtime.strict_intent,
            )
            contexts.append(context)
            built_arms.append(
                BenchmarkBlockArm(
                    comparison_arm=ComparisonArm(
                        arm_id=arm_id,
                        release_id=arm_spec.release_id,
                        bindings=bindings,
                    ),
                    intervention=intervention,
                    context=context,
                    predictor=predictor,
                )
            )

        profile = ComparisonProfile(
            profile_id=_PROFILE_ID,
            kind=manifest.estimand_profile_kind,
            required_slots=(
                "dataset",
                "model",
                "downstream_predictor",
                "driver_config",
            ),
            varied_slots=("intervention",),
            seed_policy="fixed",
            cache_policy="frozen",
        )
        execution_adapter: BenchmarkAdapter = adapter
        if isinstance(adapter, PreparedRetrievalBenchmarkAdapter) and any(
            arm.family is SemanticFamily.CURATE
            for arm in manifest.arms
            if arm.arm_id != self._config.baseline_arm_id
        ):
            execution_adapter = _preload_retrieval_candidates(
                adapter,
                split=cell.split,
                example_id=cell.unit_id,
                retriever=built_arms[0].context.retriever,
                top_k=_retrieval_top_k(benchmark_binding.predictor_builder.parameters),
            )
        block = ControlledMultiArmBlockRunner().run_one(
            execution_adapter,
            cell.split,
            cell.unit_id,
            profile,
            tuple(built_arms),
            planned_arm_ids=cell.arm_ids,
            baseline_arm_id=self._config.baseline_arm_id,
            budget=self._config.execution_budget,
            pre_action_observer=self._pre_action_receipt,
        )
        outcomes = tuple(
            _arm_payload(record, arm_spec=arm_specs[record.arm_id]) for record in block.arms
        )
        if tuple(item["arm_id"] for item in outcomes) != cell.arm_ids:
            raise ValidationError("static production driver produced a partial arm inventory")
        receipts = (
            *_resource_receipts(contexts),
            *(receipt for record in block.arms for receipt in record.downstream_resource_receipts),
        )
        return ExperimentCellResult(
            cell_id=cell.cell_id,
            status="succeeded",
            payload={
                "benchmark_id": block.benchmark_id,
                "split": block.split,
                "unit_id": block.example_id,
                "state_hash": block.state_hash,
                "profile_id": block.profile_id,
                "baseline_arm_id": block.baseline_arm_id,
                "planned_arm_ids": block.planned_arm_ids,
                "pre_action": block.pre_action,
                "arms": outcomes,
                "metric_deltas": block.metric_deltas,
                "balance": to_jsonable(block.balance),
                "observed_dependency_hashes": block.observed_dependency_hashes,
                "dataset_identity_sha256": content_hash(dataset_identity),
                "driver_config_sha256": self._config_sha256,
                "target_source_revision": target.source_revision,
            },
            resource_receipts=receipts,
        )

    def _manifest_blockers(self, manifest: ExperimentManifest) -> list[str]:
        if not isinstance(manifest, ExperimentManifest):
            return ["manifest must be an ExperimentManifest"]
        blockers: list[str] = []
        if set(manifest.model_slots) != set(self._config.model_slots):
            blockers.append("manifest model_slots differ from the locked driver config")
        manifest_arms = {arm.arm_id: arm for arm in manifest.arms}
        if set(manifest_arms) != set(self._config.arms):
            blockers.append("manifest arms differ from the locked driver config")
        else:
            for arm_id, arm in manifest_arms.items():
                configured = self._config.arms[arm_id]
                if configured.release_id != arm.release_id:
                    blockers.append(
                        f"arm {arm_id}: manifest release differs from the locked config"
                    )
        no_op_ids = tuple(arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.NO_OP)
        if no_op_ids != (self._config.baseline_arm_id,):
            blockers.append(
                "manifest requires exactly one NO_OP arm matching configured baseline_arm_id"
            )
        seen_cells: set[str] = set()
        for target in manifest.benchmarks:
            if target.external:
                blockers.append(
                    f"benchmark {target.cell_id}: stateful/external targets require "
                    "an atomic replay driver"
                )
                continue
            if target.cell_id not in _STATIC_SPLITS:
                blockers.append(
                    f"benchmark {target.cell_id}: target is not one of the seven static "
                    "production adapters"
                )
                continue
            if target.cell_id in seen_cells:
                blockers.append(f"benchmark {target.cell_id}: duplicate target")
            seen_cells.add(target.cell_id)
            if target.split != _STATIC_SPLITS[target.cell_id]:
                blockers.append(
                    f"benchmark {target.cell_id}: split differs from the production lock"
                )
            selected = target.arm_ids or tuple(manifest_arms)
            if self._config.baseline_arm_id not in selected:
                blockers.append(
                    f"benchmark {target.cell_id}: selected arm block omits the NO_OP baseline"
                )
        return blockers

    def _validate_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentBenchmarkTarget:
        if not isinstance(cell, ExperimentCell):
            raise TypeError("cell must be ExperimentCell")
        if cell.study_id != manifest.study_id or cell.manifest_hash != manifest.manifest_hash:
            raise ValidationError("experiment cell does not belong to this manifest")
        matches = tuple(
            target
            for target in manifest.benchmarks
            if target.cell_id == cell.benchmark_id and target.split == cell.split
        )
        if len(matches) != 1:
            raise ValidationError("experiment cell target is absent or ambiguous")
        target = matches[0]
        expected_arms = target.arm_ids or tuple(arm.arm_id for arm in manifest.arms)
        if cell.arm_ids != expected_arms:
            raise ValidationError("experiment cell arms differ from the frozen target block")
        if cell.model_slot not in manifest.model_slots:
            raise ValidationError("experiment cell model_slot is outside the manifest")
        if cell.seed not in manifest.seeds:
            raise ValidationError("experiment cell seed is outside the manifest")
        if target.unit_ids and cell.unit_id not in target.unit_ids:
            raise ValidationError("experiment cell unit is outside the frozen target inventory")
        return target

    def _build_runtime(self, cell: ExperimentCell) -> StaticRuntimeComponents:
        spec = self._config.model_slots[cell.model_slot]
        builder = _resolve_locked_builder(spec, expected_kind="backend")
        runtime = builder.build(
            StaticBackendBuildRequest(
                model_slot=cell.model_slot,
                seed=cell.seed,
                builder_parameters=spec.parameters,
            )
        )
        if not isinstance(runtime, StaticRuntimeComponents):
            raise ValidationError(
                f"backend builder must return StaticRuntimeComponents, got {type(runtime).__name__}"
            )
        _require_product_sha256(
            static_runtime_product_sha256(runtime),
            expected=spec.expected_product_sha256,
            label="static runtime product",
        )
        return runtime

    def _build_predictor(
        self,
        cell: ExperimentCell,
        *,
        runtime: StaticRuntimeComponents,
        binding: _BenchmarkBinding,
    ) -> DownstreamPredictor:
        spec = binding.predictor_builder
        builder = _resolve_locked_builder(spec, expected_kind="predictor")
        predictor = builder.build(
            StaticPredictorBuildRequest(
                benchmark_id=cell.benchmark_id,
                split=cell.split,
                model_slot=cell.model_slot,
                seed=cell.seed,
                builder_parameters=spec.parameters,
                runtime=runtime,
            )
        )
        if not isinstance(predictor, DownstreamPredictor):
            raise ValidationError(
                "predictor builder must return a DownstreamPredictor, "
                f"got {type(predictor).__name__}"
            )
        _require_product_sha256(
            static_predictor_product_sha256(predictor),
            expected=spec.expected_product_sha256,
            label="static predictor product",
        )
        return predictor

    def _build_intervention(
        self,
        cell: ExperimentCell,
        *,
        arm_spec: ExperimentArmSpec,
        runtime: StaticRuntimeComponents,
    ) -> Intervention:
        binding = self._config.arms[arm_spec.arm_id]
        spec = binding.intervention_builder
        builder = _resolve_locked_builder(spec, expected_kind="intervention")
        intervention = builder.build(
            StaticArmBuildRequest(
                arm_id=arm_spec.arm_id,
                release_id=arm_spec.release_id,
                family=arm_spec.family,
                profile_kind=arm_spec.profile_kind,
                model_slot=cell.model_slot,
                seed=cell.seed,
                builder_parameters=spec.parameters,
                arm_parameters=arm_spec.parameters,
                runtime=runtime,
            )
        )
        if not isinstance(intervention, Intervention):
            raise ValidationError(
                "intervention builder must return an Intervention, "
                f"got {type(intervention).__name__}"
            )
        if intervention.release_id != arm_spec.release_id:
            raise ValidationError("intervention builder returned another release_id")
        if intervention.family is not arm_spec.family:
            raise ValidationError("intervention builder returned another semantic family")
        return intervention

    def _pre_action_receipt(self, state: FrozenAgentState) -> Mapping[str, Any]:
        observation = self._feature_extractor.extract(state)
        features = observation.features
        return {
            "feature_schema_id": self._feature_extractor.feature_schema_id,
            "feature_schema_hash": self._feature_extractor.feature_schema_hash,
            "state_hash": observation.state_hash,
            "feature_hash": observation.feature_hash,
            "provenance_hash": content_hash(observation.provenance),
            "defect_scores": {
                SemanticFamily.REFORMULATE.value: features.reformulation_need_score,
                SemanticFamily.DECOMPOSE.value: features.multi_step_score,
                SemanticFamily.ACQUIRE.value: features.retrieval_need_score,
                SemanticFamily.CLARIFY.value: features.ambiguity_score,
                SemanticFamily.CURATE.value: features.context_pressure,
            },
        }


def factory() -> StaticProductionExperimentDriver:
    """Zero-argument factory driven only by two explicit environment bindings."""

    path = os.environ.get(CONFIG_PATH_ENV)
    expected_sha256 = os.environ.get(CONFIG_SHA256_ENV)
    if not path or not expected_sha256:
        raise ArtifactError(
            f"set both {CONFIG_PATH_ENV} and {CONFIG_SHA256_ENV} before loading "
            "the static production driver"
        )
    return StaticProductionExperimentDriver.from_file(
        path,
        expected_sha256=expected_sha256,
    )


def _resolve_locked_builder(
    spec: LockedBuilderSpec,
    *,
    expected_kind: str,
) -> LockedComponentBuilder:
    if expected_kind not in _BUILDER_KINDS:  # pragma: no cover - internal invariant
        raise ValueError("unknown builder kind")
    candidate = resolve_driver_entrypoint(spec.entrypoint)
    if not callable(candidate):
        raise ArtifactError("component builder entrypoint must be a zero-argument factory")
    try:
        builder = candidate()
    except TypeError as exc:
        raise ArtifactError("component builder entrypoint must be a zero-argument factory") from exc
    if not isinstance(builder, LockedComponentBuilder):
        raise ArtifactError("component builder does not implement the locked builder contract")
    if builder.builder_kind != expected_kind:
        raise ArtifactError(
            f"component builder kind is {builder.builder_kind!r}, expected {expected_kind!r}"
        )
    identity = _runtime_fingerprint(builder, label=f"{expected_kind} builder")
    if content_hash(identity) != spec.fingerprint_sha256:
        raise ArtifactError(f"{expected_kind} builder fingerprint checksum mismatch")
    return builder


def _builder_preflight(
    builder: LockedComponentBuilder,
    parameters: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers = builder.preflight(parameters)
    if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
        raise ValidationError("component builder preflight must return a sequence of strings")
    normalized = tuple(blockers)
    if any(not isinstance(item, str) or not item.strip() for item in normalized):
        raise ValidationError("component builder preflight blockers must be non-empty strings")
    return normalized


def _comparison_bindings(
    *,
    dataset_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    predictor_identity: Mapping[str, Any],
    arm_spec: ExperimentArmSpec,
    builder_spec: LockedBuilderSpec,
    intervention: Intervention,
    driver_config_sha256: str,
) -> tuple[DependencyBinding, ...]:
    return (
        DependencyBinding(
            slot="dataset",
            role=DependencyRole.NUISANCE,
            implementation_id=str(dataset_identity["benchmark_id"]),
            revision=content_hash(dataset_identity),
            config={"identity": dataset_identity},
        ),
        DependencyBinding(
            slot="model",
            role=DependencyRole.NUISANCE,
            implementation_id="operator_backend",
            revision=content_hash(model_identity),
            config={"fingerprint": model_identity},
        ),
        DependencyBinding(
            slot="downstream_predictor",
            role=DependencyRole.NUISANCE,
            implementation_id="operator_predictor",
            revision=content_hash(predictor_identity),
            config={"fingerprint": predictor_identity},
        ),
        DependencyBinding(
            slot="driver_config",
            role=DependencyRole.NUISANCE,
            implementation_id=_DRIVER_IMPLEMENTATION_ID,
            revision=driver_config_sha256,
            config={
                "schema_version": _CONFIG_SCHEMA_VERSION,
                "config_sha256": driver_config_sha256,
            },
        ),
        DependencyBinding(
            slot="intervention",
            role=DependencyRole.TREATMENT,
            implementation_id=intervention.method_id,
            revision=builder_spec.fingerprint_sha256,
            config={
                "release_id": arm_spec.release_id,
                "family": arm_spec.family.value,
                "profile_kind": arm_spec.profile_kind.value,
                "arm_parameters": arm_spec.parameters,
                "builder": builder_spec.binding_identity,
            },
        ),
    )


def _context_run_config(
    value: Mapping[str, Any],
    *,
    dataset_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    run_config = dict(value)
    declared = run_config.pop("dependency_fingerprints", {})
    if not isinstance(declared, Mapping):
        raise ValidationError("runtime dependency_fingerprints must be an object")
    dependencies = dict(declared)
    collisions = {"dataset", "model"} & set(dependencies)
    if collisions:
        raise ValidationError(
            "backend builder cannot override driver-owned dependency slots: "
            + ", ".join(sorted(collisions))
        )
    dependencies["dataset"] = dataset_identity
    dependencies["model"] = model_identity
    run_config["dependency_fingerprints"] = dependencies
    return freeze_mapping(run_config)


def _dataset_identity(
    adapter: BenchmarkAdapter,
    *,
    split: str,
) -> Mapping[str, Any]:
    if isinstance(adapter, PreparedQABenchmarkAdapter):
        assets = {"query": _prepared_identity(adapter.prepared)}
    elif isinstance(adapter, PreparedRetrievalBenchmarkAdapter):
        assets = {
            "query": _prepared_identity(adapter.query_prepared),
            "corpus": _prepared_identity(adapter.corpus_prepared),
        }
    else:
        raise ValidationError("static production driver received an unsupported adapter")
    return freeze_mapping(
        {
            "benchmark_id": adapter.benchmark_id,
            "split": split,
            "assets": assets,
        }
    )


def _prepared_identity(prepared: Any) -> Mapping[str, Any]:
    required = (
        "asset_id",
        "cell_id",
        "source_revision",
        "source_sha256",
        "processor",
        "processor_revision",
        "counts",
        "file_sha256",
    )
    if any(not hasattr(prepared, name) for name in required):
        raise ValidationError("prepared benchmark does not expose its audit identity")
    return freeze_mapping({name: getattr(prepared, name) for name in required})


def _runtime_fingerprint(value: Any, *, label: str) -> Mapping[str, Any]:
    fingerprint = getattr(value, "fingerprint", None)
    if not callable(fingerprint):
        raise ValidationError(f"{label} must expose fingerprint()")
    identity = fingerprint()
    if not isinstance(identity, Mapping) or not identity:
        raise ValidationError(f"{label} fingerprint must be a non-empty mapping")
    try:
        return freeze_mapping(identity)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} fingerprint must be canonical JSON") from exc


def static_runtime_product_sha256(runtime: StaticRuntimeComponents) -> str:
    """Hash every object and option that can affect a static arm execution.

    This is intentionally stricter than locking the backend builder. A builder
    can stay byte-for-byte identical while its endpoint routing, model, index,
    tools, runtime assets, prompt defaults, intent policy, or pricing inputs
    drift. Operators place this returned hash in ``expected_product_sha256``;
    every fresh product is checked before it can execute.
    """

    if not isinstance(runtime, StaticRuntimeComponents):
        raise TypeError("runtime product hashing requires StaticRuntimeComponents")
    component_slots = (
        "backend",
        "dense_embedder",
        "retriever",
        "vector_searcher",
        "tools",
        "external_runtime",
        "artifact_manager",
    )
    components: dict[str, Any] = {}
    for slot in component_slots:
        value = getattr(runtime, slot)
        components[slot] = (
            None if value is None else _runtime_fingerprint(value, label=f"runtime {slot}")
        )
    product_identity = {
        "components": components,
        "run_config": runtime.run_config,
        "intent_contract": runtime.intent_contract,
        "pricing_catalog": runtime.pricing_catalog,
        "strict_intent": runtime.strict_intent,
    }
    try:
        return content_hash(product_identity)
    except (TypeError, ValueError) as exc:
        raise ValidationError("static runtime product identity must be canonical JSON") from exc


def static_predictor_product_sha256(predictor: DownstreamPredictor) -> str:
    """Hash the actual downstream predictor returned by its locked builder."""

    if not isinstance(predictor, DownstreamPredictor):
        raise TypeError("predictor product hashing requires a DownstreamPredictor")
    return content_hash(_runtime_fingerprint(predictor, label="downstream predictor"))


def _require_product_sha256(
    observed: str,
    *,
    expected: str | None,
    label: str,
) -> str:
    if expected is None:  # pragma: no cover - rejected by config construction
        raise ArtifactError(f"{label} has no configured checksum")
    if observed != expected:
        raise ArtifactError(f"{label} checksum mismatch")
    return observed


def _arm_payload(record: Any, *, arm_spec: ExperimentArmSpec) -> Mapping[str, Any]:
    base: dict[str, Any] = {
        "arm_id": record.arm_id,
        "release_id": record.release_id,
        "family": arm_spec.family.value,
        "profile_kind": arm_spec.profile_kind.value,
        "status": record.status,
        "usage": record.total_usage.as_dict(),
        "downstream_duration_ms": record.downstream_duration_ms,
        "downstream_resource_receipt_hashes": tuple(
            content_hash(receipt) for receipt in record.downstream_resource_receipts
        ),
    }
    intervention_record = record.intervention_record
    if intervention_record is not None:
        patch = intervention_record.patch
        if patch.profile_kind != arm_spec.profile_kind.value:
            raise ValidationError(
                f"arm {record.arm_id!r} patch profile_kind differs from the manifest"
            )
        base.update(
            {
                "intervention_status": intervention_record.status.value,
                "intervention_duration_ms": intervention_record.duration_ms,
                "monitor_run_id": intervention_record.monitor_run_id,
                "patch_hash": patch.patch_hash,
                "trace_hash": patch.trace.trace_hash,
            }
        )
    if record.status == "succeeded":
        if record.benchmark_record is None:  # pragma: no cover - record invariant
            raise RuntimeError("successful arm lost its benchmark record")
        base["metrics"] = record.benchmark_record.metrics
        base["evaluation_provenance"] = record.benchmark_record.evaluation_provenance
    else:
        if record.failure is None:  # pragma: no cover - record invariant
            raise RuntimeError("failed arm lost its failure")
        base["failure"] = {
            "scope": "algorithm",
            "category": record.failure.category,
            "message": _redact_secret_text(record.failure.message),
            "monitor_run_id": record.failure.monitor_run_id,
        }
    return freeze_mapping(base)


def _resource_receipts(
    contexts: Sequence[InterventionContext],
) -> tuple[ResourceReceipt, ...]:
    receipts: list[ResourceReceipt] = []
    for context in contexts:
        monitor = context.monitor
        if not isinstance(monitor, RunMonitor):  # pragma: no cover - construction invariant
            continue
        for snapshot in monitor.list_snapshots():
            receipts.extend(snapshot.resource_receipts)
    return tuple(receipts)


class _PreloadedRetrievalAdapter:
    """Single-unit view that exposes one shared ranked candidate context.

    Gold remains owned by the wrapped production adapter.  Both arms receive
    the exact same visible candidate list; the wrapper forwards scoring to the
    original example so evaluator isolation and official metrics are unchanged.
    """

    pairing_mode = "stateless_score_twice"

    def __init__(
        self,
        base: PreparedRetrievalBenchmarkAdapter,
        original: BenchmarkExample,
        visible: BenchmarkExample,
    ) -> None:
        self.benchmark_id = base.benchmark_id
        self._base = base
        self._original = original
        self._visible = visible

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self._base.split:
            raise ValidationError("preloaded retrieval adapter split differs")
        return (self._visible,)

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if example is not self._visible:
            raise ValidationError("preloaded retrieval example differs from the frozen unit")
        return self._base.score(self._original, prediction)


def _preload_retrieval_candidates(
    adapter: PreparedRetrievalBenchmarkAdapter,
    *,
    split: str,
    example_id: str,
    retriever: Any,
    top_k: int,
) -> BenchmarkAdapter:
    if not isinstance(retriever, Retriever):
        raise ValidationError(
            "curation on a retrieval benchmark requires the shared locked retriever"
        )
    original = adapter.example(split, example_id)
    results = tuple(retriever.retrieve(original.query, top_k=top_k))
    if not results or any(not isinstance(item, RetrievalResult) for item in results):
        raise ValidationError("shared retriever returned no valid curation candidates")
    expected_ranks = tuple(range(1, len(results) + 1))
    if tuple(item.rank for item in results) != expected_ranks:
        raise ValidationError("curation candidates require contiguous ranks from one")
    document_ids = tuple(item.document.doc_id for item in results)
    if len(document_ids) != len(set(document_ids)):
        raise ValidationError("curation candidates contain duplicate document IDs")
    context = tuple(
        ContextBlock(
            block_id=item.document.doc_id,
            text="\n".join(
                part for part in (item.document.title, item.document.text) if part.strip()
            ),
            source="shared_locked_retriever",
            score=item.score,
            metadata={
                "document_id": item.document.doc_id,
                "retrieval_rank": item.rank,
                "retrieval_score": item.score,
                "corpus_revision": item.document.corpus_revision,
                "candidate_set_role": "pre_action_shared_context",
            },
        )
        for item in results
    )
    visible = BenchmarkExample(
        example_id=original.example_id,
        query=original.query,
        instruction=original.instruction,
        visible_context=context,
        visible_messages=original.visible_messages,
        visible_metadata={
            **dict(original.visible_metadata),
            "preloaded_candidate_count": len(context),
            "preloaded_candidate_ids_sha256": content_hash(document_ids),
        },
        evaluator_gold={},
        visible_tool_names=original.visible_tool_names,
        optimizer_visible_id=original.optimizer_visible_id,
    )
    return _PreloadedRetrievalAdapter(adapter, original, visible)


def _retrieval_top_k(parameters: Mapping[str, Any]) -> int:
    retrieval = parameters.get("retrieval", {})
    if not isinstance(retrieval, Mapping):
        raise ValidationError("retrieval predictor parameters are invalid")
    value = retrieval.get("top_k", 100)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError("retrieval predictor top_k must be positive")
    return value


def _benchmark_binding(value: Any, *, path: str) -> _BenchmarkBinding:
    data = _strict_mapping(
        value,
        path=path,
        keys={"split", "predictor_builder"},
        required={"split", "predictor_builder"},
    )
    return _BenchmarkBinding(
        split=_text(data["split"], f"{path}.split"),
        predictor_builder=LockedBuilderSpec.from_mapping(
            data["predictor_builder"],
            path=f"{path}.predictor_builder",
        ),
    )


def _arm_binding(value: Any, *, path: str) -> _ArmBinding:
    data = _strict_mapping(
        value,
        path=path,
        keys={"release_id", "intervention_builder"},
        required={"release_id", "intervention_builder"},
    )
    return _ArmBinding(
        release_id=_text(data["release_id"], f"{path}.release_id"),
        intervention_builder=LockedBuilderSpec.from_mapping(
            data["intervention_builder"],
            path=f"{path}.intervention_builder",
        ),
    )


def _execution_budget(value: Any, *, path: str) -> ExecutionBudget:
    data = _strict_mapping(
        value,
        path=path,
        keys={
            "max_model_calls",
            "max_retrieval_calls",
            "max_tool_calls",
            "max_output_tokens",
            "max_context_tokens",
            "max_cost_usd",
        },
        required=set(),
    )
    try:
        return ExecutionBudget(**data)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{path} is invalid: {exc}") from exc


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _reject_json_constant(value),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("static production config is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValidationError("static production config root must be an object")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _strict_mapping(
    value: Any,
    *,
    path: str,
    keys: set[str],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{path} keys must be strings")
    unknown = set(value) - keys
    missing = required - set(value)
    if unknown:
        raise ValidationError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValidationError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return value


def _strict_object(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"{path} must be a non-empty object")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValidationError(f"{path} keys must be non-empty strings")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string")
    return value


def _require_nonempty(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")


def _require_seed(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("seed must be a non-negative integer")


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")


def _safe_blocker_message(error: Exception, *, prepared_root: Path) -> str:
    return f"{type(error).__name__}: " + _safe_operator_message(
        str(error),
        prepared_root=prepared_root,
    )


def _safe_operator_message(value: str, *, prepared_root: Path) -> str:
    message = value[:2048].replace(str(prepared_root), "<prepared_data_root>")
    return _redact_secret_text(message)


def _redact_secret_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_TEXT_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


__all__ = [
    "CONFIG_PATH_ENV",
    "CONFIG_SHA256_ENV",
    "LockedBuilderSpec",
    "LockedComponentBuilder",
    "StaticArmBuildRequest",
    "StaticBackendBuildRequest",
    "StaticPredictorBuildRequest",
    "StaticProductionExperimentDriver",
    "StaticRuntimeComponents",
    "factory",
    "static_predictor_product_sha256",
    "static_runtime_product_sha256",
]
