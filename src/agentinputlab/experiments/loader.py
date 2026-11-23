"""Side-effect-free manifest loading and explicit driver preflight checks.

Loading never imports operator code.  Preflight is an explicit boundary that
may construct the selected driver and invoke its read-only ``preflight`` hook;
it never calls ``execute_cell``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.registry import AlgorithmRegistry

from .bindings import (
    FrozenBindingReceipt,
    build_guarded_driver_fingerprint,
    resolve_frozen_binding_receipt,
)
from .parity_identity import (
    ParityImplementationContract,
    build_parity_implementation_contract,
)
from .schema import (
    ExperimentArmKind,
    ExperimentBenchmarkTarget,
    ExperimentManifest,
    ExperimentStage,
    PrerequisiteKind,
    require_sha256,
)
from .secrets import redact_secret_text, require_secret_free

_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PreflightReport:
    study_id: str
    manifest_hash: str
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved_unit_counts: Mapping[str, int]
    driver_entrypoint: str | None
    binding_receipt: FrozenBindingReceipt | None = None
    parity_contract: ParityImplementationContract | None = None
    operator_driver_fingerprint: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
    )
    guarded_driver_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(self.blockers))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self,
            "resolved_unit_counts",
            freeze_mapping(self.resolved_unit_counts),
        )
        object.__setattr__(
            self,
            "operator_driver_fingerprint",
            freeze_mapping(self.operator_driver_fingerprint),
        )
        if (
            self.binding_receipt is not None
            and self.binding_receipt.manifest_hash != self.manifest_hash
        ):
            raise ValidationError("preflight binding receipt belongs to another manifest")
        if self.guarded_driver_hash is not None:
            require_sha256(
                self.guarded_driver_hash,
                label="preflight guarded_driver_hash",
            )
        if self.ready and (
            self.binding_receipt is None
            or not self.operator_driver_fingerprint
            or self.guarded_driver_hash is None
        ):
            raise ValidationError(
                "ready preflight must bind its receipt and effective driver identity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "manifest_hash": self.manifest_hash,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "resolved_unit_counts": dict(self.resolved_unit_counts),
            "driver_entrypoint": self.driver_entrypoint,
            "binding_receipt": (
                self.binding_receipt.to_dict() if self.binding_receipt is not None else None
            ),
            "parity_contract": (
                self.parity_contract.to_dict() if self.parity_contract is not None else None
            ),
            "guarded_driver_hash": self.guarded_driver_hash,
        }


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    """Load one strict JSON manifest without resolving operator resources."""

    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"experiment manifest is not valid UTF-8 JSON: {source}") from exc
    try:
        return ExperimentManifest.from_mapping(raw)
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive boundary
        raise ValidationError("invalid experiment manifest") from exc


def resolve_driver_entrypoint(entrypoint: str) -> Any:
    """Import an operator-selected experiment driver only on explicit request."""

    module_name, attribute_path = entrypoint.split(":", 1)
    try:
        value: Any = importlib.import_module(module_name)
        for segment in attribute_path.split("."):
            value = getattr(value, segment)
        return value
    except (ImportError, AttributeError) as exc:
        raise ArtifactError(f"cannot resolve experiment driver {entrypoint!r}: {exc}") from exc


def preflight_experiment(
    manifest: ExperimentManifest,
    *,
    driver_entrypoint: str | None = None,
    environment: Mapping[str, str] | None = None,
    resolve_units: bool = True,
) -> tuple[PreflightReport, Mapping[str, tuple[str, ...]]]:
    """Validate local prerequisites and invoke only the driver's preflight hook.

    Environment values are deliberately never included in the report.  File
    paths are likewise represented only by the environment-variable name.  A
    user-selected zero-argument driver factory may be constructed, fingerprinted,
    and asked for read-only blockers; ``execute_cell`` is never called.
    """

    env = dict(os.environ if environment is None else environment)
    blockers: list[str] = []
    warnings: list[str] = []
    units: dict[str, tuple[str, ...]] = {}
    benchmark_registry = BenchmarkRegistry.default()
    algorithm_registry = AlgorithmRegistry.default()
    effective_driver = driver_entrypoint or manifest.driver_entrypoint
    binding_receipt: FrozenBindingReceipt | None = None
    parity_contract: ParityImplementationContract | None = None
    operator_fingerprint: Mapping[str, Any] = {}
    guarded_driver_hash: str | None = None

    revision_environment = manifest.metadata.get("source_revision_placeholder_must_match_env")
    if revision_environment is not None:
        if not isinstance(revision_environment, str) or not revision_environment:
            blockers.append("source_revision_placeholder_must_match_env metadata is invalid")
        else:
            expected_revision = env.get(revision_environment)
            if expected_revision:
                for target in manifest.benchmarks:
                    if target.external and target.source_revision != expected_revision:
                        blockers.append(
                            f"benchmark {target.cell_id}: source revision is not bound "
                            f"to {revision_environment}; reload the blueprint after "
                            "setting the environment"
                        )

    for target in manifest.benchmarks:
        if not target.external:
            try:
                cell = benchmark_registry.get(target.cell_id)
            except Exception as exc:
                blockers.append(f"benchmark {target.cell_id}: {exc}")
            else:
                selected_arm_ids = set(target.arm_ids) or {arm.arm_id for arm in manifest.arms}
                unsupported = sorted(
                    arm.arm_id
                    for arm in manifest.arms
                    if arm.arm_id in selected_arm_ids
                    and arm.arm_kind is ExperimentArmKind.RELEASE
                    and arm.family.value != "no_op"
                    and arm.family not in cell.families
                )
                if unsupported:
                    blockers.append(
                        f"benchmark {target.cell_id}: selected arms are outside the "
                        f"declared family support: {', '.join(unsupported)}"
                    )
                declared_revision = manifest.metadata.get("benchmark_revisions", {})
                if isinstance(declared_revision, Mapping):
                    pinned = declared_revision.get(target.cell_id)
                    if pinned is not None and pinned != cell.source_revision:
                        blockers.append(
                            f"benchmark {target.cell_id}: manifest revision differs from registry"
                        )
        elif target.source_revision == "operator-pin-required-before-execution":
            blockers.append(
                f"benchmark {target.cell_id}: external source revision is still a placeholder"
            )
        if resolve_units:
            try:
                units[_target_key(target)] = resolve_target_units(target, environment=env)
            except (ArtifactError, ValidationError) as exc:
                blockers.append(f"benchmark {target.cell_id}: {exc}")

    for arm in manifest.arms:
        if arm.arm_kind is ExperimentArmKind.POLICY:
            blockers.extend(_policy_arm_blockers(arm, manifest=manifest))
            continue
        if arm.release_id == "baseline.no_op":
            continue
        try:
            method, release = algorithm_registry.get_release(arm.release_id)
        except Exception as exc:
            blockers.append(f"arm {arm.arm_id}: {exc}")
            continue
        if method.semantic_family is not arm.family:
            blockers.append(
                f"arm {arm.arm_id}: release belongs to {method.semantic_family.value}, "
                f"not {arm.family.value}"
            )
        if arm.profile_kind not in release.profile_modes:
            supported = ", ".join(mode.value for mode in release.profile_modes) or "none"
            blockers.append(
                f"arm {arm.arm_id}: release does not support "
                f"{arm.profile_kind.value}; supported profile modes: {supported}"
            )
        if release.entrypoint is None:
            reason = "; ".join(release.blocking_reasons) or "entrypoint is absent"
            blockers.append(f"arm {arm.arm_id}: release is unavailable: {reason}")

    prerequisite_blockers: list[str] = []
    for prerequisite in manifest.prerequisites:
        if prerequisite.kind is PrerequisiteKind.ENV:
            if not env.get(prerequisite.name):
                prerequisite_blockers.append(
                    f"missing environment prerequisite {prerequisite.name}: "
                    f"{prerequisite.description}"
                )
        elif prerequisite.kind is PrerequisiteKind.EXECUTABLE:
            if shutil.which(prerequisite.name, path=env.get("PATH")) is None:
                prerequisite_blockers.append(
                    f"missing executable prerequisite {prerequisite.name}: "
                    f"{prerequisite.description}"
                )
        elif prerequisite.kind in {
            PrerequisiteKind.FILE_ENV,
            PrerequisiteKind.DIRECTORY_ENV,
        }:
            raw = env.get(prerequisite.name)
            if not raw:
                prerequisite_blockers.append(
                    f"missing environment prerequisite {prerequisite.name}: "
                    f"{prerequisite.description}"
                )
                continue
            candidate = Path(raw).expanduser()
            expected_type = (
                candidate.is_file()
                if prerequisite.kind is PrerequisiteKind.FILE_ENV
                else candidate.is_dir()
            )
            if not expected_type:
                noun = "file" if prerequisite.kind is PrerequisiteKind.FILE_ENV else "directory"
                prerequisite_blockers.append(
                    f"{prerequisite.name} does not identify an existing {noun}"
                )
                continue
            if prerequisite.sha256_env is not None:
                expected = env.get(prerequisite.sha256_env)
                if not expected:
                    prerequisite_blockers.append(
                        f"missing checksum prerequisite {prerequisite.sha256_env}"
                    )
                else:
                    try:
                        require_sha256(
                            expected,
                            label=f"environment {prerequisite.sha256_env}",
                        )
                    except ValidationError as exc:
                        prerequisite_blockers.append(str(exc))
                    else:
                        actual = _sha256_file(candidate)
                        if actual != expected:
                            prerequisite_blockers.append(
                                f"{prerequisite.name} checksum differs from "
                                f"{prerequisite.sha256_env}"
                            )
    blockers.extend(prerequisite_blockers)
    if not prerequisite_blockers:
        try:
            binding_receipt = resolve_frozen_binding_receipt(
                manifest,
                environment=env,
            )
        except (ArtifactError, ValidationError) as exc:
            blockers.append(f"frozen binding receipt: {exc}")

    if effective_driver is None:
        blockers.append("experiment driver is not configured; pass --driver module:attribute")
    else:
        try:
            driver_factory = resolve_driver_entrypoint(effective_driver)
        except ArtifactError as exc:
            blockers.append(str(exc))
        else:
            driver_blockers, operator_fingerprint = _preflight_driver(
                driver_factory,
                manifest=manifest,
                resolved_units=(units if len(units) == len(manifest.benchmarks) else None),
            )
            blockers.extend(driver_blockers)
            if operator_fingerprint and binding_receipt is not None:
                if manifest.stage is ExperimentStage.PARITY:
                    try:
                        parity_contract = build_parity_implementation_contract(
                            manifest,
                            binding_receipt,
                            operator_fingerprint,
                        )
                    except (ArtifactError, ValidationError) as exc:
                        blockers.append(f"parity implementation identity: {exc}")
                if manifest.stage is not ExperimentStage.PARITY or parity_contract is not None:
                    guarded_fingerprint = build_guarded_driver_fingerprint(
                        operator_fingerprint,
                        binding_receipt,
                        parity_contract=(
                            parity_contract.to_dict() if parity_contract is not None else None
                        ),
                    )
                    guarded_driver_hash = content_hash(guarded_fingerprint)

    if not resolve_units:
        warnings.append("unit inventories were not materialized")
    blockers = sorted(set(blockers))
    warnings = sorted(set(warnings))
    report = PreflightReport(
        study_id=manifest.study_id,
        manifest_hash=manifest.manifest_hash,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        resolved_unit_counts={key: len(value) for key, value in sorted(units.items())},
        driver_entrypoint=effective_driver,
        binding_receipt=binding_receipt,
        parity_contract=parity_contract,
        operator_driver_fingerprint=operator_fingerprint,
        guarded_driver_hash=guarded_driver_hash,
    )
    return report, freeze_mapping(units)


def resolve_target_units(
    target: ExperimentBenchmarkTarget,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    if target.unit_ids:
        return target.unit_ids
    env = os.environ if environment is None else environment
    variable = target.unit_inventory_env
    if variable is None:  # pragma: no cover - schema invariant
        raise ValidationError("benchmark target has no unit source")
    raw_path = env.get(variable)
    if not raw_path:
        raise ArtifactError(f"unit inventory environment {variable} is not set")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ArtifactError(f"unit inventory from {variable} is not an existing file")
    if target.unit_inventory_sha256_env is not None:
        checksum_name = target.unit_inventory_sha256_env
        expected = env.get(checksum_name)
        if not expected:
            raise ArtifactError(f"unit inventory checksum environment {checksum_name} is not set")
        require_sha256(expected, label=f"environment {checksum_name}")
        if _sha256_file(path) != expected:
            raise ArtifactError(f"unit inventory from {variable} does not match {checksum_name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"unit inventory from {variable} is not valid UTF-8 JSON") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ArtifactError(f"unit inventory from {variable} must be a non-empty string array")
    if len(value) != len(set(value)):
        raise ArtifactError(f"unit inventory from {variable} contains duplicate IDs")
    return tuple(value)


def _target_key(target: ExperimentBenchmarkTarget) -> str:
    return f"{target.cell_id}:{target.split}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preflight_driver(
    factory: Any,
    *,
    manifest: ExperimentManifest,
    resolved_units: Mapping[str, tuple[str, ...]] | None,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """Construct a selected driver and run its read-only preflight contract."""

    if not callable(factory):
        return (
            ("experiment driver entrypoint must be a zero-argument factory",),
            {},
        )
    try:
        driver = factory()
    except Exception as exc:
        return (
            ("experiment driver factory failed during preflight: " + _redacted_message(exc),),
            {},
        )
    fingerprint = getattr(driver, "fingerprint", None)
    driver_preflight = getattr(driver, "preflight", None)
    execute_cell = getattr(driver, "execute_cell", None)
    if not callable(fingerprint) or not callable(driver_preflight) or not callable(execute_cell):
        return (
            ("experiment driver does not implement the required driver protocol",),
            {},
        )
    try:
        frozen_fingerprint = freeze_mapping(fingerprint())
    except Exception as exc:
        return (
            ("experiment driver fingerprint is invalid: " + _redacted_message(exc),),
            {},
        )
    if not frozen_fingerprint:
        return (("experiment driver fingerprint must not be empty",), {})
    try:
        require_secret_free(
            frozen_fingerprint,
            label="experiment driver fingerprint",
        )
    except ArtifactError as exc:
        return ((_redacted_message(exc),), {})
    try:
        raw_blockers = driver_preflight(manifest)
    except Exception as exc:
        return (
            ("experiment driver preflight failed: " + _redacted_message(exc),),
            frozen_fingerprint,
        )
    if (
        isinstance(raw_blockers, (str, bytes))
        or not isinstance(raw_blockers, Sequence)
        or any(not isinstance(item, str) or not item.strip() for item in raw_blockers)
    ):
        return (
            ("experiment driver preflight returned invalid blockers",),
            frozen_fingerprint,
        )
    blockers = ["driver preflight: " + _redacted_text(item[:2048]) for item in raw_blockers]
    unit_hook = getattr(driver, "preflight_units", None)
    require_unit_hook = manifest.metadata.get("require_driver_unit_preflight") is True
    if resolved_units is not None:
        if unit_hook is None and require_unit_hook:
            blockers.append(
                "driver preflight: manifest requires preflight_units(), but the "
                "selected driver does not implement it"
            )
        elif unit_hook is not None:
            if not callable(unit_hook):
                blockers.append("driver preflight: preflight_units must be callable")
            else:
                try:
                    raw_unit_blockers = unit_hook(manifest, resolved_units)
                except Exception as exc:
                    blockers.append("driver unit preflight failed: " + _redacted_message(exc))
                else:
                    if (
                        isinstance(raw_unit_blockers, (str, bytes))
                        or not isinstance(raw_unit_blockers, Sequence)
                        or any(
                            not isinstance(item, str) or not item.strip()
                            for item in raw_unit_blockers
                        )
                    ):
                        blockers.append(
                            "driver preflight: preflight_units returned invalid blockers"
                        )
                    else:
                        blockers.extend(
                            "driver unit preflight: " + _redacted_text(item[:2048])
                            for item in raw_unit_blockers
                        )
    return tuple(blockers), frozen_fingerprint


def _policy_arm_blockers(
    arm: Any,
    *,
    manifest: ExperimentManifest,
) -> tuple[str, ...]:
    """Validate frozen policy bindings without opening evaluator outcomes."""

    specifications = {
        "policy.locked_best_eligible_global": {
            "policy_kind": "locked_best_eligible_global",
            "required_parameters": {
                "policy_kind",
                "policy_manifest_env",
                "policy_manifest_sha256_env",
            },
            "required_prerequisites": {
                ("file_env", "policy_manifest_env", "policy_manifest_sha256_env"),
            },
        },
        "policy.eligibility_aware_selector": {
            "policy_kind": "frozen_selector",
            "required_parameters": {
                "policy_kind",
                "policy_manifest_env",
                "policy_manifest_sha256_env",
                "selector_identity_env",
                "selector_bundle_env",
                "selector_bundle_sha256_env",
            },
            "required_prerequisites": {
                ("file_env", "policy_manifest_env", "policy_manifest_sha256_env"),
                ("file_env", "selector_bundle_env", "selector_bundle_sha256_env"),
                ("env", "selector_identity_env", None),
            },
        },
    }
    try:
        specification = specifications[arm.release_id]
    except KeyError:
        return (f"arm {arm.arm_id}: unsupported frozen policy identity",)
    parameters = dict(arm.parameters)
    if set(parameters) != specification["required_parameters"]:
        return (f"arm {arm.arm_id}: frozen policy binding fields are invalid",)
    if parameters.get("policy_kind") != specification["policy_kind"]:
        return (f"arm {arm.arm_id}: frozen policy kind differs from its identity",)
    prerequisites = {
        (item.kind.value, item.name, item.sha256_env) for item in manifest.prerequisites
    }
    blockers: list[str] = []
    for kind, name_key, checksum_key in specification["required_prerequisites"]:
        name = parameters.get(name_key)
        checksum = parameters.get(checksum_key) if checksum_key is not None else None
        if (
            not isinstance(name, str)
            or not _ENVIRONMENT_NAME.fullmatch(name)
            or (
                checksum_key is not None
                and (not isinstance(checksum, str) or not _ENVIRONMENT_NAME.fullmatch(checksum))
            )
        ):
            blockers.append(f"arm {arm.arm_id}: frozen policy environment binding is invalid")
            continue
        if (kind, name, checksum) not in prerequisites:
            blockers.append(
                f"arm {arm.arm_id}: frozen policy binding {name} is not a matching prerequisite"
            )
    return tuple(blockers)


def _redacted_message(error: BaseException) -> str:
    return _redacted_text(str(error)[:2048])


def _redacted_text(value: str) -> str:
    return redact_secret_text(value)


__all__ = [
    "PreflightReport",
    "load_experiment_manifest",
    "preflight_experiment",
    "resolve_driver_entrypoint",
    "resolve_target_units",
]
