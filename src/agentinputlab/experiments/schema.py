"""Strict, execution-neutral schemas for reproducible experiment manifests.

The schemas in this module describe work without performing it.  In
particular, loading a manifest never imports an operator driver, downloads an
asset, opens a model endpoint, or creates a result directory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.enums import ProfileKind, SemanticFamily, StrEnum
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExperimentStage(StrEnum):
    PARITY = "parity"
    SMOKE = "smoke"
    DISCOVERY = "discovery"
    CONFIRMATORY = "confirmatory"
    SELECTOR_CONFIRMATION = "selector_confirmation"
    REPLICATION = "replication"
    TRANSFER = "transfer"


class PrerequisiteKind(StrEnum):
    """Kinds that can be checked without executing an experiment."""

    ENV = "env"
    FILE_ENV = "file_env"
    DIRECTORY_ENV = "directory_env"
    EXECUTABLE = "executable"


class ExperimentArmKind(StrEnum):
    """Whether an arm binds one release or an already-frozen policy."""

    RELEASE = "release"
    POLICY = "policy"


@dataclass(frozen=True, slots=True)
class ExperimentPrerequisite:
    kind: PrerequisiteKind
    name: str
    description: str
    sha256_env: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PrerequisiteKind):
            raise ValidationError("experiment prerequisite kind must be typed")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("experiment prerequisite name must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValidationError("experiment prerequisite description must be non-empty")
        if self.kind is not PrerequisiteKind.EXECUTABLE and not _ENVIRONMENT_NAME.fullmatch(
            self.name
        ):
            raise ValidationError("environment prerequisite names must be uppercase identifiers")
        if self.sha256_env is not None:
            if self.kind is not PrerequisiteKind.FILE_ENV:
                raise ValidationError("sha256_env is valid only for file_env prerequisites")
            if not _ENVIRONMENT_NAME.fullmatch(self.sha256_env):
                raise ValidationError("sha256_env must be an uppercase environment identifier")

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> ExperimentPrerequisite:
        data = _strict_mapping(
            value,
            path=path,
            keys={"kind", "name", "description", "sha256_env"},
            required={"kind", "name", "description"},
        )
        return cls(
            kind=_enum(PrerequisiteKind, data["kind"], f"{path}.kind"),
            name=_text(data["name"], f"{path}.name"),
            description=_text(data["description"], f"{path}.description"),
            sha256_env=_optional_text(data.get("sha256_env"), f"{path}.sha256_env"),
        )


@dataclass(frozen=True, slots=True)
class ExperimentBenchmarkTarget:
    """One benchmark split plus an exact or operator-supplied unit inventory."""

    cell_id: str
    split: str
    unit_ids: tuple[str, ...] = ()
    unit_inventory_env: str | None = None
    unit_inventory_sha256_env: str | None = None
    external: bool = False
    source_revision: str | None = None
    arm_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.cell_id, label="benchmark cell_id")
        _identity(self.split, label="benchmark split")
        units = tuple(self.unit_ids)
        if any(not isinstance(item, str) or not item.strip() for item in units):
            raise ValidationError("benchmark unit IDs must be non-empty strings")
        if len(units) != len(set(units)):
            raise ValidationError("benchmark unit IDs must be unique")
        if bool(units) == bool(self.unit_inventory_env):
            raise ValidationError(
                "benchmark target requires exactly one of unit_ids or unit_inventory_env"
            )
        if self.unit_inventory_env is not None and not _ENVIRONMENT_NAME.fullmatch(
            self.unit_inventory_env
        ):
            raise ValidationError("unit_inventory_env must be an uppercase identifier")
        if self.unit_inventory_sha256_env is not None:
            if self.unit_inventory_env is None:
                raise ValidationError("unit_inventory_sha256_env requires unit_inventory_env")
            if not _ENVIRONMENT_NAME.fullmatch(self.unit_inventory_sha256_env):
                raise ValidationError("unit_inventory_sha256_env must be an uppercase identifier")
        if not isinstance(self.external, bool):
            raise ValidationError("benchmark external flag must be boolean")
        if self.external and (
            not isinstance(self.source_revision, str) or not self.source_revision.strip()
        ):
            raise ValidationError("external benchmark targets require source_revision")
        if not self.external and self.source_revision is not None:
            raise ValidationError("registered benchmark revisions come from the benchmark registry")
        arm_ids = tuple(self.arm_ids)
        if arm_ids:
            if len(arm_ids) < 2:
                raise ValidationError(
                    "benchmark-specific arm_ids require at least two comparison arms"
                )
            if len(arm_ids) != len(set(arm_ids)):
                raise ValidationError("benchmark-specific arm_ids must be unique")
            for arm_id in arm_ids:
                _identity(arm_id, label="benchmark-specific arm_id")
        object.__setattr__(self, "unit_ids", units)
        object.__setattr__(self, "arm_ids", arm_ids)

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> ExperimentBenchmarkTarget:
        data = _strict_mapping(
            value,
            path=path,
            keys={
                "cell_id",
                "split",
                "unit_ids",
                "unit_inventory_env",
                "unit_inventory_sha256_env",
                "external",
                "source_revision",
                "arm_ids",
            },
            required={"cell_id", "split"},
        )
        return cls(
            cell_id=_text(data["cell_id"], f"{path}.cell_id"),
            split=_text(data["split"], f"{path}.split"),
            unit_ids=_string_tuple(data.get("unit_ids", []), f"{path}.unit_ids"),
            unit_inventory_env=_optional_text(
                data.get("unit_inventory_env"),
                f"{path}.unit_inventory_env",
            ),
            unit_inventory_sha256_env=_optional_text(
                data.get("unit_inventory_sha256_env"),
                f"{path}.unit_inventory_sha256_env",
            ),
            external=_boolean(data.get("external", False), f"{path}.external"),
            source_revision=_optional_text(
                data.get("source_revision"),
                f"{path}.source_revision",
            ),
            arm_ids=_string_tuple(data.get("arm_ids", []), f"{path}.arm_ids"),
        )


@dataclass(frozen=True, slots=True)
class ExperimentArmSpec:
    arm_id: str
    release_id: str
    family: SemanticFamily
    profile_kind: ProfileKind
    arm_kind: ExperimentArmKind = ExperimentArmKind.RELEASE
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identity(self.arm_id, label="experiment arm_id")
        _identity(self.release_id, label="experiment release_id")
        if not isinstance(self.family, SemanticFamily):
            raise ValidationError("experiment arm family must be typed")
        if not isinstance(self.profile_kind, ProfileKind):
            raise ValidationError("experiment arm profile_kind must be typed")
        if not isinstance(self.arm_kind, ExperimentArmKind):
            raise ValidationError("experiment arm_kind must be typed")
        if self.arm_kind is ExperimentArmKind.POLICY:
            if self.family is not SemanticFamily.COMPOSITE:
                raise ValidationError("experiment policy arms must use family=composite")
            if not self.release_id.startswith("policy."):
                raise ValidationError("experiment policy arm identity must start with policy.")
        elif self.release_id.startswith("policy."):
            raise ValidationError("policy.* identities require arm_kind=policy")
        try:
            frozen = freeze_mapping(self.parameters)
        except (TypeError, ValueError) as exc:
            raise ValidationError("experiment arm parameters must be canonical JSON") from exc
        object.__setattr__(self, "parameters", frozen)

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> ExperimentArmSpec:
        data = _strict_mapping(
            value,
            path=path,
            keys={
                "arm_id",
                "release_id",
                "family",
                "profile_kind",
                "arm_kind",
                "parameters",
            },
            required={"arm_id", "release_id", "family", "profile_kind"},
        )
        parameters = data.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValidationError(f"{path}.parameters must be an object")
        return cls(
            arm_id=_text(data["arm_id"], f"{path}.arm_id"),
            release_id=_text(data["release_id"], f"{path}.release_id"),
            family=_enum(SemanticFamily, data["family"], f"{path}.family"),
            profile_kind=_enum(ProfileKind, data["profile_kind"], f"{path}.profile_kind"),
            arm_kind=_enum(
                ExperimentArmKind,
                data.get("arm_kind", ExperimentArmKind.RELEASE.value),
                f"{path}.arm_kind",
            ),
            parameters=parameters,
        )


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Frozen operator intent for one experiment family."""

    schema_version: int
    study_id: str
    stage: ExperimentStage
    estimand_profile_kind: ProfileKind
    benchmarks: tuple[ExperimentBenchmarkTarget, ...]
    arms: tuple[ExperimentArmSpec, ...]
    seeds: tuple[int, ...]
    model_slots: tuple[str, ...] = ("default",)
    driver_entrypoint: str | None = None
    prerequisites: tuple[ExperimentPrerequisite, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported experiment manifest schema_version")
        _identity(self.study_id, label="experiment study_id")
        if not isinstance(self.stage, ExperimentStage):
            raise ValidationError("experiment stage must be typed")
        if not isinstance(self.estimand_profile_kind, ProfileKind):
            raise ValidationError("estimand_profile_kind must be typed")
        benchmarks = tuple(self.benchmarks)
        arms = tuple(self.arms)
        seeds = tuple(self.seeds)
        model_slots = tuple(self.model_slots)
        prerequisites = tuple(self.prerequisites)
        if not benchmarks or not all(
            isinstance(item, ExperimentBenchmarkTarget) for item in benchmarks
        ):
            raise ValidationError("experiment manifest requires benchmark targets")
        if len({(item.cell_id, item.split) for item in benchmarks}) != len(benchmarks):
            raise ValidationError("experiment benchmark target identities must be unique")
        if len(arms) < 2 or not all(isinstance(item, ExperimentArmSpec) for item in arms):
            raise ValidationError("experiment manifest requires at least two arms")
        if len({item.arm_id for item in arms}) != len(arms):
            raise ValidationError("experiment arm IDs must be unique")
        declared_arm_ids = {item.arm_id for item in arms}
        for benchmark in benchmarks:
            unknown_arm_ids = set(benchmark.arm_ids) - declared_arm_ids
            if unknown_arm_ids:
                raise ValidationError(
                    f"benchmark {benchmark.cell_id} references unknown arm IDs: "
                    + ", ".join(sorted(unknown_arm_ids))
                )
        mixed = {item.profile_kind for item in arms}
        if mixed != {self.estimand_profile_kind}:
            raise ValidationError(
                "one experiment manifest cannot mix official_native, controlled_adapter, "
                "or factorial_adapter estimands"
            )
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValidationError("experiment seeds must be non-negative integers")
        if len(seeds) != len(set(seeds)):
            raise ValidationError("experiment seeds must be unique")
        if (
            not model_slots
            or len(model_slots) != len(set(model_slots))
            or any(
                not isinstance(model_slot, str) or not model_slot.strip()
                for model_slot in model_slots
            )
        ):
            raise ValidationError("experiment model_slots must be unique non-empty identifiers")
        for model_slot in model_slots:
            _identity(model_slot, label="experiment model_slot")
        if self.driver_entrypoint is not None and not _ENTRYPOINT.fullmatch(self.driver_entrypoint):
            raise ValidationError("driver_entrypoint must use module:attribute syntax")
        if not all(isinstance(item, ExperimentPrerequisite) for item in prerequisites):
            raise ValidationError("experiment prerequisites must be typed")
        if len({(item.kind, item.name) for item in prerequisites}) != len(prerequisites):
            raise ValidationError("experiment prerequisites must be unique")
        try:
            metadata = freeze_mapping(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValidationError("experiment metadata must be canonical JSON") from exc
        object.__setattr__(self, "benchmarks", benchmarks)
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "model_slots", model_slots)
        object.__setattr__(self, "prerequisites", prerequisites)
        object.__setattr__(self, "metadata", metadata)

    @property
    def manifest_hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_mapping(cls, value: Any) -> ExperimentManifest:
        path = "manifest"
        data = _strict_mapping(
            value,
            path=path,
            keys={
                "schema_version",
                "study_id",
                "stage",
                "estimand_profile_kind",
                "benchmarks",
                "arms",
                "seeds",
                "model_slots",
                "driver_entrypoint",
                "prerequisites",
                "metadata",
            },
            required={
                "schema_version",
                "study_id",
                "stage",
                "estimand_profile_kind",
                "benchmarks",
                "arms",
                "seeds",
            },
        )
        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValidationError("manifest.schema_version must be an integer")
        benchmarks = _sequence(data["benchmarks"], "manifest.benchmarks")
        arms = _sequence(data["arms"], "manifest.arms")
        seeds = _sequence(data["seeds"], "manifest.seeds")
        model_slots = _sequence(
            data.get("model_slots", ["default"]),
            "manifest.model_slots",
        )
        prerequisites = _sequence(
            data.get("prerequisites", []),
            "manifest.prerequisites",
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("manifest.metadata must be an object")
        return cls(
            schema_version=schema_version,
            study_id=_text(data["study_id"], "manifest.study_id"),
            stage=_enum(ExperimentStage, data["stage"], "manifest.stage"),
            estimand_profile_kind=_enum(
                ProfileKind,
                data["estimand_profile_kind"],
                "manifest.estimand_profile_kind",
            ),
            benchmarks=tuple(
                ExperimentBenchmarkTarget.from_mapping(
                    item,
                    path=f"manifest.benchmarks[{index}]",
                )
                for index, item in enumerate(benchmarks)
            ),
            arms=tuple(
                ExperimentArmSpec.from_mapping(
                    item,
                    path=f"manifest.arms[{index}]",
                )
                for index, item in enumerate(arms)
            ),
            seeds=tuple(
                _integer(item, f"manifest.seeds[{index}]") for index, item in enumerate(seeds)
            ),
            model_slots=tuple(
                _text(item, f"manifest.model_slots[{index}]")
                for index, item in enumerate(model_slots)
            ),
            driver_entrypoint=_optional_text(
                data.get("driver_entrypoint"),
                "manifest.driver_entrypoint",
            ),
            prerequisites=tuple(
                ExperimentPrerequisite.from_mapping(
                    item,
                    path=f"manifest.prerequisites[{index}]",
                )
                for index, item in enumerate(prerequisites)
            ),
            metadata=metadata,
        )


def _strict_mapping(
    value: Any,
    *,
    path: str,
    keys: set[str],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    unknown = set(value) - keys
    missing = required - set(value)
    if unknown:
        raise ValidationError(f"{path} contains unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValidationError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{path} must be an array")
    return value


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    return tuple(
        _text(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{path} must be a trimmed non-empty string")
    return value


def _optional_text(value: Any, path: str) -> str | None:
    return None if value is None else _text(value, path)


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{path} must be an integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{path} must be a boolean")
    return value


def _enum(enum_type: type[Any], value: Any, path: str) -> Any:
    if not isinstance(value, str):
        raise ValidationError(f"{path} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValidationError(f"{path} must be one of: {allowed}") from exc


def _identity(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValidationError(f"{label} contains unsupported characters")


def require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "ExperimentArmKind",
    "ExperimentArmSpec",
    "ExperimentBenchmarkTarget",
    "ExperimentManifest",
    "ExperimentPrerequisite",
    "ExperimentStage",
    "PrerequisiteKind",
    "require_sha256",
]
