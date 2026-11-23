"""Strict registry-backed construction for static prompt interventions.

The builder is an assembly boundary, not a reimplementation of any method.
It resolves the exact entrypoint already declared by the algorithm registry,
constructs that object with checksum-locked JSON parameters. Executable
dependencies are deliberately *not* stored on the intervention: the static
driver supplies the shared runtime graph through :class:`InterventionContext`
when the intervention is executed.

Full-system/composite and offline-recipe releases are intentionally rejected:
they must run through their process or atomic-arena execution profile instead
of being mislabeled as a one-shot prompt intervention.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.lifecycle import Intervention
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.experiments.drivers.static_production import StaticArmBuildRequest
from agentinputlab.experiments.secrets import require_secret_free
from agentinputlab.registry import AlgorithmRegistry
from agentinputlab.selectors import NoOpIntervention

REGISTRY_INTERVENTION_BUILDER_PROTOCOL = "agentinputlab.registry-intervention-builder.v1"

_SCHEMA_VERSION = 1
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_IDENTITY_ARGUMENTS = frozenset({"method_id", "release_id", "family"})
_SAFE_OBJECT_PREFIXES = (
    "agentinputlab.algorithms.",
    "agentinputlab.recipes.",
    "agentinputlab.integrations.external_workers.",
)
_PARAMETER_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "constructor_kwargs",
        "runtime_bindings",
        "arm_parameter_bindings",
        "required_environment",
    }
)
_TYPED_SPEC_KINDS = frozenset({"dataclass", "enum", "path", "tuple"})


class RegistryInterventionBuilder:
    """Build an unchanged registered intervention under explicit fair settings."""

    builder_kind = "intervention"

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": REGISTRY_INTERVENTION_BUILDER_PROTOCOL,
                "schema_version": _SCHEMA_VERSION,
                "registry_source": "packaged",
                "identity_injection": sorted(_IDENTITY_ARGUMENTS),
                "no_op_profile_policy": "inherit_requested_profile.v1",
                "runtime_dependency_policy": "intervention_context_only.v1",
                "typed_spec_kinds": sorted(_TYPED_SPEC_KINDS),
                "composite_policy": "reject_and_route_to_process_or_atomic_arena",
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        """Validate a frozen construction recipe without loading model assets."""

        try:
            specification = _parse_parameters(parameters)
            if specification["release_id"] == "baseline.no_op":
                return ()
            _, _, constructor = _resolve_release(specification["release_id"])
            blockers = list(_constructor_blockers(constructor, specification))
            blockers.extend(
                f"missing required environment {name}"
                for name in specification["required_environment"]
                if not os.environ.get(name)
            )
            return tuple(blockers)
        except (ArtifactError, ValidationError) as exc:
            return (str(exc),)

    def build(self, request: StaticArmBuildRequest) -> Intervention:
        """Construct one exact release using the request's shared runtime graph."""

        if not isinstance(request, StaticArmBuildRequest):
            raise TypeError("registry intervention builder requires StaticArmBuildRequest")
        specification = _parse_parameters(request.builder_parameters)
        configured_release = specification["release_id"]
        if configured_release != request.release_id:
            raise ValidationError(
                "registry intervention builder release differs from the requested arm"
            )
        if request.release_id == "baseline.no_op":
            if request.family is not SemanticFamily.NO_OP:
                raise ValidationError("baseline.no_op requires the no_op family")
            return NoOpIntervention(profile_kind=request.profile_kind.value)

        method, release, constructor = _resolve_release(request.release_id)
        if method.semantic_family is not request.family:
            raise ValidationError("registry intervention builder family differs from the registry")
        if request.profile_kind not in release.profile_modes:
            raise ValidationError("requested profile kind is not declared by the selected release")
        blockers = _constructor_blockers(constructor, specification)
        if blockers:
            raise ArtifactError("; ".join(blockers))

        try:
            kwargs = {
                name: _materialize_value(value)
                for name, value in specification["constructor_kwargs"].items()
            }
            for argument, arm_parameter in specification["arm_parameter_bindings"].items():
                if arm_parameter not in request.arm_parameters:
                    raise ArtifactError(
                        f"arm parameter {arm_parameter!r} required for constructor "
                        f"argument {argument!r} is missing"
                    )
                kwargs[argument] = _materialize_value(request.arm_parameters[arm_parameter])
        except ArtifactError:
            raise
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"cannot materialize constructor values for {request.release_id!r}: {exc}"
            ) from exc

        signature = inspect.signature(constructor)
        implementation_method_id = method.method_id.rsplit(".", 1)[-1]
        identity_values = {
            "method_id": implementation_method_id,
            "release_id": release.id,
            "family": method.semantic_family,
        }
        for name, value in identity_values.items():
            if name in signature.parameters:
                configured = kwargs.get(name, value)
                if configured != value:
                    raise ValidationError(
                        f"constructor identity argument {name!r} differs from the registry"
                    )
                kwargs[name] = value
        try:
            intervention = constructor(**kwargs)
        except TypeError as exc:
            raise ValidationError(
                f"cannot construct registered release {request.release_id!r}: {exc}"
            ) from exc
        if not isinstance(intervention, Intervention):
            raise ValidationError(
                f"release {request.release_id!r} is not a one-shot Intervention; "
                "use its process or atomic-arena execution profile"
            )
        if intervention.method_id != implementation_method_id:
            raise ValidationError("constructed intervention method_id differs from the registry")
        if intervention.release_id != release.id:
            raise ValidationError("constructed intervention release_id differs from the registry")
        if intervention.family is not method.semantic_family:
            raise ValidationError("constructed intervention family differs from the registry")
        return intervention


def registry_intervention_builder_factory() -> RegistryInterventionBuilder:
    """Return the zero-argument builder used by locked production configs."""

    return RegistryInterventionBuilder()


def _parse_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(parameters, Mapping):
        raise ValidationError("registry intervention parameters must be an object")
    if not set(parameters).issubset(_PARAMETER_FIELDS):
        unknown = sorted(set(parameters) - _PARAMETER_FIELDS)
        raise ValidationError(f"registry intervention parameters contain unknown fields: {unknown}")
    if parameters.get("schema_version") != _SCHEMA_VERSION:
        raise ValidationError("unsupported registry intervention schema_version")
    release_id = parameters.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        raise ValidationError("registry intervention release_id must be non-empty")

    constructor_kwargs = _string_mapping(
        parameters.get("constructor_kwargs", {}),
        label="constructor_kwargs",
    )
    runtime_bindings = _string_mapping(
        parameters.get("runtime_bindings", {}),
        label="runtime_bindings",
        values_must_be_strings=True,
    )
    if runtime_bindings:
        raise ValidationError(
            "runtime_bindings are forbidden: executable dependencies must be supplied "
            "through InterventionContext by the experiment driver"
        )
    arm_bindings = _string_mapping(
        parameters.get("arm_parameter_bindings", {}),
        label="arm_parameter_bindings",
        values_must_be_strings=True,
    )
    targets = (
        set(constructor_kwargs),
        set(runtime_bindings),
        set(arm_bindings),
    )
    if targets[0] & targets[1] or targets[0] & targets[2] or targets[1] & targets[2]:
        raise ValidationError(
            "constructor, runtime, and arm-parameter bindings must target distinct arguments"
        )
    for argument in set().union(*targets):
        _constructor_argument(argument)
    for name in arm_bindings.values():
        if not name.strip() or name != name.strip():
            raise ValidationError("arm parameter binding names must be trimmed and non-empty")

    raw_environment = parameters.get("required_environment", ())
    if isinstance(raw_environment, (str, bytes)) or not isinstance(
        raw_environment,
        Sequence,
    ):
        raise ValidationError("required_environment must be an array")
    required_environment = tuple(raw_environment)
    if len(required_environment) != len(set(required_environment)) or any(
        not isinstance(name, str) or _ENVIRONMENT_NAME.fullmatch(name) is None
        for name in required_environment
    ):
        raise ValidationError("required_environment must contain unique environment-variable names")
    require_secret_free(parameters, label="registry intervention parameters")
    for value in constructor_kwargs.values():
        _validate_typed_value(value)
    result = {
        "schema_version": _SCHEMA_VERSION,
        "release_id": release_id,
        "constructor_kwargs": constructor_kwargs,
        "runtime_bindings": runtime_bindings,
        "arm_parameter_bindings": arm_bindings,
        "required_environment": required_environment,
    }
    try:
        return freeze_mapping(result)
    except (TypeError, ValueError) as exc:
        raise ValidationError("registry intervention parameters must be canonical JSON") from exc


def _resolve_release(release_id: str) -> tuple[Any, Any, Any]:
    registry = AlgorithmRegistry.default()
    try:
        method, release = registry.get_release(release_id)
        constructor = registry.resolve_entrypoint(release_id)
    except Exception as exc:
        raise ArtifactError(f"cannot resolve registered release {release_id!r}: {exc}") from exc
    if not callable(constructor):
        raise ArtifactError(f"registered release {release_id!r} is not callable")
    return method, release, constructor


def _constructor_blockers(
    constructor: Any,
    specification: Mapping[str, Any],
) -> tuple[str, ...]:
    if not hasattr(constructor, "propose"):
        return (
            f"release {specification['release_id']!r} is composite/offline rather "
            "than a one-shot intervention; use method_process or external_arena",
        )
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("registered intervention has no inspectable signature") from exc
    supplied = (
        set(specification["constructor_kwargs"])
        | set(specification["runtime_bindings"])
        | set(specification["arm_parameter_bindings"])
        | (set(signature.parameters) & _IDENTITY_ARGUMENTS)
    )
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    blockers: list[str] = []
    for name in supplied:
        parameter = signature.parameters.get(name)
        if parameter is None and not accepts_kwargs:
            blockers.append(f"constructor does not accept configured argument {name!r}")
        elif parameter is not None and parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            blockers.append(f"constructor argument {name!r} is positional-only")
    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is inspect.Parameter.empty and name not in supplied:
            blockers.append(f"required constructor argument {name!r} has no frozen binding")
    return tuple(blockers)


def _string_mapping(
    value: Any,
    *,
    label: str,
    values_must_be_strings: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        _constructor_argument(key)
        if values_must_be_strings and (
            not isinstance(item, str) or not item.strip() or item != item.strip()
        ):
            raise ValidationError(f"{label} values must be trimmed non-empty strings")
        result[key] = item
    return freeze_mapping(result)


def _constructor_argument(name: Any) -> None:
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValidationError("constructor argument names must be public Python identifiers")


def _validate_typed_value(value: Any) -> None:
    if isinstance(value, Mapping):
        if "$type" in value:
            kind = value.get("$type")
            if kind not in _TYPED_SPEC_KINDS:
                raise ValidationError(f"unsupported typed constructor value {kind!r}")
            _validate_typed_spec(value, kind=kind)
            return
        for item in value.values():
            _validate_typed_value(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _validate_typed_value(item)


def _validate_typed_spec(value: Mapping[str, Any], *, kind: str) -> None:
    if kind == "dataclass":
        if set(value) != {"$type", "entrypoint", "kwargs"}:
            raise ValidationError("dataclass typed values require $type, entrypoint, and kwargs")
        candidate = _resolve_safe_type(value["entrypoint"])
        if not dataclasses.is_dataclass(candidate):
            raise ValidationError("dataclass typed value entrypoint is not a dataclass type")
        kwargs = _string_mapping(value["kwargs"], label="dataclass kwargs")
        for item in kwargs.values():
            _validate_typed_value(item)
        _signature_accepts(candidate, set(kwargs), label="dataclass typed value")
        return
    if kind == "enum":
        if set(value) != {"$type", "entrypoint", "value"}:
            raise ValidationError("enum typed values require $type, entrypoint, and value")
        candidate = _resolve_safe_type(value["entrypoint"])
        if not inspect.isclass(candidate) or not issubclass(candidate, Enum):
            raise ValidationError("enum typed value entrypoint is not an Enum type")
        try:
            candidate(value["value"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("enum typed value is not a declared member value") from exc
        return
    if kind == "path":
        if set(value) != {"$type", "value"}:
            raise ValidationError("path typed values require $type and value")
        raw = value["value"]
        if not isinstance(raw, str) or not raw.strip():
            raise ValidationError("path typed value must be non-empty")
        if not Path(raw).expanduser().is_absolute():
            raise ValidationError("path typed value must be absolute")
        return
    if kind == "tuple":
        if set(value) != {"$type", "items"}:
            raise ValidationError("tuple typed values require $type and items")
        items = value["items"]
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise ValidationError("tuple typed value items must be an array")
        for item in items:
            _validate_typed_value(item)


def _resolve_safe_type(entrypoint: Any) -> Any:
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise ValidationError("typed value entrypoint must use module:attribute syntax")
    module_name, attribute_path = entrypoint.split(":", 1)
    if not module_name.startswith(_SAFE_OBJECT_PREFIXES):
        raise ValidationError("typed value entrypoint is outside the algorithm allowlist")
    if any(
        not segment.isidentifier() or segment.startswith("_")
        for segment in (module_name.split(".") + attribute_path.split("."))
    ):
        raise ValidationError("typed value entrypoint contains a private or invalid segment")
    try:
        candidate: Any = importlib.import_module(module_name)
        for segment in attribute_path.split("."):
            candidate = getattr(candidate, segment)
    except (ImportError, AttributeError) as exc:
        raise ArtifactError(f"cannot resolve typed value entrypoint {entrypoint!r}") from exc
    if not inspect.isclass(candidate):
        raise ValidationError("typed value entrypoint must resolve to a class")
    return candidate


def _signature_accepts(candidate: Any, names: set[str], *, label: str) -> None:
    signature = inspect.signature(candidate)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    unknown = sorted(set(names) - set(signature.parameters))
    if unknown and not accepts_kwargs:
        raise ValidationError(f"{label} has unsupported arguments: {unknown}")
    missing = sorted(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        and parameter.default is inspect.Parameter.empty
        and name not in names
    )
    if missing:
        raise ValidationError(f"{label} is missing required arguments: {missing}")


def _materialize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        kind = value.get("$type")
        if kind is None:
            return {key: _materialize_value(item) for key, item in value.items()}
        _validate_typed_spec(value, kind=kind)
        if kind == "dataclass":
            candidate = _resolve_safe_type(value["entrypoint"])
            kwargs = {key: _materialize_value(item) for key, item in value["kwargs"].items()}
            return candidate(**kwargs)
        if kind == "enum":
            return _resolve_safe_type(value["entrypoint"])(value["value"])
        if kind == "path":
            return Path(value["value"]).expanduser()
        if kind == "tuple":
            return tuple(_materialize_value(item) for item in value["items"])
        raise ValidationError(f"unsupported typed constructor value {kind!r}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_materialize_value(item) for item in value]
    return value


__all__ = [
    "REGISTRY_INTERVENTION_BUILDER_PROTOCOL",
    "RegistryInterventionBuilder",
    "registry_intervention_builder_factory",
]
