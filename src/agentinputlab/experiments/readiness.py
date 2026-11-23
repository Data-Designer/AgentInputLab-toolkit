"""Strict, claim-safe catalogs for method execution readiness.

These catalogs separate three facts that must not be conflated:

* a method exists in the runtime registry;
* a real benchmark/parity adapter has been implemented and reviewed;
* paper-result parity has a method-specific, frozen acceptance rule.

The default repository catalogs are intentionally allowed to say ``blocked``.
They are operator gates, not optimistic implementation inventories.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash

ACCEPTANCE_PROTOCOL = "agentinputlab.parity-acceptance-catalog.v1"
ADAPTER_READINESS_PROTOCOL = "agentinputlab.adapter-readiness-matrix.v1"

_ACCEPTANCE_STATUSES = frozenset({"blocked", "specified"})
_ADAPTER_STATUSES = frozenset({"blocked", "implemented"})
_INTEGRATION_KINDS = frozenset(
    {
        "dual_parity_adapter",
        "method_process_adapter",
        "atomic_external_bridge",
    }
)
_COMPARISONS = frozenset(
    {
        "absolute_delta",
        "relative_delta",
        "non_inferiority",
        "exact_match",
    }
)
_AGGREGATIONS = frozenset({"mean", "median", "micro", "macro", "exact_rate"})


@dataclass(frozen=True, slots=True)
class ParityAcceptanceSpec:
    """One method's frozen parity-analysis gate."""

    method_id: str
    status: str
    blockers: tuple[str, ...] = ()
    fixture_id: str | None = None
    metric_name: str | None = None
    aggregation: str | None = None
    comparison: str | None = None
    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None
    minimum_units: int | None = None
    reference_source: str | None = None

    @property
    def acceptance_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class AdapterGate:
    """Readiness of one concrete integration surface."""

    status: str
    integration_kind: str
    blockers: tuple[str, ...]

    @property
    def gate_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class MethodAdapterReadiness:
    """Parity and discovery readiness for one catalogued method."""

    method_id: str
    family: str
    profile_kind: str
    parity: AdapterGate
    discovery: AdapterGate

    @property
    def readiness_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ReadinessCatalog:
    """An exact-set catalog plus its byte and semantic identities."""

    path: Path
    file_sha256: str
    catalog_hash: str
    entries: Mapping[str, Any]


def load_parity_acceptance_catalog(
    path: str | Path,
    *,
    expected_method_ids: Sequence[str] | None = None,
) -> ReadinessCatalog:
    """Load an exact, strict parity acceptance catalog."""

    source, raw, file_sha256 = _load_json(path, label="parity acceptance catalog")
    data = _strict_mapping(
        raw,
        {"schema_version", "protocol", "method_count", "methods"},
        label="parity acceptance catalog",
    )
    if data["schema_version"] != 1 or data["protocol"] != ACCEPTANCE_PROTOCOL:
        raise ValidationError("unsupported parity acceptance catalog")
    methods = _method_array(data, label="parity acceptance catalog")
    entries: dict[str, ParityAcceptanceSpec] = {}
    for index, item in enumerate(methods):
        label = f"parity acceptance methods[{index}]"
        base = _mapping(item, label=label)
        status = _choice(base.get("status"), _ACCEPTANCE_STATUSES, f"{label}.status")
        if status == "blocked":
            values = _strict_mapping(
                base,
                {"method_id", "status", "blockers"},
                label=label,
            )
            spec = ParityAcceptanceSpec(
                method_id=_text(values["method_id"], f"{label}.method_id"),
                status=status,
                blockers=_blockers(values["blockers"], f"{label}.blockers"),
            )
        else:
            values = _strict_mapping(
                base,
                {
                    "method_id",
                    "status",
                    "blockers",
                    "fixture_id",
                    "metric_name",
                    "aggregation",
                    "comparison",
                    "absolute_tolerance",
                    "relative_tolerance",
                    "minimum_units",
                    "reference_source",
                },
                label=label,
            )
            blockers = _blockers(
                values["blockers"],
                f"{label}.blockers",
                allow_empty=True,
            )
            if blockers:
                raise ValidationError(f"{label}.blockers must be empty when specified")
            comparison = _choice(
                values["comparison"],
                _COMPARISONS,
                f"{label}.comparison",
            )
            absolute = _optional_nonnegative_number(
                values["absolute_tolerance"],
                f"{label}.absolute_tolerance",
            )
            relative = _optional_nonnegative_number(
                values["relative_tolerance"],
                f"{label}.relative_tolerance",
            )
            if comparison == "exact_match":
                if absolute not in {None, 0.0} or relative not in {None, 0.0}:
                    raise ValidationError(f"{label}: exact_match cannot have a positive tolerance")
            elif absolute is None and relative is None:
                raise ValidationError(
                    f"{label}: a non-exact comparison requires at least one tolerance"
                )
            spec = ParityAcceptanceSpec(
                method_id=_text(values["method_id"], f"{label}.method_id"),
                status=status,
                blockers=(),
                fixture_id=_text(values["fixture_id"], f"{label}.fixture_id"),
                metric_name=_text(values["metric_name"], f"{label}.metric_name"),
                aggregation=_choice(
                    values["aggregation"],
                    _AGGREGATIONS,
                    f"{label}.aggregation",
                ),
                comparison=comparison,
                absolute_tolerance=absolute,
                relative_tolerance=relative,
                minimum_units=_positive_integer(
                    values["minimum_units"],
                    f"{label}.minimum_units",
                ),
                reference_source=_text(
                    values["reference_source"],
                    f"{label}.reference_source",
                ),
            )
        if spec.method_id in entries:
            raise ValidationError("parity acceptance catalog contains duplicate methods")
        entries[spec.method_id] = spec
    _require_expected_method_set(entries, expected_method_ids, label="parity acceptance catalog")
    return ReadinessCatalog(
        path=source,
        file_sha256=file_sha256,
        catalog_hash=content_hash(raw),
        entries=MappingProxyType(entries),
    )


def load_adapter_readiness_matrix(
    path: str | Path,
    *,
    expected_methods: Mapping[str, Mapping[str, str]] | None = None,
) -> ReadinessCatalog:
    """Load the exact adapter/bridge readiness matrix."""

    source, raw, file_sha256 = _load_json(path, label="adapter readiness matrix")
    data = _strict_mapping(
        raw,
        {"schema_version", "protocol", "method_count", "methods"},
        label="adapter readiness matrix",
    )
    if data["schema_version"] != 1 or data["protocol"] != ADAPTER_READINESS_PROTOCOL:
        raise ValidationError("unsupported adapter readiness matrix")
    methods = _method_array(data, label="adapter readiness matrix")
    entries: dict[str, MethodAdapterReadiness] = {}
    for index, item in enumerate(methods):
        label = f"adapter readiness methods[{index}]"
        values = _strict_mapping(
            item,
            {"method_id", "family", "profile_kind", "parity", "discovery"},
            label=label,
        )
        method_id = _text(values["method_id"], f"{label}.method_id")
        family = _text(values["family"], f"{label}.family")
        profile_kind = _text(values["profile_kind"], f"{label}.profile_kind")
        readiness = MethodAdapterReadiness(
            method_id=method_id,
            family=family,
            profile_kind=profile_kind,
            parity=_adapter_gate(
                values["parity"],
                label=f"{label}.parity",
                expected_kind="dual_parity_adapter",
            ),
            discovery=_adapter_gate(
                values["discovery"],
                label=f"{label}.discovery",
                expected_kind=(
                    "atomic_external_bridge"
                    if family in {"clarify", "composite"}
                    else "method_process_adapter"
                ),
            ),
        )
        if method_id in entries:
            raise ValidationError("adapter readiness matrix contains duplicate methods")
        if expected_methods is not None:
            expected = expected_methods.get(method_id)
            if expected is None:
                raise ValidationError(
                    f"adapter readiness method {method_id!r} is outside the method catalog"
                )
            if family != expected["family"] or profile_kind != expected["profile_kind"]:
                raise ValidationError(
                    f"adapter readiness method {method_id!r} differs from the method catalog"
                )
        entries[method_id] = readiness
    expected_ids = tuple(expected_methods) if expected_methods is not None else None
    _require_expected_method_set(entries, expected_ids, label="adapter readiness matrix")
    return ReadinessCatalog(
        path=source,
        file_sha256=file_sha256,
        catalog_hash=content_hash(raw),
        entries=MappingProxyType(entries),
    )


def _adapter_gate(
    value: Any,
    *,
    label: str,
    expected_kind: str,
) -> AdapterGate:
    data = _strict_mapping(
        value,
        {"status", "integration_kind", "blockers"},
        label=label,
    )
    status = _choice(data["status"], _ADAPTER_STATUSES, f"{label}.status")
    kind = _choice(
        data["integration_kind"],
        _INTEGRATION_KINDS,
        f"{label}.integration_kind",
    )
    if kind != expected_kind:
        raise ValidationError(f"{label}.integration_kind must be {expected_kind!r}")
    blockers = _blockers(
        data["blockers"],
        f"{label}.blockers",
        allow_empty=status == "implemented",
    )
    if status == "implemented" and blockers:
        raise ValidationError(f"{label}.blockers must be empty when implemented")
    return AdapterGate(status=status, integration_kind=kind, blockers=blockers)


def _load_json(path: str | Path, *, label: str) -> tuple[Path, Any, str]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ArtifactError(f"{label} must not be a symlink")
    source = unresolved.resolve()
    if not source.is_file():
        raise ArtifactError(f"{label} is unavailable")
    try:
        payload = source.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    return source, raw, hashlib.sha256(payload).hexdigest()


def _method_array(data: Mapping[str, Any], *, label: str) -> tuple[Any, ...]:
    count = data["method_count"]
    methods = data["methods"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(methods, (str, bytes))
        or not isinstance(methods, Sequence)
        or len(methods) != count
    ):
        raise ValidationError(f"{label} method_count/methods are inconsistent")
    return tuple(methods)


def _require_expected_method_set(
    entries: Mapping[str, Any],
    expected_method_ids: Sequence[str] | None,
    *,
    label: str,
) -> None:
    if expected_method_ids is None:
        return
    expected = set(expected_method_ids)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise ValidationError(f"{label} method set differs; missing={missing}, extra={extra}")


def _strict_mapping(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    data = _mapping(value, label=label)
    if set(data) != fields:
        raise ValidationError(f"{label} fields are invalid")
    return data


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _choice(value: Any, choices: frozenset[str], label: str) -> str:
    text = _text(value, label)
    if text not in choices:
        raise ValidationError(f"{label} must be one of {sorted(choices)}")
    return text


def _blockers(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be a string array")
    blockers = tuple(_text(item, f"{label} item") for item in value)
    if (not blockers and not allow_empty) or len(blockers) != len(set(blockers)):
        raise ValidationError(f"{label} must contain unique blocker strings")
    return blockers


def _optional_nonnegative_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValidationError(f"{label} must be null or a finite non-negative number")
    return float(value)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


__all__ = [
    "ACCEPTANCE_PROTOCOL",
    "ADAPTER_READINESS_PROTOCOL",
    "AdapterGate",
    "MethodAdapterReadiness",
    "ParityAcceptanceSpec",
    "ReadinessCatalog",
    "load_adapter_readiness_matrix",
    "load_parity_acceptance_catalog",
]
