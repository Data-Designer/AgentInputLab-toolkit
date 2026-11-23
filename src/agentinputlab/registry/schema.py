"""Typed, dependency-free schema for the algorithm registry.

The source registry is deliberately JSON-compatible YAML.  Keeping the schema
in the standard library lets ``agentinputlab list`` and ``doctor`` work in the
minimal installation, before optional model or YAML dependencies are present.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from agentinputlab.contracts.enums import (
    ExecutionKind,
    ImplementationKind,
    ProfileKind,
    ReleaseTier,
    SemanticFamily,
    SupportStatus,
)
from agentinputlab.contracts.errors import RegistryError

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_DELIVERY_TIERS = frozenset({"must_ship", "stretch", "blocked_or_post_v1"})
_EnumT = TypeVar("_EnumT")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{path} must be an object")
    return value


def _required_string(data: Mapping[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{path}.{key} must be a non-empty string")
    return value


def _optional_string(data: Mapping[str, Any], key: str, path: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{path}.{key} must be null or a non-empty string")
    return value


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RegistryError(f"{path} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RegistryError(f"{path}[{index}] must be a non-empty string")
        result.append(item)
    return tuple(result)


def _enum(enum_type: type[_EnumT], value: Any, path: str) -> _EnumT:
    if not isinstance(value, str):
        raise RegistryError(f"{path} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)  # type: ignore[attr-defined]
        raise RegistryError(f"{path} must be one of: {allowed}") from exc


def _enum_tuple(enum_type: type[_EnumT], value: Any, path: str) -> tuple[_EnumT, ...]:
    if not isinstance(value, list):
        raise RegistryError(f"{path} must be an array")
    return tuple(_enum(enum_type, item, f"{path}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class Publication:
    """Publication identity used for citation and roster audits."""

    venue: str
    year: int
    paper_url: str

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> Publication:
        data = _mapping(raw, path)
        year = data.get("year")
        if not isinstance(year, int) or isinstance(year, bool) or year < 2000:
            raise RegistryError(f"{path}.year must be an integer publication year")
        return cls(
            venue=_required_string(data, "venue", path),
            year=year,
            paper_url=_required_string(data, "paper_url", path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"venue": self.venue, "year": self.year, "paper_url": self.paper_url}


@dataclass(frozen=True, slots=True)
class UpstreamSource:
    """Pinned public source identity and source-level redistribution policy."""

    repo_url: str
    commit: str
    license_id: str
    vendoring_allowed: bool
    source_subdir: str | None = None
    license_notes: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> UpstreamSource:
        data = _mapping(raw, path)
        commit = _required_string(data, "commit", path)
        if not _FULL_GIT_SHA.fullmatch(commit):
            raise RegistryError(f"{path}.commit must be a lowercase 40-character Git SHA")
        vendoring_allowed = data.get("vendoring_allowed")
        if not isinstance(vendoring_allowed, bool):
            raise RegistryError(f"{path}.vendoring_allowed must be a boolean")
        license_id = _required_string(data, "license_id", path)
        if license_id == "NOASSERTION" and vendoring_allowed:
            raise RegistryError(
                f"{path}.vendoring_allowed must be false when license_id is NOASSERTION"
            )
        return cls(
            repo_url=_required_string(data, "repo_url", path),
            commit=commit,
            license_id=license_id,
            vendoring_allowed=vendoring_allowed,
            source_subdir=_optional_string(data, "source_subdir", path),
            license_notes=_optional_string(data, "license_notes", path),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "repo_url": self.repo_url,
            "commit": self.commit,
            "license_id": self.license_id,
            "vendoring_allowed": self.vendoring_allowed,
        }
        if self.source_subdir is not None:
            result["source_subdir"] = self.source_subdir
        if self.license_notes is not None:
            result["license_notes"] = self.license_notes
        return result


@dataclass(frozen=True, slots=True)
class ReleaseProfile:
    """One runnable, replay, component, or training identity for a method."""

    id: str
    implementation_kind: ImplementationKind
    execution_kind: ExecutionKind
    status: SupportStatus
    delivery_tier: str
    profile_modes: tuple[ProfileKind, ...]
    entrypoint: str | None
    blocking_reasons: tuple[str, ...]

    @property
    def must_ship(self) -> bool:
        return self.delivery_tier == "must_ship"

    @property
    def runnable(self) -> bool:
        return self.entrypoint is not None and self.status in {
            SupportStatus.EXPERIMENTAL,
            SupportStatus.SUPPORTED,
        }

    @property
    def entrypoint_available(self) -> bool:
        """Whether this release exposes any importable integration entrypoint."""

        return self.entrypoint is not None

    @property
    def availability(self) -> str:
        """Canonical user-facing availability independent of maturity claims."""

        if self.runnable:
            return "runnable"
        if self.entrypoint_available:
            return "gated"
        return "locked"

    @property
    def blocker(self) -> str:
        """Concise reason why a release is not mechanically runnable."""

        if self.runnable:
            return ""
        if self.blocking_reasons:
            return "; ".join(self.blocking_reasons)
        return f"release status is {self.status.value}"

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> ReleaseProfile:
        data = _mapping(raw, path)
        release_id = _required_string(data, "id", path)
        if not _IDENTIFIER.fullmatch(release_id):
            raise RegistryError(f"{path}.id contains unsupported characters")
        delivery_tier = _required_string(data, "delivery_tier", path)
        if delivery_tier not in _DELIVERY_TIERS:
            allowed = ", ".join(sorted(_DELIVERY_TIERS))
            raise RegistryError(f"{path}.delivery_tier must be one of: {allowed}")
        entrypoint = _optional_string(data, "entrypoint", path)
        if entrypoint is not None and ":" not in entrypoint:
            raise RegistryError(f"{path}.entrypoint must use module:attribute syntax")
        profile = cls(
            id=release_id,
            implementation_kind=_enum(
                ImplementationKind,
                data.get("implementation_kind"),
                f"{path}.implementation_kind",
            ),
            execution_kind=_enum(
                ExecutionKind, data.get("execution_kind"), f"{path}.execution_kind"
            ),
            status=_enum(SupportStatus, data.get("status"), f"{path}.status"),
            delivery_tier=delivery_tier,
            profile_modes=_enum_tuple(
                ProfileKind, data.get("profile_modes", []), f"{path}.profile_modes"
            ),
            entrypoint=entrypoint,
            blocking_reasons=_string_tuple(
                data.get("blocking_reasons", []), f"{path}.blocking_reasons"
            ),
        )
        if (
            profile.status
            in {
                SupportStatus.EXPERIMENTAL,
                SupportStatus.SUPPORTED,
            }
            and profile.entrypoint is None
        ):
            raise RegistryError(f"{path}: {profile.status.value} releases require an entrypoint")
        if profile.entrypoint is None and not profile.blocking_reasons:
            raise RegistryError(f"{path}: releases without an entrypoint require blocking_reasons")
        if (
            profile.status
            in {
                SupportStatus.BLOCKED,
                SupportStatus.EXTERNAL_ONLY,
                SupportStatus.NATIVE_ONLY,
                SupportStatus.RETRAIN_REQUIRED,
            }
            and not profile.blocking_reasons
        ):
            raise RegistryError(f"{path}: status {profile.status.value} requires blocking_reasons")
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "implementation_kind": self.implementation_kind.value,
            "execution_kind": self.execution_kind.value,
            "status": self.status.value,
            "delivery_tier": self.delivery_tier,
            "profile_modes": [item.value for item in self.profile_modes],
            "entrypoint": self.entrypoint,
            "blocking_reasons": list(self.blocking_reasons),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmCard:
    """Canonical paper-method identity with one or more release profiles."""

    method_id: str
    display_name: str
    paper_title: str
    semantic_family: SemanticFamily
    release_tier: ReleaseTier
    status: SupportStatus
    publication: Publication
    upstream: UpstreamSource
    releases: tuple[ReleaseProfile, ...]
    semantic_tags: tuple[str, ...] = ()
    notes: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> AlgorithmCard:
        data = _mapping(raw, path)
        method_id = _required_string(data, "method_id", path)
        if not _IDENTIFIER.fullmatch(method_id):
            raise RegistryError(f"{path}.method_id contains unsupported characters")
        raw_releases = data.get("releases")
        if not isinstance(raw_releases, list) or not raw_releases:
            raise RegistryError(f"{path}.releases must be a non-empty array")
        releases = tuple(
            ReleaseProfile.from_mapping(item, f"{path}.releases[{index}]")
            for index, item in enumerate(raw_releases)
        )
        local_ids = [item.id for item in releases]
        if len(local_ids) != len(set(local_ids)):
            raise RegistryError(f"{path}.releases contains duplicate release IDs")
        return cls(
            method_id=method_id,
            display_name=_required_string(data, "display_name", path),
            paper_title=_required_string(data, "paper_title", path),
            semantic_family=_enum(
                SemanticFamily, data.get("semantic_family"), f"{path}.semantic_family"
            ),
            release_tier=_enum(ReleaseTier, data.get("release_tier"), f"{path}.release_tier"),
            status=_enum(SupportStatus, data.get("status"), f"{path}.status"),
            publication=Publication.from_mapping(data.get("publication"), f"{path}.publication"),
            upstream=UpstreamSource.from_mapping(data.get("upstream"), f"{path}.upstream"),
            releases=releases,
            semantic_tags=_string_tuple(data.get("semantic_tags", []), f"{path}.semantic_tags"),
            notes=_optional_string(data, "notes", path),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "paper_title": self.paper_title,
            "semantic_family": self.semantic_family.value,
            "release_tier": self.release_tier.value,
            "status": self.status.value,
            "publication": self.publication.to_dict(),
            "upstream": self.upstream.to_dict(),
            "semantic_tags": list(self.semantic_tags),
            "releases": [item.to_dict() for item in self.releases],
        }
        if self.notes is not None:
            result["notes"] = self.notes
        return result


@dataclass(frozen=True, slots=True)
class RegistryDocument:
    """Validated immutable registry document."""

    schema_version: int
    registry_id: str
    audit_date: str
    reserve_pool_status: str
    algorithms: tuple[AlgorithmCard, ...]

    @property
    def releases(self) -> tuple[ReleaseProfile, ...]:
        return tuple(release for method in self.algorithms for release in method.releases)

    @classmethod
    def from_mapping(cls, raw: Any) -> RegistryDocument:
        data = _mapping(raw, "registry")
        schema_version = data.get("schema_version")
        if schema_version != 1:
            raise RegistryError("registry.schema_version must equal 1")
        raw_algorithms = data.get("algorithms")
        if not isinstance(raw_algorithms, list):
            raise RegistryError("registry.algorithms must be an array")
        algorithms = tuple(
            AlgorithmCard.from_mapping(item, f"registry.algorithms[{index}]")
            for index, item in enumerate(raw_algorithms)
        )
        document = cls(
            schema_version=schema_version,
            registry_id=_required_string(data, "registry_id", "registry"),
            audit_date=_required_string(data, "audit_date", "registry"),
            reserve_pool_status=_required_string(data, "reserve_pool_status", "registry"),
            algorithms=algorithms,
        )
        document.validate_roster()
        return document

    def validate_roster(self) -> None:
        """Enforce the frozen v0.2 roster and release-level identity rules."""

        if len(self.algorithms) != 27:
            raise RegistryError(f"registry must contain 27 methods, found {len(self.algorithms)}")
        method_ids = [item.method_id for item in self.algorithms]
        if len(method_ids) != len(set(method_ids)):
            raise RegistryError("registry contains duplicate method IDs")
        release_ids = [item.id for item in self.releases]
        if len(release_ids) != len(set(release_ids)):
            raise RegistryError("registry contains duplicate release IDs")

        core = [item for item in self.algorithms if item.release_tier is ReleaseTier.CORE]
        optional = [item for item in self.algorithms if item.release_tier is ReleaseTier.OPTIONAL]
        if len(core) != 20 or len(optional) != 7:
            raise RegistryError(
                f"registry tier counts must be core=20 optional=7, "
                f"found core={len(core)} optional={len(optional)}"
            )
        expected_family_counts = {
            SemanticFamily.REFORMULATE: 4,
            SemanticFamily.DECOMPOSE: 5,
            SemanticFamily.ACQUIRE: 5,
            SemanticFamily.CLARIFY: 2,
            SemanticFamily.CURATE: 4,
        }
        actual_family_counts = {
            family: sum(item.semantic_family is family for item in core)
            for family in expected_family_counts
        }
        if actual_family_counts != expected_family_counts:
            rendered = {key.value: value for key, value in actual_family_counts.items()}
            raise RegistryError(
                f"core semantic-family counts do not match frozen roster: {rendered}"
            )

        must_ship = {item.id for item in self.releases if item.must_ship}
        expected_must_ship = {
            "rewrite.query2doc.official_cached",
            "rewrite.query2doc.querygym_live",
            "plan.plan_and_solve.ps_prompt",
            "plan.decomp.full",
            "plan.self_ask.full_loop",
            "plan.plan_and_budget.native_cached_soft_budget",
            "acquire.hyde.paper_reimplementation",
            "acquire.ircot.controlled",
            "clarify.udcs.repo_native_prompt_route",
            "curate.longllmlingua.official",
            "curate.llmlingua2.xlm_roberta_official",
            "recipe.ace.offline",
        }
        if must_ship != expected_must_ship:
            missing = sorted(expected_must_ship - must_ship)
            extra = sorted(must_ship - expected_must_ship)
            raise RegistryError(
                f"must-ship release set differs from the frozen plan; missing={missing}, extra={extra}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "audit_date": self.audit_date,
            "reserve_pool_status": self.reserve_pool_status,
            "algorithms": [item.to_dict() for item in self.algorithms],
        }


__all__ = [
    "AlgorithmCard",
    "Publication",
    "RegistryDocument",
    "ReleaseProfile",
    "UpstreamSource",
]
