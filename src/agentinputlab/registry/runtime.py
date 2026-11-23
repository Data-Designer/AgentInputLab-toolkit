"""Runtime lookup, filtering, diagnostics, and lazy entry-point resolution."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from agentinputlab.contracts.enums import ReleaseTier, SemanticFamily, SupportStatus
from agentinputlab.contracts.errors import AlgorithmUnavailableError, RegistryError

from .loader import load_packaged_registry
from .schema import AlgorithmCard, RegistryDocument, ReleaseProfile

# Algorithm package initializers re-export sibling profiles. Serialize cold
# imports across registry instances so concurrent first requests cannot acquire
# sibling module locks in opposite orders. Warm model execution stays parallel.
_ENTRYPOINT_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class RegistryDoctorReport:
    """Human- and machine-readable registry health summary."""

    ok: bool
    method_count: int
    release_count: int
    core_count: int
    optional_count: int
    must_ship_count: int
    entrypoint_count: int
    locked_count: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method_count": self.method_count,
            "release_count": self.release_count,
            "core_count": self.core_count,
            "optional_count": self.optional_count,
            "must_ship_count": self.must_ship_count,
            "entrypoint_count": self.entrypoint_count,
            "locked_count": self.locked_count,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class AlgorithmRegistry:
    """Immutable indexed view of a validated :class:`RegistryDocument`."""

    def __init__(self, document: RegistryDocument):
        document.validate_roster()
        self._document = document
        self._methods = {item.method_id: item for item in document.algorithms}
        self._releases: dict[str, tuple[AlgorithmCard, ReleaseProfile]] = {}
        for method in document.algorithms:
            for release in method.releases:
                self._releases[release.id] = (method, release)

    @classmethod
    def default(cls) -> AlgorithmRegistry:
        return cls(load_packaged_registry())

    @property
    def document(self) -> RegistryDocument:
        return self._document

    def list_methods(
        self,
        *,
        family: SemanticFamily | str | None = None,
        tier: ReleaseTier | str | None = None,
        status: SupportStatus | str | None = None,
    ) -> tuple[AlgorithmCard, ...]:
        family_value = SemanticFamily(family) if isinstance(family, str) else family
        tier_value = ReleaseTier(tier) if isinstance(tier, str) else tier
        status_value = SupportStatus(status) if isinstance(status, str) else status
        return tuple(
            item
            for item in self._document.algorithms
            if (family_value is None or item.semantic_family is family_value)
            and (tier_value is None or item.release_tier is tier_value)
            and (status_value is None or item.status is status_value)
        )

    def list_releases(
        self,
        *,
        family: SemanticFamily | str | None = None,
        tier: ReleaseTier | str | None = None,
        status: SupportStatus | str | None = None,
        delivery_tier: str | None = None,
    ) -> tuple[tuple[AlgorithmCard, ReleaseProfile], ...]:
        status_value = SupportStatus(status) if isinstance(status, str) else status
        methods = self.list_methods(family=family, tier=tier)
        return tuple(
            (method, release)
            for method in methods
            for release in method.releases
            if (status_value is None or release.status is status_value)
            and (delivery_tier is None or release.delivery_tier == delivery_tier)
        )

    def get_method(self, method_id: str) -> AlgorithmCard:
        try:
            return self._methods[method_id]
        except KeyError as exc:
            raise RegistryError(f"unknown algorithm method ID: {method_id}") from exc

    def get_release(self, release_id: str) -> tuple[AlgorithmCard, ReleaseProfile]:
        try:
            return self._releases[release_id]
        except KeyError as exc:
            if release_id in self._methods:
                choices = ", ".join(r.id for r in self._methods[release_id].releases)
                raise RegistryError(
                    f"{release_id} is a method, not an exact release ID; choose one of: {choices}"
                ) from exc
            matches = get_close_matches(release_id, self._releases, n=3)
            hint = f"; did you mean: {', '.join(matches)}?" if matches else "; use agentinputlab list --releases"
            raise RegistryError(f"unknown algorithm release ID: {release_id}{hint}") from exc

    def get(self, identity: str) -> AlgorithmCard | tuple[AlgorithmCard, ReleaseProfile]:
        if identity in self._methods:
            return self._methods[identity]
        if identity in self._releases:
            return self._releases[identity]
        matches = get_close_matches(identity, (*self._methods, *self._releases), n=3)
        hint = f"; did you mean: {', '.join(matches)}?" if matches else "; use agentinputlab list --releases"
        raise RegistryError(f"unknown algorithm or release ID: {identity}{hint}")

    def resolve_entrypoint(self, release_id: str) -> Any:
        """Import a declared adapter only when it is explicitly requested."""

        _, release = self.get_release(release_id)
        if release.entrypoint is None:
            reasons = "; ".join(release.blocking_reasons) or "implementation is not registered"
            raise AlgorithmUnavailableError(f"{release_id} is unavailable: {reasons}")
        module_name, attribute_path = release.entrypoint.split(":", 1)
        try:
            with _ENTRYPOINT_IMPORT_LOCK:
                value: Any = importlib.import_module(module_name)
                for segment in attribute_path.split("."):
                    value = getattr(value, segment)
                return value
        except (ImportError, AttributeError) as exc:
            raise AlgorithmUnavailableError(
                f"cannot resolve entrypoint {release.entrypoint!r} for {release_id}: {exc}"
            ) from exc

    def doctor(self) -> RegistryDoctorReport:
        warnings: list[str] = []
        errors: list[str] = []
        for method in self._document.algorithms:
            if method.upstream.license_id == "NOASSERTION":
                warnings.append(
                    f"{method.method_id}: upstream license is NOASSERTION; external/clean-room only"
                )
            for release in method.releases:
                if release.must_ship and release.status not in {
                    SupportStatus.EXPERIMENTAL,
                    SupportStatus.SUPPORTED,
                }:
                    warnings.append(
                        f"{release.id}: must-ship profile is {release.status.value}, "
                        "not locally runnable without its declared gate"
                    )
                if (
                    release.status
                    in {
                        SupportStatus.EXPERIMENTAL,
                        SupportStatus.SUPPORTED,
                    }
                    and release.entrypoint is None
                ):
                    errors.append(f"{release.id}: {release.status.value} release has no entrypoint")
        methods = self._document.algorithms
        releases = self._document.releases
        return RegistryDoctorReport(
            ok=not errors,
            method_count=len(methods),
            release_count=len(releases),
            core_count=sum(item.release_tier is ReleaseTier.CORE for item in methods),
            optional_count=sum(item.release_tier is ReleaseTier.OPTIONAL for item in methods),
            must_ship_count=sum(item.must_ship for item in releases),
            entrypoint_count=sum(item.entrypoint_available for item in releases),
            locked_count=sum(not item.entrypoint_available for item in releases),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    def __iter__(self) -> Iterable[AlgorithmCard]:
        return iter(self._document.algorithms)

    def __len__(self) -> int:
        return len(self._document.algorithms)


__all__ = ["AlgorithmRegistry", "RegistryDoctorReport"]
