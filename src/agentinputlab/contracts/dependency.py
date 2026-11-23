"""Dependency-slot identities used by controlled and factorial comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import StrEnum
from .errors import ValidationError
from .serialization import content_hash, freeze_mapping


class DependencyRole(StrEnum):
    TREATMENT = "treatment"
    NUISANCE = "nuisance"
    CROSSED = "crossed"


@dataclass(frozen=True, slots=True)
class DependencyBinding:
    slot: str
    role: DependencyRole
    implementation_id: str
    revision: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slot or not self.implementation_id:
            raise ValidationError("dependency binding requires slot and implementation_id")
        try:
            object.__setattr__(self, "config", freeze_mapping(self.config))
        except TypeError as exc:
            raise ValidationError(
                f"dependency binding config must be canonically serializable: {exc}"
            ) from exc

    @property
    def resolved_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class TreatmentDefinition:
    method_id: str
    release_id: str
    treatment_slots: tuple[str, ...]
    nuisance_slots: tuple[str, ...] = ()
    crossed_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups = [
            set(self.treatment_slots),
            set(self.nuisance_slots),
            set(self.crossed_slots),
        ]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValidationError("treatment, nuisance, and crossed slots must be disjoint")
