"""Dependency-locked public HyDE profile using the unchanged clean-room formula.

Only immutable data is retained on the intervention. Executable dependencies
come from the runner's instrumented context, so generation, embedding and search
remain observable and budgeted. Fingerprint equality is an operator identity
binding, not independent verification of model weights or benchmark parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agentinputlab.contracts import (
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .hyde import HyDECleanRoom, HyDEConfig

_DEFAULT_CONFIG = HyDEConfig()


def _identity(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"HyDE {label} fingerprint must be a non-empty mapping")
    try:
        result = freeze_mapping(value)
        content_hash(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"HyDE {label} fingerprint must be finite canonical JSON") from exc
    return result


def _context_identities(context: InterventionContext) -> dict[str, Mapping[str, Any]]:
    identities = {}
    for label, dependency, operation in (
        ("generator", context.backend, "generate"),
        ("embedder", context.dense_embedder, "embed"),
        ("searcher", context.vector_searcher, "search"),
    ):
        fingerprint = getattr(dependency, "fingerprint", None)
        if not callable(fingerprint) or not callable(getattr(dependency, operation, None)):
            raise ExternalDependencyError(
                f"controlled HyDE requires context {label} with {operation}() and fingerprint()"
            )
        identities[label] = _identity(fingerprint(), label=label)
    return identities


@dataclass(frozen=True, slots=True)
class HyDEControlledBindings:
    """Freeze shared runtime identities, method settings and resource ceilings.

    Supply reviewed fingerprints explicitly or capture them before dispatch with
    ``from_context``. The latter records the caller's current dependencies; it
    does not independently certify those dependencies or freeze their internals.
    A changed fingerprint is rejected both before and after execution.
    """

    generator: Mapping[str, Any]
    embedder: Mapping[str, Any]
    searcher: Mapping[str, Any]
    budget: ExecutionBudget
    config: HyDEConfig = _DEFAULT_CONFIG

    def __post_init__(self) -> None:
        if not isinstance(self.budget, ExecutionBudget):
            raise ValidationError("controlled HyDE bindings require an ExecutionBudget")
        if not isinstance(self.config, HyDEConfig):
            raise ValidationError("controlled HyDE bindings require a HyDEConfig")
        for name in ("generator", "embedder", "searcher"):
            object.__setattr__(self, name, _identity(getattr(self, name), label=name))

    @classmethod
    def from_context(
        cls,
        context: InterventionContext,
        *,
        budget: ExecutionBudget,
        config: HyDEConfig = _DEFAULT_CONFIG,
    ) -> HyDEControlledBindings:
        """Capture identities without generating, embedding, searching or downloading."""

        if not isinstance(context, InterventionContext):
            raise ValidationError("controlled HyDE requires an InterventionContext")
        return cls(**_context_identities(context), budget=budget, config=config)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HyDEControlledBindings:
        """Read a data-only construction recipe for the registry/CLI builder."""

        required = {"generator", "embedder", "searcher", "budget"}
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise ValidationError("controlled HyDE binding recipe is missing required fields")
        if set(value) - required - {"config"}:
            raise ValidationError("controlled HyDE binding recipe contains unknown fields")
        if not isinstance(value["budget"], Mapping) or not isinstance(
            value.get("config", {}), Mapping
        ):
            raise ValidationError("controlled HyDE budget and config recipes must be mappings")
        try:
            return cls(
                generator=value["generator"],
                embedder=value["embedder"],
                searcher=value["searcher"],
                budget=ExecutionBudget(**value["budget"]),
                config=HyDEConfig(**value.get("config", {})),
            )
        except TypeError as exc:
            raise ValidationError("controlled HyDE recipe has invalid setting fields") from exc

    @property
    def binding_hash(self) -> str:
        return content_hash(self)

    def require_match(self, context: InterventionContext) -> None:
        for name, actual in _context_identities(context).items():
            if content_hash(actual) != content_hash(getattr(self, name)):
                raise ValidationError(f"controlled HyDE {name} fingerprint differs from binding")


@dataclass(frozen=True, slots=True)
class HyDEControlled:
    """Run genuine HyDE under explicit shared dependencies and fixed settings.

    ``factorial_adapter`` labels a caller-controlled configuration; it is not
    proof of experimental comparability or original-paper result parity.
    """

    bindings: HyDEControlledBindings | Mapping[str, Any]
    profile_kind: str = "controlled_adapter"
    method_id: str = "hyde"
    release_id: str = "acquire.hyde.controlled"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def __post_init__(self) -> None:
        if isinstance(self.bindings, Mapping):
            object.__setattr__(self, "bindings", HyDEControlledBindings.from_mapping(self.bindings))
        if not isinstance(self.bindings, HyDEControlledBindings):
            raise ValidationError("controlled HyDE requires HyDEControlledBindings")
        if self.profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("controlled HyDE profile_kind must be controlled or factorial")
        if (
            self.method_id != "hyde"
            or self.release_id != "acquire.hyde.controlled"
            or self.family is not SemanticFamily.ACQUIRE
        ):
            raise ValidationError("controlled HyDE identity differs from its registered profile")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if state.budget != self.bindings.budget:
            raise ValidationError("controlled HyDE state budget differs from binding")
        self.bindings.require_match(context)
        patch = HyDECleanRoom(
            config=self.bindings.config,
            method_id=self.method_id,
            release_id=self.release_id,
            family=self.family,
        ).propose(state, context)
        self.bindings.require_match(context)
        return replace(
            patch,
            profile_kind=self.profile_kind,
            trace=replace(
                patch.trace,
                events=(
                    TraceEvent(
                        TracePhase.PREPARE,
                        "controlled_dependency_binding",
                        {
                            "binding_hash": self.bindings.binding_hash,
                            "budget": self.bindings.budget,
                            "config": self.bindings.config,
                            "identity_verification": "operator_fingerprint_equality",
                        },
                    ),
                    *patch.trace.events,
                ),
            ),
            metadata={
                **patch.metadata,
                "controlled_binding_sha256": self.bindings.binding_hash,
                "dependency_identity_verification": "operator_fingerprint_equality",
                "weight_files_verified_by_binding": False,
                "original_paper_result_parity_claimed": False,
            },
        )


__all__ = ["HyDEControlled", "HyDEControlledBindings"]
