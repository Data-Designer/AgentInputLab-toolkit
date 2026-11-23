"""Dependency-locked QueryGym live Query2doc; native generation logic is unchanged."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agentinputlab.backends.llm import GenerationRequest
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

from .models import QueryGymMode
from .querygym_bridge import QUERYGYM_REVISION, QueryGymLiveBridge


@dataclass(frozen=True, slots=True)
class Query2DocLiveConfig:
    mode: QueryGymMode | str = QueryGymMode.ZERO_SHOT
    demonstration_ids: tuple[str, ...] = ()
    temperature: float = 0.7
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.mode, (str, QueryGymMode)):
            raise ValidationError("controlled Query2doc mode must be text")
        object.__setattr__(self, "mode", QueryGymMode.parse(self.mode))
        if isinstance(self.demonstration_ids, (str, bytes)):
            raise ValidationError("controlled Query2doc demonstrations must be an ID sequence")
        ids = tuple(self.demonstration_ids)
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValidationError("controlled Query2doc demonstration IDs must be unique nonempty text")
        object.__setattr__(self, "demonstration_ids", ids)
        if self.mode is QueryGymMode.FEW_SHOT and not ids:
            raise ValidationError("controlled Query2doc few-shot mode requires frozen demonstration IDs")
        if self.mode is not QueryGymMode.FEW_SHOT and ids:
            raise ValidationError("controlled Query2doc demonstrations require few-shot mode")
        # Reuse the public request's finite numeric/token validation without a
        # model call. Prompt rendering still belongs to the pinned native worker.
        GenerationRequest(prompt="configuration validation", temperature=self.temperature,
                          max_tokens=self.max_tokens)


_DEFAULT_CONFIG = Query2DocLiveConfig()


def _identity(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"controlled Query2doc {label} fingerprint must be nonempty")
    try:
        frozen = freeze_mapping(value)
        content_hash(frozen)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"controlled Query2doc {label} fingerprint must be canonical JSON") from exc
    return frozen


def _runtime_identities(context: InterventionContext) -> dict[str, Mapping[str, Any]]:
    values = {}
    for label, dependency, operation in (
        ("generator", context.backend, "generate"),
        ("retriever", context.retriever, "retrieve"),
        ("runtime", context.external_runtime, "reformulate"),
    ):
        fingerprint = getattr(dependency, "fingerprint", None)
        if not callable(fingerprint) or not callable(getattr(dependency, operation, None)):
            raise ExternalDependencyError(
                f"controlled Query2doc context requires {label} with {operation}() and fingerprint()"
            )
        values[label] = _identity(fingerprint(), label)
    return values


@dataclass(frozen=True, slots=True)
class Query2DocLiveBindings:
    """Immutable identities/settings; executable dependencies stay in the context."""

    generator: Mapping[str, Any]
    retriever: Mapping[str, Any]
    runtime: Mapping[str, Any]
    budget: ExecutionBudget
    config: Query2DocLiveConfig = _DEFAULT_CONFIG

    def __post_init__(self) -> None:
        if not isinstance(self.budget, ExecutionBudget) or not isinstance(self.config, Query2DocLiveConfig):
            raise ValidationError("controlled Query2doc bindings require typed budget/config")
        for label in ("generator", "retriever", "runtime"):
            object.__setattr__(self, label, _identity(getattr(self, label), label))
        if self.runtime.get("revision") != QUERYGYM_REVISION:
            raise ValidationError("controlled Query2doc runtime differs from the pinned QueryGym revision")
        closure = self.runtime.get("source_closure_sha256")
        if not isinstance(closure, str) or re.fullmatch(r"[0-9a-f]{64}", closure) is None:
            raise ValidationError("controlled Query2doc requires a runtime source-closure SHA-256")
        if content_hash(self.runtime.get("backend")) != content_hash(self.generator):
            raise ValidationError("controlled Query2doc runtime uses a different generator identity")

    @classmethod
    def from_context(
        cls, context: InterventionContext, *, budget: ExecutionBudget,
        config: Query2DocLiveConfig = _DEFAULT_CONFIG,
    ) -> Query2DocLiveBindings:
        if not isinstance(context, InterventionContext):
            raise ValidationError("controlled Query2doc requires an InterventionContext")
        return cls(**_runtime_identities(context), budget=budget, config=config)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Query2DocLiveBindings:
        required = {"generator", "retriever", "runtime", "budget"}
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise ValidationError("controlled Query2doc binding recipe is missing required fields")
        if set(value) - required - {"config"}:
            raise ValidationError("controlled Query2doc binding recipe contains unknown fields")
        if not isinstance(value["budget"], Mapping) or not isinstance(value.get("config", {}), Mapping):
            raise ValidationError("controlled Query2doc budget/config recipes must be mappings")
        try:
            return cls(generator=value["generator"], retriever=value["retriever"], runtime=value["runtime"],
                       budget=ExecutionBudget(**value["budget"]),
                       config=Query2DocLiveConfig(**value.get("config", {})))
        except TypeError as exc:
            raise ValidationError("controlled Query2doc recipe contains invalid settings") from exc

    @property
    def binding_hash(self) -> str:
        return content_hash(self)

    def require_match(self, context: InterventionContext) -> None:
        for label, actual in _runtime_identities(context).items():
            if content_hash(actual) != content_hash(getattr(self, label)):
                raise ValidationError(f"controlled Query2doc {label} fingerprint differs from binding")


@dataclass(frozen=True, slots=True)
class Query2DocControlledLive:
    """Native QueryGym reformulation with locked shared-comparison dependencies.

    Retrieval is not performed by a query rewriter. Binding the shared retriever
    prevents a dispatch-time swap; the downstream benchmark driver must use it
    with its own declared retrieval depth and remaining budget.
    """

    bindings: Query2DocLiveBindings | Mapping[str, Any]
    profile_kind: str = "controlled_adapter"
    method_id: str = "query2doc"
    release_id: str = "rewrite.query2doc.controlled_live"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        if isinstance(self.bindings, Mapping):
            object.__setattr__(self, "bindings", Query2DocLiveBindings.from_mapping(self.bindings))
        if not isinstance(self.bindings, Query2DocLiveBindings):
            raise ValidationError("controlled Query2doc requires Query2DocLiveBindings")
        if self.profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("controlled Query2doc profile_kind must be controlled or factorial")
        if (self.method_id != "query2doc" or self.release_id != "rewrite.query2doc.controlled_live"
                or self.family is not SemanticFamily.REFORMULATE):
            raise ValidationError("controlled Query2doc identity differs from its registered profile")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if state.budget != self.bindings.budget:
            raise ValidationError("controlled Query2doc state budget differs from binding")
        self.bindings.require_match(context)
        settings = self.bindings.config
        native = QueryGymLiveBridge(mode=settings.mode, demonstration_ids=settings.demonstration_ids,
            temperature=settings.temperature, max_tokens=settings.max_tokens).propose(state, context)
        self.bindings.require_match(context)
        return replace(native, release_id=self.release_id, profile_kind=self.profile_kind,
            trace=replace(native.trace, release_id=self.release_id, events=(
                TraceEvent(TracePhase.PREPARE, "controlled_dependency_binding", {
                    "binding_sha256": self.bindings.binding_hash, "budget": self.bindings.budget,
                    "config": settings, "shared_retriever_bound_not_invoked": True}),
                *native.trace.events)),
            metadata={**native.metadata, "native_source_release": native.release_id,
                "controlled_binding_sha256": self.bindings.binding_hash,
                "dependency_identity_verification": "operator_fingerprint_equality",
                "downstream_retrieval_performed": False, "weight_files_verified_by_binding": False,
                "original_paper_result_parity_claimed": False})


__all__ = ["Query2DocControlledLive", "Query2DocLiveBindings", "Query2DocLiveConfig"]
