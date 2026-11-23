"""Lifecycle-aware intervention protocols.

The semantic protocols are deliberately distinct even though all of them return
an :class:`InputPatch`.  This prevents stateful retrieval or clarification from
being mislabeled as a one-shot string rewriter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .enums import SemanticFamily
from .errors import ValidationError
from .intent import IntentContract
from .patch import InputPatch
from .serialization import content_hash, freeze_mapping, to_jsonable
from .state import ContextBlock, FrozenAgentState
from .trace import ExecutionTrace


@dataclass(frozen=True, slots=True)
class InterventionContext:
    backend: Any = None
    dense_embedder: Any = None
    retriever: Any = None
    vector_searcher: Any = None
    tools: Any = None
    external_runtime: Any = None
    artifact_manager: Any = None
    seed: int = 0
    run_config: Mapping[str, Any] = field(default_factory=dict)
    intent_contract: IntentContract | None = None
    budget_ledger: Any = None
    monitor: Any = None
    pricing_catalog: Any = None
    strict_intent: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_config", freeze_mapping(self.run_config))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("seed must be a non-negative integer")
        if self.intent_contract is not None and not isinstance(
            self.intent_contract, IntentContract
        ):
            raise TypeError("intent_contract must be an IntentContract")


@runtime_checkable
class Intervention(Protocol):
    method_id: str
    release_id: str
    family: SemanticFamily

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        """Propose a patch without mutating the frozen state."""


@runtime_checkable
class Reformulator(Intervention, Protocol):
    family: SemanticFamily


@runtime_checkable
class Decomposer(Intervention, Protocol):
    family: SemanticFamily


@runtime_checkable
class EvidenceAcquirer(Intervention, Protocol):
    family: SemanticFamily


@runtime_checkable
class Clarifier(Intervention, Protocol):
    family: SemanticFamily


@runtime_checkable
class ContextCurator(Intervention, Protocol):
    family: SemanticFamily


@dataclass(frozen=True, slots=True)
class CompositeResult:
    """Terminal result of a full agent/search architecture.

    Composite methods keep their final answer and multi-step trajectory instead
    of pretending to be a one-shot input patch.  A downstream experiment may
    separately derive a controlled evidence patch, but that is a different
    release profile and estimand.
    """

    method_id: str
    release_id: str
    source_state_hash: str
    final_answer: str
    evidence: tuple[ContextBlock, ...]
    trace: ExecutionTrace
    implementation_identity: str
    profile_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        if (
            not isinstance(self.method_id, str)
            or not self.method_id.strip()
            or not isinstance(self.release_id, str)
            or not self.release_id.strip()
        ):
            raise ValidationError("composite result requires method and release IDs")
        if not isinstance(self.source_state_hash, str) or not self.source_state_hash.strip():
            raise ValidationError("composite result requires source_state_hash")
        if not isinstance(self.final_answer, str) or not self.final_answer.strip():
            raise ValidationError("composite final_answer must be non-empty text")
        if not all(isinstance(block, ContextBlock) for block in evidence):
            raise ValidationError("composite evidence must contain ContextBlock instances")
        if not isinstance(self.trace, ExecutionTrace):
            raise ValidationError("composite result requires an ExecutionTrace")
        if self.trace.method_id != self.method_id or self.trace.release_id != self.release_id:
            raise ValidationError("composite trace identity does not match result identity")
        if (
            not isinstance(self.implementation_identity, str)
            or not self.implementation_identity.strip()
            or not isinstance(self.profile_kind, str)
            or not self.profile_kind.strip()
        ):
            raise ValidationError("composite implementation/profile identity must not be empty")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def result_hash(self) -> str:
        return content_hash(self)

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@runtime_checkable
class CompositeIntervention(Protocol):
    method_id: str
    release_id: str
    family: SemanticFamily

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        """Run a full architecture and return its terminal outcome and trace."""


@runtime_checkable
class OfflineRecipe(Protocol):
    method_id: str
    release_id: str

    def fit(
        self,
        train_examples: tuple[Any, ...],
        *,
        validation_examples: tuple[Any, ...] = (),
        context: InterventionContext,
    ) -> Any:
        """Produce a versioned, replayable artifact without consulting test data."""
