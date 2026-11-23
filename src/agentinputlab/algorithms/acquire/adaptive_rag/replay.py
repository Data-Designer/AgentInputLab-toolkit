"""Replay the released Adaptive-RAG router choice and matched strategy output.

The upstream method selects among no retrieval, one-shot retrieval, and IRCoT
outputs generated under one locked dependency bundle.  This adapter therefore
does not silently rerun those arms with different models or retrievers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts import (
    ArtifactError,
    ContextBlock,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import freeze_mapping

ADAPTIVE_RAG_UPSTREAM_COMMIT = "0c88670af8707667eb5c1163151bb5ce61b14acb"


class AdaptiveRAGStrategy(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    SINGLE_RETRIEVAL = "single_retrieval"
    IRCOT = "ircot"

    @property
    def route_label(self) -> str:
        return {
            AdaptiveRAGStrategy.NO_RETRIEVAL: "A",
            AdaptiveRAGStrategy.SINGLE_RETRIEVAL: "B",
            AdaptiveRAGStrategy.IRCOT: "C",
        }[self]


class AdaptiveRAGPredictionSource(str, Enum):
    """Permitted origins for an evaluation-time router prediction."""

    UPSTREAM_ROUTER = "upstream_router"
    REPRODUCED_ROUTER = "reproduced_router"


class AdaptiveRAGPredictionType(str, Enum):
    """How the router converted its model output into a route."""

    GENERATIVE_CLASSIFICATION = "generative_classification"
    LOGIT_ARGMAX = "logit_argmax"


class AdaptiveRAGEvaluationLabelAccess(str, Enum):
    """Reference-label access used while producing an evaluation route."""

    NONE = "none"
    GOLD = "gold"
    SILVER = "silver"
    ORACLE = "oracle"


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactError(f"Adaptive-RAG {label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ArtifactError(f"Adaptive-RAG {label} must be hexadecimal") from exc
    return value.lower()


def _require_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"Adaptive-RAG {label} must be a non-empty revision")
    revision = value.strip()
    if revision.lower() in {"default", "head", "latest", "main", "master"}:
        raise ArtifactError(f"Adaptive-RAG {label} cannot be a mutable revision")
    return revision


@dataclass(frozen=True, slots=True)
class AdaptiveRAGRouterPrediction:
    """Auditable provenance for one evaluation-time router decision.

    Adaptive-RAG's training labels are derived by comparing the three strategy
    arms.  Those silver labels are valid training targets, but replaying them as
    selector outputs on an evaluation split would turn the selector into an
    oracle.  Consequently, this contract only accepts routes produced by an
    identified router inference and rejects any reference-label access.
    """

    predicted_strategy: AdaptiveRAGStrategy
    router_checkpoint: str
    router_revision: str
    prediction_source: AdaptiveRAGPredictionSource
    prediction_type: AdaptiveRAGPredictionType
    prediction_output_hash: str
    logits_hash: str
    evaluation_label_access: AdaptiveRAGEvaluationLabelAccess

    def __post_init__(self) -> None:
        enum_fields = (
            ("predicted_strategy", AdaptiveRAGStrategy),
            ("prediction_source", AdaptiveRAGPredictionSource),
            ("prediction_type", AdaptiveRAGPredictionType),
            ("evaluation_label_access", AdaptiveRAGEvaluationLabelAccess),
        )
        for field_name, enum_type in enum_fields:
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                try:
                    value = enum_type(value)
                except (TypeError, ValueError) as exc:
                    raise ArtifactError(
                        f"Adaptive-RAG router prediction has invalid {field_name}"
                    ) from exc
                object.__setattr__(self, field_name, value)

        if self.evaluation_label_access is not AdaptiveRAGEvaluationLabelAccess.NONE:
            raise ArtifactError(
                "Adaptive-RAG evaluation routes cannot use gold, silver, or oracle labels"
            )
        object.__setattr__(
            self,
            "router_checkpoint",
            _require_revision(self.router_checkpoint, label="router_checkpoint"),
        )
        object.__setattr__(
            self,
            "router_revision",
            _require_revision(self.router_revision, label="router_revision"),
        )
        object.__setattr__(
            self,
            "prediction_output_hash",
            _require_sha256(
                self.prediction_output_hash,
                label="prediction_output_hash",
            ),
        )
        object.__setattr__(
            self,
            "logits_hash",
            _require_sha256(self.logits_hash, label="logits_hash"),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRAGRecord:
    record_id: str
    split: str
    query: str
    strategy: AdaptiveRAGStrategy
    router_prediction: AdaptiveRAGRouterPrediction
    dependency_hash: str
    strategy_output_hash: str
    contexts: tuple[ContextBlock, ...] = ()
    selected_answer: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.split.strip() or not self.query.strip():
            raise ArtifactError("Adaptive-RAG records require ID, split, and query")
        if not isinstance(self.strategy, AdaptiveRAGStrategy):
            try:
                object.__setattr__(self, "strategy", AdaptiveRAGStrategy(str(self.strategy)))
            except ValueError as exc:
                raise ArtifactError("Adaptive-RAG record has an unsupported strategy") from exc
        if not isinstance(self.router_prediction, AdaptiveRAGRouterPrediction):
            raise ArtifactError("Adaptive-RAG records require typed router_prediction provenance")
        if self.router_prediction.predicted_strategy is not self.strategy:
            raise ArtifactError("Adaptive-RAG replay strategy does not match the router prediction")
        for name in ("dependency_hash", "strategy_output_hash"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), label=name))
        contexts = tuple(self.contexts)
        if not all(isinstance(block, ContextBlock) for block in contexts):
            raise ArtifactError("Adaptive-RAG contexts must be ContextBlock instances")
        if self.strategy is AdaptiveRAGStrategy.NO_RETRIEVAL and contexts:
            raise ArtifactError("Adaptive-RAG no-retrieval records cannot contain evidence")
        if self.strategy is not AdaptiveRAGStrategy.NO_RETRIEVAL and not contexts:
            raise ArtifactError("Adaptive-RAG retrieval records require frozen evidence")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(slots=True)
class AdaptiveRAGSelectorReplay:
    records: Sequence[AdaptiveRAGRecord]
    split: str
    dependency_hash: str
    artifact_id: str
    artifact_sha256: str
    method_id: str = "adaptive_rag"
    release_id: str = "acquire.adaptive_rag.selector_replay"
    family: SemanticFamily = SemanticFamily.ACQUIRE
    _records: Mapping[str, AdaptiveRAGRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.split.strip() or not self.artifact_id.strip():
            raise ValidationError("Adaptive-RAG replay requires split and artifact_id")
        try:
            dependency_hash = _require_sha256(self.dependency_hash, label="dependency_hash")
            artifact_sha256 = _require_sha256(self.artifact_sha256, label="artifact_sha256")
        except ArtifactError as exc:
            raise ValidationError(str(exc)) from exc
        object.__setattr__(self, "dependency_hash", dependency_hash)
        object.__setattr__(self, "artifact_sha256", artifact_sha256)
        indexed: dict[str, AdaptiveRAGRecord] = {}
        router_identity: tuple[object, ...] | None = None
        for record in self.records:
            if not isinstance(record, AdaptiveRAGRecord):
                raise ValidationError("Adaptive-RAG records have an invalid type")
            if record.split != self.split:
                raise ValidationError("Adaptive-RAG replay contains a different split")
            if record.dependency_hash != self.dependency_hash:
                raise ValidationError("Adaptive-RAG replay mixes dependency bundles")
            if record.record_id in indexed:
                raise ValidationError("Adaptive-RAG replay contains duplicate IDs")
            prediction = record.router_prediction
            record_router_identity = (
                prediction.router_checkpoint,
                prediction.router_revision,
                prediction.prediction_source,
                prediction.prediction_type,
            )
            if router_identity is None:
                router_identity = record_router_identity
            elif record_router_identity != router_identity:
                raise ValidationError("Adaptive-RAG replay mixes router identities")
            indexed[record.record_id] = record
        if not indexed:
            raise ValidationError("Adaptive-RAG replay requires at least one record")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "_records", MappingProxyType(dict(indexed)))

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        declared_hash = context.run_config.get("adaptive_rag_dependency_hash")
        if declared_hash is not None and declared_hash != self.dependency_hash:
            raise ArtifactError("Adaptive-RAG runtime dependency bundle does not match replay")
        record_id = state.task_id or str(state.metadata.get("record_id", ""))
        try:
            record = self._records[record_id]
        except KeyError as exc:
            raise ArtifactError(f"Adaptive-RAG replay has no record for {record_id!r}") from exc
        if record.query != state.query:
            raise ArtifactError("Adaptive-RAG replay query mismatch")
        prediction = record.router_prediction
        if prediction.evaluation_label_access is not AdaptiveRAGEvaluationLabelAccess.NONE:
            raise ArtifactError(
                "Adaptive-RAG evaluation routes cannot use gold, silver, or oracle labels"
            )
        if prediction.predicted_strategy is not record.strategy:
            raise ArtifactError("Adaptive-RAG replay strategy does not match the router prediction")
        if record.strategy is AdaptiveRAGStrategy.NO_RETRIEVAL:
            operation = PatchOperation(
                PatchOp.NO_OP,
                rationale="Adaptive-RAG classifier selected the no-retrieval arm",
            )
        else:
            operation = PatchOperation(
                PatchOp.ADD_CONTEXT,
                record.contexts,
                target="retrieved_evidence",
                rationale=(
                    "frozen strategy output selected by the Adaptive-RAG complexity classifier"
                ),
            )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "selector_artifact",
                    {
                        "artifact_id": self.artifact_id,
                        "artifact_sha256": self.artifact_sha256,
                        "split": self.split,
                        "dependency_hash": self.dependency_hash,
                        "test_gold_visible": False,
                        "evaluation_label_access": (prediction.evaluation_label_access.value),
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "complexity_route",
                    {
                        "strategy": record.strategy.value,
                        "predicted_route_label": record.strategy.route_label,
                        "router_checkpoint": prediction.router_checkpoint,
                        "router_revision": prediction.router_revision,
                        "prediction_source": prediction.prediction_source.value,
                        "prediction_type": prediction.prediction_type.value,
                        "prediction_output_hash": prediction.prediction_output_hash,
                        "logits_hash": prediction.logits_hash,
                        "strategy_output_hash": record.strategy_output_hash,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "selected_strategy_output",
                    {
                        "document_ids": tuple(block.block_id for block in record.contexts),
                        "selected_answer": record.selected_answer,
                        "answer_is_intervention_input": False,
                    },
                ),
            ),
            counters={
                "replay_records": 1,
                "historical_strategy_calls": 1,
                "selected_documents": len(record.contexts),
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity=(
                f"artifact_replay:starsuzi/Adaptive-RAG@{ADAPTIVE_RAG_UPSTREAM_COMMIT}:"
                f"{self.artifact_sha256}"
            ),
            profile_kind="official_native",
            metadata={
                "implementation_origin": "artifact_replay",
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "strategy": record.strategy.value,
                "predicted_route_label": record.strategy.route_label,
                "router_checkpoint": prediction.router_checkpoint,
                "router_revision": prediction.router_revision,
                "prediction_source": prediction.prediction_source.value,
                "prediction_type": prediction.prediction_type.value,
                "prediction_output_hash": prediction.prediction_output_hash,
                "logits_hash": prediction.logits_hash,
                "evaluation_label_access": prediction.evaluation_label_access.value,
                "dependency_hash": self.dependency_hash,
                "strategy_output_hash": record.strategy_output_hash,
                "selected_answer": record.selected_answer,
                "historical_generation_not_rebilled": True,
                "record_metadata": dict(record.metadata),
            },
        )


__all__ = [
    "ADAPTIVE_RAG_UPSTREAM_COMMIT",
    "AdaptiveRAGEvaluationLabelAccess",
    "AdaptiveRAGPredictionSource",
    "AdaptiveRAGPredictionType",
    "AdaptiveRAGRecord",
    "AdaptiveRAGRouterPrediction",
    "AdaptiveRAGSelectorReplay",
    "AdaptiveRAGStrategy",
]
