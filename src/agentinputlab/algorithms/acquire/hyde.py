"""Clean-room HyDE formula with injectable generation, embedding, and search.

This module implements the formula described in the HyDE paper rather than
copying the unlicensed upstream source.  Hypothetical documents are query
surrogates only.  Only documents returned by the injected corpus searcher are
materialized as evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from agentinputlab.backends.llm import GenerationRequest, LLMBackend
from agentinputlab.backends.retrieval import RetrievalResult
from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
    ExecutionTrace,
    ExternalDependencyError,
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

Vector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HyDEConfig:
    num_hypotheses: int = 8
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 5
    prompt_template: str = (
        "Write a passage that would answer the question. "
        "The passage should be factual and self-contained.\n\n"
        "Question: {query}\n\nPassage:"
    )

    def __post_init__(self) -> None:
        for name in ("num_hypotheses", "max_tokens", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"HyDE {name} must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValidationError("HyDE temperature must be finite and non-negative")
        if not isinstance(self.prompt_template, str):
            raise ValidationError("HyDE prompt_template must be text")
        try:
            self.prompt_template.format(query="probe")
        except (KeyError, ValueError) as exc:
            raise ValidationError(
                "HyDE prompt_template must accept only the {query} field"
            ) from exc


@dataclass(slots=True)
class HyDECleanRoom:
    """Paper-formula implementation: mean(query, hypothetical-doc embeddings)."""

    embedder: DenseEmbedder | None = None
    searcher: VectorSearcher | None = None
    generator: LLMBackend | None = None
    config: HyDEConfig = HyDEConfig()
    method_id: str = "hyde"
    release_id: str = "acquire.hyde.paper_reimplementation"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("HyDE requires a non-empty retrieval query")
        backend = self.generator or context.backend
        if backend is None:
            raise ExternalDependencyError("HyDE requires an injected LLM generation backend")
        embedder = self.embedder or context.dense_embedder
        searcher = self.searcher or context.vector_searcher
        if not isinstance(embedder, DenseEmbedder):
            raise ExternalDependencyError("HyDE requires an injected DenseEmbedder")
        if not isinstance(searcher, VectorSearcher):
            raise ExternalDependencyError("HyDE requires an injected VectorSearcher")

        try:
            prompt = self.config.prompt_template.format(query=state.query)
        except (KeyError, ValueError) as exc:
            raise ValidationError("invalid HyDE prompt template") from exc

        hypotheses: list[str] = []
        empty_generations: list[int] = []
        for index in range(self.config.num_hypotheses):
            response = backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=context.seed + index,
                    metadata={
                        "method_id": self.method_id,
                        "release_id": self.release_id,
                        "hypothesis_index": index,
                    },
                )
            )
            if not isinstance(response.text, str):
                raise AlgorithmOutputError("HyDE generator returned a non-text response")
            text = response.text.strip()
            if text:
                hypotheses.append(text)
            else:
                empty_generations.append(index)
        if not hypotheses:
            raise AlgorithmOutputError("HyDE generator returned no usable hypothetical documents")

        encoded = embedder.embed((state.query, *hypotheses))
        vectors = _validate_vectors(encoded, expected_count=1 + len(hypotheses))
        mean_vector = _arithmetic_mean(vectors)
        results = _validate_results(
            searcher.search(mean_vector, top_k=self.config.top_k),
            top_k=self.config.top_k,
        )
        if not results:
            raise AlgorithmOutputError("HyDE vector search returned no corpus documents")

        evidence = tuple(_context_from_result(result) for result in results)
        warnings = (f"empty_hypothesis_indices={empty_generations}",) if empty_generations else ()
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "hyde_config",
                    {
                        "num_hypotheses": self.config.num_hypotheses,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "top_k": self.config.top_k,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "hypothetical_documents",
                    {
                        "documents": tuple(hypotheses),
                        "semantic_role": "synthetic_query_surrogate",
                        "empty_generation_indices": tuple(empty_generations),
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "mean_embedding",
                    {
                        "vector_count": len(vectors),
                        "dimension": len(mean_vector),
                        "mean_vector": mean_vector,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "real_corpus_results",
                    {
                        "document_ids": tuple(item.document.doc_id for item in results),
                        "ranks": tuple(item.rank for item in results),
                    },
                ),
            ),
            counters={
                "model_calls": self.config.num_hypotheses,
                "hypotheses_used": len(hypotheses),
                "embedding_calls": 1,
                "retrieval_calls": 1,
            },
            warnings=warnings,
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    evidence,
                    target="retrieved_evidence",
                    rationale="real documents retrieved with the HyDE mean embedding",
                ),
            ),
            trace=trace,
            implementation_identity="clean_room_paper_formula",
            profile_kind="controlled_adapter",
            metadata={
                "synthetic_documents_are_evidence": False,
                "generator_fingerprint": dict(backend.fingerprint()),
                "embedder_fingerprint": dict(embedder.fingerprint()),
                "searcher_fingerprint": dict(searcher.fingerprint()),
            },
        )


def _validate_vectors(
    encoded: Sequence[Sequence[float]], *, expected_count: int
) -> tuple[Vector, ...]:
    try:
        vectors = tuple(tuple(float(value) for value in row) for row in encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlgorithmOutputError("HyDE embedder returned non-numeric vectors") from exc
    if len(vectors) != expected_count:
        raise AlgorithmOutputError(
            f"HyDE embedder returned {len(vectors)} vectors; expected {expected_count}"
        )
    if not vectors or not vectors[0]:
        raise AlgorithmOutputError("HyDE embeddings must have a non-zero dimension")
    dimension = len(vectors[0])
    for vector in vectors:
        if len(vector) != dimension:
            raise AlgorithmOutputError("HyDE embeddings have inconsistent dimensions")
        if not all(math.isfinite(value) for value in vector):
            raise AlgorithmOutputError("HyDE embeddings must contain only finite values")
    return vectors


def _arithmetic_mean(vectors: Sequence[Vector]) -> Vector:
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))
    )


def _validate_results(
    values: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(values)
    except TypeError as exc:
        raise AlgorithmOutputError("HyDE searcher returned a non-sequence result") from exc
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise AlgorithmOutputError("HyDE searcher returned a non-RetrievalResult item")
    if len(results) > top_k:
        raise AlgorithmOutputError("HyDE searcher returned more than top_k documents")
    document_ids = [item.document.doc_id for item in results]
    if len(document_ids) != len(set(document_ids)):
        raise AlgorithmOutputError("HyDE searcher returned duplicate document IDs")
    if tuple(item.rank for item in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("HyDE searcher ranks must be contiguous and one-based")
    if not all(math.isfinite(item.score) for item in results):
        raise AlgorithmOutputError("HyDE search scores must be finite")
    return results


def _context_from_result(result: RetrievalResult) -> ContextBlock:
    document = result.document
    if not document.doc_id or not document.corpus_revision:
        raise AlgorithmOutputError(
            "HyDE search results require document ID and corpus revision provenance"
        )
    reserved = {
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "rank",
        "retrieval_features",
        "title",
    }
    overlap = reserved & set(document.metadata)
    if overlap:
        raise AlgorithmOutputError(
            "HyDE document metadata uses reserved provenance keys: " + ", ".join(sorted(overlap))
        )
    return ContextBlock(
        block_id=document.doc_id,
        text=document.text,
        source=document.source if hasattr(document, "source") else document.corpus_revision,
        score=result.score,
        metadata={
            **dict(document.metadata),
            "document_id": document.doc_id,
            "title": document.title,
            "corpus_revision": document.corpus_revision,
            "rank": result.rank,
            "retrieval_features": dict(result.features),
            "evidence_kind": "real_corpus_document",
        },
    )
