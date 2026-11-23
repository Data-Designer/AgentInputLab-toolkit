"""Hermetic Adaptive-RAG runtime used only by explicit fixture tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .driver import (
    GenerationResult,
    RetrievedDocument,
    RouterResult,
    RuntimeBundle,
    RuntimeIdentity,
)


class HermeticRouter:
    """Return raw logits; the worker, not this fixture, chooses the route."""

    def __init__(
        self,
        identity: RuntimeIdentity,
        logits_by_question: Mapping[str, Mapping[str, float]],
    ) -> None:
        self.identity = identity
        self.logits_by_question = dict(logits_by_question)
        self.calls = 0

    def component_manifest(self) -> Mapping[str, object]:
        return {
            "component_type": "router",
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "fixture_runtime": True,
            "revision": self.identity.router_revision,
            "snapshot_sha256": self.identity.router_snapshot_sha256,
            "weights_sha256": self.identity.router_weights_sha256,
            "tokenizer_sha256": self.identity.router_tokenizer_sha256,
        }

    def estimate_input_tokens(self, question: str) -> int:
        return max(1, len(question.encode("utf-8")))

    def first_decoder_step(self, question: str, *, seed: int) -> RouterResult:
        del seed
        self.calls += 1
        return RouterResult(
            logits=dict(self.logits_by_question[question]),
            input_tokens=self.estimate_input_tokens(question),
        )


class HermeticGenerator:
    def __init__(self, identity: RuntimeIdentity) -> None:
        self.identity = identity
        self.operations: list[str] = []

    def component_manifest(self) -> Mapping[str, object]:
        return {
            "component_type": "generator",
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "fixture_runtime": True,
            "revision": self.identity.base_model_revision,
            "artifact_sha256": self.identity.generator_artifact_sha256,
            "config_sha256": self.identity.generator_config_sha256,
        }

    def estimate_input_tokens(
        self,
        *,
        operation: str,
        question: str,
        documents: Sequence[RetrievedDocument],
        reasoning: Sequence[str],
        prompt: str,
    ) -> int:
        del operation
        text = question + prompt + "".join(item.text for item in documents) + "".join(reasoning)
        return max(1, len(text.encode("utf-8")))

    def generate(
        self,
        *,
        operation: str,
        question: str,
        documents: Sequence[RetrievedDocument],
        reasoning: Sequence[str],
        prompt: str,
        max_output_tokens: int,
        seed: int,
    ) -> GenerationResult:
        del seed
        self.operations.append(operation)
        if operation.startswith("ircot.reason."):
            text = "retrieve fixture-second"
        else:
            text = f"{operation} fixture answer"
        return GenerationResult(
            text=text,
            input_tokens=self.estimate_input_tokens(
                operation=operation,
                question=question,
                documents=documents,
                reasoning=reasoning,
                prompt=prompt,
            ),
            output_tokens=min(max_output_tokens, max(1, len(text.split()))),
        )


class HermeticRetriever:
    def __init__(
        self,
        identity: RuntimeIdentity,
        documents_by_query: Mapping[str, Sequence[RetrievedDocument]],
    ) -> None:
        self.identity = identity
        self.documents_by_query = {
            query: tuple(documents) for query, documents in documents_by_query.items()
        }
        self.queries: list[str] = []

    def component_manifest(self) -> Mapping[str, object]:
        return {
            "component_type": "retriever",
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "fixture_runtime": True,
            "revision": self.identity.retriever_revision,
            "implementation_sha256": self.identity.retriever_implementation_sha256,
            "config_sha256": self.identity.retriever_config_sha256,
            "corpus_revision": self.identity.corpus_revision,
            "corpus_manifest_sha256": self.identity.corpus_manifest_sha256,
            "index_revision": self.identity.index_revision,
            "index_manifest_sha256": self.identity.index_manifest_sha256,
        }

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedDocument]:
        self.queries.append(query)
        return self.documents_by_query.get(query, ())[:top_k]


def build_fixture_runtime(
    *,
    identity: RuntimeIdentity,
    logits_by_question: Mapping[str, Mapping[str, float]],
    documents_by_query: Mapping[str, Sequence[RetrievedDocument]],
) -> RuntimeBundle:
    """Build a runtime that is permanently and visibly fixture-marked."""

    if not identity.fixture_runtime:
        raise ValueError("hermetic Adaptive-RAG identity must set fixture_runtime=True")
    return RuntimeBundle(
        identity=identity,
        router=HermeticRouter(identity, logits_by_question),
        generator=HermeticGenerator(identity),
        retriever=HermeticRetriever(identity, documents_by_query),
    )


__all__ = [
    "HermeticGenerator",
    "HermeticRetriever",
    "HermeticRouter",
    "build_fixture_runtime",
]
