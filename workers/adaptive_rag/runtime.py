"""Concrete local-files-only router seam for Adaptive-RAG production workers."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentinputlab.backends import GenerationRequest
from agentinputlab.backends.llm.base import LLMBackend
from agentinputlab.backends.retrieval import Retriever

from .driver import (
    GenerationResult,
    RetrievedDocument,
    RouterResult,
    RuntimeIdentity,
    _path_closure_sha256,
)


class OpenAICompatibleGenerator:
    """Bind the controlled arms to one explicit toolkit backend, without retries.

    The driver owns call/token budgets; this adapter preserves its requested seed
    and output cap. It forwards final text only, never promotes reasoning or
    inserts a second answer-extraction call. The explicit input reservation must
    upper-bound the deployment's context capacity (or use a separate adapter
    with an exact local tokenizer); it is not charged as observed token usage.
    """

    def __init__(
        self, *, backend: LLMBackend, identity: RuntimeIdentity, input_token_upper_bound: int
    ) -> None:
        if identity.fixture_runtime:
            raise ValueError("production generator cannot use fixture identity")
        if type(input_token_upper_bound) is not int or input_token_upper_bound < 1:
            raise ValueError("input_token_upper_bound must be a positive integer")
        self.backend, self.identity = backend, identity
        self.input_token_upper_bound = input_token_upper_bound

    def component_manifest(self) -> dict[str, object]:
        return {
            "component_type": "generator",
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "fixture_runtime": False,
            "revision": self.identity.base_model_revision,
            "artifact_sha256": self.identity.generator_artifact_sha256,
            "config_sha256": self.identity.generator_config_sha256,
        }

    @staticmethod
    def request(
        *,
        operation: str,
        question: str,
        documents: Sequence[RetrievedDocument],
        reasoning: Sequence[str],
        prompt: str,
        max_output_tokens: int,
        seed: int,
    ) -> GenerationRequest:
        evidence = "\n\n".join(f"[{d.document_id}] {d.title}\n{d.text}" for d in documents)
        history = "\n".join(reasoning)
        content = f"{prompt}\n\nQuestion: {question}"
        if evidence:
            content += f"\n\nRetrieved evidence:\n{evidence}"
        if history:
            content += f"\n\nReasoning so far:\n{history}"
        if operation.startswith("ircot.reason."):
            content += (
                "\n\nWrite the next reasoning step; its last nonempty line is the retrieval query."
            )
        else:
            content += "\n\nGive the final answer, without an explanation."
        return GenerationRequest(
            prompt=content, seed=seed, temperature=0.0, max_tokens=max_output_tokens
        )

    def estimate_input_tokens(self, **kwargs: Any) -> int:
        return self.input_token_upper_bound

    def generate(self, **kwargs: Any) -> GenerationResult:
        response = self.backend.generate(self.request(**kwargs))
        return GenerationResult(
            text=response.text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
        )


class ToolkitRetriever:
    """Expose an explicitly bound toolkit retriever to the controlled worker.

    This does not acquire a corpus or construct a full index. Keep its corpus,
    subset selection, implementation and configuration in the deployment's
    independently hashed manifests. Protected/gold document metadata is never
    copied into the generator-visible result.
    """

    def __init__(self, *, retriever: Retriever, identity: RuntimeIdentity) -> None:
        if identity.fixture_runtime:
            raise ValueError("production retriever cannot use fixture identity")
        self.retriever, self.identity = retriever, identity

    def component_manifest(self) -> dict[str, object]:
        identity = self.identity
        return {
            "component_type": "retriever",
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "fixture_runtime": False,
            "revision": identity.retriever_revision,
            "implementation_sha256": identity.retriever_implementation_sha256,
            "config_sha256": identity.retriever_config_sha256,
            "corpus_revision": identity.corpus_revision,
            "corpus_manifest_sha256": identity.corpus_manifest_sha256,
            "index_revision": identity.index_revision,
            "index_manifest_sha256": identity.index_manifest_sha256,
        }

    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedDocument, ...]:
        return tuple(
            RetrievedDocument(
                document_id=row.document.doc_id,
                text=row.document.text,
                title=row.document.title,
                score=row.score,
            )
            for row in self.retriever.retrieve(query, top_k=top_k)
        )


class LocalTransformersRouter:
    """Compute the same first generated-step A/B/C scores used upstream.

    Model and tokenizer snapshots must already exist locally. Network fetches
    and remote code are disabled. The worker driver separately verifies the
    router weight-file hash and deployment revision.
    """

    def __init__(
        self,
        *,
        snapshot_root: Path,
        identity: RuntimeIdentity,
        weights_path: Path | None = None,
        tokenizer_path: Path | None = None,
        device: str = "cpu",
        max_input_tokens: int | None = None,
    ) -> None:
        if max_input_tokens is not None and (
            type(max_input_tokens) is not int or max_input_tokens < 1
        ):
            raise ValueError("Adaptive-RAG router max_input_tokens must be a positive integer")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError(
                "torch and transformers are required by the Adaptive-RAG router"
            ) from exc
        self._torch = torch
        self._lock = threading.Lock()
        self.identity = identity
        if identity.fixture_runtime:
            raise RuntimeError("LocalTransformersRouter cannot use a fixture runtime identity")
        effective_weights = weights_path or snapshot_root
        effective_tokenizer = tokenizer_path or snapshot_root
        observed = {
            "router_snapshot_sha256": _path_closure_sha256(snapshot_root),
            "router_weights_sha256": _path_closure_sha256(effective_weights),
            "router_tokenizer_sha256": _path_closure_sha256(effective_tokenizer),
        }
        for name, digest in observed.items():
            if digest != getattr(identity, name):
                raise RuntimeError(f"Adaptive-RAG local router closure mismatch: {name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_root),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(snapshot_root),
            local_files_only=True,
            trust_remote_code=False,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_input_tokens = (
            None
            if max_input_tokens is None
            else min(max_input_tokens, self.tokenizer.model_max_length)
        )
        self.label_token_ids = {
            label: int(self.tokenizer(label).input_ids[0]) for label in ("A", "B", "C")
        }
        if len(set(self.label_token_ids.values())) != 3:
            raise RuntimeError("Adaptive-RAG A/B/C tokenizer IDs must be distinct")

    def component_manifest(self) -> dict[str, object]:
        return {
            "component_type": "router",
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "fixture_runtime": False,
            "revision": self.identity.router_revision,
            "snapshot_sha256": self.identity.router_snapshot_sha256,
            "weights_sha256": self.identity.router_weights_sha256,
            "tokenizer_sha256": self.identity.router_tokenizer_sha256,
        }

    def estimate_input_tokens(self, question: str) -> int:
        encoded = self._encode_question(question)
        return int(encoded["input_ids"].numel())

    def _encode_question(self, question: str) -> dict[str, Any]:
        options: dict[str, Any] = dict(return_tensors="pt", truncation=True)
        if self.max_input_tokens is not None:
            options["max_length"] = self.max_input_tokens
        return self.tokenizer(question, **options)

    def first_decoder_step(self, question: str, *, seed: int) -> RouterResult:
        with self._lock:
            return self._first_decoder_step(question, seed=seed)

    def _first_decoder_step(self, question: str, *, seed: int) -> RouterResult:
        torch = self._torch
        torch.manual_seed(seed)
        encoded = self._encode_question(question)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        first_scores = generated.scores[0][0]
        logits = {
            label: float(first_scores[token_id].detach().cpu().item())
            for label, token_id in self.label_token_ids.items()
        }
        return RouterResult(
            logits=logits,
            input_tokens=int(encoded["input_ids"].numel()),
            output_tokens=1,
        )


__all__ = ["LocalTransformersRouter", "OpenAICompatibleGenerator", "ToolkitRetriever"]
