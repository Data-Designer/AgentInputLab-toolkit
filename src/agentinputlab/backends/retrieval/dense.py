"""Auditable in-memory dense retrieval with lazy SentenceTransformers loading.

The implementation deliberately separates a *profile identity* from weight
availability.  The package ships model repository names, immutable revisions,
and prompt contracts, but it never bundles or silently downloads checkpoints.
By default, the SentenceTransformers loader is restricted to an existing local
cache.  Tests and institutional deployments may instead inject a compatible
encoder with an explicit fingerprint.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.embedding.base import (
    EmbeddingBackend,
    EmbeddingRequest,
    EmbeddingResponse,
    validate_embedding_response,
)
from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .base import Document, RetrievalResult, Retriever

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_DENSE_BACKEND = "sentence_transformers_dense"
_BM25_BACKEND = "memory_bm25"


@dataclass(frozen=True, slots=True)
class RetrieverProfile:
    """Immutable retriever configuration used in comparison fingerprints.

    ``weights_bundled`` and ``weight_files_verified`` are intentionally fixed
    to ``False`` for the model profiles below.  A repository revision is useful
    provenance, but it is not a substitute for hashing a user's local weight
    files.
    """

    profile_id: str
    backend: str
    model_id: str = ""
    model_revision: str = ""
    query_prefix: str = ""
    document_prefix: str = ""
    expected_dimensions: int | None = None
    metric: str = "cosine"
    tokenizer_padding_side: str | None = None
    license_spdx: str = "NOASSERTION"
    source_url: str = ""
    weights_bundled: bool = False
    weight_files_verified: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id or not self.backend:
            raise ValidationError("retriever profile requires profile_id and backend")
        if self.backend not in {_BM25_BACKEND, _DENSE_BACKEND}:
            raise ValidationError(f"unsupported retriever profile backend: {self.backend}")
        if self.weights_bundled or self.weight_files_verified:
            raise ValidationError("built-in retriever profiles must not claim bundled weights")
        if self.backend == _BM25_BACKEND:
            if self.model_id or self.model_revision or self.expected_dimensions is not None:
                raise ValidationError("BM25 profile must not declare dense model fields")
            if self.metric != "bm25":
                raise ValidationError("BM25 profile metric must be bm25")
            return
        if not self.model_id or not _HEX40.fullmatch(self.model_revision):
            raise ValidationError("dense profile requires a model repository and 40-hex revision")
        if (
            self.expected_dimensions is None
            or isinstance(self.expected_dimensions, bool)
            or not isinstance(self.expected_dimensions, int)
            or self.expected_dimensions < 1
        ):
            raise ValidationError("dense profile requires positive expected_dimensions")
        if self.metric != "cosine":
            raise ValidationError("SentenceTransformer dense profiles use cosine")
        if self.tokenizer_padding_side not in {None, "left", "right"}:
            raise ValidationError("tokenizer_padding_side must be left, right, or None")


_PROFILE_ITEMS = (
    RetrieverProfile(
        profile_id="sparse.bm25.memory.v1",
        backend=_BM25_BACKEND,
        metric="bm25",
        license_spdx="MIT",
        source_url="https://github.com/Data-Designer/AgentInputLab",
    ),
    RetrieverProfile(
        profile_id="dense.bge.base_en_v1_5",
        backend=_DENSE_BACKEND,
        model_id="BAAI/bge-base-en-v1.5",
        model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        query_prefix="Represent this sentence for searching relevant passages: ",
        expected_dimensions=768,
        license_spdx="MIT",
        source_url="https://huggingface.co/BAAI/bge-base-en-v1.5",
    ),
    RetrieverProfile(
        profile_id="dense.bge.m3",
        backend=_DENSE_BACKEND,
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        expected_dimensions=1024,
        license_spdx="MIT",
        source_url="https://huggingface.co/BAAI/bge-m3",
    ),
    RetrieverProfile(
        profile_id="dense.e5.base_v2",
        backend=_DENSE_BACKEND,
        model_id="intfloat/e5-base-v2",
        model_revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        query_prefix="query: ",
        document_prefix="passage: ",
        expected_dimensions=768,
        license_spdx="MIT",
        source_url="https://huggingface.co/intfloat/e5-base-v2",
    ),
    RetrieverProfile(
        profile_id="dense.e5.large_v2",
        backend=_DENSE_BACKEND,
        model_id="intfloat/e5-large-v2",
        model_revision="f169b11e22de13617baa190a028a32f3493550b6",
        query_prefix="query: ",
        document_prefix="passage: ",
        expected_dimensions=1024,
        license_spdx="MIT",
        source_url="https://huggingface.co/intfloat/e5-large-v2",
    ),
    RetrieverProfile(
        profile_id="dense.e5.multilingual_large",
        backend=_DENSE_BACKEND,
        model_id="intfloat/multilingual-e5-large",
        model_revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        query_prefix="query: ",
        document_prefix="passage: ",
        expected_dimensions=1024,
        source_url="https://huggingface.co/intfloat/multilingual-e5-large",
    ),
    RetrieverProfile(
        profile_id="dense.qwen3.embedding_0_6b",
        backend=_DENSE_BACKEND,
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer "
            "the query\nQuery:"
        ),
        expected_dimensions=1024,
        tokenizer_padding_side="left",
        license_spdx="Apache-2.0",
        source_url="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    ),
    RetrieverProfile(
        profile_id="dense.contriever.msmarco",
        backend=_DENSE_BACKEND,
        model_id="facebook/contriever-msmarco",
        model_revision="abe8c1493371369031bcb1e02acb754cf4e162fa",
        expected_dimensions=768,
        license_spdx="CC-BY-NC-4.0",
        source_url="https://huggingface.co/facebook/contriever-msmarco",
    ),
)

RETRIEVER_PROFILES: Mapping[str, RetrieverProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _PROFILE_ITEMS}
)
DENSE_RETRIEVER_PROFILE_IDS = tuple(
    profile.profile_id for profile in _PROFILE_ITEMS if profile.backend == _DENSE_BACKEND
)


def retriever_profile(profile_id: str) -> RetrieverProfile:
    """Return one built-in profile, failing closed on an unknown identity."""

    try:
        return RETRIEVER_PROFILES[profile_id]
    except KeyError as exc:
        raise ValidationError(f"unknown retriever profile: {profile_id}") from exc


@runtime_checkable
class BatchTextEncoder(Protocol):
    """Minimal SentenceTransformers-compatible injection seam."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any:
        """Encode one bounded batch in input order."""


@dataclass(frozen=True, slots=True)
class EmbeddingBackendEncoder:
    """Adapt an AgentInputLab embedding backend to the dense-retriever seam.

    The backend may be a local callable, SDK/RPC bridge, replay fixture, or
    self-hosted/API client. Its secret-free fingerprint is captured once and
    becomes part of the retriever identity. Prompt prefixes remain the selected
    :class:`RetrieverProfile`'s responsibility and are applied before this
    adapter is called.
    """

    backend: EmbeddingBackend
    dimensions: int | None = None
    request_metadata: Mapping[str, Any] = field(default_factory=dict)
    _backend_identity: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, EmbeddingBackend):
            raise ValidationError("embedding encoder requires an EmbeddingBackend")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValidationError("embedding encoder dimensions must be positive or None")
        identity = self.backend.fingerprint()
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError("embedding backend fingerprint must be a non-empty mapping")
        try:
            frozen_identity = freeze_mapping(identity)
            frozen_metadata = freeze_mapping(self.request_metadata)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "embedding encoder identity must be canonically serializable"
            ) from exc
        object.__setattr__(self, "_backend_identity", frozen_identity)
        object.__setattr__(self, "request_metadata", frozen_metadata)

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> tuple[tuple[float, ...], ...]:
        del batch_size, show_progress_bar, convert_to_numpy
        self._require_stable_backend_identity()
        request = EmbeddingRequest(
            texts=tuple(sentences),
            normalize=normalize_embeddings,
            dimensions=self.dimensions,
            metadata=self.request_metadata,
        )
        response = self.backend.embed(request)
        if not isinstance(response, EmbeddingResponse):
            raise ValidationError("embedding backend returned a non-EmbeddingResponse")
        return validate_embedding_response(
            request,
            response,
            backend_dimensions=self.dimensions,
        ).vectors

    def fingerprint(self) -> Mapping[str, Any]:
        self._require_stable_backend_identity()
        return {
            "implementation": "agentinputlab.embedding_backend_encoder.v1",
            "backend_identity": self._backend_identity,
            "dimensions": self.dimensions,
            "request_metadata_hash": content_hash(self.request_metadata),
        }

    def _require_stable_backend_identity(self) -> None:
        identity = self.backend.fingerprint()
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError("embedding backend fingerprint must be a non-empty mapping")
        try:
            current = freeze_mapping(identity)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "embedding encoder identity must be canonically serializable"
            ) from exc
        if current != self._backend_identity:
            raise ValidationError("embedding backend fingerprint changed after adapter creation")


@dataclass(frozen=True, slots=True)
class DenseIndexSummary:
    document_count: int
    dimensions: int
    vector_hash: str


class SentenceTransformerDenseRetriever(Retriever):
    """Batch-index documents and rank them by deterministic cosine similarity.

    Construction and :meth:`fingerprint` never import SentenceTransformers or
    touch the network.  The first :meth:`build_index` or :meth:`retrieve` call
    loads the pinned model.  ``local_files_only=True`` is the safe default; set
    it to ``False`` explicitly if downloading the pinned revision is intended.
    """

    __slots__ = (
        "_batch_size",
        "_dependency_version",
        "_device",
        "_documents",
        "_encoder",
        "_encoder_identity",
        "_encoder_identity_binding",
        "_encoder_identity_probe",
        "_index_summary",
        "_local_files_only",
        "_model_path",
        "_profile",
        "_retriever_id",
        "_runtime_kind",
        "_vectors",
    )

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        profile: RetrieverProfile | str,
        batch_size: int = 32,
        encoder: BatchTextEncoder | None = None,
        encoder_identity: Mapping[str, Any] | None = None,
        device: str | None = None,
        local_files_only: bool = True,
        model_path: Path | str | None = None,
        retriever_id: str | None = None,
    ) -> None:
        resolved = retriever_profile(profile) if isinstance(profile, str) else profile
        if not isinstance(resolved, RetrieverProfile) or resolved.backend != _DENSE_BACKEND:
            raise ValidationError("SentenceTransformerDenseRetriever requires a dense profile")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValidationError("dense retriever batch_size must be a positive integer")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValidationError("dense retriever device must be non-empty or None")
        if not isinstance(local_files_only, bool):
            raise ValidationError("local_files_only must be boolean")
        resolved_model_path: Path | None = None
        if model_path is not None:
            try:
                resolved_model_path = Path(model_path)
            except TypeError as exc:
                raise ValidationError("dense model_path must be path-like or None") from exc
            if not resolved_model_path.is_absolute():
                raise ValidationError("dense model_path must be absolute")
        prepared_documents = tuple(documents)
        if not all(isinstance(document, Document) for document in prepared_documents):
            raise ValidationError("dense retriever documents must be Document instances")
        document_ids = tuple(document.doc_id for document in prepared_documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValidationError("dense retriever document IDs must be unique")
        if encoder is None and encoder_identity is not None:
            raise ValidationError("encoder_identity is only valid with an injected encoder")
        if encoder is not None and not isinstance(encoder, BatchTextEncoder):
            raise ValidationError("injected dense encoder must implement encode()")
        if encoder is not None:
            identity = encoder_identity
            identity_probe = None
            if identity is None:
                fingerprint = getattr(encoder, "fingerprint", None)
                identity = fingerprint() if callable(fingerprint) else None
                identity_probe = fingerprint if callable(fingerprint) else None
            if not isinstance(identity, Mapping) or not identity:
                raise ValidationError("injected dense encoder requires a non-empty identity")
            try:
                frozen_encoder_identity = freeze_mapping(identity)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "injected dense encoder identity must be canonically serializable"
                ) from exc
            identity_binding = (
                "derived_fingerprint_monitored"
                if identity_probe is not None
                else "explicit_identity_unverified"
            )
            runtime_kind = "injected_encoder_unverified"
            dependency_version = "injected"
        else:
            frozen_encoder_identity = freeze_mapping(
                {
                    "model_id": resolved.model_id,
                    "model_revision": resolved.model_revision,
                    "loader": "sentence_transformers.SentenceTransformer",
                }
            )
            identity_probe = None
            identity_binding = "pinned_profile_revision"
            runtime_kind = "sentence_transformers_lazy"
            dependency_version = _distribution_version("sentence-transformers")

        resolved_retriever_id = retriever_id or f"dense:{resolved.profile_id}"
        if not resolved_retriever_id:
            raise ValidationError("dense retriever_id must be non-empty")
        self._documents = prepared_documents
        self._profile = resolved
        self._batch_size = batch_size
        self._device = device
        self._local_files_only = local_files_only
        self._model_path = resolved_model_path
        self._retriever_id = resolved_retriever_id
        self._encoder = encoder
        self._encoder_identity = frozen_encoder_identity
        self._encoder_identity_binding = identity_binding
        self._encoder_identity_probe = identity_probe
        self._runtime_kind = runtime_kind
        self._dependency_version = dependency_version
        self._vectors: tuple[tuple[float, ...], ...] | None = None
        self._index_summary: DenseIndexSummary | None = None

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._documents

    @property
    def profile(self) -> RetrieverProfile:
        return self._profile

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def local_files_only(self) -> bool:
        return self._local_files_only

    @property
    def retriever_id(self) -> str:
        return self._retriever_id

    @property
    def embedder_id(self) -> str:
        """Identity exposed through the paper-level ``DenseEmbedder`` seam."""

        return f"{self._retriever_id}:embedder"

    @property
    def searcher_id(self) -> str:
        """Identity exposed through the paper-level ``VectorSearcher`` seam."""

        return f"{self._retriever_id}:index"

    @property
    def indexed(self) -> bool:
        return self._vectors is not None

    def build_index(self) -> DenseIndexSummary:
        """Encode the fixed corpus in bounded batches exactly once."""

        if self._index_summary is not None:
            return self._index_summary
        if not self.documents:
            summary = DenseIndexSummary(
                document_count=0,
                dimensions=self.profile.expected_dimensions or 0,
                vector_hash=content_hash(()),
            )
            self._vectors = ()
            self._index_summary = summary
            return summary
        inputs = tuple(self._document_input(document) for document in self.documents)
        vectors = self._encode_batched(inputs, role="documents")
        summary = DenseIndexSummary(
            document_count=len(self.documents),
            dimensions=len(vectors[0]),
            vector_hash=content_hash(vectors),
        )
        self._vectors = vectors
        self._index_summary = summary
        return summary

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        if not isinstance(query, str):
            raise ValidationError("dense retrieval query must be text")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValidationError("dense retrieval top_k must be an integer")
        if top_k < 1 or not self.documents:
            return ()
        self.build_index()
        assert self._vectors is not None  # established by build_index
        query_vector = self._encode_batched(
            (f"{self.profile.query_prefix}{query}",),
            role="query",
        )[0]
        return self.search(query_vector, top_k=top_k)

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Encode query surrogates for HyDE using the profile's query contract."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValidationError("dense embed texts must be a sequence of strings")
        if any(not isinstance(text, str) for text in texts):
            raise ValidationError("dense embed texts must contain only strings")
        return self._encode_batched(
            tuple(f"{self.profile.query_prefix}{text}" for text in texts),
            role="query",
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        """Search the fixed corpus with an already-computed dense query vector."""

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValidationError("dense vector-search top_k must be an integer")
        if top_k < 1 or not self.documents:
            return ()
        self.build_index()
        assert self._vectors is not None
        try:
            prepared = tuple(float(value) for value in vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("dense vector-search query must be numeric") from exc
        expected = self.profile.expected_dimensions
        if len(prepared) != expected:
            raise ValidationError(
                f"dense vector-search query dimension does not match profile {expected}"
            )
        query_vector = _normalized_vector(prepared, role="query")
        scored = [
            (sum(left * right for left, right in zip(query_vector, item, strict=True)), document)
            for document, item in zip(self.documents, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].doc_id))
        return tuple(
            RetrievalResult(
                document=document,
                score=score,
                rank=rank,
                features={"cosine_score": score},
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        )

    def fingerprint(self) -> Mapping[str, Any]:
        """Return a stable recipe identity without loading model weights."""

        self._require_stable_encoder_identity()
        fingerprint = {
            "retriever_id": self.retriever_id,
            "implementation": "agentinputlab.sentence_transformer_dense.v3",
            "profile_id": self.profile.profile_id,
            "model_id": self.profile.model_id,
            "model_revision": self.profile.model_revision,
            "expected_dimensions": self.profile.expected_dimensions,
            "metric": "cosine",
            "normalization": "agentinputlab.l2.v1",
            "query_prefix": self.profile.query_prefix,
            "document_prefix": self.profile.document_prefix,
            "document_format": "title_newline_text.v1",
            "tokenizer_padding_side": self.profile.tokenizer_padding_side,
            "corpus_hash": content_hash(self.documents),
            "document_count": len(self.documents),
            "batch_size": self.batch_size,
            "device": self.device or "auto",
            "local_files_only": self.local_files_only,
            "model_source": "local_snapshot" if self._model_path is not None else "profile_cache",
            "runtime_kind": self._runtime_kind,
            "encoder_identity_hash": content_hash(self._encoder_identity),
            "encoder_identity_binding": self._encoder_identity_binding,
            "sentence_transformers_version": self._dependency_version,
            "weights_bundled": False,
            "weight_files_verified": False,
        }
        self._require_stable_encoder_identity()
        return fingerprint

    def _document_input(self, document: Document) -> str:
        body = f"{document.title}\n{document.text}" if document.title else document.text
        return f"{self.profile.document_prefix}{body}"

    def _encoder_instance(self) -> BatchTextEncoder:
        if self._encoder is not None:
            return self._encoder
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise ExternalDependencyError(
                "SentenceTransformers is unavailable; install agentinputlab[dense-retrieval]"
            ) from exc
        constructor = getattr(sentence_transformers, "SentenceTransformer", None)
        if not callable(constructor):
            raise ExternalDependencyError(
                "sentence_transformers does not expose SentenceTransformer"
            )
        kwargs: dict[str, Any] = {
            "revision": self.profile.model_revision,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.profile.tokenizer_padding_side is not None:
            kwargs["tokenizer_kwargs"] = {
                "padding_side": self.profile.tokenizer_padding_side,
            }
        try:
            model_source = (
                str(self._model_path) if self._model_path is not None else self.profile.model_id
            )
            encoder = constructor(model_source, **kwargs)
        except Exception as exc:
            raise ExternalDependencyError(
                "the pinned dense model is unavailable; prefetch the exact revision or "
                "explicitly allow a download"
            ) from exc
        if not isinstance(encoder, BatchTextEncoder):
            raise ExternalDependencyError("loaded SentenceTransformer has no compatible encode()")
        self._encoder = encoder
        return encoder

    def _encode_batched(
        self,
        inputs: Sequence[str],
        *,
        role: str,
    ) -> tuple[tuple[float, ...], ...]:
        self._require_stable_encoder_identity()
        encoder = self._encoder_instance()
        rows: list[tuple[float, ...]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = tuple(inputs[start : start + self.batch_size])
            self._require_stable_encoder_identity()
            try:
                raw = encoder.encode(
                    batch,
                    batch_size=len(batch),
                    normalize_embeddings=False,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except ValidationError:
                raise
            except Exception as exc:
                raise ExternalDependencyError(
                    f"dense encoder failed while encoding {role}"
                ) from exc
            self._require_stable_encoder_identity()
            try:
                converted = tuple(tuple(float(value) for value in vector) for vector in raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(f"dense encoder returned invalid {role} vectors") from exc
            if len(converted) != len(batch):
                raise ValidationError(f"dense encoder changed the {role} batch cardinality")
            rows.extend(converted)
        expected = self.profile.expected_dimensions
        if any(len(vector) != expected for vector in rows):
            raise ValidationError(
                f"dense encoder output dimension does not match profile {expected}"
            )
        normalized = tuple(_normalized_vector(vector, role=role) for vector in rows)
        return normalized

    def _require_stable_encoder_identity(self) -> None:
        probe = self._encoder_identity_probe
        if probe is None:
            return
        try:
            identity = probe()
        except Exception as exc:
            raise ValidationError("injected dense encoder fingerprint failed") from exc
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError("injected dense encoder fingerprint must be a non-empty mapping")
        try:
            current = freeze_mapping(identity)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "injected dense encoder identity must be canonically serializable"
            ) from exc
        if current != self._encoder_identity:
            raise ValidationError("injected dense encoder fingerprint changed after construction")


def build_retriever(
    profile_id: str,
    documents: Iterable[Document],
    **options: Any,
) -> Retriever:
    """Construct a built-in retriever profile without caller-side dispatch.

    BM25 and dense profiles deliberately accept disjoint option sets. Unknown
    or cross-family options fail closed instead of being silently ignored.
    Dense callers may inject either a ``BatchTextEncoder`` or an
    ``EmbeddingBackend``; the latter is wrapped with
    :class:`EmbeddingBackendEncoder` and fingerprinted as part of the runtime.
    """

    profile = retriever_profile(profile_id)
    if profile.backend == _BM25_BACKEND:
        allowed = {"retriever_id", "k1", "b", "tokenizer", "tokenizer_id"}
        _require_known_options(profile_id, options, allowed)
        from .memory import InMemoryBM25Retriever

        bm25_options = dict(options)
        bm25_options.setdefault("retriever_id", f"bm25:{profile_id}")
        return InMemoryBM25Retriever(documents=documents, **bm25_options)

    allowed = {
        "batch_size",
        "device",
        "embedding_backend",
        "embedding_request_metadata",
        "encoder",
        "encoder_identity",
        "local_files_only",
        "model_path",
        "retriever_id",
    }
    _require_known_options(profile_id, options, allowed)
    dense_options = dict(options)
    embedding_backend = dense_options.pop("embedding_backend", None)
    request_metadata = dense_options.pop("embedding_request_metadata", None)
    if embedding_backend is not None:
        if dense_options.get("encoder") is not None:
            raise ValidationError("dense retriever accepts encoder or embedding_backend, not both")
        if dense_options.get("encoder_identity") is not None:
            raise ValidationError(
                "encoder_identity is derived from embedding_backend and must not be supplied"
            )
        dense_options["encoder"] = EmbeddingBackendEncoder(
            backend=embedding_backend,
            dimensions=profile.expected_dimensions,
            request_metadata=request_metadata or {},
        )
    elif request_metadata is not None:
        raise ValidationError("embedding_request_metadata requires embedding_backend")
    return SentenceTransformerDenseRetriever(documents, profile=profile, **dense_options)


def _require_known_options(
    profile_id: str,
    options: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValidationError(
            f"retriever profile {profile_id} does not accept option(s): {', '.join(unknown)}"
        )


def _normalized_vector(vector: tuple[float, ...], *, role: str) -> tuple[float, ...]:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValidationError(f"dense encoder returned a non-finite or empty {role} vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValidationError(f"dense encoder returned a zero-norm {role} vector")
    return tuple(value / norm for value in vector)


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


__all__ = [
    "DENSE_RETRIEVER_PROFILE_IDS",
    "RETRIEVER_PROFILES",
    "BatchTextEncoder",
    "DenseIndexSummary",
    "EmbeddingBackendEncoder",
    "RetrieverProfile",
    "SentenceTransformerDenseRetriever",
    "build_retriever",
    "retriever_profile",
]
