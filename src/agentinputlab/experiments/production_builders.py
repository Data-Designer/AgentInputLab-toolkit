"""Production component builders for static cells and official external arenas.

The static experiment driver deliberately accepts operator-owned builders
instead of guessing deployments.  This module provides a strict, reusable
implementation of the common production builders:

* :class:`ProductionBackendBuilder` creates a checksum-bound Replay,
  OpenAI-compatible, or local Transformers backend.  It may also attach a
  benchmark-keyed set of checksum-bound retrievers or one checksum-bound
  dense embedder/vector-search runtime for acquisition methods such as HyDE.
* :class:`PreparedCorpusRetrieverBuilder` builds BM25 or dense retrieval over
  an explicit normalized corpus and checksum-locked local model assets.
* :class:`ProductionPredictorBuilder` creates the evaluator-blind downstream
  predictor used after an intervention.  Retrieval cells emit ranked document
  scores; QA cells generate a strict answer/support/evidence JSON object from
  visible state only.

Construction and preflight never contact a model endpoint or download model
weights.  Secrets are accepted only through environment-variable names.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.llm import (
    GenerationRequest,
    GenerationResponse,
    LLMBackend,
    ModelCapability,
    OpenAICompatibleBackend,
    ReplayBackend,
    TransformersBackend,
    Usage,
    require_capabilities,
)
from agentinputlab.backends.retrieval import (
    Document,
    RetrievalResult,
    Retriever,
    build_retriever,
    retriever_profile,
)
from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher
from agentinputlab.benchmarks import (
    DownstreamPrediction,
    QAPrediction,
    RetrievalPrediction,
)
from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS
from agentinputlab.benchmarks.production import production_benchmark_specs
from agentinputlab.connectors.tools import SchemaOnlyToolConnector
from agentinputlab.contracts.errors import (
    ArtifactError,
    CapabilityError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.serialization import (
    canonical_json,
    content_hash,
    freeze_mapping,
    to_jsonable,
)
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.experiments.drivers.static_production import (
    StaticBackendBuildRequest,
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
    static_predictor_product_sha256,
    static_runtime_product_sha256,
)
from agentinputlab.experiments.loader import resolve_driver_entrypoint
from agentinputlab.observability import ResourceUsage

_SCHEMA_VERSION = 1
_BACKEND_IMPLEMENTATION = "agentinputlab.production_backend_builder.v4"
_PREDICTOR_IMPLEMENTATION = "agentinputlab.production_downstream_predictor.v7"
_CORPUS_RETRIEVER_IMPLEMENTATION = "agentinputlab.prepared_corpus_retriever_builder.v1"
_RETRIEVER_ROUTER_IMPLEMENTATION = "agentinputlab.production_retriever_router.v2"
_TRANSFORMERS_IMPLEMENTATION = "agentinputlab.local_only_transformers.v2"
_QUERYGYM_RUNTIME_IMPLEMENTATION = "agentinputlab.querygym.pinned_runtime.v1"
_RECOMP_RUNTIME_IMPLEMENTATION = "agentinputlab.recomp.current_transformers_loader.v1"
_MAX_REPLAY_BYTES = 64 * 1024 * 1024
_MAX_CORPUS_LINE_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|authorization|password|secret|"
    r"access[_-]?token|bearer|credential)(?:$|[_-])"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
)
_SENSITIVE_HEADER = re.compile(
    r"(?i)^(?:authorization|proxy-authorization|x-api-key|api-key|"
    r"x-goog-api-key|x-auth-token|x-amz-security-token|cookie|set-cookie)$"
)

# These native sample adapters supply their own pinned source and scorer. The
# predictor only performs retrieval; this does not enable implicit downloads or
# turn a bounded sample into a full-corpus production dataset.
_NATIVE_SAMPLE_RETRIEVAL_CELLS = frozenset({"trec_dl.2019", "trec_dl.2020", "beir.cqadupstack"})
_CELL_SPLITS = MappingProxyType(
    {
        **{spec.cell_id: spec.split for spec in production_benchmark_specs()
           if spec.adapter_kind in {"qa", "retrieval"}},
        **{cell: "test" for cell in BEIR_ARCHIVE_LOCKS},
        **{cell: "test" for cell in _NATIVE_SAMPLE_RETRIEVAL_CELLS},
    }
)
_RETRIEVAL_CELLS = frozenset(
    {
        *(spec.cell_id for spec in production_benchmark_specs() if spec.adapter_kind == "retrieval"),
        *BEIR_ARCHIVE_LOCKS,
        *_NATIVE_SAMPLE_RETRIEVAL_CELLS,
    }
)
_QA_CELLS = frozenset(set(_CELL_SPLITS) - _RETRIEVAL_CELLS)
_INTERACTIVE_RETRIEVER_CELLS = frozenset(
    {
        "tau3.banking_knowledge",
        "tau3.retail",
        "state_bench.common_support",
        "toolsandbox.common_support",
        "tooltalk.full",
    }
)
_RETRIEVER_ROUTE_CELLS = frozenset({*_CELL_SPLITS, *_INTERACTIVE_RETRIEVER_CELLS})


@dataclass(frozen=True, slots=True)
class ProductionRetrieverBuildRequest:
    """Typed request passed to an operator or built-in retriever builder."""

    benchmark_id: str
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.benchmark_id, label="retriever benchmark_id")
        _require_text(self.model_slot, label="model_slot")
        _require_seed(self.seed)
        object.__setattr__(
            self,
            "builder_parameters",
            _canonical_mapping(self.builder_parameters, label="retriever builder parameters"),
        )


@runtime_checkable
class ProductionRetrieverBuilderProtocol(Protocol):
    """Builder seam used by :class:`ProductionRetrieverRouter`."""

    builder_kind: str

    def fingerprint(self) -> Mapping[str, Any]: ...

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]: ...

    def build(self, request: ProductionRetrieverBuildRequest) -> Retriever: ...


@dataclass(frozen=True, slots=True)
class LockedRetrieverSpec:
    """One retriever builder and its exact expected runtime product."""

    entrypoint: str
    fingerprint_sha256: str
    expected_product_sha256: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entrypoint, str) or _ENTRYPOINT.fullmatch(self.entrypoint) is None:
            raise ValidationError("retriever entrypoint must use module:factory syntax")
        _require_sha256(self.fingerprint_sha256, label="retriever fingerprint_sha256")
        _require_sha256(
            self.expected_product_sha256,
            label="retriever expected_product_sha256",
        )
        _reject_embedded_secrets(self.parameters, path="retriever parameters")
        object.__setattr__(
            self,
            "parameters",
            _canonical_mapping(self.parameters, label="retriever parameters"),
        )

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> LockedRetrieverSpec:
        data = _strict_mapping(
            value,
            path=path,
            allowed={
                "entrypoint",
                "fingerprint_sha256",
                "expected_product_sha256",
                "parameters",
            },
            required={
                "entrypoint",
                "fingerprint_sha256",
                "expected_product_sha256",
            },
        )
        return cls(
            entrypoint=_require_text(data["entrypoint"], label=f"{path}.entrypoint"),
            fingerprint_sha256=_require_text(
                data["fingerprint_sha256"],
                label=f"{path}.fingerprint_sha256",
            ),
            expected_product_sha256=_require_text(
                data["expected_product_sha256"],
                label=f"{path}.expected_product_sha256",
            ),
            parameters=_mapping_or_empty(data.get("parameters"), path=f"{path}.parameters"),
        )

    @property
    def binding_identity(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "entrypoint": self.entrypoint,
                "fingerprint_sha256": self.fingerprint_sha256,
                "expected_product_sha256": self.expected_product_sha256,
                "parameters_sha256": content_hash(self.parameters),
            }
        )


class _BoundLLMBackend:
    """Add an immutable operator binding without exposing secret values."""

    __slots__ = ("_binding", "_inner")

    def __init__(self, inner: LLMBackend, binding: Mapping[str, Any]) -> None:
        if not isinstance(inner, LLMBackend):
            raise TypeError("bound backend requires an LLMBackend")
        self._inner = inner
        self._binding = _canonical_mapping(binding, label="backend binding")

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self._inner.capabilities

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self._inner.generate(request)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": "agentinputlab.bound_llm_backend.v1",
            "backend": _fingerprint(self._inner, label="inner backend"),
            "binding": self._binding,
        }


class _LocalOnlyTransformersBackend(TransformersBackend):
    """Transformers backend bound to one complete local snapshot tree."""

    __slots__ = (
        "_artifact_files",
        "_model_path",
        "_snapshot_sha256",
        "local_files_only",
    )

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        model_revision: str,
        tokenizer_revision: str,
        artifact_files: tuple[Mapping[str, Any], ...],
        generation_kwargs: Mapping[str, Any],
        capabilities: frozenset[ModelCapability],
        local_files_only: bool,
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            generation_kwargs=generation_kwargs,
            capabilities=capabilities,
        )
        if not isinstance(local_files_only, bool):
            raise ValidationError("transformers local_files_only must be boolean")
        if not isinstance(model_path, Path) or not model_path.is_absolute():
            raise ValidationError("transformers model_path must be an absolute Path")
        self._model_path = model_path
        self._artifact_files = artifact_files
        self._snapshot_sha256 = _transformers_snapshot_sha256(artifact_files)
        self.local_files_only = local_files_only

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        _verify_transformers_snapshot(
            self._model_path,
            self._artifact_files,
        )
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ExternalDependencyError(
                "Transformers backend requires `pip install agentinputlab[local]`"
            ) from exc
        tokenizer_factory = getattr(transformers, "AutoTokenizer", None)
        model_factory = getattr(transformers, "AutoModelForCausalLM", None)
        if tokenizer_factory is None or model_factory is None:
            raise ExternalDependencyError(
                "transformers does not expose AutoTokenizer and AutoModelForCausalLM"
            )
        try:
            self.tokenizer = tokenizer_factory.from_pretrained(
                str(self._model_path),
                revision=self.tokenizer_revision,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
            self.model = model_factory.from_pretrained(
                str(self._model_path),
                revision=self.model_revision,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise ExternalDependencyError(
                "the checksum-bound Transformers deployment is unavailable locally"
            ) from exc

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            **dict(super().fingerprint()),
            "implementation": _TRANSFORMERS_IMPLEMENTATION,
            "local_files_only": self.local_files_only,
            "snapshot_sha256": self._snapshot_sha256,
            "trust_remote_code": False,
        }


class ProductionRetrieverRouter:
    """Lazy, benchmark-keyed retriever with an identity stable across binding.

    ``StaticBackendBuildRequest`` intentionally contains a model slot but not a
    benchmark ID.  The predictor builder therefore binds this router exactly
    once using the driver-validated benchmark ID.  The fingerprint covers all
    available routes and expected products and does not change after binding.
    """

    retriever_id = "production-retriever-router"

    def __init__(
        self,
        specs: Mapping[str, LockedRetrieverSpec],
        *,
        model_slot: str,
        seed: int,
    ) -> None:
        routes = dict(specs)
        if not routes:
            raise ValidationError("production retriever router requires at least one route")
        for benchmark_id, spec in routes.items():
            _require_retriever_cell(benchmark_id)
            if not isinstance(spec, LockedRetrieverSpec):
                raise TypeError("retriever route must contain LockedRetrieverSpec values")
        _require_text(model_slot, label="model_slot")
        _require_seed(seed)
        self._specs = MappingProxyType(routes)
        self._model_slot = model_slot
        self._seed = seed
        self._active_benchmark_id: str | None = None
        self._instances: dict[str, Retriever] = {}

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def has_route(self, benchmark_id: str) -> bool:
        return benchmark_id in self._specs

    def bind(self, benchmark_id: str) -> None:
        _require_retriever_cell(benchmark_id)
        if benchmark_id not in self._specs:
            raise CapabilityError(
                f"runtime has no checksum-bound retriever route for {benchmark_id!r}"
            )
        if self._active_benchmark_id is not None and self._active_benchmark_id != benchmark_id:
            raise ValidationError("one runtime retriever router cannot be rebound to another cell")
        self._active_benchmark_id = benchmark_id

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        if self._active_benchmark_id is None:
            raise CapabilityError(
                "production retriever route is unbound; build a benchmark predictor first"
            )
        retriever = self._retriever_for(self._active_benchmark_id)
        return retriever.retrieve(query, top_k=top_k)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": _RETRIEVER_ROUTER_IMPLEMENTATION,
            "retriever_id": self.retriever_id,
            "binding_policy": "driver_validated_benchmark_id.v1",
            "routes": {
                benchmark_id: spec.binding_identity
                for benchmark_id, spec in sorted(self._specs.items())
            },
        }

    def _retriever_for(self, benchmark_id: str) -> Retriever:
        cached = self._instances.get(benchmark_id)
        if cached is not None:
            return cached
        spec = self._specs[benchmark_id]
        builder = _resolve_retriever_builder(spec)
        product = builder.build(
            ProductionRetrieverBuildRequest(
                benchmark_id=benchmark_id,
                model_slot=self._model_slot,
                seed=self._seed,
                builder_parameters=spec.parameters,
            )
        )
        if not isinstance(product, Retriever):
            raise ValidationError(
                "production retriever builder must return a Retriever, "
                f"got {type(product).__name__}"
            )
        observed = retriever_product_sha256(product)
        _validate_product_hash(
            observed,
            spec.expected_product_sha256,
            label=f"retriever route {benchmark_id}",
        )
        self._instances[benchmark_id] = product
        return product


class _BoundRetriever:
    """Bind local corpus/model receipts to a built-in retriever fingerprint."""

    __slots__ = ("_binding", "_inner")

    def __init__(self, inner: Retriever, binding: Mapping[str, Any]) -> None:
        if not isinstance(inner, Retriever):
            raise TypeError("bound retriever requires a Retriever")
        self._inner = inner
        self._binding = _canonical_mapping(binding, label="retriever binding")

    @property
    def retriever_id(self) -> str:
        return self._inner.retriever_id

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        return self._inner.retrieve(query, top_k=top_k)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": "agentinputlab.bound_retriever.v1",
            "retriever": _fingerprint(self._inner, label="inner retriever"),
            "binding": self._binding,
        }


class _BoundVectorRuntime(_BoundRetriever):
    """Bind one dense embedder/index pair to corpus and model asset receipts."""

    def __init__(self, inner: Retriever, binding: Mapping[str, Any]) -> None:
        super().__init__(inner, binding)
        if not isinstance(inner, DenseEmbedder) or not isinstance(inner, VectorSearcher):
            raise TypeError("bound vector runtime requires dense embedder and searcher protocols")

    @property
    def embedder_id(self) -> str:
        return self._inner.embedder_id

    @property
    def searcher_id(self) -> str:
        return self._inner.searcher_id

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._inner.embed(texts)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        return self._inner.search(vector, top_k=top_k)


class ProductionBackendBuilder:
    """Strict backend/runtime builder used by static production experiments."""

    builder_kind = "backend"

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": _BACKEND_IMPLEMENTATION,
            "schema_version": _SCHEMA_VERSION,
            "backend_kinds": ("openai_compatible", "replay", "transformers"),
            "tool_connector_kinds": ("schema_only",),
            "external_runtime_kinds": (
                "querygym_query2doc",
                "qrag_common_support",
                "r2ag_common_support",
                "ask_or_assume_worker",
                "acon_observation_worker",
                "reformer_worker",
                "infocqr_worker",
                "recomp_extractive_worker",
                "smartsearch_worker",
            ),
            "secret_policy": "environment_indirection_only.v1",
            "network_policy": "no_preflight_or_construction_network.v1",
            "retriever_binding": _RETRIEVER_ROUTER_IMPLEMENTATION,
            "vector_runtime_binding": "checksum_locked_dense_embedder_and_index.v1",
            "recomp_compatibility_loader": _RECOMP_RUNTIME_IMPLEMENTATION,
        }

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            config = _parse_backend_parameters(parameters)
        except (ArtifactError, TypeError, ValueError) as exc:
            return (f"invalid production backend configuration: {exc}",)
        return tuple(_backend_preflight_blockers(config))

    def build(self, request: StaticBackendBuildRequest) -> StaticRuntimeComponents:
        if not isinstance(request, StaticBackendBuildRequest):
            raise TypeError("production backend builder requires StaticBackendBuildRequest")
        config = _parse_backend_parameters(request.builder_parameters)
        blockers = _backend_preflight_blockers(config)
        if blockers:
            raise ArtifactError("production backend preflight failed: " + "; ".join(blockers))
        backend = _build_llm_backend(config["backend"])
        required = config["required_capabilities"]
        require_capabilities(backend, required)
        external_runtime = _build_external_runtime(config["external_runtime"], backend)
        retriever_specs = config["retrievers"]
        retriever = (
            ProductionRetrieverRouter(
                retriever_specs,
                model_slot=request.model_slot,
                seed=request.seed,
            )
            if retriever_specs
            else None
        )
        vector_runtime = _build_locked_vector_runtime(
            config["vector_runtime"],
            model_slot=request.model_slot,
            seed=request.seed,
        )
        tools = _build_tool_connector(config["tools"])
        return StaticRuntimeComponents(
            backend=backend,
            dense_embedder=vector_runtime,
            retriever=retriever,
            vector_searcher=vector_runtime,
            tools=tools,
            external_runtime=external_runtime,
            run_config=config["run_config"],
            strict_intent=config["strict_intent"],
        )

    def product_sha256(self, request: StaticBackendBuildRequest) -> str:
        """Build, validate, and hash the exact runtime product."""

        return static_runtime_product_sha256(self.build(request))


def _build_locked_vector_runtime(
    spec: LockedRetrieverSpec | None,
    *,
    model_slot: str,
    seed: int,
) -> DenseEmbedder | VectorSearcher | None:
    if spec is None:
        return None
    benchmark_id = _require_text(
        spec.parameters.get("benchmark_id"),
        label="vector runtime benchmark_id",
    )
    builder = _resolve_retriever_builder(spec)
    product = builder.build(
        ProductionRetrieverBuildRequest(
            benchmark_id=benchmark_id,
            model_slot=model_slot,
            seed=seed,
            builder_parameters=spec.parameters,
        )
    )
    if not isinstance(product, DenseEmbedder) or not isinstance(product, VectorSearcher):
        raise ValidationError(
            "production vector runtime must implement DenseEmbedder and VectorSearcher"
        )
    observed = retriever_product_sha256(product)
    _validate_product_hash(observed, spec.expected_product_sha256, label="vector runtime")
    return product


def _build_tool_connector(config: Mapping[str, Any] | None) -> SchemaOnlyToolConnector | None:
    if config is None:
        return None
    if config["kind"] != "schema_only":  # pragma: no cover - parser invariant
        raise ValidationError("unsupported production tool connector kind")
    return SchemaOnlyToolConnector.from_openai_definitions(
        config["definitions"],
        connector_id=config["connector_id"],
    )


class PreparedCorpusRetrieverBuilder:
    """Build BM25/BGE/E5/Qwen/Contriever over one locked normalized corpus."""

    builder_kind = "retriever"

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": _CORPUS_RETRIEVER_IMPLEMENTATION,
            "schema_version": _SCHEMA_VERSION,
            "corpus_schema": "id-title-text.jsonl.v1",
            "network_policy": "local_files_only.v1",
            "retriever_factory": "agentinputlab.build_retriever.v1",
        }

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            config = _parse_corpus_retriever_parameters(parameters)
        except (ArtifactError, TypeError, ValueError) as exc:
            return (f"invalid prepared corpus retriever configuration: {exc}",)
        blockers: list[str] = []
        try:
            documents = _load_locked_corpus(config)
            if len(documents) != config["expected_document_count"]:
                blockers.append(
                    "prepared corpus document count mismatch: "
                    f"expected {config['expected_document_count']}, got {len(documents)}"
                )
        except (ArtifactError, OSError, UnicodeError, ValueError) as exc:
            blockers.append(str(exc))
        profile = retriever_profile(config["profile_id"])
        if profile.backend == "sentence_transformers_dense":
            if importlib.util.find_spec("sentence_transformers") is None:
                blockers.append(
                    "sentence-transformers is unavailable; install agentinputlab[dense-retrieval]"
                )
            if not config["artifact_files"]:
                blockers.append(
                    "dense production retrieval requires checksum-locked artifact_files"
                )
        for artifact in config["artifact_files"]:
            try:
                _verify_checksum_file(
                    artifact["path"],
                    artifact["sha256"],
                    label="retriever artifact",
                )
            except ArtifactError as exc:
                blockers.append(str(exc))
        return tuple(blockers)

    def build(self, request: ProductionRetrieverBuildRequest) -> Retriever:
        if not isinstance(request, ProductionRetrieverBuildRequest):
            raise TypeError(
                "prepared corpus retriever builder requires ProductionRetrieverBuildRequest"
            )
        config = _parse_corpus_retriever_parameters(request.builder_parameters)
        if config["benchmark_id"] != request.benchmark_id:
            raise ValidationError(
                "prepared corpus retriever benchmark differs from the requested route"
            )
        blockers = self.preflight(request.builder_parameters)
        if blockers:
            raise ArtifactError(
                "prepared corpus retriever preflight failed: " + "; ".join(blockers)
            )
        documents = _load_locked_corpus(config)
        retriever = build_retriever(
            config["profile_id"],
            documents,
            **dict(config["options"]),
        )
        bound_type = (
            _BoundVectorRuntime
            if retriever_profile(config["profile_id"]).backend == "sentence_transformers_dense"
            else _BoundRetriever
        )
        return bound_type(
            retriever,
            {
                "builder": _CORPUS_RETRIEVER_IMPLEMENTATION,
                "benchmark_id": request.benchmark_id,
                "profile_id": config["profile_id"],
                "corpus_sha256": config["corpus_sha256"],
                "corpus_revision": config["corpus_revision"],
                "document_count": len(documents),
                "artifact_files": tuple(
                    {
                        "name": artifact["name"],
                        "sha256": artifact["sha256"],
                    }
                    for artifact in config["artifact_files"]
                ),
                "options": config["options"],
            },
        )

    def product_sha256(self, request: ProductionRetrieverBuildRequest) -> str:
        return retriever_product_sha256(self.build(request))


class ProductionPredictorBuilder:
    """Build one dataset-aware predictor without exposing evaluator gold."""

    builder_kind = "predictor"

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": "agentinputlab.production_predictor_builder.v1",
            "schema_version": _SCHEMA_VERSION,
            "predictor_implementation": _PREDICTOR_IMPLEMENTATION,
            "supported_cells": tuple(sorted(_CELL_SPLITS)),
            "gold_boundary": "frozen_agent_state_only.v1",
            "qa_output": (
                "structured_json_object.v1",
                "structured_json_schema.v1",
                "answer_only.v1",
            ),
            "retrieval_output": "ranked_document_scores.v1",
        }

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            _parse_predictor_parameters(parameters)
        except (ArtifactError, TypeError, ValueError) as exc:
            return (f"invalid production predictor configuration: {exc}",)
        return ()

    def build(self, request: StaticPredictorBuildRequest, *,
              prediction_required: bool = True) -> ProductionDownstreamPredictor:
        if type(prediction_required) is not bool:
            raise ValidationError("prediction_required must be boolean")
        if not isinstance(request, StaticPredictorBuildRequest):
            raise TypeError("production predictor builder requires StaticPredictorBuildRequest")
        config = _parse_predictor_parameters(request.builder_parameters)
        _require_cell(request.benchmark_id)
        expected_split = _CELL_SPLITS[request.benchmark_id]
        if not _supports_cell_split(request.benchmark_id, request.split):
            raise ValidationError(
                f"production predictor expects split {expected_split!r} for "
                f"{request.benchmark_id!r}"
            )
        inferred_task = "retrieval" if request.benchmark_id in _RETRIEVAL_CELLS else "qa"
        if config["task"] not in {"auto", inferred_task}:
            raise ValidationError(
                f"predictor task {config['task']!r} is incompatible with {request.benchmark_id!r}"
            )

        retriever = request.runtime.retriever
        if isinstance(retriever, ProductionRetrieverRouter) and retriever.has_route(
            request.benchmark_id
        ):
            retriever.bind(request.benchmark_id)
        if inferred_task == "retrieval" and prediction_required:
            if retriever is None or not isinstance(retriever, Retriever):
                raise CapabilityError(
                    f"retrieval cell {request.benchmark_id!r} requires a locked runtime retriever"
                )
            if isinstance(retriever, ProductionRetrieverRouter):
                retriever.bind(request.benchmark_id)
        elif inferred_task == "qa" and prediction_required:
            backend = request.runtime.backend
            if not isinstance(backend, LLMBackend):
                raise CapabilityError("QA prediction requires an LLMBackend")
            required = set(config["qa"]["required_capabilities"])
            if config["qa"]["use_seed"]:
                required.add(ModelCapability.SEED)
            require_capabilities(backend, required)

        return ProductionDownstreamPredictor(
            benchmark_id=request.benchmark_id,
            split=request.split,
            seed=request.seed,
            task=inferred_task,
            runtime=request.runtime,
            config=config,
            prediction_enabled=prediction_required,
        )

    def product_sha256(self, request: StaticPredictorBuildRequest) -> str:
        return static_predictor_product_sha256(self.build(request))


class ProductionDownstreamPredictor:
    """Evaluator-blind predictor shared by the supported static cells."""

    __slots__ = ("_benchmark_id", "_config", "_prediction_enabled", "_runtime", "_seed", "_split", "_task")

    def __init__(
        self,
        *,
        benchmark_id: str,
        split: str,
        seed: int,
        task: str,
        runtime: StaticRuntimeComponents,
        config: Mapping[str, Any],
        prediction_enabled: bool = True,
    ) -> None:
        _require_cell(benchmark_id)
        if not _supports_cell_split(benchmark_id, split):
            raise ValidationError("production predictor split differs from the locked cell")
        _require_seed(seed)
        if task not in {"qa", "retrieval"}:
            raise ValidationError("production predictor task must be qa or retrieval")
        if not isinstance(runtime, StaticRuntimeComponents):
            raise TypeError("production predictor requires StaticRuntimeComponents")
        self._benchmark_id = benchmark_id
        self._split = split
        self._seed = seed
        self._task = task
        self._runtime = runtime
        self._config = _canonical_mapping(config, label="predictor configuration")
        if type(prediction_enabled) is not bool:
            raise ValidationError("prediction_enabled must be boolean")
        self._prediction_enabled = prediction_enabled

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "implementation": _PREDICTOR_IMPLEMENTATION,
            "benchmark_id": self._benchmark_id,
            "split": self._split,
            "task": self._task,
            "configuration_sha256": content_hash(self._config),
            # The concrete backend/retriever is already locked by the model-slot
            # runtime product. Keeping it out of the predictor recipe allows one
            # predictor lock to be reused fairly across multiple model slots.
            "runtime_binding_slot": "backend" if self._task == "qa" else "retriever",
            "visible_input_schema": "frozen_agent_state_without_task_or_metadata.v1",
            **({"prediction_enabled": False} if not self._prediction_enabled else {}),
        }

    def predict(self, state: FrozenAgentState) -> DownstreamPrediction:
        if not self._prediction_enabled:
            raise CapabilityError("score-only predictor cannot generate a downstream prediction")
        if not isinstance(state, FrozenAgentState):
            raise TypeError("production downstream predictor requires FrozenAgentState")
        if self._task == "retrieval":
            return self._predict_retrieval(state)
        return self._predict_qa(state)

    def _predict_retrieval(self, state: FrozenAgentState) -> DownstreamPrediction:
        options = self._config["retrieval"]
        if options["source_mode"] == "context_or_retriever":
            context_ranking = _retrieval_prediction_from_context(
                state,
                top_k=options["top_k"],
            )
            if context_ranking is not None:
                prediction, document_ids = context_ranking
                return DownstreamPrediction(
                    prediction=prediction,
                    usage=ResourceUsage(),
                    metadata={
                        "prediction_kind": "retrieval_visible_context",
                        "visible_context_sha256": content_hash(
                            tuple(
                                {
                                    "block_id": block.block_id,
                                    "text": block.text,
                                    "metadata": block.metadata,
                                }
                                for block in state.context
                            )
                        ),
                        "returned_documents": len(document_ids),
                        "top_k": options["top_k"],
                    },
                )
        retriever = self._runtime.retriever
        if not isinstance(retriever, Retriever):
            raise CapabilityError("production retrieval predictor has no runtime retriever")
        if state.budget.max_retrieval_calls < 1:
            raise CapabilityError("retrieval prediction has no remaining retrieval-call budget")
        query = (
            state.query
            if options["query_mode"] == "query_only"
            else "\n".join(part for part in (state.instruction, state.query) if part)
        )
        if not query.strip():
            raise ValidationError("production retrieval query is empty")
        results = tuple(retriever.retrieve(query, top_k=options["top_k"]))
        if any(not isinstance(item, RetrievalResult) for item in results):
            raise ValidationError("runtime retriever returned a non-RetrievalResult")
        document_ids = tuple(item.document.doc_id for item in results)
        if len(document_ids) != len(set(document_ids)):
            raise ValidationError("runtime retriever returned duplicate document IDs")
        expected_ranks = tuple(range(1, len(results) + 1))
        if tuple(item.rank for item in results) != expected_ranks:
            raise ValidationError("runtime retriever ranks must be contiguous from one")
        score_mode = options.get("score_mode", "raw")
        prediction = (
            RetrievalPrediction.from_ranked_ids(document_ids)
            if score_mode == "rank"
            else RetrievalPrediction({item.document.doc_id: item.score for item in results})
        )
        return DownstreamPrediction(
            prediction=prediction,
            usage=ResourceUsage(retrieval_calls=1),
            metadata={
                "prediction_kind": "retrieval",
                **({"score_mode": "rank"} if score_mode == "rank" else {}),
                "query_sha256": content_hash(query),
                "returned_documents": len(results),
                "top_k": options["top_k"],
            },
        )

    def _predict_qa(self, state: FrozenAgentState) -> DownstreamPrediction:
        backend = self._runtime.backend
        if not isinstance(backend, LLMBackend):
            raise CapabilityError("production QA predictor has no LLM backend")
        if state.budget.max_model_calls < 1:
            raise CapabilityError("QA prediction has no remaining model-call budget")
        options = self._config["qa"]
        request_limit = min(options["max_tokens"], state.budget.max_output_tokens)
        if request_limit < 1:
            raise CapabilityError("QA prediction has no remaining output-token budget")
        request = GenerationRequest(
            prompt=_render_visible_qa_prompt(state),
            system=options["system_prompt"],
            temperature=options["temperature"],
            max_tokens=request_limit,
            seed=self._seed if options["use_seed"] else None,
            stop=options["stop"],
            response_format=(
                _qa_response_format(
                    benchmark_id=self._benchmark_id,
                    output_format=options["output_format"],
                    structured_output_mode=options["structured_output_mode"],
                )
            ),
            metadata={
                "benchmark_id_sha256": content_hash(self._benchmark_id),
                "prompt_revision": options["prompt_revision"],
            },
        )
        response = backend.generate(request)
        if not isinstance(response, GenerationResponse):
            raise ValidationError("LLM backend returned a non-GenerationResponse")
        if not isinstance(response.text, str):
            raise ValidationError("LLM backend response text must be a string")
        usage = response.usage
        if not isinstance(usage, Usage):
            raise ValidationError("LLM backend response usage must be Usage")
        cost = usage.cost_usd
        prediction, support_audit = _parse_visible_qa_prediction(
            response.text,
            benchmark_id=self._benchmark_id,
            state=state,
            output_format=options["output_format"],
            identifier_policy=options["identifier_policy"],
            allow_json_fence=options.get("allow_json_fence", False),
        )
        return DownstreamPrediction(
            prediction=prediction,
            usage=ResourceUsage(
                model_calls=1,
                unpriced_model_calls=1 if cost is None else 0,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=0.0 if cost is None else cost,
            ),
            metadata={
                "prediction_kind": f"qa_{options['output_format']}",
                "request_sha256": request.request_hash,
                "model": response.model,
                "finish_reason": response.finish_reason,
                **support_audit,
            },
        )


def _retrieval_prediction_from_context(
    state: FrozenAgentState,
    *,
    top_k: int,
) -> tuple[RetrievalPrediction, tuple[str, ...]] | None:
    """Recover a ranked candidate list from provenance-preserving context.

    Auxiliary retrieval tasks preload the same candidate set for both arms
    when a curation method is evaluated. Acquisition methods may instead add
    their own retrieved documents. Only visible provenance is consumed.
    """

    document_ids: list[str] = []
    for block in state.context:
        raw_ids: tuple[str, ...]
        direct = block.metadata.get("document_id")
        if not isinstance(direct, str) or not direct.strip():
            direct = block.metadata.get("source_context_block_id")
        if isinstance(direct, str) and direct.strip():
            raw_ids = (direct,)
        else:
            lineage = block.metadata.get("source_block_ids")
            if isinstance(lineage, (tuple, list)) and all(
                isinstance(item, str) and item.strip() for item in lineage
            ):
                raw_ids = tuple(lineage)
            else:
                raw_ids = ()
        for document_id in raw_ids:
            if document_id not in document_ids:
                document_ids.append(document_id)
            if len(document_ids) >= top_k:
                break
        if len(document_ids) >= top_k:
            break
    if not document_ids:
        return None
    size = len(document_ids)
    return (
        RetrievalPrediction(
            {document_id: float(size - rank) for rank, document_id in enumerate(document_ids)}
        ),
        tuple(document_ids),
    )


def production_backend_builder_factory() -> ProductionBackendBuilder:
    """Zero-argument entrypoint for ``LockedBuilderSpec``."""

    return ProductionBackendBuilder()


def production_predictor_builder_factory() -> ProductionPredictorBuilder:
    """Zero-argument entrypoint for ``LockedBuilderSpec``."""

    return ProductionPredictorBuilder()


def prepared_corpus_retriever_builder_factory() -> PreparedCorpusRetrieverBuilder:
    """Zero-argument entrypoint for a nested ``LockedRetrieverSpec``."""

    return PreparedCorpusRetrieverBuilder()


def builder_fingerprint_sha256(builder: Any) -> str:
    """Return the checksum placed in a locked builder specification."""

    return content_hash(_fingerprint(builder, label="builder"))


def retriever_product_sha256(retriever: Retriever) -> str:
    """Hash an exact retriever product without performing retrieval."""

    if not isinstance(retriever, Retriever):
        raise TypeError("retriever product hashing requires a Retriever")
    return content_hash(_fingerprint(retriever, label="retriever"))


def validate_runtime_product_sha256(
    runtime: StaticRuntimeComponents,
    expected_sha256: str,
) -> str:
    """Validate and return the static driver's exact runtime product hash."""

    observed = static_runtime_product_sha256(runtime)
    return _validate_product_hash(observed, expected_sha256, label="runtime product")


def validate_predictor_product_sha256(
    predictor: ProductionDownstreamPredictor,
    expected_sha256: str,
) -> str:
    """Validate and return the static driver's exact predictor product hash."""

    observed = static_predictor_product_sha256(predictor)
    return _validate_product_hash(observed, expected_sha256, label="predictor product")


def validate_retriever_product_sha256(
    retriever: Retriever,
    expected_sha256: str,
) -> str:
    """Validate and return an exact retriever product hash."""

    observed = retriever_product_sha256(retriever)
    return _validate_product_hash(observed, expected_sha256, label="retriever product")


def _parse_backend_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _reject_embedded_secrets(parameters, path="backend parameters")
    data = _strict_mapping(
        parameters,
        path="backend parameters",
        allowed={
            "schema_version",
            "backend",
            "external_runtime",
            "required_capabilities",
            "required_env",
            "retrievers",
            "vector_runtime",
            "run_config",
            "strict_intent",
            "tools",
        },
        required={"schema_version", "backend"},
    )
    _require_schema_version(data["schema_version"], path="backend parameters.schema_version")
    backend = _parse_backend_block(data["backend"])
    external_runtime = _parse_external_runtime_block(data.get("external_runtime"))
    required_capabilities = _capabilities(
        data.get("required_capabilities", (ModelCapability.COMPLETION.value,)),
        path="backend parameters.required_capabilities",
    )
    if not required_capabilities:
        raise ValidationError("backend required_capabilities must not be empty")
    retriever_values = _mapping_or_empty(
        data.get("retrievers"),
        path="backend parameters.retrievers",
    )
    retrievers: dict[str, LockedRetrieverSpec] = {}
    for benchmark_id, value in retriever_values.items():
        _require_retriever_cell(benchmark_id)
        retrievers[benchmark_id] = LockedRetrieverSpec.from_mapping(
            value,
            path=f"backend parameters.retrievers.{benchmark_id}",
        )
    vector_runtime = None
    if data.get("vector_runtime") is not None:
        vector_runtime = LockedRetrieverSpec.from_mapping(
            data["vector_runtime"],
            path="backend parameters.vector_runtime",
        )
        _require_text(
            vector_runtime.parameters.get("benchmark_id"),
            label="vector runtime benchmark_id",
        )
    tools = _parse_tool_connector_block(data.get("tools"))
    required_env = _environment_names(
        data.get("required_env", ()),
        path="backend parameters.required_env",
    )
    if backend["kind"] == "openai_compatible" and backend["require_api_key"]:
        required_env = tuple(sorted({*required_env, backend["api_key_env"]}))
    if external_runtime is not None and external_runtime["kind"] in {
        "acon_observation_worker",
        "ask_or_assume_worker",
        "reformer_worker",
        "infocqr_worker",
    }:
        api_key_env = external_runtime.get("api_key_env")
        if api_key_env is not None:
            required_env = tuple(sorted({*required_env, api_key_env}))
    if backend["kind"] == "openai_compatible" and backend["deployment_revision_env"] is not None:
        required_env = tuple(sorted({*required_env, backend["deployment_revision_env"]}))
    strict_intent = data.get("strict_intent", False)
    if not isinstance(strict_intent, bool):
        raise ValidationError("backend parameters.strict_intent must be boolean")
    if strict_intent:
        raise ValidationError(
            "ProductionBackendBuilder cannot infer a per-example IntentContract; "
            "use a custom locked runtime builder that supplies intent_contract, or set "
            "strict_intent=false and report intent preservation as an evaluation metric"
        )
    return MappingProxyType(
        {
            "backend": backend,
            "external_runtime": external_runtime,
            "required_capabilities": required_capabilities,
            "required_env": required_env,
            "retrievers": MappingProxyType(retrievers),
            "vector_runtime": vector_runtime,
            "tools": tools,
            "run_config": _canonical_mapping(
                _mapping_or_empty(
                    data.get("run_config"),
                    path="backend parameters.run_config",
                ),
                label="backend run_config",
            ),
            "strict_intent": strict_intent,
        }
    )


def _parse_tool_connector_block(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    data = _strict_mapping(
        value,
        path="backend parameters.tools",
        allowed={"kind", "connector_id", "definitions"},
        required={"kind", "connector_id", "definitions"},
    )
    kind = _require_text(data["kind"], label="tool connector kind")
    if kind != "schema_only":
        raise ValidationError("production tool connector kind must be schema_only")
    connector_id = _require_text(data["connector_id"], label="tool connector ID")
    definitions = data["definitions"]
    if isinstance(definitions, (str, bytes)) or not isinstance(definitions, Sequence):
        raise ValidationError("tool connector definitions must be a sequence")
    normalized: list[Mapping[str, Any]] = []
    for index, definition in enumerate(definitions):
        if not isinstance(definition, Mapping):
            raise ValidationError(f"tool connector definition {index} must be an object")
        normalized.append(
            _canonical_mapping(definition, label=f"tool connector definition {index}")
        )
    # Construct once during parsing so malformed OpenAI declarations fail preflight.
    SchemaOnlyToolConnector.from_openai_definitions(
        normalized,
        connector_id=connector_id,
    )
    return MappingProxyType(
        {
            "kind": kind,
            "connector_id": connector_id,
            "definitions": tuple(normalized),
        }
    )


def _parse_external_runtime_block(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    data = _strict_mapping(
        value,
        path="backend parameters.external_runtime",
        allowed={
            "kind",
            "checkout",
            "artifact_root",
            "manifest_sha256",
            "source_root",
            "base_model_root",
            "base_model_load_set_sha256",
            "feature_model_root",
            "feature_model_load_set_sha256",
            "reader_device",
            "feature_device",
            "checkpoint_root",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            "runtime_manifest_path",
            "runtime_manifest_sha256",
            "embedder_root",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "embedder_manifest_sha256",
            "device",
            "single_candidate_axis_compat",
            "allow_short_candidates",
            "driver_path",
            "driver_sha256",
            "python_executable",
            "python_executable_sha256",
            "base_url",
            "api_key_env",
            "model_id",
            "optimizer_model_revision",
            "target_model_revision",
            "tokenizer_revision",
            "require_api_key",
            "extra_body",
            "response_model_aliases",
            "timeout_seconds",
            "max_input_tokens_per_call",
            "max_output_tokens",
            "temperature",
            "input_cost_per_million",
            "output_cost_per_million",
        },
        required={"kind"},
    )
    kind = _require_text(data["kind"], label="external runtime kind")
    if kind == "querygym_query2doc":
        _require_exact_backend_fields(
            data,
            {"kind", "checkout"},
            required={"kind", "checkout"},
            kind="QueryGym external runtime",
        )
        return MappingProxyType(
            {
                "kind": kind,
                "checkout": _absolute_path(data["checkout"], label="QueryGym checkout"),
            }
        )
    if kind == "recomp_extractive_worker":
        required = {
            "kind",
            "checkout",
            "checkpoint_root",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "device",
        }
        _require_exact_backend_fields(
            data,
            required,
            required=required,
            kind="RECOMP extractive external runtime",
        )
        device = _require_text(data["device"], label="RECOMP device")
        if re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
            raise ValidationError("RECOMP device must be cpu, cuda, or cuda:<index>")
        return MappingProxyType(
            {
                "kind": kind,
                "checkout": _absolute_path(data["checkout"], label="RECOMP checkout"),
                "checkpoint_root": _absolute_path(
                    data["checkpoint_root"], label="RECOMP checkpoint root"
                ),
                "checkpoint_manifest_path": _absolute_path(
                    data["checkpoint_manifest_path"], label="RECOMP checkpoint manifest"
                ),
                "checkpoint_manifest_sha256": _sha256_text(
                    data["checkpoint_manifest_sha256"],
                    label="RECOMP checkpoint manifest SHA-256",
                ),
                "compatibility_manifest_path": _absolute_path(
                    data["compatibility_manifest_path"],
                    label="RECOMP compatibility manifest",
                ),
                "compatibility_manifest_sha256": _sha256_text(
                    data["compatibility_manifest_sha256"],
                    label="RECOMP compatibility manifest SHA-256",
                ),
                "device": device,
            }
        )
    if kind == "acon_observation_worker":
        required = {
            "kind",
            "checkout",
            "driver_path",
            "driver_sha256",
            "python_executable",
            "python_executable_sha256",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "base_url",
            "model_id",
            "optimizer_model_revision",
            "target_model_revision",
            "tokenizer_revision",
        }
        optional = {
            "extra_body",
            "response_model_aliases",
            "require_api_key",
            "timeout_seconds",
            "max_output_tokens",
            "temperature",
        }
        _require_exact_backend_fields(
            data,
            required | optional,
            required=required,
            kind="ACON observation external runtime",
        )
        driver_path = _absolute_path(data["driver_path"], label="ACON driver")
        compatibility_manifest_path = _absolute_path(
            data["compatibility_manifest_path"], label="ACON compatibility manifest"
        )
        if compatibility_manifest_path != driver_path.with_name("compatibility_manifest.json"):
            raise ValidationError(
                "ACON compatibility manifest must be the worker driver's sibling manifest"
            )
        require_api_key = data.get("require_api_key", False)
        if not isinstance(require_api_key, bool):
            raise ValidationError("ACON require_api_key must be boolean")
        temperature = _nonnegative_number(data.get("temperature", 0.0), label="ACON temperature")
        if temperature > 2:
            raise ValidationError("ACON temperature must not exceed 2")
        return MappingProxyType(
            {
                "kind": kind,
                "checkout": _absolute_path(data["checkout"], label="ACON checkout"),
                "driver_path": driver_path,
                "driver_sha256": _sha256_text(data["driver_sha256"], label="ACON driver SHA-256"),
                "python_executable": _absolute_path(
                    data["python_executable"], label="ACON Python executable", resolve_symlinks=False
                ),
                "python_executable_sha256": _sha256_text(
                    data["python_executable_sha256"], label="ACON Python executable SHA-256"
                ),
                "compatibility_manifest_path": compatibility_manifest_path,
                "compatibility_manifest_sha256": _sha256_text(
                    data["compatibility_manifest_sha256"],
                    label="ACON compatibility manifest SHA-256",
                ),
                "base_url": _absolute_http_url(data["base_url"], label="ACON base_url"),
                "api_key_env": "AGENTINPUTLAB_ACON_API_KEY" if require_api_key else None,
                "model_id": _require_text(data["model_id"], label="ACON model_id"),
                **_optional_response_model_aliases(data),
                "optimizer_model_revision": _require_text(
                    data["optimizer_model_revision"], label="ACON optimizer_model_revision"
                ),
                "target_model_revision": _require_text(
                    data["target_model_revision"], label="ACON target_model_revision"
                ),
                "tokenizer_revision": _require_text(
                    data["tokenizer_revision"], label="ACON tokenizer_revision"
                ),
                "extra_body": _canonical_mapping(
                    _mapping_or_empty(data.get("extra_body"), path="ACON extra_body"),
                    label="ACON extra_body",
                ),
                "timeout_seconds": _positive_number(
                    data.get("timeout_seconds", 600.0), label="ACON timeout_seconds"
                ),
                "max_output_tokens": _positive_int(
                    data.get("max_output_tokens", 512), label="ACON max_output_tokens"
                ),
                "temperature": temperature,
            }
        )
    if kind == "ask_or_assume_worker":
        required = {
            "kind",
            "checkout",
            "driver_path",
            "driver_sha256",
            "python_executable",
            "python_executable_sha256",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "base_url",
            "api_key_env",
            "model_id",
        }
        optional = {
            "timeout_seconds",
            "max_output_tokens",
            "input_cost_per_million",
            "output_cost_per_million",
        }
        _require_exact_backend_fields(
            data,
            required | optional,
            required=required,
            kind="Ask-or-Assume external runtime",
        )
        return MappingProxyType(
            {
                "kind": kind,
                "checkout": _absolute_path(data["checkout"], label="Ask-or-Assume checkout"),
                "driver_path": _absolute_path(data["driver_path"], label="Ask-or-Assume driver"),
                "driver_sha256": _sha256_text(
                    data["driver_sha256"], label="Ask-or-Assume driver SHA-256"
                ),
                "python_executable": _absolute_path(
                    data["python_executable"], label="Ask-or-Assume Python executable", resolve_symlinks=False
                ),
                "python_executable_sha256": _sha256_text(
                    data["python_executable_sha256"],
                    label="Ask-or-Assume Python executable SHA-256",
                ),
                "compatibility_manifest_path": _absolute_path(
                    data["compatibility_manifest_path"],
                    label="Ask-or-Assume compatibility manifest",
                ),
                "compatibility_manifest_sha256": _sha256_text(
                    data["compatibility_manifest_sha256"],
                    label="Ask-or-Assume compatibility manifest SHA-256",
                ),
                "base_url": _absolute_http_url(data["base_url"], label="Ask-or-Assume base_url"),
                "api_key_env": _require_env_name(
                    data["api_key_env"], label="Ask-or-Assume api_key_env"
                ),
                "model_id": _require_text(data["model_id"], label="Ask-or-Assume model_id"),
                "timeout_seconds": _positive_number(
                    data.get("timeout_seconds", 600.0),
                    label="Ask-or-Assume timeout_seconds",
                ),
                "max_output_tokens": _positive_int(
                    data.get("max_output_tokens", 512),
                    label="Ask-or-Assume max_output_tokens",
                ),
                "input_cost_per_million": _nonnegative_number(
                    data.get("input_cost_per_million", 0.0),
                    label="Ask-or-Assume input_cost_per_million",
                ),
                "output_cost_per_million": _nonnegative_number(
                    data.get("output_cost_per_million", 0.0),
                    label="Ask-or-Assume output_cost_per_million",
                ),
            }
        )
    if kind in {"reformer_worker", "infocqr_worker"}:
        required = {
            "kind",
            "checkout",
            "driver_path",
            "driver_sha256",
            "python_executable",
            "python_executable_sha256",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "base_url",
            "api_key_env",
            "model_id",
        }
        optional = {
            "extra_body",
            "timeout_seconds",
            "max_input_tokens_per_call",
            "max_output_tokens",
            "temperature",
            "input_cost_per_million",
            "output_cost_per_million",
        }
        _require_exact_backend_fields(
            data, required | optional, required=required, kind="ReFormeR external runtime"
        )
        temperature = _nonnegative_number(
            data.get("temperature", 0.0), label="ReFormeR temperature"
        )
        if temperature > 2:
            raise ValidationError("ReFormeR temperature must not exceed 2")
        return MappingProxyType(
            {
                "kind": kind,
                "checkout": _absolute_path(data["checkout"], label="ReFormeR checkout"),
                "driver_path": _absolute_path(data["driver_path"], label="ReFormeR driver"),
                "driver_sha256": _sha256_text(
                    data["driver_sha256"], label="ReFormeR driver SHA-256"
                ),
                "python_executable": _absolute_path(
                    data["python_executable"], label="ReFormeR Python executable", resolve_symlinks=False
                ),
                "python_executable_sha256": _sha256_text(
                    data["python_executable_sha256"],
                    label="ReFormeR Python executable SHA-256",
                ),
                "compatibility_manifest_path": _absolute_path(
                    data["compatibility_manifest_path"],
                    label="ReFormeR compatibility manifest",
                ),
                "compatibility_manifest_sha256": _sha256_text(
                    data["compatibility_manifest_sha256"],
                    label="ReFormeR compatibility manifest SHA-256",
                ),
                "base_url": _absolute_http_url(data["base_url"], label="ReFormeR base_url"),
                "api_key_env": _require_env_name(data["api_key_env"], label="ReFormeR api_key_env"),
                "model_id": _require_text(data["model_id"], label="ReFormeR model_id"),
                "extra_body": _canonical_mapping(
                    _mapping_or_empty(data.get("extra_body"), path="ReFormeR extra_body"),
                    label="ReFormeR extra_body",
                ),
                "timeout_seconds": _positive_number(
                    data.get("timeout_seconds", 600.0), label="ReFormeR timeout_seconds"
                ),
                "max_input_tokens_per_call": _positive_int(
                    data.get("max_input_tokens_per_call", 16384),
                    label="ReFormeR max_input_tokens_per_call",
                ),
                "max_output_tokens": _positive_int(
                    data.get("max_output_tokens", 512), label="ReFormeR max_output_tokens"
                ),
                "temperature": temperature,
                "input_cost_per_million": _nonnegative_number(
                    data.get("input_cost_per_million", 0.0),
                    label="ReFormeR input_cost_per_million",
                ),
                "output_cost_per_million": _nonnegative_number(
                    data.get("output_cost_per_million", 0.0),
                    label="ReFormeR output_cost_per_million",
                ),
            }
        )
    if kind == "qrag_common_support":
        required = {
            "kind",
            "source_root",
            "checkpoint_root",
            "embedder_root",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "embedder_manifest_sha256",
        }
        _require_exact_backend_fields(
            data,
            required | {"device", "single_candidate_axis_compat"},
            required=required,
            kind="Q-RAG external runtime",
        )
        axis_compat = data.get("single_candidate_axis_compat", False)
        if not isinstance(axis_compat, bool):
            raise ValidationError("Q-RAG single_candidate_axis_compat must be boolean")
        return MappingProxyType(
            {
                "kind": kind,
                "source_root": _absolute_path(data["source_root"], label="Q-RAG source root"),
                "checkpoint_root": _absolute_path(
                    data["checkpoint_root"], label="Q-RAG checkpoint root"
                ),
                "embedder_root": _absolute_path(data["embedder_root"], label="Q-RAG embedder root"),
                "compatibility_manifest_path": _absolute_path(
                    data["compatibility_manifest_path"],
                    label="Q-RAG compatibility manifest",
                ),
                "compatibility_manifest_sha256": _sha256_text(
                    data["compatibility_manifest_sha256"],
                    label="Q-RAG compatibility manifest SHA-256",
                ),
                "embedder_manifest_sha256": _sha256_text(
                    data["embedder_manifest_sha256"],
                    label="Q-RAG embedder manifest SHA-256",
                ),
                "device": _require_text(data.get("device", "cuda"), label="Q-RAG device"),
                "single_candidate_axis_compat": axis_compat,
            }
        )
    if kind == "smartsearch_worker":
        required = {
            "kind",
            "checkout",
            "driver_path",
            "driver_sha256",
            "python_executable",
            "python_executable_sha256",
            "compatibility_manifest_path",
            "compatibility_manifest_sha256",
            "runtime_manifest_path",
            "runtime_manifest_sha256",
        }
        _require_exact_backend_fields(
            data,
            required | {"timeout_seconds"},
            required=required,
            kind="SmartSearch external runtime",
        )
        driver_path = _absolute_path(data["driver_path"], label="SmartSearch driver")
        compatibility_manifest_path = _absolute_path(
            data["compatibility_manifest_path"], label="SmartSearch compatibility manifest"
        )
        if compatibility_manifest_path != driver_path.with_name("compatibility_manifest.json"):
            raise ValidationError(
                "SmartSearch compatibility manifest must be the worker driver's sibling manifest"
            )
        return MappingProxyType(
            {
                "kind": kind,
                "checkout": _absolute_path(data["checkout"], label="SmartSearch checkout"),
                "driver_path": driver_path,
                "driver_sha256": _sha256_text(
                    data["driver_sha256"], label="SmartSearch driver SHA-256"
                ),
                "python_executable": _absolute_path(
                    data["python_executable"], label="SmartSearch Python executable"
                ),
                "python_executable_sha256": _sha256_text(
                    data["python_executable_sha256"],
                    label="SmartSearch Python executable SHA-256",
                ),
                "compatibility_manifest_path": compatibility_manifest_path,
                "compatibility_manifest_sha256": _sha256_text(
                    data["compatibility_manifest_sha256"],
                    label="SmartSearch compatibility manifest SHA-256",
                ),
                "runtime_manifest_path": _absolute_path(
                    data["runtime_manifest_path"], label="SmartSearch runtime manifest"
                ),
                "runtime_manifest_sha256": _sha256_text(
                    data["runtime_manifest_sha256"],
                    label="SmartSearch runtime manifest SHA-256",
                ),
                "timeout_seconds": _positive_number(
                    data.get("timeout_seconds", 1800.0),
                    label="SmartSearch timeout_seconds",
                ),
            }
        )
    if kind == "r2ag_common_support":
        required = {
            "kind",
            "artifact_root",
            "manifest_sha256",
            "source_root",
            "base_model_root",
            "base_model_load_set_sha256",
            "feature_model_root",
            "feature_model_load_set_sha256",
        }
        _require_exact_backend_fields(
            data,
            required | {"reader_device", "feature_device", "allow_short_candidates"},
            required=required,
            kind="R²AG external runtime",
        )
        allow_short = data.get("allow_short_candidates", False)
        if not isinstance(allow_short, bool):
            raise ValidationError("R²AG allow_short_candidates must be boolean")
        return MappingProxyType(
            {
                "kind": kind,
                "artifact_root": _absolute_path(data["artifact_root"], label="R²AG artifact root"),
                "manifest_sha256": _sha256_text(
                    data["manifest_sha256"], label="R²AG manifest_sha256"
                ),
                "source_root": _absolute_path(data["source_root"], label="R²AG source root"),
                "base_model_root": _absolute_path(
                    data["base_model_root"], label="R²AG base model root"
                ),
                "base_model_load_set_sha256": _sha256_text(
                    data["base_model_load_set_sha256"],
                    label="R²AG base model load-set SHA-256",
                ),
                "feature_model_root": _absolute_path(
                    data["feature_model_root"], label="R²AG feature model root"
                ),
                "feature_model_load_set_sha256": _sha256_text(
                    data["feature_model_load_set_sha256"],
                    label="R²AG feature model load-set SHA-256",
                ),
                "reader_device": _require_text(
                    data.get("reader_device", "cuda"), label="R²AG reader device"
                ),
                "feature_device": _require_text(
                    data.get("feature_device", "cuda"), label="R²AG feature device"
                ),
                "allow_short_candidates": allow_short,
            }
        )
    raise ValidationError(f"unsupported production external runtime kind: {kind!r}")


def _response_model_aliases(value: Any) -> tuple[str, ...]:
    if (not isinstance(value, (tuple, list)) or
        any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in value) or
        len(set(value)) != len(value)):
        raise ValidationError("response_model_aliases must be a unique string array")
    return tuple(value)


def _optional_response_model_aliases(value: Mapping[str, Any]) -> dict[str, Any]:
    aliases = _response_model_aliases(value.get("response_model_aliases", ()))
    return {"response_model_aliases": aliases} if aliases else {}


def _parse_backend_block(value: Any) -> Mapping[str, Any]:
    base = _strict_mapping(
        value,
        path="backend parameters.backend",
        allowed={
            "kind",
            "model_id",
            "capabilities",
            "fixture_path",
            "fixture_sha256",
            "base_url",
            "deployment_identity",
            "deployment_revision_env",
            "api_key_env",
            "require_api_key",
            "timeout_seconds",
            "extra_headers",
            "extra_body",
            "response_model_aliases",
            "structured_output_transport",
            "model_revision",
            "tokenizer_revision",
            "model_path",
            "generation_kwargs",
            "local_files_only",
            "artifact_files",
        },
        required={"kind", "model_id"},
    )
    kind = _require_text(base["kind"], label="backend kind")
    if kind not in {"replay", "openai_compatible", "transformers"}:
        raise ValidationError(f"unsupported production backend kind: {kind!r}")
    model_id = _require_text(base["model_id"], label="backend model_id")
    defaults = {
        "replay": (
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.SEED,
        ),
        "openai_compatible": (
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.JSON_OBJECT,
            ModelCapability.TOOL_CALLING,
            ModelCapability.SEED,
        ),
        "transformers": (
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
        ),
    }
    capabilities = _capabilities(
        base.get("capabilities", tuple(item.value for item in defaults[kind])),
        path="backend capabilities",
    )
    common = {
        "kind": kind,
        "model_id": model_id,
        "capabilities": capabilities,
    }
    if kind == "replay":
        _require_exact_backend_fields(
            base,
            {"kind", "model_id", "capabilities", "fixture_path", "fixture_sha256"},
            required={"fixture_path", "fixture_sha256"},
            kind=kind,
        )
        return MappingProxyType(
            {
                **common,
                "fixture_path": _absolute_path(
                    base["fixture_path"],
                    label="replay fixture_path",
                ),
                "fixture_sha256": _sha256_text(
                    base["fixture_sha256"],
                    label="replay fixture_sha256",
                ),
            }
        )
    if kind == "openai_compatible":
        _require_exact_backend_fields(
            base,
            {
                "kind",
                "model_id",
                "capabilities",
                "base_url",
                "deployment_identity",
                "deployment_revision_env",
                "api_key_env",
                "require_api_key",
                "timeout_seconds",
                "extra_headers",
                "extra_body",
                "response_model_aliases",
                "structured_output_transport",
            },
            required={"base_url", "deployment_identity"},
            kind=kind,
        )
        api_key_env = base.get("api_key_env", "OPENAI_API_KEY")
        structured_transport = base.get("structured_output_transport", "chat_completions")
        if structured_transport not in ("chat_completions", "responses_strict"):
            raise ValidationError("structured_output_transport must be chat_completions or responses_strict")
        if structured_transport == "responses_strict" and ModelCapability.JSON_SCHEMA not in capabilities:
            raise ValidationError("responses_strict requires the json_schema capability")
        _require_env_name(api_key_env, label="OpenAI-compatible api_key_env")
        require_api_key = base.get("require_api_key", False)
        if not isinstance(require_api_key, bool):
            raise ValidationError("OpenAI-compatible require_api_key must be boolean")
        timeout = _positive_number(
            base.get("timeout_seconds", 120.0),
            label="OpenAI-compatible timeout_seconds",
        )
        headers = _mapping_or_empty(
            base.get("extra_headers"),
            path="OpenAI-compatible extra_headers",
        )
        for name, header_value in headers.items():
            _require_text(name, label="OpenAI-compatible header name")
            if _SENSITIVE_HEADER.fullmatch(name):
                raise ValidationError(f"sensitive header {name!r} is forbidden; use api_key_env")
            _require_text(header_value, label=f"OpenAI-compatible header {name}")
        identity = _canonical_mapping(
            _mapping_or_empty(
                base["deployment_identity"],
                path="OpenAI-compatible deployment_identity",
            ),
            label="OpenAI-compatible deployment_identity",
        )
        if not identity.get("revision"):
            raise ValidationError(
                "OpenAI-compatible deployment_identity requires a non-empty revision"
            )
        deployment_revision_env = base.get("deployment_revision_env")
        if deployment_revision_env is not None:
            _require_env_name(
                deployment_revision_env,
                label="OpenAI-compatible deployment_revision_env",
            )
        return MappingProxyType(
            {
                **common,
                "base_url": _absolute_http_url(
                    base["base_url"],
                    label="OpenAI-compatible base_url",
                ),
                "deployment_identity": identity,
                **_optional_response_model_aliases(base),
                **({"structured_output_transport": structured_transport}
                   if structured_transport != "chat_completions" else {}),
                "deployment_revision_env": deployment_revision_env,
                "api_key_env": api_key_env,
                "require_api_key": require_api_key,
                "timeout_seconds": timeout,
                "extra_headers": _canonical_mapping(headers, label="extra_headers"),
                "extra_body": _canonical_mapping(
                    _mapping_or_empty(
                        base.get("extra_body"),
                        path="OpenAI-compatible extra_body",
                    ),
                    label="extra_body",
                ),
            }
        )
    _require_exact_backend_fields(
        base,
        {
            "kind",
            "model_id",
            "capabilities",
            "model_revision",
            "tokenizer_revision",
            "model_path",
            "generation_kwargs",
            "local_files_only",
            "artifact_files",
        },
        required={
            "model_path",
            "model_revision",
            "tokenizer_revision",
            "artifact_files",
        },
        kind=kind,
    )
    local_files_only = base.get("local_files_only", True)
    if local_files_only is not True:
        raise ValidationError("production Transformers backend requires local_files_only=true")
    artifacts = _artifact_file_locks(
        base["artifact_files"],
        path="Transformers artifact_files",
    )
    if not artifacts:
        raise ValidationError(
            "production Transformers backend requires checksum-locked artifact_files"
        )
    model_path = _absolute_path(
        base["model_path"],
        label="Transformers model_path",
    )
    _validate_transformers_artifact_layout(model_path, artifacts)
    return MappingProxyType(
        {
            **common,
            "model_path": model_path,
            "model_revision": _require_text(
                base["model_revision"],
                label="Transformers model_revision",
            ),
            "tokenizer_revision": _require_text(
                base["tokenizer_revision"],
                label="Transformers tokenizer_revision",
            ),
            "generation_kwargs": _canonical_mapping(
                _mapping_or_empty(
                    base.get("generation_kwargs"),
                    path="Transformers generation_kwargs",
                ),
                label="Transformers generation_kwargs",
            ),
            "local_files_only": True,
            "artifact_files": artifacts,
        }
    )


def _require_exact_backend_fields(
    data: Mapping[str, Any],
    allowed: set[str],
    *,
    required: set[str],
    kind: str,
) -> None:
    unknown = set(data) - allowed
    missing = required - set(data)
    if unknown:
        raise ValidationError(
            f"{kind} backend has incompatible fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValidationError(f"{kind} backend is missing fields: {', '.join(sorted(missing))}")


def _backend_preflight_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers = [
        f"required environment variable {name!r} is not set"
        for name in config["required_env"]
        if not os.environ.get(name)
    ]
    backend_config = config["backend"]
    if (
        backend_config["kind"] == "openai_compatible"
        and backend_config["deployment_revision_env"] is not None
    ):
        revision_environment = backend_config["deployment_revision_env"]
        observed_revision = os.environ.get(revision_environment)
        if (
            observed_revision
            and observed_revision != backend_config["deployment_identity"]["revision"]
        ):
            blockers.append(
                "OpenAI-compatible deployment revision differs from its environment lock"
            )
    try:
        backend = _build_llm_backend(backend_config)
        require_capabilities(backend, config["required_capabilities"])
        _build_external_runtime(config["external_runtime"], backend)
    except (ArtifactError, CapabilityError, ExternalDependencyError, ValidationError) as exc:
        blockers.append(str(exc))
    if backend_config["kind"] == "transformers":
        if importlib.util.find_spec("transformers") is None:
            blockers.append("transformers is unavailable; install agentinputlab[local]")
        try:
            _verify_transformers_snapshot(
                backend_config["model_path"],
                backend_config["artifact_files"],
            )
        except ArtifactError as exc:
            blockers.append(str(exc))
    for benchmark_id, spec in config["retrievers"].items():
        try:
            builder = _resolve_retriever_builder(spec)
            messages = builder.preflight(spec.parameters)
            if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
                raise ValidationError("retriever builder preflight must return a sequence")
            for message in messages:
                if not isinstance(message, str) or not message.strip():
                    raise ValidationError(
                        "retriever builder preflight blockers must be non-empty strings"
                    )
                blockers.append(f"retriever {benchmark_id}: {message}")
        except (ArtifactError, TypeError, ValueError) as exc:
            blockers.append(f"retriever {benchmark_id}: {exc}")
    vector_spec = config["vector_runtime"]
    if vector_spec is not None:
        try:
            builder = _resolve_retriever_builder(vector_spec)
            messages = builder.preflight(vector_spec.parameters)
            if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
                raise ValidationError("vector runtime preflight must return a sequence")
            for message in messages:
                if not isinstance(message, str) or not message.strip():
                    raise ValidationError(
                        "vector runtime preflight blockers must be non-empty strings"
                    )
                blockers.append(f"vector runtime: {message}")
        except (ArtifactError, TypeError, ValueError) as exc:
            blockers.append(f"vector runtime: {exc}")
    return blockers


def _build_llm_backend(config: Mapping[str, Any]) -> LLMBackend:
    kind = config["kind"]
    if kind == "replay":
        fixture = _load_replay_fixture(
            config["fixture_path"],
            config["fixture_sha256"],
        )
        inner: LLMBackend = ReplayBackend(
            fixture["responses"],
            model_id=config["model_id"],
            capabilities=frozenset(config["capabilities"]),
            strict_request_hashes=fixture["strict_request_hashes"],
        )
        binding = {
            "kind": kind,
            "fixture_sha256": config["fixture_sha256"],
            "fixture_schema_version": _SCHEMA_VERSION,
        }
    elif kind == "openai_compatible":
        inner = OpenAICompatibleBackend(
            model_id=config["model_id"],
            base_url=config["base_url"],
            deployment_identity=config["deployment_identity"],
            api_key=None,
            api_key_env=config["api_key_env"],
            timeout_seconds=config["timeout_seconds"],
            extra_headers=config["extra_headers"],
            extra_body=config["extra_body"],
            response_model_aliases=config.get("response_model_aliases",()),
            structured_output_transport=config.get("structured_output_transport", "chat_completions"),
            capabilities=frozenset(config["capabilities"]),
        )
        binding = {
            "kind": kind,
            "deployment_revision_source": (
                {
                    "kind": "environment",
                    "name": config["deployment_revision_env"],
                }
                if config["deployment_revision_env"] is not None
                else {"kind": "operator_declaration"}
            ),
            "api_key_source": {
                "kind": "environment",
                "name": config["api_key_env"],
                "required": config["require_api_key"],
            },
        }
    else:
        inner = _LocalOnlyTransformersBackend(
            model_id=config["model_id"],
            model_path=config["model_path"],
            model_revision=config["model_revision"],
            tokenizer_revision=config["tokenizer_revision"],
            artifact_files=config["artifact_files"],
            generation_kwargs=config["generation_kwargs"],
            capabilities=frozenset(config["capabilities"]),
            local_files_only=config["local_files_only"],
        )
        binding = {
            "kind": kind,
            "snapshot_sha256": _transformers_snapshot_sha256(config["artifact_files"]),
            "artifact_files": tuple(
                {"name": artifact["name"], "sha256": artifact["sha256"]}
                for artifact in config["artifact_files"]
            ),
            "local_files_only": True,
        }
    return _BoundLLMBackend(inner, binding)


@lru_cache(maxsize=4)
def _cached_recomp_extractive_model(
    checkpoint_root: str,
    device: str,
) -> tuple[Any, Any]:
    """Load the frozen RECOMP artifact with the installed Transformers API.

    RECOMP's scorer is invoked unchanged by its worker.  Only the historical
    ``from_pretrained`` compatibility surface is adapted; this identity and
    the installed package versions remain visible in the worker fingerprint.
    """

    try:
        from transformers import AutoModel, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise ExternalDependencyError(
            "RECOMP current-runtime loader requires transformers"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        checkpoint_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _load_recomp_extractive_current_runtime(
    mode: str,
    checkpoint_root: Path,
    device: str,
) -> tuple[Any, Any]:
    if mode != "extractive":
        raise ValidationError("RECOMP production runtime only accepts the extractive release")
    return _cached_recomp_extractive_model(str(checkpoint_root), device)


def _build_external_runtime(
    config: Mapping[str, Any] | None,
    backend: LLMBackend,
) -> Any:
    if config is None:
        return None
    if config["kind"] == "querygym_query2doc":
        from agentinputlab.algorithms.reformulate.query2doc import PinnedQueryGymRuntime

        runtime = PinnedQueryGymRuntime(config["checkout"], backend)
        identity = _fingerprint(runtime, label="QueryGym external runtime")
        if identity.get("implementation") != _QUERYGYM_RUNTIME_IMPLEMENTATION:
            raise ValidationError("QueryGym runtime implementation identity is invalid")
        return runtime
    if config["kind"] == "recomp_extractive_worker":
        from workers.recomp.driver import RECOMPDriver

        _verify_locked_file(
            config["checkpoint_manifest_path"],
            config["checkpoint_manifest_sha256"],
            label="RECOMP checkpoint manifest",
        )
        _verify_locked_file(
            config["compatibility_manifest_path"],
            config["compatibility_manifest_sha256"],
            label="RECOMP compatibility manifest",
        )
        return RECOMPDriver(
            checkout_path=config["checkout"],
            checkpoint_path=config["checkpoint_root"],
            checkpoint_manifest_path=config["checkpoint_manifest_path"],
            device=config["device"],
            manifest_path=config["compatibility_manifest_path"],
            model_loader=_load_recomp_extractive_current_runtime,
            checkpoint_license_acknowledged=True,
        )
    if config["kind"] == "acon_observation_worker":
        from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

        driver_sha256 = _verify_locked_file(
            config["driver_path"], config["driver_sha256"], label="ACON driver"
        )
        executable_sha256 = _verify_locked_file(
            config["python_executable"],
            config["python_executable_sha256"],
            label="ACON Python executable",
        )
        manifest_sha256 = _verify_locked_file(
            config["compatibility_manifest_path"],
            config["compatibility_manifest_sha256"],
            label="ACON compatibility manifest",
        )
        model_fingerprint = dict(_fingerprint(backend, label="ACON target backend"))
        environment = {
            "AGENTINPUTLAB_ACON_CHECKOUT": str(config["checkout"]),
            "AGENTINPUTLAB_ACON_OPENAI_BASE_URL": config["base_url"],
            "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL": config["model_id"],
            "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL_REVISION": (config["optimizer_model_revision"]),
            "AGENTINPUTLAB_ACON_TARGET_MODEL_REVISION": config["target_model_revision"],
            "AGENTINPUTLAB_ACON_TOKENIZER_REVISION": config["tokenizer_revision"],
            "AGENTINPUTLAB_ACON_TEMPERATURE": str(config["temperature"]),
            "AGENTINPUTLAB_ACON_TIMEOUT_SECONDS": str(config["timeout_seconds"]),
            "AGENTINPUTLAB_ACON_MAX_TOKENS": str(config["max_output_tokens"]),
            "AGENTINPUTLAB_ACON_EXTRA_BODY_JSON": json.dumps(
                to_jsonable(config["extra_body"]),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            **({"AGENTINPUTLAB_ACON_RESPONSE_MODEL_ALIASES_JSON": json.dumps(list(config["response_model_aliases"]))}
               if config.get("response_model_aliases") else {}),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        pass_environment = (config["api_key_env"],) if config["api_key_env"] is not None else ()
        return JsonLineSubprocessWorker(
            command=(str(config["python_executable"]), str(config["driver_path"])),
            worker_id="acon-common-support-observation",
            identity={
                "implementation": "agentinputlab.acon.process_worker.v1",
                "release_id": "curate.acon.common_support_observation",
                "component_release_id": "curate.acon.observation_runtime",
                "upstream_commit": "d63f9ae18959dc7215ff62899c94c5e8c56847ae",
                "driver_sha256": driver_sha256,
                "python_executable_sha256": executable_sha256,
                "compatibility_manifest_sha256": manifest_sha256,
                "optimizer_endpoint": config["base_url"],
                "optimizer_model_revision": config["optimizer_model_revision"],
                "target_model_revision": config["target_model_revision"],
                "tokenizer_revision": config["tokenizer_revision"],
                "optimizer_extra_body_hash": content_hash(config["extra_body"]),
                **({"optimizer_response_model_aliases": tuple(config["response_model_aliases"])}
                   if config.get("response_model_aliases") else {}),
                "target_backend": model_fingerprint,
            },
            timeout_seconds=config["timeout_seconds"],
            pass_environment=pass_environment,
            working_directory=str(config["driver_path"].parents[2]),
            environment=environment,
        )
    if config["kind"] == "ask_or_assume_worker":
        from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

        driver_sha256 = _verify_locked_file(
            config["driver_path"],
            config["driver_sha256"],
            label="Ask-or-Assume driver",
        )
        executable_sha256 = _verify_locked_file(
            config["python_executable"],
            config["python_executable_sha256"],
            label="Ask-or-Assume Python executable",
        )
        manifest_sha256 = _verify_locked_file(
            config["compatibility_manifest_path"],
            config["compatibility_manifest_sha256"],
            label="Ask-or-Assume compatibility manifest",
        )
        model_fingerprint = dict(_fingerprint(backend, label="Ask-or-Assume backend"))
        role = {
            "model": config["model_id"],
            "fingerprint": model_fingerprint,
            "max_output_tokens": config["max_output_tokens"],
            "input_cost_per_million": config["input_cost_per_million"],
            "output_cost_per_million": config["output_cost_per_million"],
        }
        backend_config = {
            "base_url": config["base_url"],
            "api_key_env": config["api_key_env"],
            "timeout_seconds": config["timeout_seconds"],
            "roles": {"intent": role, "coding": role},
        }
        return JsonLineSubprocessWorker(
            command=(
                str(config["python_executable"]),
                str(config["driver_path"]),
                "--checkout",
                str(config["checkout"]),
                "--backend-config-json",
                json.dumps(
                    to_jsonable(backend_config),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            ),
            worker_id="ask-or-assume-intent-monitor",
            identity={
                "implementation": "agentinputlab.ask_or_assume.process_worker.v1",
                "release_id": "clarify.ask_or_assume.intent_monitor_component",
                "upstream_commit": "728464a755f15fd448aff61a5cb32dce97fd9c1d",
                "driver_sha256": driver_sha256,
                "python_executable_sha256": executable_sha256,
                "compatibility_manifest_sha256": manifest_sha256,
                "backend": model_fingerprint,
            },
            timeout_seconds=config["timeout_seconds"],
            pass_environment=(config["api_key_env"],),
            working_directory=str(config["driver_path"].parents[2]),
        )
    if config["kind"] in {"reformer_worker", "infocqr_worker"}:
        from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

        is_infocqr = config["kind"] == "infocqr_worker"
        runtime_label = "InfoCQR" if is_infocqr else "ReFormeR"
        driver_sha256 = _verify_locked_file(
            config["driver_path"], config["driver_sha256"], label=f"{runtime_label} driver"
        )
        executable_sha256 = _verify_locked_file(
            config["python_executable"],
            config["python_executable_sha256"],
            label="ReFormeR Python executable",
        )
        manifest_sha256 = _verify_locked_file(
            config["compatibility_manifest_path"],
            config["compatibility_manifest_sha256"],
            label=f"{runtime_label} compatibility manifest",
        )
        model_fingerprint = dict(_fingerprint(backend, label="ReFormeR backend"))
        backend_config = {
            "base_url": config["base_url"],
            "api_key_env": config["api_key_env"],
            "model": config["model_id"],
            "fingerprint": model_fingerprint,
            "extra_body": config["extra_body"],
            "timeout_seconds": config["timeout_seconds"],
            "max_input_tokens_per_call": config["max_input_tokens_per_call"],
            "max_output_tokens": config["max_output_tokens"],
            "temperature": config["temperature"],
            "input_cost_per_million": config["input_cost_per_million"],
            "output_cost_per_million": config["output_cost_per_million"],
        }
        return JsonLineSubprocessWorker(
            command=(
                str(config["python_executable"]),
                str(config["driver_path"]),
                "--checkout",
                str(config["checkout"]),
                "--manifest" if is_infocqr else "--compatibility-manifest",
                str(config["compatibility_manifest_path"]),
                "--backend-config-json",
                json.dumps(
                    to_jsonable(backend_config),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "--acknowledge-noassertion-license",
            ),
            worker_id=(
                "infocqr-common-support-rewriter" if is_infocqr else "reformer-pattern-reformulator"
            ),
            identity={
                "implementation": (
                    "agentinputlab.infocqr.process_worker.v1"
                    if is_infocqr
                    else "agentinputlab.reformer.process_worker.v1"
                ),
                "release_id": (
                    "rewrite.infocqr.common_support_rw_zsl"
                    if is_infocqr
                    else "rewrite.reformer.repo_native_r_only"
                ),
                "upstream_commit": (
                    "2953c461aa0c6817b1b7656dbd5168050227b31d"
                    if is_infocqr
                    else "72e52450a922bc0051b5b39c3a6307186a555137"
                ),
                "driver_sha256": driver_sha256,
                "python_executable_sha256": executable_sha256,
                "compatibility_manifest_sha256": manifest_sha256,
                "backend": model_fingerprint,
            },
            timeout_seconds=config["timeout_seconds"],
            pass_environment=(config["api_key_env"],),
            working_directory=str(config["driver_path"].parents[2]),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
        )
    if config["kind"] == "qrag_common_support":
        from agentinputlab.algorithms.acquire.qrag import LocalQRAGFrozenPolicyRuntime

        return LocalQRAGFrozenPolicyRuntime(
            source_root=config["source_root"],
            checkpoint_root=config["checkpoint_root"],
            embedder_root=config["embedder_root"],
            compatibility_manifest_path=config["compatibility_manifest_path"],
            compatibility_manifest_sha256=config["compatibility_manifest_sha256"],
            embedder_manifest_sha256=config["embedder_manifest_sha256"],
            device=config["device"],
            single_candidate_axis_compat=config["single_candidate_axis_compat"],
        )
    if config["kind"] == "smartsearch_worker":
        from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

        driver_sha256 = _verify_locked_file(
            config["driver_path"], config["driver_sha256"], label="SmartSearch driver"
        )
        executable_sha256 = _verify_locked_file(
            config["python_executable"],
            config["python_executable_sha256"],
            label="SmartSearch Python executable",
        )
        manifest_sha256 = _verify_locked_file(
            config["compatibility_manifest_path"],
            config["compatibility_manifest_sha256"],
            label="SmartSearch compatibility manifest",
        )
        runtime_manifest_sha256 = _verify_locked_file(
            config["runtime_manifest_path"],
            config["runtime_manifest_sha256"],
            label="SmartSearch runtime manifest",
        )
        model_fingerprint = dict(_fingerprint(backend, label="SmartSearch target backend"))
        return JsonLineSubprocessWorker(
            command=(str(config["python_executable"]), str(config["driver_path"])),
            worker_id="smartsearch-common-support-queries",
            identity={
                "implementation": "agentinputlab.smartsearch.process_worker.v2",
                "release_id": "rewrite.smartsearch.common_support_queries",
                "component_release_id": "recipe.smartsearch.frozen_3b_inference",
                "upstream_commit": "5e9e7ec0d6af34f45fad145d4bff442529e7e6ff",
                "driver_sha256": driver_sha256,
                "python_executable_sha256": executable_sha256,
                "compatibility_manifest_sha256": manifest_sha256,
                "runtime_manifest_sha256": runtime_manifest_sha256,
                "target_backend": model_fingerprint,
            },
            timeout_seconds=config["timeout_seconds"],
            working_directory=str(config["driver_path"].parents[2]),
            environment={
                "AGENTINPUTLAB_SMARTSEARCH_CHECKOUT": str(config["checkout"]),
                "AGENTINPUTLAB_SMARTSEARCH_RUNTIME_MANIFEST": str(config["runtime_manifest_path"]),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    if config["kind"] == "r2ag_common_support":
        import os

        from agentinputlab.algorithms.acquire.r2ag import (
            R2AG_RUNTIME_SOCKET_ENV,
            R2AG_RUNTIME_TIMEOUT_ENV,
            LocalR2AGCommonSupportReader,
            LocalR2AGCommonSupportRuntime,
            LocalSentenceTransformerR2AGFeatureEncoder,
            UnixSocketR2AGCommonSupportRuntime,
        )

        socket_path = os.environ.get(R2AG_RUNTIME_SOCKET_ENV)
        if socket_path:
            runtime = UnixSocketR2AGCommonSupportRuntime(
                socket_path=Path(socket_path),
                timeout_seconds=float(os.environ.get(R2AG_RUNTIME_TIMEOUT_ENV, "3600")),
            )
            if config.get("allow_short_candidates", False):
                from agentinputlab.algorithms.acquire.r2ag.candidate_count import (
                    R2AG_SHORT_CANDIDATE_COMPAT,
                )

                if runtime.fingerprint().get("reader", {}).get("candidate_count_adapter") != R2AG_SHORT_CANDIDATE_COMPAT:
                    raise ValidationError("R²AG socket reader must explicitly enable short-candidate compatibility")
            return runtime

        short_options = {"allow_short_candidates": True} if config.get("allow_short_candidates", False) else {}
        return LocalR2AGCommonSupportRuntime(
            reader=LocalR2AGCommonSupportReader(
                artifact_root=config["artifact_root"],
                manifest_sha256=config["manifest_sha256"],
                source_root=config["source_root"],
                base_model_root=config["base_model_root"],
                base_model_load_set_sha256=config["base_model_load_set_sha256"],
                device=config["reader_device"],
                **short_options,
            ),
            feature_encoder=LocalSentenceTransformerR2AGFeatureEncoder(
                model_root=config["feature_model_root"],
                model_load_set_sha256=config["feature_model_load_set_sha256"],
                device=config["feature_device"],
            ),
            **short_options,
        )
    raise ValidationError(f"unsupported external runtime kind: {config['kind']!r}")


def _parse_predictor_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _reject_embedded_secrets(parameters, path="predictor parameters")
    data = _strict_mapping(
        parameters,
        path="predictor parameters",
        allowed={"schema_version", "task", "qa", "retrieval"},
        required={"schema_version"},
    )
    _require_schema_version(data["schema_version"], path="predictor parameters.schema_version")
    task = data.get("task", "auto")
    if task not in {"auto", "qa", "retrieval"}:
        raise ValidationError("predictor task must be auto, qa, or retrieval")
    qa = _strict_mapping(
        data.get("qa", {}),
        path="predictor parameters.qa",
        allowed={
            "system_prompt",
            "prompt_revision",
            "output_format",
            "structured_output_mode",
            "identifier_policy",
            "allow_json_fence",
            "max_tokens",
            "temperature",
            "stop",
            "use_seed",
            "required_capabilities",
        },
        required=set(),
    )
    output_format = qa.get("output_format", "structured_json")
    if output_format not in {"structured_json", "answer_only"}:
        raise ValidationError("QA output_format must be structured_json or answer_only")
    allow_json_fence = qa.get("allow_json_fence", False)
    if type(allow_json_fence) is not bool:
        raise ValidationError("QA allow_json_fence must be boolean")
    if allow_json_fence and output_format != "structured_json":
        raise ValidationError("QA allow_json_fence requires output_format=structured_json")
    structured_output_mode = qa.get("structured_output_mode", "json_object")
    if structured_output_mode not in {"json_object", "json_schema"}:
        raise ValidationError("QA structured_output_mode must be json_object or json_schema")
    identifier_policy = qa.get("identifier_policy", "native_sets.v1")
    if not isinstance(identifier_policy, str) or identifier_policy not in {
        "native_sets.v1", "strict_strings.v1"
    }:
        raise ValidationError("QA identifier_policy must be native_sets.v1 or strict_strings.v1")
    default_system_prompt = (
        "Answer using only the visible request and context. Return exactly one JSON "
        "object with keys answer, support_ids, evidence_ids, and evidence_texts. "
        "Each non-answer value must be an array of unique strings. Use the visible "
        "support identifiers when present; otherwise use visible block IDs. "
        "For 2Wiki evidence_ids, use canonical JSON strings of inferred "
        "[subject, relation, object] triples. Do not add markdown."
        if output_format == "structured_json"
        else (
            "Answer the question using only the visible request and context. "
            "Return only the answer text."
        )
    )
    system_prompt = qa.get("system_prompt", default_system_prompt)
    if not isinstance(system_prompt, str):
        raise ValidationError("QA system_prompt must be a string")
    prompt_revision = _require_text(
        qa.get("prompt_revision", "visible_qa_text.v1"),
        label="QA prompt_revision",
    )
    max_tokens = _positive_int(qa.get("max_tokens", 512), label="QA max_tokens")
    temperature = _nonnegative_number(qa.get("temperature", 0.0), label="QA temperature")
    stop = _text_sequence(qa.get("stop", ()), path="QA stop", allow_empty=True)
    use_seed = qa.get("use_seed", False)
    if not isinstance(use_seed, bool):
        raise ValidationError("QA use_seed must be boolean")
    required = _capabilities(
        qa.get("required_capabilities", (ModelCapability.COMPLETION.value,)),
        path="QA required_capabilities",
    )
    if not required:
        raise ValidationError("QA required_capabilities must not be empty")

    retrieval = _strict_mapping(
        data.get("retrieval", {}),
        path="predictor parameters.retrieval",
        allowed={"top_k", "query_mode", "source_mode", "score_mode"},
        required=set(),
    )
    top_k = _positive_int(retrieval.get("top_k", 100), label="retrieval top_k")
    if top_k > 10_000:
        raise ValidationError("retrieval top_k must not exceed 10000")
    query_mode = retrieval.get("query_mode", "query_only")
    if query_mode not in {"query_only", "instruction_and_query"}:
        raise ValidationError("retrieval query_mode must be query_only or instruction_and_query")
    source_mode = retrieval.get("source_mode", "retriever")
    if source_mode not in {"retriever", "context_or_retriever"}:
        raise ValidationError("retrieval source_mode must be retriever or context_or_retriever")
    score_mode = retrieval.get("score_mode", "raw")
    if not isinstance(score_mode, str) or score_mode not in {"raw", "rank"}:
        raise ValidationError("retrieval score_mode must be raw or rank")
    return MappingProxyType(
        {
            "schema_version": _SCHEMA_VERSION,
            "task": task,
            "qa": MappingProxyType(
                {
                    "system_prompt": system_prompt,
                    "prompt_revision": prompt_revision,
                    "output_format": output_format,
                    "structured_output_mode": structured_output_mode,
                    "identifier_policy": identifier_policy,
                    **({"allow_json_fence": True} if allow_json_fence else {}),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "use_seed": use_seed,
                    "required_capabilities": required,
                }
            ),
            "retrieval": MappingProxyType(
                {
                    "top_k": top_k,
                    "query_mode": query_mode,
                    "source_mode": source_mode,
                    **({"score_mode": "rank"} if score_mode == "rank" else {}),
                }
            ),
        }
    )


def _parse_corpus_retriever_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _reject_embedded_secrets(parameters, path="corpus retriever parameters")
    data = _strict_mapping(
        parameters,
        path="corpus retriever parameters",
        allowed={
            "schema_version",
            "benchmark_id",
            "profile_id",
            "corpus_path",
            "corpus_sha256",
            "corpus_revision",
            "expected_document_count",
            "options",
            "artifact_files",
        },
        required={
            "schema_version",
            "benchmark_id",
            "profile_id",
            "corpus_path",
            "corpus_sha256",
            "corpus_revision",
            "expected_document_count",
        },
    )
    _require_schema_version(
        data["schema_version"],
        path="corpus retriever parameters.schema_version",
    )
    benchmark_id = _require_text(data["benchmark_id"], label="retriever benchmark_id")
    profile_id = _require_text(data["profile_id"], label="retriever profile_id")
    profile = retriever_profile(profile_id)
    options = _mapping_or_empty(
        data.get("options"),
        path="corpus retriever options",
    )
    if profile.backend == "memory_bm25":
        allowed_options = {"retriever_id", "k1", "b"}
    else:
        allowed_options = {
            "retriever_id",
            "batch_size",
            "device",
            "local_files_only",
            "model_path",
        }
    unknown_options = set(options) - allowed_options
    if unknown_options:
        raise ValidationError(
            "corpus retriever options contain unsupported fields: "
            + ", ".join(sorted(unknown_options))
        )
    normalized_options = dict(options)
    if profile.backend == "sentence_transformers_dense":
        if normalized_options.get("local_files_only", True) is not True:
            raise ValidationError("production dense retriever requires local_files_only=true")
        normalized_options["local_files_only"] = True
        if "model_path" in normalized_options:
            normalized_options["model_path"] = str(
                _absolute_path(normalized_options["model_path"], label="dense model_path")
            )
    return freeze_mapping(
        {
            "benchmark_id": benchmark_id,
            "profile_id": profile_id,
            "corpus_path": _absolute_path(data["corpus_path"], label="corpus_path"),
            "corpus_sha256": _sha256_text(
                data["corpus_sha256"],
                label="corpus_sha256",
            ),
            "corpus_revision": _require_text(
                data["corpus_revision"],
                label="corpus_revision",
            ),
            "expected_document_count": _positive_int(
                data["expected_document_count"],
                label="expected_document_count",
            ),
            "options": _canonical_mapping(
                normalized_options,
                label="corpus retriever options",
            ),
            "artifact_files": _artifact_file_locks(
                data.get("artifact_files", ()),
                path="corpus retriever artifact_files",
            ),
        }
    )


def _load_replay_fixture(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    raw = _read_checksum_bytes(
        path,
        expected_sha256,
        max_bytes=_MAX_REPLAY_BYTES,
        label="replay fixture",
    )
    parsed = _strict_json_bytes(raw, label="replay fixture")
    data = _strict_mapping(
        parsed,
        path="replay fixture",
        allowed={"schema_version", "responses", "strict_request_hashes"},
        required={"schema_version", "responses"},
    )
    _require_schema_version(data["schema_version"], path="replay fixture.schema_version")
    values = data["responses"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ArtifactError("replay fixture responses must be a non-empty array")
    responses: list[str | GenerationResponse] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            responses.append(value)
            continue
        item = _strict_mapping(
            value,
            path=f"replay fixture.responses[{index}]",
            allowed={"text", "model", "finish_reason", "usage", "raw"},
            required={"text"},
        )
        text = item["text"]
        if not isinstance(text, str):
            raise ArtifactError(f"replay fixture.responses[{index}].text must be a string")
        usage_data = _strict_mapping(
            item.get("usage", {}),
            path=f"replay fixture.responses[{index}].usage",
            allowed={"input_tokens", "output_tokens", "cost_usd"},
            required=set(),
        )
        try:
            usage = Usage(**usage_data)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"replay fixture.responses[{index}].usage is invalid: {exc}"
            ) from exc
        responses.append(
            GenerationResponse(
                text=text,
                model=str(item.get("model", "")),
                finish_reason=str(item.get("finish_reason", "replay")),
                usage=usage,
                raw=_mapping_or_empty(
                    item.get("raw"),
                    path=f"replay fixture.responses[{index}].raw",
                ),
            )
        )
    hashes = _text_sequence(
        data.get("strict_request_hashes", ()),
        path="replay fixture.strict_request_hashes",
        allow_empty=True,
    )
    if hashes and len(hashes) != len(responses):
        raise ArtifactError("replay fixture strict_request_hashes must match response count")
    for value in hashes:
        _require_sha256(value, label="replay strict request hash")
    return MappingProxyType(
        {
            "responses": tuple(responses),
            "strict_request_hashes": hashes,
        }
    )


def _load_locked_corpus(config: Mapping[str, Any]) -> tuple[Document, ...]:
    path = Path(config["corpus_path"])
    identity = _regular_file_identity(path, label="prepared corpus")
    return _load_locked_corpus_snapshot(
        str(path),
        config["corpus_sha256"],
        config["corpus_revision"],
        identity,
    )


@lru_cache(maxsize=16)
def _load_locked_corpus_snapshot(
    path_text: str,
    expected_sha256: str,
    corpus_revision: str,
    identity: tuple[int, int, int, int, int],
) -> tuple[Document, ...]:
    """Stream and cache one immutable corpus snapshot across arm runtimes."""

    path = Path(path_text)
    digest = hashlib.sha256()
    documents: list[Document] = []
    identifiers: set[str] = set()
    try:
        with path.open("rb") as handle:
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ArtifactError("prepared corpus changed before it could be read")
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                line = raw_line.rstrip(b"\r\n")
                if not line:
                    raise ArtifactError(f"prepared corpus contains a blank line at {line_number}")
                if len(line) > _MAX_CORPUS_LINE_BYTES:
                    raise ArtifactError(f"prepared corpus line {line_number} exceeds size limit")
                value = _strict_json_bytes(
                    line,
                    label=f"prepared corpus line {line_number}",
                )
                item = _strict_mapping(
                    value,
                    path=f"prepared corpus line {line_number}",
                    allowed={"id", "title", "text"},
                    required={"id", "title", "text"},
                )
                document_id = _require_text(
                    item["id"],
                    label=f"prepared corpus line {line_number}.id",
                )
                if document_id in identifiers:
                    raise ArtifactError(f"prepared corpus contains duplicate ID {document_id!r}")
                identifiers.add(document_id)
                title = item["title"]
                text = item["text"]
                if not isinstance(title, str):
                    raise ArtifactError(
                        f"prepared corpus line {line_number}.title must be a string"
                    )
                if not isinstance(text, str) or not text:
                    raise ArtifactError(
                        f"prepared corpus line {line_number}.text must be non-empty text"
                    )
                documents.append(
                    Document(
                        doc_id=document_id,
                        title=title,
                        text=text,
                        corpus_revision=corpus_revision,
                    )
                )
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ArtifactError("prepared corpus changed while it was being read")
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(f"prepared corpus is unavailable: {path}") from exc
    if digest.hexdigest() != expected_sha256:
        raise ArtifactError(f"prepared corpus checksum mismatch: {path}")
    if not documents:
        raise ArtifactError("prepared corpus contains no documents")
    return tuple(documents)


def _regular_file_identity(path: Path, *, label: str) -> tuple[int, int, int, int, int]:
    if not path.is_absolute():
        raise ArtifactError(f"{label} path must be absolute")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ArtifactError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactError(f"{label} is not a regular file: {path}")
    return _stat_identity(metadata)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _verify_checksum_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    source = Path(path)
    identity = _regular_file_identity(source, label=label)
    _require_sha256(expected_sha256, label=f"{label} SHA-256")
    observed = _hash_file_snapshot(str(source), identity)
    if observed != expected_sha256:
        raise ArtifactError(f"{label} checksum mismatch: {source}")


def _hash_file_snapshot(
    path_text: str,
    identity: tuple[int, int, int, int, int],
) -> str:
    path = Path(path_text)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ArtifactError(f"artifact changed before it could be read: {path}")
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ArtifactError(f"artifact changed while it was being read: {path}")
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(f"artifact is unavailable: {path}") from exc
    return digest.hexdigest()


def _render_visible_qa_prompt(state: FrozenAgentState) -> str:
    """Render only fields deliberately exposed by ``FrozenAgentState``.

    ``task_id`` and ``metadata`` are intentionally omitted because either may
    carry benchmark identity.  Evaluator gold is physically absent from this
    type and never enters the predictor API.
    """

    sections: list[str] = []
    if state.instruction:
        sections.append(f"Instruction:\n{state.instruction}")
    if state.messages:
        conversation = "\n".join(f"{message.role}: {message.content}" for message in state.messages)
        sections.append(f"Conversation:\n{conversation}")
    sections.append(f"Question:\n{state.query}")
    if state.context:
        context = "\n\n".join(
            _render_visible_context_block(block, index=index)
            for index, block in enumerate(state.context, start=1)
        )
        sections.append(f"Visible context:\n{context}")
    if state.tool_names:
        sections.append("Available tool names:\n" + "\n".join(state.tool_names))
    return "\n\n".join(sections)


def _qa_response_format(
    *,
    benchmark_id: str,
    output_format: str,
    structured_output_mode: str,
) -> Mapping[str, Any] | None:
    if output_format == "answer_only":
        return None
    if structured_output_mode == "json_object":
        return {"type": "json_object"}
    if structured_output_mode != "json_schema":  # pragma: no cover - parsed invariant
        raise ValidationError("unsupported structured QA output mode")
    identifier_limit = 0 if benchmark_id == "qasper.v0_3_dev" else 16
    evidence_text_limit = 3 if benchmark_id == "qasper.v0_3_dev" else 16
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "support_ids", "evidence_ids", "evidence_texts"],
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "support_ids": {
                "type": "array",
                "maxItems": identifier_limit,
                "items": {"type": "string", "minLength": 1},
            },
            "evidence_ids": {
                "type": "array",
                "maxItems": identifier_limit,
                "items": {"type": "string", "minLength": 1},
            },
            "evidence_texts": {
                "type": "array",
                "maxItems": evidence_text_limit,
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_prediction",
            "strict": True,
            "schema": schema,
        },
    }


def _render_visible_context_block(block: Any, *, index: int) -> str:
    support_id = block.metadata.get("official_support_id", block.block_id)
    source = f"\nSource: {block.source}" if block.source else ""
    return (
        f"[Context {index}]\nBlock ID: {block.block_id}\n"
        f"Support ID: {support_id}{source}\nText: {block.text}"
    )


def _parse_visible_qa_prediction(
    text: str,
    *,
    benchmark_id: str,
    state: FrozenAgentState,
    output_format: str,
    identifier_policy: str = "native_sets.v1",
    allow_json_fence: bool = False,
    preserve_empty_answer: bool = False,
) -> tuple[QAPrediction, Mapping[str, Any]]:
    if type(allow_json_fence) is not bool:
        raise ValidationError("QA allow_json_fence must be boolean")
    if allow_json_fence and output_format != "structured_json":
        raise ValidationError("QA allow_json_fence requires output_format=structured_json")
    if not isinstance(identifier_policy, str) or identifier_policy not in {
        "native_sets.v1", "strict_strings.v1"
    }:
        raise ValidationError("unsupported QA identifier policy")
    if output_format == "answer_only":
        return QAPrediction(answer=text.strip()), {
            "identifier_policy": identifier_policy,
            "support_id_validation": "not_applicable_answer_only.v1",
            "valid_support_id_count": 0,
            "invalid_support_id_count": 0,
            "invalid_support_ids_sha256": None,
            "evidence_id_validation": "not_applicable_answer_only.v1",
            "valid_evidence_id_count": 0,
            "invalid_evidence_id_count": 0,
            "invalid_evidence_ids_sha256": None,
        }
    if output_format != "structured_json":  # pragma: no cover - parsed config invariant
        raise ValidationError("unsupported QA output format")
    envelope_audit: dict[str, Any] = {}
    if allow_json_fence:
        if not isinstance(text, str):
            raise ValidationError("QA backend response must be JSON text")
        original_text = text
        fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text.strip(), re.DOTALL)
        if fenced is not None:
            text = fenced.group(1)
        envelope_audit = {
            "json_envelope_policy": "single_json_fence.v1",
            "json_fence_removed": fenced is not None,
            "original_response_text_sha256": content_hash(original_text),
        }
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: _reject_json_constant(item),
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationError("QA backend response is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValidationError("structured QA response must be a JSON object")
    expected = {"answer", "support_ids", "evidence_ids", "evidence_texts"}
    if set(value) != expected:
        raise ValidationError(
            "structured QA response schema mismatch: "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    answer = value["answer"]
    if not isinstance(answer, str) or (not answer.strip() and not preserve_empty_answer):
        raise ValidationError("structured QA answer must be non-empty text")
    identifier_audit: dict[str, Any] = {"identifier_policy": identifier_policy, **envelope_audit}
    malformed_support_ids: tuple[Any, ...] = ()
    malformed_evidence_ids: tuple[Any, ...] = ()
    if identifier_policy == "strict_strings.v1":
        support_ids = _prediction_string_array(
            value["support_ids"], label="structured QA support_ids"
        )
        evidence_ids = _prediction_string_array(
            value["evidence_ids"], label="structured QA evidence_ids"
        )
    else:
        support_ids, malformed_support_ids, support_audit = _normalize_qa_identifier_array(
            value["support_ids"],
            label="structured QA support_ids",
            kind=(
                "support_pair"
                if benchmark_id in {"2wiki.multihop", "hotpotqa.distractor"}
                else "opaque"
            ),
        )
        evidence_ids, malformed_evidence_ids, evidence_audit = _normalize_qa_identifier_array(
            value["evidence_ids"],
            label="structured QA evidence_ids",
            kind="evidence_triple" if benchmark_id == "2wiki.multihop" else "opaque",
        )
        identifier_audit.update(
            {f"support_{key}": item for key, item in support_audit.items()}
        )
        identifier_audit.update(
            {f"evidence_{key}": item for key, item in evidence_audit.items()}
        )
    evidence_texts, duplicate_evidence_texts = _prediction_string_array_keep_first(
        value["evidence_texts"],
        label="structured QA evidence_texts",
    )
    visible_support_ids = {
        str(block.metadata.get("official_support_id", block.block_id)) for block in state.context
    }
    valid_support_ids, invalid_support_ids = _partition_visible_support_ids(
        support_ids,
        visible_support_ids=visible_support_ids,
        benchmark_id=benchmark_id,
    )
    all_invalid_support_ids = (*malformed_support_ids, *invalid_support_ids)
    support_validation = (
        "filter_unknown_or_invalid_canonical_support.v1"
        if benchmark_id in {"2wiki.multihop", "hotpotqa.distractor"}
        else "filter_unknown_visible_only.v1"
    )
    valid_evidence_ids = evidence_ids
    invalid_evidence_ids: tuple[str, ...] = ()
    evidence_validation = "not_applicable_non_2wiki.v1"
    if benchmark_id == "2wiki.multihop":
        valid_evidence_ids, invalid_evidence_ids = _partition_two_wiki_evidence_ids(evidence_ids)
        evidence_validation = "filter_invalid_2wiki_canonical_triples.v1"
    all_invalid_evidence_ids = (*malformed_evidence_ids, *invalid_evidence_ids)
    return (
        QAPrediction(
            answer=answer.strip(),
            support_ids=valid_support_ids,
            evidence_ids=valid_evidence_ids,
            evidence_texts=evidence_texts,
        ),
        {
            **identifier_audit,
            "support_id_validation": support_validation,
            "valid_support_id_count": len(valid_support_ids),
            "invalid_support_id_count": len(all_invalid_support_ids),
            "invalid_support_ids_sha256": (
                content_hash(all_invalid_support_ids) if all_invalid_support_ids else None
            ),
            "evidence_id_validation": evidence_validation,
            "valid_evidence_id_count": len(valid_evidence_ids),
            "invalid_evidence_id_count": len(all_invalid_evidence_ids),
            "invalid_evidence_ids_sha256": (
                content_hash(all_invalid_evidence_ids) if all_invalid_evidence_ids else None
            ),
            "evidence_text_duplicate_policy": "deduplicate_exact_trimmed_keep_first.v1",
            "duplicate_evidence_text_count": len(duplicate_evidence_texts),
            "duplicate_evidence_texts_sha256": (
                content_hash(duplicate_evidence_texts) if duplicate_evidence_texts else None
            ),
        },
    )


def _normalize_qa_identifier_array(
    value: Any,
    *,
    label: str,
    kind: str,
) -> tuple[tuple[str, ...], tuple[Any, ...], Mapping[str, Any]]:
    """Normalize set-like IDs without inferring evidence or reading references.

    Native support pairs may be JSON strings, nested pairs, or an unambiguous
    flat title/integer stream. Native 2Wiki triples must be JSON strings or
    nested triples: a flat string stream is ambiguous and is never grouped.
    Unknown IDs still pass through the visible-support filter. Invalid typed
    entries are excluded and hash-audited; no coercion or title-only lookup is
    permitted. All policies keep strict JSON and the required response fields.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    identifiers: list[str] = []
    invalid: list[Any] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    normalized_count = 0
    index = 0
    while index < len(value):
        raw = value[index]
        index += 1
        # Only a non-boolean, non-negative integer makes this shape unambiguous.
        # Never turn e.g. [title, true], [title, "0"], or title alone into an ID.
        if (
            kind == "support_pair"
            and isinstance(raw, str)
            and raw.strip()
            and index < len(value)
            and isinstance(value[index], int)
            and not isinstance(value[index], bool)
            and value[index] >= 0
        ):
            raw = [raw, value[index]]
            index += 1
        if isinstance(raw, str) and raw.strip():
            identifier = raw.strip()
            if kind != "opaque":
                try:
                    native = json.loads(identifier)
                except json.JSONDecodeError:
                    native = None
                if _is_native_qa_identifier(native, kind=kind):
                    identifier = canonical_json(native)
        elif _is_native_qa_identifier(raw, kind=kind):
            identifier = canonical_json(raw)
        else:
            invalid.append(raw)
            continue
        if not isinstance(raw, str) or identifier != raw:
            normalized_count += 1
        if identifier in seen:
            duplicates.append(identifier)
            continue
        seen.add(identifier)
        identifiers.append(identifier)
    return (
        tuple(identifiers),
        tuple(invalid),
        {
            "raw_identifier_array_sha256": content_hash(value),
            "normalized_id_count": normalized_count,
            "duplicate_id_count": len(duplicates),
            "duplicate_ids_sha256": content_hash(duplicates) if duplicates else None,
        },
    )


def _is_native_qa_identifier(value: Any, *, kind: str) -> bool:
    if not isinstance(value, list):
        return False
    if kind == "support_pair":
        return (
            len(value) == 2
            and isinstance(value[0], str)
            and bool(value[0].strip())
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
            and value[1] >= 0
        )
    return (
        kind == "evidence_triple"
        and len(value) == 3
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _partition_visible_support_ids(
    identifiers: tuple[str, ...],
    *,
    visible_support_ids: set[str],
    benchmark_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    require_canonical_pair = benchmark_id in {
        "2wiki.multihop",
        "hotpotqa.distractor",
    }
    valid: list[str] = []
    invalid: list[str] = []
    for identifier in identifiers:
        if identifier not in visible_support_ids or (
            require_canonical_pair and not _is_canonical_support_pair(identifier)
        ):
            invalid.append(identifier)
        else:
            valid.append(identifier)
    return tuple(valid), tuple(invalid)


def _is_canonical_support_pair(identifier: str) -> bool:
    try:
        value = json.loads(identifier)
    except json.JSONDecodeError:
        return False
    return bool(
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and value[0].strip()
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and value[1] >= 0
        and canonical_json(value) == identifier
    )


def _partition_two_wiki_evidence_ids(
    identifiers: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    valid: list[str] = []
    invalid: list[str] = []
    for identifier in identifiers:
        try:
            value = json.loads(identifier)
        except json.JSONDecodeError:
            invalid.append(identifier)
            continue
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(item, str) and item.strip() for item in value)
            or canonical_json(value) != identifier
        ):
            invalid.append(identifier)
            continue
        valid.append(identifier)
    return tuple(valid), tuple(invalid)


def _prediction_string_array(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValidationError(f"{label} must contain non-empty strings")
    normalized = tuple(item.strip() for item in items)
    if len(normalized) != len(set(normalized)):
        raise ValidationError(f"{label} must not contain duplicates")
    return normalized


def _prediction_string_array_keep_first(
    value: Any,
    *,
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize a model-produced set-like array and audit exact repeats.

    Evidence-text scoring is set based, so an otherwise valid model response
    must not abort a run solely because it repeated the same visible span.
    The first occurrence is retained, subsequent exact trimmed repeats are
    returned for hash-only audit metadata, and malformed entries still fail
    closed.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValidationError(f"{label} must contain non-empty strings")
    unique: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if normalized in seen:
            duplicates.append(normalized)
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique), tuple(duplicates)


def _resolve_retriever_builder(
    spec: LockedRetrieverSpec,
) -> ProductionRetrieverBuilderProtocol:
    candidate = resolve_driver_entrypoint(spec.entrypoint)
    if not callable(candidate):
        raise ArtifactError("retriever builder entrypoint must be a zero-argument factory")
    try:
        builder = candidate()
    except TypeError as exc:
        raise ArtifactError("retriever builder entrypoint must be a zero-argument factory") from exc
    if not isinstance(builder, ProductionRetrieverBuilderProtocol):
        raise ArtifactError("retriever builder does not implement the production protocol")
    if builder.builder_kind != "retriever":
        raise ArtifactError("nested production builder kind must be 'retriever'")
    observed = builder_fingerprint_sha256(builder)
    if observed != spec.fingerprint_sha256:
        raise ArtifactError("retriever builder fingerprint checksum mismatch")
    return builder


def _fingerprint(value: Any, *, label: str) -> Mapping[str, Any]:
    fingerprint = getattr(value, "fingerprint", None)
    if not callable(fingerprint):
        raise ValidationError(f"{label} must expose fingerprint()")
    identity = fingerprint()
    if not isinstance(identity, Mapping) or not identity:
        raise ValidationError(f"{label} fingerprint must be a non-empty mapping")
    return _canonical_mapping(identity, label=f"{label} fingerprint")


def _validate_product_hash(observed: str, expected: str, *, label: str) -> str:
    _require_sha256(observed, label=f"{label} observed SHA-256")
    _require_sha256(expected, label=f"{label} expected SHA-256")
    if observed != expected:
        raise ArtifactError(f"{label} checksum mismatch")
    return observed


def _artifact_file_locks(value: Any, *, path: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{path} must be an array")
    locks: list[Mapping[str, Any]] = []
    names: set[str] = set()
    paths: set[Path] = set()
    for index, item in enumerate(value):
        data = _strict_mapping(
            item,
            path=f"{path}[{index}]",
            allowed={"name", "path", "sha256"},
            required={"name", "path", "sha256"},
        )
        name = _require_text(data["name"], label=f"{path}[{index}].name")
        artifact_path = _absolute_path(data["path"], label=f"{path}[{index}].path")
        digest = _sha256_text(data["sha256"], label=f"{path}[{index}].sha256")
        if name in names or artifact_path in paths:
            raise ValidationError(f"{path} contains duplicate names or paths")
        names.add(name)
        paths.add(artifact_path)
        locks.append(
            freeze_mapping(
                {
                    "name": name,
                    "path": artifact_path,
                    "sha256": digest,
                }
            )
        )
    return tuple(locks)


def _validate_transformers_artifact_layout(
    model_path: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    """Require artifact names to be exact paths within the local snapshot."""

    for artifact in artifacts:
        artifact_path = Path(artifact["path"])
        try:
            relative = artifact_path.relative_to(model_path).as_posix()
        except ValueError as exc:
            raise ValidationError("Transformers artifact paths must be inside model_path") from exc
        if not relative or relative == ".":
            raise ValidationError("Transformers artifacts must identify files")
        if artifact["name"] != relative:
            raise ValidationError(
                "Transformers artifact names must equal their model-relative POSIX paths"
            )


def _transformers_snapshot_sha256(
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    return content_hash(
        {
            "protocol": "agentinputlab.transformers-snapshot-tree.v1",
            "files": tuple(
                {
                    "path": artifact["name"],
                    "sha256": artifact["sha256"],
                }
                for artifact in sorted(artifacts, key=lambda item: item["name"])
            ),
        }
    )


def _verify_transformers_snapshot(
    model_path: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    """Verify the complete, closed model tree and every declared file digest."""

    _validate_transformers_artifact_layout(model_path, artifacts)
    try:
        root_metadata = model_path.lstat()
    except OSError as exc:
        raise ArtifactError(f"Transformers model snapshot is unavailable: {model_path}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ArtifactError(f"Transformers model_path is not a closed directory: {model_path}")

    observed_paths: set[Path] = set()
    try:
        for candidate in model_path.rglob("*"):
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactError(
                    "Transformers model snapshot contains a symlink: "
                    f"{candidate.relative_to(model_path)}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactError(
                    "Transformers model snapshot contains a non-regular entry: "
                    f"{candidate.relative_to(model_path)}"
                )
            observed_paths.add(candidate.resolve())
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(
            f"Transformers model snapshot cannot be enumerated: {model_path}"
        ) from exc

    expected_paths = {Path(artifact["path"]) for artifact in artifacts}
    if observed_paths != expected_paths:
        missing = sorted(
            path.relative_to(model_path).as_posix() for path in observed_paths - expected_paths
        )
        stale = sorted(
            path.relative_to(model_path).as_posix() for path in expected_paths - observed_paths
        )
        raise ArtifactError(
            "Transformers artifact_files do not cover the complete model snapshot; "
            f"unlisted={missing[:5]}, absent={stale[:5]}"
        )
    for artifact in artifacts:
        _verify_checksum_file(
            artifact["path"],
            artifact["sha256"],
            label=f"Transformers artifact {artifact['name']!r}",
        )


def _read_checksum_bytes(
    path: str | Path,
    expected_sha256: str,
    *,
    max_bytes: int | None,
    label: str,
) -> bytes:
    path = Path(path)
    if not path.is_absolute():
        raise ArtifactError(f"{label} path must be absolute")
    _require_sha256(expected_sha256, label=f"{label} SHA-256")
    try:
        size = path.stat().st_size
        if not path.is_file():
            raise ArtifactError(f"{label} is not a regular file: {path}")
        if max_bytes is not None and size > max_bytes:
            raise ArtifactError(f"{label} exceeds the size limit: {path}")
        raw = path.read_bytes()
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(f"{label} is unavailable: {path}") from exc
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ArtifactError(f"{label} checksum mismatch: {path}")
    return raw


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _reject_json_constant(value),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not strict UTF-8 JSON") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValidationError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_embedded_secrets(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path} keys must be strings")
            child = f"{path}.{key}"
            if key == "require_api_key" or path.endswith(".extra_headers"):
                _reject_embedded_secrets(item, path=child)
                continue
            # OpenAI tool declarations use arbitrary domain field names as keys
            # below JSON Schema ``properties``.  A field such as
            # ``new_password`` names an argument; it is not itself a credential.
            # Keep recursively scanning the property's schema so secret-like
            # defaults, examples, descriptions, and other literal values still
            # fail closed.
            if (
                path.startswith("backend parameters.tools.definitions[")
                and ".function.parameters" in path
                and path.endswith(".properties")
            ):
                _reject_embedded_secrets(item, path=child)
                continue
            if _SENSITIVE_KEY.search(key):
                if not key.casefold().endswith("_env"):
                    raise ValidationError(
                        f"{child} looks secret-bearing; store only an environment-variable name"
                    )
                _require_env_name(item, label=child)
                continue
            _reject_embedded_secrets(item, path=child)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_embedded_secrets(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        raise ValidationError(
            f"{path} contains secret-like text; use environment-variable indirection"
        )


def _capabilities(value: Any, *, path: str) -> tuple[ModelCapability, ...]:
    names = _text_sequence(value, path=path, allow_empty=True)
    try:
        return tuple(sorted((ModelCapability(name) for name in names), key=lambda item: item.value))
    except ValueError as exc:
        raise ValidationError(f"{path} contains an unknown capability") from exc


def _environment_names(value: Any, *, path: str) -> tuple[str, ...]:
    names = _text_sequence(value, path=path, allow_empty=True)
    for name in names:
        _require_env_name(name, label=path)
    return tuple(sorted(names))


def _text_sequence(value: Any, *, path: str, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{path} must be an array of strings")
    items = tuple(value)
    if not allow_empty and not items:
        raise ValidationError(f"{path} must not be empty")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValidationError(f"{path} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise ValidationError(f"{path} must not contain duplicates")
    return items


def _strict_mapping(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{path} keys must be strings")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValidationError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValidationError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return value


def _mapping_or_empty(value: Any, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    return value


def _canonical_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        return freeze_mapping(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be canonical JSON") from exc


def _require_schema_version(value: Any, *, path: str) -> None:
    if value != _SCHEMA_VERSION or isinstance(value, bool):
        raise ValidationError(f"{path} must equal {_SCHEMA_VERSION}")


def _supports_cell_split(benchmark_id: str, split: str) -> bool:
    # FiQA has an explicitly validated, train/test-disjoint native dev
    # adapter. Keep every other cell on its registered default split.
    return split == _CELL_SPLITS[benchmark_id] or (benchmark_id == "beir.fiqa_2018" and split == "dev")


def _require_cell(value: str) -> None:
    if value not in _CELL_SPLITS:
        raise ValidationError(
            f"unsupported static production benchmark {value!r}; "
            f"available: {', '.join(sorted(_CELL_SPLITS))}"
        )


def _require_retriever_cell(value: str) -> None:
    if value not in _RETRIEVER_ROUTE_CELLS:
        raise ValidationError(
            f"unsupported production retriever benchmark {value!r}; "
            f"available: {', '.join(sorted(_RETRIEVER_ROUTE_CELLS))}"
        )


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value


def _verify_locked_file(path: Path, expected_sha256: str, *, label: str) -> str:
    if not path.is_file():
        raise ArtifactError(f"{label} is unavailable: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ArtifactError(f"{label} checksum mismatch")
    return observed


def _absolute_path(value: Any, *, label: str, resolve_symlinks: bool = True) -> Path:
    text = _require_text(value, label=label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValidationError(f"{label} must be an absolute path")
    # A venv Python is commonly a symlink; executing its target loses the venv.
    return path.resolve() if resolve_symlinks else path


def _absolute_http_url(value: Any, *, label: str) -> str:
    text = _require_text(value, label=label)
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError(f"{label} must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValidationError(f"{label} must not contain credentials, query, or fragment")
    return text.rstrip("/")


def _sha256_text(value: Any, *, label: str) -> str:
    text = _require_text(value, label=label)
    _require_sha256(text, label=label)
    return text


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")


def _require_env_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _ENV_NAME.fullmatch(value) is None:
        raise ValidationError(f"{label} must be an environment-variable name")
    return value


def _require_seed(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("seed must be a non-negative integer")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, *, label: str) -> float:
    number = _nonnegative_number(value, label=label)
    if number <= 0:
        raise ValidationError(f"{label} must be positive")
    return number


def _nonnegative_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValidationError(f"{label} must be finite and non-negative")
    return float(value)


__all__ = [
    "LockedRetrieverSpec",
    "PreparedCorpusRetrieverBuilder",
    "ProductionBackendBuilder",
    "ProductionDownstreamPredictor",
    "ProductionPredictorBuilder",
    "ProductionRetrieverBuildRequest",
    "ProductionRetrieverBuilderProtocol",
    "ProductionRetrieverRouter",
    "builder_fingerprint_sha256",
    "prepared_corpus_retriever_builder_factory",
    "production_backend_builder_factory",
    "production_predictor_builder_factory",
    "retriever_product_sha256",
    "validate_predictor_product_sha256",
    "validate_retriever_product_sha256",
    "validate_runtime_product_sha256",
]
