"""Thin external bridge for the prompt Rewrite-Retrieve-Read system.

The unlicensed upstream checkout, query parser, search environment, snippet
serialization, and reader remain in a user-mounted worker.  AgentInputLab only
binds controlled dependencies, validates the emitted trajectory, and converts
it to a typed patch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend
from agentinputlab.backends.retrieval import RetrievalResult, Retriever
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
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    dependency_fingerprint,
    finite_number,
    mapping_sequence,
    nonnegative_integer,
    optional_text,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

RRR_UPSTREAM_COMMIT = "6bab6243e60c7f3e526b6ee44f95381b1024d471"
RRR_PROTOCOL = "agentinputlab.rrr.inference.v2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RRRPromptControls:
    # The current AgentInputLab state contract has one reader query.  Supporting
    # upstream multi-query RRR by concatenating rewrites would silently change
    # its retrieval semantics, so this release deliberately exposes only the
    # faithful single-query profile.
    max_queries: int = 1
    top_k: int = 5
    query_parser: str = "upstream_semicolon_terminator_single"
    search_mode: str = "bing_cache"
    rewriter_profile: str = "prompt"
    search_window_start: str = ""
    search_window_end: str = ""
    max_rewriter_tokens: int = 100
    max_reader_tokens: int = 300
    max_input_chars: int = 200_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_queries, bool)
            or not isinstance(self.max_queries, int)
            or self.max_queries != 1
        ):
            raise ValidationError(
                "RRR prompt_external currently supports exactly one parsed query; "
                "multi-query RRR requires a query-set execution contract"
            )
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not 1 <= self.top_k <= 100
        ):
            raise ValidationError("RRR top_k must be positive")
        if self.search_mode not in {"bing_cache", "controlled_static"}:
            raise ValidationError("RRR search_mode must be bing_cache or controlled_static")
        if self.rewriter_profile not in {"prompt", "warmup_checkpoint", "ppo_checkpoint"}:
            raise ValidationError("RRR rewriter_profile is unsupported")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.max_rewriter_tokens,
                self.max_reader_tokens,
                self.max_input_chars,
            )
        ):
            raise ValidationError("RRR generation/input budgets must be positive")
        if self.search_window_start > self.search_window_end:
            raise ValidationError("RRR search time window is reversed")
        if not self.query_parser.strip():
            raise ValidationError("RRR query_parser must not be empty")


@dataclass(slots=True)
class RRRPromptExternal:
    """Prompt-RRR runtime; warm-up and PPO training remain separate recipes."""

    client: WorkerClient | None = None
    source_license_acknowledged: bool = False
    checkpoint_license_acknowledged: bool = False
    controls: RRRPromptControls = RRRPromptControls()
    prompt_hash: str = ""
    reader_prompt_hash: str = ""
    search_artifact_hash: str = ""
    corpus_revision: str = ""
    index_revision: str = ""
    rewriter_checkpoint_revision: str = "prompt-api-frozen"
    reader_checkpoint_revision: str = "frozen-reader"
    method_id: str = "rrr"
    release_id: str = "rewrite.rrr.prompt_external"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        expected_release = {
            "prompt": "rewrite.rrr.prompt_external",
            "warmup_checkpoint": "rewrite.rrr.warmup_external",
            "ppo_checkpoint": "rewrite.rrr.ppo_external",
        }[self.controls.rewriter_profile]
        if self.release_id != expected_release:
            raise ValidationError("RRR release_id does not match rewriter_profile")
        if not isinstance(self.source_license_acknowledged, bool) or not isinstance(
            self.checkpoint_license_acknowledged, bool
        ):
            raise ValidationError("RRR license acknowledgements must be boolean")
        for label, value in (
            ("prompt_hash", self.prompt_hash),
            ("reader_prompt_hash", self.reader_prompt_hash),
            ("search_artifact_hash", self.search_artifact_hash),
        ):
            if not _HEX64.fullmatch(value):
                raise ValidationError(f"RRR {label} must be a SHA-256 digest")
        if (
            not self.rewriter_checkpoint_revision.strip()
            or not self.reader_checkpoint_revision.strip()
        ):
            raise ValidationError("RRR rewriter/reader revisions must not be empty")
        if self.controls.search_mode == "controlled_static" and (
            not self.corpus_revision.strip() or not self.index_revision.strip()
        ):
            raise ValidationError("controlled RRR requires corpus_revision and index_revision")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("RRR requires a non-empty query")
        _require_source_license_acknowledgement(self.source_license_acknowledged)
        _require_checkpoint_license_acknowledgement(self.checkpoint_license_acknowledged)
        worker = resolve_worker(self.client, context, label="RRR")
        worker_fp = worker_fingerprint(worker, label="RRR")
        backend_fp = dependency_fingerprint(
            context.backend, label="RRR reader/generator", required=True
        )
        retriever_fp = dependency_fingerprint(
            context.retriever,
            label="RRR retriever",
            required=self.controls.search_mode == "controlled_static",
        )
        backend_hash = content_hash(backend_fp)
        retriever_hash = content_hash(retriever_fp)
        input_hash = content_hash(
            {
                "query": state.query,
                "instruction": state.instruction,
                "messages": state.messages,
                "controls": asdict(self.controls),
            }
        )
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=RRR_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "query": state.query,
                "instruction": state.instruction,
                "messages": state.messages,
                "controls": asdict(self.controls),
                "prompt_hash": self.prompt_hash,
                "reader_prompt_hash": self.reader_prompt_hash,
                "search_artifact_hash": self.search_artifact_hash,
                "corpus_revision": self.corpus_revision,
                "index_revision": self.index_revision,
                "rewriter_checkpoint_revision": self.rewriter_checkpoint_revision,
                "reader_checkpoint_revision": self.reader_checkpoint_revision,
                "source_license_acknowledged": self.source_license_acknowledged,
                "checkpoint_license_acknowledged": self.checkpoint_license_acknowledged,
                "backend_artifact_hash": backend_hash,
                "retriever_artifact_hash": retriever_hash,
                "input_artifact_hash": input_hash,
                "seed": context.seed,
            },
            dependency_fingerprints={
                "backend": backend_fp,
                "retriever": retriever_fp,
            },
            minimum_reservation={"model_calls": 2, "retrieval_calls": 1},
            protocol_version=RRR_PROTOCOL,
        )
        response = call_worker(worker, request)
        payload = response.payload
        for key, expected in (
            ("prompt_hash", self.prompt_hash),
            ("reader_prompt_hash", self.reader_prompt_hash),
            ("search_artifact_hash", self.search_artifact_hash),
            ("rewriter_checkpoint_revision", self.rewriter_checkpoint_revision),
            ("reader_checkpoint_revision", self.reader_checkpoint_revision),
            ("backend_artifact_hash", backend_hash),
            ("retriever_artifact_hash", retriever_hash),
            ("input_artifact_hash", input_hash),
        ):
            if payload.get(key) != expected:
                raise AlgorithmOutputError(f"RRR {key} mismatch")
        for key in ("source_manifest_sha256", "trajectory_hash"):
            if not isinstance(payload.get(key), str) or not _HEX64.fullmatch(payload[key]):
                raise AlgorithmOutputError(f"RRR {key} must be a SHA-256 digest")
        if payload.get("execution_evidence") not in {
            "fake_checkout_plumbing",
            "pinned_tarball_compatible_chain",
            "pinned_checkout_compatible_chain",
        }:
            raise AlgorithmOutputError("RRR execution evidence is invalid")
        if worker_fp.get("source_manifest_sha256") != payload["source_manifest_sha256"]:
            raise AlgorithmOutputError("RRR source manifest attestation mismatch")
        queries = string_tuple(
            payload.get("queries"),
            label="RRR queries",
            allow_empty=False,
            unique=True,
        )
        if len(queries) != 1:
            raise AlgorithmOutputError("RRR prompt_external requires exactly one parsed query")
        reader_query = required_text(payload, "reader_query", label="RRR response")
        if reader_query != state.query:
            raise AlgorithmOutputError(
                "RRR reader_query must preserve the original frozen-reader task query"
            )
        documents = _parse_documents(
            payload.get("documents"),
            query_count=len(queries),
            top_k=self.controls.top_k,
            expected_corpus_revision=self.corpus_revision,
        )
        answer = optional_text(payload, "answer", label="RRR response")
        parser_mode = required_text(payload, "parser_mode", label="RRR response")
        if parser_mode != self.controls.query_parser:
            raise AlgorithmOutputError("RRR worker query parser identity mismatch")
        stage_order = string_tuple(
            payload.get("stage_order"), label="RRR stage_order", allow_empty=False
        )
        if stage_order != ("rewrite", "retrieve", "snippet", "reader"):
            raise AlgorithmOutputError("RRR worker trajectory stage order drifted")
        if response.usage.model_calls != 2 or response.usage.retrieval_calls != 1:
            raise AlgorithmOutputError(
                "RRR worker usage must report two model calls and one search"
            )

        operations: list[PatchOperation] = [
            PatchOperation(
                PatchOp.REPLACE_QUERY,
                queries[0],
                target="query",
                rationale="query sequence emitted by the pinned prompt-RRR parser",
            )
        ]
        if documents:
            operations.append(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    documents,
                    target="retrieved_evidence",
                    rationale="snippets retrieved by the prompt-RRR search stage",
                )
            )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "rrr_controls",
                    {
                        **asdict(self.controls),
                        "prompt_hash": self.prompt_hash,
                        "corpus_revision": self.corpus_revision,
                        "index_revision": self.index_revision,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "parsed_search_queries",
                    {"queries": queries, "reader_query": reader_query},
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "retrieved_snippets",
                    {"document_ids": tuple(block.block_id for block in documents)},
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "reader_output",
                    {"answer": answer, "answer_is_intervention_input": False},
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "rewritten_queries": len(queries),
                "retrieved_documents": len(documents),
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=tuple(operations),
            trace=trace,
            implementation_identity=f"thin_external_wrapper:xbmxb/RAG-query-rewriting@{RRR_UPSTREAM_COMMIT}",
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "prompt_only_release": self.controls.rewriter_profile == "prompt",
                "warmup_or_ppo": self.controls.rewriter_profile != "prompt",
                "source_license_id": "NOASSERTION",
                "source_license_acknowledged": self.source_license_acknowledged,
                "checkpoint_license_acknowledged": self.checkpoint_license_acknowledged,
                "rewriter_profile": self.controls.rewriter_profile,
                "search_profile": self.controls.search_mode,
                "search_artifact_hash": self.search_artifact_hash,
                "input_artifact_hash": input_hash,
                "trajectory_hash": payload["trajectory_hash"],
                "execution_evidence": payload["execution_evidence"],
                "worker_fingerprint": worker_fp,
                "backend_fingerprint": backend_fp,
                "retriever_fingerprint": retriever_fp,
                "reader_answer": answer,
            },
        )


def _parse_documents(
    raw: Any,
    *,
    query_count: int,
    top_k: int,
    expected_corpus_revision: str,
) -> tuple[ContextBlock, ...]:
    rows = mapping_sequence(raw, label="RRR documents")
    if len(rows) > top_k:
        raise AlgorithmOutputError("RRR worker returned more than top_k documents")
    blocks: list[ContextBlock] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        document_id = required_text(row, "document_id", label=f"RRR document {index}")
        if document_id in seen:
            raise AlgorithmOutputError("RRR worker returned duplicate document IDs")
        seen.add(document_id)
        text = required_text(row, "text", label=f"RRR document {index}")
        corpus_revision = required_text(
            row,
            "corpus_revision",
            label=f"RRR document {index}",
        )
        if expected_corpus_revision and corpus_revision != expected_corpus_revision:
            raise AlgorithmOutputError("RRR document corpus revision mismatch")
        query_index = nonnegative_integer(
            row.get("query_index"),
            label=f"RRR document {index} query_index",
        )
        if query_index >= query_count:
            raise AlgorithmOutputError("RRR document query_index is out of range")
        rank = nonnegative_integer(row.get("rank"), label=f"RRR document {index} rank")
        if rank < 1 or rank > top_k:
            raise AlgorithmOutputError("RRR document rank is outside the configured top_k")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AlgorithmOutputError("RRR document metadata must be a mapping")
        reserved = {"corpus_revision", "document_id", "query_index", "rank", "title"}
        if reserved & set(metadata):
            raise AlgorithmOutputError("RRR document metadata uses reserved provenance keys")
        score = finite_number(row.get("score", 0.0), label=f"RRR document {index} score")
        blocks.append(
            ContextBlock(
                block_id=document_id,
                text=text,
                source=corpus_revision,
                score=score,
                metadata={
                    **dict(metadata),
                    "document_id": document_id,
                    "title": str(row.get("title", "")),
                    "corpus_revision": corpus_revision,
                    "query_index": query_index,
                    "rank": rank,
                },
            )
        )
    return tuple(blocks)


def _require_source_license_acknowledgement(acknowledged: bool) -> None:
    if not acknowledged:
        raise ExternalDependencyError(
            "RRR upstream source license is NOASSERTION; explicitly acknowledge the "
            "user-mounted source before execution"
        )


def _require_checkpoint_license_acknowledgement(acknowledged: bool) -> None:
    if not acknowledged:
        raise ExternalDependencyError(
            "RRR rewriter/reader checkpoints are NOASSERTION; explicitly acknowledge "
            "the user-mounted assets before execution"
        )


@dataclass(frozen=True, slots=True)
class RRRCommonSupportConfig:
    """Source-locked prompt RRR under the shared arena backend/retriever."""

    rewrite_prompt: str
    reader_prompt: str
    prompt_source_sha256: str
    rewrite_prompt_sha256: str
    reader_prompt_sha256: str
    top_k: int = 5
    max_rewriter_tokens: int = 100
    max_reader_tokens: int = 300
    temperature: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("prompt_source_sha256", self.prompt_source_sha256),
            ("rewrite_prompt_sha256", self.rewrite_prompt_sha256),
            ("reader_prompt_sha256", self.reader_prompt_sha256),
        ):
            if _HEX64.fullmatch(value) is None:
                raise ValidationError(f"RRR common-support {label} must be a SHA-256 digest")
        if "{query}" not in self.rewrite_prompt:
            raise ValidationError("RRR rewrite prompt must contain {query}")
        if not {"{query}", "{background}"} <= {
            token for token in ("{query}", "{background}") if token in self.reader_prompt
        }:
            raise ValidationError("RRR reader prompt must contain {query} and {background}")
        if content_hash(self.rewrite_prompt) != self.rewrite_prompt_sha256:
            raise ValidationError("RRR rewrite prompt hash does not match its content")
        if content_hash(self.reader_prompt) != self.reader_prompt_sha256:
            raise ValidationError("RRR reader prompt hash does not match its content")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.top_k, self.max_rewriter_tokens, self.max_reader_tokens)
        ):
            raise ValidationError("RRR common-support budgets must be positive integers")
        if self.top_k > 100:
            raise ValidationError("RRR common-support top_k must not exceed 100")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValidationError("RRR common-support temperature must be numeric")
        if self.temperature < 0 or self.temperature > 2:
            raise ValidationError("RRR common-support temperature must be in [0, 2]")


@dataclass(slots=True)
class RRRCommonSupport:
    """Controlled source-faithful rewrite→retrieve→snippet→reader execution.

    This profile preserves the upstream prompt rows, single-query parser, stage
    order, and frozen-reader use of the original query. It deliberately binds
    both generations and retrieval to AgentInputLab's common-support
    dependencies; it is not the released warm-up/PPO checkpoint profile.
    """

    config: RRRCommonSupportConfig
    method_id: str = "rrr"
    release_id: str = "rewrite.rrr.common_support_prompt"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("RRR common support requires a non-empty query")
        backend = context.backend
        retriever = context.retriever
        if not isinstance(backend, LLMBackend):
            raise ExternalDependencyError("RRR common support requires an LLMBackend")
        if not isinstance(retriever, Retriever):
            raise ExternalDependencyError("RRR common support requires a Retriever")

        rewrite_prompt = self.config.rewrite_prompt.replace("{query}", state.query)
        rewritten = backend.generate(
            GenerationRequest(
                prompt=rewrite_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_rewriter_tokens,
                seed=context.seed,
                stop=("**",),
                metadata={
                    "method_id": self.method_id,
                    "release_id": self.release_id,
                    "stage": "rewrite",
                },
            )
        )
        query = _parse_common_support_query(rewritten.text)
        results = _validate_common_support_results(
            retriever.retrieve(query, top_k=self.config.top_k),
            top_k=self.config.top_k,
        )
        if not results:
            raise AlgorithmOutputError("RRR common-support retrieval returned no documents")
        evidence = tuple(_common_support_context(item) for item in results)
        snippet = " ".join(item.document.text for item in results)
        reader_prompt = self.config.reader_prompt.replace("{background}", snippet).replace(
            "{query}", state.query
        )
        reader = backend.generate(
            GenerationRequest(
                prompt=reader_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_reader_tokens,
                seed=context.seed + 1,
                stop=("**",),
                metadata={
                    "method_id": self.method_id,
                    "release_id": self.release_id,
                    "stage": "reader",
                },
            )
        )
        if not isinstance(reader.text, str):
            raise AlgorithmOutputError("RRR common-support reader returned non-text output")

        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "source_locked_prompt_profile",
                    {
                        "upstream_commit": RRR_UPSTREAM_COMMIT,
                        "prompt_source_sha256": self.config.prompt_source_sha256,
                        "rewrite_prompt_sha256": self.config.rewrite_prompt_sha256,
                        "reader_prompt_sha256": self.config.reader_prompt_sha256,
                        "profile": "upstream_prompt_pid2_step2_pid1",
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "official_single_query_parser",
                    {
                        "query": query,
                        "parser": "semicolon_then_double_asterisk_single.v1",
                        "raw_rewrite_sha256": content_hash(rewritten.text),
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "real_retrieved_snippets",
                    {
                        "document_ids": tuple(item.document.doc_id for item in results),
                        "ranks": tuple(item.rank for item in results),
                        "reader_uses_original_query": True,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "frozen_reader_output",
                    {
                        "answer": reader.text.strip(),
                        "answer_is_intervention_input": False,
                        "stage_order": ("rewrite", "retrieve", "snippet", "reader"),
                    },
                ),
            ),
            counters={
                "model_calls": 2,
                "retrieval_calls": 1,
                "rewritten_queries": 1,
                "retrieved_documents": len(results),
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_QUERY,
                    query,
                    target="query",
                    rationale="single search query emitted by the source-locked RRR prompt",
                ),
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    evidence,
                    target="retrieved_evidence",
                    rationale="real documents returned by the shared locked retriever",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"controlled_source_locked:xbmxb/RAG-query-rewriting@{RRR_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "controlled_source_locked",
                "upstream_stage_order_preserved": True,
                "upstream_parser_preserved": True,
                "paper_checkpoint_used": False,
                "ppo_training_used": False,
                "reader_answer": reader.text.strip(),
                "backend_fingerprint": dict(backend.fingerprint()),
                "retriever_fingerprint": dict(retriever.fingerprint()),
            },
        )


def _parse_common_support_query(raw: Any) -> str:
    if not isinstance(raw, str):
        raise AlgorithmOutputError("RRR common-support rewriter returned non-text output")
    value = raw.split("**", 1)[0].strip()
    if "Query:" in value:
        value = value.rsplit("Query:", 1)[1].strip()
    queries = tuple(item.strip() for item in value.split(";") if item.strip())
    if len(queries) != 1:
        raise AlgorithmOutputError("RRR common-support parser requires exactly one query")
    return queries[0]


def _validate_common_support_results(raw: Any, *, top_k: int) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(raw)
    except TypeError as exc:
        raise AlgorithmOutputError("RRR retriever returned a non-sequence") from exc
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise AlgorithmOutputError("RRR retriever returned a non-RetrievalResult item")
    if len(results) > top_k:
        raise AlgorithmOutputError("RRR retriever returned more than top_k documents")
    ids = tuple(item.document.doc_id for item in results)
    if len(ids) != len(set(ids)):
        raise AlgorithmOutputError("RRR retriever returned duplicate document IDs")
    if tuple(item.rank for item in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("RRR retriever ranks must be contiguous and one-based")
    return results


def _common_support_context(result: RetrievalResult) -> ContextBlock:
    document = result.document
    if not document.doc_id or not document.corpus_revision:
        raise AlgorithmOutputError("RRR documents require ID and corpus revision provenance")
    reserved = {
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "rank",
        "retrieval_features",
        "title",
    }
    if reserved & set(document.metadata):
        raise AlgorithmOutputError("RRR document metadata uses reserved provenance keys")
    return ContextBlock(
        block_id=document.doc_id,
        text=document.text,
        source=document.corpus_revision,
        score=float(result.score),
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


__all__ = [
    "RRR_PROTOCOL",
    "RRR_UPSTREAM_COMMIT",
    "RRRCommonSupport",
    "RRRCommonSupportConfig",
    "RRRPromptControls",
    "RRRPromptExternal",
]
