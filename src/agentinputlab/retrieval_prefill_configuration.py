"""Explicit text-only context preparation for Plan-and-Budget IR workflows."""

from collections.abc import Mapping
from dataclasses import replace

from agentinputlab.backends.retrieval import RetrievalResult
from agentinputlab.contracts import ContextBlock, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable


def options(value):
    if (not isinstance(value, Mapping)
            or not {"top_k", "max_document_chars"} <= set(value)
            or set(value) - {"top_k", "max_document_chars", "projection"}
            or any(type(value[k]) is not int or value[k] < 1 for k in ("top_k", "max_document_chars"))):
        raise ValidationError("context_prefill requires positive integer top_k and max_document_chars")
    if value.get("projection", "text_only.v1") not in ("text_only.v1", "native_ir_context.v1"):
        raise ValidationError("unknown context_prefill projection")
    return dict(value)


def prepare(state, retriever, settings):
    """Use only the visible query; never replace original passages or read gold."""
    if state.context or state.messages or state.tool_names:
        raise ValidationError("context_prefill requires an empty stateless retrieval context; no replacement")
    native_projection = settings.get("projection") == "native_ir_context.v1"
    if native_projection and (not isinstance(state.metadata.get("benchmark_cell"), str)
                              or not state.metadata["benchmark_cell"].strip()):
        raise ValidationError("native_ir_context requires the dataset's visible benchmark_cell identity")
    results = tuple(retriever.retrieve(state.query, top_k=settings["top_k"]))
    if (not results or len(results) > settings["top_k"]
            or any(not isinstance(r, RetrievalResult) for r in results)):
        raise ValidationError("context_prefill requires a nonempty bounded RetrievalResult sequence")
    ids = [r.document.doc_id for r in results]
    if (len(set(ids)) != len(ids) or any(not i.strip() for i in ids)
            or [r.rank for r in results] != list(range(1, len(results) + 1))):
        raise ValidationError("context_prefill requires unique document IDs and contiguous ranks")
    limit = settings["max_document_chars"]
    if native_projection:
        blocks = tuple(ContextBlock(block_id=r.document.doc_id, text=r.document.text[:limit],
            source=state.metadata["benchmark_cell"], score=r.score,
            metadata={"document_id": r.document.doc_id, "initial_capture_rank": r.rank,
                      "initial_capture_corpus_revision": r.document.corpus_revision}) for r in results)
    else:
        blocks = tuple(ContextBlock(block_id=r.document.doc_id, text=r.document.text[:limit],
            source=r.document.corpus_revision) for r in results)
    if any(not b.text.strip() for b in blocks):
        raise ValidationError("context_prefill received an empty document")
    prepared = replace(state, context=blocks, metadata={**state.metadata, "initial_capture_retrieval_calls": 1}) if native_projection else replace(state, context=blocks)
    return prepared, {"mode": "native_ir_context.v1" if native_projection else "visible_query_text_only_prefill.v1", "configuration": settings,
        "source_state_sha256": state.state_hash, "prepared_state_sha256": prepared.state_hash,
        "retriever": to_jsonable(retriever.fingerprint()), "document_ids": ids,
        "context_sha256": content_hash(blocks),
        "clipped_documents": sum(len(r.document.text) > limit for r in results),
        "projection": ("document_id,text,benchmark_source,score,rank,corpus_revision; no title or arbitrary document metadata"
                       if native_projection else "document_id,text,corpus_revision; no title, score, or metadata injection")}
