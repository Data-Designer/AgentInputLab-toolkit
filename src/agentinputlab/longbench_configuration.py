"""File binding to LongBench's source-locked templates and native text scorers."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.benchmarks.execution import DownstreamPrediction
from agentinputlab.configuration import _known, _object, _unique
from agentinputlab.contracts.errors import CapabilityError, ValidationError

SOURCE_KEYS = {"data_zip", "scorer_path", "prompts_path"}
LONG_TEXT_REWRITE_RELEASES = frozenset({"rewrite.query2doc.querygym_live", "rewrite.reformer.repo_native_r_only"})
LONG_TEXT_REWRITE_CELLS = frozenset({"longbench.multifieldqa_en", "longbench.qmsum"})


def _visible_ranked_chunks(original, query, *, retriever, top_k):
    from agentinputlab.contracts import ContextBlock

    if retriever is None:
        raise ValidationError("LongBench projection requires a visible retriever")
    blocks = {b.block_id: b for b in original.context}
    results = tuple(retriever.retrieve(query, top_k=top_k))
    if not results or len(results) > top_k:
        raise ValidationError("rewrite retrieval returned no valid visible chunks")
    projected = []
    for rank, result in enumerate(results, start=1):
        document = result.document
        source = document.metadata.get("visible_chunk_source")
        if not isinstance(source, Mapping) or source.get("block_id") not in blocks:
            raise ValidationError("rewritten query returned a document outside the original case")
        text = blocks[source["block_id"]].text
        start, end = source.get("char_start"), source.get("char_end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or document.text != text[start:end] or result.rank != rank
                or source.get("block_sha256") != hashlib.sha256(text.encode()).hexdigest()):
            raise ValidationError("rewritten query returned an invalid chunk/source binding")
        projected.append(ContextBlock(text=document.text, block_id=document.doc_id,
                                      source=document.corpus_revision, metadata=document.metadata))
    if len({b.block_id for b in projected}) != len(projected):
        raise ValidationError("rewritten query returned duplicate visible chunks")
    return tuple(projected)


def prepare_reformer_documents(original, retriever, *, top_k):
    """Bind the pattern selector to actual ranked passages from the original case."""
    projected = _visible_ranked_chunks(original, original.query, retriever=retriever, top_k=top_k)
    prepared = replace(original, context=projected)
    return prepared, {"kind": "original_query_visible_chunks_in_rank_order.v1",
        "selected_ids": [b.block_id for b in projected], "source_state_sha256": original.state_hash,
        "prepared_state_sha256": prepared.state_hash, "added_model_calls": 0, "added_retrieval_calls": 1}


def project_native_question(original, transformed, *, retriever, top_k, prefilled=None):
    """Use only actual original-document chunks, then restore the scored question."""
    from agentinputlab.contracts.serialization import content_hash

    audit = {"native_question_sha256": content_hash(original.query),
             "rewrite_sha256": content_hash(transformed.query),
             "added_model_calls": 0, "added_retrieval_calls": 0}
    expected = prefilled if prefilled is not None else original
    if prefilled is not None and (top_k is None or transformed.context != prefilled.context):
        raise ValidationError("prefilled rewriter unexpectedly changed evidence")
    if transformed.query == original.query:
        retained = replace(transformed, context=original.context) if prefilled is not None else transformed
        return retained, {**audit, "projection": "unchanged_query"}
    if top_k is None:
        return replace(transformed, query=original.query), {**audit, "projection": "preserve_native_question"}
    if transformed.context != expected.context:
        raise ValidationError("rewrite retrieval requires unchanged case context")
    projected = _visible_ranked_chunks(original, transformed.query, retriever=retriever, top_k=top_k)
    block_order = {block.block_id: i for i, block in enumerate(original.context)}
    ordered = tuple(sorted(projected, key=lambda b: (
        block_order[b.metadata["visible_chunk_source"]["block_id"]],
        b.metadata["visible_chunk_source"]["char_start"],
    )))
    return replace(transformed, query=original.query, context=ordered), {
        **audit, "projection": "rewrite_retrieves_original_visible_chunks.v1",
        "ranked_ids": [b.block_id for b in projected],
        "rendered_ids": [b.block_id for b in ordered], "added_retrieval_calls": 1,
    }


def prepare_qrag_context(state, *, adapter, observe_retriever):
    """Explicit lossless-350-word/BM25-top6 input preparation, never use gold."""
    from agentinputlab.backends import Document, InMemoryBM25Retriever
    from agentinputlab.contracts import ContextBlock
    from agentinputlab.contracts.serialization import content_hash

    documents = []
    for index, block in enumerate(state.context):
        tokens = list(re.finditer(r"\S+", block.text))
        if not tokens:
            raise ValidationError("LongBench candidate source has an empty visible block")
        for offset in range(0, len(tokens), 350):
            start = 0 if offset == 0 else tokens[offset].start()
            stop = len(block.text) if offset + 350 >= len(tokens) else tokens[offset + 350].start()
            documents.append(Document(doc_id=f"{block.block_id or index}:words350:{start}:{stop}",
                text=block.text[start:stop],
                corpus_revision=block.source or content_hash({"original_visible_state": state.state_hash}),
                metadata={"source_visible_block_index": index,
                    "source_visible_block_sha256": hashlib.sha256(block.text.encode()).hexdigest(),
                    "source_char_start": start, "source_char_end": stop,
                    "corpus_adapter": "lossless_visible_word_chunks350_v1"}))
    if len({d.doc_id for d in documents}) != len(documents):
        raise ValidationError("LongBench candidate document IDs must be unique")
    if len(documents) < 2:
        raise ValidationError("Q-RAG requires at least two real visible candidate chunks; no padding")
    selected = tuple(observe_retriever(InMemoryBM25Retriever(tuple(documents))).retrieve(state.query, top_k=6))
    marker = "[AGENTINPUTLAB_SELECTED_VISIBLE_PASSAGES]"
    if marker in state.query or marker in state.instruction or any(marker in b.text for b in state.context):
        raise ValidationError("LongBench visible input contains the reserved projection marker")
    instruction = adapter.template.format(context=marker, input=state.query)
    if state.instruction:
        instruction += "\n\n" + state.instruction
    if instruction.count(marker) != 1:
        raise ValidationError("LongBench template must contain exactly one context marker")
    context = tuple(ContextBlock(block_id=r.document.doc_id, text=r.document.text,
        source=r.document.corpus_revision, metadata=r.document.metadata) for r in selected)
    prepared = replace(state, instruction=instruction, context=context)
    return prepared, {"kind": "lossless_words350_bm25_top6.v1",
        "source_state_sha256": state.state_hash, "prepared_state_sha256": prepared.state_hash,
        "visible_chunk_count": len(documents), "selected_ids": [b.block_id for b in context],
        "native_template_sha256": content_hash(adapter.template),
        "corpus_lossless": True, "selection_is_full_context": len(selected) == len(documents),
        "preparation_retrieval_calls": 1, "new_model_calls": 0,
        "model_and_output_token_budget_unchanged": True}


def read_sources(path):
    path = Path(path).expanduser().absolute()
    try:
        with path.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValidationError("LongBench source configuration exceeds 1 MiB")
        data = _object(json.loads(raw, object_pairs_hook=_unique), "longbench_sources")
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValidationError("cannot read LongBench source configuration: use a UTF-8 JSON object") from None
    data = sources(data)
    return {key: str(path.parent / value) if not Path(value).is_absolute() else value
            for key, value in data.items()}


def sources(value):
    data = _object(value, "longbench_sources")
    _known(data, SOURCE_KEYS, "longbench_sources")
    if set(data) != SOURCE_KEYS or any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValidationError("longbench_sources requires data_zip, scorer_path and prompts_path")
    return {key: str(Path(value).expanduser()) for key, value in data.items()}


def options(value, *, split, backend, prediction_required=True):
    if type(prediction_required) is not bool:
        raise ValidationError("prediction_required must be boolean")
    if split != "test":
        raise ValidationError("LongBench v1 exposes only the test split")
    data = _object(value, "LongBench predictor")
    _known(data, {"schema_version", "native_text"}, "LongBench predictor")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValidationError("LongBench predictor schema_version must be 1")
    settings = _object(data.get("native_text", {}), "native_text")
    _known(settings, {"max_output_tokens", "temperature", "composite_output_format",
                      "allow_json_fence", "preserve_original_question", "rewrite_retrieval_top_k",
                      "reformer_context_top_k"}, "native_text")
    output_format = settings.get("composite_output_format", "text")
    if output_format not in ("text", "structured_qa_json"):
        raise ValidationError("LongBench composite_output_format must be text or structured_qa_json")
    allow_fence = settings.get("allow_json_fence", False)
    if type(allow_fence) is not bool:
        raise ValidationError("LongBench allow_json_fence must be boolean")
    if allow_fence and output_format != "structured_qa_json":
        raise ValidationError("LongBench allow_json_fence requires structured_qa_json")
    preserve = settings.get("preserve_original_question", False)
    if type(preserve) is not bool or (preserve and not prediction_required):
        raise ValidationError("preserve_original_question requires the native downstream reader")
    top_k = settings.get("rewrite_retrieval_top_k")
    if "rewrite_retrieval_top_k" in settings and (
            not preserve or type(top_k) is not int or not 1 <= top_k <= 100):
        raise ValidationError("rewrite_retrieval_top_k requires preserve_original_question and an integer in [1,100]")
    reformer_top_k = settings.get("reformer_context_top_k")
    if "reformer_context_top_k" in settings and (
            top_k is None or type(reformer_top_k) is not int or reformer_top_k < 1):
        raise ValidationError("reformer_context_top_k requires rewrite retrieval and a positive integer")
    request = GenerationRequest(prompt="validation", max_tokens=settings.get("max_output_tokens", 512),
        temperature=settings.get("temperature", 0.0))
    if backend is None and prediction_required:
        raise CapabilityError("LongBench evaluation requires a generation backend")
    result = {"max_output_tokens": request.max_tokens, "temperature": request.temperature}
    if output_format != "text":
        result.update(composite_output_format=output_format, allow_json_fence=allow_fence)
    if preserve:
        result.update(preserve_original_question=True, rewrite_retrieval_top_k=top_k,
                      reformer_context_top_k=reformer_top_k)
    return result


def composite_prediction(text, *, state, benchmark_id, settings):
    """Decode an explicitly declared answer envelope, without another reader."""
    metadata = {"composite_answer_no_extra_call": True,
                "prediction_kind": "native_longbench_text", "native_template_used": False}
    if settings.get("composite_output_format", "text") == "text":
        return text, metadata
    from agentinputlab.experiments.production_builders import _parse_visible_qa_prediction

    prediction, audit = _parse_visible_qa_prediction(
        text, benchmark_id=benchmark_id, state=state, output_format="structured_json",
        identifier_policy="native_sets.v1", allow_json_fence=settings["allow_json_fence"],
        preserve_answer_text=True)
    return prediction.answer, {**metadata, "composite_output_format": "structured_qa_json",
        "identifier_audit": audit, "scoring_fields": ["answer"],
        "support_and_evidence_scored": False}


def predict(state, *, adapter, backend, settings, seed):
    if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
        raise CapabilityError("LongBench generation has no remaining model budget")
    request = GenerationRequest(prompt=adapter.render(state),
        max_tokens=min(settings["max_output_tokens"], state.budget.max_output_tokens),
        temperature=settings["temperature"], seed=seed)
    response = backend.generate(request)
    if response.usage.output_tokens > request.max_tokens:
        raise ValidationError("LongBench provider exceeded the per-call output cap")
    return DownstreamPrediction(prediction=response.text, metadata={
        "prediction_kind": "native_longbench_text", "native_template_used": True,
        "official_aggregate_leaderboard_claimed": False})
