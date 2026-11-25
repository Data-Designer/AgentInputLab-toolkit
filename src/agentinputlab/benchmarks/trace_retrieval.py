"""Project only observed corpus document IDs from supported retrieval traces."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.contracts import ExecutionTrace, TracePhase, ValidationError

from .retrieval import RetrievalPrediction

_FINAL_RANK = {
    "plan.decomp.common_support_odqa": ("decomp_common_support_result", "retrieved_document_ids"),
    # The no-retrieval route legitimately emits an empty document sequence.
    "acquire.adaptive_rag.common_support": ("adaptive_rag_selected_arm_exit", "document_ids"),
    "acquire.ircot.common_support": ("ircot_exit", "active_document_ids"),
    "acquire.qrag.common_support": ("qrag_selected_real_context", "selected_document_ids"),
}
_SELF_ASK = "plan.self_ask.full_loop"
_RRR = "rewrite.rrr.common_support_prompt"
TRACE_RANK_RELEASES = frozenset((*_FINAL_RANK, _SELF_ASK, _RRR))
TRACE_RANK_BENCHMARKS = frozenset({"bright.earth_science", "beir.fiqa_2018"})


def validate_trace_ranking(options, parameters, *, benchmark_id, release_id, has_retriever):
    """Validate the opt-in protocol before the evaluator or workflow dispatches."""
    if (options != {"protocol": "observed_rank.v1"} or release_id not in TRACE_RANK_RELEASES
            or benchmark_id not in TRACE_RANK_BENCHMARKS or not has_retriever):
        raise ValidationError("trace_ranking requires a supported static retrieval release and corpus retriever")
    if set(parameters) - {"schema_version", "retriever_capability"}:
        raise ValidationError("trace_ranking preserves the method's native ranking; do not configure another prediction or preparation mode")
    if "retriever_capability" in parameters and (
            release_id != _SELF_ASK or parameters["retriever_capability"] != {"visible_name": "search"}):
        raise ValidationError("trace_ranking permits retriever_capability=search only for Self-Ask")


@dataclass(slots=True)
class CaseRetrievalEvidence:
    """Capture actual returns without changing ranks, calls, or the index identity."""

    inner: Retriever
    calls: list[tuple[str, ...]] = field(default_factory=list, init=False)

    @property
    def retriever_id(self) -> str:
        return self.inner.retriever_id

    def fingerprint(self):
        return self.inner.fingerprint()

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        results = self.inner.retrieve(query, top_k=top_k)
        self.calls.append(tuple(item.document.doc_id for item in results))
        return results

    @property
    def observed_ids(self) -> frozenset[str]:
        return frozenset(doc_id for call in self.calls for doc_id in call)


def _ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError("retrieval trace requires an ordered document-ID sequence")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValidationError("retrieval trace contains an invalid document ID")
    if len(set(value)) != len(value):
        raise ValidationError("a ranked retrieval step contains duplicate IDs")
    return tuple(value)


def prediction_from_trace(trace: ExecutionTrace, *, observed_document_ids: Collection[str]) -> RetrievalPrediction:
    """Retain native order, or first occurrence across Self-Ask's ordered search steps.

    This projection cannot retrieve documents or add a model reader. Empty
    completed ranks remain empty; unverified IDs are rejected, not backfilled.
    """
    release = trace.release_id
    if release in _FINAL_RANK:
        kind, field = _FINAL_RANK[release]
        matches = [event for event in trace.events if event.name == kind and event.phase is TracePhase.FINALIZE]
        if len(matches) != 1:
            raise ValidationError("missing or ambiguous final ranking event")
        ranked = _ids(matches[0].payload.get(field))
    elif release == _RRR:
        matches = [e for e in trace.events if e.name == "real_retrieved_snippets" and e.phase is TracePhase.OBSERVE]
        readers = [e for e in trace.events if e.name == "frozen_reader_output" and e.phase is TracePhase.FINALIZE]
        if len(matches) != 1 or len(readers) != 1:
            raise ValidationError("missing or ambiguous RRR search or native reader event")
        ranked = _ids(matches[0].payload.get("document_ids"))
        if matches[0].payload.get("ranks") != tuple(range(1, len(ranked) + 1)):
            raise ValidationError("RRR trace does not preserve native retrieval ranks")
    elif release == _SELF_ASK:
        if sum(event.name == "self_ask_final" and event.phase is TracePhase.FINALIZE for event in trace.events) != 1:
            raise ValidationError("missing or ambiguous Self-Ask final event")
        ordered = {}
        for event in trace.events:
            if event.name != "self_ask_step" or event.phase is not TracePhase.OBSERVE:
                continue
            ids = _ids(event.payload.get("document_ids"))
            if event.payload.get("answer_source") == "retrieval":
                if not ids:
                    raise ValidationError("Self-Ask search evidence contains no retrieved ID")
                ordered.update(dict.fromkeys(ids))
            elif ids:
                raise ValidationError("non-retrieval Self-Ask step cannot supply evidence IDs")
        ranked = tuple(ordered)
    else:
        raise ValidationError("release has no verified trace-to-ranking projection")
    if not set(ranked).issubset(observed_document_ids):
        raise ValidationError("ranking contains a document absent from the actual retriever output")
    return RetrievalPrediction.from_ranked_ids(ranked)
