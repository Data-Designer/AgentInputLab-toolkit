"""Controlled IRCoT runtime for a shared benchmark corpus.

The upstream Apache-2.0 worker remains the native-path integration.  This
module is the explicitly adapted common-support release used when every arm
must share AgentInputLab's frozen retriever and downstream agent.  It preserves
the upstream controller order and prompt construction while keeping generated
reasoning private: only retrieved corpus documents become visible evidence.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from agentinputlab.backends.llm import GenerationRequest, LLMBackend
from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CapabilityError,
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
from agentinputlab.integrations.external_workers.ircot import IRCOT_UPSTREAM_COMMIT

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ANSWER = re.compile(r".* answer is:? (.*)\.?")
_SENTENCE = re.compile(r"^(.+?(?:[.!?](?=\s|$)|$))", re.DOTALL)
_REASONING_STARTERS = ("thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence")
_ARITHMETIC_REASONING = re.compile(
    r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = "
    r"(\d[\d,]*\.?\d+|\d+)(.*)"
)
_WH_WORDS = frozenset({"who", "what", "when", "where", "why", "which", "how", "does", "is"})


@dataclass(frozen=True, slots=True)
class IRCoTCommonSupportConfig:
    """Frozen settings for the controlled shared-retriever profile."""

    demonstrations: str
    prompt_sha256: str
    max_steps: int = 4
    retrieval_top_k: int = 6
    max_total_documents: int = 15
    max_tokens_per_call: int = 200
    max_document_words: int = 350
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.demonstrations.strip():
            raise ValidationError("IRCoT common-support requires a frozen demonstration prompt")
        if _SHA256.fullmatch(self.prompt_sha256) is None:
            raise ValidationError("IRCoT prompt_sha256 must be lowercase SHA-256")
        actual = hashlib.sha256(self.demonstrations.encode("utf-8")).hexdigest()
        if actual != self.prompt_sha256:
            raise ValidationError("IRCoT demonstration prompt checksum mismatch")
        for name in (
            "max_steps",
            "retrieval_top_k",
            "max_total_documents",
            "max_tokens_per_call",
            "max_document_words",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"IRCoT {name} must be a positive integer")
        if self.retrieval_top_k > self.max_total_documents:
            raise ValidationError("IRCoT retrieval_top_k cannot exceed max_total_documents")
        if self.temperature < 0:
            raise ValidationError("IRCoT temperature must be non-negative")


@dataclass(frozen=True, slots=True)
class IRCoTCommonSupportStep:
    index: int
    generated_sentence: str
    retrieval_query: str | None
    retrieved_document_ids: tuple[str, ...]
    new_document_ids: tuple[str, ...]
    active_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IRCoTCommonSupportResult:
    answer: str
    exit_reason: str
    initial_query: str
    initial_retrieval_query: str
    initial_results: tuple[RetrievalResult, ...]
    steps: tuple[IRCoTCommonSupportStep, ...]
    active_results: tuple[RetrievalResult, ...]
    model_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int


def _remove_wh_words(text: str) -> str:
    """Match upstream ``remove_wh_words`` without importing legacy code."""

    return " ".join(word for word in text.split(" ") if word.strip().lower() not in _WH_WORDS)


def _is_reasoning_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return lowered.startswith(_REASONING_STARTERS) or bool(_ARITHMETIC_REASONING.match(sentence))


def _first_sentence(text: str) -> str:
    """Controlled substitute for upstream spaCy ``doc.sents[0]``.

    The substitution is intentionally small and declared in patch metadata so
    it cannot be mistaken for native component parity.
    """

    cleaned = text.strip()
    match = _SENTENCE.match(cleaned)
    return match.group(1).strip() if match else ""


def _answer(sentence: str) -> str:
    match = _ANSWER.match(sentence)
    if match is None:
        return ""
    answer = match.group(1).strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return answer.strip()


def _validated_results(
    values: Sequence[RetrievalResult], *, top_k: int
) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(values)
    except TypeError as exc:
        raise AlgorithmOutputError("IRCoT retriever returned a non-sequence") from exc
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise AlgorithmOutputError("IRCoT retriever returned an invalid result")
    if len(results) > top_k:
        raise AlgorithmOutputError("IRCoT retriever returned more than top_k documents")
    ids = tuple(item.document.doc_id for item in results)
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise AlgorithmOutputError("IRCoT retrieval document IDs must be non-empty and unique")
    if tuple(item.rank for item in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("IRCoT retrieval ranks must be contiguous and one-based")
    return results


def _paragraph(result: RetrievalResult, max_words: int) -> str:
    document = result.document
    paragraph = " ".join(document.text.split(" ")[:max_words])
    if paragraph.strip().startswith("Wikipedia Title: "):
        return paragraph.strip()
    return f"Wikipedia Title: {document.title}\n{paragraph.strip()}"


@dataclass(slots=True)
class IRCoTCommonSupport:
    """IRCoT controller over injected, checksum-locked common dependencies."""

    config: IRCoTCommonSupportConfig
    method_id: str = "ircot"
    release_id: str = "acquire.ircot.common_support"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    @staticmethod
    def _backend(value: object) -> LLMBackend:
        if value is None:
            raise ExternalDependencyError("IRCoT common-support requires an LLMBackend")
        if not hasattr(value, "generate") or not hasattr(value, "fingerprint"):
            raise CapabilityError("IRCoT backend does not implement LLMBackend")
        return value  # type: ignore[return-value]

    @staticmethod
    def _retriever(value: object) -> Retriever:
        if value is None:
            raise ExternalDependencyError("IRCoT common-support requires a Retriever")
        if not hasattr(value, "retrieve") or not hasattr(value, "fingerprint"):
            raise CapabilityError("IRCoT retriever does not implement Retriever")
        return value  # type: ignore[return-value]

    def run(
        self, state: FrozenAgentState, context: InterventionContext
    ) -> IRCoTCommonSupportResult:
        if not state.query.strip():
            raise ValidationError("IRCoT requires a non-empty question")
        backend = self._backend(context.backend)
        retriever = self._retriever(context.retriever)
        model_calls = retrieval_calls = input_tokens = output_tokens = 0
        active: dict[str, RetrievalResult] = {}
        generated: list[str] = []
        steps: list[IRCoTCommonSupportStep] = []

        def retrieve(query: str) -> tuple[RetrievalResult, ...]:
            nonlocal retrieval_calls
            if retrieval_calls >= state.budget.max_retrieval_calls:
                raise BudgetExceededError("IRCoT exhausted max_retrieval_calls")
            normalized = _remove_wh_words(query)
            results = _validated_results(
                retriever.retrieve(normalized, top_k=self.config.retrieval_top_k),
                top_k=self.config.retrieval_top_k,
            )
            retrieval_calls += 1
            return results

        initial_retrieval_query = _remove_wh_words(state.query)
        initial = retrieve(state.query)
        if not initial:
            raise AlgorithmOutputError("IRCoT initial retrieval returned no documents")
        for result in initial:
            if len(active) >= self.config.max_total_documents:
                break
            active.setdefault(result.document.doc_id, result)

        answer = ""
        exit_reason = "max_steps"
        for index in range(self.config.max_steps):
            if model_calls >= state.budget.max_model_calls:
                raise BudgetExceededError("IRCoT exhausted max_model_calls")
            remaining_output = state.budget.max_output_tokens - output_tokens
            if remaining_output < 1:
                raise BudgetExceededError("IRCoT exhausted max_output_tokens")
            paragraphs = "\n\n".join(
                _paragraph(result, self.config.max_document_words) for result in active.values()
            )
            reasoning = " ".join(generated)
            question = "Answer the following question by reasoning step-by-step.\n" + state.query
            test_example = f"{paragraphs}\n\nQ: {question}\nA: {reasoning}".strip()
            prompt = f"{self.config.demonstrations.rstrip()}\n\n\n{test_example}"
            response = backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=min(self.config.max_tokens_per_call, remaining_output),
                    seed=context.seed + model_calls,
                    metadata={
                        "method_id": self.method_id,
                        "release_id": self.release_id,
                        "step_index": index,
                    },
                )
            )
            model_calls += 1
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            if output_tokens > state.budget.max_output_tokens:
                raise BudgetExceededError("IRCoT exceeded max_output_tokens")
            if (
                state.budget.max_context_tokens is not None
                and input_tokens > state.budget.max_context_tokens
            ):
                raise BudgetExceededError("IRCoT exceeded max_context_tokens")
            sentence = _first_sentence(response.text)
            if not sentence:
                raise AlgorithmOutputError("IRCoT generator returned no sentence")
            generated.append(sentence)
            answer = _answer(sentence)
            if answer:
                exit_reason = "answer_detected"
                steps.append(
                    IRCoTCommonSupportStep(
                        index=index,
                        generated_sentence=sentence,
                        retrieval_query=None,
                        retrieved_document_ids=(),
                        new_document_ids=(),
                        active_document_ids=tuple(active),
                    )
                )
                break
            if index + 1 == self.config.max_steps:
                answer = " ".join(generated)
                steps.append(
                    IRCoTCommonSupportStep(
                        index=index,
                        generated_sentence=sentence,
                        retrieval_query=None,
                        retrieved_document_ids=(),
                        new_document_ids=(),
                        active_document_ids=tuple(active),
                    )
                )
                break

            factual = [item for item in generated if not _is_reasoning_sentence(item)]
            retrieval_query = factual[-1].strip() if factual else state.query
            results = retrieve(retrieval_query)
            retrieved_ids = tuple(item.document.doc_id for item in results)
            new_ids: list[str] = []
            for result in results:
                document_id = result.document.doc_id
                if document_id in active or len(active) >= self.config.max_total_documents:
                    continue
                active[document_id] = result
                new_ids.append(document_id)
            steps.append(
                IRCoTCommonSupportStep(
                    index=index,
                    generated_sentence=sentence,
                    retrieval_query=_remove_wh_words(retrieval_query),
                    retrieved_document_ids=retrieved_ids,
                    new_document_ids=tuple(new_ids),
                    active_document_ids=tuple(active),
                )
            )

        if not active:
            raise AlgorithmOutputError("IRCoT returned no active corpus evidence")
        return IRCoTCommonSupportResult(
            answer=answer,
            exit_reason=exit_reason,
            initial_query=state.query,
            initial_retrieval_query=initial_retrieval_query,
            initial_results=initial,
            steps=tuple(steps),
            active_results=tuple(active.values()),
            model_calls=model_calls,
            retrieval_calls=retrieval_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        backend = self._backend(context.backend)
        retriever = self._retriever(context.retriever)
        result = self.run(state, context)
        events = [
            TraceEvent(
                TracePhase.PREPARE,
                "ircot_common_support_configuration",
                {
                    "upstream_commit": IRCOT_UPSTREAM_COMMIT,
                    "prompt_sha256": self.config.prompt_sha256,
                    "max_steps": self.config.max_steps,
                    "retrieval_top_k": self.config.retrieval_top_k,
                    "max_total_documents": self.config.max_total_documents,
                    "sentence_segmenter": "terminal_punctuation_first_sentence.v1",
                    "backend": dict(backend.fingerprint()),
                    "retriever": dict(retriever.fingerprint()),
                },
            ),
            TraceEvent(
                TracePhase.OBSERVE,
                "initial_retrieval",
                {
                    "query": result.initial_query,
                    "normalized_query": result.initial_retrieval_query,
                    "document_ids": tuple(item.document.doc_id for item in result.initial_results),
                },
            ),
        ]
        events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "ircot_step",
                {
                    "index": step.index,
                    "generated_sentence": step.generated_sentence,
                    "retrieval_query": step.retrieval_query,
                    "retrieved_document_ids": step.retrieved_document_ids,
                    "new_document_ids": step.new_document_ids,
                    "active_document_ids": step.active_document_ids,
                },
            )
            for step in result.steps
        )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "ircot_exit",
                {
                    "exit_reason": result.exit_reason,
                    "answer": result.answer,
                    "active_document_ids": tuple(
                        item.document.doc_id for item in result.active_results
                    ),
                    "answer_visible_to_downstream": False,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(events),
            counters={
                "model_calls": result.model_calls,
                "retrieval_calls": result.retrieval_calls,
                "reasoning_steps": len(result.steps),
                "active_documents": len(result.active_results),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        evidence = tuple(_context_from_result(item) for item in result.active_results)
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    evidence,
                    target="retrieved_evidence",
                    rationale="real corpus evidence selected by the controlled IRCoT loop",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"controlled_minimal_engineering:StonyBrookNLP/ircot@{IRCOT_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "official_controller_minimal_engineering",
                "upstream_commit": IRCOT_UPSTREAM_COMMIT,
                "prompt_sha256": self.config.prompt_sha256,
                "generated_reasoning_is_evidence": False,
                "generated_answer_visible_to_downstream": False,
                "query_normalization": "upstream_remove_wh_words",
                "retrieval_query_policy": "question_or_last_non_reasoning_sentence",
                "sentence_segmenter_substitution": "terminal_punctuation_first_sentence.v1",
            },
        )


def _context_from_result(result: RetrievalResult) -> ContextBlock:
    document = result.document
    if not document.corpus_revision:
        raise AlgorithmOutputError("IRCoT evidence requires corpus revision provenance")
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
            "IRCoT document metadata uses reserved provenance keys: " + ", ".join(sorted(overlap))
        )
    return ContextBlock(
        block_id=document.doc_id,
        text=document.text,
        source=document.corpus_revision,
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


__all__ = [
    "IRCoTCommonSupport",
    "IRCoTCommonSupportConfig",
    "IRCoTCommonSupportResult",
    "IRCoTCommonSupportStep",
]
