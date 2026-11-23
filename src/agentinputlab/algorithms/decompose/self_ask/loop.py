"""Stateful Self-Ask loop with injected language-model and search backends."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from agentinputlab.backends.llm.base import GenerationRequest, LLMBackend
from agentinputlab.backends.retrieval.base import RetrievalResult, Retriever
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
from agentinputlab.contracts.serialization import content_hash

SELF_ASK_UPSTREAM_REVISION = "559e56ec1ba8dc93cf63c4148ec1f4717caf4b60"


@dataclass(frozen=True, slots=True)
class SelfAskMarkers:
    followup_needed: str = "Are follow up questions needed here:"
    followup: str = "Follow up:"
    intermediate_answer: str = "Intermediate answer:"
    final_answer: str = "So the final answer is:"


DEFAULT_SELF_ASK_MARKERS = SelfAskMarkers()


@dataclass(frozen=True, slots=True)
class SelfAskConfig:
    demonstrations: str
    max_followups: int = 5
    retrieval_top_k: int = 1
    allow_model_search_fallback: bool = True
    max_tokens_per_call: int = 256
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.demonstrations.strip():
            raise ValidationError(
                "Self-Ask full_loop requires an explicit frozen demonstration artifact"
            )
        if self.max_followups < 0:
            raise ValidationError("max_followups must be non-negative")
        if self.retrieval_top_k < 1:
            raise ValidationError("retrieval_top_k must be positive")
        if self.max_tokens_per_call < 1:
            raise ValidationError("max_tokens_per_call must be positive")


@dataclass(frozen=True, slots=True)
class SelfAskStep:
    index: int
    followup_question: str
    intermediate_answer: str
    answer_source: str
    document_ids: tuple[str, ...] = ()
    retrieval_records: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class SelfAskResult:
    transcript: str
    final_answer: str
    steps: tuple[SelfAskStep, ...]
    model_calls: int
    retrieval_calls: int
    model_fallbacks: int
    termination: str = "final_answer_marker"


def _last_marker_value(text: str, marker: str) -> str | None:
    pattern = re.compile(re.escape(marker) + r"\s*(?P<value>[^\n\r]+)")
    matches = tuple(pattern.finditer(text))
    if not matches:
        return None
    value = matches[-1].group("value").strip()
    return value or None


def parse_followup_question(
    text: str, markers: SelfAskMarkers = DEFAULT_SELF_ASK_MARKERS
) -> str | None:
    return _last_marker_value(text, markers.followup)


def parse_final_answer(text: str, markers: SelfAskMarkers = DEFAULT_SELF_ASK_MARKERS) -> str | None:
    return _last_marker_value(text, markers.final_answer)


def render_initial_self_ask_prompt(
    query: str, config: SelfAskConfig, markers: SelfAskMarkers = DEFAULT_SELF_ASK_MARKERS,
) -> str:
    """Serialize the first stage from frozen demonstrations and the current query."""
    return (
        f"{config.demonstrations.rstrip()}\n\n"
        f"Question: {query}\n{markers.followup_needed}"
    )


class SelfAskLoop:
    method_id = "self_ask"
    release_id = "plan.self_ask.full_loop"
    family = SemanticFamily.DECOMPOSE

    def __init__(
        self,
        config: SelfAskConfig,
        *,
        markers: SelfAskMarkers = DEFAULT_SELF_ASK_MARKERS,
    ) -> None:
        self.config = config
        self.markers = markers

    @staticmethod
    def _require_backend(value: object) -> LLMBackend:
        if value is None:
            raise ExternalDependencyError("Self-Ask requires an injected LLMBackend")
        if not hasattr(value, "generate") or not hasattr(value, "fingerprint"):
            raise CapabilityError("Self-Ask backend does not implement LLMBackend")
        return value  # type: ignore[return-value]

    @staticmethod
    def _require_retriever(value: object) -> Retriever:
        if value is None:
            raise ExternalDependencyError(
                "Self-Ask full_loop requires a frozen Retriever/search backend"
            )
        if not hasattr(value, "retrieve") or not hasattr(value, "fingerprint"):
            raise CapabilityError("Self-Ask retriever does not implement Retriever")
        return value  # type: ignore[return-value]

    def run(self, state: FrozenAgentState, context: InterventionContext) -> SelfAskResult:
        backend = self._require_backend(context.backend)
        retriever = self._require_retriever(context.retriever)
        transcript = render_initial_self_ask_prompt(state.query, self.config, self.markers)
        model_calls = 0
        retrieval_calls = 0
        model_fallbacks = 0
        steps: list[SelfAskStep] = []
        termination = "final_answer_marker"

        def generate(prompt: str, *, stop: tuple[str, ...] = ()) -> str:
            nonlocal model_calls
            if model_calls >= state.budget.max_model_calls:
                raise BudgetExceededError("Self-Ask exhausted max_model_calls")
            if state.budget.max_output_tokens < 1:
                raise BudgetExceededError("Self-Ask has no output-token budget")
            response = backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=min(
                        self.config.max_tokens_per_call,
                        state.budget.max_output_tokens,
                    ),
                    seed=context.seed + model_calls,
                    stop=stop,
                    metadata={
                        "method_id": self.method_id,
                        "release_id": self.release_id,
                    },
                )
            )
            model_calls += 1
            if not isinstance(response.text, str) or not response.text.strip():
                raise AlgorithmOutputError("Self-Ask backend returned an empty response")
            return response.text.strip()

        continuation = generate(
            transcript,
            stop=(f"\n{self.markers.intermediate_answer}",),
        )
        transcript = f"{transcript} {continuation}"

        while True:
            # Parse only the current continuation.  The frozen demonstrations
            # intentionally contain final-answer markers of their own.
            final_answer = parse_final_answer(continuation, self.markers)
            if final_answer is not None:
                break
            followup = parse_followup_question(continuation, self.markers)
            if followup is None:
                prompt = f"{transcript}\n{self.markers.final_answer}"
                final_completion = generate(prompt)
                transcript = f"{prompt} {final_completion}"
                final_answer = parse_final_answer(final_completion, self.markers)
                if final_answer is None:
                    final_answer = final_completion.splitlines()[0].strip()
                    termination = "first_line_fallback"
                if not final_answer:
                    raise AlgorithmOutputError("Self-Ask could not parse a final answer")
                break
            if len(steps) >= self.config.max_followups:
                raise BudgetExceededError("Self-Ask reached the registered follow-up safety cap")
            if retrieval_calls >= state.budget.max_retrieval_calls:
                raise BudgetExceededError("Self-Ask exhausted max_retrieval_calls")
            results: tuple[RetrievalResult, ...] = tuple(
                retriever.retrieve(followup, top_k=self.config.retrieval_top_k)
            )
            retrieval_calls += 1
            if results:
                intermediate = results[0].document.text.strip()
                source = "retrieval"
                document_ids = tuple(item.document.doc_id for item in results)
                retrieval_records = tuple(
                    {
                        "document_id": item.document.doc_id,
                        "corpus_revision": item.document.corpus_revision,
                        "rank": item.rank,
                        "score": item.score,
                    }
                    for item in results
                )
            elif self.config.allow_model_search_fallback:
                fallback_prompt = f"{transcript}\n{self.markers.intermediate_answer}"
                intermediate = generate(fallback_prompt, stop=("\n",))
                model_fallbacks += 1
                source = "model_fallback"
                document_ids = ()
                retrieval_records = ()
            else:
                raise AlgorithmOutputError(f"Self-Ask search returned no result for {followup!r}")
            step = SelfAskStep(
                index=len(steps),
                followup_question=followup,
                intermediate_answer=intermediate,
                answer_source=source,
                document_ids=document_ids,
                retrieval_records=retrieval_records,
            )
            steps.append(step)
            transcript = f"{transcript}\n{self.markers.intermediate_answer} {intermediate}"
            continuation = generate(
                transcript,
                stop=(f"\n{self.markers.intermediate_answer}",),
            )
            transcript = f"{transcript}\n{continuation}"

        return SelfAskResult(
            transcript=transcript,
            final_answer=final_answer,
            steps=tuple(steps),
            model_calls=model_calls,
            retrieval_calls=retrieval_calls,
            model_fallbacks=model_fallbacks,
            termination=termination,
        )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        result = self.run(state, context)
        events = [
            TraceEvent(
                TracePhase.PREPARE,
                "self_ask_configuration",
                {
                    "demonstration_hash": content_hash(self.config.demonstrations),
                    "marker_protocol": {
                        "followup_needed": self.markers.followup_needed,
                        "followup": self.markers.followup,
                        "intermediate_answer": self.markers.intermediate_answer,
                        "final_answer": self.markers.final_answer,
                    },
                    "max_followups": self.config.max_followups,
                    "retriever": dict(context.retriever.fingerprint()),
                    "backend": dict(context.backend.fingerprint()),
                },
            )
        ]
        for step in result.steps:
            events.append(
                TraceEvent(
                    TracePhase.OBSERVE,
                    "self_ask_step",
                    {
                        "index": step.index,
                        "followup_question": step.followup_question,
                        "intermediate_answer": step.intermediate_answer,
                        "answer_source": step.answer_source,
                        "document_ids": step.document_ids,
                        "retrieval_records": step.retrieval_records,
                    },
                )
            )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "self_ask_final",
                {
                    "final_answer": result.final_answer,
                    "transcript": result.transcript,
                    "termination": result.termination,
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
                "followups": len(result.steps),
                "model_search_fallbacks": result.model_fallbacks,
            },
        )
        context_block = ContextBlock(
            text=result.transcript,
            block_id=f"{state.task_id or state.state_hash}:self_ask",
            source="self_ask_intermediate_trace",
            metadata={
                "method_id": self.method_id,
                "release_id": self.release_id,
                "final_answer": result.final_answer,
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    value=context_block,
                    target="context",
                    rationale="materialize Self-Ask intermediate reasoning/search trajectory",
                ),
                PatchOperation(
                    PatchOp.SET_METADATA,
                    value={
                        "self_ask_final_answer": result.final_answer,
                        "self_ask_trace_hash": trace.trace_hash,
                    },
                    target="metadata",
                ),
            ),
            trace=trace,
            implementation_identity=f"minimal_engineering:ofirpress/self-ask@{SELF_ASK_UPSTREAM_REVISION}",
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "official_notebook_minimal_engineering",
                "full_loop": True,
                "search_mode": str(
                    context.run_config.get("search_mode", "frozen_or_user_supplied")
                ),
            },
        )
