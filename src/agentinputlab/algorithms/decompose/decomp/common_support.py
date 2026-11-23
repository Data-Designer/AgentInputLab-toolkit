"""Controlled DecomP-ODQA adapter for the common-support paper arenas."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend
from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
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

DECOMP_ODQA_REVISION = "9e0f7e717f6243a8b58b22a86eba4843088c2151"
DECOMP_COMMON_SUPPORT_RELEASE = "plan.decomp.common_support_odqa"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMAND = re.compile(
    r"^\((?P<operation>select)\) \[(?P<handler>retrieve_odqa|multihop_titleqa)\] "
    r"(?P<question>.+)$"
)
_DECOMPOSER_CHAT_SYSTEM = (
    "You are the DecomP controller. Continue only the final QS line using the "
    "demonstrated syntax. On the first step, always emit exactly (select) "
    "[retrieve_odqa] <one self-contained evidence-seeking subquestion>. After "
    "evidence exists, emit another retrieve_odqa step only if needed; otherwise "
    "emit (select) [multihop_titleqa] <aggregation question>. Never answer the "
    "user, explain, emit [EOQ] on the first step, or add a QS prefix."
)
_HANDLER_CHAT_SYSTEM = (
    "You are the DecomP retrieve_odqa handler. Answer the final question using "
    "only the supplied retrieved passages. Return one concise answer line, "
    "without explanation, labels, or markup."
)


@dataclass(frozen=True, slots=True)
class DecompCommonSupportConfig:
    """Frozen DecomP-ODQA prompt and bounded common-support controls."""

    demonstrations: str
    prompt_source_sha256: str
    demonstrations_sha256: str
    max_steps: int = 3
    retrieval_top_k: int = 2
    max_decomposer_tokens: int = 160
    max_handler_tokens: int = 128
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.demonstrations.strip():
            raise ValidationError("DecomP common support requires frozen demonstrations")
        for label, value in (
            ("prompt_source_sha256", self.prompt_source_sha256),
            ("demonstrations_sha256", self.demonstrations_sha256),
        ):
            if _HEX64.fullmatch(value) is None:
                raise ValidationError(f"DecomP {label} must be a SHA-256 digest")
        if content_hash(self.demonstrations) != self.demonstrations_sha256:
            raise ValidationError("DecomP demonstrations hash does not match the content")
        for label, value in (
            ("max_steps", self.max_steps),
            ("retrieval_top_k", self.retrieval_top_k),
            ("max_decomposer_tokens", self.max_decomposer_tokens),
            ("max_handler_tokens", self.max_handler_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"DecomP {label} must be a positive integer")
        if self.max_steps > 8:
            raise ValidationError("DecomP max_steps must not exceed the ODQA source cap")
        if self.retrieval_top_k > 100:
            raise ValidationError("DecomP retrieval_top_k must not exceed 100")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 2
        ):
            raise ValidationError("DecomP temperature must be numeric and in [0, 2]")


@dataclass(frozen=True, slots=True)
class DecompCommonSupportStep:
    index: int
    command: str
    operation: str
    handler: str
    question: str
    answer: str
    document_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class DecompCommonSupport:
    """Run the source-locked QC/QS/A controller with shared dependencies.

    The pinned ODQA system ends by routing the accumulated subtask state to a
    multihop answer handler.  In this input-intervention profile that final
    handler is deliberately delegated to the benchmark's common downstream
    agent.  The optimizer therefore injects the modular subquestion/handler
    trace and real retrieved evidence, never its own final answer.
    """

    config: DecompCommonSupportConfig
    method_id: str = "decomp"
    release_id: str = DECOMP_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.DECOMPOSE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("DecomP common support requires a non-empty query")
        backend = context.backend
        retriever = context.retriever
        if not isinstance(backend, LLMBackend):
            raise ExternalDependencyError("DecomP common support requires an LLMBackend")
        if not isinstance(retriever, Retriever):
            raise ExternalDependencyError("DecomP common support requires a Retriever")

        steps: list[DecompCommonSupportStep] = []
        documents: dict[str, ContextBlock] = {}
        model_calls = 0
        retrieval_calls = 0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        seen_commands: set[str] = set()
        termination = "max_steps"

        def generate(prompt: str, *, system: str, stage: str, max_tokens: int) -> str:
            nonlocal model_calls, input_tokens, output_tokens, cost_usd
            if model_calls >= state.budget.max_model_calls:
                raise BudgetExceededError("DecomP exhausted max_model_calls")
            remaining = state.budget.max_output_tokens - output_tokens
            if remaining < 1:
                raise BudgetExceededError("DecomP exhausted max_output_tokens")
            response = backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    system=system,
                    temperature=self.config.temperature,
                    max_tokens=min(max_tokens, remaining),
                    seed=context.seed + model_calls,
                    stop=("\n",),
                    metadata={
                        "method_id": self.method_id,
                        "release_id": self.release_id,
                        "stage": stage,
                    },
                )
            )
            model_calls += 1
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            cost_usd += response.usage.cost_usd or 0.0
            if output_tokens > state.budget.max_output_tokens:
                raise BudgetExceededError("DecomP exceeded max_output_tokens")
            if (
                state.budget.max_context_tokens is not None
                and input_tokens > state.budget.max_context_tokens
            ):
                raise BudgetExceededError("DecomP exceeded max_context_tokens")
            if state.budget.max_cost_usd is not None and cost_usd > state.budget.max_cost_usd:
                raise BudgetExceededError("DecomP exceeded max_cost_usd")
            if not isinstance(response.text, str) or not response.text.strip():
                raise AlgorithmOutputError(f"DecomP {stage} returned empty output")
            return response.text.strip()

        for index in range(self.config.max_steps):
            prompt = _render_decomposer_prompt(
                self.config.demonstrations,
                state.query,
                steps,
            )
            raw = generate(
                prompt,
                system=_DECOMPOSER_CHAT_SYSTEM,
                stage=f"decomposer_{index}",
                max_tokens=self.config.max_decomposer_tokens,
            )
            parsed = parse_decomp_command(raw)
            if parsed is None:
                termination = "eoq"
                break
            operation, handler, question, command = parsed
            if command in seen_commands:
                raise AlgorithmOutputError("DecomP emitted a duplicate controller command")
            seen_commands.add(command)
            if handler == "multihop_titleqa":
                steps.append(
                    DecompCommonSupportStep(
                        index=index,
                        command=command,
                        operation=operation,
                        handler=handler,
                        question=question,
                        answer="",
                    )
                )
                termination = "aggregation_delegated_to_common_agent"
                break
            if retrieval_calls >= state.budget.max_retrieval_calls:
                raise BudgetExceededError("DecomP exhausted max_retrieval_calls")
            results = _validated_results(
                retriever.retrieve(question, top_k=self.config.retrieval_top_k),
                top_k=self.config.retrieval_top_k,
            )
            retrieval_calls += 1
            if not results:
                raise AlgorithmOutputError("DecomP retrieve_odqa handler returned no documents")
            for result in results:
                block = _document_block(result)
                documents.setdefault(block.block_id, block)
            answer = generate(
                _render_handler_prompt(question, results),
                system=_HANDLER_CHAT_SYSTEM,
                stage=f"retrieve_odqa_handler_{index}",
                max_tokens=self.config.max_handler_tokens,
            )
            answer_payload = json.dumps(
                {
                    "titles": [item.document.title or item.document.doc_id for item in results],
                    "answer": [answer],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            steps.append(
                DecompCommonSupportStep(
                    index=index,
                    command=command,
                    operation=operation,
                    handler=handler,
                    question=question,
                    answer=answer_payload,
                    document_ids=tuple(item.document.doc_id for item in results),
                )
            )

        if not steps:
            raise AlgorithmOutputError("DecomP produced no executable modular step")
        if not documents:
            raise AlgorithmOutputError(
                "DecomP delegated aggregation before materializing retrieved evidence"
            )
        trace_text = _materialized_trace(state.query, steps, termination)
        trace_block = ContextBlock(
            text=trace_text,
            block_id=f"{state.task_id or state.state_hash}:decomp-trace",
            source="decomp_odqa_common_support",
            metadata={
                "method_id": self.method_id,
                "release_id": self.release_id,
                "termination": termination,
                "final_answer_injected": False,
                "evidence_kind": "decomposition_trace",
            },
        )
        events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "source_locked_odqa_profile",
                {
                    "decomp_upstream_revision": DECOMP_REVISION,
                    "odqa_upstream_revision": DECOMP_ODQA_REVISION,
                    "prompt_source_sha256": self.config.prompt_source_sha256,
                    "demonstrations_sha256": self.config.demonstrations_sha256,
                    "serialization": "QC/QS/A",
                    "controller": "bounded_best_first_single_candidate",
                    "decomposer_chat_compatibility_sha256": content_hash(_DECOMPOSER_CHAT_SYSTEM),
                    "handler_chat_compatibility_sha256": content_hash(_HANDLER_CHAT_SYSTEM),
                },
            )
        ]
        events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "decomp_participant_call",
                {
                    "index": step.index,
                    "command": step.command,
                    "operation": step.operation,
                    "handler": step.handler,
                    "question": step.question,
                    "answer": step.answer,
                    "document_ids": step.document_ids,
                },
            )
            for step in steps
        )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "decomp_common_support_result",
                {
                    "termination": termination,
                    "trace_block_id": trace_block.block_id,
                    "retrieved_document_ids": tuple(documents),
                    "final_answer_injected": False,
                    "downstream_common_agent_required": True,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(events),
            counters={
                "model_calls": model_calls,
                "retrieval_calls": retrieval_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "participant_calls": len(steps),
                "retrieved_documents": len(documents),
            },
        )
        blocks = (*documents.values(), trace_block)
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    value=blocks,
                    target="decomposition_evidence",
                    rationale="materialize DecomP participant trace and real handler evidence",
                ),
            ),
            trace=trace,
            implementation_identity=(
                "controlled_source_locked:allenai/DecomP@"
                f"{DECOMP_REVISION}+HarshTrivedi/DecomP-ODQA@{DECOMP_ODQA_REVISION}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "controlled_source_locked",
                "upstream_qc_qs_a_serialization_preserved": True,
                "upstream_participant_routing_preserved": True,
                "best_first_width": 1,
                "chat_compatibility_wrapper": True,
                "final_answer_injected": False,
                "aggregation_delegated_to_common_agent": True,
                "termination": termination,
                "backend_fingerprint": dict(backend.fingerprint()),
                "retriever_fingerprint": dict(retriever.fingerprint()),
            },
        )


def parse_decomp_command(raw: Any) -> tuple[str, str, str, str] | None:
    if not isinstance(raw, str):
        raise AlgorithmOutputError("DecomP decomposer returned non-text output")
    line = raw.strip().splitlines()[0].strip()
    if line.startswith("QS:"):
        line = line.removeprefix("QS:").strip()
    if line == "[EOQ]":
        return None
    match = _COMMAND.fullmatch(line)
    if match is None:
        raise AlgorithmOutputError("DecomP command violates the source router grammar")
    question = match.group("question").strip()
    if not question:
        raise AlgorithmOutputError("DecomP command contains an empty subquestion")
    return match.group("operation"), match.group("handler"), question, line


def _render_decomposer_prompt(
    demonstrations: str,
    query: str,
    steps: list[DecompCommonSupportStep],
) -> str:
    value = f"{demonstrations.rstrip()}\n\n\nQC: {query.strip()}"
    for step in steps:
        value += f"\nQS: {step.command}"
        if step.answer:
            value += f"\nA: {step.answer}"
    return f"{value}\nQS:"


def _render_handler_prompt(question: str, results: tuple[RetrievalResult, ...]) -> str:
    context = "\n\n".join(
        f"Title: {item.document.title or item.document.doc_id}\n{item.document.text}"
        for item in results
    )
    return f"{context}\n\nQ: {question}\nA:"


def _validated_results(raw: Any, *, top_k: int) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(raw)
    except TypeError as exc:
        raise AlgorithmOutputError("DecomP retriever returned a non-sequence") from exc
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise AlgorithmOutputError("DecomP retriever returned a non-RetrievalResult item")
    if len(results) > top_k:
        raise AlgorithmOutputError("DecomP retriever returned more than top_k documents")
    ids = tuple(item.document.doc_id for item in results)
    if len(ids) != len(set(ids)):
        raise AlgorithmOutputError("DecomP retriever returned duplicate document IDs")
    if tuple(item.rank for item in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("DecomP retriever ranks must be contiguous and one-based")
    return results


def _document_block(result: RetrievalResult) -> ContextBlock:
    document = result.document
    if not document.corpus_revision:
        raise AlgorithmOutputError("DecomP documents require a corpus revision")
    reserved = {
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "rank",
        "retrieval_features",
        "title",
    }
    if reserved & set(document.metadata):
        raise AlgorithmOutputError("DecomP document metadata uses reserved provenance keys")
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


def _materialized_trace(
    query: str,
    steps: list[DecompCommonSupportStep],
    termination: str,
) -> str:
    lines = [
        "Decomposed Prompting modular execution trace (QC/QS/A protocol).",
        f"QC: {query}",
    ]
    for step in steps:
        lines.append(f"QS: {step.command}")
        if step.answer:
            lines.append(f"A: {step.answer}")
    lines.append(f"Termination: {termination}")
    lines.append("Use the subtask evidence above to complete the original task.")
    return "\n".join(lines)


# Imported late to avoid a circular import through the package __init__.
from .worker_bridge import DECOMP_REVISION  # noqa: E402

__all__ = [
    "DECOMP_COMMON_SUPPORT_RELEASE",
    "DECOMP_ODQA_REVISION",
    "DecompCommonSupport",
    "DecompCommonSupportConfig",
    "DecompCommonSupportStep",
    "parse_decomp_command",
]
