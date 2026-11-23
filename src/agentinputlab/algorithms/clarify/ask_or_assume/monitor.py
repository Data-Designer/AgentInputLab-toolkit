"""External component bridge for Ask or Assume?'s intent monitor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionTrace,
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
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    dependency_fingerprint,
    mapping_sequence,
    optional_text,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

ASK_OR_ASSUME_UPSTREAM_COMMIT = "728464a755f15fd448aff61a5cb32dce97fd9c1d"
ASK_OR_ASSUME_INTENT_PROMPT_SHA256 = (
    "c0a150af1020090a0016930c0a319f1df0e19eff40c034d07d141c690b611190"
)
_TRANSITIONS = {"delegate", "verdict", "force", "await", "resume", "skip_first_turn"}


@dataclass(frozen=True, slots=True)
class AskOrAssumeControls:
    skip_first_turn: bool = True
    max_coding_turns: int = 100
    max_user_turns: int = 3
    max_model_calls: int = 2
    max_tokens: int = 20_000
    max_cost_usd: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.skip_first_turn, bool):
            raise ValidationError("Ask-or-Assume skip_first_turn must be boolean")
        for name in (
            "max_coding_turns",
            "max_user_turns",
            "max_model_calls",
            "max_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"Ask-or-Assume {name} must be positive")
        if (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(float(self.max_cost_usd))
            or self.max_cost_usd < 0
        ):
            raise ValidationError("Ask-or-Assume max_cost_usd must be non-negative")


@dataclass(frozen=True, slots=True)
class AskOrAssumeDecision:
    needs_clarification: bool
    reasons: tuple[str, ...]
    clarification_question: str
    clarification_questions: tuple[Mapping[str, object], ...]
    intent_summary: str
    message: str
    decision_trace: tuple[Mapping[str, object], ...]
    transition: str
    delegated: bool
    forced: bool
    first_turn_skipped: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AskOrAssumeDecision:
        needs_clarification = value.get("needs_clarification")
        if not isinstance(needs_clarification, bool):
            raise AlgorithmOutputError("Ask-or-Assume needs_clarification must be a strict boolean")
        reasons = string_tuple(
            value.get("reasons", ()),
            label="Ask-or-Assume reasons",
            unique=True,
        )
        if needs_clarification and not reasons:
            raise AlgorithmOutputError(
                "Ask-or-Assume clarification verdict requires IntentAgent reasons"
            )
        question = optional_text(
            value,
            "clarification_question",
            label="Ask-or-Assume response",
        )
        if needs_clarification and not question.strip():
            raise AlgorithmOutputError("Ask-or-Assume ask verdict requires a question")
        if not needs_clarification and question.strip():
            raise AlgorithmOutputError(
                "Ask-or-Assume continue verdict cannot carry a clarification question"
            )
        transition = required_text(value, "transition", label="Ask-or-Assume response")
        if transition not in _TRANSITIONS:
            raise AlgorithmOutputError("Ask-or-Assume returned an invalid state transition")
        booleans: dict[str, bool] = {}
        for key in ("delegated", "forced", "first_turn_skipped"):
            item = value.get(key, False)
            if not isinstance(item, bool):
                raise AlgorithmOutputError(f"Ask-or-Assume {key} must be boolean")
            booleans[key] = item
        return cls(
            needs_clarification=needs_clarification,
            reasons=reasons,
            clarification_question=question,
            clarification_questions=mapping_sequence(
                value.get("clarification_questions", ()),
                label="Ask-or-Assume clarification questions",
            ),
            intent_summary=optional_text(
                value,
                "intent_summary",
                label="Ask-or-Assume response",
            ),
            message=optional_text(value, "message", label="Ask-or-Assume response"),
            decision_trace=mapping_sequence(
                value.get("decision_trace", ()),
                label="Ask-or-Assume decision trace",
            ),
            transition=transition,
            delegated=booleans["delegated"],
            forced=booleans["forced"],
            first_turn_skipped=booleans["first_turn_skipped"],
        )


@dataclass(slots=True)
class AskOrAssumeIntentMonitor:
    """Component-only uncertainty verdict; it does not impersonate OpenHands full."""

    client: WorkerClient | None = None
    controls: AskOrAssumeControls = AskOrAssumeControls()
    intent_prompt_hash: str = ASK_OR_ASSUME_INTENT_PROMPT_SHA256
    method_id: str = "ask_or_assume"
    release_id: str = "clarify.ask_or_assume.intent_monitor_component"
    family: SemanticFamily = SemanticFamily.CLARIFY

    def __post_init__(self) -> None:
        if self.intent_prompt_hash != ASK_OR_ASSUME_INTENT_PROMPT_SHA256:
            raise ValidationError("Ask-or-Assume requires the pinned official IntentAgent prompt")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        turn_index_raw = state.metadata.get("turn_index", len(state.messages))
        if isinstance(turn_index_raw, bool) or not isinstance(turn_index_raw, int):
            raise ValidationError("Ask-or-Assume turn_index must be an integer")
        if turn_index_raw < 0:
            raise ValidationError("Ask-or-Assume turn_index must be non-negative")
        coding_turns = state.metadata.get("ask_or_assume_coding_turns", turn_index_raw)
        if isinstance(coding_turns, bool) or not isinstance(coding_turns, int):
            raise ValidationError("ask_or_assume_coding_turns must be an integer")
        if coding_turns < 0:
            raise ValidationError("ask_or_assume_coding_turns must be non-negative")
        clarification_turns = state.metadata.get(
            "ask_or_assume_user_turns",
            state.metadata.get("ask_or_assume_clarification_turns", 0),
        )
        if (
            isinstance(clarification_turns, bool)
            or not isinstance(clarification_turns, int)
            or clarification_turns < 0
        ):
            raise ValidationError(
                "ask_or_assume_clarification_turns must be a non-negative integer"
            )
        if self.controls.skip_first_turn and turn_index_raw == 0:
            return self._first_turn_patch(state)
        if coding_turns >= self.controls.max_coding_turns:
            raise AlgorithmOutputError("Ask-or-Assume effective coding-turn budget is exhausted")

        worker = resolve_worker(self.client, context, label="Ask-or-Assume")
        coding_backend_fp = dependency_fingerprint(
            context.backend,
            label="Ask-or-Assume coding/clarification model",
            required=True,
        )
        configured_intent_fp = context.run_config.get("ask_or_assume_intent_model_fingerprint")
        if configured_intent_fp is None:
            intent_backend_fp = coding_backend_fp
        elif isinstance(configured_intent_fp, Mapping) and configured_intent_fp:
            intent_backend_fp = dict(configured_intent_fp)
        else:
            raise ValidationError(
                "ask_or_assume_intent_model_fingerprint must be a non-empty mapping"
            )
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=ASK_OR_ASSUME_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "task_view": {
                    "query": state.query,
                    "instruction": state.instruction,
                    "context": state.context,
                    "tool_names": state.tool_names,
                },
                "history": state.messages,
                "turn_index": coding_turns,
                "clarification_turns": clarification_turns,
                "controls": asdict(self.controls),
                "intent_prompt_hash": self.intent_prompt_hash,
                "seed": context.seed,
            },
            dependency_fingerprints={
                "intent_model": intent_backend_fp,
                "coding_model": coding_backend_fp,
            },
            minimum_reservation={"model_calls": 1},
            protocol_version="agentinputlab.ask-or-assume.monitor.v1",
        )
        response = call_worker(worker, request)
        decision = AskOrAssumeDecision.from_mapping(response.payload)
        minimum_calls = 2 if decision.needs_clarification else 1
        if response.usage.model_calls < minimum_calls:
            raise AlgorithmOutputError(
                "Ask-or-Assume usage omits the IntentAgent or Clarify-question model call"
            )
        if decision.first_turn_skipped:
            raise AlgorithmOutputError(
                "Ask-or-Assume worker cannot retroactively skip a delegated turn"
            )
        if not decision.delegated:
            raise AlgorithmOutputError(
                "Ask-or-Assume post-first-turn verdict must come from delegated IntentAgent"
            )
        if decision.forced and not decision.needs_clarification:
            raise AlgorithmOutputError("Ask-or-Assume forced transition must request clarification")
        if decision.needs_clarification and not decision.forced:
            raise AlgorithmOutputError(
                "Ask-or-Assume clarification must use the forced Clarify-only transition"
            )
        if decision.needs_clarification and decision.transition not in {"force", "await"}:
            raise AlgorithmOutputError(
                "Ask-or-Assume clarification requires a force/await controller transition"
            )
        if not decision.needs_clarification and decision.transition in {"force", "await"}:
            raise AlgorithmOutputError(
                "Ask-or-Assume continue verdict cannot enter a clarification transition"
            )
        if decision.needs_clarification and clarification_turns >= self.controls.max_user_turns:
            raise AlgorithmOutputError(
                "Ask-or-Assume worker requested clarification after max_user_turns"
            )
        if decision.needs_clarification:
            operations = (
                PatchOperation(
                    PatchOp.REQUEST_CLARIFICATION,
                    decision.clarification_question,
                    target="user",
                    rationale="Ask-or-Assume intent monitor requested missing task information",
                ),
                PatchOperation(
                    PatchOp.SET_METADATA,
                    {
                        "ask_or_assume_clarification_turns": clarification_turns + 1,
                        "ask_or_assume_user_turns": clarification_turns + 1,
                    },
                    target="metadata",
                    rationale="track the release-level clarification-turn safety limit",
                ),
            )
        else:
            operations = (
                PatchOperation(
                    PatchOp.NO_OP,
                    rationale="Ask-or-Assume intent monitor allowed the coding agent to proceed",
                ),
            )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "intent_monitor_controls",
                    {
                        **asdict(self.controls),
                        "intent_prompt_hash": self.intent_prompt_hash,
                        "turn_index": turn_index_raw,
                        "effective_coding_turns": coding_turns,
                        "clarification_turns": clarification_turns,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "intent_verdict",
                    {
                        "needs_clarification": decision.needs_clarification,
                        "reasons": decision.reasons,
                        "intent_summary": decision.intent_summary,
                        "message": decision.message,
                        "clarification_question": decision.clarification_question,
                        "clarification_questions": decision.clarification_questions,
                        "worker_decision_trace": decision.decision_trace,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "controller_transition",
                    {
                        "transition": decision.transition,
                        "delegated": decision.delegated,
                        "forced": decision.forced,
                        "first_turn_skipped": decision.first_turn_skipped,
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "clarification_requests": int(decision.needs_clarification),
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=operations,
            trace=trace,
            implementation_identity=(
                f"official_component_adapter:nedwards99/ask-or-assume@"
                f"{ASK_OR_ASSUME_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "official_component_adapter",
                "component_only": True,
                "openhands_full_system": False,
                "needs_clarification": decision.needs_clarification,
                "clarification_turns_before": clarification_turns,
                "clarification_turns_after": (
                    clarification_turns + int(decision.needs_clarification)
                ),
                "reasons": decision.reasons,
                "transition": decision.transition,
                "worker_fingerprint": worker_fingerprint(worker, label="Ask-or-Assume"),
                "intent_model_fingerprint": intent_backend_fp,
                "coding_model_fingerprint": coding_backend_fp,
                "effective_coding_turns": coding_turns,
                "effective_user_turns": clarification_turns,
            },
        )

    def _first_turn_patch(self, state: FrozenAgentState) -> InputPatch:
        """Preserve the upstream rule: the coding agent acts once before delegation."""

        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.FINALIZE,
                    "skip_first_coding_turn",
                    {"transition": "skip_first_turn", "worker_called": False},
                ),
            ),
            counters={"worker_calls": 0, "clarification_requests": 0},
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.NO_OP,
                    rationale="Ask-or-Assume delegates only after the first coding turn",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"official_component_adapter:nedwards99/ask-or-assume@"
                f"{ASK_OR_ASSUME_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "official_component_adapter",
                "component_only": True,
                "openhands_full_system": False,
                "first_turn_skipped": True,
            },
        )


__all__ = [
    "ASK_OR_ASSUME_INTENT_PROMPT_SHA256",
    "ASK_OR_ASSUME_UPSTREAM_COMMIT",
    "AskOrAssumeControls",
    "AskOrAssumeDecision",
    "AskOrAssumeIntentMonitor",
]
