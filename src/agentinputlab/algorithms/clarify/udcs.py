"""Prompt-native UDCS component preserving the repository routing semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.backends.llm.base import require_capabilities
from agentinputlab.contracts import (
    AlgorithmOutputError,
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

_PROBABILITY = r"[\d.]+"


@dataclass(frozen=True, slots=True)
class UDCSDecision:
    think: str
    u_request: float
    u_request_explanation: str
    action: str
    confidence: float
    explanation: str
    raw_text: str
    clamped_fields: tuple[str, ...] = ()

    @property
    def requests_clarification(self) -> bool:
        return self.action.strip().lower().startswith("request_clarification")


def parse_udcs_response(text: str) -> UDCSDecision:
    """Parse the six repository tags.

    This intentionally preserves the upstream behavior: ``think`` and both
    explanations may be empty, action falls back to a fenced block/full output,
    and numeric uncertainty/confidence values are clamped to ``[0, 1]``.
    """

    if not isinstance(text, str) or not text.strip():
        raise AlgorithmOutputError("UDCS response must be non-empty text")
    think = _optional_tag(text, "think")
    action = _parse_action(text)
    u_raw = _required_probability(text, "u_request")
    confidence_raw = _required_probability(text, "confidence")
    clamped: list[str] = []
    u_request = _clamp(u_raw)
    confidence = _clamp(confidence_raw)
    if u_request != u_raw:
        clamped.append("u_request")
    if confidence != confidence_raw:
        clamped.append("confidence")
    return UDCSDecision(
        think=think,
        u_request=u_request,
        u_request_explanation=_optional_tag(text, "u_request_explanation"),
        action=action,
        confidence=confidence,
        explanation=_optional_tag(text, "explanation"),
        raw_text=text,
        clamped_fields=tuple(clamped),
    )


def _optional_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _required_probability(text: str, tag: str) -> float:
    match = re.search(rf"<{tag}>\s*({_PROBABILITY})\s*</{tag}>", text)
    if not match:
        raise AlgorithmOutputError(f"UDCS response is missing a valid <{tag}> value")
    try:
        return float(match.group(1))
    except ValueError as exc:
        raise AlgorithmOutputError(f"UDCS <{tag}> value is not numeric") from exc


def _parse_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    fenced = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class UDCSPromptNativePolicy:
    """One-step prompt-native component from the UDCS Proposed-v4 runner.

    The threshold is stated in the prompt but is deliberately *not* enforced
    in Python.  This reproduces the pinned repository semantics.  A
    deterministic threshold router must use a different release ID.
    """

    backend: LLMBackend | None = None
    clarification_threshold: float = 0.5
    temperature: float = 0.0
    max_tokens: int = 800
    method_id: str = "udcs"
    release_id: str = "clarify.udcs.repo_native_prompt_route"
    family: SemanticFamily = SemanticFamily.CLARIFY

    def __post_init__(self) -> None:
        if not 0.0 <= self.clarification_threshold <= 1.0:
            raise ValidationError("UDCS clarification_threshold must be in [0, 1]")
        if self.temperature < 0:
            raise ValidationError("UDCS temperature must be non-negative")
        if self.max_tokens < 1:
            raise ValidationError("UDCS max_tokens must be positive")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        backend = self.backend or context.backend
        if backend is None:
            raise ExternalDependencyError("UDCS requires an injected chat backend")
        require_capabilities(backend, {ModelCapability.CHAT})
        response = backend.generate(
            GenerationRequest(
                prompt=_build_user_prompt(state, self.clarification_threshold),
                system=_system_prompt(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=context.seed,
                stop=(),
                metadata={
                    "method_id": self.method_id,
                    "release_id": self.release_id,
                    "routing_semantics": "prompt_native",
                },
            )
        )
        decision = parse_udcs_response(response.text)
        warns = tuple(f"clamped_{name}" for name in decision.clamped_fields)
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "prompt_native_threshold",
                    {
                        "threshold": self.clarification_threshold,
                        "python_enforced": False,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "parsed_udcs_decision",
                    {
                        "think": decision.think,
                        "u_request": decision.u_request,
                        "u_request_explanation": decision.u_request_explanation,
                        "action": decision.action,
                        "confidence": decision.confidence,
                        "explanation": decision.explanation,
                        "threshold_disagreement": (
                            decision.u_request >= self.clarification_threshold
                            and not decision.requests_clarification
                        ),
                    },
                ),
            ),
            counters={"model_calls": 1},
            warnings=warns,
        )
        if decision.requests_clarification:
            operation = PatchOperation(
                PatchOp.REQUEST_CLARIFICATION,
                _clarification_request(decision),
                target="user",
                rationale=decision.u_request_explanation,
            )
        else:
            operation = PatchOperation(
                PatchOp.NO_OP,
                rationale=(
                    "UDCS prompt-native model selected a non-clarification action; "
                    "the repository does not enforce the numerical threshold"
                ),
            )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity="thin_prompt_native_component",
            profile_kind="official_native",
            metadata={
                "u_request": decision.u_request,
                "confidence": decision.confidence,
                "raw_action": decision.action,
                "threshold": self.clarification_threshold,
                "threshold_enforced": False,
                "backend_fingerprint": dict(backend.fingerprint()),
            },
        )


def _system_prompt() -> str:
    return (
        "You are an interactive agent. Follow the available-action contract. "
        "Return all six XML-like fields requested by the user message."
    )


def _build_user_prompt(state: FrozenAgentState, threshold: float) -> str:
    context = "\n\n".join(block.text for block in state.context) or "(none)"
    actions = "\n".join(f"- {name}" for name in state.tool_names) or "(none)"
    history = "\n".join(f"{message.role}: {message.content}" for message in state.messages)
    return (
        f"# Goal\n\n{state.query}\n\n"
        f"# Instruction\n\n{state.instruction or '(none)'}\n\n"
        f"# Current Observation\n\n{context}\n\n"
        f"# Available Actions\n\n{actions}\n"
        "- request_clarification\n\n"
        f"# History\n\n{history or '(none)'}\n\n"
        "# Next action\n\n"
        "Reason inside <think>...</think>. Assess request uncertainty from 0.0 to 1.0 "
        "inside <u_request>...</u_request>, then explain it inside "
        "<u_request_explanation>...</u_request_explanation>. "
        f"If u_request >= {threshold}, your action MUST be request_clarification. "
        "Put the selected action inside <action>...</action>. Finally provide "
        "<confidence>...</confidence> and <explanation>...</explanation>."
    )


def _clarification_request(decision: UDCSDecision) -> str:
    action = decision.action.strip()
    if ":" in action:
        requested = action.split(":", 1)[1].strip()
        if requested:
            return requested
    if decision.u_request_explanation:
        return decision.u_request_explanation
    return "Please clarify the missing or ambiguous details in your request."
