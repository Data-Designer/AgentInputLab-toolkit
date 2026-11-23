"""Paper Algorithm 1 UDCS routing, distinct from repository prompt-only routing.

This is a one-step input intervention. It records the routed action/history;
it never executes that action or fabricates a user/environment response.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, ModelCapability
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
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.contracts.state import Message

from .udcs import (
    UDCSDecision,
    _build_user_prompt,
    _clarification_request,
    _system_prompt,
    parse_udcs_response,
)

UDCS_DETERMINISTIC_RELEASE = "clarify.udcs.paper_spec_deterministic_route"
UDCS_PAPER_SPEC = "arxiv:2606.19559v1:Algorithm1"
UDCS_HISTORY_KEY = "udcs.paper_spec.history.v1"


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValidationError("UDCS threshold must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class UDCSDeterministicRoute:
    proposed_action: str
    routed_action: str
    threshold_triggered: bool
    action_overridden: bool
    requests_clarification: bool
    question: str


def route_udcs_decision(decision: UDCSDecision, threshold: float = 0.5) -> UDCSDeterministicRoute:
    """Implement both branches of Algorithm 1 without a second model call."""
    value = _threshold(threshold)
    if not isinstance(decision, UDCSDecision):
        raise ValidationError("UDCS router requires a parsed decision")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in (decision.u_request, decision.confidence)):
        raise AlgorithmOutputError("UDCS routing signals must be finite probabilities")
    triggered = decision.u_request >= value
    action = "request_clarification" if triggered else decision.action
    asks = triggered or decision.requests_clarification
    if not asks:
        question = ""
    elif decision.requests_clarification:
        question = _clarification_request(decision)
    else:
        # A colon in a non-clarification tool action is not a question.
        question = decision.u_request_explanation or "Please clarify the missing or ambiguous details in your request."
    return UDCSDeterministicRoute(decision.action, action, triggered,
        triggered and not decision.requests_clarification, asks, question)


def _history(state: FrozenAgentState) -> tuple[Mapping[str, Any], ...]:
    raw = state.metadata.get(UDCS_HISTORY_KEY, ())
    if not isinstance(raw, (list, tuple)):
        raise ValidationError("UDCS history must be an ordered entry sequence")
    fields = {"source_state_hash", "observation", "think", "u_request", "u_request_explanation",
              "proposed_action", "routed_action", "confidence", "explanation", "threshold", "threshold_triggered"}
    result = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValidationError("UDCS history entry fields differ from the protocol")
        for key in fields - {"observation", "u_request", "confidence", "threshold", "threshold_triggered"}:
            if not isinstance(row[key], str):
                raise ValidationError("UDCS history text fields must be strings")
        if (not isinstance(row["observation"], (tuple, list))
                or any(not isinstance(item, str) for item in row["observation"])
                or not isinstance(row["threshold_triggered"], bool)):
            raise ValidationError("UDCS history observation/route fields differ")
        for key in ("u_request", "confidence", "threshold"):
            _threshold(row[key])
        result.append(freeze_mapping(row))
    return tuple(result)


def _history_message(row: Mapping[str, Any]) -> Message:
    # Preserve the signal-before-action ordering. These are visible previous
    # observations and model emissions, not a guessed future environment state.
    text = "# Previous UDCS step\nObservation:\n" + "\n\n".join(row["observation"])
    for tag, key in (("think", "think"), ("u_request", "u_request"),
                     ("u_request_explanation", "u_request_explanation"), ("action", "routed_action"),
                     ("confidence", "confidence"), ("explanation", "explanation")):
        text += f"\n<{tag}>{row[key]}</{tag}>"
    return Message(role="assistant", content=text)


@dataclass(frozen=True, slots=True)
class UDCSPaperSpecDeterministicPolicy:
    clarification_threshold: float = 0.5
    temperature: float = 0.0
    max_tokens: int = 512
    max_history_entries: int = 64
    profile_kind: str = "controlled_adapter"
    method_id: str = "udcs"
    release_id: str = UDCS_DETERMINISTIC_RELEASE
    family: SemanticFamily = SemanticFamily.CLARIFY

    def __post_init__(self):
        _threshold(self.clarification_threshold)
        GenerationRequest(prompt="configuration validation", temperature=self.temperature, max_tokens=self.max_tokens)
        if type(self.max_history_entries) is not int or self.max_history_entries < 1:
            raise ValidationError("UDCS max_history_entries must be positive")
        if self.profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("UDCS deterministic routing requires controlled or factorial mode")
        if (self.method_id != "udcs" or self.release_id != UDCS_DETERMINISTIC_RELEASE
                or self.family is not SemanticFamily.CLARIFY):
            raise ValidationError("UDCS deterministic identity differs from its registered profile")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        from dataclasses import replace

        history = _history(state)
        if len(history) >= self.max_history_entries:
            raise ValidationError("UDCS history safety bound reached; no silent truncation or model call")
        backend = context.backend
        if backend is None:
            raise ExternalDependencyError("UDCS deterministic routing needs a context chat backend")
        require_capabilities(backend, {ModelCapability.CHAT})
        prompt_state = replace(state, messages=(*state.messages, *(_history_message(row) for row in history)))
        response = backend.generate(GenerationRequest(
            prompt=_build_user_prompt(prompt_state, self.clarification_threshold), system=_system_prompt(),
            temperature=self.temperature, max_tokens=self.max_tokens, seed=context.seed, stop=(),
            metadata={"method_id": self.method_id, "release_id": self.release_id,
                      "routing_semantics": "paper_spec_deterministic"}))
        decision = parse_udcs_response(response.text)
        route = route_udcs_decision(decision, self.clarification_threshold)
        step = freeze_mapping({"source_state_hash": state.state_hash,
            "observation": tuple(block.text for block in state.context), "think": decision.think,
            "u_request": decision.u_request, "u_request_explanation": decision.u_request_explanation,
            "proposed_action": decision.action, "routed_action": route.routed_action,
            "confidence": decision.confidence, "explanation": decision.explanation,
            "threshold": self.clarification_threshold, "threshold_triggered": route.threshold_triggered})
        operations = [PatchOperation(PatchOp.REQUEST_CLARIFICATION, route.question, target="user",
            rationale=decision.u_request_explanation) if route.requests_clarification else
            PatchOperation(PatchOp.NO_OP, rationale="paper Algorithm1 retained the proposed action; no tool execution here")]
        metadata_update = {UDCS_HISTORY_KEY: (*history, step)}
        if not route.requests_clarification and "clarification_request" in state.metadata:
            metadata_update["clarification_request"] = None
        operations.append(PatchOperation(PatchOp.SET_METADATA, metadata_update,
            target="udcs_history", rationale="retain the observed step and emitted uncertainty fields"))
        return InputPatch(method_id=self.method_id, release_id=self.release_id, source_state_hash=state.state_hash,
            operations=tuple(operations), trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id,
                events=(TraceEvent(TracePhase.PREPARE, "paper_spec_threshold", {
                    "threshold": self.clarification_threshold, "python_enforced": True,
                    "source_spec": UDCS_PAPER_SPEC, "previous_history_entries": len(history)}),
                    TraceEvent(TracePhase.PROPOSE, "deterministic_udcs_route", {
                        "u_request": decision.u_request, "confidence": decision.confidence,
                        "proposed_action": route.proposed_action, "routed_action": route.routed_action,
                        "threshold_triggered": route.threshold_triggered, "action_overridden": route.action_overridden,
                        "requests_clarification": route.requests_clarification})),
                counters={"model_calls": 1}, warnings=tuple(f"clamped_{f}" for f in decision.clamped_fields)),
            implementation_identity="paper_algorithm1_one_step_deterministic_route",
            profile_kind=self.profile_kind, metadata={"source_spec": UDCS_PAPER_SPEC,
                "u_request": decision.u_request, "confidence": decision.confidence,
                "raw_action": decision.action, "routed_action": route.routed_action,
                "threshold": self.clarification_threshold, "threshold_enforced": True,
                "threshold_triggered": route.threshold_triggered, "action_overridden": route.action_overridden,
                "requests_clarification": route.requests_clarification,
                "backend_fingerprint": dict(backend.fingerprint()), "environment_action_executed": False,
                "user_reply_generated": False, "parser_semantics": "repository_compatible_clamp_and_action_fallback",
                "history_entries": len(history)+1, "original_native_runner_parity_claimed": False})


__all__ = ["UDCS_DETERMINISTIC_RELEASE", "UDCS_HISTORY_KEY", "UDCS_PAPER_SPEC",
    "UDCSDeterministicRoute", "UDCSPaperSpecDeterministicPolicy", "route_udcs_decision"]
