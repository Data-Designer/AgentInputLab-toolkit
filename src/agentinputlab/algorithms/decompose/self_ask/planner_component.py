"""First-continuation component of the existing controlled Self-Ask adapter.

The source prompt, first-call stop and parsing markers are preserved. Search,
intermediate answers and iterative solver execution are not part of this API.
"""
from __future__ import annotations

import re

from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
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
from agentinputlab.contracts.serialization import content_hash

from .loop import (
    DEFAULT_SELF_ASK_MARKERS,
    SELF_ASK_UPSTREAM_REVISION,
    SelfAskConfig,
    SelfAskLoop,
    SelfAskMarkers,
    parse_final_answer,
    parse_followup_question,
    render_initial_self_ask_prompt,
)


class SelfAskPlannerComponent:
    """Suggest the first native follow-up in one paid, budgeted generation.

    A native ``No`` decision yields an explicit NO_OP. A continuation that
    already contains the final-answer marker remains visible as such; it is
    not a searched fact, verified answer or complete Self-Ask trajectory.
    """

    method_id = "self_ask"
    release_id = "plan.self_ask.planner_component"
    family = SemanticFamily.DECOMPOSE

    def __init__(self, config: SelfAskConfig, *, markers: SelfAskMarkers = DEFAULT_SELF_ASK_MARKERS,
                 allow_no_decision_explanation: bool = False):
        if not isinstance(config, SelfAskConfig) or not isinstance(markers, SelfAskMarkers):
            raise ValidationError("Self-Ask planner requires SelfAskConfig and SelfAskMarkers")
        if type(allow_no_decision_explanation) is not bool:
            raise ValidationError("allow_no_decision_explanation must be boolean")
        self.config, self.markers = config, markers
        self.allow_no_decision_explanation = allow_no_decision_explanation

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("Self-Ask planner requires a non-empty query")
        if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
            raise BudgetExceededError("Self-Ask planner has no generation budget")
        backend = SelfAskLoop._require_backend(context.backend)
        prompt = render_initial_self_ask_prompt(state.query, self.config, self.markers)
        request = GenerationRequest(prompt=prompt, temperature=self.config.temperature,
            max_tokens=min(self.config.max_tokens_per_call, state.budget.max_output_tokens),
            seed=context.seed, stop=(f"\n{self.markers.intermediate_answer}",),
            metadata={"method_id": self.method_id, "release_id": self.release_id})
        response = backend.generate(request)
        continuation = response.text.strip()
        if not continuation:
            raise AlgorithmOutputError("Self-Ask planner returned an empty initial continuation")
        final = parse_final_answer(continuation, self.markers)
        followup = None if final is not None else parse_followup_question(continuation, self.markers)
        compatibility_applied = False
        if final is not None:
            decision = "initial_final_answer"
        elif followup is not None:
            decision = "first_followup"
        elif re.fullmatch(
            rf"(?:{re.escape(self.markers.followup_needed)}\s*)?No[.!]?",
            continuation, flags=re.IGNORECASE,
        ):
            decision = "no_followup"
        elif self.allow_no_decision_explanation and _no_decision_with_explanation(
            continuation, self.markers
        ):
            decision = "no_followup"
            compatibility_applied = True
        else:
            raise AlgorithmOutputError("Self-Ask planner initial continuation has no supported marker/No decision")
        operation = (
            PatchOperation(PatchOp.NO_OP, rationale="Self-Ask initial planner requested no follow-up")
            if decision == "no_followup" else
            PatchOperation(PatchOp.APPEND_INSTRUCTION,
                value="Self-Ask initial planner continuation (not retrieved evidence):\n" + continuation,
                target="instruction", rationale="materialize only the native first continuation")
        )
        counters = {"model_calls": 1, "optimizer_model_calls": 1, "retrieval_calls": 0,
                    "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        if response.usage.cost_usd is not None:
            counters["cost_usd"] = response.usage.cost_usd
        return InputPatch(method_id=self.method_id, release_id=self.release_id,
            source_state_hash=state.state_hash, operations=(operation,),
            implementation_identity=f"controlled_self_ask_first_continuation@{SELF_ASK_UPSTREAM_REVISION}",
            profile_kind="controlled_adapter",
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id,
                events=(TraceEvent(TracePhase.PREPARE, "self_ask_initial_request", {
                    "demonstration_hash": content_hash(self.config.demonstrations),
                    "prompt_hash": content_hash(prompt), "optimizer_input_projection": "query_only",
                    "upstream_revision": SELF_ASK_UPSTREAM_REVISION,
                    "max_tokens": request.max_tokens, "seed": request.seed, "stop": request.stop}),
                    TraceEvent(TracePhase.PROPOSE, "self_ask_initial_continuation", {
                        "continuation": continuation, "decision": decision,
                        "first_followup": followup, "initial_answer_present": final is not None,
                        **({"response_compatibility": "standalone_no_with_explanation.v1",
                            "compatibility_applied": compatibility_applied}
                           if self.allow_no_decision_explanation else {})})),
                counters=counters),
            metadata={"component_only": True, "full_loop": False,
                "source_stage": "full_loop_initial_continuation", "planner_decision": decision,
                "initial_answer_present": final is not None,
                "search_executed": False, "iterative_solver_executed": False,
                "optimizer_input_projection": "query_only", "gold_fields_visible_to_optimizer": False,
                "paper_result_parity_claimed": False,
                **({"response_compatibility": "standalone_no_with_explanation.v1",
                    "compatibility_applied": compatibility_applied}
                   if self.allow_no_decision_explanation else {})})


def _no_decision_with_explanation(text: str, markers: SelfAskMarkers) -> bool:
    """Accept a standalone No line plus prose, never infer a missing decision."""
    first, separator, explanation = text.partition("\n")
    if not separator or not explanation.strip():
        return False
    if not re.fullmatch(
        rf"(?:{re.escape(markers.followup_needed)}\s*)?No[.!]?", first.strip(), re.IGNORECASE
    ):
        return False
    # Do not silently discard a conflicting/repeated protocol decision, an
    # incomplete follow-up marker, or a case-variant final-answer marker.
    reserved = (markers.followup_needed, markers.followup,
                markers.intermediate_answer, markers.final_answer)
    if any(marker.casefold() in explanation.casefold() for marker in reserved):
        return False
    return re.search(r"(?im)^\s*(?:Yes|No)[.!]?\s*$", explanation) is None


__all__ = ["SelfAskPlannerComponent"]
