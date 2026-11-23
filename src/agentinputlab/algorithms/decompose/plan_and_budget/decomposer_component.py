"""Evaluator-blind first-stage component of the controlled P&B adaptation.

Shares its existing generalized decomposition prompt/parser unchanged. This is
not credit assessment, budget allocation, the upstream full solver or a claim
of native released-cache/paper-result parity.
"""

from __future__ import annotations

import math

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.contracts import (
    BudgetExceededError,
    CapabilityError,
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

from .cached_soft_budget import P_AND_B_UPSTREAM_REVISION
from .regenerated_soft_budget import (
    _decomposition_prompt,
    _decomposition_system,
    _parse_decomposition,
)


class PlanAndBudgetDecomposerComponent:
    method_id = "plan_and_budget"
    release_id = "plan.plan_and_budget.decomposer_component"
    family = SemanticFamily.DECOMPOSE

    def __init__(self, *, domain="general question answering", initial_level=3,
                 decomposition_max_tokens=768, temperature=0.0):
        if not isinstance(domain, str) or not domain.strip():
            raise ValidationError("Plan-and-Budget domain must be non-empty")
        if type(initial_level) is not int or not 1 <= initial_level <= 5:
            raise ValidationError("Plan-and-Budget initial_level must be in [1, 5]")
        if type(decomposition_max_tokens) is not int or decomposition_max_tokens < 1:
            raise ValidationError("decomposition_max_tokens must be positive")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValidationError("temperature must be finite and non-negative")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValidationError("temperature must be finite and non-negative")
        self.domain, self.initial_level = domain, initial_level
        self.decomposition_max_tokens, self.temperature = decomposition_max_tokens, temperature

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("Plan-and-Budget requires a non-empty query")
        if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
            raise BudgetExceededError("decomposer component has no generation budget")
        backend = context.backend
        if backend is None:
            raise ExternalDependencyError("decomposer component requires an injected LLMBackend")
        if not isinstance(backend, LLMBackend) or ModelCapability.COMPLETION not in backend.capabilities:
            raise CapabilityError("decomposer component requires completion capability")
        system = _decomposition_system(self.domain)
        prompt = _decomposition_prompt(state.query, level=self.initial_level)
        # Match the existing controlled full profile's FIRST-stage reservation.
        # The component does not spend the reserved half on a credit assessor.
        cap = min(self.decomposition_max_tokens, max(1, state.budget.max_output_tokens // 2))
        request = GenerationRequest(prompt=prompt, system=system, temperature=self.temperature,
            max_tokens=cap, seed=context.seed, metadata={"method_id": self.method_id,
                "release_id": self.release_id, "stage": "decomposition",
                "upstream_revision": P_AND_B_UPSTREAM_REVISION})
        response = backend.generate(request)
        steps, warnings = _parse_decomposition(response.text)
        instruction = "Suggested sub-questions:\n" + "\n\n".join(
            f"{index}. {step}" for index, step in enumerate(steps, 1)
        )
        return InputPatch(method_id=self.method_id, release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.APPEND_INSTRUCTION, value=instruction,
                target="instruction", rationale="controlled P&B decomposition-only component"),),
            implementation_identity=f"controlled_p_and_b_first_stage@{P_AND_B_UPSTREAM_REVISION}",
            profile_kind="controlled_adapter",
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id,
                events=(TraceEvent(TracePhase.PROPOSE, "decomposition_component", {
                    "steps": steps, "initial_level": self.initial_level,
                    "decomposition_request_sha256": content_hash({"system": system, "prompt": prompt}),
                    "optimizer_input_projection": "query_only", "gold_visible_to_optimizer": False,
                    "credit_assessment_executed": False, "solver_executed": False}),),
                counters={"model_calls": 1, "optimizer_model_calls": 1,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens, "planned_steps": len(steps)},
                warnings=warnings),
            metadata={"component_only": True, "full_system": False, "steps": steps,
                "live_decomposition": True, "optimizer_input_projection": "query_only",
                "credit_assessment_executed": False, "budget_allocation_executed": False,
                "solver_executed": False, "hard_decoder_cap": False,
                "gold_fields_visible_to_optimizer": False, "paper_result_parity_claimed": False,
                "source_profile": "native_regenerated_soft_budget_first_stage",
                "decomposition_fallback": "decomposition_parse_fallback_to_direct_solve" in warnings})


__all__ = ["PlanAndBudgetDecomposerComponent"]
