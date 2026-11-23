"""Controlled Plan-and-Budget adaptation with live, evaluator-blind planning.

The official repository prepares decompositions and credit assignments before
answer generation.  Its released caches cover MATH-500, NaturalInstructions,
and TravelPlanner, so they cannot be reused for a new benchmark such as
QASPER.  This module preserves the released two-stage preparation protocol and
the original weighted soft-word allocation, while regenerating the two cached
fields from the visible input only.  It is therefore a controlled adapter, not
an official-native or paper-parity claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.contracts import (
    AlgorithmOutputError,
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

from .cached_soft_budget import (
    P_AND_B_UPSTREAM_REVISION,
    CachedPlanBudgetRecord,
    DecaySchedule,
    allocate_tokens,
    render_soft_budget_instruction,
)

_NUMBERED_ITEM = re.compile(r"(?ms)^\s*(?P<index>\d+)\.\s*(?P<text>.*?)(?=^\s*\d+\.\s|\Z)")


@dataclass(frozen=True, slots=True)
class RegeneratedPlanBudgetConfig:
    """Frozen controls for the general-benchmark P&B preparation calls."""

    domain: str = "general question answering"
    initial_level: int = 3
    total_word_budget: int = 512
    schedule: DecaySchedule | str = DecaySchedule.CONSTANT
    decomposition_max_tokens: int = 768
    assessment_max_tokens: int = 768
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValidationError("Plan-and-Budget domain must be non-empty")
        if not 1 <= self.initial_level <= 5:
            raise ValidationError("Plan-and-Budget initial_level must be in [1, 5]")
        if self.total_word_budget < 1:
            raise ValidationError("Plan-and-Budget total_word_budget must be positive")
        for label, value in (
            ("decomposition_max_tokens", self.decomposition_max_tokens),
            ("assessment_max_tokens", self.assessment_max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"Plan-and-Budget {label} must be positive")
        if self.temperature < 0:
            raise ValidationError("Plan-and-Budget temperature must be non-negative")
        object.__setattr__(self, "schedule", DecaySchedule.parse(self.schedule))


class PlanAndBudgetRegeneratedSoftBudget:
    """Regenerate official cache fields from visible state, then apply P&B."""

    method_id = "plan_and_budget"
    release_id = "plan.plan_and_budget.native_regenerated_soft_budget"
    family = SemanticFamily.DECOMPOSE

    def __init__(
        self,
        *,
        domain: str = "general question answering",
        initial_level: int = 3,
        total_word_budget: int = 512,
        schedule: str | DecaySchedule = DecaySchedule.CONSTANT,
        decomposition_max_tokens: int = 768,
        assessment_max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> None:
        self.config = RegeneratedPlanBudgetConfig(
            domain=domain,
            initial_level=initial_level,
            total_word_budget=total_word_budget,
            schedule=schedule,
            decomposition_max_tokens=decomposition_max_tokens,
            assessment_max_tokens=assessment_max_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _backend(context: InterventionContext) -> LLMBackend:
        backend = context.backend
        if backend is None:
            raise ExternalDependencyError(
                "regenerated Plan-and-Budget requires an injected LLMBackend"
            )
        if not isinstance(backend, LLMBackend):
            raise CapabilityError("Plan-and-Budget backend does not implement LLMBackend")
        capabilities = set(backend.capabilities)
        if ModelCapability.COMPLETION not in capabilities:
            raise CapabilityError("Plan-and-Budget backend lacks capability: completion")
        if not capabilities.intersection(
            {ModelCapability.JSON_OBJECT, ModelCapability.JSON_SCHEMA}
        ):
            raise CapabilityError(
                "Plan-and-Budget backend requires json_object or json_schema capability"
            )
        return backend

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("Plan-and-Budget requires a non-empty query")
        if state.budget.max_model_calls < 2:
            raise BudgetExceededError("regenerated Plan-and-Budget requires two model calls")
        if state.budget.max_output_tokens < 2:
            raise BudgetExceededError("regenerated Plan-and-Budget has no output-token budget")
        backend = self._backend(context)

        decomposition_system = _decomposition_system(self.config.domain)
        decomposition_prompt = _decomposition_prompt(
            state.query,
            level=self.config.initial_level,
        )
        first_cap = min(
            self.config.decomposition_max_tokens,
            max(1, state.budget.max_output_tokens // 2),
        )
        decomposition = backend.generate(
            GenerationRequest(
                prompt=decomposition_prompt,
                system=decomposition_system,
                temperature=self.config.temperature,
                max_tokens=first_cap,
                seed=context.seed,
                metadata={
                    "method_id": self.method_id,
                    "release_id": self.release_id,
                    "stage": "decomposition",
                    "upstream_revision": P_AND_B_UPSTREAM_REVISION,
                },
            )
        )
        steps, decomposition_warnings = _parse_decomposition(decomposition.text)

        remaining_output = state.budget.max_output_tokens - decomposition.usage.output_tokens
        if remaining_output < 1:
            raise BudgetExceededError(
                "Plan-and-Budget decomposition exhausted the output-token budget"
            )
        assessment_system = _assessment_system(self.config.domain)
        assessment_prompt = _assessment_prompt(state.query, steps)
        assessment_response_format = _assessment_response_format(
            len(steps),
            use_json_schema=ModelCapability.JSON_SCHEMA in backend.capabilities,
        )
        assessment = backend.generate(
            GenerationRequest(
                prompt=assessment_prompt,
                system=assessment_system,
                temperature=self.config.temperature,
                max_tokens=min(self.config.assessment_max_tokens, remaining_output),
                seed=context.seed + 1,
                response_format=assessment_response_format,
                metadata={
                    "method_id": self.method_id,
                    "release_id": self.release_id,
                    "stage": "credit_assessment",
                    "upstream_revision": P_AND_B_UPSTREAM_REVISION,
                },
            )
        )
        level, credits, assessment_warnings = _parse_assessment(
            assessment.text,
            step_count=len(steps),
        )
        audit_warnings = (*decomposition_warnings, *assessment_warnings)
        record = CachedPlanBudgetRecord(
            record_id=state.task_id or state.state_hash,
            query=state.query,
            steps=steps,
            credits=credits,
            level=level,
            audit_warnings=audit_warnings,
        )
        allocations = allocate_tokens(
            record.credits,
            self.config.total_word_budget,
            self.config.schedule,
        )
        instruction = render_soft_budget_instruction(record, allocations)
        input_tokens = decomposition.usage.input_tokens + assessment.usage.input_tokens
        output_tokens = decomposition.usage.output_tokens + assessment.usage.output_tokens
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "live_decomposition",
                    {
                        "steps": steps,
                        "initial_level": self.config.initial_level,
                        "decomposition_request_sha256": content_hash(
                            {
                                "system": decomposition_system,
                                "prompt": decomposition_prompt,
                            }
                        ),
                        "gold_visible_to_optimizer": False,
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "live_credit_assessment",
                    {
                        "evaluated_level": level,
                        "credits": credits,
                        "credit_sum": sum(credits),
                        "assessment_request_sha256": content_hash(
                            {"system": assessment_system, "prompt": assessment_prompt}
                        ),
                        "assessment_response_format": assessment_response_format["type"],
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "soft_word_budget_allocation",
                    {
                        "schedule": self.config.schedule.value,
                        "total_budget": self.config.total_word_budget,
                        "budget_unit": "soft_words",
                        "allocation_tie_policy": "stable_source_index",
                        "allocations": allocations,
                        "decoder_hard_cap_applied": False,
                        "instruction": instruction,
                    },
                ),
            ),
            counters={
                "model_calls": 2,
                "optimizer_model_calls": 2,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "planned_steps": len(steps),
                "allocated_soft_words": sum(allocations),
            },
            warnings=audit_warnings,
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.APPEND_INSTRUCTION,
                    value=instruction,
                    target="instruction",
                    rationale="Plan-and-Budget live decomposition and weighted soft-word budgets",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"official_mit_generalized_adapter:junhongmit/P-and-B@{P_AND_B_UPSTREAM_REVISION}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "official_mit_controlled_generalization",
                "cached_decomposition": False,
                "live_decomposition": True,
                "budget_unit": "soft_words",
                "allocation_tie_policy": "stable_source_index",
                "numpy_tie_order_parity_claimed": False,
                "hard_decoder_cap": False,
                "gold_fields_visible_to_optimizer": False,
                "paper_result_parity_claimed": False,
            },
        )


def _decomposition_system(domain: str) -> str:
    # Faithfully preserves the released prompt's role, 2--5 step constraint,
    # hint requirement, and prohibition on exposing a solution.
    return (
        f"You are an experienced expert in {domain} and an exam question designer. "
        "Break a challenging problem into 2 to 5 high-level sub-questions. Each "
        "sub-question should build on the previous one, and each must include a "
        "brief Hint explaining its purpose. Do not reveal internal chain-of-thought "
        "or the final answer. Do not solve the problem. Output only numbered items "
        "in the form '1. ...\nHint: ...'."
    )


def _decomposition_prompt(query: str, *, level: int) -> str:
    return (
        "A student has presented the following problem:\n"
        f"Problem: {query}\n"
        f"Initial difficulty level: {level} out of 5\n"
        "Decomposed Sub-questions:"
    )


def _assessment_system(domain: str) -> str:
    # This is the released credit-assessment schema without dataset-specific
    # benchmark examples.  Omitting those examples is explicit in the adapter
    # identity and prevents cross-dataset gold or label leakage.
    return (
        f"You are an experienced expert in {domain} and an exam question designer. "
        "Evaluate the overall problem and the supplied sub-questions. Assign the "
        "problem an integer difficulty from 1 to 5 and assign each sub-question a "
        "positive integer credit. Credits must sum exactly to 100. Do not add, "
        "remove, or solve sub-questions. Return only one JSON object in exactly this "
        'shape: {"problem":{"evaluated_level":3},'
        '"1":{"evaluated_level":2,"credit":40},'
        '"2":{"evaluated_level":3,"credit":60}}. The numeric keys must be '
        "top-level siblings of 'problem', never children of 'problem'. Include one "
        "numeric key for every supplied sub-question and no other keys."
    )


def _assessment_prompt(query: str, steps: tuple[str, ...]) -> str:
    rendered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return f"Problem: {query}\nSub-questions:\n{rendered}\nOutput:"


def _assessment_response_format(
    step_count: int,
    *,
    use_json_schema: bool,
) -> dict[str, Any]:
    if not use_json_schema:
        return {"type": "json_object"}
    step_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["evaluated_level", "credit"],
        "properties": {
            "evaluated_level": {"type": "integer"},
            "credit": {"type": "integer"},
        },
    }
    properties: dict[str, Any] = {
        "problem": {
            "type": "object",
            "additionalProperties": False,
            "required": ["evaluated_level"],
            "properties": {"evaluated_level": {"type": "integer"}},
        }
    }
    properties.update({str(index): step_schema for index in range(1, step_count + 1)})
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "plan_and_budget_credit_assessment",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["problem", *(str(index) for index in range(1, step_count + 1))],
                "properties": properties,
            },
        },
    }


def _parse_decomposition(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(text, str) or not text.strip():
        raise AlgorithmOutputError("Plan-and-Budget decomposition returned empty text")
    matches = tuple(_NUMBERED_ITEM.finditer(text.strip()))
    steps = tuple(match.group("text").strip() for match in matches if match.group("text").strip())
    warnings: list[str] = []
    if not steps:
        # Matches the released preprocessing fallback exactly.
        steps = ("Directly solve the problem. Hint: None.",)
        warnings.append("decomposition_parse_fallback_to_direct_solve")
    elif len(steps) > 5:
        raise AlgorithmOutputError("Plan-and-Budget decomposition returned more than five steps")
    if not 2 <= len(steps) <= 5:
        warnings.append("generated_step_count_outside_prompt_intended_2_to_5")
    return steps, tuple(warnings)


def _parse_assessment(
    text: str,
    *,
    step_count: int,
) -> tuple[int, tuple[float, ...], tuple[str, ...]]:
    data = _json_object(text)
    problem = data.get("problem")
    if not isinstance(problem, dict):
        raise AlgorithmOutputError("Plan-and-Budget assessment lacks problem metadata")
    level = _integer(problem.get("evaluated_level"), label="problem evaluated_level")
    if not 1 <= level <= 5:
        raise AlgorithmOutputError("Plan-and-Budget evaluated problem level must be in [1, 5]")
    credits: list[float] = []
    for index in range(1, step_count + 1):
        item = data.get(str(index))
        if not isinstance(item, dict):
            raise AlgorithmOutputError(f"Plan-and-Budget assessment lacks sub-question {index}")
        sublevel = _integer(item.get("evaluated_level"), label=f"step {index} evaluated_level")
        credit = _integer(item.get("credit"), label=f"step {index} credit")
        if not 1 <= sublevel <= 5 or not 1 <= credit <= 100:
            raise AlgorithmOutputError(
                "Plan-and-Budget step level/credit is outside the released range"
            )
        credits.append(float(credit))
    if set(data) != {"problem", *(str(index) for index in range(1, step_count + 1))}:
        raise AlgorithmOutputError("Plan-and-Budget assessment returned unexpected step keys")
    warnings = () if sum(credits) == 100 else ("generated_credit_sum_not_100",)
    return level, tuple(credits), warnings


def _json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise AlgorithmOutputError("Plan-and-Budget assessment returned empty text")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise AlgorithmOutputError("Plan-and-Budget assessment did not return JSON")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AlgorithmOutputError("Plan-and-Budget assessment returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise AlgorithmOutputError("Plan-and-Budget assessment JSON must be an object")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlgorithmOutputError(f"Plan-and-Budget {label} must be an integer")
    return value
