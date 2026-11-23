"""Complete P&B preparation, native weighted prompt and observed answer call."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    CapabilityError,
    CompositeResult,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    Message,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)

from .cached_soft_budget import (
    P_AND_B_UPSTREAM_REVISION,
    CachedPlanBudgetRecord,
    DecaySchedule,
    PlanAndBudgetCache,
)
from .regenerated_soft_budget import PlanAndBudgetRegeneratedSoftBudget


class PlanAndBudgetFull:
    """Run a complete answer-producing workflow, not an input patch.

    Cached mode requires an exact ID/query match. Live mode regenerates two
    solver-only fields without labels. Both use the original native weighted
    reasoning prompt and level-dependent soft word budgets, followed by one
    generation. The caller retains the benchmark's independent official scorer.
    """

    method_id = "plan_and_budget"
    release_id = "plan.plan_and_budget.full"
    family = SemanticFamily.DECOMPOSE

    def __init__(
        self, *, output_format: str,
        records: Mapping[str, CachedPlanBudgetRecord] | PlanAndBudgetCache | None = None,
        domain: str = "general question answering", initial_level: int = 3,
        schedule: str | DecaySchedule = DecaySchedule.CONSTANT,
        decomposition_max_tokens: int = 512, assessment_max_tokens: int = 512,
        max_solver_tokens: int = 512, temperature: float = 0.0,
    ):
        if not isinstance(output_format, str) or not output_format.strip():
            raise ValidationError("P&B full requires an explicit output format")
        if (type(max_solver_tokens) is not int or max_solver_tokens < 1
                or type(temperature) not in (int, float)
                or not math.isfinite(temperature) or temperature < 0):
            raise ValidationError("P&B solver token cap/temperature is invalid")
        self.output_format, self.max_solver_tokens = output_format, max_solver_tokens
        self.temperature = temperature
        self.schedule = DecaySchedule.parse(schedule)
        self.cache_sha256 = records.artifact_sha256 if isinstance(records, PlanAndBudgetCache) else None
        self.cache_verification = records.verification if isinstance(records, PlanAndBudgetCache) else None
        if isinstance(records, PlanAndBudgetCache):
            records = records.records
        if records is not None and (not records or any(
            not isinstance(value, CachedPlanBudgetRecord) or key != value.record_id
            for key, value in records.items()
        )):
            raise ArtifactError("P&B full requires solver-only records keyed by exact ID")
        self.records = MappingProxyType(dict(records)) if records is not None else None
        self.preparation = PlanAndBudgetRegeneratedSoftBudget(
            domain=domain, initial_level=initial_level, schedule=self.schedule,
            decomposition_max_tokens=decomposition_max_tokens,
            assessment_max_tokens=assessment_max_tokens, temperature=temperature,
        )

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        from agentinputlab.integrations.full_architecture.plan_and_budget import (
            PAB_NATIVE_PROMPT_MODE,
            PlanBudgetPromptRequest,
        )

        if not state.query.strip():
            raise ValidationError("P&B full requires a non-empty query")
        if state.messages or state.tool_names:
            raise CapabilityError("P&B native solver accepts visible QA context, not a tool/history loop")
        needed_calls = 1 if self.records is not None else 3
        if (state.budget.max_model_calls < needed_calls
                or state.budget.max_output_tokens < needed_calls):
            raise BudgetExceededError(f"P&B full requires at least {needed_calls} model calls/tokens")
        backend = context.backend
        if not isinstance(backend, LLMBackend):
            raise ExternalDependencyError("P&B full requires an injected LLMBackend")
        if ModelCapability.CHAT not in backend.capabilities:
            raise CapabilityError("P&B full requires the controlled chat backend binding")
        runtime = context.external_runtime
        if (runtime is None or not callable(getattr(runtime, "run", None))
                or not callable(getattr(runtime, "fingerprint", None))):
            raise ExternalDependencyError("P&B full requires PlanAndBudgetNativePromptRuntime")
        identity = runtime.fingerprint()
        if (identity.get("binding_mode") != PAB_NATIVE_PROMPT_MODE
                or identity.get("upstream_revision") != P_AND_B_UPSTREAM_REVISION):
            raise ArtifactError("P&B full native runtime binding/revision differs")
        events, warnings = (), ()
        input_tokens = output_tokens = prep_calls = 0
        if self.records is not None:
            record = self.records.get(state.task_id)
            if record is None or record.query != state.query:
                raise ArtifactError("P&B full cached ID/query does not match; no live fallback")
            warnings = record.audit_warnings
        else:
            # Reserve at least one output token/call for the solver. Actual
            # provider usage, including failed preparation, is ledger-observed.
            prep_state = replace(state, budget=replace(
                state.budget, max_model_calls=state.budget.max_model_calls - 1,
                max_output_tokens=state.budget.max_output_tokens - 1,
            ))
            patch = self.preparation.propose(prep_state, context)
            decomposition, assessment = patch.trace.events[:2]
            record = CachedPlanBudgetRecord(
                record_id=state.task_id or state.state_hash, query=state.query,
                steps=tuple(decomposition.payload["steps"]),
                credits=tuple(assessment.payload["credits"]),
                level=assessment.payload["evaluated_level"],
                audit_warnings=patch.trace.warnings,
            )
            events, warnings = patch.trace.events[:2], patch.trace.warnings
            input_tokens = patch.trace.counters["input_tokens"]
            output_tokens = patch.trace.counters["output_tokens"]
            prep_calls = 2
        envelope = runtime.run(PlanBudgetPromptRequest(
            record=record, instruction=state.instruction,
            reference="\n\n".join(
                (f"[{block.block_id}] {block.text}" if block.block_id else block.text)
                + f"\nSupport ID: {block.metadata.get('official_support_id', block.block_id)}"
                + (f"\nSource: {block.source}" if block.source else "")
                for block in state.context
            ),
            output_format=self.output_format, schedule=self.schedule.value,
        ))
        if not isinstance(envelope, Mapping) or envelope.get("identity") != identity:
            raise ArtifactError("P&B native prompt response identity differs")
        remaining = state.budget.max_output_tokens - output_tokens
        if remaining < 1:
            raise BudgetExceededError("P&B preparation exhausted the solver output budget")
        request = GenerationRequest(
            messages=tuple(Message(**message) for message in envelope["messages"]),
            temperature=self.temperature, max_tokens=min(self.max_solver_tokens, remaining),
            seed=context.seed + prep_calls,
            metadata={"release_id": self.release_id, "stage": "native_weighted_solver",
                      "upstream_revision": P_AND_B_UPSTREAM_REVISION},
        )
        response = backend.generate(request)
        if not isinstance(response.text, str) or not response.text.strip():
            raise AlgorithmOutputError("P&B solver returned an empty final answer; no retry")
        if response.usage.output_tokens > request.max_tokens:
            raise BudgetExceededError("P&B solver usage exceeded its requested output cap")
        return CompositeResult(
            method_id=self.method_id, release_id=self.release_id,
            source_state_hash=state.state_hash, final_answer=response.text,
            evidence=state.context,
            trace=ExecutionTrace(
                method_id=self.method_id, release_id=self.release_id,
                events=(*events, TraceEvent(TracePhase.PREPARE, "native_soft_word_allocation", {
                    "allocations": envelope["allocations"],
                    "total_soft_words": envelope["soft_word_budget"],
                    "schedule": self.schedule.value, "tie_policy": "native_numpy_argsort",
                }), TraceEvent(TracePhase.OBSERVE, "solver_answer", {
                    "request_hash": request.request_hash, "finish_reason": response.finish_reason,
                    "max_output_tokens": request.max_tokens, "seed": request.seed,
                })),
                counters={"model_calls": prep_calls + 1, "optimizer_model_calls": prep_calls,
                          "input_tokens": input_tokens + response.usage.input_tokens,
                          "output_tokens": output_tokens + response.usage.output_tokens},
                warnings=warnings,
            ),
            implementation_identity=f"native_weighted_solver:P-and-B@{P_AND_B_UPSTREAM_REVISION}",
            profile_kind="controlled_adapter",
            metadata={
                "cached_decomposition": self.records is not None,
                "cache_artifact_sha256": self.cache_sha256,
                "cache_verification": self.cache_verification,
                "native_prompt_runtime": identity,
                "solver_calls": 1, "per_step_model_calls": False,
                "budget_unit": "soft_words", "per_step_hard_token_cap": False,
                "decoder_output_cap": request.max_tokens,
                "native_sampling_defaults_preserved": False,
                "sampling_policy": "explicit_caller_temperature_seed_cap_and_backend_defaults",
                "native_retry_policy_preserved": False,
                "transport_policy": "one_attempt_no_implicit_retry",
                "native_gold_evaluator_executed": False,
                "official_scoring": "caller_owned_separate_benchmark_scorer",
                "gold_fields_visible_to_optimizer": False,
                "paper_result_parity_claimed": False,
            },
        )


__all__ = ["PlanAndBudgetFull"]
