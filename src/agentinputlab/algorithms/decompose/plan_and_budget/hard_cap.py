"""Budget-enforced P&B extension: observed subproblem calls and final synthesis.

This is a controlled extension, not the native one-call soft-word solver.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
    FrozenAgentState,
    InterventionContext,
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
    _schedule_weights,
    allocate_tokens,
)
from .regenerated_soft_budget import PlanAndBudgetRegeneratedSoftBudget

PAB_HARD_CAP_RELEASE = "plan.plan_and_budget.hard_cap_extension"
PAB_HARD_CAP_PROTOCOL = "agentinputlab.pab.credit_weighted_decode_caps.v1"


def allocate_hard_token_caps(credits: Sequence[float], total_tokens: int,
                             per_step_cap: int = 512,
                             schedule: str | DecaySchedule = DecaySchedule.CONSTANT) -> tuple[int, ...]:
    """Minimum one/step, then weighted extras, redistributing saturation only.

    Zero-credit steps keep their one-token minimum. If every positive-credit
    step saturates, unused allocation remains unused rather than inventing weight.
    This deterministic extension uses source-index ties, not NumPy tie parity.
    """
    if (type(total_tokens) is not int or total_tokens < 0
            or type(per_step_cap) is not int or per_step_cap < 1):
        raise ValidationError("hard token pool/cap must be nonnegative/positive integers")
    if isinstance(credits, (str, bytes)):
        raise ValidationError("hard allocation requires a credit sequence")
    values = tuple(credits)
    if (not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or not math.isfinite(sum(values)) or sum(values) <= 0):
        raise ValidationError("hard allocation requires finite nonnegative credits with positive mass")
    decay = DecaySchedule.parse(schedule)
    weights = tuple(v*w for v,w in zip(values, _schedule_weights(decay,len(values)),strict=True))
    if total_tokens < len(values):
        raise BudgetExceededError("hard allocation needs at least one output token per subproblem")
    caps = [1]*len(values)
    remaining = min(total_tokens,len(values)*per_step_cap)-len(values)
    active = [i for i,w in enumerate(weights) if w > 0 and caps[i] < per_step_cap]
    while remaining and active:
        additions = allocate_tokens(tuple(weights[i] for i in active),remaining)
        spent = 0
        for i, proposed in zip(active,additions,strict=True):
            granted = min(proposed,per_step_cap-caps[i])
            caps[i] += granted
            spent += granted
        assert spent > 0
        remaining -= spent
        active = [i for i in active if caps[i] < per_step_cap]
    return tuple(caps)


class _HardCapPreparation(PlanAndBudgetRegeneratedSoftBudget):
    # Original preparation prompts/parser; distinguish the new execution identity.
    release_id = PAB_HARD_CAP_RELEASE

    @staticmethod
    def _backend(context):
        backend = PlanAndBudgetRegeneratedSoftBudget._backend(context)

        class RequestedCapGuard:
            capabilities = backend.capabilities

            def fingerprint(self):
                return backend.fingerprint()

            def generate(self,request):
                response = backend.generate(request)
                if response.usage.output_tokens > request.max_tokens:
                    raise BudgetExceededError("hard-cap P&B preparation exceeded the requested output cap")
                return response

        return RequestedCapGuard()


class PlanAndBudgetHardCapExtension:
    """Live or exact-cache preparation, capped sequential solving and synthesis."""
    method_id = "plan_and_budget"
    release_id = PAB_HARD_CAP_RELEASE
    family = SemanticFamily.DECOMPOSE

    def __init__(self, *, output_format: str,
                 records: Mapping[str,CachedPlanBudgetRecord] | PlanAndBudgetCache | None = None,
                 domain: str = "general question answering", initial_level: int = 3,
                 schedule: str | DecaySchedule = DecaySchedule.CONSTANT,
                 decomposition_max_tokens: int = 512, assessment_max_tokens: int = 512,
                 total_step_tokens: int = 512, per_step_max_tokens: int = 512, final_max_tokens: int = 512,
                 temperature: float = 0.0, profile_kind: str = "controlled_adapter"):
        if not isinstance(output_format,str) or not output_format.strip():
            raise ValidationError("hard-cap P&B requires an explicit final output format")
        if type(initial_level) is not int or not 1 <= initial_level <= 5:
            raise ValidationError("hard-cap P&B initial level must be an integer in [1,5]")
        for cap in (decomposition_max_tokens,assessment_max_tokens,total_step_tokens,per_step_max_tokens,final_max_tokens):
            if type(cap) is not int or cap < 1:
                raise ValidationError("hard-cap P&B generation caps must be positive integers")
        GenerationRequest(prompt="configuration",max_tokens=per_step_max_tokens,temperature=temperature)
        if profile_kind not in {"controlled_adapter","factorial_adapter"}:
            raise ValidationError("hard-cap P&B is a controlled/factorial extension")
        self.output_format,self.per_step_max_tokens,self.final_max_tokens = output_format,per_step_max_tokens,final_max_tokens
        self.total_step_tokens = total_step_tokens
        self.temperature,self.profile_kind = temperature,profile_kind
        self.schedule = DecaySchedule.parse(schedule)
        self.cache_sha256 = records.artifact_sha256 if isinstance(records,PlanAndBudgetCache) else None
        self.cache_verification = records.verification if isinstance(records,PlanAndBudgetCache) else None
        if isinstance(records,PlanAndBudgetCache):
            records = records.records
        if records is not None and not isinstance(records,Mapping):
            raise ArtifactError("hard-cap P&B cache must be keyed by exact task ID")
        if records is not None and (not records or any(
            not isinstance(v,CachedPlanBudgetRecord) or k != v.record_id or not 1 <= len(v.steps) <= 5
            or any(not isinstance(s,str) or not s.strip() for s in v.steps)
            for k,v in records.items())):
            raise ArtifactError("hard-cap P&B requires solver-only records with one to five nonempty steps")
        self.records = MappingProxyType(dict(records)) if records is not None else None
        self.preparation = _HardCapPreparation(domain=domain,initial_level=initial_level,schedule=self.schedule,
            decomposition_max_tokens=decomposition_max_tokens,assessment_max_tokens=assessment_max_tokens,
            temperature=temperature)

    def run(self,state: FrozenAgentState,context: InterventionContext) -> CompositeResult:
        if not state.query.strip():
            raise ValidationError("hard-cap P&B requires a non-empty query")
        if state.messages or state.tool_names:
            raise CapabilityError("hard-cap P&B supports visible QA context, not tool/history environments")
        backend = context.backend
        if not isinstance(backend,LLMBackend):
            raise CapabilityError("hard-cap P&B requires an injected LLMBackend")
        if ModelCapability.CHAT not in backend.capabilities:
            raise CapabilityError("hard-cap P&B requires chat capability")
        calls = inputs = outputs = 0
        events = []
        if self.records is not None:
            record = self.records.get(state.task_id)
            if record is None or record.query != state.query:
                raise ArtifactError("hard-cap P&B cached ID/query mismatch; no live fallback")
            if state.budget.max_model_calls < len(record.steps)+1 or state.budget.max_output_tokens < len(record.steps)+1:
                raise BudgetExceededError("hard-cap P&B budget cannot cover all cached steps and final synthesis")
            warnings = record.audit_warnings
        else:
            if state.budget.max_model_calls < 4 or state.budget.max_output_tokens < 4:
                raise BudgetExceededError("live hard-cap P&B needs preparation, at least one step and a final call")
            prep_state = replace(state,budget=replace(state.budget,
                max_model_calls=2,max_output_tokens=state.budget.max_output_tokens-2))
            patch = self.preparation.propose(prep_state,context)
            decomposition,assessment = patch.trace.events[:2]
            record = CachedPlanBudgetRecord(state.task_id or state.state_hash,state.query,
                tuple(decomposition.payload["steps"]),tuple(assessment.payload["credits"]),
                assessment.payload["evaluated_level"],patch.trace.warnings)
            calls,inputs,outputs = 2,patch.trace.counters["input_tokens"],patch.trace.counters["output_tokens"]
            events.extend(patch.trace.events[:2])
            warnings = patch.trace.warnings
            if calls+len(record.steps)+1 > state.budget.max_model_calls:
                raise BudgetExceededError("generated plan exceeds remaining call budget; no steps silently dropped")
        remaining = state.budget.max_output_tokens-outputs
        count = len(record.steps)
        if remaining < count+1:
            raise BudgetExceededError("remaining output budget cannot cover all steps and final synthesis")
        final_reservation = min(self.final_max_tokens,remaining-count)
        pool = min(self.total_step_tokens,remaining-final_reservation,count*self.per_step_max_tokens)
        caps = allocate_hard_token_caps(record.credits,pool,self.per_step_max_tokens,self.schedule)
        events.append(TraceEvent(TracePhase.PREPARE,"hard_decoder_allocation",{
            "protocol":PAB_HARD_CAP_PROTOCOL,"steps":record.steps,"credits":record.credits,
            "schedule":self.schedule.value,"caps":caps,"pool":pool,"unallocated_pool":pool-sum(caps),
            "final_reservation":final_reservation,"budget_unit":"output_tokens",
            "minimum_per_step":1,"tie_policy":"stable_source_index","unused_step_cap_redistributed":False}))
        reference = "\n\n".join(
            (f"[{b.block_id}] " if b.block_id else "")+b.text
            +(f"\nSource: {b.source}" if b.source else "")
            +f"\nSupport ID: {b.metadata.get('official_support_id',b.block_id)}" for b in state.context)
        answers = []

        def generate(prompt,cap,stage):
            nonlocal calls,inputs,outputs
            if calls >= state.budget.max_model_calls or cap > state.budget.max_output_tokens-outputs:
                raise BudgetExceededError("hard-cap P&B cannot reserve the next requested generation")
            request = GenerationRequest(prompt=prompt,system=state.instruction,temperature=self.temperature,
                max_tokens=cap,seed=context.seed+calls,metadata={"release_id":self.release_id,"stage":stage,
                    "hard_cap_protocol":PAB_HARD_CAP_PROTOCOL})
            response = backend.generate(request)
            calls += 1
            inputs += response.usage.input_tokens
            outputs += response.usage.output_tokens
            if response.usage.output_tokens > cap:
                raise BudgetExceededError("hard-cap P&B provider exceeded the requested decoder output cap")
            if not response.text.strip():
                raise AlgorithmOutputError(f"hard-cap P&B {stage} returned empty text; no retry")
            events.append(TraceEvent(TracePhase.OBSERVE,stage,{
                "request_hash":request.request_hash,"max_tokens":cap,"output_tokens":response.usage.output_tokens,
                "finish_reason":response.finish_reason,"seed":request.seed,"answer":response.text}))
            return response.text

        for index,(step,cap) in enumerate(zip(record.steps,caps,strict=True),1):
            prior = "\n\n".join(f"Sub-question {i}: {s}\nAnswer: {a}" for i,(s,a) in
                enumerate(zip(record.steps,answers,strict=False),1))
            prompt = (f"Original question: {state.query}\n\nVisible references:\n{reference}\n\n"
                f"Previous sub-answers:\n{prior}\n\nCurrent sub-question {index}: {step}\n"
                "Return a concise answer to this sub-question using the visible information. Do not invent evidence.")
            answers.append(generate(prompt,cap,f"hard_capped_subproblem_{index}"))
        prior = "\n\n".join(f"Sub-question {i}: {s}\nAnswer: {a}" for i,(s,a) in
            enumerate(zip(record.steps,answers,strict=True),1))
        final = generate(f"Original question: {state.query}\n\nVisible references:\n{reference}\n\n"
            f"Sub-answers:\n{prior}\n\nSynthesize the final answer to the original question.\n{self.output_format}",
            min(self.final_max_tokens,state.budget.max_output_tokens-outputs),"hard_capped_final_synthesis")
        return CompositeResult(method_id=self.method_id,release_id=self.release_id,source_state_hash=state.state_hash,
            final_answer=final,evidence=state.context,trace=ExecutionTrace(method_id=self.method_id,release_id=self.release_id,
                events=tuple(events),counters={"model_calls":calls,"optimizer_model_calls":2 if self.records is None else 0,
                    "subproblem_model_calls":count,"input_tokens":inputs,"output_tokens":outputs},warnings=warnings),
            implementation_identity=PAB_HARD_CAP_PROTOCOL,profile_kind=self.profile_kind,
            metadata={"protocol":PAB_HARD_CAP_PROTOCOL,"source_preparation_revision":P_AND_B_UPSTREAM_REVISION,
                "cached_decomposition":self.records is not None,"cache_artifact_sha256":self.cache_sha256,
                "cache_verification":self.cache_verification,"per_step_hard_token_cap":True,"step_caps":caps,
                "total_step_tokens":self.total_step_tokens,
                "per_step_max_tokens":self.per_step_max_tokens,"final_max_tokens":self.final_max_tokens,
                "actual_model_calls":calls,"actual_output_tokens":outputs,"final_call_in_shared_budget":True,
                "allocation_unit":"output_tokens_not_words","official_native_parity_claimed":False,
                "paper_result_parity_claimed":False,"automatic_retry":False,
                "official_scoring":"caller_owned_independent_scorer","gold_fields_visible":False})


__all__ = ["PAB_HARD_CAP_PROTOCOL","PAB_HARD_CAP_RELEASE","PlanAndBudgetHardCapExtension","allocate_hard_token_caps"]
