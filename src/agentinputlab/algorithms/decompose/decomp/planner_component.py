"""One native DecomP planner transition, explicitly not its full solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
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
from agentinputlab.observability import external_usage_counters, normalize_external_usage

from .worker_bridge import DECOMP_REVISION, DecompWorkerRequest, decomp_context_hash

DECOMP_PLANNER_SCHEMA = "agentinputlab.decomp.planner.v1"
DECOMP_PLANNER_RELEASE = "plan.decomp.planner_component"
DECOMP_PLANNER_MODE = "controlled_native_planner_callbacks.v1"


@dataclass(frozen=True, slots=True)
class DecompPlannerRequest(DecompWorkerRequest):
    schema_version: str = DECOMP_PLANNER_SCHEMA

    def __post_init__(self):
        DecompWorkerRequest.__post_init__(self)
        if self.max_steps != 1 or self.schema_version != DECOMP_PLANNER_SCHEMA:
            raise ValidationError("DecomP planner requires its own schema and one transition")
        if self.upstream_revision != DECOMP_REVISION:
            raise ValidationError("DecomP planner requires the pinned source revision")


class DecompPlannerComponent:
    """Attach the unexecuted native next subquestion to the original reader input."""

    method_id = "decomp"
    release_id = DECOMP_PLANNER_RELEASE
    family = SemanticFamily.DECOMPOSE

    def __init__(self, *, config_artifact_id: str, profile_kind="controlled_adapter"):
        if not isinstance(config_artifact_id, str) or not config_artifact_id.strip():
            raise ValidationError("DecomP planner requires an explicit native config artifact")
        if profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("DecomP planner requires a controlled/factorial profile")
        self.config_artifact_id, self.profile_kind = config_artifact_id, profile_kind

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("DecomP planner query must be non-empty")
        if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
            raise BudgetExceededError("DecomP planner has no generation budget")
        runtime = context.external_runtime
        if (
            runtime is None
            or not callable(getattr(runtime, "run", None))
            or not callable(getattr(runtime, "fingerprint", None))
        ):
            raise ExternalDependencyError("DecomP planner requires its dedicated native runtime")
        identity = runtime.fingerprint()
        if identity.get("backend_binding_mode") != DECOMP_PLANNER_MODE:
            raise ValidationError("full DecomP runtime cannot satisfy the planner component")
        if identity.get("upstream_revision") != DECOMP_REVISION:
            raise ValidationError("DecomP planner runtime revision differs")
        input_bound = identity.get("per_call_input_token_bound")
        if state.budget.max_cost_usd is not None or (
                state.budget.max_context_tokens is not None and
                (type(input_bound) is not int or input_bound < 1)):
            raise ValidationError("native DecomP finite input-token/cost ceilings require pre-call estimators")
        if (state.budget.max_context_tokens is not None
                and input_bound > state.budget.max_context_tokens):
            raise BudgetExceededError("DecomP planner input budget cannot reserve the native upper bound")
        backend = context.backend
        if backend is None or not callable(getattr(backend, "fingerprint", None)):
            raise ExternalDependencyError("DecomP planner requires an explicitly bound backend")
        backend_binding = dict(backend.fingerprint())
        if content_hash(identity.get("backend")) != content_hash(backend_binding):
            raise ValidationError("DecomP planner runtime backend differs from context")
        paras = tuple(block.text for block in state.context)
        ids = tuple(block.block_id for block in state.context)
        request = DecompPlannerRequest(
            case_id=state.task_id or state.state_hash,
            question=state.query,
            paras=paras,
            paragraph_ids=ids,
            context_hash=decomp_context_hash(ids, paras),
            config_artifact_id=self.config_artifact_id,
            backend_binding=backend_binding,
            max_steps=1,
            execution_budget=state.budget,
            seed=context.seed,
        )
        response = runtime.run(request)
        request.raise_for_external_status(response)
        if not isinstance(response, Mapping):
            raise AlgorithmOutputError("invalid DecomP planner response")
        expected = {
            "schema_version": DECOMP_PLANNER_SCHEMA,
            "profile_id": self.release_id,
            "case_id": request.case_id,
            "upstream_revision": DECOMP_REVISION,
            "context_hash": request.context_hash,
            "component_only": True,
            "router_executed": False,
            "qa_executed": False,
        }
        if any(response.get(k) != v for k, v in expected.items()) or "final_answer" in response:
            raise AlgorithmOutputError("DecomP planner identity/boundary differs")
        if tuple(response.get("paragraph_ids", ())) != ids:
            raise AlgorithmOutputError("DecomP planner changed paragraph identity")
        selected = response.get("selected")
        if not isinstance(selected, Mapping) or not isinstance(selected.get("question"), str):
            raise AlgorithmOutputError("DecomP planner next question is unavailable")
        question = selected["question"].strip()
        if not question or question == "[EOQ]":
            raise AlgorithmOutputError(
                "native DecomP planner returned no substantive next question"
            )
        candidates = response.get("candidates")
        if not isinstance(candidates, (list, tuple)) or selected not in candidates:
            raise AlgorithmOutputError("selected DecomP plan is outside native candidates")
        artifacts = response.get("native_artifacts")
        if (
            not isinstance(artifacts, Mapping)
            or artifacts.get("config_artifact_id") != self.config_artifact_id
            or artifacts.get("backend_binding_mode") != DECOMP_PLANNER_MODE
        ):
            raise AlgorithmOutputError("DecomP planner config artifact identity differs")
        usage = normalize_external_usage(response.get("usage", {}))
        if usage.model_calls != 1:
            raise AlgorithmOutputError("planner must consume exactly one native generation")
        if content_hash(runtime.fingerprint()) != content_hash(identity):
            raise AlgorithmOutputError("DecomP planner runtime identity changed")
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.APPEND_INSTRUCTION,
                    value="Native DecomP next subquestion (unexecuted):\n" + question,
                    target="instruction",
                    rationale="first native planner transition; router/QA are not executed",
                ),
            ),
            implementation_identity=f"native_decomp_first_planner_transition@{DECOMP_REVISION}",
            profile_kind=self.profile_kind,
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                events=(
                    TraceEvent(
                        TracePhase.PROPOSE,
                        "native_planner_component",
                        {
                            "selected": selected,
                            "candidates": candidates,
                            "native_artifacts": response.get("native_artifacts", {}),
                            "context_hash": request.context_hash,
                            "paragraph_ids": ids,
                            "runtime_identity_hash": content_hash(identity),
                        },
                    ),
                ),
                counters=external_usage_counters(usage),
            ),
            metadata={
                "component_only": True,
                "full_system": False,
                "selected": selected,
                "native_artifacts": response.get("native_artifacts", {}),
                "router_executed": False,
                "qa_executed": False,
                "solver_executed": False,
                "optimizer_input_projection": "query_and_exact_visible_context",
                "gold_fields_visible_to_optimizer": False,
                "paper_result_parity_claimed": False,
            },
        )


__all__ = [
    "DECOMP_PLANNER_MODE",
    "DECOMP_PLANNER_RELEASE",
    "DECOMP_PLANNER_SCHEMA",
    "DecompPlannerComponent",
    "DecompPlannerRequest",
]
