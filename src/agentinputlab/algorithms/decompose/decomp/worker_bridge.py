"""Typed bridge for the unmodified DecomP full-system control flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CompositeResult,
    ExecutionBudget,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.observability import external_usage_counters, normalize_external_usage

DECOMP_REVISION = "8b157f6fe9ff5676ae51bd054f1df7b015adf278"
DECOMP_PROFILE_ID = "plan.decomp.full"
DECOMP_WORKER_SCHEMA = "agentinputlab.decomp.worker.v2"


def decomp_context_hash(
    paragraph_ids: tuple[str, ...],
    paras: tuple[str, ...],
) -> str:
    """Hash the exact paragraph identity and text passed to upstream DecomP."""

    return content_hash({"paragraph_ids": paragraph_ids, "paras": paras})


@dataclass(frozen=True, slots=True)
class DecompWorkerRequest:
    case_id: str
    question: str
    paras: tuple[str, ...]
    paragraph_ids: tuple[str, ...]
    context_hash: str
    config_artifact_id: str
    backend_binding: Mapping[str, Any]
    max_steps: int
    upstream_revision: str = DECOMP_REVISION
    schema_version: str = DECOMP_WORKER_SCHEMA
    execution_budget: ExecutionBudget | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.case_id or not self.question:
            raise ValidationError("DecomP worker request requires case_id and question")
        if not self.config_artifact_id:
            raise ValidationError("DecomP full requires a pinned config artifact")
        if self.max_steps < 1:
            raise ValidationError("DecomP max_steps must be positive")
        if self.execution_budget is not None and not isinstance(self.execution_budget, ExecutionBudget):
            raise ValidationError("DecomP execution_budget must be ExecutionBudget")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("DecomP seed must be non-negative")
        paras = tuple(self.paras)
        paragraph_ids = tuple(self.paragraph_ids)
        if len(paras) != len(paragraph_ids):
            raise ValidationError("DecomP paras and paragraph_ids must have equal length")
        if any(not isinstance(value, str) or not value for value in paras):
            raise ValidationError("DecomP paras must contain non-empty text")
        if any(not isinstance(value, str) or not value.strip() for value in paragraph_ids):
            raise ValidationError("DecomP paragraph_ids must contain non-empty IDs")
        if len(paragraph_ids) != len(set(paragraph_ids)):
            raise ValidationError("DecomP paragraph_ids must be unique")
        expected_context_hash = decomp_context_hash(paragraph_ids, paras)
        if self.context_hash != expected_context_hash:
            raise ValidationError("DecomP context_hash does not match paras and paragraph_ids")
        object.__setattr__(self, "paras", paras)
        object.__setattr__(self, "paragraph_ids", paragraph_ids)
        object.__setattr__(self, "backend_binding", freeze_mapping(self.backend_binding))

    def external_reservation(self) -> Mapping[str, int]:
        """A successful full DecomP response must account for a model call."""

        return {"model_calls": 1}

    def raise_for_external_status(self, result: object) -> None:
        """Let the monitored boundary settle actual usage before raising failure."""
        if isinstance(result, Mapping) and result.get("status") == "error":
            raise ExternalDependencyError("DecomP native worker failed with " + str(result.get("failure_code", "INVALID_RESPONSE")))


@dataclass(frozen=True, slots=True)
class DecompGraphNode:
    node_id: str
    question: str
    answer: str = ""
    handler: str = ""
    depth: int = 0


@dataclass(frozen=True, slots=True)
class DecompParticipantCall:
    index: int
    participant: str
    input_state_id: str
    output_state_id: str
    command: str


@dataclass(frozen=True, slots=True)
class DecompFrontierEvent:
    index: int
    state_id: str
    score: float
    next_command: str
    action: str


@dataclass(frozen=True, slots=True)
class DecompWorkerResponse:
    case_id: str
    profile_id: str
    upstream_revision: str
    paragraph_ids: tuple[str, ...]
    context_hash: str
    nodes: tuple[DecompGraphNode, ...]
    dependency_edges: tuple[tuple[str, str], ...]
    participant_calls: tuple[DecompParticipantCall, ...]
    frontier: tuple[DecompFrontierEvent, ...]
    reasoning_chain: str
    final_answer: str
    termination: str
    component_only: bool = False
    usage: Mapping[str, int | float] = field(default_factory=dict)
    schema_version: str = DECOMP_WORKER_SCHEMA
    native_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "paragraph_ids", tuple(self.paragraph_ids))
        object.__setattr__(self, "dependency_edges", tuple(self.dependency_edges))
        object.__setattr__(self, "participant_calls", tuple(self.participant_calls))
        object.__setattr__(self, "frontier", tuple(self.frontier))
        object.__setattr__(self, "usage", freeze_mapping(self.usage))


@runtime_checkable
class DecompWorkerRuntime(Protocol):
    def run(self, request: DecompWorkerRequest) -> DecompWorkerResponse | Mapping[str, Any]: ...

    def fingerprint(self) -> Mapping[str, Any]: ...


def _validate_acyclic(node_ids: set[str], edges: tuple[tuple[str, str], ...]) -> None:
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        if source not in node_ids or target not in node_ids:
            raise AlgorithmOutputError("DecomP dependency edge references an unknown node")
        children[source].append(target)
        indegree[target] += 1
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(node_ids):
        raise AlgorithmOutputError("DecomP typed dependency graph contains a cycle")


def validate_decomp_full_response(response: DecompWorkerResponse) -> None:
    """Reject planner-only or lossy responses under the full-system release ID."""

    if response.schema_version != DECOMP_WORKER_SCHEMA:
        raise AlgorithmOutputError(f"DecomP worker schema mismatch: {response.schema_version!r}")
    if response.profile_id != DECOMP_PROFILE_ID:
        raise AlgorithmOutputError("DecomP worker returned the wrong profile identity")
    if response.upstream_revision != DECOMP_REVISION:
        raise AlgorithmOutputError("DecomP worker returned an unpinned upstream revision")
    if len(response.paragraph_ids) != len(set(response.paragraph_ids)) or any(
        not isinstance(value, str) or not value for value in response.paragraph_ids
    ):
        raise AlgorithmOutputError("DecomP worker returned invalid paragraph identities")
    if (
        not isinstance(response.context_hash, str)
        or len(response.context_hash) != 64
        or any(character not in "0123456789abcdef" for character in response.context_hash)
    ):
        raise AlgorithmOutputError("DecomP worker returned an invalid context_hash")
    if response.component_only:
        raise AlgorithmOutputError("planner-component output cannot satisfy plan.decomp.full")
    if not response.nodes:
        raise AlgorithmOutputError("DecomP full response lost native decomposition nodes")
    if not response.participant_calls:
        raise AlgorithmOutputError("DecomP full response lost participant routing")
    if not response.frontier:
        raise AlgorithmOutputError("DecomP full response lost best-first frontier events")
    if not response.reasoning_chain:
        raise AlgorithmOutputError("DecomP full response requires the native reasoning chain")
    if not response.final_answer.strip():
        raise AlgorithmOutputError("DecomP full response requires a terminal final answer")
    if not response.termination:
        raise AlgorithmOutputError("DecomP full response requires a termination reason")
    node_ids = [node.node_id for node in response.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise AlgorithmOutputError("DecomP response contains duplicate node IDs")
    if any(node.depth < 0 for node in response.nodes):
        raise AlgorithmOutputError("DecomP node depth cannot be negative")
    if tuple(event.index for event in response.frontier) != tuple(range(len(response.frontier))):
        raise AlgorithmOutputError("DecomP frontier event indices must be contiguous")
    if tuple(call.index for call in response.participant_calls) != tuple(
        range(len(response.participant_calls))
    ):
        raise AlgorithmOutputError("DecomP participant-call indices must be contiguous")
    _validate_acyclic(set(node_ids), response.dependency_edges)
    usage = normalize_external_usage(response.usage)
    if usage.model_calls < 1:
        raise AlgorithmOutputError("DecomP full response must account for its model calls")


class DecompFullBridge:
    """Run pinned full DecomP and preserve its terminal answer as a composite result."""

    method_id = "decomp"
    release_id = DECOMP_PROFILE_ID
    family = SemanticFamily.DECOMPOSE

    def __init__(
        self,
        runtime: DecompWorkerRuntime | None = None,
        *,
        config_artifact_id: str,
        max_steps: int = 20,
    ) -> None:
        self.runtime = runtime
        self.config_artifact_id = config_artifact_id
        self.max_steps = max_steps

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        runtime = self._resolve_runtime(context)
        if runtime is None:
            raise ExternalDependencyError(
                "plan.decomp.full requires the pinned external DecomP worker; "
                "the core package intentionally has no simplified fallback"
            )
        runtime_identity = self._validate_runtime_identity(runtime)
        controlled = runtime_identity.get("backend_binding_mode") == "controlled_native_completion_callbacks.v1"
        if controlled and (state.instruction or state.messages):
            raise ValidationError("native DecomP accepts query/context only; explicitly project instruction/history")
        input_bound = runtime_identity.get("per_call_input_token_bound")
        if controlled and (state.budget.max_cost_usd is not None or (
                state.budget.max_context_tokens is not None and
                (type(input_bound) is not int or input_bound < 1))):
            raise ValidationError("native DecomP finite input-token/cost ceilings require pre-call estimators")
        if (controlled and state.budget.max_context_tokens is not None
                and input_bound > state.budget.max_context_tokens):
            raise BudgetExceededError("DecomP input budget cannot reserve the native upper bound")
        backend_binding = (
            dict(context.backend.fingerprint())
            if context.backend is not None and hasattr(context.backend, "fingerprint")
            else {}
        )
        if controlled and "backend" in runtime_identity and content_hash(runtime_identity["backend"]) != content_hash(backend_binding):
            raise ValidationError("native DecomP runtime backend differs from the context backend")
        paras = tuple(block.text for block in state.context)
        paragraph_ids = tuple(block.block_id for block in state.context)
        if any(not paragraph_id.strip() for paragraph_id in paragraph_ids):
            raise ValidationError(
                "plan.decomp.full requires a non-empty block_id for every context paragraph"
            )
        context_hash = decomp_context_hash(paragraph_ids, paras)
        request = DecompWorkerRequest(
            case_id=state.task_id or state.state_hash,
            question=state.query,
            paras=paras,
            paragraph_ids=paragraph_ids,
            context_hash=context_hash,
            config_artifact_id=self.config_artifact_id,
            backend_binding=backend_binding,
            max_steps=self.max_steps,
            execution_budget=state.budget,
            seed=context.seed,
        )
        try:
            response = runtime.run(request)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ExternalDependencyError(
                "the pinned DecomP worker is missing an optional upstream dependency"
            ) from exc
        request.raise_for_external_status(response)
        if not isinstance(response, DecompWorkerResponse):
            raise AlgorithmOutputError("DecomP worker returned an invalid response envelope")
        if response.case_id != request.case_id:
            raise AlgorithmOutputError("DecomP worker response case_id mismatch")
        if response.paragraph_ids != request.paragraph_ids:
            raise AlgorithmOutputError("DecomP worker response paragraph_ids mismatch")
        if response.context_hash != request.context_hash:
            raise AlgorithmOutputError("DecomP worker response context_hash mismatch")
        validate_decomp_full_response(response)
        usage = normalize_external_usage(response.usage)
        events = [
            TraceEvent(
                TracePhase.PREPARE,
                "decomp_worker_request",
                {
                    "config_artifact_id": self.config_artifact_id,
                    "max_steps": self.max_steps,
                    "upstream_revision": DECOMP_REVISION,
                    "backend_binding": backend_binding,
                    "paragraph_ids": paragraph_ids,
                    "context_hash": context_hash,
                    "runtime_identity_hash": content_hash(runtime_identity),
                },
            )
        ]
        events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "decomp_frontier",
                {
                    "index": event.index,
                    "state_id": event.state_id,
                    "score": event.score,
                    "next_command": event.next_command,
                    "action": event.action,
                },
            )
            for event in response.frontier
        )
        events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "decomp_participant_call",
                {
                    "index": call.index,
                    "participant": call.participant,
                    "input_state_id": call.input_state_id,
                    "output_state_id": call.output_state_id,
                    "command": call.command,
                },
            )
            for call in response.participant_calls
        )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "decomp_full_result",
                {
                    "nodes": response.nodes,
                    "dependency_edges": response.dependency_edges,
                    "reasoning_chain": response.reasoning_chain,
                    "final_answer": response.final_answer,
                    "termination": response.termination,
                    "native_artifacts": dict(response.native_artifacts),
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(events),
            counters={
                **external_usage_counters(usage),
                "participant_calls": len(response.participant_calls),
                "frontier_events": len(response.frontier),
                "graph_nodes": len(response.nodes),
                "input_paragraphs": len(paragraph_ids),
            },
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=response.final_answer,
            evidence=state.context,
            trace=trace,
            implementation_identity=f"thin_external_wrapper:allenai/DecomP@{DECOMP_REVISION}",
            profile_kind="controlled_adapter" if controlled else "official_native",
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "full_system": True,
                "component_only": False,
                "answer_is_intervention_input": False,
                "reasoning_chain": response.reasoning_chain,
                "termination": response.termination,
                "paragraph_ids": paragraph_ids,
                "context_hash": context_hash,
                "native_artifacts": dict(response.native_artifacts),
                "worker_fingerprint": dict(runtime_identity),
            },
        )

    def _resolve_runtime(self, context: InterventionContext) -> DecompWorkerRuntime | None:
        injected = context.external_runtime
        if injected is not None and self.runtime is not None and injected is not self.runtime:
            raise ValidationError(
                "DecomP runtime is ambiguous; use either the constructor or "
                "InterventionContext.external_runtime"
            )
        return injected if injected is not None else self.runtime

    @staticmethod
    def _validate_runtime_identity(runtime: DecompWorkerRuntime) -> Mapping[str, Any]:
        fingerprint = getattr(runtime, "fingerprint", None)
        if not callable(fingerprint):
            raise ValidationError("DecomP runtime requires a stable fingerprint()")
        identity = fingerprint()
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError("DecomP runtime fingerprint must be a non-empty mapping")
        return freeze_mapping(identity)
