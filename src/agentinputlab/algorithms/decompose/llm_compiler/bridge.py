"""Faithful transport boundary for the pinned LLMCompiler runtime.

Planning, streaming scheduling, tool execution, join, and replan stay in the
official MIT worker.  The bridge validates the task DAG and scheduler history
instead of replacing the system with a one-shot planning prompt.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agentinputlab.connectors.tools import ToolConnector
from agentinputlab.contracts import (
    AlgorithmOutputError,
    CompositeResult,
    ContextBlock,
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
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    dependency_fingerprint,
    mapping_sequence,
    nonnegative_integer,
    optional_text,
    required_text,
    resolve_worker,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

LLMCOMPILER_UPSTREAM_COMMIT = "a00c9d35507507da70e8c637eee64efc8c1857ae"
LLMCOMPILER_COMPATIBILITY_MANIFEST_SHA256 = (
    "4b22356ddd587b0fac952511ab53dcf1ae0007b9645e02cd6375122a41ddedd6"
)
_OFFICIAL_NATIVE_WORKER_IDS = frozenset({"agentinputlab.llmcompiler.official-native.pinned.v1"})
_OFFICIAL_NATIVE_ATTESTATION = {
    "schema_version": "agentinputlab.llmcompiler.official-native-attestation.v1",
    "upstream_commit": LLMCOMPILER_UPSTREAM_COMMIT,
    "compatibility_manifest_sha256": LLMCOMPILER_COMPATIBILITY_MANIFEST_SHA256,
    "source_files_verified": True,
    "required_versions_verified": True,
    "import_origins_verified": True,
    "model_transport": "upstream_native",
    "tool_transport": "upstream_native",
}
_TASK_STATES = ("planned", "ready", "running", "completed", "failed")


def _profile_identity(
    worker_identity: Mapping[str, Any],
    *,
    full_system: bool,
) -> tuple[str, str, str]:
    """Return a conservative profile label from a strict worker attestation."""

    attestation = worker_identity.get("official_native_attestation")
    official = (
        full_system
        and worker_identity.get("worker_id") in _OFFICIAL_NATIVE_WORKER_IDS
        and worker_identity.get("upstream_commit") == LLMCOMPILER_UPSTREAM_COMMIT
        and isinstance(attestation, Mapping)
        and dict(attestation) == _OFFICIAL_NATIVE_ATTESTATION
    )
    if official:
        return "official_native", "official_external_adapter", "official_native_allowlisted"
    return "controlled_adapter", "controlled_external_adapter", "controlled_default"


@dataclass(frozen=True, slots=True)
class LLMCompilerControls:
    max_tasks: int = 32
    max_replans: int = 2
    concurrency: int = 8
    timeout_seconds: float = 300.0
    streaming: bool = True

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.max_tasks, self.concurrency)
        ):
            raise ValidationError("LLMCompiler task/concurrency limits must be positive")
        if (
            isinstance(self.max_replans, bool)
            or not isinstance(self.max_replans, int)
            or self.max_replans < 0
        ):
            raise ValidationError("LLMCompiler max_replans must be non-negative")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValidationError("LLMCompiler timeout_seconds must be positive")
        if not isinstance(self.streaming, bool):
            raise ValidationError("LLMCompiler streaming must be boolean")


@dataclass(frozen=True, slots=True)
class CompilerTask:
    task_id: int
    tool_name: str
    arguments: Mapping[str, Any]
    dependencies: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.task_id < 1 or not self.tool_name.strip():
            raise AlgorithmOutputError("LLMCompiler task ID/tool name is invalid")
        object.__setattr__(self, "arguments", freeze_mapping(self.arguments))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in self.dependencies
        ):
            raise AlgorithmOutputError("LLMCompiler dependencies must be positive task IDs")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise AlgorithmOutputError("LLMCompiler task contains duplicate dependencies")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompilerTask:
        task_id = nonnegative_integer(value.get("task_id"), label="LLMCompiler task_id")
        tool_name = required_text(value, "tool_name", label="LLMCompiler task")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise AlgorithmOutputError("LLMCompiler task arguments must be a mapping")
        dependencies = value.get("dependencies", ())
        if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
            raise AlgorithmOutputError("LLMCompiler task dependencies must be a sequence")
        return cls(
            task_id=task_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            dependencies=tuple(dependencies),
        )


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    index: int
    task_id: int
    state: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchedulerEvent:
        index = nonnegative_integer(value.get("index"), label="scheduler event index")
        task_id = nonnegative_integer(value.get("task_id"), label="scheduler event task_id")
        state = required_text(value, "state", label="scheduler event")
        if state not in _TASK_STATES:
            raise AlgorithmOutputError("LLMCompiler scheduler event has an unsupported state")
        return cls(index=index, task_id=task_id, state=state)


@dataclass(slots=True)
class _LLMCompilerBridge:
    client: WorkerClient | None = None
    controls: LLMCompilerControls = LLMCompilerControls()
    prompt_hash: str = ""
    full_system: bool = False
    release_id: str = "plan.llm_compiler.planner_component"
    method_id: str = "llm_compiler"
    family: SemanticFamily = SemanticFamily.DECOMPOSE

    def __post_init__(self) -> None:
        if not self.prompt_hash.strip():
            raise ValidationError("LLMCompiler requires a prompt_hash")

    def _execute_patch(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("LLMCompiler requires a non-empty query")
        worker = resolve_worker(self.client, context, label="LLMCompiler")
        worker_identity = worker_fingerprint(worker, label="LLMCompiler")
        if not isinstance(context.tools, ToolConnector):
            raise ExternalDependencyError("LLMCompiler requires an injected ToolConnector")
        tool_specs = context.tools.list_tools()
        allowed_tools = tuple(spec.name for spec in tool_specs)
        if not allowed_tools:
            raise ExternalDependencyError("LLMCompiler requires at least one available tool")
        backend_fp = dependency_fingerprint(
            context.backend,
            label="LLMCompiler planner backend",
            required=True,
        )
        tools_fp = dependency_fingerprint(
            context.tools,
            label="LLMCompiler tools",
            required=True,
        )
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=LLMCOMPILER_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "query": state.query,
                "instruction": state.instruction,
                "messages": state.messages,
                "tool_specs": tool_specs,
                "controls": asdict(self.controls),
                "prompt_hash": self.prompt_hash,
                "execution_mode": "full" if self.full_system else "planner_component",
                "seed": context.seed,
                "execution_budget": asdict(state.budget),
                "tool_argument_order": worker_identity.get("positional_argument_order", {}),
            },
            dependency_fingerprints={"backend": backend_fp, "tools": tools_fp},
            minimum_reservation={"model_calls": 1},
            protocol_version="agentinputlab.llm-compiler.v1",
        )
        response = call_worker(worker, request)
        payload = response.payload
        tasks = tuple(
            CompilerTask.from_mapping(row)
            for row in mapping_sequence(payload.get("tasks"), label="LLMCompiler tasks")
        )
        _validate_dag(tasks, max_tasks=self.controls.max_tasks, allowed_tools=set(allowed_tools))
        raw_events = mapping_sequence(
            payload.get("scheduler_events", ()),
            label="LLMCompiler scheduler_events",
        )
        events = tuple(SchedulerEvent.from_mapping(row) for row in raw_events)
        _validate_scheduler(events, tasks, full_system=self.full_system)
        replans = nonnegative_integer(payload.get("replans", 0), label="LLMCompiler replans")
        if replans > self.controls.max_replans:
            raise AlgorithmOutputError("LLMCompiler worker exceeded max_replans")
        streaming = payload.get("streaming")
        if not isinstance(streaming, bool) or streaming != self.controls.streaming:
            raise AlgorithmOutputError("LLMCompiler worker streaming mode mismatch")
        raw_plan_hash = required_text(payload, "raw_plan_hash", label="LLMCompiler response")
        termination = required_text(payload, "termination", label="LLMCompiler response")
        if termination not in {"planner_complete", "finish", "replan_limit", "tool_failure"}:
            raise AlgorithmOutputError("LLMCompiler response has unsupported termination")
        final_answer = optional_text(payload, "final_answer", label="LLMCompiler response")
        if self.full_system and (termination != "finish" or not final_answer.strip()):
            raise AlgorithmOutputError(
                "LLMCompiler full profile requires successful finish and a final answer"
            )
        if not self.full_system and termination != "planner_complete":
            raise AlgorithmOutputError(
                "LLMCompiler planner component requires planner_complete termination"
            )
        if not self.full_system and events:
            raise AlgorithmOutputError("LLMCompiler planner component must not execute tasks")

        task_rows = tuple(to_jsonable(task) for task in tasks)
        canonical_plan = json.dumps(
            task_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        operations: list[PatchOperation] = [
            PatchOperation(
                PatchOp.ADD_CONTEXT,
                ContextBlock(
                    block_id=f"{state.task_id or state.state_hash}:llm-compiler-plan",
                    text=canonical_plan,
                    source="llm_compiler_task_dag",
                    metadata={
                        "raw_plan_hash": raw_plan_hash,
                        "task_count": len(tasks),
                        "full_system": self.full_system,
                    },
                ),
                target="execution_plan",
                rationale="materialize the parsed LLMCompiler dependency DAG",
            ),
            PatchOperation(
                PatchOp.SET_METADATA,
                {
                    "llm_compiler_tasks": task_rows,
                    "llm_compiler_termination": termination,
                    "llm_compiler_replans": replans,
                    "llm_compiler_final_answer": final_answer,
                },
                target="metadata",
            ),
        ]
        trace_events = [
            TraceEvent(
                TracePhase.PREPARE,
                "compiler_controls",
                {
                    **asdict(self.controls),
                    "prompt_hash": self.prompt_hash,
                    "tool_names": allowed_tools,
                },
            ),
            TraceEvent(
                TracePhase.PROPOSE,
                "task_dag",
                {"tasks": task_rows, "raw_plan_hash": raw_plan_hash},
            ),
        ]
        trace_events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "scheduler_event",
                {"index": event.index, "task_id": event.task_id, "state": event.state},
            )
            for event in events
        )
        trace_events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "join_result",
                {
                    "termination": termination,
                    "replans": replans,
                    "final_answer": final_answer,
                    "full_system": self.full_system,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(trace_events),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "planned_tasks": len(tasks),
                "scheduler_events": len(events),
                "replans": replans,
            },
        )
        profile_kind, implementation_origin, profile_attestation = _profile_identity(
            worker_identity,
            full_system=self.full_system,
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=tuple(operations),
            trace=trace,
            implementation_identity=(
                f"{implementation_origin}:SqueezeAILab/LLMCompiler@{LLMCOMPILER_UPSTREAM_COMMIT}"
            ),
            profile_kind=profile_kind,
            metadata={
                "implementation_origin": implementation_origin,
                "profile_attestation": profile_attestation,
                "full_system": self.full_system,
                "requires_tools": True,
                "terminal_answer_in_metadata": self.full_system,
                "worker_fingerprint": worker_identity,
                "backend_fingerprint": backend_fp,
                "tools_fingerprint": tools_fp,
                "tool_schema_hash": content_hash(tool_specs),
                "final_answer": final_answer,
                "task_dag": task_rows,
            },
        )


class LLMCompilerPlannerBridge(_LLMCompilerBridge):
    def __init__(
        self,
        client: WorkerClient | None = None,
        *,
        prompt_hash: str = "",
        controls: LLMCompilerControls | None = None,
    ) -> None:
        super().__init__(
            client=client,
            controls=controls if controls is not None else LLMCompilerControls(),
            prompt_hash=prompt_hash,
            full_system=False,
            release_id="plan.llm_compiler.planner_component",
        )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        return self._execute_patch(state, context)


class LLMCompilerFullBridge(_LLMCompilerBridge):
    def __init__(
        self,
        client: WorkerClient | None = None,
        *,
        prompt_hash: str = "",
        controls: LLMCompilerControls | None = None,
    ) -> None:
        super().__init__(
            client=client,
            controls=controls if controls is not None else LLMCompilerControls(),
            prompt_hash=prompt_hash,
            full_system=True,
            release_id="plan.llm_compiler.full",
        )

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        patch = self._execute_patch(state, context)
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=str(patch.metadata["final_answer"]),
            evidence=(),
            trace=patch.trace,
            implementation_identity=patch.implementation_identity,
            profile_kind=patch.profile_kind,
            metadata={
                **dict(patch.metadata),
                "internal_validation_patch_hash": patch.patch_hash,
            },
        )


def _validate_dag(
    tasks: tuple[CompilerTask, ...],
    *,
    max_tasks: int,
    allowed_tools: set[str],
) -> None:
    if not tasks:
        raise AlgorithmOutputError("LLMCompiler worker returned an empty task DAG")
    if len(tasks) > max_tasks:
        raise AlgorithmOutputError("LLMCompiler worker exceeded max_tasks")
    ids = tuple(task.task_id for task in tasks)
    if len(ids) != len(set(ids)):
        raise AlgorithmOutputError("LLMCompiler task IDs must be unique")
    if ids != tuple(sorted(ids)):
        raise AlgorithmOutputError("LLMCompiler task IDs must be monotonically increasing")
    known = set(ids)
    for task in tasks:
        if task.tool_name not in allowed_tools:
            raise AlgorithmOutputError(f"LLMCompiler planned unknown tool {task.tool_name!r}")
        if not set(task.dependencies) <= known:
            raise AlgorithmOutputError("LLMCompiler task references an unknown dependency")
        if task.task_id in task.dependencies:
            raise AlgorithmOutputError("LLMCompiler task cannot depend on itself")
    graph = {task.task_id: set(task.dependencies) for task in tasks}
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> None:
        if node in visiting:
            raise AlgorithmOutputError("LLMCompiler task graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for task_id in ids:
        visit(task_id)


def _validate_scheduler(
    events: tuple[SchedulerEvent, ...],
    tasks: tuple[CompilerTask, ...],
    *,
    full_system: bool,
) -> None:
    if not full_system:
        return
    if not events:
        raise AlgorithmOutputError("LLMCompiler full worker requires scheduler events")
    if tuple(event.index for event in events) != tuple(range(len(events))):
        raise AlgorithmOutputError("LLMCompiler scheduler indices must be contiguous")
    by_id = {task.task_id: task for task in tasks}
    state_index = {state: index for index, state in enumerate(_TASK_STATES)}
    current: dict[int, str] = {}
    completed: set[int] = set()
    for event in events:
        if event.task_id not in by_id:
            raise AlgorithmOutputError("LLMCompiler scheduler references an unknown task")
        previous = current.get(event.task_id)
        if previous is not None and state_index[event.state] < state_index[previous]:
            raise AlgorithmOutputError("LLMCompiler scheduler state moved backwards")
        if event.state == "running" and not set(by_id[event.task_id].dependencies) <= completed:
            raise AlgorithmOutputError("LLMCompiler task started before all dependencies completed")
        current[event.task_id] = event.state
        if event.state == "completed":
            completed.add(event.task_id)
    missing = set(by_id) - set(current)
    if missing:
        raise AlgorithmOutputError(
            "LLMCompiler full worker omitted scheduler histories for planned tasks"
        )
    if completed != set(by_id):
        raise AlgorithmOutputError(
            "LLMCompiler full worker finished before every planned task completed"
        )


__all__ = [
    "LLMCOMPILER_UPSTREAM_COMMIT",
    "CompilerTask",
    "LLMCompilerControls",
    "LLMCompilerFullBridge",
    "LLMCompilerPlannerBridge",
    "SchedulerEvent",
]
