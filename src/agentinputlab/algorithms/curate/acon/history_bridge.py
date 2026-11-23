"""External bridge for ACON's history optimizer and MemoryManager assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    Message,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    optional_text,
    required_text,
    resolve_worker,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

from .common import (
    ACON_HISTORY_PROTOCOL,
    ACON_UPSTREAM_COMMIT,
    ACONArtifactIdentity,
    ACONHistoryControls,
    history_text,
    preserved_turn_indices,
    target_backend_fingerprint,
    validate_runtime_identity,
    validate_threshold_response,
)


@dataclass(slots=True)
class ACONHistoryRuntimeExternal:
    """Apply pinned ACON with a budget-safe preflight/execute protocol."""

    client: WorkerClient | None = None
    artifacts: ACONArtifactIdentity | None = None
    controls: ACONHistoryControls = field(default_factory=ACONHistoryControls)
    method_id: str = "acon"
    release_id: str = "curate.acon.history_runtime"
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, ACONArtifactIdentity):
            raise ValidationError("ACON history runtime requires frozen artifact identities")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        messages = state.messages
        if len(messages) < 2 or messages[0].role != "system" or messages[1].role != "user":
            raise ValidationError(
                "ACON history runtime requires a session beginning with system then user"
            )
        assert self.artifacts is not None
        worker = resolve_worker(self.client, context, label="ACON history")
        target_fingerprint = target_backend_fingerprint(context.backend, self.artifacts)
        original_prompt = state.metadata.get("acon_original_first_user_prompt")
        if original_prompt is None:
            if "<HISTORY_SUMMARY>" in messages[1].content:
                raise ValidationError(
                    "cannot recover ACON original prompt from an already summarized session"
                )
            original_prompt = messages[1].content
        if not isinstance(original_prompt, str) or not original_prompt.strip():
            raise ValidationError("acon_original_first_user_prompt metadata must be text")
        preserved_indices = preserved_turn_indices(messages, self.controls.preserve_last_k_turns)
        preserved = tuple(messages[index] for index in preserved_indices)

        # The pinned MemoryManager searches by role/content for the first preserved
        # item.  Reproduce that behavior, including its duplicate-content edge case.
        preserved_start = len(messages)
        if preserved:
            first = preserved[0]
            for index, message in enumerate(messages):
                if message.role == first.role and message.content == first.content:
                    preserved_start = index
                    break
        summarization_messages = messages[:preserved_start] if preserved_start > 2 else ()
        material = history_text(summarization_messages)
        previous_summary = state.metadata.get("acon_previous_history_summary", "")
        if not isinstance(previous_summary, str):
            raise ValidationError("acon_previous_history_summary metadata must be text")
        counted_material = f"{previous_summary}\n{material}" if previous_summary else material
        task = state.instruction if state.instruction.strip() else state.query
        base_payload = {
            "task": task,
            "raw_messages": messages,
            "history_text": material,
            "history_text_hash": content_hash(material),
            "counted_material_hash": content_hash(counted_material),
            "previous_history_summary": previous_summary,
            "preserved_message_indices": preserved_indices,
            "preserved_messages": preserved,
            "controls": asdict(self.controls),
            "artifacts": self.artifacts.as_payload(),
            "seed": context.seed,
        }
        preflight_request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=ACON_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                **base_payload,
                "phase": "preflight",
                "decision_token": "",
            },
            dependency_fingerprints={"target_backend": target_fingerprint},
            protocol_version=ACON_HISTORY_PROTOCOL,
        )
        preflight = call_worker(worker, preflight_request)
        if preflight.usage.model_calls != 0:
            raise AlgorithmOutputError("ACON preflight cannot bill a model call")
        payload = preflight.payload
        profile_id, manifest_hash, attestation = validate_runtime_identity(payload, self.artifacts)
        if required_text(
            payload, "counted_material_hash", label="ACON history response"
        ) != content_hash(counted_material):
            raise AlgorithmOutputError("ACON history worker counted different material")
        token_count, threshold_triggered = validate_threshold_response(
            payload,
            threshold=self.controls.summarization_threshold,
        )
        interval_ready = (
            self.controls.summary_interval == -1
            or len(messages) // 2 >= self.controls.summary_interval
        )
        expected_applied = bool(material.strip()) and threshold_triggered and interval_ready
        applied = payload.get("applied")
        if not isinstance(applied, bool) or applied is not expected_applied:
            raise AlgorithmOutputError(
                "ACON history applied flag disagrees with history/threshold/interval branches"
            )
        decision_token = required_text(payload, "decision_token", label="ACON preflight")
        response = preflight
        if applied:
            execute_request = MethodWorkerRequest(
                method_id=self.method_id,
                release_id=self.release_id,
                upstream_commit=ACON_UPSTREAM_COMMIT,
                state_hash=state.state_hash,
                payload={
                    **base_payload,
                    "phase": "execute",
                    "decision_token": decision_token,
                },
                dependency_fingerprints={"target_backend": target_fingerprint},
                minimum_reservation={"model_calls": 1},
                protocol_version=ACON_HISTORY_PROTOCOL,
            )
            response = call_worker(worker, execute_request)
            payload = response.payload
            execute_identity = validate_runtime_identity(payload, self.artifacts)
            if execute_identity != (profile_id, manifest_hash, attestation):
                raise AlgorithmOutputError("ACON runtime identity changed after preflight")
            if payload.get("decision_token") != decision_token:
                raise AlgorithmOutputError("ACON execute decision token mismatch")
        raw_response = optional_text(payload, "raw_response", label="ACON history response")
        summary = optional_text(payload, "summary", label="ACON history response")
        if applied:
            if not summary.strip() or not raw_response.strip():
                raise AlgorithmOutputError(
                    "applied ACON history compression requires parsed summary and raw response"
                )
            if response.usage.model_calls < 1:
                raise AlgorithmOutputError(
                    "applied ACON history compression must report a model call"
                )
            optimized_messages = _assemble_optimized_messages(
                messages,
                preserved,
                summary,
                summary_rule=self.controls.summary_rule,
                original_prompt=original_prompt,
            )
        else:
            if summary or raw_response:
                raise AlgorithmOutputError("skipped ACON history compression returned model output")
            if response.usage.model_calls != 0:
                raise AlgorithmOutputError(
                    "skipped ACON history compression cannot bill model calls"
                )
            optimized_messages = messages

        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "history_branch_inputs",
                    {
                        "controls": asdict(self.controls),
                        "history_text_hash": content_hash(material),
                        "previous_summary_hash": content_hash(previous_summary),
                        "preserved_message_indices": preserved_indices,
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "strict_threshold_decision",
                    {
                        "raw_input_tokens": token_count,
                        "threshold": self.controls.summarization_threshold,
                        "threshold_triggered": threshold_triggered,
                        "interval_ready": interval_ready,
                        "history_available": bool(material.strip()),
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "history_optimization_result",
                    {
                        "applied": applied,
                        "summary": summary,
                        "raw_response": raw_response,
                        "output_message_count": len(optimized_messages),
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 2 if applied else 1,
                "input_messages": len(messages),
                "output_messages": len(optimized_messages),
            },
            warnings=("acon_runtime_code_predates_camera_ready",),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_MESSAGES,
                    optimized_messages,
                    target="agent_session_history",
                    rationale=(
                        "pinned ACON MemoryManager history assembly"
                        if applied
                        else "upstream ACON history threshold/interval skip"
                    ),
                ),
                PatchOperation(
                    PatchOp.SET_METADATA,
                    {
                        "acon_original_first_user_prompt": original_prompt,
                        "acon_previous_history_summary": summary if applied else previous_summary,
                        "acon_runtime_attestation": dict(attestation),
                        "acon_profile_id": profile_id,
                        "acon_compatibility_manifest_sha256": manifest_hash,
                    },
                    target="acon_history_state",
                    rationale="persist ACON cross-round summary and frozen source identity",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"thin_external_wrapper:microsoft/acon@{ACON_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "paper_version": "ICML_2026_camera_ready",
                "code_artifact_date": "2025-10-14",
                "applied": applied,
                "artifacts": self.artifacts.as_payload(),
                "preserved_message_indices": preserved_indices,
                "raw_history_hash": content_hash(messages),
                "raw_response": raw_response,
                "profile_id": profile_id,
                "compatibility_manifest_sha256": manifest_hash,
                "runtime_attestation": dict(attestation),
                "worker_fingerprint": worker_fingerprint(worker, label="ACON history"),
            },
        )


def _assemble_optimized_messages(
    messages: tuple[Message, ...],
    preserved: tuple[Message, ...],
    summary: str,
    *,
    summary_rule: str,
    original_prompt: str,
) -> tuple[Message, ...]:
    base = original_prompt if summary_rule == "reset" else messages[1].content
    user_prompt = f"{base}\n\n<HISTORY_SUMMARY>\n{summary}\n</HISTORY_SUMMARY>"
    return (
        Message(role="system", content=messages[0].content),
        Message(role="user", content=user_prompt),
        *preserved,
    )


__all__ = ["ACONHistoryRuntimeExternal"]
