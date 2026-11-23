"""External bridge for ACON's observation optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
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
    ACON_OBSERVATION_PROTOCOL,
    ACON_RESERVED_OPT_ARGS,
    ACON_UPSTREAM_COMMIT,
    ACONArtifactIdentity,
    ACONObservationControls,
    history_text,
    target_backend_fingerprint,
    validate_runtime_identity,
    validate_threshold_response,
)


@dataclass(slots=True)
class ACONObservationRuntimeExternal:
    """Compress one tool observation and preserve all source-block lineage."""

    client: WorkerClient | None = None
    artifacts: ACONArtifactIdentity | None = None
    controls: ACONObservationControls = field(default_factory=ACONObservationControls)
    method_id: str = "acon"
    release_id: str = "curate.acon.observation_runtime"
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, ACONArtifactIdentity):
            raise ValidationError("ACON observation runtime requires frozen artifact identities")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.context:
            raise ValidationError("ACON observation runtime requires a context observation")
        assert self.artifacts is not None
        index = self.controls.context_index
        if index < 0:
            index += len(state.context)
        if index < 0 or index >= len(state.context):
            raise ValidationError("ACON observation context_index is out of range")
        source = state.context[index]
        if not source.block_id:
            raise ValidationError("ACON observation requires a non-empty source block ID")
        previous_summary = state.metadata.get("acon_previous_history_summary", "")
        if not isinstance(previous_summary, str):
            raise ValidationError("acon_previous_history_summary metadata must be text")
        current_history = history_text(state.messages)
        if previous_summary and self.controls.add_previous_history_summary:
            current_history = (
                f"PREVIOUS HISTORY SUMMARY:\n{previous_summary}\n\n"
                f"LATEST HISTORY:\n{current_history}"
            )
        opt_args = context.run_config.get("acon_opt_args", {})
        if not isinstance(opt_args, Mapping):
            raise ValidationError("acon_opt_args run setting must be a mapping")
        collisions = set(opt_args) & ACON_RESERVED_OPT_ARGS
        if collisions:
            raise ValidationError(
                "acon_opt_args cannot override reserved upstream inputs: "
                + ", ".join(sorted(collisions))
            )
        task = state.instruction if state.instruction.strip() else state.query
        worker = resolve_worker(self.client, context, label="ACON observation")
        target_fingerprint = target_backend_fingerprint(context.backend, self.artifacts)
        base_payload = {
            "task": task,
            "observation": source.text,
            "source_context_id": source.block_id,
            "history_text": current_history,
            "history_text_hash": content_hash(current_history),
            "raw_messages": state.messages,
            "opt_args": opt_args,
            "opt_args_hash": content_hash(opt_args),
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
            protocol_version=ACON_OBSERVATION_PROTOCOL,
        )
        preflight = call_worker(worker, preflight_request)
        if preflight.usage.model_calls != 0:
            raise AlgorithmOutputError("ACON preflight cannot bill a model call")
        payload = preflight.payload
        profile_id, manifest_hash, attestation = validate_runtime_identity(payload, self.artifacts)
        if (
            required_text(payload, "source_context_id", label="ACON observation response")
            != source.block_id
        ):
            raise AlgorithmOutputError("ACON observation source block mismatch")
        if required_text(
            payload, "history_text_hash", label="ACON observation response"
        ) != content_hash(current_history):
            raise AlgorithmOutputError("ACON observation worker used different history")
        if required_text(
            payload, "opt_args_hash", label="ACON observation response"
        ) != content_hash(opt_args):
            raise AlgorithmOutputError("ACON observation worker used different opt_args")
        token_count, threshold_triggered = validate_threshold_response(
            payload,
            threshold=self.controls.summarization_threshold,
        )
        applied = payload.get("applied")
        if not isinstance(applied, bool) or applied is not threshold_triggered:
            raise AlgorithmOutputError(
                "ACON observation applied flag disagrees with strict threshold decision"
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
                protocol_version=ACON_OBSERVATION_PROTOCOL,
            )
            response = call_worker(worker, execute_request)
            payload = response.payload
            execute_identity = validate_runtime_identity(payload, self.artifacts)
            if execute_identity != (profile_id, manifest_hash, attestation):
                raise AlgorithmOutputError("ACON runtime identity changed after preflight")
            if payload.get("decision_token") != decision_token:
                raise AlgorithmOutputError("ACON execute decision token mismatch")
        optimized = required_text(
            payload, "optimized_observation", label="ACON observation response"
        )
        raw_response = optional_text(payload, "raw_response", label="ACON observation response")
        if applied:
            if not raw_response.strip() or response.usage.model_calls < 1:
                raise AlgorithmOutputError(
                    "applied ACON observation compression requires response and model usage"
                )
            replacement = ContextBlock(
                block_id=source.block_id,
                text=optimized,
                source=source.source,
                score=source.score,
                metadata={
                    **dict(source.metadata),
                    "acon_source_state_hash": state.state_hash,
                    "acon_source_text_hash": content_hash(source.text),
                    "acon_raw_response_hash": content_hash(raw_response),
                    "acon_applied": True,
                },
            )
            optimized_context = (
                *state.context[:index],
                replacement,
                *state.context[index + 1 :],
            )
        else:
            if optimized != source.text or raw_response:
                raise AlgorithmOutputError(
                    "skipped ACON observation compression must return the raw observation"
                )
            if response.usage.model_calls != 0:
                raise AlgorithmOutputError(
                    "skipped ACON observation compression cannot bill model calls"
                )
            optimized_context = state.context

        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "observation_branch_inputs",
                    {
                        "controls": asdict(self.controls),
                        "source_context_id": source.block_id,
                        "history_text_hash": content_hash(current_history),
                        "opt_args_hash": content_hash(opt_args),
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "strict_threshold_decision",
                    {
                        "raw_input_tokens": token_count,
                        "threshold": self.controls.summarization_threshold,
                        "threshold_triggered": threshold_triggered,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "observation_optimization_result",
                    {
                        "applied": applied,
                        "source_text_hash": content_hash(source.text),
                        "output_text_hash": content_hash(optimized),
                        "raw_response": raw_response,
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 2 if applied else 1,
                "input_observation_characters": len(source.text),
                "output_observation_characters": len(optimized),
            },
            warnings=("acon_runtime_code_predates_camera_ready",),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_CONTEXT,
                    optimized_context,
                    target="tool_observation_context",
                    rationale=(
                        "pinned ACON observation optimizer output"
                        if applied
                        else "upstream ACON observation threshold skip"
                    ),
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
                "source_context_id": source.block_id,
                "raw_observation_hash": content_hash(source.text),
                "raw_response": raw_response,
                "profile_id": profile_id,
                "compatibility_manifest_sha256": manifest_hash,
                "runtime_attestation": dict(attestation),
                "worker_fingerprint": worker_fingerprint(worker, label="ACON observation"),
            },
        )


__all__ = ["ACONObservationRuntimeExternal"]
