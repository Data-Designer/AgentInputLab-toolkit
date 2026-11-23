"""External ReFormeR bridge with explicit paper/code divergence profiles."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from agentinputlab.contracts import (
    AlgorithmOutputError,
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
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    dependency_fingerprint,
    optional_text,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

REFORMER_UPSTREAM_COMMIT = "72e52450a922bc0051b5b39c3a6307186a555137"
REFORMER_PROMPT_ARTIFACT_SHA256 = "536bd3bd9596baf65c6b8f98f05e1320d59f221ae9b8d1799461374cd3fad652"
_SEMANTICS = {"r_only", "q_concat_r"}
_PATTERN_MODES = {"automatic", "manual"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ReFormeRControls:
    max_model_calls: int = 2
    max_tokens: int = 4_096
    max_cost_usd: float = 10.0

    def __post_init__(self) -> None:
        for name in ("max_model_calls", "max_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"ReFormeR {name} must be positive")
        if (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(float(self.max_cost_usd))
            or self.max_cost_usd < 0
        ):
            raise ValidationError("ReFormeR max_cost_usd must be finite and non-negative")


@dataclass(slots=True)
class ReFormeRExternal:
    """Delegate pattern selection/application while making output semantics explicit."""

    client: WorkerClient | None = None
    source_license_acknowledged: bool = False
    pattern_artifact_hash: str = ""
    semantics: str = "r_only"
    pattern_mode: str = "automatic"
    manual_pattern_id: str = ""
    release_id: str = "rewrite.reformer.repo_native_r_only"
    profile_kind: str = "controlled_adapter"
    top_document_count: int = 3
    concat_separator: str = " "
    controls: ReFormeRControls = ReFormeRControls()
    method_id: str = "reformer"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        if not isinstance(self.source_license_acknowledged, bool):
            raise ValidationError("ReFormeR source_license_acknowledged must be boolean")
        if _SHA256.fullmatch(self.pattern_artifact_hash) is None:
            raise ValidationError("ReFormeR pattern_artifact_hash must be a SHA-256 hash")
        if self.semantics not in _SEMANTICS:
            raise ValidationError("ReFormeR semantics must be r_only or q_concat_r")
        if self.pattern_mode not in _PATTERN_MODES:
            raise ValidationError("ReFormeR pattern_mode must be automatic or manual")
        if self.pattern_mode == "manual" and not self.manual_pattern_id.strip():
            raise ValidationError("manual ReFormeR profiles require manual_pattern_id")
        if self.pattern_mode == "automatic" and self.manual_pattern_id:
            raise ValidationError("automatic ReFormeR profiles cannot set manual_pattern_id")
        if self.profile_kind == "official_native":
            raise ValidationError(
                "ReFormeR official_native is unavailable until a fixed official checkpoint, "
                "complete artifact attestation, and parity evidence are released"
            )
        if self.profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("invalid ReFormeR profile_kind")
        if (
            isinstance(self.top_document_count, bool)
            or not isinstance(self.top_document_count, int)
            or self.top_document_count < 1
        ):
            raise ValidationError("ReFormeR top_document_count must be positive")
        if not isinstance(self.concat_separator, str):
            raise ValidationError("ReFormeR concat_separator must be text")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("ReFormeR requires a non-empty query")
        _require_source_license_acknowledgement(self.source_license_acknowledged)
        worker = resolve_worker(self.client, context, label="ReFormeR")
        backend_fp = _bound_dependency(
            context,
            dependency=context.backend,
            run_config_key="reformer_model_fingerprint",
            label="ReFormeR generator",
            required=True,
        )
        retriever_fp = _bound_dependency(
            context,
            dependency=context.retriever,
            run_config_key="reformer_retriever_fingerprint",
            label="ReFormeR context retriever",
            required=self.pattern_mode == "automatic",
        )
        supplied_context_ids = tuple(
            block.block_id for block in state.context[: self.top_document_count]
        )
        expected_calls = 2 if self.pattern_mode == "automatic" and supplied_context_ids else 1
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=REFORMER_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "query": state.query,
                "context": state.context,
                "top_document_count": self.top_document_count,
                "pattern_artifact_hash": self.pattern_artifact_hash,
                "prompt_artifact_hash": REFORMER_PROMPT_ARTIFACT_SHA256,
                "semantics": self.semantics,
                "pattern_mode": self.pattern_mode,
                "manual_pattern_id": self.manual_pattern_id,
                "source_license_acknowledged": self.source_license_acknowledged,
                "controls": asdict(self.controls),
                "seed": context.seed,
            },
            dependency_fingerprints={
                "backend": backend_fp,
                "retriever": retriever_fp,
            },
            minimum_reservation={"model_calls": expected_calls},
            protocol_version="agentinputlab.reformer.v2",
        )
        response = call_worker(worker, request)
        payload = response.payload
        returned_hash = required_text(payload, "pattern_library_hash", label="ReFormeR response")
        if returned_hash != self.pattern_artifact_hash:
            raise AlgorithmOutputError("ReFormeR pattern library hash mismatch")
        pattern_id = required_text(payload, "selected_pattern_id", label="ReFormeR response")
        pattern = required_text(payload, "selected_pattern", label="ReFormeR response")
        reformulation = required_text(payload, "reformulation", label="ReFormeR response")
        returned_semantics = required_text(payload, "semantics", label="ReFormeR response")
        returned_pattern_mode = required_text(payload, "pattern_mode", label="ReFormeR response")
        if returned_semantics != self.semantics or returned_pattern_mode != self.pattern_mode:
            raise AlgorithmOutputError("ReFormeR worker profile identity mismatch")
        top_document_ids = string_tuple(
            payload.get("top_document_ids", ()),
            label="ReFormeR top_document_ids",
            unique=True,
        )
        if len(top_document_ids) > self.top_document_count:
            raise AlgorithmOutputError("ReFormeR used too many context documents")
        if (
            supplied_context_ids
            and top_document_ids != supplied_context_ids[: len(top_document_ids)]
        ):
            raise AlgorithmOutputError("ReFormeR worker changed the supplied top-document ordering")
        parse_status = required_text(payload, "selection_parse_status", label="ReFormeR response")
        if parse_status not in {
            "valid_json",
            "first_pattern_fallback",
            "no_documents_first_pattern",
            "manual",
        }:
            raise AlgorithmOutputError("ReFormeR returned an unsupported selection parse status")
        fallback_pattern_id = optional_text(
            payload,
            "fallback_first_pattern_id",
            label="ReFormeR response",
        )
        if parse_status == "first_pattern_fallback" and fallback_pattern_id != pattern_id:
            raise AlgorithmOutputError(
                "ReFormeR first-pattern fallback does not match selected_pattern_id"
            )
        if self.pattern_mode == "manual" and pattern_id != self.manual_pattern_id:
            raise AlgorithmOutputError("ReFormeR worker changed the manual pattern selection")
        if response.usage.model_calls != expected_calls:
            raise AlgorithmOutputError("ReFormeR worker usage mismatches its model-call trajectory")
        output_query = (
            reformulation
            if self.semantics == "r_only"
            else f"{state.query}{self.concat_separator}{reformulation}"
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "reformer_context",
                    {
                        "top_document_ids": top_document_ids,
                        "pattern_library_hash": returned_hash,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "pattern_selection",
                    {
                        "selected_pattern_id": pattern_id,
                        "selected_pattern": pattern,
                        "parse_status": parse_status,
                        "pattern_mode": self.pattern_mode,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "reformulation_semantics",
                    {
                        "repository_reformulation": reformulation,
                        "semantics": self.semantics,
                        "materialized_query": output_query,
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "context_documents": len(top_document_ids),
            },
            warnings=(
                ("repository_first_pattern_fallback_applied",)
                if parse_status == "first_pattern_fallback"
                else ()
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_QUERY,
                    output_query,
                    target="query",
                    rationale="ReFormeR selected and applied an explicit reformulation pattern",
                ),
            ),
            trace=trace,
            implementation_identity=f"thin_external_wrapper:aminbigdeli/ReFormeR@{REFORMER_UPSTREAM_COMMIT}",
            profile_kind=self.profile_kind,
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "source_license_id": "NOASSERTION",
                "source_license_acknowledged": self.source_license_acknowledged,
                "paper_code_semantics": self.semantics,
                "pattern_mode": self.pattern_mode,
                "manual_pattern_id": self.manual_pattern_id,
                "pattern_artifact_hash": returned_hash,
                "worker_adapter_revision": payload.get("adapter_revision", "legacy"),
                "selection_library_match": payload.get("selection_library_match", "exact"),
                "raw_selected_pattern_hash": payload.get("raw_selected_pattern_hash", ""),
                "worker_fingerprint": worker_fingerprint(worker, label="ReFormeR"),
                "backend_fingerprint": backend_fp,
                "retriever_fingerprint": retriever_fp,
            },
        )


class ReFormeRRepoNative(ReFormeRExternal):
    def __init__(
        self,
        client: WorkerClient | None = None,
        *,
        source_license_acknowledged: bool = False,
        pattern_artifact_hash: str = "",
        top_document_count: int = 3,
        controls: ReFormeRControls | None = None,
    ) -> None:
        super().__init__(
            client=client,
            source_license_acknowledged=source_license_acknowledged,
            pattern_artifact_hash=pattern_artifact_hash,
            semantics="r_only",
            pattern_mode="automatic",
            release_id="rewrite.reformer.repo_native_r_only",
            profile_kind="controlled_adapter",
            top_document_count=top_document_count,
            controls=controls or ReFormeRControls(),
        )


class ReFormeRPaperConcat(ReFormeRExternal):
    def __init__(
        self,
        client: WorkerClient | None = None,
        *,
        source_license_acknowledged: bool = False,
        pattern_artifact_hash: str = "",
        top_document_count: int = 3,
        controls: ReFormeRControls | None = None,
    ) -> None:
        super().__init__(
            client=client,
            source_license_acknowledged=source_license_acknowledged,
            pattern_artifact_hash=pattern_artifact_hash,
            semantics="q_concat_r",
            pattern_mode="automatic",
            release_id="rewrite.reformer.paper_spec_q_concat_r",
            profile_kind="controlled_adapter",
            top_document_count=top_document_count,
            controls=controls or ReFormeRControls(),
        )


class ReFormeRControlledROnly(ReFormeRExternal):
    def __init__(
        self,
        client: WorkerClient | None = None,
        *,
        source_license_acknowledged: bool = False,
        pattern_artifact_hash: str = "",
        manual_pattern_id: str = "",
        top_document_count: int = 3,
        controls: ReFormeRControls | None = None,
    ) -> None:
        super().__init__(
            client=client,
            source_license_acknowledged=source_license_acknowledged,
            pattern_artifact_hash=pattern_artifact_hash,
            semantics="r_only",
            pattern_mode="manual",
            manual_pattern_id=manual_pattern_id,
            release_id="rewrite.reformer.controlled_r_only",
            profile_kind="controlled_adapter",
            top_document_count=top_document_count,
            controls=controls or ReFormeRControls(max_model_calls=1),
        )


class ReFormeRControlledConcat(ReFormeRExternal):
    def __init__(
        self,
        client: WorkerClient | None = None,
        *,
        source_license_acknowledged: bool = False,
        pattern_artifact_hash: str = "",
        manual_pattern_id: str = "",
        top_document_count: int = 3,
        controls: ReFormeRControls | None = None,
    ) -> None:
        super().__init__(
            client=client,
            source_license_acknowledged=source_license_acknowledged,
            pattern_artifact_hash=pattern_artifact_hash,
            semantics="q_concat_r",
            pattern_mode="manual",
            manual_pattern_id=manual_pattern_id,
            release_id="rewrite.reformer.controlled_q_concat_r",
            profile_kind="controlled_adapter",
            top_document_count=top_document_count,
            controls=controls or ReFormeRControls(max_model_calls=1),
        )


def _require_source_license_acknowledgement(acknowledged: bool) -> None:
    if not acknowledged:
        raise ExternalDependencyError(
            "ReFormeR upstream source license is NOASSERTION; explicitly acknowledge the "
            "user-mounted source before execution"
        )


def _bound_dependency(
    context: InterventionContext,
    *,
    dependency: object,
    run_config_key: str,
    label: str,
    required: bool,
):
    if dependency is not None:
        return dependency_fingerprint(dependency, label=label, required=True)
    configured = context.run_config.get(run_config_key)
    if configured is None:
        return dependency_fingerprint(None, label=label, required=required)
    if not isinstance(configured, Mapping) or not configured:
        raise ValidationError(f"{run_config_key} must be a non-empty mapping")
    return configured


__all__ = [
    "REFORMER_PROMPT_ARTIFACT_SHA256",
    "REFORMER_UPSTREAM_COMMIT",
    "ReFormeRControlledConcat",
    "ReFormeRControlledROnly",
    "ReFormeRControls",
    "ReFormeRExternal",
    "ReFormeRPaperConcat",
    "ReFormeRRepoNative",
]
