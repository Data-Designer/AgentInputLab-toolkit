"""External bridges for the pinned InfoCQR teacher and official author caches."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

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
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

from .cache_replay import INFOCQR_UPSTREAM_COMMIT

INFOCQR_PROTOCOL_VERSION = "agentinputlab.infocqr.v1"
INFOCQR_POSTPROCESS_SHA256 = "7201ceaeb16beaaf3af874faf083ce4f64dd630636a2e302edf398cc4da8d3dd"
INFOCQR_PROMPT_SOURCE_SHA256 = {
    "rw_zsl": "93d6ba61da042f15ed6d506b3d69bac1a16d3e33453b84ec8f6b7317d018defc",
    "rw_fsl": "9c447f5292a6c936160ab847d3eb18db4704ea62f3d46a896ef5613aac2ce6a6",
    "t5_edit": "5c71071d213ac562651d5577e623f73d0b98ffc3279f2dc1727127e8475d1bba",
    "self_edit": "2fba6bb3b24a7b332c7896b89c7ae6480c2d21272f42bad893ee5151c7739373",
}
_LIVE_PROFILES = frozenset(INFOCQR_PROMPT_SOURCE_SHA256)
_CACHE_RELEASES = {
    "rw_zsl": "rewrite.infocqr.cached_rw_zsl",
    "rw_fsl": "rewrite.infocqr.cached_rw_fsl",
    "t5_edit": "rewrite.infocqr.cached_edit",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class InfoCQRExternalControls:
    max_model_calls: int = 2
    max_tokens: int = 8_192
    max_cost_usd: float = 10.0

    def __post_init__(self) -> None:
        for name in ("max_model_calls", "max_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"InfoCQR {name} must be non-negative")
        if (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(float(self.max_cost_usd))
            or self.max_cost_usd < 0
        ):
            raise ValidationError("InfoCQR max_cost_usd must be finite and non-negative")


@dataclass(slots=True)
class InfoCQRLiveTeacherExternal:
    """Call the exact pinned teacher prompt function through an external worker."""

    prompt_profile: str = "rw_fsl"
    client: WorkerClient | None = None
    source_license_acknowledged: bool = False
    dataset_id: str = "qrecc"
    dataset_revision: str = ""
    split: str = "test"
    retrieval_corpus_id: str = ""
    profile_kind: str = "controlled_adapter"
    initial_rewrite: str = ""
    initial_rewrite_source_sha256: str = ""
    controls: InfoCQRExternalControls = InfoCQRExternalControls()
    release_id: str = "rewrite.infocqr.live_teacher_external"
    method_id: str = "infocqr"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        common_support = self.release_id == "rewrite.infocqr.common_support_rw_zsl"
        if self.prompt_profile not in _LIVE_PROFILES or (
            common_support and self.prompt_profile != "rw_zsl"
        ):
            raise ValidationError("unsupported InfoCQR live teacher profile")
        _validate_common_configuration(
            source_license_acknowledged=self.source_license_acknowledged,
            dataset_revision=self.dataset_revision,
            split=self.split,
            retrieval_corpus_id=self.retrieval_corpus_id,
        )
        if self.profile_kind not in {"official_native", "controlled_adapter"}:
            raise ValidationError("InfoCQR live profile_kind is invalid")
        if not self.dataset_id.strip():
            raise ValidationError("InfoCQR dataset_id must not be empty")
        if common_support and self.profile_kind != "controlled_adapter":
            raise ValidationError("InfoCQR common-support execution is a controlled adapter")
        if not common_support and self.dataset_id.lower() != "qrecc":
            raise ValidationError("native InfoCQR execution requires QReCC")
        if self.prompt_profile == "t5_edit":
            if not self.initial_rewrite.strip() or not _is_hash(self.initial_rewrite_source_sha256):
                raise ValidationError(
                    "InfoCQR t5_edit requires an initial rewrite and artifact SHA-256"
                )
            if self.profile_kind != "controlled_adapter":
                raise ValidationError("user-supplied T5 candidates are controlled_adapter inputs")
        elif self.initial_rewrite or self.initial_rewrite_source_sha256:
            raise ValidationError("InfoCQR profile does not accept an initial rewrite")
        expected_calls = 2 if self.prompt_profile == "self_edit" else 1
        if self.controls.max_model_calls < expected_calls:
            raise ValidationError("InfoCQR controls cannot reserve required teacher calls")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        conversation = _conversation_view(
            state, common_support=self.release_id == "rewrite.infocqr.common_support_rw_zsl"
        )
        _require_source_license_acknowledgement(self.source_license_acknowledged)
        worker = resolve_worker(self.client, context, label="InfoCQR")
        backend_fp = _bound_fingerprint(
            context.backend,
            context.run_config.get("infocqr_model_fingerprint"),
            label="InfoCQR teacher",
        )
        retriever_fp = _bound_fingerprint(
            context.retriever,
            context.run_config.get("infocqr_retriever_fingerprint"),
            label="InfoCQR retrieval corpus",
        )
        expected_calls = 2 if self.prompt_profile == "self_edit" else 1
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=INFOCQR_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload=_request_payload(
                state,
                conversation,
                dataset_id=self.dataset_id,
                operation="live_teacher",
                prompt_profile=self.prompt_profile,
                prompt_source_sha256=INFOCQR_PROMPT_SOURCE_SHA256[self.prompt_profile],
                dataset_revision=self.dataset_revision,
                split=self.split,
                retrieval_corpus_id=self.retrieval_corpus_id,
                profile_kind=self.profile_kind,
                initial_rewrite=self.initial_rewrite,
                initial_rewrite_source_sha256=self.initial_rewrite_source_sha256,
                cache_artifact_id="",
                cache_artifact_sha256="",
                allow_first_turn_truth_rewrite=False,
                source_license_acknowledged=self.source_license_acknowledged,
                controls=self.controls,
                seed=context.seed,
            ),
            dependency_fingerprints={"backend": backend_fp, "retriever": retriever_fp},
            minimum_reservation={"model_calls": expected_calls},
            protocol_version=INFOCQR_PROTOCOL_VERSION,
        )
        response = call_worker(worker, request)
        payload = _validate_response(
            response.payload,
            state=state,
            conversation=conversation,
            dataset_id=self.dataset_id,
            operation="live_teacher",
            prompt_profile=self.prompt_profile,
            prompt_source_sha256=INFOCQR_PROMPT_SOURCE_SHA256[self.prompt_profile],
            dataset_revision=self.dataset_revision,
            split=self.split,
            retrieval_corpus_id=self.retrieval_corpus_id,
            profile_kind=self.profile_kind,
            cache_artifact_id="",
            cache_artifact_sha256="",
        )
        if response.usage.model_calls != expected_calls:
            raise AlgorithmOutputError("InfoCQR worker usage omits or adds a teacher call")
        return _materialize_patch(
            state,
            payload,
            response.usage,
            worker=worker,
            release_id=self.release_id,
            profile_kind=self.profile_kind,
            implementation_origin="thin_external_wrapper",
            source_license_acknowledged=self.source_license_acknowledged,
            backend_fingerprint=backend_fp,
            retriever_fingerprint=retriever_fp,
        )


@dataclass(slots=True)
class InfoCQRCommonSupportZeroShot(InfoCQRLiveTeacherExternal):
    """Bound zero-shot visible-dialogue profile, not native QReCC execution."""

    prompt_profile: str = 'rw_zsl'
    release_id: str = 'rewrite.infocqr.common_support_rw_zsl'


@dataclass(slots=True)
class InfoCQRMountedCacheReplay:
    """Replay an official-format InfoCQR author cache mounted by the operator."""

    prompt_profile: str
    cache_artifact_id: str
    cache_artifact_sha256: str
    client: WorkerClient | None = None
    source_license_acknowledged: bool = False
    dataset_revision: str = ""
    split: str = "test"
    retrieval_corpus_id: str = ""
    allow_first_turn_truth_rewrite: bool = False
    controls: InfoCQRExternalControls = InfoCQRExternalControls(
        max_model_calls=0,
        max_tokens=0,
        max_cost_usd=0.0,
    )
    method_id: str = "infocqr"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        if self.prompt_profile not in _CACHE_RELEASES:
            raise ValidationError("unsupported InfoCQR author-cache profile")
        if not self.cache_artifact_id.strip() or not _is_hash(self.cache_artifact_sha256):
            raise ValidationError("InfoCQR author cache requires artifact ID and SHA-256")
        _validate_common_configuration(
            source_license_acknowledged=self.source_license_acknowledged,
            dataset_revision=self.dataset_revision,
            split=self.split,
            retrieval_corpus_id=self.retrieval_corpus_id,
        )
        if not isinstance(self.allow_first_turn_truth_rewrite, bool):
            raise ValidationError("allow_first_turn_truth_rewrite must be boolean")
        if self.controls.max_model_calls != 0:
            raise ValidationError("InfoCQR cache replay cannot reserve live model calls")

    @property
    def release_id(self) -> str:
        return _CACHE_RELEASES[self.prompt_profile]

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        conversation = _qrecc_conversation(state)
        _require_source_license_acknowledgement(self.source_license_acknowledged)
        worker = resolve_worker(self.client, context, label="InfoCQR")
        retriever_fp = _bound_fingerprint(
            context.retriever,
            context.run_config.get("infocqr_retriever_fingerprint"),
            label="InfoCQR retrieval corpus",
        )
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=INFOCQR_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload=_request_payload(
                state,
                conversation,
                dataset_id="qrecc",
                operation="cached_replay",
                prompt_profile=self.prompt_profile,
                prompt_source_sha256=INFOCQR_PROMPT_SOURCE_SHA256[self.prompt_profile],
                dataset_revision=self.dataset_revision,
                split=self.split,
                retrieval_corpus_id=self.retrieval_corpus_id,
                profile_kind="artifact_replay",
                initial_rewrite="",
                initial_rewrite_source_sha256="",
                cache_artifact_id=self.cache_artifact_id,
                cache_artifact_sha256=self.cache_artifact_sha256,
                allow_first_turn_truth_rewrite=self.allow_first_turn_truth_rewrite,
                source_license_acknowledged=self.source_license_acknowledged,
                controls=self.controls,
                seed=context.seed,
            ),
            dependency_fingerprints={"backend": {}, "retriever": retriever_fp},
            minimum_reservation={"model_calls": 0},
            protocol_version=INFOCQR_PROTOCOL_VERSION,
        )
        response = call_worker(worker, request)
        payload = _validate_response(
            response.payload,
            state=state,
            conversation=conversation,
            dataset_id="qrecc",
            operation="cached_replay",
            prompt_profile=self.prompt_profile,
            prompt_source_sha256=INFOCQR_PROMPT_SOURCE_SHA256[self.prompt_profile],
            dataset_revision=self.dataset_revision,
            split=self.split,
            retrieval_corpus_id=self.retrieval_corpus_id,
            profile_kind="artifact_replay",
            cache_artifact_id=self.cache_artifact_id,
            cache_artifact_sha256=self.cache_artifact_sha256,
        )
        if (
            response.usage.model_calls
            or response.usage.input_tokens
            or response.usage.output_tokens
        ):
            raise AlgorithmOutputError("InfoCQR replay incorrectly billed historical generation")
        return _materialize_patch(
            state,
            payload,
            response.usage,
            worker=worker,
            release_id=self.release_id,
            profile_kind="controlled_adapter",
            implementation_origin="compatibility_replay",
            source_license_acknowledged=self.source_license_acknowledged,
            backend_fingerprint={},
            retriever_fingerprint=retriever_fp,
        )


def _request_payload(
    state: FrozenAgentState,
    conversation: Mapping[str, Any],
    *,
    dataset_id: str,
    operation: str,
    prompt_profile: str,
    prompt_source_sha256: str,
    dataset_revision: str,
    split: str,
    retrieval_corpus_id: str,
    profile_kind: str,
    initial_rewrite: str,
    initial_rewrite_source_sha256: str,
    cache_artifact_id: str,
    cache_artifact_sha256: str,
    allow_first_turn_truth_rewrite: bool,
    source_license_acknowledged: bool,
    controls: InfoCQRExternalControls,
    seed: int,
) -> Mapping[str, Any]:
    return {
        "operation": operation,
        "task_id": state.task_id
        or f"{conversation['conversation_source']}:"
        f"{conversation['conversation_no']}:{conversation['turn_no']}",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "conversation_source": conversation["conversation_source"],
        "conversation_no": conversation["conversation_no"],
        "turn_no": conversation["turn_no"],
        "query": state.query,
        "new_context": conversation["new_context"],
        "prompt_profile": prompt_profile,
        "prompt_source_sha256": prompt_source_sha256,
        "cache_artifact_id": cache_artifact_id,
        "cache_artifact_sha256": cache_artifact_sha256,
        "allow_first_turn_truth_rewrite": allow_first_turn_truth_rewrite,
        "source_license_acknowledged": source_license_acknowledged,
        "retrieval_corpus_id": retrieval_corpus_id,
        "profile_kind": profile_kind,
        "initial_rewrite": initial_rewrite,
        "initial_rewrite_source_sha256": initial_rewrite_source_sha256,
        "controls": asdict(controls),
        "seed": seed,
    }


def _validate_response(
    payload: Mapping[str, Any],
    *,
    state: FrozenAgentState,
    conversation: Mapping[str, Any],
    dataset_id: str,
    operation: str,
    prompt_profile: str,
    prompt_source_sha256: str,
    dataset_revision: str,
    split: str,
    retrieval_corpus_id: str,
    profile_kind: str,
    cache_artifact_id: str,
    cache_artifact_sha256: str,
) -> Mapping[str, Any]:
    rewrite = required_text(payload, "rewrite", label="InfoCQR response")
    del rewrite
    expected_text = {
        "operation": operation,
        "prompt_profile": prompt_profile,
        "prompt_source_sha256": prompt_source_sha256,
        "postprocess_source_sha256": INFOCQR_POSTPROCESS_SHA256,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "conversation_source": conversation["conversation_source"],
        "retrieval_corpus_id": retrieval_corpus_id,
        "profile_kind": profile_kind,
        "cache_artifact_id": cache_artifact_id,
        "cache_artifact_sha256": cache_artifact_sha256,
    }
    for key, expected in expected_text.items():
        observed = payload.get(key, "")
        if observed != expected:
            raise AlgorithmOutputError(f"InfoCQR worker {key} identity mismatch")
    if (
        payload.get("conversation_no") != conversation["conversation_no"]
        or payload.get("turn_no") != conversation["turn_no"]
    ):
        raise AlgorithmOutputError("InfoCQR worker conversation turn mismatch")
    expected_view_hash = content_hash(
        {"query": state.query, "new_context": conversation["new_context"]}
    )
    if payload.get("conversation_view_hash") != expected_view_hash:
        raise AlgorithmOutputError("InfoCQR worker conversation view hash mismatch")
    raw_candidates = string_tuple(
        payload.get("raw_candidates", ()),
        label="InfoCQR raw_candidates",
        allow_empty=False,
    )
    del raw_candidates
    official_calls = string_tuple(
        payload.get("official_calls", ()),
        label="InfoCQR official_calls",
    )
    reconstructed_calls = string_tuple(
        payload.get("reconstructed_calls", ()),
        label="InfoCQR reconstructed_calls",
    )
    execution_label = payload.get("execution_label")
    if operation == "cached_replay":
        if execution_label != "compatibility_replay" or official_calls or not reconstructed_calls:
            raise AlgorithmOutputError(
                "InfoCQR cache must disclose compatibility replay without official calls"
            )
    elif execution_label != "official_function_calls" or not official_calls or reconstructed_calls:
        raise AlgorithmOutputError("InfoCQR live teacher must disclose its official function calls")
    if payload.get("source_license_id") != "NOASSERTION":
        raise AlgorithmOutputError("InfoCQR worker changed source license identity")
    if payload.get("source_license_acknowledged") is not True:
        raise AlgorithmOutputError("InfoCQR worker omitted license acknowledgement")
    if (
        payload.get("vendored_upstream_source") is not False
        or payload.get("t5_student_checkpoint_used") is not False
        or payload.get("paper_result_parity_claimed") is not False
    ):
        raise AlgorithmOutputError("InfoCQR worker safety declaration mismatch")
    return payload


def _materialize_patch(
    state: FrozenAgentState,
    payload: Mapping[str, Any],
    usage: Any,
    *,
    worker: WorkerClient,
    release_id: str,
    profile_kind: str,
    implementation_origin: str,
    source_license_acknowledged: bool,
    backend_fingerprint: Mapping[str, Any],
    retriever_fingerprint: Mapping[str, Any],
) -> InputPatch:
    rewrite = str(payload["rewrite"])
    trace = ExecutionTrace(
        method_id="infocqr",
        release_id=release_id,
        events=(
            TraceEvent(
                TracePhase.PREPARE,
                "dataset_identity",
                {
                    "dataset_id": payload["dataset_id"],
                    "dataset_revision": payload["dataset_revision"],
                    "split": payload["split"],
                    "conversation_source": payload["conversation_source"],
                    "conversation_no": payload["conversation_no"],
                    "turn_no": payload["turn_no"],
                    "conversation_view_hash": payload["conversation_view_hash"],
                },
            ),
            TraceEvent(
                TracePhase.PROPOSE,
                "official_teacher_or_cache",
                {
                    "operation": payload["operation"],
                    "prompt_profile": payload["prompt_profile"],
                    "execution_label": payload["execution_label"],
                    "execution_calls": (
                        payload["official_calls"]
                        if payload["execution_label"] == "official_function_calls"
                        else payload["reconstructed_calls"]
                    ),
                    "raw_candidates": payload["raw_candidates"],
                },
            ),
            TraceEvent(
                TracePhase.FINALIZE,
                "official_post_processing",
                {
                    "postprocess_status": payload["postprocess_status"],
                    "rewrite": rewrite,
                },
            ),
        ),
        counters={**external_usage_counters(usage), "worker_calls": 1},
    )
    return InputPatch(
        method_id="infocqr",
        release_id=release_id,
        source_state_hash=state.state_hash,
        operations=(
            PatchOperation(
                PatchOp.REPLACE_QUERY,
                rewrite,
                target="query",
                rationale="pinned InfoCQR teacher/cache conversational rewrite",
            ),
        ),
        trace=trace,
        implementation_identity=(
            f"{implementation_origin}:smartyfh/InfoCQR@{INFOCQR_UPSTREAM_COMMIT}:"
            f"{payload['prompt_source_sha256']}"
        ),
        profile_kind=profile_kind,
        metadata={
            "implementation_origin": implementation_origin,
            "source_license_id": "NOASSERTION",
            "source_license_acknowledged": source_license_acknowledged,
            "operation": payload["operation"],
            "prompt_profile": payload["prompt_profile"],
            "prompt_source_sha256": payload["prompt_source_sha256"],
            "postprocess_source_sha256": payload["postprocess_source_sha256"],
            "cache_artifact_id": payload["cache_artifact_id"],
            "cache_artifact_sha256": payload["cache_artifact_sha256"],
            "retrieval_corpus_id": payload["retrieval_corpus_id"],
            "worker_fingerprint": worker_fingerprint(worker, label="InfoCQR"),
            "backend_fingerprint": backend_fingerprint,
            "retriever_fingerprint": retriever_fingerprint,
        },
    )


def _conversation_view(
    state: FrozenAgentState, *, common_support: bool = False
) -> Mapping[str, Any]:
    if common_support:
        return _common_support_conversation(state)
    return _qrecc_conversation(state)


def _qrecc_conversation(state: FrozenAgentState) -> Mapping[str, Any]:
    if not state.query.strip():
        raise ValidationError("InfoCQR requires a non-empty query")
    metadata = state.metadata
    source = metadata.get("conversation_source")
    conversation_no = metadata.get("conversation_no")
    turn_no = metadata.get("turn_no")
    new_context = metadata.get("new_context")
    if not isinstance(source, str) or not source.strip():
        raise ValidationError("InfoCQR requires metadata.conversation_source")
    for name, value in (("conversation_no", conversation_no), ("turn_no", turn_no)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"InfoCQR requires positive metadata.{name}")
    if isinstance(new_context, (str, bytes)) or not isinstance(new_context, Sequence):
        raise ValidationError("InfoCQR requires metadata.new_context as Q/A sequence")
    normalized = tuple(new_context)
    if not all(isinstance(item, str) for item in normalized):
        raise ValidationError("InfoCQR new_context must contain strings")
    if len(normalized) % 2:
        raise ValidationError("InfoCQR new_context must contain complete Q/A pairs")
    if turn_no == 1 and normalized:
        raise ValidationError("InfoCQR first turn cannot have new_context")
    return {
        "conversation_source": source,
        "conversation_no": conversation_no,
        "turn_no": turn_no,
        "new_context": normalized,
    }


def _common_support_conversation(state: FrozenAgentState) -> Mapping[str, Any]:
    """Serialize only complete visible user/assistant pairs into InfoCQR's schema."""

    if not state.query.strip():
        raise ValidationError("InfoCQR requires a non-empty query")
    messages = list(state.messages)
    if (
        messages
        and messages[-1].role == "user"
        and messages[-1].content.strip() == state.query.strip()
    ):
        messages.pop()
    pairs: list[str] = []
    pending_user: str | None = None
    for message in messages:
        role = message.role.lower()
        if role == "user":
            pending_user = message.content
        elif role == "assistant" and pending_user is not None:
            pairs.extend((pending_user, message.content))
            pending_user = None
    benchmark_id = state.metadata.get("benchmark_id", "common_support")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValidationError("InfoCQR common-support state requires metadata.benchmark_id")
    return {
        "conversation_source": benchmark_id,
        "conversation_no": 1,
        "turn_no": len(pairs) // 2 + 1,
        "new_context": tuple(pairs),
    }


def _bound_fingerprint(
    dependency: Any,
    configured: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    if dependency is not None:
        fingerprint = getattr(dependency, "fingerprint", None)
        if not callable(fingerprint):
            raise ExternalDependencyError(f"{label} must expose fingerprint()")
        identity = fingerprint()
    else:
        identity = configured
    if not isinstance(identity, Mapping) or not identity:
        raise ExternalDependencyError(f"{label} fingerprint is required")
    return identity


def _validate_common_configuration(
    *,
    source_license_acknowledged: bool,
    dataset_revision: str,
    split: str,
    retrieval_corpus_id: str,
) -> None:
    if not isinstance(source_license_acknowledged, bool):
        raise ValidationError("InfoCQR source_license_acknowledged must be boolean")
    if not dataset_revision.strip() or not split.strip() or not retrieval_corpus_id.strip():
        raise ValidationError(
            "InfoCQR requires dataset revision, split, and retrieval corpus identity"
        )


def _require_source_license_acknowledgement(acknowledged: bool) -> None:
    if not acknowledged:
        raise ExternalDependencyError(
            "InfoCQR upstream source/cache license is NOASSERTION; explicitly acknowledge "
            "the user-mounted material before execution"
        )


def _is_hash(value: str) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value.lower()) is not None


__all__ = [
    "INFOCQR_POSTPROCESS_SHA256",
    "INFOCQR_PROMPT_SOURCE_SHA256",
    "INFOCQR_PROTOCOL_VERSION",
    "InfoCQRCommonSupportZeroShot",
    "InfoCQRExternalControls",
    "InfoCQRLiveTeacherExternal",
    "InfoCQRMountedCacheReplay",
]
