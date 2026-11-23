"""InfoCQR's trained T5 student, separate from its teacher and saved rewrites."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
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

from .cache_replay import INFOCQR_UPSTREAM_COMMIT
from .external import _qrecc_conversation

INFOCQR_STUDENT_RELEASE = "rewrite.infocqr.t5_student_external"
INFOCQR_STUDENT_MODE = "source_pinned_t5_student.v1"
INFOCQR_STUDENT_SCHEMA = "agentinputlab.infocqr.student-request.v1"
INFOCQR_STUDENT_SOURCE_HASHES = {
    "T5QR/T5test.py": "c9c1b78d8147035ea3c300e7b29c791fb171ce399f8c1269a6e1a11b9bc05eb8",
    "T5QR/T5train.py": "d40bc17e365d9f626fd5bf1fd14fffd3073c0b6772434a78eac6412d6fd2f4de",
    "T5QR/utils/dataCollator.py": "3e4a466b1f729cba306b4953f1e825efde14316e83586c0952dc0e404a974d5d",
    "T5QR/data_preprocessing.ipynb": "6a7dd60ed6f84d89d1313850732599b94e55889cdb694b00f7d2c42df97f0695",
}


@dataclass(frozen=True, slots=True)
class InfoCQRStudentRequest:
    state_hash: str
    task_id: str
    query: str
    conversation_source: str
    conversation_no: int
    turn_no: int
    new_context: tuple[str, ...]
    dataset_revision: str
    checkpoint_manifest_sha256: str
    seed: int
    phase: str = "prepare"
    schema_version: str = INFOCQR_STUDENT_SCHEMA

    def __post_init__(self):
        object.__setattr__(self, "new_context", tuple(self.new_context))
        if self.schema_version != INFOCQR_STUDENT_SCHEMA:
            raise ValidationError("InfoCQR student request schema differs")
        if self.phase not in {"prepare", "generate", "discard"}:
            raise ValidationError("InfoCQR student request phase differs")
        for value in (self.state_hash, self.checkpoint_manifest_sha256):
            if not _sha256(value):
                raise ValidationError("InfoCQR student requires SHA-256 bindings")
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.query, self.conversation_source, self.dataset_revision)):
            raise ValidationError("InfoCQR student requires visible query/conversation/dataset")
        if any(type(v) is not int or v < 1 for v in (self.conversation_no, self.turn_no)):
            raise ValidationError("InfoCQR student conversation/turn IDs must be positive")
        if (len(self.new_context) % 2 or any(not isinstance(v, str) for v in self.new_context)
                or (self.turn_no == 1 and self.new_context)):
            raise ValidationError("InfoCQR student requires complete visible Q/A history")
        if type(self.seed) is not int or self.seed < 0:
            raise ValidationError("InfoCQR student seed must be non-negative")

    @property
    def request_id(self):
        return content_hash(self)

    def external_reservation(self):
        return {"model_calls": 1 if self.phase == "generate" else 0}

    def raise_for_external_status(self, result):
        if isinstance(result, Mapping) and result.get("status") == "empty_output":
            raise AlgorithmOutputError("InfoCQR T5 student produced an empty rewrite; no retry")
        if isinstance(result, Mapping) and result.get("status") == "decode_failed":
            raise AlgorithmOutputError("InfoCQR T5 student decoding failed after generation; usage retained")


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class InfoCQRStudentExternal:
    """Rewrite the native QReCC conversational view using an explicit T5 artifact.

    The rewriter is a fixed auxiliary student, not the downstream backbone. Its
    generation is independently accounted through context.external_runtime.
    Neither author caches, teacher calls nor a generic t5-base can substitute.
    """

    method_id = "infocqr"
    release_id = INFOCQR_STUDENT_RELEASE
    family = SemanticFamily.REFORMULATE

    def __init__(self, *, checkpoint_manifest_sha256: str, dataset_revision: str,
                 profile_kind: str = "controlled_adapter", source_license_acknowledged: bool = False):
        if not _sha256(checkpoint_manifest_sha256):
            raise ValidationError("InfoCQR student requires a trusted checkpoint manifest SHA-256")
        if not isinstance(dataset_revision, str) or not dataset_revision.strip():
            raise ValidationError("InfoCQR student requires the QReCC dataset revision")
        if profile_kind not in {"official_native", "controlled_adapter"}:
            raise ValidationError("InfoCQR student profile kind differs")
        if source_license_acknowledged is not True:
            raise ValidationError("InfoCQR student requires explicit external-source acknowledgement")
        self.checkpoint_manifest_sha256 = checkpoint_manifest_sha256
        self.dataset_revision, self.profile_kind = dataset_revision, profile_kind
        self.source_license_acknowledged = True

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        conversation = _qrecc_conversation(state)
        if state.metadata.get("benchmark_id", "qrecc") not in {"qrecc", "qrecc.main", "qrecc.retrieval"}:
            raise ValidationError("InfoCQR student native release requires QReCC, not a renamed QA dataset")
        if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 63:
            raise BudgetExceededError("InfoCQR student requires one call and its native 63-new-token ceiling")
        runtime = context.external_runtime
        if runtime is None or not callable(getattr(runtime, "run", None)) or not callable(getattr(runtime, "fingerprint", None)):
            raise ExternalDependencyError("InfoCQR student requires InfoCQRStudentRuntime with trained weights")
        identity = runtime.fingerprint()
        if (identity.get("mode") != INFOCQR_STUDENT_MODE
                or identity.get("upstream_revision") != INFOCQR_UPSTREAM_COMMIT
                or identity.get("checkpoint_manifest_sha256") != self.checkpoint_manifest_sha256
                or identity.get("dataset_revision") != self.dataset_revision):
            raise ArtifactError("InfoCQR student runtime/source/checkpoint/dataset binding differs")
        if self.profile_kind == "official_native" and identity.get("provenance_kind") != "author_student":
            raise ArtifactError("a user-trained student must not be labeled as the author's checkpoint")
        request = InfoCQRStudentRequest(
            state_hash=state.state_hash, task_id=state.task_id, query=state.query,
            dataset_revision=self.dataset_revision,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            seed=context.seed, **conversation,
        )
        # Loading weights/tokenization is a zero-generation execution preflight,
        # never a failed scientific attempt charged as one model invocation.
        prepared = runtime.run(request)
        if (not isinstance(prepared, Mapping) or prepared.get("status") != "ready"
                or prepared.get("request_id") != request.request_id
                or prepared.get("identity") != identity
                or type(prepared.get("input_tokens")) is not int
                or not 1 <= prepared["input_tokens"] <= 384
                or normalize_external_usage(prepared.get("usage")).model_calls != 0):
            raise AlgorithmOutputError("InfoCQR student execution preflight differs")
        if (state.budget.max_context_tokens is not None
                and prepared["input_tokens"] > state.budget.max_context_tokens):
            runtime.run(replace(request, phase="discard"))
            raise BudgetExceededError("InfoCQR student exceeds the input-token budget before generation")
        request = replace(request, phase="generate")
        response = runtime.run(request)
        if (not isinstance(response, Mapping) or response.get("request_id") != request.request_id
                or response.get("identity") != identity or response.get("status") != "ok"):
            raise AlgorithmOutputError("InfoCQR student returned another request/runtime or failed status")
        rewrite = response.get("rewrite")
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise AlgorithmOutputError("InfoCQR student returned an empty rewrite; no fallback")
        usage = normalize_external_usage(response.get("usage"))
        if usage.model_calls != 1 or usage.retrieval_calls or usage.output_tokens > 63:
            raise AlgorithmOutputError("InfoCQR student native generation usage differs")
        return InputPatch(
            method_id=self.method_id, release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.REPLACE_QUERY, rewrite, target="query",
                                       rationale="source-pinned trained InfoCQR T5 student"),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id,
                events=(TraceEvent(TracePhase.OBSERVE, "native_t5_student", {
                    "request_id": request.request_id, "source_text_hash": response["source_text_hash"],
                    "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
                    "max_source_length": 384, "generation_max_length_including_start": 64,
                    "target_labels_visible_to_generator": False,
                }),), counters=external_usage_counters(usage)),
            implementation_identity=f"external_student:InfoCQR@{INFOCQR_UPSTREAM_COMMIT}",
            profile_kind=self.profile_kind,
            metadata={"student_runtime": identity, "fixed_auxiliary_not_downstream_backbone": True,
                      "native_source_logic": True, "native_environment_parity_claimed": False,
                      "teacher_or_cache_fallback": False, "implicit_retry": False,
                      "checkpoint_provenance_verification": "caller_receipt_not_authorship_certificate",
                      "paper_result_parity_claimed": False},
        )


__all__ = ["InfoCQRStudentExternal", "InfoCQRStudentRequest"]
