"""Stable public enums used by manifests, traces, and result files."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return self.value


class SemanticFamily(StrEnum):
    REFORMULATE = "reformulate"
    DECOMPOSE = "decompose"
    ACQUIRE = "acquire"
    CLARIFY = "clarify"
    CURATE = "curate"
    COMPOSITE = "composite"
    NO_OP = "no_op"


class ReleaseTier(StrEnum):
    CORE = "core"
    OPTIONAL = "optional"


class ExecutionKind(StrEnum):
    RUNTIME = "runtime"
    COMPONENT = "component"
    COMPOSITE = "composite"
    TRAINING_RECIPE = "training_recipe"
    ARTIFACT_REPLAY = "artifact_replay"


class ImplementationKind(StrEnum):
    OFFICIAL_ADAPTER = "official_adapter"
    COMMUNITY_IMPLEMENTATION = "community_implementation"
    PAPER_REIMPLEMENTATION = "paper_reimplementation"
    THIN_EXTERNAL_WRAPPER = "thin_external_wrapper"
    ARTIFACT_REPLAY = "artifact_replay"


class ProfileKind(StrEnum):
    OFFICIAL_NATIVE = "official_native"
    CONTROLLED_ADAPTER = "controlled_adapter"
    FACTORIAL_ADAPTER = "factorial_adapter"


class SupportStatus(StrEnum):
    SOURCE_AUDITED = "source_audited"
    PLANNED = "planned"
    EXTERNAL_ONLY = "external_only"
    NATIVE_ONLY = "native_only"
    RETRAIN_REQUIRED = "retrain_required"
    BLOCKED = "blocked"
    EXPERIMENTAL = "experimental"
    SUPPORTED = "supported"


class PatchOp(StrEnum):
    REPLACE_QUERY = "replace_query"
    APPEND_QUERY = "append_query"
    REPLACE_INSTRUCTION = "replace_instruction"
    APPEND_INSTRUCTION = "append_instruction"
    ADD_CONTEXT = "add_context"
    REPLACE_CONTEXT = "replace_context"
    ADD_MESSAGE = "add_message"
    REPLACE_MESSAGES = "replace_messages"
    SET_METADATA = "set_metadata"
    REQUEST_CLARIFICATION = "request_clarification"
    NO_OP = "no_op"


class TracePhase(StrEnum):
    PREPARE = "prepare"
    PROPOSE = "propose"
    OBSERVE = "observe"
    FINALIZE = "finalize"
    ERROR = "error"
