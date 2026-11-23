"""Typed, auditable input patches and deterministic materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .enums import PatchOp
from .errors import StateConflictError, ValidationError
from .serialization import content_hash, deep_freeze_jsonable, freeze_mapping, to_jsonable
from .state import ContextBlock, FrozenAgentState, Message
from .trace import ExecutionTrace


@dataclass(frozen=True, slots=True)
class PatchOperation:
    op: PatchOp
    value: Any = None
    target: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_patch_value(self.value))
        if self.op is PatchOp.NO_OP and self.value not in (None, ""):
            raise ValidationError("NO_OP cannot carry a replacement value")


@dataclass(frozen=True, slots=True)
class InputPatch:
    method_id: str
    release_id: str
    source_state_hash: str
    operations: tuple[PatchOperation, ...]
    trace: ExecutionTrace
    implementation_identity: str
    profile_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not all(isinstance(operation, PatchOperation) for operation in operations):
            raise ValidationError("patch operations must be PatchOperation instances")
        if not isinstance(self.trace, ExecutionTrace):
            raise ValidationError("patch trace must be an ExecutionTrace")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.operations:
            raise ValidationError("patch requires at least one operation; use NO_OP explicitly")
        if self.trace.method_id != self.method_id or self.trace.release_id != self.release_id:
            raise ValidationError("trace identity does not match patch identity")

    @classmethod
    def no_op(
        cls,
        state: FrozenAgentState,
        *,
        method_id: str = "no_op",
        release_id: str = "baseline.no_op",
        rationale: str = "",
        profile_kind: str = "controlled_adapter",
    ) -> InputPatch:
        trace = ExecutionTrace(method_id=method_id, release_id=release_id)
        return cls(
            method_id=method_id,
            release_id=release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.NO_OP, rationale=rationale),),
            trace=trace,
            implementation_identity="builtin",
            profile_kind=profile_kind,
        )

    @property
    def patch_hash(self) -> str:
        return content_hash(self)

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def apply(self, state: FrozenAgentState) -> FrozenAgentState:
        """Apply operations in order after checking the paired-state identity."""

        if state.state_hash != self.source_state_hash:
            raise StateConflictError(
                f"patch expected state {self.source_state_hash}, got {state.state_hash}"
            )
        current = state
        for operation in self.operations:
            current = _apply_operation(current, operation)
        return current


def _apply_operation(state: FrozenAgentState, operation: PatchOperation) -> FrozenAgentState:
    if operation.op is PatchOp.NO_OP:
        return state
    if operation.op is PatchOp.REPLACE_QUERY:
        return replace(state, query=str(operation.value))
    if operation.op is PatchOp.APPEND_QUERY:
        separator = "\n\n" if state.query else ""
        return replace(state, query=f"{state.query}{separator}{operation.value}")
    if operation.op is PatchOp.REPLACE_INSTRUCTION:
        return replace(state, instruction=str(operation.value))
    if operation.op is PatchOp.APPEND_INSTRUCTION:
        separator = "\n\n" if state.instruction else ""
        return replace(state, instruction=f"{state.instruction}{separator}{operation.value}")
    if operation.op is PatchOp.ADD_CONTEXT:
        blocks = _coerce_context(operation.value)
        return replace(state, context=(*state.context, *blocks))
    if operation.op is PatchOp.REPLACE_CONTEXT:
        return replace(state, context=_coerce_context(operation.value))
    if operation.op is PatchOp.ADD_MESSAGE:
        message = operation.value
        if isinstance(message, Mapping):
            message = Message(**message)
        if not isinstance(message, Message):
            raise ValidationError("ADD_MESSAGE requires Message or mapping")
        return replace(state, messages=(*state.messages, message))
    if operation.op is PatchOp.REPLACE_MESSAGES:
        return replace(state, messages=_coerce_messages(operation.value))
    if operation.op is PatchOp.SET_METADATA:
        if not isinstance(operation.value, Mapping):
            raise ValidationError("SET_METADATA requires a mapping")
        merged = dict(state.metadata)
        merged.update(operation.value)
        return replace(state, metadata=merged)
    if operation.op is PatchOp.REQUEST_CLARIFICATION:
        merged = dict(state.metadata)
        merged["clarification_request"] = str(operation.value)
        return replace(state, metadata=merged)
    raise ValidationError(f"unsupported patch operation: {operation.op}")


def _coerce_context(value: Any) -> tuple[ContextBlock, ...]:
    values = value if isinstance(value, (tuple, list)) else (value,)
    blocks: list[ContextBlock] = []
    for item in values:
        if isinstance(item, ContextBlock):
            blocks.append(item)
        elif isinstance(item, str):
            blocks.append(ContextBlock(text=item))
        elif isinstance(item, Mapping):
            blocks.append(ContextBlock(**item))
        else:
            raise ValidationError("context operation requires ContextBlock/string/mapping")
    return tuple(blocks)


def _coerce_messages(value: Any) -> tuple[Message, ...]:
    values = value if isinstance(value, (tuple, list)) else (value,)
    messages: list[Message] = []
    for item in values:
        if isinstance(item, Message):
            messages.append(item)
        elif isinstance(item, Mapping):
            messages.append(Message(**item))
        else:
            raise ValidationError("REPLACE_MESSAGES requires Message instances or mappings")
    return tuple(messages)


def _freeze_patch_value(value: Any) -> Any:
    if isinstance(value, (ContextBlock, Message)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_patch_value(item) for item in value)
    return deep_freeze_jsonable(value)
