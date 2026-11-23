"""ACON history at real agent user-turn boundaries with a shared run budget.

The native environment remains authoritative. Only the agent-visible message
view is compressed; tool databases, simulator state and evaluator data never
enter this adapter. Tool calls/results are serialized for the text optimizer
but preserved verbatim whenever retained in the downstream protocol.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from agentinputlab.contracts import ContextBlock, ExecutionBudget, FrozenAgentState, Message
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner


class NativeHistoryInputError(ValidationError):
    """A native input cannot enter this history interface, before worker/model IO.

    ``code`` is stable for caller-side routing. It does not classify a model
    outcome or make an entire benchmark inapplicable. No message content is
    included in the error metadata.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_native_history_prefix(messages, *, prefix_policy="strict_system_user") -> int:
    """Validate native prefix framing without constructing a worker or model.

    Return the original first-user index. This check neither inserts a system
    frame nor validates/compresses subsequent tool episodes; the native protocol
    adapter remains responsible for their complete roundtrip.
    """
    if not isinstance(prefix_policy, str) or prefix_policy not in {
            "strict_system_user", "preserve_pre_user_assistant"}:
        raise NativeHistoryInputError("unknown_prefix_policy", "unknown native history prefix policy")
    if (not isinstance(messages, Sequence) or isinstance(messages, (str, bytes))
            or any(not isinstance(row, Mapping) for row in messages)):
        raise NativeHistoryInputError("invalid_message_sequence", "native history requires a message sequence")
    missing = "live ACON requires the native session to start with system then user"
    if len(messages) < 2 or messages[0].get("role") != "system":
        raise NativeHistoryInputError("native_system_user_boundary_missing", missing)
    if messages[1].get("role") == "user":
        return 1
    if prefix_policy != "preserve_pre_user_assistant":
        raise NativeHistoryInputError("native_system_user_boundary_missing", missing)
    first_user = next((i for i, row in enumerate(messages) if row.get("role") == "user"), None)
    if first_user is None:
        raise NativeHistoryInputError("native_first_user_missing", "native greeting prefix has no first user")
    prefix = messages[1:first_user]
    if not prefix or any(row.get("role") != "assistant"
            or not isinstance(row.get("content"), str) or not row["content"].strip()
            or row.get("tool_calls") or row.get("function_call") for row in prefix):
        raise NativeHistoryInputError("native_greeting_prefix_unsupported",
                                     "native pre-user prefix must contain only plain assistant greetings")
    return first_user


def _view(message: Mapping[str, Any]) -> Message:
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValidationError("native history message has no role")
    content = message.get("content")
    if message.get("tool_calls") or role in {"tool", "function"}:
        # Upstream MemoryManager consumes USER/ASSISTANT text. Preserve tool
        # observations as explicit text in that view, never as hidden metadata.
        from agentinputlab.contracts.serialization import canonical_json
        content = canonical_json(dict(message))
        role = "assistant"
    if not isinstance(content, str):
        raise ValidationError("native text history message has no string content")
    return Message(role=role, content=content, name=message.get("name"))


def _balanced_tool_frames(messages: Sequence[Mapping[str, Any]]) -> bool:
    pending = set()
    for message in messages:
        if message.get("function_call") or message.get("role") == "function":
            return False  # legacy function framing needs its own explicit adapter
        calls = message.get("tool_calls") or []
        if calls:
            if pending:
                return False
            pending = {call.get("id") for call in calls}
            if None in pending or len(pending) != len(calls):
                return False
        elif message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending:
                return False
            pending.remove(identifier)
        elif pending:
            return False
    return not pending


class ACONLiveHistory:
    """One instance per native episode; keep metadata and causal history across turns.

    The context must bind the real optimizer and native worker. A downstream
    call and its per-call output allowance are reserved before any optimizer
    request. Initial, intra-tool-turn and insufficient-budget branches do not
    consume optimizer calls. Genuine optimizer failures propagate, never become
    successful NO_OPs. ``records`` retains every decision and observed usage.
    """

    def __init__(self, method, context, *, per_call_output_tokens: int = 512,
                 prefix_policy: str = "strict_system_user"):
        if method.release_id != "curate.acon.history_runtime":
            raise ValidationError("live ACON history requires the exact history runtime release")
        if type(per_call_output_tokens) is not int or per_call_output_tokens < 1:
            raise ValidationError("per-call history output allowance must be positive")
        if not isinstance(prefix_policy, str) or prefix_policy not in {
                "strict_system_user", "preserve_pre_user_assistant"}:
            raise ValidationError("unknown native history prefix policy")
        self.method, self.context = method, context
        self.per_call_output_tokens = per_call_output_tokens
        self.prefix_policy = prefix_policy
        self.records: list[dict] = []
        self._raw: list[dict] = []
        self._logical: list[dict] = []
        self._metadata: dict = {}

    def fingerprint(self) -> dict:
        fingerprint = {"release": self.method.release_id, "integration": "acon.native_user_boundary.v2",
                "per_call_output_tokens": self.per_call_output_tokens,
                "optimizer_tokenizer": "source-native-bound-not-target-approximation",
                "agent_tool_frames_preserved": True,
                "worker": dict(self.context.external_runtime.fingerprint())}
        if self.prefix_policy != "strict_system_user":
            fingerprint["prefix_policy"] = self.prefix_policy
        return to_jsonable(fingerprint)

    def _native_optimizer_view(self, logical):
        """Keep a declared native greeting immutable, outside the optimizer view.

        No message is invented, moved or dropped from the downstream wire.
        The native optimizer still receives its original system/user framing;
        only plain assistant greetings preceding the first user are excluded
        from that mutable view and reinserted verbatim after materialization.
        """
        first_user = validate_native_history_prefix(logical, prefix_policy=self.prefix_policy)
        if first_user == 1:
            return logical, []
        prefix = logical[1:first_user]
        return [logical[0], *logical[first_user:]], deepcopy(prefix)

    def transform(self, messages, *, state, remaining_budget):
        raw = deepcopy(list(messages))
        if self._raw and (len(raw) < len(self._raw) or raw[:len(self._raw)] != self._raw):
            raise ValidationError("native history changed before its append-only boundary")
        logical = deepcopy(self._logical) + raw[len(self._raw):] if self._raw else deepcopy(raw)
        native_logical, immutable_prefix = self._native_optimizer_view(logical)
        initial = {"input_raw_hash": content_hash(raw), "input_logical_hash": content_hash(logical),
                   "remaining_budget": dict(remaining_budget)}
        if immutable_prefix:
            initial.update(prefix_policy=self.prefix_policy,
                           immutable_native_prefix_hash=content_hash(immutable_prefix),
                           immutable_native_prefix_length=len(immutable_prefix))
        reason = None
        if logical[-1].get("role") != "user":
            reason = "intra_user_turn_tool_protocol_preserved"
        elif (remaining_budget["max_model_calls"] < 2
              or remaining_budget["max_output_tokens"] < 2 * self.per_call_output_tokens):
            reason = "downstream_call_and_output_reserved"
        if reason:
            self._raw, self._logical = raw, logical
            record = {**initial, "status": "integration_skip", "reason": reason,
                      "usage": {"model_calls": 0, "output_tokens": 0}}
            self.records.append(record)
            return {"messages": deepcopy(logical), "usage": dict(record["usage"])}
        view = tuple(_view(message) for message in native_logical)
        # Reject protocol-orphaning assembly before the optimizer has side effects.
        from agentinputlab.algorithms.curate.acon.common import preserved_turn_indices
        indices = preserved_turn_indices(view, self.method.controls.preserve_last_k_turns)
        candidate_tail = [native_logical[index] for index in indices]
        if not _balanced_tool_frames(candidate_tail):
            self._raw, self._logical = raw, logical
            record = {**initial, "status": "integration_skip", "reason": "preserved_tail_has_unpaired_tool_frames",
                      "usage": {"model_calls": 0, "output_tokens": 0}}
            self.records.append(record)
            return {"messages": deepcopy(logical), "usage": dict(record["usage"])}
        frozen = FrozenAgentState(
            query=state["query"], instruction=state.get("instruction", ""),
            task_id=state.get("task_id", ""), messages=view,
            context=tuple(ContextBlock(**block) for block in state.get("context", [])),
            tool_names=tuple(state.get("tool_names", [])),
            metadata={**state.get("metadata", {}), **self._metadata},
            budget=ExecutionBudget(max_model_calls=remaining_budget["max_model_calls"] - 1,
                                   max_output_tokens=remaining_budget["max_output_tokens"] - self.per_call_output_tokens))
        monitor = RunMonitor()
        try:
            result = InterventionRunner(monitor=monitor).run(self.method, frozen, self.context)
        except Exception as exc:
            self.records.append({**initial, "status": "failed_terminal", "error_type": type(exc).__name__,
                                 "error": str(exc), "monitor_snapshots": to_jsonable(monitor.list_snapshots())})
            raise
        patched = result.patch.apply(frozen)
        if result.patch.metadata.get("applied"):
            if tuple(patched.messages[2:]) != tuple(view[index] for index in indices):
                raise ValidationError("ACON preserved messages differ from source-bound indices")
            materialized = [dict(native_logical[0], content=patched.messages[0].content),
                            *deepcopy(immutable_prefix),
                            dict(native_logical[1], content=patched.messages[1].content), *deepcopy(candidate_tail)]
        else:
            materialized = logical
        if not _balanced_tool_frames(materialized):
            raise ValidationError("ACON materialization would orphan a native tool call")
        self._metadata = {key: value for key, value in patched.metadata.items() if key.startswith("acon_")}
        self._raw, self._logical = raw, materialized
        usage = to_jsonable(result.usage)
        self.records.append({**initial, "status": "applied" if result.patch.metadata.get("applied") else "native_threshold_skip",
                             "output_logical_hash": content_hash(materialized), "usage": usage,
                             "intervention": to_jsonable(result)})
        return {"messages": deepcopy(materialized), "usage": dict(usage)}
