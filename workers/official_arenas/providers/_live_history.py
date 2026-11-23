"""Optional native-message history view; no native state/tool object is rewritten."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from ..protocol import OfficialArenaError


def validate_history_transform(hook):
    if hook is not None and not all(callable(getattr(hook, name, None))
                                    for name in ("transform", "fingerprint")):
        raise OfficialArenaError("native history requires transform and fingerprint methods")


class NativeHistorySession:
    """Separate helper usage from target usage, enforcing their shared caps."""

    def __init__(self, hook, materialized, budget):
        validate_history_transform(hook)
        self.hook = hook
        self.materialized = copy.deepcopy(dict(materialized))
        self.budget = dict(budget)
        self.usage = {}

    def remaining(self, agent_usage):
        fields = {"max_model_calls": "model_calls", "max_output_tokens": "output_tokens"}
        return {cap: int(self.budget[cap]) - int(agent_usage.get(name, 0))
                - int(self.usage.get(name, 0)) for cap, name in fields.items()}

    def messages(self, native, *, encode, agent_usage):
        """Map only system/first-user text; reuse retained native objects exactly."""
        from agentinputlab.integrations.live_history import _balanced_tool_frames

        raw = encode(native)
        logical = self._transform(raw, agent_usage=agent_usage)
        first_user = next((i for i, row in enumerate(raw) if row.get("role") == "user"), None)
        if (first_user is None or first_user == 0 or raw[0].get("role") != "system"
                or len(logical) <= first_user or logical[0].get("role") != "system"
                or logical[first_user].get("role") != "user"):
            raise OfficialArenaError("native history changed the system/first-user framing")
        if list(logical[1:first_user]) != raw[1:first_user]:
            raise OfficialArenaError("native history changed a pre-user native frame")
        text_slots = {0, first_user}
        result, cursor = [], 0
        for i, row in enumerate(logical):
            row = dict(row)
            if i in text_slots:
                original = raw[i]
                if ({k: v for k, v in row.items() if k != "content"}
                        != {k: v for k, v in original.items() if k != "content"}
                        or not isinstance(row.get("content"), str)):
                    raise OfficialArenaError("native history changed non-text first-turn fields")
                result.append(native[i].model_copy(update={"content": row["content"]}, deep=True))
                cursor = i + 1
            else:
                index = next((j for j in range(cursor, len(raw)) if raw[j] == row), None)
                if index is None:
                    raise OfficialArenaError("native history invented or reordered a retained frame")
                result.append(copy.deepcopy(native[index]))
                cursor = index + 1
        if encode(result) != list(logical):
            raise OfficialArenaError("native history message roundtrip changed wire fields")
        if not _balanced_tool_frames(logical):
            raise OfficialArenaError("native history orphaned tool frames")
        return result

    def _transform(self, raw, *, agent_usage):
        """Shared measured helper budget; protocol-specific roundtrips stay separate."""
        from agentinputlab.observability import external_usage_counters, normalize_external_usage

        remaining = self.remaining(agent_usage)
        answer = self.hook.transform(copy.deepcopy(raw), state=copy.deepcopy(self.materialized),
                                     remaining_budget=remaining)
        if not isinstance(answer, Mapping) or set(answer) != {"messages", "usage"}:
            raise OfficialArenaError("native history response fields differ")
        usage = answer["usage"]
        if not isinstance(usage, Mapping) or not {"model_calls", "output_tokens"} <= usage.keys():
            raise OfficialArenaError("native history requires actual call/output usage")
        if usage["model_calls"] and not {"input_tokens", "retrieval_calls"} <= usage.keys():
            raise OfficialArenaError("native history calls require complete actual usage")
        usage = external_usage_counters(normalize_external_usage({
            "model_calls": 0, "retrieval_calls": 0, "input_tokens": 0, "output_tokens": 0, **usage}))
        for key, value in usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
        if not usage["model_calls"] and (usage["input_tokens"] or usage["output_tokens"]):
            raise OfficialArenaError("skipped native history cannot report model tokens")
        if usage["tool_calls"] or usage["retrieval_calls"] or usage["embedding_calls"]:
            raise OfficialArenaError("native history cannot execute tools or retrieval")
        if usage["model_calls"] >= remaining["max_model_calls"] or usage["output_tokens"] >= remaining["max_output_tokens"]:
            raise OfficialArenaError("native history failed to reserve downstream budget")
        logical = answer["messages"]
        if not isinstance(logical, (list, tuple)) or any(not isinstance(row, Mapping) for row in logical):
            raise OfficialArenaError("native history messages are invalid")
        return list(logical)

    def combined_usage(self, target_usage):
        result = dict(target_usage)
        for key, value in self.usage.items():
            result[key] = result.get(key, 0) + value
        return result

    def write_receipt(self, path: Path, *, target_usage):
        value = dict(fingerprint=self.hook.fingerprint(), records=list(getattr(self.hook, "records", ())),
                     history_usage=dict(self.usage), target_usage=dict(target_usage),
                     agent_and_history_usage=self.combined_usage(target_usage),
                     original_shared_budget=self.budget,
                     native_environment_history_NOT_modified=True)
        if hasattr(self, "responses_records"):
            value["responses_view"] = dict(protocol="native.responses.user_boundary.v1",
                                           records=list(self.responses_records))
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise OfficialArenaError("native history receipt changed before finalization")
            return
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)


class ResponsesHistorySession(NativeHistorySession):
    """A reversible optimizer view, never a Chat-Completions wire conversion.

    Real instructions and injected system items are immutable. Each complete
    assistant/tool episode between actual user messages is one atomic text view
    for ACON. Retained Items are deep-copied from the original native objects,
    including reasoning, annotations, call IDs, arguments and tool results.
    Invoke only at new user turns, not inside a function-call continuation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.responses_records = []

    def responses(self, native, *, instructions, agent_usage):
        from agentinputlab.contracts.serialization import canonical_json, content_hash

        if not isinstance(instructions, str) or not instructions:
            raise OfficialArenaError("Responses history requires real text instructions")
        if not isinstance(native, (list, tuple)):
            raise OfficialArenaError("Responses history requires native input Items")
        rows = []
        for item in native:
            row = dict(item) if isinstance(item, Mapping) else item.model_dump(mode="json")
            if not isinstance(row, dict):
                raise OfficialArenaError("Responses history Item is not serializable")
            rows.append(row)
        system_indices = [i for i, row in enumerate(rows) if row.get("role") in {"system", "developer"}]
        policy = canonical_json(dict(instructions=instructions,
                                     input_policy_items=[rows[i] for i in system_indices]))
        raw = [dict(role="system", content=policy)]
        groups = []
        pending_group = []

        def flush():
            if not pending_group:
                return
            pending, seen = set(), set()
            for i in pending_group:
                row = rows[i]
                if row.get("type") == "function_call":
                    call = row.get("call_id")
                    if not isinstance(call, str) or not call or call in seen:
                        raise OfficialArenaError("Responses history has duplicate or missing call_id")
                    pending.add(call)
                    seen.add(call)
                elif row.get("type") == "function_call_output":
                    call = row.get("call_id")
                    if call not in pending:
                        raise OfficialArenaError("Responses history has an orphaned function result")
                    pending.remove(call)
            if pending:
                raise OfficialArenaError("Responses history has an incomplete function episode")
            raw.append(dict(role="assistant", content=canonical_json([rows[i] for i in pending_group])))
            groups.append(list(pending_group))
            pending_group.clear()

        for i, row in enumerate(rows):
            if i in system_indices:
                continue  # Preserved on the native wire and included in the policy view.
            if row.get("role") == "user":
                flush()
                content = row.get("content")
                if isinstance(content, list) and len(content) == 1 and isinstance(content[0], Mapping):
                    if content[0].get("type") != "input_text":
                        raise OfficialArenaError("Responses history user Item is not plain text")
                    content = content[0].get("text")
                if not isinstance(content, str):
                    raise OfficialArenaError("Responses history user Item is not plain text")
                raw.append(dict(role="user", content=content))
                groups.append([i])
            else:
                pending_group.append(i)
        flush()
        if len(raw) < 2 or raw[1]["role"] != "user" or raw[-1]["role"] != "user":
            raise OfficialArenaError("Responses history requires an actual new user-turn boundary")
        logical = self._transform(raw, agent_usage=agent_usage)
        if (len(logical) < 2 or logical[0] != raw[0]
                or set(logical[1]) != {"role", "content"}
                or logical[1].get("role") != "user" or not isinstance(logical[1].get("content"), str)):
            raise OfficialArenaError("Responses history changed immutable policy or first-user framing")
        selected = set(system_indices + groups[0])
        cursor = 2
        for row in logical[2:]:
            index = next((j for j in range(cursor, len(raw)) if raw[j] == row), None)
            if index is None:
                raise OfficialArenaError("Responses history invented or reordered native Items")
            selected.update(groups[index - 1])
            cursor = index + 1
        # The current real user request must remain; no model-generated turn is inserted.
        if not set(groups[-1]) <= selected:
            raise OfficialArenaError("Responses history removed the current user turn")
        result = [copy.deepcopy(item) for i, item in enumerate(native) if i in selected]
        first_index = groups[0][0]
        first_output = sorted(selected).index(first_index)
        original = rows[first_index]
        content = logical[1]["content"]
        if isinstance(original["content"], list):
            content = [{**original["content"][0], "text": content}]
        if isinstance(result[first_output], Mapping):
            result[first_output] = {**result[first_output], "content": content}
        else:
            result[first_output] = result[first_output].model_copy(update={"content": content}, deep=True)
        self.responses_records.append(dict(native_input_hash=content_hash(rows),
            optimizer_view_hash=content_hash(raw), retained_source_item_indices=sorted(selected),
            policy_unchanged=True, native_items_NOT_chat_messages=True))
        return result, instructions
