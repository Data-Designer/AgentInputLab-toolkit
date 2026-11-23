"""Official ToolTalk full-set provider at the pinned Microsoft revision."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..protocol import (
    OfficialArenaError,
    ProviderArmResult,
    content_hash,
    intervention_usage,
    materialized_state,
    native_extra_body,
    native_timeout_options,
    require_materialized_budget,
    resolve_shared_budget_config,
    validate_dependency_hashes,
    visible_shared_budget,
)
from ._message_compat import coalesce_chat_system_messages

_BUDGET_STOP = "I could not complete the task within the shared run budget."
_SENT2VEC_REVISION = "c2830440298372a76a5deccdac1b25da71530009"
_DISTILBERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def _install_sent2vec_compatibility() -> None:
    """Install pdrm83/sent2vec's exact BERT-CLS path without unused extras.

    ToolTalk imports ``sent2vec.vectorizer.Vectorizer`` globally, while the
    official package imports gensim and spaCy even though its default runtime
    uses only DistilBERT.  This source-locked shim preserves the 0.3.0 default
    algorithm (tokenizer.encode, zero padding, first hidden-state CLS vector)
    and pins the otherwise floating Hugging Face model revision.
    """

    if "sent2vec.vectorizer" in sys.modules:
        return

    class Vectorizer:
        def __init__(self, pretrained_weights: str = "distilbert-base-uncased", **_: Any):
            if pretrained_weights != "distilbert-base-uncased":
                raise OfficialArenaError("ToolTalk sent2vec model differs")
            import torch
            from transformers import DistilBertModel, DistilBertTokenizer

            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = DistilBertTokenizer.from_pretrained(
                pretrained_weights,
                revision=_DISTILBERT_REVISION,
            )
            self._model = DistilBertModel.from_pretrained(
                pretrained_weights,
                revision=_DISTILBERT_REVISION,
            )
            self.vectors: list[Any] = []

        def run(self, sentences: list[str], **_: Any) -> None:
            import numpy as np

            if not isinstance(sentences, list) or any(
                not isinstance(item, str) for item in sentences
            ):
                raise TypeError("sent2vec requires a list of strings")
            tokenized = [
                self._tokenizer.encode(item, add_special_tokens=True) for item in sentences
            ]
            maximum = max(len(item) for item in tokenized)
            padded = np.array([item + [0] * (maximum - len(item)) for item in tokenized])
            input_ids = self._torch.tensor(padded).type(self._torch.LongTensor).to(self._device)
            model = self._model.to(self._device)
            model.eval()
            with self._torch.no_grad():
                vectors = model(input_ids)[0][:, 0, :].cpu().numpy()
            self.vectors.extend(vectors[index] for index in range(vectors.shape[0]))

    package = types.ModuleType("sent2vec")
    package.__path__ = []  # type: ignore[attr-defined]
    vectorizer = types.ModuleType("sent2vec.vectorizer")
    vectorizer.Vectorizer = Vectorizer
    package.vectorizer = vectorizer  # type: ignore[attr-defined]
    sys.modules["sent2vec"] = package
    sys.modules["sent2vec.vectorizer"] = vectorizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _system_prompt(metadata: Mapping[str, Any]) -> str:
    return (
        "You are a helpful assistant. Here is some user data:"
        f"\nlocation: {metadata['location']}"
        f"\ntimestamp: {metadata['timestamp']}"
        f"\nusername (if logged in): {metadata.get('username')}"
    )


def _tool_schemas() -> list[dict[str, Any]]:
    from tooltalk.apis import ALL_APIS

    schemas: list[dict[str, Any]] = []
    for api in ALL_APIS:
        legacy = dict(api.to_openai_doc(False))
        required = legacy.pop("required", [])
        parameters = dict(legacy["parameters"])
        parameters["required"] = list(required)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": legacy["name"],
                    "description": legacy["description"],
                    "parameters": parameters,
                },
            }
        )
    return schemas


class _Predictor:
    """OpenAI-compatible predictor preserving ToolTalk's official turn loop."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        materialized: Mapping[str, Any],
        budget: Mapping[str, Any],
        optimizer_usage: Mapping[str, int | float],
        seed: int,
        max_output_tokens_per_call: int,
        history_transform: Any = None,
        timeout_seconds: float = 600.0,
        extra_body: Mapping[str, Any] | None = None,
        extra_body_mode: str = "merge",
    ) -> None:
        from openai import OpenAI

        self.extra_body = native_extra_body({"extra_body": {} if extra_body is None else extra_body,
            "extra_body_mode": extra_body_mode},
            "extra_body", defaults={"chat_template_kwargs": {"enable_thinking": False}})
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                            **native_timeout_options({"timeout_seconds": timeout_seconds}, "timeout_seconds"))
        self.model = model
        self.materialized = materialized
        self.seed = seed
        self.tools = _tool_schemas()
        self.first_prediction = True
        self.model_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_model_calls = max(
            0, int(budget["max_model_calls"]) - int(optimizer_usage.get("model_calls", 0))
        )
        self.max_tool_calls = max(
            0, int(budget["max_tool_calls"]) - int(optimizer_usage.get("tool_calls", 0))
        )
        self.max_output_tokens = max(
            0, int(budget["max_output_tokens"]) - int(optimizer_usage.get("output_tokens", 0))
        )
        context_cap = budget.get("max_context_tokens")
        self.max_input_tokens = (
            2**63 - 1
            if context_cap is None
            else max(0, int(context_cap) - int(optimizer_usage.get("input_tokens", 0)))
        )
        self.max_output_tokens_per_call = max_output_tokens_per_call
        self.hard_stops = 0
        self.history_transform = history_transform
        self.history_usage: dict[str, int | float] = {}
        self._history_raw_boundary: list[dict[str, Any]] | None = None
        self._history_logical_boundary: list[dict[str, Any]] | None = None

    def _history_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.history_transform is None:
            return messages
        # ToolTalk's official executor starts every user turn from committed
        # ground-truth history. Predicted tool actions are a temporary rollout,
        # not an append-only extension of that history across user turns. The
        # transformer therefore observes only committed user boundaries; reuse
        # its view plus the exact temporary tool suffix within the same turn.
        if not messages or messages[-1].get("role") != "user":
            raw = self._history_raw_boundary
            logical = self._history_logical_boundary
            if raw is None or logical is None:
                raise OfficialArenaError("ToolTalk live history requires an initial user boundary")
            if len(messages) < len(raw) or messages[:len(raw)] != raw:
                raise OfficialArenaError("ToolTalk temporary tool rollout changed its user boundary")
            result = copy.deepcopy(logical) + copy.deepcopy(messages[len(raw):])
            from agentinputlab.integrations.live_history import _balanced_tool_frames
            if not _balanced_tool_frames(result):
                raise OfficialArenaError("ToolTalk temporary rollout orphaned native tool frames")
            return result
        from agentinputlab.observability import external_usage_counters, normalize_external_usage

        remaining = {
            "max_model_calls": self.max_model_calls - self.model_calls
            - int(self.history_usage.get("model_calls", 0)),
            "max_output_tokens": self.max_output_tokens - self.output_tokens
            - int(self.history_usage.get("output_tokens", 0)),
        }
        transformed = self.history_transform.transform(
            copy.deepcopy(messages), state=copy.deepcopy(self.materialized),
            remaining_budget=dict(remaining),
        )
        if not isinstance(transformed, Mapping) or set(transformed) != {"messages", "usage"}:
            raise OfficialArenaError("ToolTalk history transform response fields differ")
        raw_usage = transformed["usage"]
        if not isinstance(raw_usage, Mapping) or not {"model_calls", "output_tokens"} <= raw_usage.keys():
            raise OfficialArenaError("ToolTalk history transform requires actual call/output usage")
        if raw_usage["model_calls"] and not {"input_tokens", "retrieval_calls"} <= raw_usage.keys():
            raise OfficialArenaError("ToolTalk history model calls require complete actual usage")
        usage = external_usage_counters(normalize_external_usage({
            "model_calls": 0, "retrieval_calls": 0, "input_tokens": 0, "output_tokens": 0,
            **raw_usage,
        }))
        if not usage["model_calls"] and (usage["input_tokens"] or usage["output_tokens"]):
            raise OfficialArenaError("ToolTalk skipped history transform cannot report model tokens")
        for key, value in usage.items():
            self.history_usage[key] = self.history_usage.get(key, 0) + value
        if usage["tool_calls"] or usage["retrieval_calls"] or usage["embedding_calls"]:
            raise OfficialArenaError("ToolTalk history transform cannot execute tools or retrieval")
        if usage["model_calls"] >= remaining["max_model_calls"] or usage["output_tokens"] >= remaining["max_output_tokens"]:
            raise OfficialArenaError("ToolTalk history transform failed to reserve downstream budget")
        result = transformed["messages"]
        if not isinstance(result, (list, tuple)) or len(result) < 2 or any(
            not isinstance(row, Mapping) or not isinstance(row.get("role"), str)
            for row in result
        ):
            raise OfficialArenaError("ToolTalk history transform messages are invalid")
        if result[0]["role"] != "system" or result[1]["role"] != "user":
            raise OfficialArenaError("ToolTalk history transform must preserve system/user session framing")
        from agentinputlab.integrations.live_history import _balanced_tool_frames
        if not _balanced_tool_frames(result):
            raise OfficialArenaError("ToolTalk history transform orphaned native tool frames")
        self._history_raw_boundary = copy.deepcopy(messages)
        self._history_logical_boundary = copy.deepcopy([dict(row) for row in result])
        return copy.deepcopy(self._history_logical_boundary)

    def _messages(
        self,
        metadata: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        additions = [str(self.materialized.get("instruction", ""))]
        additions.extend(
            str(item.get("text", ""))
            for item in self.materialized.get("context", [])
            if isinstance(item, Mapping)
        )
        official = _system_prompt(metadata)
        injected = "\n\n".join(item for item in additions if item and item != official)
        system = official if not injected else official + "\n\n" + injected
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        first_user = True
        tool_index = 0
        for turn in history:
            role = turn.get("role")
            if role in {"user", "assistant"}:
                text = turn.get("text", "")
                if role == "user" and first_user:
                    text = self.materialized["query"]
                    first_user = False
                messages.append({"role": role, "content": str(text)})
            elif role == "api":
                tool_index += 1
                call_id = f"tooltalk_{tool_index:03d}"
                request = turn["request"]
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": request["api_name"],
                                    "arguments": json.dumps(
                                        request["parameters"],
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(
                            {
                                "response": turn.get("response"),
                                "exception": turn.get("exception"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
        return coalesce_chat_system_messages(messages)

    def __call__(
        self,
        metadata: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        visible_metadata = self.materialized.get("metadata", {})
        clarification = (
            visible_metadata.get("clarification_request")
            if isinstance(visible_metadata, Mapping)
            else None
        )
        if self.first_prediction and isinstance(clarification, str) and clarification.strip():
            self.first_prediction = False
            return {"role": "assistant", "text": clarification}
        self.first_prediction = False

        remaining_output = self.max_output_tokens - self.output_tokens - int(self.history_usage.get("output_tokens", 0))
        if self.model_calls + self.history_usage.get("model_calls", 0) >= self.max_model_calls or remaining_output < 1:
            self.hard_stops += 1
            return {"role": "assistant", "text": _BUDGET_STOP}
        messages = self._messages(metadata, history)
        messages = self._history_messages(messages)
        remaining_output = self.max_output_tokens - self.output_tokens - int(self.history_usage.get("output_tokens", 0))
        if self.input_tokens + self.history_usage.get("input_tokens", 0) > self.max_input_tokens:
            self.hard_stops += 1
            return {"role": "assistant", "text": _BUDGET_STOP}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0,
            seed=self.seed + self.model_calls,
            max_tokens=min(
                self.max_output_tokens_per_call,
                remaining_output,
            ),
            extra_body=copy.deepcopy(self.extra_body),
        )
        self.model_calls += 1
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        if (self.input_tokens + self.history_usage.get("input_tokens", 0) > self.max_input_tokens
                or self.output_tokens + self.history_usage.get("output_tokens", 0) > self.max_output_tokens):
            self.hard_stops += 1
            return {"role": "assistant", "text": _BUDGET_STOP}
        message = response.choices[0].message
        calls = list(message.tool_calls or ())
        if calls:
            if self.tool_calls >= self.max_tool_calls:
                self.hard_stops += 1
                return {"role": "assistant", "text": _BUDGET_STOP}
            call = calls[0]
            self.tool_calls += 1
            try:
                parameters = json.loads(call.function.arguments)
                if not isinstance(parameters, dict):
                    parameters = None
            except (TypeError, json.JSONDecodeError):
                parameters = None
            return {
                "role": "api",
                "request": {
                    "api_name": call.function.name,
                    "parameters": parameters,
                },
                "metadata": {
                    "model": self.model,
                    "finish_reason": response.choices[0].finish_reason,
                },
            }
        return {"role": "assistant", "text": message.content or ""}

    def usage(self) -> dict[str, int | float]:
        usage = {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": 0.0,
            "unpriced_model_calls": self.model_calls,
        }
        for key, value in self.history_usage.items():
            usage[key] = usage.get(key, 0) + value
        return usage


class ToolTalkProvider:
    benchmark_id = "tooltalk.full"
    source_revision = "e05f4ce6132c80ed33392b81535b077d56ab28fd"
    scorer_id = "tooltalk.official_conversation_success.v1"
    scorer_revision = "122fb0872f116c9a3b531ea24ba3d1b19bd8c75e4785863dfb4add9617fa157e"
    conversation_set_sha256 = (
        "ce9a691f52830a3aef7e5c0665bf30676c536bfdd0ffb0e534770d10d793227d"
    )

    _REQUIRED = frozenset(
        {
            "dataset_parts",
            "limit",
            "api_mode",
            "agent_model",
            "agent_base_url",
            "agent_key_env",
            "locked_dependency_hashes",
        }
    )
    _OPTIONAL = frozenset({"budget_override", "agent_timeout_seconds", "agent_extra_body", "agent_extra_body_mode"})

    def __init__(self, *, history_transform: Any = None) -> None:
        if history_transform is not None and not all(
            callable(getattr(history_transform, name, None)) for name in ("transform", "fingerprint")
        ):
            raise OfficialArenaError("ToolTalk history transform requires transform and fingerprint")
        self.history_transform = history_transform

    def doctor(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            self._validate_config(config)
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if head != self.source_revision:
                raise OfficialArenaError("ToolTalk checkout revision differs")
            _install_sent2vec_compatibility()
            from tooltalk.apis import ALL_APIS
            from tooltalk.evaluation.tool_executor import ToolExecutor

            if len(ALL_APIS) != 28:
                raise OfficialArenaError("ToolTalk official tool catalog differs")
            counts = {part: len(list((Path("data") / part).glob("*.json"))) for part in ("easy", "tooltalk")}
            if counts != {"easy": 28, "tooltalk": 50}:
                raise OfficialArenaError("ToolTalk full-set inventory differs")
            ToolExecutor(init_database_dir="data/databases")
        except Exception as exc:
            return (f"{type(exc).__name__}: {exc}",)
        return ()

    def capture(
        self,
        config: Mapping[str, Any],
        *,
        output_root: Path,
    ) -> tuple[Mapping[str, Any], ...]:
        del output_root
        parsed = self._validate_config(config)
        _install_sent2vec_compatibility()
        schemas = _tool_schemas()
        tool_names = [item["function"]["name"] for item in schemas]
        rows: list[Mapping[str, Any]] = []
        for part in parsed["dataset_parts"]:
            for path in sorted((Path("data") / part).glob("*.json")):
                raw = path.read_bytes()
                conversation = json.loads(raw)
                opening = conversation["conversation"][0]
                if opening.get("role") != "user":
                    raise OfficialArenaError("ToolTalk conversation has no opening user turn")
                checkpoint_id = f"{part}:{conversation['conversation_id']}"
                instruction = _system_prompt(conversation["metadata"])
                visible_state = {
                    "query": opening["text"],
                    "task_id": checkpoint_id,
                    "instruction": instruction,
                    "context": [
                        {
                            "text": json.dumps(schemas, ensure_ascii=False, sort_keys=True),
                            "block_id": "official_tool_catalog",
                            "source": "tooltalk.apis.ALL_APIS",
                            "score": None,
                            "metadata": {"api_mode": "all"},
                        }
                    ],
                    "messages": [
                        {"role": "user", "content": opening["text"], "name": None}
                    ],
                    "tool_names": tool_names,
                    "metadata": {
                        "benchmark_id": self.benchmark_id,
                        "difficulty": part,
                        "conversation_id": conversation["conversation_id"],
                        "checkpoint_phase": "turn_zero",
                        "clarification_roundtrip_available": True,
                        "state_restore_available": True,
                    },
                    "budget": visible_shared_budget(
                        parsed,
                        {
                            "max_model_calls": 16,
                            "max_retrieval_calls": 8,
                            "max_tool_calls": 9,
                            "max_output_tokens": 8192,
                            "max_context_tokens": None,
                            "max_cost_usd": None,
                        },
                    ),
                }
                rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "group_key": checkpoint_id,
                        "visible_state": visible_state,
                        "eligible_families": [
                            "reformulate",
                            "decompose",
                            "acquire",
                            "clarify",
                            "curate",
                        ],
                        "protected": {
                            "dataset_part": part,
                            "relative_path": path.as_posix(),
                            "conversation_sha256": hashlib.sha256(raw).hexdigest(),
                            "conversation_id": conversation["conversation_id"],
                            "locked_dependency_hashes": parsed["locked_dependency_hashes"],
                        },
                    }
                )
                if len(rows) >= parsed["limit"]:
                    return tuple(rows)
        if not rows:
            raise OfficialArenaError("ToolTalk capture found no conversations")
        return tuple(rows)

    def run_arm(
        self,
        config: Mapping[str, Any],
        *,
        replay_record: Mapping[str, Any],
        arm_payload: Mapping[str, Any],
        seed: int,
        role: str,
        work_dir: Path,
    ) -> ProviderArmResult:
        parsed = self._validate_config(config)
        prediction_path = work_dir / "predicted-conversation.json"
        evaluation_path = work_dir / "native-evaluation.json"
        trajectory_path = work_dir / "trajectory.json"
        history_path = work_dir / "live-history.json"
        artifacts = (prediction_path, evaluation_path, trajectory_path)
        if self.history_transform is not None:
            artifacts += (history_path,)
        if any(path.exists() for path in artifacts):
            raise OfficialArenaError("ToolTalk requires a fresh work directory")
        state = materialized_state(arm_payload)
        budget = require_materialized_budget(state, parsed)
        optimizer_usage = intervention_usage(arm_payload)
        protected = replay_record["protected"]
        path = Path(protected["relative_path"])
        raw = path.read_bytes()
        if _sha256(path) != protected["conversation_sha256"]:
            raise OfficialArenaError("ToolTalk protected conversation differs")
        conversation = json.loads(raw)
        if conversation.get("conversation_id") != protected["conversation_id"]:
            raise OfficialArenaError("ToolTalk protected conversation identity differs")
        api_key = os.environ.get(parsed["agent_key_env"])
        if not api_key:
            raise OfficialArenaError(
                f"ToolTalk environment {parsed['agent_key_env']} is missing"
            )
        override = parsed.get("budget_override")
        per_call_cap = (
            int(override["max_output_tokens_per_call"])
            if isinstance(override, Mapping)
            else 512
        )
        # Fresh execution processes need the pinned import shim before the
        # predictor imports tooltalk.apis; capture()/doctor() need not precede
        # every restored native episode in the same Python process.
        _install_sent2vec_compatibility()
        predictor = _Predictor(
            model=parsed["agent_model"],
            base_url=parsed["agent_base_url"],
            api_key=api_key,
            materialized=state,
            budget=budget,
            optimizer_usage=optimizer_usage,
            seed=seed,
            max_output_tokens_per_call=per_call_cap,
            history_transform=self.history_transform,
            timeout_seconds=parsed.get("agent_timeout_seconds", 600.0),
            extra_body=parsed.get("agent_extra_body"),
            extra_body_mode=parsed.get("agent_extra_body_mode", "merge"),
        )
        from tooltalk.evaluation.tool_executor import ToolExecutor

        executor = ToolExecutor(init_database_dir="data/databases")
        try:
            evaluated = executor.run_conversation(copy.deepcopy(conversation), predictor)
        finally:
            if self.history_transform is not None:
                with history_path.open("x", encoding="utf-8") as handle:
                    json.dump({
                        "fingerprint": self.history_transform.fingerprint(),
                        "records": list(getattr(self.history_transform, "records", ())),
                        "history_usage": predictor.history_usage,
                        "agent_and_history_usage": predictor.usage(),
                        "original_shared_budget": dict(budget),
                        "native_boundary_protocol": "tooltalk.committed_user_turns.v1",
                    }, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                    handle.write("\n")
        # Preserve the actual episode before scoring, including scorer failures.
        # Exclusive creation keeps an earlier attempt's evidence immutable.
        with prediction_path.open("x", encoding="utf-8") as handle:
            json.dump(evaluated, handle, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
        evaluated = executor.evaluate_predictions(evaluated)
        with evaluation_path.open("x", encoding="utf-8") as handle:
            json.dump(evaluated, handle, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
        metrics = evaluated["metrics"]
        trace = {
            "provider": "tooltalk",
            "source_revision": self.source_revision,
            "conversation_id": protected["conversation_id"],
            "dataset_part": protected["dataset_part"],
            "role": role,
            "seed": seed,
            "metrics": metrics,
            "generation_controls": {
                "hard_stops": predictor.hard_stops,
                "shared_budget": dict(budget),
                "optimizer_usage": dict(optimizer_usage),
            },
            "prediction_trace_sha256": content_hash(evaluated["conversation"]),
            "native_artifacts": {
                "prediction": {"path": prediction_path.name, "sha256": _sha256(prediction_path)},
                "evaluation": {"path": evaluation_path.name, "sha256": _sha256(evaluation_path)},
            },
        }
        if self.history_transform is not None:
            trace["live_history_transform"] = self.history_transform.fingerprint()
            trace["live_history_artifact"] = {"path": history_path.name, "sha256": _sha256(history_path)}
            trace["live_history_boundary_protocol"] = "tooltalk.committed_user_turns.v1"
        with trajectory_path.open("x", encoding="utf-8") as handle:
            json.dump(trace, handle, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
        return ProviderArmResult(
            metrics={
                "official_success": float(metrics["success"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "action_precision": float(metrics["action_precision"]),
                "bad_action_rate": float(metrics["bad_action_rate"]),
                "soft_success": float(metrics["soft_success"]),
                "adapter_hard_stops": float(predictor.hard_stops),
            },
            trace=trace,
            usage=predictor.usage(),
            runtime_instance_id=content_hash(
                {
                    "provider": "tooltalk",
                    "checkpoint_id": replay_record["checkpoint_id"],
                    "role": role,
                    "seed": seed,
                    "work_dir": str(work_dir),
                }
            ),
        )

    def _validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        config = resolve_shared_budget_config(config)
        if not isinstance(config, Mapping) or not (
            self._REQUIRED <= set(config) <= self._REQUIRED | self._OPTIONAL
        ):
            raise OfficialArenaError("ToolTalk provider config fields are invalid")
        if list(config["dataset_parts"]) != ["easy", "tooltalk"]:
            raise OfficialArenaError("ToolTalk full set must use easy then tooltalk")
        limit = config["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 78:
            raise OfficialArenaError("ToolTalk limit must be between 1 and 78")
        if config["api_mode"] != "all":
            raise OfficialArenaError("ToolTalk paper protocol requires api_mode=all")
        for name in ("agent_model", "agent_base_url", "agent_key_env"):
            if not isinstance(config[name], str) or not config[name].strip():
                raise OfficialArenaError(f"ToolTalk {name} is invalid")
        validate_dependency_hashes(config["locked_dependency_hashes"], label="ToolTalk")
        native_timeout_options(config, "agent_timeout_seconds")
        native_extra_body(config, "agent_extra_body")
        return config


def factory() -> ToolTalkProvider:
    return ToolTalkProvider()
