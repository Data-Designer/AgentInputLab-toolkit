"""Production provider for tau3 Banking Knowledge at the pinned revision."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..protocol import (
    OfficialArenaError,
    ProviderArmResult,
    content_hash,
    intervention_usage,
    materialized_state,
    require_materialized_budget,
    resolve_shared_budget_config,
    validate_dependency_hashes,
    visible_shared_budget,
)
from ._live_history import NativeHistorySession, validate_history_transform

_SHARED_BUDGET_STOP_MESSAGE = "I could not complete the task within the shared run budget."


def _is_replayable_initial_user_message(message: Any) -> bool:
    """Return whether a captured user turn is safe to replay as a checkpoint.

    tau3 user messages may contain a user-side tool call together with visible
    text. Stopping capture on that message leaves a dangling call without the
    following ToolMessage, and environment replay then fails before the agent
    can run. Let the orchestrator finish that tool exchange and checkpoint the
    next ordinary visible user turn instead.
    """

    content = getattr(message, "content", None)
    if not content:
        return False
    is_tool_call = getattr(message, "is_tool_call", None)
    return not (callable(is_tool_call) and is_tool_call())


def _remaining_agent_budget(
    budget: Mapping[str, Any],
    optimizer_usage: Mapping[str, int | float],
) -> dict[str, Any]:
    """Subtract optimizer consumption from every cumulative agent-side cap."""

    remaining = dict(budget)
    fields = {
        "max_model_calls": "model_calls",
        "max_retrieval_calls": "retrieval_calls",
        "max_tool_calls": "tool_calls",
        "max_output_tokens": "output_tokens",
        "max_context_tokens": "input_tokens",
    }
    for budget_name, usage_name in fields.items():
        cap = budget.get(budget_name)
        if cap is None:
            remaining[budget_name] = None
            continue
        remaining[budget_name] = max(
            0,
            int(cap) - int(optimizer_usage.get(usage_name, 0)),
        )
    return remaining


def _apply_tool_call_budget(
    response: Any,
    *,
    remaining_tool_calls: int,
) -> tuple[Any, int, int]:
    """Truncate generated tool calls before tau3 can execute beyond the cap.

    The returned counters are the number of calls retained for execution and
    the number discarded by the hard budget boundary.  Generated text-only
    answers remain legal when the tool budget is exhausted.
    """

    if isinstance(remaining_tool_calls, bool) or remaining_tool_calls < 0:
        raise OfficialArenaError("tau3 remaining tool-call budget is invalid")
    calls = getattr(response, "tool_calls", None)
    if calls is None:
        return response, 0, 0
    if not isinstance(calls, (list, tuple)):
        raise OfficialArenaError("tau3 generated tool_calls are invalid")
    generated = len(calls)
    kept = min(generated, remaining_tool_calls)
    discarded = generated - kept
    if discarded == 0:
        return response, kept, 0
    copier = getattr(response, "model_copy", None)
    if not callable(copier):
        raise OfficialArenaError("tau3 assistant response cannot be budget-truncated")
    if kept:
        response = copier(update={"tool_calls": list(calls[:kept])}, deep=True)
    else:
        response = copier(
            update={
                "tool_calls": None,
                "content": _SHARED_BUDGET_STOP_MESSAGE,
            },
            deep=True,
        )
    return response, kept, discarded


def _budget_exhausted_result(
    *,
    replay_record: Mapping[str, Any],
    role: str,
    seed: int,
    work_dir: Path,
    optimizer_usage: Mapping[str, int | float],
    remaining_budget: Mapping[str, int | float],
) -> ProviderArmResult:
    """Materialize a scored, auditable zero-action arm after optimizer exhaustion.

    A shared resource cap is an experimental outcome, not an infrastructure
    failure.  In particular, the canonical eight-document HyDE intervention
    consumes all eight x1 model calls before the downstream tau3 agent can
    act.  Preserve the paired baseline, report a hard stop, and let the common
    observation normalizer mark the treatment contract invalid with valid
    success zero.
    """

    trace = {
        "provider": "tau3",
        "source_revision": Tau3Provider.source_revision,
        "task_id": replay_record["protected"]["task_id"],
        "role": role,
        "seed": seed,
        "termination_reason": "shared_budget_exhausted_before_agent",
        "messages": [],
        "reward_info": None,
        "generation_controls": {
            "hard_stops": 1,
            "optimizer_usage": dict(optimizer_usage),
            "remaining_budget": dict(remaining_budget),
        },
        "score_provenance": "paper_shared_budget_zero_action",
    }
    work_dir.joinpath("budget-stop.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ProviderArmResult(
        metrics={
            "official_reward": 0.0,
            "cost": 0.0,
            "adapter_hard_stops": 1.0,
        },
        trace=trace,
        usage={
            "model_calls": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "unpriced_model_calls": 0,
        },
        runtime_instance_id=content_hash(
            {
                "provider": "tau3",
                "checkpoint_id": replay_record["checkpoint_id"],
                "role": role,
                "seed": seed,
                "termination_reason": trace["termination_reason"],
                "work_dir": str(work_dir),
            }
        ),
    )


def _model_output_invalid_result(
    *,
    replay_record: Mapping[str, Any],
    role: str,
    seed: int,
    work_dir: Path,
    error: Exception,
    messages: list[object],
    termination_reason: str = "generated_tool_arguments_invalid",
    failed_generation_scope: str = "evaluated_agent",
    measured_usage: Mapping[str, int | float] | None = None,
) -> ProviderArmResult:
    """Score a recognized malformed generation as an arm outcome, not infra loss.

    ``tau2`` raises before producing ``reward_info`` when either the evaluated
    agent emits malformed tool arguments or the benchmark user simulator emits
    an empty message. Both are real trajectory terminal states. The latter
    must retain the evaluated agent's already-consumed usage while excluding
    the benchmark user's failed generation from the paper-wide agent budget.
    """

    serialized_messages = []
    for message in messages:
        dump = getattr(message, "model_dump", None)
        if callable(dump):
            serialized_messages.append(dump(mode="json"))
    trace = {
        "provider": "tau3",
        "source_revision": Tau3Provider.source_revision,
        "task_id": replay_record["protected"]["task_id"],
        "role": role,
        "seed": seed,
        "termination_reason": termination_reason,
        "messages": serialized_messages,
        "reward_info": None,
        "generation_controls": {
            "hard_stops": 1,
            "failure_type": type(error).__name__,
            "failed_generation_scope": failed_generation_scope,
            "failed_generation_usage_unknown": True,
        },
        "score_provenance": "paper_model_output_invalid_zero_action",
    }
    work_dir.joinpath("model-output-invalid.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    usage = (
        {
            "model_calls": 1,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "unpriced_model_calls": 1,
        }
        if measured_usage is None
        else dict(measured_usage)
    )
    return ProviderArmResult(
        metrics={
            "official_reward": 0.0,
            "cost": 0.0,
            "adapter_hard_stops": 1.0,
        },
        trace=trace,
        usage=usage,
        runtime_instance_id=content_hash(
            {
                "provider": "tau3",
                "checkpoint_id": replay_record["checkpoint_id"],
                "role": role,
                "seed": seed,
                "termination_reason": trace["termination_reason"],
                "work_dir": str(work_dir),
            }
        ),
    )


def _is_generated_tool_argument_validation(error: Exception) -> bool:
    errors = getattr(error, "errors", None)
    if not callable(errors):
        return False
    return any(
        item.get("type") == "dict_type" and tuple(item.get("loc", ()))[-1:] == ("arguments",)
        for item in errors()
    )


def _is_generated_user_message_validation(error: Exception) -> bool:
    """Recognize only tau2's explicit empty generated-user-message failure."""

    return isinstance(error, ValueError) and str(error).startswith(
        "UserMessage must have either content or tool_calls. Got UserMessage"
    )


def _downstream_agent_usage(
    messages: list[object],
    *,
    initial_history_length: int,
    assistant_message_type: type,
    measured_model_calls: int | None = None,
    measured_input_tokens: int | None = None,
    measured_output_tokens: int | None = None,
) -> dict[str, int]:
    """Measure only calls made by the evaluated agent after the replay checkpoint.

    The tau3 user simulator is part of the benchmark environment.  Its generated
    ``UserMessage`` objects may also carry provider usage, but charging those
    tokens to the evaluated arm can falsely exceed the paper-wide agent budget.
    """

    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    tool_calls = 0
    for message in messages[initial_history_length:]:
        if not isinstance(message, assistant_message_type):
            continue
        model_calls += 1
        usage = getattr(message, "usage", None)
        if isinstance(usage, Mapping):
            input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        calls = getattr(message, "tool_calls", None)
        if calls:
            tool_calls += len(calls)
    measured = {
        "model_calls": model_calls if measured_model_calls is None else measured_model_calls,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens if measured_input_tokens is None else measured_input_tokens,
        "output_tokens": (
            output_tokens if measured_output_tokens is None else measured_output_tokens
        ),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in measured.values()
    ):
        raise OfficialArenaError("tau3 evaluated-agent usage counters are invalid")
    return measured


def _orchestrator_step_cap(
    *,
    remaining_budget: Mapping[str, int | float],
    materialized: Mapping[str, Any],
) -> int:
    """Translate the evaluated-agent call budget into half-duplex steps.

    tau3 counts each USER/AGENT/ENV transition as an orchestrator step, whereas
    the paper budget counts only generations made by the evaluated agent.  A
    normal agent turn therefore needs two orchestrator steps.  A materialized
    clarification contributes one deterministic, zero-generation agent turn
    before the remaining model calls.
    """

    model_calls = int(remaining_budget["max_model_calls"])
    metadata = materialized.get("metadata", {})
    clarification = metadata.get("clarification_request") if isinstance(metadata, Mapping) else None
    deterministic_agent_turns = int(isinstance(clarification, str) and bool(clarification.strip()))
    return 2 * (model_calls + deterministic_agent_turns)


def _local_tau_model_ids(config: Mapping[str, Any]) -> frozenset[str]:
    """Return model ids whose tau3 endpoint is explicitly loopback-local."""

    local: set[str] = set()
    for role in ("agent", "user"):
        args = config.get(f"llm_args_{role}")
        model = config.get(f"llm_{role}")
        if not isinstance(args, Mapping) or not isinstance(model, str):
            continue
        raw_url = args.get("api_base") or args.get("base_url")
        if not isinstance(raw_url, str):
            continue
        hostname = urlparse(raw_url).hostname
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            continue
        local.add(model.rsplit("/", 1)[-1])
    return frozenset(local)


def _install_local_tau_cost_accounting(config: Mapping[str, Any]) -> frozenset[str]:
    """Keep local unpriced models out of LiteLLM's remote price lookup path."""

    local_models = _local_tau_model_ids(config)
    if not local_models:
        return local_models
    from tau2.utils import llm_utils

    current = llm_utils.get_response_cost
    registered = getattr(current, "_agentinputlab_local_models", None)
    if isinstance(registered, set):
        registered.update(local_models)
        return local_models
    original = current
    registered = set(local_models)

    def local_aware_cost(response: Any) -> float:
        model = str(getattr(response, "model", "")).rsplit("/", 1)[-1]
        if model in registered:
            return 0.0
        return float(original(response))

    local_aware_cost._agentinputlab_local_models = registered  # type: ignore[attr-defined]
    llm_utils.get_response_cost = local_aware_cost
    return local_models


def _local_tau_nl_evaluator_binding(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Bind τ³'s NL-assertion judge to the declared local model endpoint.

    The pinned τ³ scorer evaluates the final Banking Knowledge task with an
    LLM-backed natural-language assertion.  Upstream defaults that judge to a
    closed GPT endpoint.  A fully local paper profile must not silently leave
    the declared runtime and contact that endpoint.  For explicit loopback
    profiles, use the same frozen model/load set and deterministic decoding
    controls as the two simulated participants.  External/API profiles retain
    the upstream evaluator binding.
    """

    local_models = _local_tau_model_ids(config)
    model = config.get("llm_agent")
    raw_args = config.get("llm_args_agent")
    if not isinstance(model, str) or not isinstance(raw_args, Mapping):
        return None
    if model.rsplit("/", 1)[-1] not in local_models:
        return None
    args = dict(raw_args)
    args["temperature"] = 0.0
    args["response_format"] = {"type": "json_object"}
    return {"model": model, "arguments": args, "provenance": "declared_local_load_set"}


def _install_local_tau_nl_evaluator(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Install the explicit local NL-assertion evaluator in the pinned scorer."""

    binding = _local_tau_nl_evaluator_binding(config)
    if binding is None:
        return None
    from tau2.evaluator import evaluator_nl_assertions

    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = binding["model"]
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(binding["arguments"])
    return binding


class Tau3Provider:
    benchmark_id = "tau3.banking_knowledge"
    source_revision = "cf71a8070269883e38a365ffa85f78f46844c1f4"
    scorer_id = "tau3.official_v1.0.1"
    scorer_revision = "32526f9c17ba3725cc8327d43783a2b5d2c48015dc72dbc6f7af4d25a95328a6"

    _CONFIG_FIELDS = frozenset(
        {
            "task_set_name",
            "task_split_name",
            "limit",
            "llm_agent",
            "llm_user",
            "llm_args_agent",
            "llm_args_user",
            "retrieval_config",
            "retrieval_config_kwargs",
            "max_steps",
            "max_errors",
            "timeout",
            "locked_dependency_hashes",
        }
    )
    _OPTIONAL_CONFIG_FIELDS = frozenset(
        {
            "retriever_override",
            "budget_override",
            "capture_max_attempts",
            "capture_text_only_fallback",
            "capture_task_ids",
        }
    )

    def __init__(self, *, history_transform: Any = None) -> None:
        validate_history_transform(history_transform)
        self.history_transform = history_transform

    @staticmethod
    def _history_result(result, agent, work_dir):
        session = getattr(agent, "_history_session", None)
        if session is None:
            return result
        receipt = work_dir / "live-history.json"
        session.write_receipt(receipt, target_usage=agent._history_target_usage())
        metrics = dict(result.metrics)
        if "cost" in metrics:
            metrics["cost"] += float(session.usage.get("cost_usd", 0))
        return ProviderArmResult(
            metrics=metrics,
            trace={**result.trace, "live_history_transform": session.hook.fingerprint(),
                   "live_history_receipt": {"path": receipt.name, "sha256": _file_sha256(receipt)},
                   "history_usage": dict(session.usage)},
            usage=session.combined_usage(result.usage),
            runtime_instance_id=content_hash({"native_runtime": result.runtime_instance_id,
                "live_history_receipt_sha256": _file_sha256(receipt)}),
        )

    def doctor(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            self._validate_config(config)
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if head != self.source_revision:
                raise OfficialArenaError("tau3 checkout revision differs")
            from tau2.agent.llm_agent import LLMAgent
            from tau2.runner.build import build_orchestrator
            from tau2.runner.helpers import get_tasks
            from tau2.runner.simulation import run_simulation

            del LLMAgent, build_orchestrator, get_tasks, run_simulation
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
        _install_local_tau_cost_accounting(parsed)
        _install_local_tau_nl_evaluator(parsed)
        _install_agentinputlab_retriever(parsed)
        _install_tau_tool_argument_normalizer()
        from tau2.data_model.message import AssistantMessage, UserMessage
        from tau2.runner.build import build_orchestrator
        from tau2.runner.helpers import get_tasks

        run_config = self._run_config(parsed)
        tasks = get_tasks(
            parsed["task_set_name"],
            task_split_name=parsed["task_split_name"],
            num_tasks=parsed["limit"],
            task_ids=parsed.get("capture_task_ids"),
        )
        rows: list[Mapping[str, Any]] = []
        capture_attempts, text_only_fallback = _capture_attempt_policy(
            parsed.get("capture_max_attempts", 1),
            parsed.get("capture_text_only_fallback", False),
        )
        capture_policy = {
            "policy_id": "tau3.single_capture.v1" if capture_attempts == 1 else "tau3.explicit_bounded_capture.v1",
            "max_attempts": capture_attempts,
            "text_only_fallback_allowed": text_only_fallback,
            "usage_scope": "environment_preparation_not_evaluated_agent_budget",
            "attempt_count_is_not_model_call_or_token_usage": True,
        }
        for index, task in enumerate(tasks):
            orchestrator, message, capture_attempt, capture_seed = _capture_initial_visible_user(
                build_orchestrator=build_orchestrator,
                run_config=run_config,
                task=task,
                base_seed=index,
                assistant_message_type=AssistantMessage,
                user_message_type=UserMessage,
                max_attempts=capture_attempts,
                allow_text_only_fallback=text_only_fallback,
            )
            try:
                initial_history = []
                for item in orchestrator.get_messages():
                    normalized = item.model_copy(update={"timestamp": None, "turn_idx": None})
                    initial_history.append(normalized.model_dump(mode="json"))
                tools = orchestrator.environment.get_tools()
                tool_schemas = [tool.openai_schema for tool in tools]
                tool_names = tuple(
                    getattr(tool, "name", None) or getattr(tool, "__name__", None) for tool in tools
                )
                if any(not isinstance(name, str) or not name for name in tool_names):
                    raise OfficialArenaError("tau3 tool inventory contains unnamed tools")
                policy = orchestrator.environment.get_policy()
                if (
                    not isinstance(tool_schemas, list)
                    or not tool_schemas
                    or any(not isinstance(item, Mapping) for item in tool_schemas)
                ):
                    raise OfficialArenaError("tau3 tool schemas are unavailable")
                visible_state = {
                    "query": message.content,
                    "task_id": task.id,
                    "instruction": policy,
                    "context": [
                        {
                            "text": policy,
                            "block_id": "official_policy",
                            "source": "tau3.environment.policy",
                            "score": None,
                            "metadata": {"domain": "banking_knowledge"},
                        },
                        {
                            "text": json.dumps(
                                tool_schemas,
                                default=str,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "block_id": "official_tool_catalog",
                            "source": "tau3.environment.tools",
                            "score": None,
                            "metadata": {
                                "domain": "banking_knowledge",
                                "tool_defs_hash": content_hash(tool_schemas),
                            },
                        },
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": message.content,
                            "name": None,
                        }
                    ],
                    "tool_names": list(tool_names),
                    "metadata": {
                        "benchmark_id": self.benchmark_id,
                        "checkpoint_phase": "after_initial_user_message",
                        "clarification_roundtrip_available": True,
                        "state_restore_available": True,
                        "retrieval_config": parsed["retrieval_config"],
                        "capture_attempt": capture_attempt,
                        "capture_seed": capture_seed,
                        "capture_policy": capture_policy,
                    },
                    "budget": visible_shared_budget(
                        parsed,
                        {
                            "max_model_calls": parsed["max_steps"],
                            "max_retrieval_calls": parsed["max_steps"],
                            "max_tool_calls": parsed["max_steps"],
                            "max_output_tokens": 65536,
                            "max_context_tokens": None,
                            "max_cost_usd": None,
                        },
                    ),
                }
                rows.append(
                    {
                        "checkpoint_id": str(task.id),
                        "group_key": str(task.id),
                        "visible_state": visible_state,
                        "eligible_families": [
                            "reformulate",
                            "decompose",
                            "acquire",
                            "clarify",
                            "curate",
                        ],
                        "protected": {
                            "task_id": str(task.id),
                            "initial_user_message": message.model_dump(mode="json"),
                            "initial_message_history": initial_history,
                            "capture_attempt": capture_attempt,
                            "capture_seed": capture_seed,
                            "capture_policy": capture_policy,
                            "locked_dependency_hashes": parsed["locked_dependency_hashes"],
                        },
                    }
                )
            finally:
                orchestrator._cleanup()
        if not rows:
            raise OfficialArenaError("tau3 capture found no tasks")
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
        if self.history_transform is not None and (work_dir / "live-history.json").exists():
            raise OfficialArenaError("native history output must be a fresh episode")
        parsed = self._validate_config(config)
        local_unpriced_models = _install_local_tau_cost_accounting(parsed)
        nl_evaluator_binding = _install_local_tau_nl_evaluator(parsed)
        _install_agentinputlab_retriever(parsed)
        _install_tau_tool_argument_normalizer()
        state = materialized_state(arm_payload)
        budget = require_materialized_budget(state, parsed)
        optimizer_usage = intervention_usage(arm_payload)
        remaining_budget = _remaining_agent_budget(budget, optimizer_usage)
        if remaining_budget["max_model_calls"] < 1 or remaining_budget["max_output_tokens"] < 1:
            return _budget_exhausted_result(
                replay_record=replay_record,
                role=role,
                seed=seed,
                work_dir=work_dir,
                optimizer_usage=optimizer_usage,
                remaining_budget=remaining_budget,
            )
        protected = replay_record["protected"]

        from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
        from tau2.data_model.tasks import InitialState
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
        from tau2.runner.build import (
            _build_env_kwargs,
            build_environment,
            build_user,
        )
        from tau2.runner.helpers import get_tasks
        from tau2.runner.simulation import run_simulation

        task_id = protected["task_id"]
        tasks = get_tasks(
            parsed["task_set_name"],
            task_split_name=parsed["task_split_name"],
            task_ids=[task_id],
        )
        if len(tasks) != 1:
            raise OfficialArenaError("tau3 protected task identity is unavailable")
        task = tasks[0]
        raw_history = protected.get("initial_message_history")
        if isinstance(raw_history, list) and raw_history:
            message_types = {
                "assistant": AssistantMessage,
                "user": UserMessage,
                "tool": ToolMessage,
            }
            try:
                initial_history = [
                    message_types[str(item["role"])].model_validate(item) for item in raw_history
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise OfficialArenaError(
                    "tau3 protected initial message history is invalid"
                ) from exc
        else:
            initial_user = UserMessage.model_validate(protected["initial_user_message"])
            initial_history = [copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE), initial_user]
        original_initial = task.initial_state
        task = task.model_copy(
            update={
                "initial_state": InitialState(
                    initialization_data=(
                        original_initial.initialization_data
                        if original_initial is not None
                        else None
                    ),
                    initialization_actions=(
                        original_initial.initialization_actions
                        if original_initial is not None
                        else None
                    ),
                    message_history=copy.deepcopy(initial_history),
                )
            },
            deep=True,
        )
        run_config = self._run_config(parsed, budget=remaining_budget)
        env_kwargs = _build_env_kwargs(run_config, task)
        environment = build_environment(
            "banking_knowledge",
            env_kwargs=env_kwargs,
        )
        agent = _intervened_tau_agent_class()(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=parsed["llm_agent"],
            llm_args=parsed["llm_args_agent"],
            materialized=state,
            budget=remaining_budget,
            history_transform=self.history_transform,
        )
        user = build_user(
            "user_simulator",
            environment,
            task,
            llm=parsed["llm_user"],
            llm_args=parsed["llm_args_user"],
        )
        from tau2.orchestrator.orchestrator import Orchestrator

        orchestrator = Orchestrator(
            domain="banking_knowledge",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=_orchestrator_step_cap(
                remaining_budget=remaining_budget,
                materialized=state,
            ),
            max_errors=parsed["max_errors"],
            seed=seed,
            simulation_id=f"agentinputlab-{role}-{task_id}-{seed}",
            timeout=parsed["timeout"],
        )
        try:
            simulation = run_simulation(
                orchestrator,
                env_kwargs=env_kwargs,
            )
        except json.JSONDecodeError as error:
            result = _model_output_invalid_result(
                replay_record=replay_record,
                role=role,
                seed=seed,
                work_dir=work_dir,
                error=error,
                messages=orchestrator.get_messages(),
            )
            return self._history_result(result, agent, work_dir)
        except ValueError as error:
            if _is_generated_user_message_validation(error):
                measured_usage = _downstream_agent_usage(
                    orchestrator.get_messages(),
                    initial_history_length=len(initial_history),
                    assistant_message_type=AssistantMessage,
                    measured_model_calls=agent._budget_model_calls,
                    measured_input_tokens=agent._budget_input_tokens,
                    measured_output_tokens=agent._budget_output_tokens,
                )
                result = _model_output_invalid_result(
                    replay_record=replay_record,
                    role=role,
                    seed=seed,
                    work_dir=work_dir,
                    error=error,
                    messages=orchestrator.get_messages(),
                    termination_reason="generated_user_message_invalid",
                    failed_generation_scope="benchmark_user_simulator",
                    measured_usage={
                        **measured_usage,
                        "cost_usd": 0.0,
                        "unpriced_model_calls": measured_usage["model_calls"],
                    },
                )
                return self._history_result(result, agent, work_dir)
            if not _is_generated_tool_argument_validation(error):
                raise
            result = _model_output_invalid_result(
                replay_record=replay_record,
                role=role,
                seed=seed,
                work_dir=work_dir,
                error=error,
                messages=orchestrator.get_messages(),
            )
            return self._history_result(result, agent, work_dir)
        finally:
            if getattr(agent, "_history_session", None) is not None:
                agent._history_session.write_receipt(work_dir / "live-history.json",
                                                     target_usage=agent._history_target_usage())
        if simulation.reward_info is None:
            raise OfficialArenaError("tau3 official evaluator did not return reward_info")

        messages = simulation.get_messages()
        # The replay history contains the fixed initial assistant greeting. It
        # is not a model call made by this arm and must not consume the shared
        # execution budget a second time. The user simulator is benchmark
        # environment state, so its calls and tokens are deliberately excluded.
        agent_usage = _downstream_agent_usage(
            messages,
            initial_history_length=len(initial_history),
            assistant_message_type=AssistantMessage,
            # The intervened agent owns the authoritative counters for real
            # generations.  A materialized clarification and the deterministic
            # budget-stop response are AssistantMessages too, but neither is a
            # model call and neither may consume the shared generation budget.
            measured_model_calls=agent._budget_model_calls,
            measured_input_tokens=agent._budget_input_tokens,
            measured_output_tokens=agent._budget_output_tokens,
        )
        if agent_usage["tool_calls"] != agent._budget_tool_calls:
            raise OfficialArenaError(
                "tau3 executed tool-call trace differs from its enforced budget counter"
            )
        trace = {
            "provider": "tau3",
            "source_revision": self.source_revision,
            "task_id": task_id,
            "role": role,
            "seed": seed,
            "cost_accounting": (
                "local_unpriced_zero"
                if str(parsed["llm_agent"]).rsplit("/", 1)[-1] in local_unpriced_models
                else "upstream_litellm"
            ),
            "nl_assertion_evaluator": (
                {
                    "model": nl_evaluator_binding["model"],
                    "provenance": nl_evaluator_binding["provenance"],
                }
                if nl_evaluator_binding is not None
                else {"model": "upstream_default", "provenance": "pinned_tau3"}
            ),
            "termination_reason": simulation.termination_reason.value,
            "messages": [message.model_dump(mode="json") for message in messages],
            "reward_info": simulation.reward_info.model_dump(mode="json"),
            "generation_controls": {
                "hard_stops": agent._budget_hard_stops,
                "executed_tool_calls": agent._budget_tool_calls,
                "discarded_tool_calls": agent._budget_discarded_tool_calls,
            },
        }
        work_dir.joinpath("simulation.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        total_cost = float(simulation.agent_cost or 0.0) + float(simulation.user_cost or 0.0)
        result = ProviderArmResult(
            metrics={
                "official_reward": float(simulation.reward_info.reward),
                "cost": total_cost,
                "adapter_hard_stops": float(agent._budget_hard_stops),
            },
            trace=trace,
            usage={
                **agent_usage,
                "cost_usd": total_cost,
                "unpriced_model_calls": (
                    agent_usage["model_calls"]
                    if (
                        str(parsed["llm_agent"]).rsplit("/", 1)[-1] in local_unpriced_models
                        or (agent_usage["input_tokens"] == 0 and agent_usage["output_tokens"] == 0)
                    )
                    else 0
                ),
            },
            runtime_instance_id=content_hash(
                {
                    "provider": "tau3",
                    "checkpoint_id": replay_record["checkpoint_id"],
                    "role": role,
                    "seed": seed,
                    "simulation_id": simulation.id,
                    "work_dir": str(work_dir),
                }
            ),
        )
        return self._history_result(result, agent, work_dir)

    def _run_config(
        self,
        config: Mapping[str, Any],
        *,
        budget: Mapping[str, Any] | None = None,
    ) -> Any:
        from tau2.data_model.simulation import TextRunConfig

        agent_args = dict(config["llm_args_agent"])
        max_steps = int(config["max_steps"])
        if budget is not None:
            max_steps = min(max_steps, int(budget["max_model_calls"]))
            override = config.get("budget_override")
            declared_per_call = (
                int(override["max_output_tokens_per_call"])
                if isinstance(override, Mapping)
                else max(1, int(budget["max_output_tokens"]) // max_steps)
            )
            per_call = int(
                min(
                    declared_per_call,
                    max(1, int(budget["max_output_tokens"]) // max_steps),
                )
            )
            agent_args["max_tokens"] = min(
                int(agent_args.get("max_tokens", per_call)),
                per_call,
                int(budget["max_output_tokens"]),
            )

        return TextRunConfig(
            domain="banking_knowledge",
            agent="llm_agent",
            llm_agent=config["llm_agent"],
            llm_args_agent=agent_args,
            user="user_simulator",
            llm_user=config["llm_user"],
            llm_args_user=dict(config["llm_args_user"]),
            max_steps=max_steps,
            max_errors=config["max_errors"],
            timeout=config["timeout"],
            retrieval_config=config["retrieval_config"],
            retrieval_config_kwargs=dict(config["retrieval_config_kwargs"]),
        )

    def _validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        config = resolve_shared_budget_config(resolve_retriever_config(config))
        if not isinstance(config, Mapping) or not (
            set(config) >= self._CONFIG_FIELDS
            and set(config) <= self._CONFIG_FIELDS | self._OPTIONAL_CONFIG_FIELDS
        ):
            raise OfficialArenaError("tau3 provider config fields are invalid")
        for name in (
            "task_set_name",
            "task_split_name",
            "llm_agent",
            "llm_user",
            "retrieval_config",
        ):
            if not isinstance(config[name], str) or not config[name].strip():
                raise OfficialArenaError(f"tau3 provider {name} is invalid")
        for name in ("llm_args_agent", "llm_args_user", "retrieval_config_kwargs"):
            if not isinstance(config[name], Mapping):
                raise OfficialArenaError(f"tau3 provider {name} must be an object")
        for name in ("limit", "max_steps", "max_errors"):
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise OfficialArenaError(f"tau3 provider {name} must be positive")
        timeout = config["timeout"]
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise OfficialArenaError("tau3 provider timeout is invalid")
        _capture_attempt_policy(
            config.get("capture_max_attempts", 1),
            config.get("capture_text_only_fallback", False),
        )
        if config.get("capture_text_only_fallback", False) and not _is_gpt_oss_model_name(config["llm_user"]):
            raise OfficialArenaError("tau3 text-only capture fallback requires GPT-OSS")
        if "capture_task_ids" in config:
            selected = config["capture_task_ids"]
            if (
                not isinstance(selected, list)
                or not selected
                or len(selected) > config["limit"]
                or any(not isinstance(value, str) or not value.strip() for value in selected)
                or len(set(selected)) != len(selected)
            ):
                raise OfficialArenaError("tau3 capture_task_ids is invalid")
        validate_dependency_hashes(
            config["locked_dependency_hashes"],
            label="tau3",
        )
        if "retriever_override" in config:
            _validate_retriever_override(config["retriever_override"])
        return config


_RETRIEVER_OVERRIDE_FIELDS = frozenset(
    {
        "profile_id",
        "backend",
        "model_path",
        "model_revision",
        "model_artifact_hash",
        "query_prefix",
        "document_prefix",
        "expected_dimensions",
        "top_k",
        "device",
        "prompt_template",
        "prompt_sha256",
    }
)
_LOCAL_ENCODERS: dict[str, Any] = {}
RETRIEVER_OVERRIDE_PATH_ENV = "AILAB_TAU3_RETRIEVER_OVERRIDE_PATH"
RETRIEVER_OVERRIDE_SHA256_ENV = "AILAB_TAU3_RETRIEVER_OVERRIDE_SHA256"
_EMPTY_ASSISTANT_RESPONSE = "[AgentInputLab: empty assistant response]"
_CAPTURE_MAX_ATTEMPTS = 16
_CAPTURE_MAX_STEPS = 32
_CAPTURE_RETRY_SEED_STRIDE = 1_000_003
_GPT_OSS_CAPTURE_FALLBACK_ATTEMPTS = 1


def _capture_attempt_policy(max_attempts: int, allow_text_only_fallback: bool) -> tuple[int, bool]:
    """Validate explicit preparation policies before building a simulator."""

    if type(allow_text_only_fallback) is not bool:
        raise OfficialArenaError("tau3 capture_text_only_fallback must be boolean")
    maximum = _CAPTURE_MAX_ATTEMPTS + (
        _GPT_OSS_CAPTURE_FALLBACK_ATTEMPTS if allow_text_only_fallback else 0
    )
    if type(max_attempts) is not int or not 1 <= max_attempts <= maximum:
        raise OfficialArenaError("tau3 capture_max_attempts is invalid")
    if allow_text_only_fallback and max_attempts != maximum:
        raise OfficialArenaError("tau3 capture_text_only_fallback requires explicit 17-attempt preparation")
    return max_attempts, allow_text_only_fallback


def _capture_retry_seed(base_seed: int, attempt: int) -> int:
    """Derive a distinct deterministic seed for a simulator-only capture retry."""

    if attempt < 0:
        raise ValueError("capture attempt must be non-negative")
    return base_seed + attempt * _CAPTURE_RETRY_SEED_STRIDE


def _capture_initial_visible_user(
    *,
    build_orchestrator: Any,
    run_config: Any,
    task: Any,
    base_seed: int,
    assistant_message_type: type,
    user_message_type: type,
    max_attempts: int = 1,
    allow_text_only_fallback: bool = False,
) -> tuple[Any, Any, int, int]:
    """Capture a visible initial user turn; default to one simulator attempt.

    The benchmark user simulator occasionally emits a long environment-tool
    preamble, or malformed tool-argument JSON, before the evaluated agent is
    allowed to run. Those are capture-infrastructure failures, not treatment
    outcomes. Additional preparation attempts require an explicit bound and
    retain deterministic attempt/seed metadata. Removing the simulator's tools
    is a separate explicit opt-in variant, never an automatic repair. Count
    all native user/agent/judge generations separately: an attempt counter is
    not token/cost telemetry, and missing usage is not free preparation.
    """

    attempts, allow_text_only_fallback = _capture_attempt_policy(max_attempts, allow_text_only_fallback)
    last_failure: Exception | None = None
    for attempt in range(attempts):
        capture_seed = _capture_retry_seed(base_seed, attempt)
        orchestrator = build_orchestrator(
            run_config,
            task,
            seed=capture_seed,
            simulation_id=f"agentinputlab-capture-{task.id}-attempt-{attempt + 1}",
        )
        capture_user = getattr(orchestrator, "user", None)
        capture_model = str(getattr(capture_user, "llm", ""))
        if allow_text_only_fallback and (
            not _is_gpt_oss_model_name(capture_model) or not hasattr(capture_user, "tools")
        ):
            orchestrator._cleanup()
            raise OfficialArenaError("tau3 text-only capture fallback requires a GPT-OSS simulator with tools")
        text_only_fallback = allow_text_only_fallback and attempt >= _CAPTURE_MAX_ATTEMPTS
        if text_only_fallback:
            # GPT-OSS can deterministically loop on user-only environment tools
            # before emitting the first visible request.  After the normal,
            # bounded capture window is exhausted, make one final simulator
            # call without those pre-agent tools.  The user scenario remains
            # unchanged, but this is an explicit preparation variant, not the
            # default capture policy or an evaluated-agent retry.
            capture_user.tools = []
        try:
            orchestrator.initialize()
            for _capture_step in range(_CAPTURE_MAX_STEPS):
                orchestrator.step()
                candidate = orchestrator.message
                if isinstance(candidate, user_message_type) and _is_replayable_initial_user_message(
                    candidate
                ):
                    return orchestrator, candidate, attempt + 1, capture_seed
                if isinstance(candidate, assistant_message_type):
                    raise OfficialArenaError(
                        "tau3 agent executed before the first visible user message"
                    )
            last_failure = OfficialArenaError(
                "tau3 capture did not produce an initial visible user message"
            )
        except json.JSONDecodeError as error:
            last_failure = error
        except Exception:
            orchestrator._cleanup()
            raise
        orchestrator._cleanup()
    raise OfficialArenaError(
        "tau3 capture did not produce an initial visible user message "
        f"after {attempts} simulator attempts"
    ) from last_failure


def _is_gpt_oss_model_name(model_name: str) -> bool:
    return model_name.casefold().removeprefix("openai/") == "gpt-oss-20b"


def _is_gpt_oss_generation_boundary_failure(error: Exception, *, model_name: str) -> bool:
    """Recognize model-output failures that should become scored hard stops."""

    if not _is_gpt_oss_model_name(model_name):
        return False
    text = str(error).casefold()
    return (
        "unexpected tokens remaining in message header" in text
        or "unexpected token " in text
        or ("input length" in text and "exceeds model's maximum context length" in text)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_retriever_override(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RETRIEVER_OVERRIDE_FIELDS:
        raise OfficialArenaError("tau3 retriever_override fields are invalid")
    if value["backend"] not in {"bm25", "sentence_transformers_dense"}:
        raise OfficialArenaError("tau3 retriever_override backend is invalid")
    for name in ("profile_id", "model_artifact_hash", "prompt_template", "prompt_sha256"):
        if not isinstance(value[name], str) or not value[name]:
            raise OfficialArenaError(f"tau3 retriever_override {name} is invalid")
    for name in ("model_artifact_hash", "prompt_sha256"):
        if len(value[name]) != 64 or any(
            character not in "0123456789abcdef" for character in value[name]
        ):
            raise OfficialArenaError(f"tau3 retriever_override {name} is not SHA-256")
    for name in ("top_k", "expected_dimensions"):
        if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 1:
            raise OfficialArenaError(f"tau3 retriever_override {name} must be positive")
    raw_prompt = Path(str(value["prompt_template"])).expanduser()
    prompt = raw_prompt.resolve()
    if (
        not prompt.is_file()
        or raw_prompt.is_symlink()
        or _file_sha256(prompt) != value["prompt_sha256"]
    ):
        raise OfficialArenaError("tau3 retriever_override prompt is absent or changed")
    if value["backend"] == "sentence_transformers_dense":
        model_path = Path(str(value["model_path"])).expanduser().resolve()
        if not model_path.is_dir() or not value["model_revision"] or not value["device"]:
            raise OfficialArenaError("tau3 dense retriever_override model binding is invalid")
    return value


def resolve_retriever_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve a hash-bound process-level τ³ retriever factor.

    Every main-arena method runner uses this provider.  The two-variable
    binding therefore lets RQ4 vary the shared downstream retrieval dependency
    without changing method logic or editing the pinned tau2 checkout.
    """

    raw_path = os.environ.get(RETRIEVER_OVERRIDE_PATH_ENV)
    raw_sha256 = os.environ.get(RETRIEVER_OVERRIDE_SHA256_ENV)
    if not raw_path and not raw_sha256:
        return config
    if not raw_path or not raw_sha256:
        raise OfficialArenaError("tau3 retriever override environment binding is incomplete")
    unresolved = Path(raw_path).expanduser()
    path = unresolved.resolve()
    if unresolved.is_symlink() or not path.is_file() or _file_sha256(path) != raw_sha256:
        raise OfficialArenaError("tau3 retriever override environment artifact differs")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialArenaError("tau3 retriever override artifact is invalid JSON") from exc
    override = _validate_retriever_override(loaded)
    existing = config.get("retriever_override")
    if existing is not None and existing != override:
        raise OfficialArenaError("tau3 explicit and environment retriever overrides differ")
    variant_name = (
        "agentinputlab_rq4_"
        + str(override["profile_id"]).replace(".", "_")
        + f"_k{override['top_k']}"
    )
    return {
        **config,
        "retrieval_config": variant_name,
        "retrieval_config_kwargs": {"top_k": int(override["top_k"])},
        "retriever_override": override,
    }


def _normalized_tool_arguments(value: object) -> str | None:
    """Return canonical object JSON for normal or double-encoded arguments."""

    if not isinstance(value, str):
        return None
    candidate = value
    for _ in range(4):
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, Mapping):
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if isinstance(parsed, str):
            candidate = parsed
            continue
        return None
    return None


def _normalized_tool_name(value: object, *, advertised_tool_names: set[str]) -> str | None:
    """Strip only a leaked Harmony channel suffix from an advertised tool name."""

    if not isinstance(value, str) or "<|channel|>" not in value:
        return None
    candidate = value.split("<|channel|>", 1)[0]
    if candidate and candidate in advertised_tool_names:
        return candidate
    return None


def _openai_tool_names(tools: object) -> set[str]:
    if isinstance(tools, (str, bytes)) or not isinstance(tools, (list, tuple)):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _normalize_empty_assistant_message(message: Any) -> bool:
    """Make an empty OpenAI assistant turn explicit for the pinned tau2 runtime.

    Some local OpenAI-compatible models emit a successful completion whose
    assistant message contains neither text nor tool calls.  The model event is
    valid at the transport layer, but tau2 rejects it before the official
    scorer can assign a task outcome.  Preserve any refusal text and otherwise
    insert a deterministic, visibly synthetic sentinel so the trajectory is
    scored as produced instead of disappearing from the experiment matrix.

    The function returns whether it changed the response.  It deliberately
    does not alter non-empty content, tool calls, finish reasons, or usage.
    """

    if message is None or getattr(message, "tool_calls", ()):
        return False
    content = getattr(message, "content", None)
    if content is not None and (not isinstance(content, str) or content.strip()):
        return False
    refusal = getattr(message, "refusal", None)
    message.content = (
        refusal if isinstance(refusal, str) and refusal.strip() else _EMPTY_ASSISTANT_RESPONSE
    )
    return True


def _install_tau_tool_argument_normalizer() -> None:
    """Bridge vLLM/OpenAI tool arguments to tau2's dictionary contract.

    OpenAI-compatible servers serialize function arguments as JSON text.  A
    few valid generations quote that JSON one additional time.  The pinned
    tau2 runtime decodes exactly once and then validates a dictionary, so the
    double-encoded form otherwise aborts an official trajectory.  Normalize
    only this transport representation; tool name, values, and scorer state
    remain unchanged.
    """

    import tau2.utils.llm_utils as llm_utils

    completion = llm_utils.completion
    if getattr(completion, "_agentinputlab_tool_argument_normalizer", False):
        return

    def normalized_completion(*args: Any, **kwargs: Any) -> Any:
        response = completion(*args, **kwargs)
        advertised_tool_names = _openai_tool_names(kwargs.get("tools"))
        choices = getattr(response, "choices", ()) or ()
        for choice in choices:
            message = getattr(choice, "message", None)
            _normalize_empty_assistant_message(message)
            for tool_call in getattr(message, "tool_calls", ()) or ():
                function = getattr(tool_call, "function", None)
                raw_name = getattr(function, "name", None)
                normalized_name = _normalized_tool_name(
                    raw_name,
                    advertised_tool_names=advertised_tool_names,
                )
                if normalized_name is not None:
                    function.name = normalized_name
                raw = getattr(function, "arguments", None)
                normalized = _normalized_tool_arguments(raw)
                if normalized is not None:
                    function.arguments = normalized
        return response

    normalized_completion._agentinputlab_tool_argument_normalizer = True  # type: ignore[attr-defined]
    llm_utils.completion = normalized_completion


def _install_agentinputlab_retriever(config: Mapping[str, Any]) -> None:
    raw = config.get("retriever_override")
    if raw is None:
        return
    override = _validate_retriever_override(raw)
    from tau2.domains.banking_knowledge.retrieval import (
        RETRIEVAL_VARIANTS,
        PipelineSpec,
        RetrievalVariant,
        standard_prompt,
    )

    variant_name = str(config["retrieval_config"])
    prompt = Path(str(override["prompt_template"])).expanduser().resolve()
    if override["backend"] == "bm25":
        spec = PipelineSpec(type="bm25", top_k=int(override["top_k"]))
    else:
        from tau2.knowledge.document_preprocessors.embedding_indexer import (
            EMBEDDER_REGISTRY as document_registry,
        )
        from tau2.knowledge.input_preprocessors.embedding_encoder import (
            EMBEDDER_REGISTRY as query_registry,
        )

        registry_key = "agentinputlab_" + str(override["model_artifact_hash"])[:16]

        def embedder_class(*, prefix: str) -> type[Any]:
            class LocalSentenceTransformerEmbedder:
                def __init__(self, model: str):
                    if model != override["profile_id"]:
                        raise ValueError("tau3 local retriever profile marker differs")
                    self.query_instruction = prefix
                    identity = str(override["model_artifact_hash"])
                    encoder = _LOCAL_ENCODERS.get(identity)
                    if encoder is None:
                        from sentence_transformers import SentenceTransformer

                        encoder = SentenceTransformer(
                            str(override["model_path"]),
                            device=str(override["device"]),
                            local_files_only=True,
                            trust_remote_code=True,
                        )
                        _LOCAL_ENCODERS[identity] = encoder
                    self._encoder = encoder

                def embed(self, texts: list[str]) -> Any:
                    values = [prefix + text for text in texts]
                    vectors = self._encoder.encode(
                        values,
                        batch_size=min(16, max(1, len(values))),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    if vectors.ndim != 2 or vectors.shape[1] != override["expected_dimensions"]:
                        raise ValueError("tau3 local retriever returned unexpected dimensions")
                    return vectors

                def get_name(self) -> str:
                    return str(override["profile_id"])

            return LocalSentenceTransformerEmbedder

        document_registry[registry_key] = embedder_class(prefix=str(override["document_prefix"]))
        query_registry[registry_key] = embedder_class(prefix=str(override["query_prefix"]))
        spec = PipelineSpec(
            type="embedding",
            embedder_type=registry_key,
            embedder_model=str(override["profile_id"]),
            top_k=int(override["top_k"]),
        )
    RETRIEVAL_VARIANTS[variant_name] = RetrievalVariant(
        name=variant_name,
        prompt_template=prompt,
        build_prompt=standard_prompt,
        kb_search=spec,
        supports_top_k=True,
    )


def _intervened_tau_agent_class() -> type[Any]:
    from tau2.agent.llm_agent import LLMAgent, LLMAgentState
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        UserMessage,
    )
    from tau2.utils.llm_utils import generate

    class IntervenedTauAgent(LLMAgent[LLMAgentState]):
        def __init__(
            self,
            *args: Any,
            materialized: Mapping[str, Any],
            budget: Mapping[str, Any],
            history_transform: Any = None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self._materialized = materialized
            self._first_turn = True
            self._budget = budget
            self._budget_stopped = False
            self._budget_model_calls = 0
            self._budget_tool_calls = 0
            self._budget_input_tokens = 0
            self._budget_output_tokens = 0
            self._budget_hard_stops = 0
            self._budget_discarded_tool_calls = 0
            self._history_session = (NativeHistorySession(history_transform, materialized, budget)
                                     if history_transform is not None else None)
            self._budget_max_context_tokens = (
                2**63 - 1
                if budget.get("max_context_tokens") is None
                else int(budget["max_context_tokens"])
            )

        def _history_target_usage(self):
            return dict(model_calls=self._budget_model_calls, input_tokens=self._budget_input_tokens,
                        output_tokens=self._budget_output_tokens, tool_calls=self._budget_tool_calls)

        @property
        def system_prompt(self) -> str:
            base = super().system_prompt
            additions = [self._materialized.get("instruction", "")]
            additions.extend(
                block.get("text", "")
                for block in self._materialized.get("context", [])
                if isinstance(block, Mapping)
            )
            visible = "\n\n".join(item for item in additions if item)
            return base + (
                "\n\n<optimizer_visible_input>\n" + visible + "\n</optimizer_visible_input>"
                if visible
                else ""
            )

        def generate_next_message(
            self,
            message: Any,
            state: LLMAgentState,
        ) -> tuple[AssistantMessage, LLMAgentState]:
            metadata = self._materialized.get("metadata", {})
            clarification = (
                metadata.get("clarification_request") if isinstance(metadata, Mapping) else None
            )
            if self._first_turn and isinstance(clarification, str) and clarification.strip():
                self._first_turn = False
                if message is not None:
                    state.messages.append(message)
                response = AssistantMessage(
                    role="assistant",
                    content=clarification,
                    cost=0.0,
                )
                state.messages.append(response)
                return response, state
            if self._first_turn and isinstance(message, UserMessage):
                message = message.model_copy(
                    update={"content": self._materialized["query"]},
                    deep=True,
                )
            self._first_turn = False
            return super().generate_next_message(message, state)

        def _generate_next_message(
            self,
            message: Any,
            state: LLMAgentState,
        ) -> AssistantMessage:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            if self._budget_stopped:
                return AssistantMessage(
                    role="assistant",
                    content=_SHARED_BUDGET_STOP_MESSAGE,
                    cost=0.0,
                )
            messages = state.system_messages + state.messages
            if self._history_session is not None:
                remaining = self._history_session.remaining(self._history_target_usage())
                if remaining["max_model_calls"] < 1 or remaining["max_output_tokens"] < 1:
                    self._budget_stopped = True
                    self._budget_hard_stops += 1
                    return AssistantMessage(role="assistant", content=_SHARED_BUDGET_STOP_MESSAGE, cost=0.0)
                from tau2.utils.llm_utils import to_litellm_messages
                messages = self._history_session.messages(messages, encode=to_litellm_messages,
                                                          agent_usage=self._history_target_usage())
            try:
                from litellm import token_counter

                normalized_messages = [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in messages
                ]
                # tau2's internal tool-call objects are not OpenAI-shaped.
                # Count their canonical JSON as plain text so every argument,
                # result, system message, and tool schema remains included
                # without asking LiteLLM to reinterpret the structure.
                token_messages = [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "messages": normalized_messages,
                                "tools": [tool.openai_schema for tool in self.tools],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ]
                next_input_tokens = int(token_counter(model=self.llm, messages=token_messages))
            except Exception as exc:
                raise OfficialArenaError(
                    "tau3 could not measure input tokens before generation"
                ) from exc
            history_usage = self._history_session.usage if self._history_session is not None else {}
            history_calls = int(history_usage.get("model_calls", 0))
            history_input = int(history_usage.get("input_tokens", 0))
            history_output = int(history_usage.get("output_tokens", 0))
            remaining_output = int(self._budget["max_output_tokens"]) - self._budget_output_tokens - history_output
            if (
                self._budget_model_calls + history_calls >= int(self._budget["max_model_calls"])
                or self._budget_input_tokens + history_input + next_input_tokens > self._budget_max_context_tokens
                or remaining_output < 1
            ):
                self._budget_stopped = True
                self._budget_hard_stops += 1
                return AssistantMessage(
                    role="assistant",
                    content=_SHARED_BUDGET_STOP_MESSAGE,
                    cost=0.0,
                )
            call_args = dict(self.llm_args)
            call_args["max_tokens"] = min(
                int(call_args.get("max_tokens", remaining_output)),
                remaining_output,
            )
            try:
                response = generate(
                    model=self.llm,
                    tools=self.tools,
                    messages=messages,
                    call_name="agent_response",
                    **call_args,
                )
            except Exception as error:
                if not _is_gpt_oss_generation_boundary_failure(
                    error,
                    model_name=self.llm,
                ):
                    raise
                # A malformed Harmony header or a request beyond the serving
                # context window is an evaluated trajectory outcome, not a
                # missing experiment row. Preserve it as the same scored hard
                # stop used for an exhausted shared budget.
                self._budget_model_calls += 1
                self._budget_input_tokens += next_input_tokens
                self._budget_stopped = True
                self._budget_hard_stops += 1
                return AssistantMessage(
                    role="assistant",
                    content=_SHARED_BUDGET_STOP_MESSAGE,
                    cost=0.0,
                )
            usage = getattr(response, "usage", None)
            if isinstance(usage, Mapping):
                actual_input = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                actual_output = int(
                    usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                )
            else:
                actual_input = next_input_tokens
                actual_output = 0
            self._budget_model_calls += 1
            self._budget_input_tokens += actual_input
            self._budget_output_tokens += actual_output
            if (
                self._budget_input_tokens + history_input > self._budget_max_context_tokens
                or self._budget_output_tokens + history_output > int(self._budget["max_output_tokens"])
            ):
                raise OfficialArenaError("tau3 provider exceeded its measured shared budget")
            response, kept_tool_calls, discarded_tool_calls = _apply_tool_call_budget(
                response,
                remaining_tool_calls=max(
                    0,
                    int(self._budget["max_tool_calls"]) - self._budget_tool_calls,
                ),
            )
            self._budget_tool_calls += kept_tool_calls
            if discarded_tool_calls:
                self._budget_stopped = True
                self._budget_hard_stops += 1
                self._budget_discarded_tool_calls += discarded_tool_calls
            return response

    return IntervenedTauAgent


def factory() -> Tau3Provider:
    return Tau3Provider()
