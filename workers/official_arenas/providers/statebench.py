"""Production provider for the pinned Microsoft STATE-Bench runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
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
from ._live_history import ResponsesHistorySession, validate_history_transform
from ._message_compat import (
    fold_responses_system_messages_into_instructions,
    normalize_qwen38_reasoning_effort,
    requires_qwen38_initial_system_message,
)


class _SharedBudgetStop(RuntimeError):
    """Internal, scored hard stop after a STATE-Bench resource cap is reached."""


class _EvaluatedAgentOutputInvalid(RuntimeError):
    """A recognized malformed model generation from the evaluated agent."""

    def __init__(self, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: evaluated agent output is invalid")
        self.error = error


def _model_output_invalid_result(
    *,
    replay_record: Mapping[str, Any],
    role: str,
    seed: int,
    work_dir: Path,
    error: Exception,
    measured_usage: Mapping[str, int | float],
) -> ProviderArmResult:
    """Materialize malformed generated tool arguments as an official zero score."""

    protected = replay_record["protected"]
    trace = {
        "provider": "state_bench",
        "source_revision": StateBenchProvider.source_revision,
        "protocol_id": protected["protocol_id"],
        "task_id": protected["task_id"],
        "domain": protected["domain"],
        "role": role,
        "seed": seed,
        "termination_reason": "generated_tool_arguments_invalid",
        "generation_controls": {
            "hard_stops": 1,
            "failure_type": type(error).__name__,
            "failed_generation_scope": "evaluated_agent",
        },
        "score_provenance": "paper_model_output_invalid_zero_action",
    }
    work_dir.joinpath("model-output-invalid.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    usage = {
        **measured_usage,
        "cost_usd": 0.0,
        "unpriced_model_calls": int(measured_usage.get("model_calls", 0)),
    }
    return ProviderArmResult(
        metrics={
            "task_completion_pass": 0.0,
            "state_requirements_met": 0.0,
            "task_requirements_met": 0.0,
            "adapter_hard_stops": 1.0,
        },
        trace=trace,
        usage=usage,
        runtime_instance_id=content_hash(
            {
                "provider": "state_bench",
                "checkpoint_id": replay_record["checkpoint_id"],
                "role": role,
                "seed": seed,
                "termination_reason": trace["termination_reason"],
                "work_dir": str(work_dir),
            }
        ),
    )


class StateBenchProvider:
    benchmark_id = "state_bench.common_support"
    source_revision = "4efcbf2d4fe60df04878859b692d9391f3d5b33a"
    scorer_id = "state_bench.official_v0.8.1"
    scorer_revision = "00f5c68a83e55535797435f1fed59e9d334878ef3431f9906d703778dbcf09e8"

    _CONFIG_FIELDS = frozenset(
        {
            "domains",
            "split",
            "limit",
            "agent_provider",
            "agent_api_key_var",
            "score_reasoning_effort",
            "locked_dependency_hashes",
        }
    )
    _OPENAI_COMPATIBLE_FIELDS = frozenset(
        {
            "agent_model",
            "agent_base_url",
            "evaluator_model",
            "evaluator_base_url",
            "evaluator_api_key_var",
        }
    )
    _OPTIONAL_CONFIG_FIELDS = frozenset({
        "budget_override", "agent_timeout_seconds", "environment_timeout_seconds", "request_attempts",
        "agent_extra_body", "environment_extra_body", "agent_extra_body_mode", "environment_extra_body_mode",
        "agent_responses_reasoning_disabled",
    })

    def __init__(self, *, history_transform=None):
        validate_history_transform(history_transform)
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
                raise OfficialArenaError("STATE-Bench checkout revision differs")
            from state_bench.client import (
                build_llm_client,
                build_locked_judge_client,
                build_user_sim_client,
            )
            from state_bench.domain import get_domain_config
            from state_bench.orchestrator import run_task
            from state_bench.scoring import combine_task_completion

            del (
                build_llm_client,
                build_locked_judge_client,
                build_user_sim_client,
                get_domain_config,
                run_task,
                combine_task_completion,
            )
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
        from state_bench.domain import get_domain_config
        from state_bench.paths import domain_tasks_dir
        from state_bench.protocol import load_default_protocol, load_split_task_ids
        from state_bench.schemas import TaskDefinition

        protocol = load_default_protocol()
        errors = protocol.validate_prompt_hashes()
        if errors:
            raise OfficialArenaError("STATE-Bench protocol prompt hashes differ")

        rows: list[Mapping[str, Any]] = []
        for domain_name in parsed["domains"]:
            domain = get_domain_config(domain_name)
            task_ids = load_split_task_ids(
                domain_name,
                parsed["split"],
                protocol.split_version,
            )
            for task_id in task_ids:
                task = TaskDefinition.load(domain_tasks_dir(domain_name) / f"{task_id}.json")
                user_id = task.user_id
                if not user_id:
                    continue
                instruction = domain.agent_system_prompt.format(
                    now=task.now,
                    user_id=user_id,
                )
                tool_names = tuple(
                    str(tool.get("name"))
                    for tool in domain.tool_schemas
                    if isinstance(tool, Mapping) and isinstance(tool.get("name"), str)
                )
                if not tool_names:
                    raise OfficialArenaError("STATE-Bench domain exposes no tool schemas")
                visible_state = {
                    "query": task.opening_message,
                    "task_id": task.task_id,
                    "instruction": instruction,
                    "context": [
                        {
                            "text": instruction,
                            "block_id": "official_policy",
                            "source": "state_bench.domain.agent_system_prompt",
                            "score": None,
                            "metadata": {"domain": domain_name},
                        },
                        {
                            "text": json.dumps(
                                domain.tool_schemas,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "block_id": "official_tool_catalog",
                            "source": "state_bench.domain.tool_schemas",
                            "score": None,
                            "metadata": {"domain": domain_name},
                        },
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": task.opening_message,
                            "name": None,
                        }
                    ],
                    "tool_names": list(tool_names),
                    "metadata": {
                        "benchmark_id": self.benchmark_id,
                        "domain": domain_name,
                        "checkpoint_phase": "turn_zero",
                        "clarification_roundtrip_available": True,
                        "state_restore_available": True,
                    },
                    "budget": visible_shared_budget(
                        parsed,
                        {
                            "max_model_calls": domain.max_agent_turns * 9,
                            "max_retrieval_calls": 32,
                            "max_tool_calls": domain.max_agent_turns * 8,
                            "max_output_tokens": 65536,
                            "max_context_tokens": None,
                            "max_cost_usd": None,
                        },
                    ),
                }
                rows.append(
                    {
                        "checkpoint_id": f"{domain_name}:{task.task_id}",
                        "group_key": f"{domain_name}:{task.task_id}",
                        "visible_state": visible_state,
                        "eligible_families": [
                            "reformulate",
                            "decompose",
                            "acquire",
                            "clarify",
                            "curate",
                        ],
                        "protected": {
                            "domain": domain_name,
                            "task_id": task.task_id,
                            "user_id": user_id,
                            "protocol_id": protocol.protocol_id,
                            "locked_dependency_hashes": parsed["locked_dependency_hashes"],
                        },
                    }
                )
                if len(rows) >= parsed["limit"]:
                    return tuple(rows)
        if not rows:
            raise OfficialArenaError("STATE-Bench capture found no protocol tasks")
        return tuple(rows)

    def run_arm(
        self, config: Mapping[str, Any], *, replay_record: Mapping[str, Any],
        arm_payload: Mapping[str, Any], seed: int, role: str, work_dir: Path,
    ) -> ProviderArmResult:
        context = {}
        try:
            result = self._run_arm(config, replay_record=replay_record, arm_payload=arm_payload,
                                   seed=seed, role=role, work_dir=work_dir, _history_context=context)
        finally:
            agent = context.get("agent")
            session = getattr(agent, "_history_session", None)
            if session is not None:
                session.write_receipt(work_dir / "live-history.json",
                                      target_usage=agent.agentinputlab_usage())
        if session is None:
            return result
        receipt_hash = hashlib.sha256((work_dir / "live-history.json").read_bytes()).hexdigest()
        return ProviderArmResult(metrics=result.metrics,
            trace={**result.trace, "live_history_transform": session.hook.fingerprint(),
                   "live_history_receipt": {"path": "live-history.json", "sha256": receipt_hash},
                   "history_usage": dict(session.usage)},
            usage=session.combined_usage(result.usage),
            runtime_instance_id=content_hash({"native_runtime": result.runtime_instance_id,
                                             "live_history_receipt_sha256": receipt_hash}))

    def _run_arm(
        self,
        config: Mapping[str, Any],
        *,
        replay_record: Mapping[str, Any],
        arm_payload: Mapping[str, Any],
        seed: int,
        role: str,
        work_dir: Path,
        _history_context: dict,
    ) -> ProviderArmResult:
        parsed = self._validate_config(config)
        state = materialized_state(arm_payload)
        budget = require_materialized_budget(state, parsed)
        optimizer_usage = intervention_usage(arm_payload)
        remaining_model_calls = max(
            0,
            int(budget["max_model_calls"]) - int(optimizer_usage.get("model_calls", 0)),
        )
        remaining_output_tokens = max(
            0,
            int(budget["max_output_tokens"]) - int(optimizer_usage.get("output_tokens", 0)),
        )
        context_cap = budget.get("max_context_tokens")
        remaining_input_tokens = (
            2**63 - 1
            if context_cap is None
            else max(
                0,
                int(context_cap) - int(optimizer_usage.get("input_tokens", 0)),
            )
        )
        remaining_tool_calls = max(
            0,
            int(budget["max_tool_calls"]) - int(optimizer_usage.get("tool_calls", 0)),
        )
        override = parsed.get("budget_override")
        per_call_output_cap = (
            int(override["max_output_tokens_per_call"])
            if isinstance(override, Mapping)
            else min(512, remaining_output_tokens)
        )
        protected = replay_record["protected"]
        domain_name = protected["domain"]
        task_id = protected["task_id"]
        user_id = protected["user_id"]

        from state_bench.agents.base import AgentRuntimeContext
        from state_bench.client import (
            build_llm_client,
            build_locked_judge_client,
            build_user_sim_client,
        )
        from state_bench.domain import get_domain_config
        from state_bench.env_loader import load_task_environment
        from state_bench.orchestrator import run_task
        from state_bench.paths import domain_tasks_dir
        from state_bench.protocol import load_default_protocol
        from state_bench.schemas import TaskDefinition
        from state_bench.scoring import (
            TaskRequirementsJudge,
            combine_task_completion,
            evaluate_state_requirements,
        )

        domain = get_domain_config(domain_name)
        task = TaskDefinition.load(domain_tasks_dir(domain_name) / f"{task_id}.json")
        env_data, _ = load_task_environment(domain, task)
        env_copy = env_data.deep_copy()
        env = domain.environment_class(env_copy, now=task.now)
        agent_system_prompt = domain.agent_system_prompt.format(
            now=task.now,
            user_id=user_id,
        )
        if self._OPENAI_COMPATIBLE_FIELDS.issubset(parsed):
            agent_client = _openai_compatible_client(
                model=parsed["agent_model"],
                base_url=parsed["agent_base_url"],
                api_key_var=parsed["agent_api_key_var"],
                single_attempt=parsed.get("request_attempts") == 1,
                extra_body=parsed.get("agent_extra_body"),
                extra_body_mode=parsed.get("agent_extra_body_mode", "merge"),
                responses_reasoning_disabled=parsed.get("agent_responses_reasoning_disabled", False),
                **native_timeout_options(parsed, "agent_timeout_seconds"),
            )
            simulator_client = _openai_compatible_client(
                model=parsed["evaluator_model"],
                base_url=parsed["evaluator_base_url"],
                api_key_var=parsed["evaluator_api_key_var"],
                single_attempt=parsed.get("request_attempts") == 1,
                extra_body=parsed.get("environment_extra_body"),
                extra_body_mode=parsed.get("environment_extra_body_mode", "merge"),
                **native_timeout_options(parsed, "environment_timeout_seconds"),
            )
            judge_client = _openai_compatible_client(
                model=parsed["evaluator_model"],
                base_url=parsed["evaluator_base_url"],
                api_key_var=parsed["evaluator_api_key_var"],
                single_attempt=parsed.get("request_attempts") == 1,
                extra_body=parsed.get("environment_extra_body"),
                extra_body_mode=parsed.get("environment_extra_body_mode", "merge"),
                **native_timeout_options(parsed, "environment_timeout_seconds"),
            )
            _install_statebench_judge_json_compat(judge_client)
        else:
            agent_client = build_llm_client(
                provider=parsed["agent_provider"],
                api_key_var=parsed["agent_api_key_var"],
            )
            simulator_client = build_user_sim_client()
            judge_client = build_locked_judge_client()
        runtime_context = AgentRuntimeContext(
            task_id=task.task_id,
            user_id=user_id,
            domain=domain.name,
            now=task.now,
            task_summary=task.task_summary,
            state_requirements=task.state_requirements,
            task_requirements=task.task_requirements,
        )
        agent = _intervened_agent_class()(
            agent_client,
            agent_system_prompt,
            domain.tool_schemas,
            env.tool_handlers,
            runtime_context=runtime_context,
            materialized=state,
            max_api_calls=remaining_model_calls,
            max_output_tokens=remaining_output_tokens,
            max_input_tokens=remaining_input_tokens,
            max_output_tokens_per_call=per_call_output_cap,
            max_tool_calls=remaining_tool_calls,
            history_transform=self.history_transform,
        )
        _history_context["agent"] = agent
        try:
            trajectory = run_task(
                task,
                env_data,
                user_id,
                client=None,
                domain=domain,
                agent=agent,
                env=env,
                trajectory_metadata={
                    "agentinputlab": {
                        "release_id": arm_payload["release_id"],
                        "role": role,
                        "seed": seed,
                        "output_state_hash": arm_payload["intervention_execution"][
                            "output_state_hash"
                        ],
                    }
                },
                simulator_client=simulator_client,
            )
        except _EvaluatedAgentOutputInvalid as failure:
            return _model_output_invalid_result(
                replay_record=replay_record,
                role=role,
                seed=seed,
                work_dir=work_dir,
                error=failure.error,
                measured_usage=agent.agentinputlab_usage(),
            )

        tool_calls = [
            call
            for message in trajectory.conversation
            for call in (message.get("tool_calls") or [])
        ]
        state_result = evaluate_state_requirements(task, trajectory.state_diff)
        protocol = load_default_protocol()
        judge_reasoning_effort = normalize_qwen38_reasoning_effort(
            parsed.get("evaluator_model"),
            parsed["score_reasoning_effort"] or protocol.judge_reasoning_effort,
        )
        judge = TaskRequirementsJudge(
            client=judge_client,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=judge_reasoning_effort,
        )
        task_result = judge.evaluate(
            task,
            trajectory.conversation,
            tool_calls,
            trajectory.state_diff,
        )
        completion = combine_task_completion(state_result, task_result)
        if completion is None:
            raise OfficialArenaError("STATE-Bench official task judge did not complete")

        trace = {
            "provider": "state_bench",
            "source_revision": self.source_revision,
            "protocol_id": protected["protocol_id"],
            "task_id": task_id,
            "domain": domain_name,
            "role": role,
            "seed": seed,
            "conversation": trajectory.conversation,
            "state_diff": trajectory.state_diff.to_dict(),
            "efficiency": (trajectory.efficiency.to_dict() if trajectory.efficiency else None),
            "task_completion_pass": completion,
            "adapter_compatibility_repairs": {
                "qwen3_4b_judge_reasoning_json": int(
                    getattr(
                        judge_client,
                        "_agentinputlab_qwen3_4b_judge_json_repairs",
                        0,
                    )
                )
            },
        }
        token_usage = trajectory.token_usage.to_dict()
        audited_agent_usage = agent.agentinputlab_usage()
        usage = {
            # Conversation rows are not API-call receipts: a clarification or
            # budget-stop message can be emitted without calling the model.
            # The wrapper counters below are incremented exactly at the model
            # and tool boundaries and are the only counters compatible with
            # the shared remaining-budget calculation above.
            "model_calls": audited_agent_usage["model_calls"],
            "tool_calls": audited_agent_usage["tool_calls"],
            "input_tokens": audited_agent_usage["input_tokens"],
            "output_tokens": audited_agent_usage["output_tokens"],
            "cost_usd": float(token_usage.get("total_cost_usd", 0.0) or 0.0),
        }
        work_dir.joinpath("trajectory.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ProviderArmResult(
            metrics={
                "task_completion_pass": float(completion),
                "state_requirements_met": float(state_result.score),
                "task_requirements_met": float(task_result.score),
            },
            trace=trace,
            usage=usage,
            runtime_instance_id=content_hash(
                {
                    "provider": "state_bench",
                    "checkpoint_id": replay_record["checkpoint_id"],
                    "role": role,
                    "seed": seed,
                    "work_dir": str(work_dir),
                }
            ),
        )

    def _validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        config = resolve_shared_budget_config(config)
        fields = set(config) if isinstance(config, Mapping) else set()
        base_fields = fields - self._OPTIONAL_CONFIG_FIELDS
        valid_fields = {
            self._CONFIG_FIELDS,
            self._CONFIG_FIELDS | self._OPENAI_COMPATIBLE_FIELDS,
        }
        if not isinstance(config, Mapping) or base_fields not in valid_fields:
            raise OfficialArenaError("STATE-Bench provider config fields are invalid")
        if "request_attempts" in config:
            if type(config["request_attempts"]) is not int or config["request_attempts"] != 1:
                raise OfficialArenaError("STATE-Bench request_attempts must be 1")
            if not self._OPENAI_COMPATIBLE_FIELDS.issubset(config):
                raise OfficialArenaError("STATE-Bench request_attempts requires the OpenAI-compatible binding")
        if "agent_responses_reasoning_disabled" in config:
            if type(config["agent_responses_reasoning_disabled"]) is not bool:
                raise OfficialArenaError("STATE-Bench agent_responses_reasoning_disabled must be boolean")
            if not self._OPENAI_COMPATIBLE_FIELDS.issubset(config):
                raise OfficialArenaError("STATE-Bench Responses reasoning option requires the OpenAI-compatible binding")
        for name in ("agent_timeout_seconds", "environment_timeout_seconds"):
            native_timeout_options(config, name)
            if name in config and not self._OPENAI_COMPATIBLE_FIELDS.issubset(config):
                raise OfficialArenaError("explicit STATE-Bench timeouts require the OpenAI-compatible binding")
        for name in ("agent_extra_body", "environment_extra_body"):
            native_extra_body(config, name)
            if name in config and not self._OPENAI_COMPATIBLE_FIELDS.issubset(config):
                raise OfficialArenaError("explicit STATE-Bench extra_body requires the OpenAI-compatible binding")
        domains = config["domains"]
        if (
            isinstance(domains, (str, bytes))
            or not isinstance(domains, Sequence)
            or not domains
            or len(domains) != len(set(domains))
            or any(not isinstance(item, str) or not item.strip() for item in domains)
        ):
            raise OfficialArenaError("STATE-Bench domains must be unique strings")
        split = config["split"]
        if split not in {"train", "test"}:
            raise OfficialArenaError("STATE-Bench split must be train or test")
        limit = config["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise OfficialArenaError("STATE-Bench limit must be positive")
        if config["agent_provider"] not in {"openai", "azure_openai"}:
            raise OfficialArenaError("STATE-Bench agent_provider is unsupported")
        key_fields = ["agent_api_key_var"]
        if self._OPENAI_COMPATIBLE_FIELDS.issubset(config):
            if config["agent_provider"] != "openai":
                raise OfficialArenaError(
                    "STATE-Bench OpenAI-compatible runtime requires agent_provider=openai"
                )
            for name in (
                "agent_model",
                "agent_base_url",
                "evaluator_model",
                "evaluator_base_url",
            ):
                if not isinstance(config[name], str) or not config[name].strip():
                    raise OfficialArenaError(f"STATE-Bench {name} is invalid")
            key_fields.append("evaluator_api_key_var")
        for name in key_fields:
            value = config[name]
            if (
                not isinstance(value, str)
                or not value.strip()
                or not value.replace("_", "").isalnum()
            ):
                raise OfficialArenaError(f"STATE-Bench {name} is invalid")
        effort = config["score_reasoning_effort"]
        if effort is not None and effort not in {"low", "medium", "high"}:
            raise OfficialArenaError("STATE-Bench score_reasoning_effort is invalid")
        validate_dependency_hashes(
            config["locked_dependency_hashes"],
            label="STATE-Bench",
        )
        return config


def _openai_compatible_client(
    *,
    model: str,
    base_url: str,
    api_key_var: str,
    timeout: float | None = None,
    single_attempt: bool = False,
    extra_body: Mapping[str, Any] | None = None,
    extra_body_mode: str = "merge",
    responses_reasoning_disabled: bool = False,
) -> Any:
    """Inject only the transport while retaining pinned STATE-Bench client logic."""
    from openai import OpenAI
    from state_bench.client import LLMClient

    if type(single_attempt) is not bool:
        raise OfficialArenaError("STATE-Bench single_attempt must be boolean")
    if type(responses_reasoning_disabled) is not bool:
        raise OfficialArenaError("STATE-Bench responses_reasoning_disabled must be boolean")
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise OfficialArenaError(
            f"STATE-Bench OpenAI-compatible environment {api_key_var} is missing"
        )
    extensions = native_extra_body({"extra_body": {} if extra_body is None else extra_body,
        "extra_body_mode": extra_body_mode}, "extra_body")
    client = LLMClient.__new__(LLMClient)
    client.provider = "openai"
    client.endpoint = base_url
    client.deployment = model
    client.deployments = [model]
    client._chat_response_history = {}
    options = {} if timeout is None else native_timeout_options({"timeout": timeout}, "timeout")
    if single_attempt:
        _bind_single_attempt(client)
    client._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, **options)
    if extensions or extra_body_mode == "replace" or responses_reasoning_disabled:
        original_create = client._client.responses.create

        def configured_create(**kwargs: Any) -> Any:
            # The pinned native methods own the request and its parsing. Only
            # explicitly configured, non-reserved transport fields are added.
            if extensions or extra_body_mode == "replace":
                inherited = kwargs.get("extra_body", {}) if extra_body_mode == "merge" else {}
                kwargs["extra_body"] = {**inherited, **copy.deepcopy(extensions)}
            if responses_reasoning_disabled and "tools" in kwargs:
                # The upstream client drops temperature when reasoning_effort
                # is set. Bind this explicit transport option at the SDK
                # boundary so native temperature and budget remain unchanged.
                disabled = {"effort": "none"}
                if "reasoning" in kwargs and kwargs["reasoning"] != disabled:
                    raise OfficialArenaError("STATE-Bench explicit reasoning conflicts with disabled reasoning")
                kwargs["reasoning"] = disabled
            return original_create(**kwargs)

        client._client.responses.create = configured_create
    _install_stateless_tool_history(client)
    return client


def _bind_single_attempt(client: Any) -> None:
    """Limit the pinned native retry decorators on this instance, not its class.

    Keep native parsing/error classification. Both JSON layers are bound so
    complete_json cannot retry an already-attempted complete_json_response.
    Legacy callers that do not request this binding retain their native policy.
    """
    methods = {}
    for name in ("complete_chat", "complete_with_tools", "complete_json_response", "complete_json"):
        method = getattr(client, name, None)
        if not callable(getattr(method, "retry_with", None)):
            raise OfficialArenaError("STATE-Bench native retry binding is unavailable: " + name)
        methods[name] = method.retry_with(stop=lambda state: True).__get__(client, type(client))
    for name, method in methods.items():
        setattr(client, name, method)


class _ResponseTextOverride:
    """Delegate a Responses object while exposing one validated text repair."""

    def __init__(self, response: Any, output_text: str) -> None:
        self._response = response
        self.output_text = output_text

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def _response_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _qwen3_4b_judge_reasoning_json(response: Any, *, model: str) -> str | None:
    """Recover only a completed Qwen3-4B judge JSON emitted as reasoning text.

    Some Qwen3-4B vLLM Responses generations place the JSON-object answer in
    a ``reasoning_text`` block while leaving the SDK's ``output_text`` empty.
    STATE-Bench then retries and ultimately drops an otherwise valid official
    judge result.  Keep the repair deliberately narrower than generic response
    promotion: it applies only to this served alias, completed responses, one
    reasoning block, and the exact top-level shape consumed by
    ``TaskRequirementsJudge``.
    """

    alias = model.strip().casefold().replace("_", "-")
    if alias != "qwen3-4b":
        return None
    if _response_field(response, "status") != "completed":
        return None
    if _response_field(response, "output_text") not in {None, ""}:
        return None

    candidates: list[str] = []
    output = _response_field(response, "output")
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        return None
    for item in output:
        if _response_field(item, "type") != "reasoning":
            continue
        content = _response_field(item, "content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for block in content:
            if _response_field(block, "type") != "reasoning_text":
                continue
            text = _response_field(block, "text")
            if isinstance(text, str) and text.strip():
                candidates.append(text)
    if len(candidates) != 1:
        return None
    try:
        parsed = json.loads(candidates[0])
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("details"), list):
        return None
    return candidates[0]


def _install_statebench_judge_json_compat(client: Any) -> None:
    """Install the narrow Qwen3-4B JSON repair on the judge client only."""

    original = client.complete_json_response
    client._agentinputlab_qwen3_4b_judge_json_repairs = 0

    def complete_json_response(**call_kwargs: Any) -> Any:
        response = original(**call_kwargs)
        recovered = _qwen3_4b_judge_reasoning_json(
            response,
            model=str(getattr(client, "deployment", "")),
        )
        if recovered is None:
            return response
        client._agentinputlab_qwen3_4b_judge_json_repairs += 1
        return _ResponseTextOverride(response, recovered)

    client.complete_json_response = complete_json_response


def _install_stateless_tool_history(client: Any) -> None:
    """Replay Responses tool history for stateless OpenAI-compatible servers.

    STATE-Bench natively chains tool turns with ``previous_response_id``.
    Self-hosted vLLM implements the Responses wire format but does not retain
    provider-side response objects.  Preserve the upstream turn loop while
    replacing only that transport assumption with explicit input replay.
    """

    original = client.complete_with_tools
    histories: dict[str, list[Any]] = {}

    def replayable(items: list[Any]) -> list[Any]:
        if str(getattr(client, "deployment", "")).casefold() != "gpt-oss-20b":
            return items
        # vLLM's Responses endpoint does not accept its own occasional
        # GPT-OSS ``mcp_call`` output as a subsequent input item. STATE-Bench
        # ignores that item (only function_call is executable), so omit only
        # this inert parser artifact from stateless replay.
        return [
            item
            for item in items
            if (
                item.get("type") if isinstance(item, Mapping) else getattr(item, "type", None)
            )
            != "mcp_call"
        ]

    def stateless_complete_with_tools(**call_kwargs: Any) -> Any:
        previous = call_kwargs.get("previous_response_id")
        incoming = list(call_kwargs.get("input") or ())
        if previous is not None:
            history = histories.get(str(previous))
            if history is None:
                raise OfficialArenaError(
                    "STATE-Bench stateless tool continuation has no local response history"
                )
            incoming = [*replayable(history), *incoming]
            call_kwargs["input"] = incoming
            call_kwargs["previous_response_id"] = None
        else:
            history_view = getattr(client, "_agentinputlab_responses_history", None)
            if history_view is not None:
                call_kwargs = history_view(dict(call_kwargs, input=incoming))
                incoming = list(call_kwargs["input"])
        if requires_qwen38_initial_system_message(client.deployment):
            normalized_input, normalized_instructions = (
                fold_responses_system_messages_into_instructions(
                    incoming,
                    call_kwargs.get("instructions"),
                )
            )
            call_kwargs["input"] = normalized_input
            call_kwargs["instructions"] = normalized_instructions
        response = original(**call_kwargs)
        response_id = getattr(response, "id", None)
        output = getattr(response, "output", None)
        if not isinstance(response_id, str) or not response_id or not isinstance(output, list):
            raise OfficialArenaError("STATE-Bench Responses result has no replayable output")
        histories[response_id] = [*incoming, *output]
        return response

    client.complete_with_tools = stateless_complete_with_tools
    client._agentinputlab_stateless_responses = True


def _is_gpt_oss_harmony_parse_failure(error: Exception, *, model_name: str) -> bool:
    """Recognize only GPT-OSS response-parser HTTP-500 failures."""

    if model_name.casefold() != "gpt-oss-20b" or getattr(error, "status_code", None) != 500:
        return False
    body = getattr(error, "body", None)
    message: Any = None
    if isinstance(body, Mapping):
        message = body.get("message")
        nested = body.get("error")
        if message is None and isinstance(nested, Mapping):
            message = nested.get("message")
    if not isinstance(message, str):
        return False
    normalized = message.casefold()
    return normalized.startswith("unexpected tokens remaining in message header") or normalized.startswith(
        "unexpected token "
    )


def _intervened_agent_class() -> type[Any]:
    from state_bench.agents.state_bench import StateBenchAgent

    class IntervenedStateBenchAgent(StateBenchAgent):
        def __init__(
            self,
            *args: Any,
            materialized: Mapping[str, Any],
            max_api_calls: int,
            max_output_tokens: int,
            max_input_tokens: int,
            max_output_tokens_per_call: int,
            max_tool_calls: int,
            history_transform=None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self._materialized = materialized
            self._first_turn = True
            self._budget_stopped = False
            self._budget_api_calls = 0
            self._budget_input_tokens = 0
            self._budget_output_tokens = 0
            self._budget_tool_calls = 0
            self._max_api_calls = max_api_calls
            self._max_input_tokens = max_input_tokens
            self._max_output_tokens = max_output_tokens
            self._max_output_tokens_per_call = max_output_tokens_per_call
            self._max_tool_calls = max_tool_calls
            self._history_session = (ResponsesHistorySession(history_transform, materialized,
                dict(max_model_calls=max_api_calls, max_output_tokens=max_output_tokens))
                if history_transform is not None else None)
            if self._history_session is not None:
                if not hasattr(self.client, "_agentinputlab_stateless_responses"):
                    raise OfficialArenaError("STATE history requires the native stateless Responses transport")

                def history_view(call_kwargs):
                    call_kwargs["input"], call_kwargs["instructions"] = self._history_session.responses(
                        call_kwargs["input"], instructions=call_kwargs.get("instructions"),
                        agent_usage=self.agentinputlab_usage())
                    remaining = self._history_session.remaining(self.agentinputlab_usage())
                    if (remaining["max_model_calls"] < 1 or remaining["max_output_tokens"] < 1
                            or self._budget_input_tokens + self._history_session.usage.get("input_tokens", 0)
                            >= self._max_input_tokens):
                        raise _SharedBudgetStop
                    call_kwargs["max_tokens"] = min(int(call_kwargs["max_tokens"]),
                                                    remaining["max_output_tokens"])
                    return call_kwargs

                self.client._agentinputlab_responses_history = history_view
            original = self.client.complete_with_tools

            def budgeted_complete_with_tools(**call_kwargs: Any) -> Any:
                remaining_output = self._max_output_tokens - self._budget_output_tokens
                calls = self._budget_api_calls
                if self._history_session is not None:
                    remaining = self._history_session.remaining(self.agentinputlab_usage())
                    remaining_output = remaining["max_output_tokens"]
                    calls += int(self._history_session.usage.get("model_calls", 0))
                if calls >= self._max_api_calls or remaining_output < 1:
                    raise _SharedBudgetStop
                requested = int(call_kwargs.get("max_tokens", 4096))
                call_kwargs["max_tokens"] = min(
                    requested,
                    self._max_output_tokens_per_call,
                    remaining_output,
                )
                response = original(**call_kwargs)
                self._budget_api_calls += 1
                usage = getattr(response, "usage", None)
                self._budget_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self._budget_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                if (
                    self._budget_input_tokens + (self._history_session.usage.get("input_tokens", 0)
                        if self._history_session is not None else 0) > self._max_input_tokens
                    or self._budget_output_tokens + (self._history_session.usage.get("output_tokens", 0)
                        if self._history_session is not None else 0) > self._max_output_tokens
                ):
                    raise _SharedBudgetStop
                return response

            self.client.complete_with_tools = budgeted_complete_with_tools
            original_handlers = dict(self.tool_handlers)
            for tool_name, handler in original_handlers.items():

                def budgeted_handler(
                    arguments: Any,
                    *,
                    _handler: Any = handler,
                ) -> Any:
                    if self._budget_tool_calls >= self._max_tool_calls:
                        raise _SharedBudgetStop
                    self._budget_tool_calls += 1
                    return _handler(arguments)

                self.tool_handlers[tool_name] = budgeted_handler

        def prepare_conversation(self, conversation: list[Any]) -> list[Any]:
            prepared = [dict(item) if isinstance(item, Mapping) else item for item in conversation]
            if prepared and isinstance(prepared[0], Mapping):
                prepared[0] = {**prepared[0], "content": self._materialized["query"]}
            additions = [self._materialized.get("instruction", "")]
            additions.extend(
                item.get("text", "")
                for item in self._materialized.get("context", [])
                if isinstance(item, Mapping)
            )
            injected = "\n\n".join(item for item in additions if item)
            return self.inject_system_message(prepared, injected)

        def act(self, conversation: list[Any]) -> tuple[str, list[dict[str, Any]], list[Any]]:
            metadata = self._materialized.get("metadata", {})
            clarification = (
                metadata.get("clarification_request") if isinstance(metadata, Mapping) else None
            )
            if self._first_turn and isinstance(clarification, str) and clarification.strip():
                self._first_turn = False
                return clarification, [], [{"role": "assistant", "content": clarification}]
            self._first_turn = False
            if self._budget_stopped:
                message = "I could not complete the task within the shared run budget."
                return message, [], [{"role": "assistant", "content": message}]
            try:
                return super().act(conversation)
            except _SharedBudgetStop:
                self._budget_stopped = True
                message = "I could not complete the task within the shared run budget."
                return message, [], [{"role": "assistant", "content": message}]
            except json.JSONDecodeError as error:
                raise _EvaluatedAgentOutputInvalid(error) from error
            except Exception as error:
                if not _is_gpt_oss_harmony_parse_failure(
                    error,
                    model_name=self.client.deployment,
                ):
                    raise
                self._budget_api_calls += 1
                raise _EvaluatedAgentOutputInvalid(error) from None

        def agentinputlab_usage(self) -> Mapping[str, int]:
            """Return counters collected at the actual model/tool boundaries."""

            return {
                "model_calls": self._budget_api_calls,
                "tool_calls": self._budget_tool_calls,
                "input_tokens": self._budget_input_tokens,
                "output_tokens": self._budget_output_tokens,
            }

    return IntervenedStateBenchAgent


def factory() -> StateBenchProvider:
    return StateBenchProvider()
