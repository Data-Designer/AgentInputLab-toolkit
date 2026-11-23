"""Production provider for the pinned ToolSandbox official runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.contracts.serialization import to_jsonable

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
from ._message_compat import (
    coalesce_chat_system_messages,
    requires_qwen38_initial_system_message,
)


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
    compatibility_repairs: list[Mapping[str, Any]],
    budget_enforcements: list[Mapping[str, int]],
    user_simulator_usage: Mapping[str, int | float],
) -> ProviderArmResult:
    """Materialize a malformed evaluated-agent response as an official zero score."""

    trace = {
        "provider": "toolsandbox",
        "source_revision": ToolSandboxProvider.source_revision,
        "scenario_name": replay_record["protected"]["scenario_name"],
        "role": role,
        "seed": seed,
        "termination_reason": "generated_agent_response_invalid",
        "generation_controls": {
            "hard_stops": 1,
            "failure_type": type(error).__name__,
            "failed_generation_scope": "evaluated_agent",
        },
        "compatibility_repairs": list(compatibility_repairs),
        "budget_enforcements": list(budget_enforcements),
        "user_simulator_usage": dict(user_simulator_usage),
        "score_provenance": "paper_model_output_invalid_zero_action",
    }
    work_dir.joinpath("model-output-invalid.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ProviderArmResult(
        metrics={
            "official_similarity": 0.0,
            "milestone_similarity": 0.0,
            "minefield_similarity": 0.0,
            "turn_count": 0.0,
            "adapter_compatibility_repairs": float(len(compatibility_repairs)),
            "adapter_hard_stops": 1.0,
        },
        trace=trace,
        usage=dict(measured_usage),
        runtime_instance_id=content_hash(
            {
                "provider": "toolsandbox",
                "checkpoint_id": replay_record["checkpoint_id"],
                "role": role,
                "seed": seed,
                "termination_reason": trace["termination_reason"],
                "work_dir": str(work_dir),
            }
        ),
    )


class ToolSandboxProvider:
    """Capture, restore, execute, and officially score ToolSandbox scenarios."""

    benchmark_id = "toolsandbox.common_support"
    source_revision = "165848b9a78cead7ca7fe7c89c688b58e6501219"
    scorer_id = "toolsandbox.official_similarity_v1"
    scorer_revision = "a81814306ce7aa229079fc530f6d026b423542cecc342a7677ac8dc3dd167481"

    _CONFIG_FIELDS = frozenset(
        {
            "scenario_names",
            "limit",
            "preferred_tool_backend",
            "agent_model",
            "agent_base_url",
            "agent_key_env",
            "user_model",
            "user_base_url",
            "user_key_env",
            "locked_dependency_hashes",
        }
    )
    _OPTIONAL_CONFIG_FIELDS = frozenset({"budget_override", "agent_timeout_seconds", "environment_timeout_seconds", "agent_extra_body", "environment_extra_body", "agent_extra_body_mode", "environment_extra_body_mode"})

    def __init__(self, *, history_transform: Any = None) -> None:
        if history_transform is not None and not all(
            callable(getattr(history_transform, name, None)) for name in ("transform", "fingerprint")
        ):
            raise OfficialArenaError("ToolSandbox history transform requires transform/fingerprint")
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
                raise OfficialArenaError("ToolSandbox checkout revision differs")
            from tool_sandbox.common.scenario import Scenario
            from tool_sandbox.roles.execution_environment import ExecutionEnvironment
            from tool_sandbox.scenarios import named_scenarios

            del ExecutionEnvironment, Scenario, named_scenarios
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
        scenarios = self._scenarios(parsed)
        rows: list[Mapping[str, Any]] = []
        for scenario_name, scenario in scenarios.items():
            serialized_context = scenario.starting_context.to_dict(serialize_console=False)
            sandbox_rows = _sandbox_rows(serialized_context)
            query = _opening_user_query(sandbox_rows)
            instruction = _agent_system_instruction(sandbox_rows)
            available_tools = _agent_available_tools(scenario.starting_context)
            tools = list(available_tools.keys())
            from tool_sandbox.common.tool_conversion import convert_to_openai_tools

            tool_schemas = convert_to_openai_tools(dict(available_tools))
            if not tool_schemas or any(not isinstance(item, Mapping) for item in tool_schemas):
                raise OfficialArenaError("ToolSandbox tool schemas are unavailable")
            visible_state = {
                "query": query,
                "task_id": scenario_name,
                "instruction": instruction,
                "context": [
                    {
                        "text": json.dumps(
                            {
                                "available_tools": tools,
                                "tool_schemas": tool_schemas,
                                "categories": [str(item) for item in scenario.categories],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "block_id": "official_tool_catalog",
                        "source": "toolsandbox.starting_context",
                        "score": None,
                        "metadata": {"scenario_name": scenario_name},
                    }
                ],
                "messages": [{"role": "user", "content": query, "name": None}],
                "tool_names": tools,
                "metadata": {
                    "benchmark_id": self.benchmark_id,
                    "checkpoint_phase": "turn_zero_before_agent_response",
                    "clarification_roundtrip_available": True,
                    "state_restore_available": True,
                    "scenario_categories": [str(item) for item in scenario.categories],
                },
                "budget": visible_shared_budget(
                    parsed,
                    {
                        "max_model_calls": scenario.max_messages,
                        "max_retrieval_calls": scenario.max_messages,
                        "max_tool_calls": scenario.max_messages,
                        "max_output_tokens": 65536,
                        "max_context_tokens": None,
                        "max_cost_usd": None,
                    },
                ),
            }
            rows.append(
                {
                    "checkpoint_id": scenario_name,
                    "group_key": scenario_name,
                    "visible_state": visible_state,
                    "eligible_families": [
                        "reformulate",
                        "decompose",
                        "acquire",
                        "clarify",
                        "curate",
                    ],
                    "protected": {
                        "scenario_name": scenario_name,
                        "starting_context": serialized_context,
                        "starting_context_hash": content_hash(serialized_context),
                        "locked_dependency_hashes": parsed["locked_dependency_hashes"],
                    },
                }
            )
        if not rows:
            raise OfficialArenaError("ToolSandbox capture found no scenarios")
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
        protected = replay_record["protected"]
        state = materialized_state(arm_payload)
        budget = require_materialized_budget(state, parsed)
        optimizer_usage = intervention_usage(arm_payload)
        budget_override = parsed.get("budget_override")
        scenario_name = protected["scenario_name"]
        scenarios = self._scenarios(
            {
                **parsed,
                "scenario_names": [scenario_name],
                "limit": 1,
            }
        )
        try:
            scenario = scenarios[scenario_name]
        except KeyError as exc:
            raise OfficialArenaError(
                "ToolSandbox protected scenario identity is unavailable"
            ) from exc

        from tool_sandbox.common.execution_context import ExecutionContext, RoleType
        from tool_sandbox.roles.execution_environment import ExecutionEnvironment

        serialized_context = protected["starting_context"]
        if content_hash(serialized_context) != protected["starting_context_hash"]:
            raise OfficialArenaError("ToolSandbox protected starting state changed")
        scenario.starting_context = ExecutionContext.from_dict(dict(serialized_context))
        agent_budget, simulator_budget = _role_budgets(
            scenario_max_messages=scenario.max_messages,
            budget=budget,
            optimizer_usage=optimizer_usage,
            budget_override=budget_override,
        )
        agent = _intervened_agent(state, parsed, seed=seed, budget=agent_budget,
                                 history_transform=self.history_transform)
        user = _tracked_user(parsed, seed=seed + 1_000_000, budget=simulator_budget)
        try:
            result = scenario.play_and_evaluate(
                roles={
                    RoleType.USER: user,
                    RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
                    RoleType.AGENT: agent,
                },
                output_directory=work_dir,
                scenario_name=scenario_name,
            )
        except _EvaluatedAgentOutputInvalid as failure:
            return _model_output_invalid_result(
                replay_record=replay_record,
                role=role,
                seed=seed,
                work_dir=work_dir,
                error=failure.error,
                measured_usage=agent.usage,
                compatibility_repairs=[
                    *agent.compatibility_repairs,
                    *user.compatibility_repairs,
                ],
                budget_enforcements=agent.budget_enforcements,
                user_simulator_usage=user.usage,
            )
        evaluation = result.evaluation_result
        ending = result.ending_context.to_dict(serialize_console=False)
        sandbox_rows = _sandbox_rows(ending)
        tool_calls = sum(
            1
            for row in sandbox_rows
            if _role_name(row.get("sender")) == "AGENT"
            and _role_name(row.get("recipient")) == "EXECUTION_ENVIRONMENT"
        )
        # The simulated user is benchmark infrastructure, not the evaluated
        # agent.  Its calls must neither consume the intervention/agent budget
        # nor enter paper usage and cost accounting.  Keep its usage in the
        # protected provider trace so the execution remains auditable.
        usage = dict(agent.usage)
        usage["tool_calls"] = tool_calls
        trace = {
            "provider": "toolsandbox",
            "live_history_transform": (
                to_jsonable(self.history_transform.fingerprint()) if self.history_transform is not None else None
            ),
            "live_history_records": (
                to_jsonable(list(getattr(self.history_transform, "records", ()))) if self.history_transform is not None else []
            ),
            "compatibility_repairs": [
                *agent.compatibility_repairs,
                *user.compatibility_repairs,
            ],
            "source_revision": self.source_revision,
            "scenario_name": scenario_name,
            "role": role,
            "seed": seed,
            "generation_controls": {
                "temperature": 0.0,
                "seed_policy": "role_base_seed_plus_call_index",
                "agent_base_seed": seed,
                "user_base_seed": seed + 1_000_000,
                "enable_thinking": False,
                "max_agent_model_calls": agent_budget["max_model_calls"],
                "max_agent_tool_calls": agent_budget["max_tool_calls"],
                "user_simulator_budget_separate": True,
                "user_simulator_model_calls": simulator_budget["model_calls"],
                "user_simulator_output_tokens": simulator_budget["output_tokens"],
                "user_simulator_hard_stops": simulator_budget["hard_stops"],
                "hard_stops": agent_budget["hard_stops"],
            },
            "budget_enforcements": list(agent.budget_enforcements),
            "user_simulator_usage": dict(user.usage),
            "ending_context": ending,
            "evaluation": {
                "similarity": evaluation.similarity,
                "milestone_similarity": evaluation.milestone_similarity,
                "minefield_similarity": evaluation.minefield_similarity,
                "turn_count": evaluation.turn_count,
                "milestone_mapping": dict(evaluation.milestone_mapping),
                "minefield_mapping": dict(evaluation.minefield_mapping),
            },
        }
        work_dir.joinpath("official_result.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ProviderArmResult(
            metrics={
                "official_similarity": float(evaluation.similarity),
                "milestone_similarity": float(evaluation.milestone_similarity),
                "minefield_similarity": float(evaluation.minefield_similarity),
                "turn_count": float(evaluation.turn_count),
                "adapter_compatibility_repairs": float(
                    len(agent.compatibility_repairs) + len(user.compatibility_repairs)
                ),
                "adapter_hard_stops": float(agent_budget["hard_stops"]),
            },
            trace=trace,
            usage=usage,
            runtime_instance_id=content_hash(
                {
                    "provider": "toolsandbox",
                    "checkpoint_id": replay_record["checkpoint_id"],
                    "role": role,
                    "seed": seed,
                    "work_dir": str(work_dir),
                }
            ),
        )

    def _scenarios(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        from tool_sandbox.common.tool_discovery import ToolBackend
        from tool_sandbox.scenarios import named_scenarios

        try:
            backend = ToolBackend(config["preferred_tool_backend"])
        except ValueError as exc:
            try:
                backend = ToolBackend[config["preferred_tool_backend"]]
            except KeyError:
                raise OfficialArenaError(
                    "ToolSandbox preferred_tool_backend is unsupported"
                ) from exc
        all_scenarios = _deterministic_scenarios(
            lambda: named_scenarios(preferred_tool_backend=backend),
            seed_material={
                "source_revision": self.source_revision,
                "preferred_tool_backend": config["preferred_tool_backend"],
            },
        )
        requested = config["scenario_names"]
        names = list(all_scenarios) if requested is None else list(requested)
        missing = set(names).difference(all_scenarios)
        if missing:
            raise OfficialArenaError(
                "ToolSandbox scenarios are unavailable: " + ", ".join(sorted(missing))
            )
        return {name: all_scenarios[name] for name in names[: config["limit"]]}

    def _validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        config = resolve_shared_budget_config(config)
        if not isinstance(config, Mapping) or not (
            set(config) >= self._CONFIG_FIELDS
            and set(config) <= self._CONFIG_FIELDS | self._OPTIONAL_CONFIG_FIELDS
        ):
            raise OfficialArenaError("ToolSandbox provider config fields are invalid")
        native_timeout_options(config, "agent_timeout_seconds")
        native_timeout_options(config, "environment_timeout_seconds")
        native_extra_body(config, "agent_extra_body")
        native_extra_body(config, "environment_extra_body")
        names = config["scenario_names"]
        if names is not None and (
            isinstance(names, (str, bytes))
            or not isinstance(names, list)
            or not names
            or any(not isinstance(item, str) or not item for item in names)
        ):
            raise OfficialArenaError("ToolSandbox scenario_names is invalid")
        limit = config["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise OfficialArenaError("ToolSandbox limit must be positive")
        for name in (
            "preferred_tool_backend",
            "agent_model",
            "agent_base_url",
            "agent_key_env",
            "user_model",
            "user_base_url",
            "user_key_env",
        ):
            value = config[name]
            if not isinstance(value, str) or not value.strip():
                raise OfficialArenaError(f"ToolSandbox {name} is invalid")
        for name in ("agent_key_env", "user_key_env"):
            environment_name = config[name]
            if not environment_name.replace("_", "").isalnum():
                raise OfficialArenaError(f"ToolSandbox {name} is invalid")
            if not os.environ.get(environment_name):
                raise OfficialArenaError(f"ToolSandbox environment {environment_name} is missing")
        validate_dependency_hashes(
            config["locked_dependency_hashes"],
            label="ToolSandbox",
        )
        return config


def _role_budgets(
    *,
    scenario_max_messages: int,
    budget: Mapping[str, Any],
    optimizer_usage: Mapping[str, Any],
    budget_override: Mapping[str, Any] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Separate the evaluated agent budget from benchmark-simulator usage."""

    per_call = int(
        budget_override["max_output_tokens_per_call"]
        if isinstance(budget_override, Mapping)
        else 512
    )
    agent = {
        "max_model_calls": max(
            0,
            min(
                scenario_max_messages,
                int(budget["max_model_calls"]) - int(optimizer_usage.get("model_calls", 0)),
            ),
        ),
        "model_calls": 0,
        "max_tool_calls": max(
            0,
            int(budget["max_tool_calls"]) - int(optimizer_usage.get("tool_calls", 0)),
        ),
        "tool_calls": 0,
        "max_output_tokens": max(
            0,
            int(budget["max_output_tokens"]) - int(optimizer_usage.get("output_tokens", 0)),
        ),
        "output_tokens": 0,
        "max_output_tokens_per_call": per_call,
        "hard_stops": 0,
    }
    simulator = {
        "max_model_calls": scenario_max_messages,
        "model_calls": 0,
        "max_output_tokens": scenario_max_messages * per_call,
        "output_tokens": 0,
        "max_output_tokens_per_call": per_call,
        "hard_stops": 0,
    }
    return agent, simulator


def _sandbox_rows(serialized_context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    databases = serialized_context.get("_dbs")
    if not isinstance(databases, Mapping):
        raise OfficialArenaError("ToolSandbox serialized context has no databases")
    rows = databases.get("SANDBOX")
    if not isinstance(rows, list):
        raise OfficialArenaError("ToolSandbox serialized context has no SANDBOX rows")
    return [row for row in rows if isinstance(row, Mapping)]


def _role_name(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1].upper()


def _opening_user_query(rows: list[Mapping[str, Any]]) -> str:
    # ToolSandbox prepends user-simulator few-shot conversations to the live
    # scenario.  The actual turn-zero task is the final USER -> AGENT message,
    # not the first demonstration message.
    for row in reversed(rows):
        if (
            _role_name(row.get("sender")) == "USER"
            and _role_name(row.get("recipient")) == "AGENT"
            and isinstance(row.get("content"), str)
            and row["content"].strip()
        ):
            return row["content"]
    raise OfficialArenaError("ToolSandbox scenario has no visible opening user query")


def _agent_system_instruction(rows: list[Mapping[str, Any]]) -> str:
    values = [
        row["content"]
        for row in rows
        if _role_name(row.get("sender")) == "SYSTEM"
        and _role_name(row.get("recipient")) == "AGENT"
        and isinstance(row.get("content"), str)
        and row["content"].strip()
    ]
    return "\n\n".join(values)


def _intervened_agent(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    budget: dict[str, int],
    history_transform: Any = None,
) -> Any:
    from openai import OpenAI
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.common.message_conversion import Message
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

    class IntervenedAgent(OpenAIAPIAgent):
        model_name = config["agent_model"]

        def __init__(self) -> None:
            # The pinned upstream constructor only creates an unused public
            # OpenAI client. Construct the bound client directly so local-only
            # execution never requires an unrelated OPENAI_API_KEY.
            self.openai_client = OpenAI(
                api_key=os.environ[config["agent_key_env"]],
                base_url=config["agent_base_url"],
                max_retries=0,
                **native_timeout_options(config, "agent_timeout_seconds"),
            )
            self.usage = _empty_usage()
            self._first_response = True
            self._first_inference = True
            self._last_response: Any = None
            self._last_advertised_tool_names: set[str] = set()
            self.compatibility_repairs: list[dict[str, Any]] = []
            self.budget_enforcements: list[dict[str, int]] = []
            self._base_seed = seed
            self._inference_calls = 0

        def respond(self, ending_index: int | None = None) -> None:
            if (
                budget["model_calls"] >= budget["max_model_calls"]
                or budget["tool_calls"] >= budget["max_tool_calls"]
                or budget["output_tokens"] >= budget["max_output_tokens"]
            ):
                if not budget["hard_stops"]:
                    budget["hard_stops"] = 1
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.AGENT,
                            recipient=RoleType.USER,
                            content="I could not complete the task within the run budget.",
                            conversation_active=False,
                        )
                    ]
                )
                return
            metadata = state.get("metadata", {})
            clarification = (
                metadata.get("clarification_request") if isinstance(metadata, Mapping) else None
            )
            if self._first_response and isinstance(clarification, str) and clarification.strip():
                self._first_response = False
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.AGENT,
                            recipient=RoleType.USER,
                            content=clarification,
                        )
                    ]
                )
                return
            self._first_response = False
            self._last_response = None
            self._last_advertised_tool_names = set()
            try:
                super().respond(ending_index=ending_index)
            except AssertionError as error:
                if not _is_empty_length_response(self._last_response):
                    if _is_generated_agent_content_invalid(self._last_response):
                        raise _EvaluatedAgentOutputInvalid(error) from error
                    raise
                budget["hard_stops"] = 1
                replacement = "I could not complete the task within the run budget."
                self.compatibility_repairs.append(
                    {
                        "kind": "empty_length_response_to_agent_budget_hard_stop",
                        "replacement": replacement,
                    }
                )
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.AGENT,
                            recipient=RoleType.USER,
                            content=replacement,
                            conversation_active=False,
                        )
                    ]
                )
            except KeyError as error:
                repair = _repair_agent_end_conversation(
                    self._last_response,
                    advertised_tool_names=self._last_advertised_tool_names,
                )
                if repair is None:
                    if _has_unadvertised_agent_tool_call(
                        self._last_response,
                        advertised_tool_names=self._last_advertised_tool_names,
                    ):
                        raise _EvaluatedAgentOutputInvalid(error) from error
                    raise
                self.compatibility_repairs.append(
                    {
                        "kind": "unadvertised_agent_end_conversation_to_user_message",
                        "replacement": repair,
                    }
                )
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.AGENT,
                            recipient=RoleType.USER,
                            content=repair,
                        )
                    ]
                )
            except json.JSONDecodeError as error:
                raise _EvaluatedAgentOutputInvalid(error) from error
            except Exception as error:
                if not _is_gpt_oss_harmony_parse_failure(error, model_name=self.model_name):
                    raise
                # The local vLLM server completed generation but its Harmony
                # response parser rejected malformed channel/header tokens.
                # This is evaluated model output, so preserve the cell as the
                # same official zero-action outcome used for other recognized
                # malformed generations instead of losing the observation.
                budget["model_calls"] += 1
                self.usage["model_calls"] += 1
                self.usage["unpriced_model_calls"] += 1
                self.compatibility_repairs.append(
                    {
                        "kind": "gpt_oss_harmony_parse_failure_to_agent_invalid_output",
                    }
                )
                raise _EvaluatedAgentOutputInvalid(error) from None

        def model_inference(self, openai_messages: Any, openai_tools: Any) -> Any:
            messages = [dict(item) for item in openai_messages]
            if history_transform is not None:
                transformed = history_transform.transform(messages, state=state, remaining_budget={
                    "max_model_calls": budget["max_model_calls"] - budget["model_calls"],
                    "max_output_tokens": budget["max_output_tokens"] - budget["output_tokens"],
                })
                # Native rendering inserts its own system preamble below;
                # never mutate the callback's private cross-turn history.
                messages = [dict(message) for message in transformed["messages"]]
                usage = transformed["usage"]
                for key in ("model_calls", "output_tokens"):
                    if type(usage.get(key, 0)) is not int or usage.get(key, 0) < 0:
                        raise OfficialArenaError("ToolSandbox history transform usage is invalid")
                    budget[key] += usage.get(key, 0)
                if (budget["model_calls"] >= budget["max_model_calls"] or
                        budget["output_tokens"] >= budget["max_output_tokens"]):
                    raise OfficialArenaError("history transform failed to reserve the downstream budget")
                self.usage = _merge_tracked_usage(self.usage, usage)
            additions = [state.get("instruction", "")]
            additions.extend(
                block.get("text", "")
                for block in state.get("context", [])
                if isinstance(block, Mapping)
            )
            optimized_context = "\n\n".join(
                item for item in additions if isinstance(item, str) and item
            )
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "<agentinputlab_visible_input>\n"
                        f"Optimized query: {state['query']}\n\n"
                        f"{optimized_context}\n"
                        "The end_conversation tool is reserved for the simulated user. "
                        "Never call it as the agent; when the task is complete, reply "
                        "to the user in natural language.\n"
                        "</agentinputlab_visible_input>"
                    ),
                },
            )
            metadata = state.get("metadata", {})
            clarification = (
                metadata.get("clarification_request") if isinstance(metadata, Mapping) else None
            )
            if self._first_inference and not clarification:
                for message in reversed(messages):
                    if message.get("role") == "user":
                        message["content"] = state["query"]
                        break
            self._first_inference = False
            if requires_qwen38_initial_system_message(self.model_name):
                messages = coalesce_chat_system_messages(messages)
            call_seed = self._base_seed + self._inference_calls
            self._inference_calls += 1
            output_cap = min(
                budget["max_output_tokens_per_call"],
                budget["max_output_tokens"] - budget["output_tokens"],
            )
            if output_cap < 1:
                raise OfficialArenaError("ToolSandbox shared output budget is exhausted")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=openai_tools,
                temperature=0.0,
                seed=call_seed,
                max_tokens=output_cap,
                extra_body=native_extra_body(config, "agent_extra_body",
                    defaults={"chat_template_kwargs": {"enable_thinking": False}}),
            )
            budget["model_calls"] += 1
            self._last_response = response
            self._last_advertised_tool_names = _openai_tool_names(openai_tools)
            enforcement = _enforce_agent_tool_budget(response, budget=budget)
            if enforcement is not None:
                self.budget_enforcements.append(enforcement)
            self.compatibility_repairs.extend(_normalize_tool_call_ids(response))
            _record_openai_usage(self.usage, response)
            reported = getattr(response, "usage", None)
            budget["output_tokens"] += int(getattr(reported, "completion_tokens", 0) or 0)
            return response

    return IntervenedAgent()


def _normalize_tool_call_ids(response: Any) -> list[dict[str, str]]:
    """Encode opaque call IDs for upstream's Python-variable representation.

    The pinned converter interpolates IDs into ``<id>_parameters`` and
    ``<id>_response``. OpenAI-compatible IDs may contain hyphens or punctuation;
    those are valid correlation IDs, not valid Python identifiers. Change only
    that representation before upstream stores both the call and its result,
    so subsequent tool-result references use the same alias. Keep names,
    arguments, order, call count, text, usage and all generation controls intact.
    Raw HTTP receipts remain unchanged; every alias is recorded separately.
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        return []
    message = getattr(choices[0], "message", None)
    calls = getattr(message, "tool_calls", None)
    if not isinstance(calls, (list, tuple)) or not calls:
        return []
    reserved = {
        call.id for call in calls
        if isinstance(getattr(call, "id", None), str) and call.id.isidentifier()
    }
    aliases: dict[str, str] = {}
    repairs: list[dict[str, str]] = []
    for call in calls:
        original = getattr(call, "id", None)
        # Missing/non-string IDs remain upstream validation errors. Do not
        # synthesize a missing model field or repair duplicate logical calls.
        if not isinstance(original, str) or not original or original.isidentifier():
            continue
        if original not in aliases:
            base = "ail_call_" + hashlib.sha256(original.encode()).hexdigest()[:32]
            alias = base
            suffix = 0
            while alias in reserved:
                suffix += 1
                alias = f"{base}_{suffix}"
            aliases[original] = alias
            reserved.add(alias)
            repairs.append({
                "kind": "tool_call_id_to_python_identifier",
                "original_id": original,
                "normalized_id": alias,
            })
        call.id = aliases[original]
    return repairs


def _enforce_agent_tool_budget(
    response: Any,
    *,
    budget: dict[str, int],
) -> dict[str, int] | None:
    """Execute at most the remaining agent tool calls from one model response.

    ToolSandbox permits parallel tool calls, so limiting model turns alone does
    not enforce ``max_tool_calls``. Truncation happens before the upstream role
    materializes calls into executable messages; the partial trajectory is then
    scored as a shared-budget hard stop instead of being discarded after replay.
    """

    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    message = getattr(choices[0], "message", None)
    calls = getattr(message, "tool_calls", None)
    if not isinstance(calls, (list, tuple)) or not calls:
        return None
    remaining = max(0, budget["max_tool_calls"] - budget["tool_calls"])
    kept = min(len(calls), remaining)
    budget["tool_calls"] += kept
    if kept == len(calls):
        return None
    # ``respond`` prevents inference when no tool budget remains, so a generated
    # tool response always retains at least one call here. Preserve upstream call
    # order and remove only the suffix that cannot be executed fairly.
    message.tool_calls = list(calls[:kept])
    budget["hard_stops"] = 1
    return {
        "generated_tool_calls": len(calls),
        "executed_tool_calls": kept,
        "discarded_tool_calls": len(calls) - kept,
    }


def _openai_tool_names(openai_tools: Any) -> set[str]:
    if isinstance(openai_tools, (str, bytes)) or not isinstance(openai_tools, (list, tuple)):
        return set()
    names: set[str] = set()
    for tool in openai_tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _repair_agent_end_conversation(
    response: Any,
    *,
    advertised_tool_names: set[str],
) -> str | None:
    """Turn one known model-protocol error into an auditable user-facing turn."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    message = getattr(choices[0], "message", None)
    calls = getattr(message, "tool_calls", None)
    if not isinstance(calls, (list, tuple)) or len(calls) != 1:
        return None
    function = getattr(calls[0], "function", None)
    name = getattr(function, "name", None)
    if name != "end_conversation" or name in advertised_tool_names:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return "I've completed the requested step."


def _repair_user_end_conversation(
    response: Any,
    *,
    tools_were_given: bool,
) -> str | None:
    """Recover the one valid ToolSandbox user terminator from a strict client mismatch."""
    if tools_were_given:
        return None
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    message = getattr(choices[0], "message", None)
    calls = getattr(message, "tool_calls", None)
    if not isinstance(calls, (list, tuple)) or len(calls) != 1:
        return None
    function = getattr(calls[0], "function", None)
    if getattr(function, "name", None) != "end_conversation":
        return None
    return "end_conversation()"


def _normalize_empty_user_tool_calls(response: Any, *, tools_were_given: bool) -> bool:
    """Normalize vLLM's empty list to the OpenAI ``null`` no-tool representation."""
    if tools_were_given:
        return False
    choices = getattr(response, "choices", None)
    if not choices:
        return False
    message = getattr(choices[0], "message", None)
    if getattr(message, "tool_calls", None) != []:
        return False
    message.tool_calls = None
    return True


def _tracked_user(
    config: Mapping[str, Any],
    *,
    seed: int,
    budget: dict[str, int],
) -> Any:
    from openai import OpenAI
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.common.message_conversion import Message
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

    class TrackedUser(OpenAIAPIUser):
        model_name = config["user_model"]

        def __init__(self) -> None:
            # OpenAIAPIUser's pinned constructor only creates the default
            # client; no role state is skipped by binding it directly here.
            self.openai_client = OpenAI(
                api_key=os.environ[config["user_key_env"]],
                base_url=config["user_base_url"],
                max_retries=0,
                **native_timeout_options(config, "environment_timeout_seconds"),
            )
            self.usage = _empty_usage()
            self._base_seed = seed
            self._inference_calls = 0
            self._last_response: Any = None
            self._tools_were_given = False
            self._last_advertised_tool_names: set[str] = set()
            self.compatibility_repairs: list[dict[str, Any]] = []

        def respond(self, ending_index: int | None = None) -> None:
            if budget["hard_stops"] or budget["model_calls"] >= budget["max_model_calls"]:
                if not budget["hard_stops"]:
                    budget["hard_stops"] = 1
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.USER,
                            recipient=RoleType.EXECUTION_ENVIRONMENT,
                            content="end_conversation()",
                            conversation_active=False,
                        )
                    ]
                )
                return
            self._last_response = None
            self._tools_were_given = False
            try:
                super().respond(ending_index=ending_index)
            except AssertionError:
                if _is_empty_length_response(self._last_response):
                    budget["hard_stops"] = 1
                    replacement = "end_conversation()"
                    self.compatibility_repairs.append(
                        {
                            "kind": "empty_length_response_to_user_budget_hard_stop",
                            "replacement": replacement,
                        }
                    )
                    self.add_messages(
                        [
                            Message(
                                sender=RoleType.USER,
                                recipient=RoleType.EXECUTION_ENVIRONMENT,
                                content=replacement,
                                conversation_active=False,
                            )
                        ]
                    )
                    return
                if _is_generated_user_content_invalid(self._last_response):
                    # The upstream ToolSandbox user role asserts that a
                    # no-tool response must contain visible text.  vLLM may
                    # legally materialize an empty assistant message instead
                    # (most often after reasoning consumed the generation
                    # allowance).  This is simulator infrastructure, not the
                    # evaluated agent, so end the simulated conversation and
                    # preserve the official scorer rather than discarding the
                    # experiment cell as a fictitious tool-schema failure.
                    budget["hard_stops"] = 1
                    replacement = "end_conversation()"
                    self.compatibility_repairs.append(
                        {
                            "kind": "empty_user_response_to_simulator_hard_stop",
                            "replacement": replacement,
                        }
                    )
                    self.add_messages(
                        [
                            Message(
                                sender=RoleType.USER,
                                recipient=RoleType.EXECUTION_ENVIRONMENT,
                                content=replacement,
                                conversation_active=False,
                            )
                        ]
                    )
                    return
                repair = _repair_user_end_conversation(
                    self._last_response,
                    tools_were_given=self._tools_were_given,
                )
                if repair is None:
                    raise OfficialArenaError(
                        "ToolSandbox user emitted unsupported tools without an "
                        f"advertised schema: {_response_tool_calls(self._last_response)!r}"
                    ) from None
                self.compatibility_repairs.append(
                    {
                        "kind": "unadvertised_user_end_conversation_to_official_tool",
                        "replacement": repair,
                    }
                )
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.USER,
                            recipient=RoleType.EXECUTION_ENVIRONMENT,
                            content=repair,
                        )
                    ]
                )
            except KeyError:
                if not _has_unadvertised_agent_tool_call(
                    self._last_response,
                    advertised_tool_names=self._last_advertised_tool_names,
                ):
                    raise
                # The user role is benchmark infrastructure. Never execute a
                # hallucinated, unadvertised simulator tool; terminate that
                # simulator trajectory explicitly so the evaluated agent is
                # still scored by the official scenario.
                budget["hard_stops"] = 1
                replacement = "end_conversation()"
                self.compatibility_repairs.append(
                    {
                        "kind": "unadvertised_user_tool_to_simulator_hard_stop",
                        "replacement": replacement,
                    }
                )
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.USER,
                            recipient=RoleType.EXECUTION_ENVIRONMENT,
                            content=replacement,
                            conversation_active=False,
                        )
                    ]
                )
            except Exception as error:
                if not _is_gpt_oss_harmony_parse_failure(error, model_name=self.model_name):
                    raise
                # This failure belongs to the simulated-user infrastructure,
                # not the evaluated agent. End the user trajectory explicitly
                # and retain the official scenario score.
                budget["model_calls"] += 1
                self.usage["model_calls"] += 1
                self.usage["unpriced_model_calls"] += 1
                budget["hard_stops"] = 1
                replacement = "end_conversation()"
                self.compatibility_repairs.append(
                    {
                        "kind": "gpt_oss_harmony_parse_failure_to_simulator_hard_stop",
                        "replacement": replacement,
                    }
                )
                self.add_messages(
                    [
                        Message(
                            sender=RoleType.USER,
                            recipient=RoleType.EXECUTION_ENVIRONMENT,
                            content=replacement,
                            conversation_active=False,
                        )
                    ]
                )

        def model_inference(self, openai_messages: Any, openai_tools: Any) -> Any:
            if (
                budget["model_calls"] >= budget["max_model_calls"]
                or budget["output_tokens"] >= budget["max_output_tokens"]
            ):
                raise OfficialArenaError("ToolSandbox shared user-simulator budget is exhausted")
            call_seed = self._base_seed + self._inference_calls
            self._inference_calls += 1
            tools_were_given = isinstance(openai_tools, (list, tuple))
            output_cap = min(
                budget["max_output_tokens_per_call"],
                budget["max_output_tokens"] - budget["output_tokens"],
            )
            messages = [dict(item) for item in openai_messages]
            if requires_qwen38_initial_system_message(self.model_name):
                messages = coalesce_chat_system_messages(messages)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto" if tools_were_given else "none",
                temperature=0.0,
                seed=call_seed,
                max_tokens=output_cap,
                extra_body=native_extra_body(config, "environment_extra_body",
                    defaults={"chat_template_kwargs": {"enable_thinking": False}}),
            )
            budget["model_calls"] += 1
            self._last_response = response
            self._tools_were_given = tools_were_given
            self._last_advertised_tool_names = _openai_tool_names(openai_tools)
            if _normalize_empty_user_tool_calls(
                response,
                tools_were_given=tools_were_given,
            ):
                self.compatibility_repairs.append(
                    {
                        "kind": "empty_user_tool_calls_to_openai_null",
                        "replacement": None,
                    }
                )
            self.compatibility_repairs.extend(_normalize_tool_call_ids(response))
            _record_openai_usage(self.usage, response)
            reported = getattr(response, "usage", None)
            budget["output_tokens"] += int(getattr(reported, "completion_tokens", 0) or 0)
            return response

    return TrackedUser()


def _response_tool_calls(response: Any) -> list[tuple[Any, Any]]:
    choices = getattr(response, "choices", None)
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return []
    return [
        (
            getattr(getattr(call, "function", None), "name", None),
            getattr(getattr(call, "function", None), "arguments", None),
        )
        for call in calls
    ]


def _is_empty_length_response(response: Any) -> bool:
    """Recognize a turn that exhausted its fair output cap before acting.

    Reasoning-model APIs may return neither visible content nor tool calls when
    the shared completion cap is exhausted. ToolSandbox's upstream role asserts
    on that otherwise valid OpenAI response. Treating it as an explicit hard
    stop preserves the official failed trajectory and scorer instead of losing
    the experiment cell to an adapter exception.
    """

    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return False
    choice = choices[0]
    if getattr(choice, "finish_reason", None) != "length":
        return False
    message = getattr(choice, "message", None)
    if message is None:
        return False
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    return not (isinstance(content, str) and content.strip()) and not tool_calls


def _is_generated_agent_content_invalid(response: Any) -> bool:
    """Recognize an otherwise valid response with neither content nor tool calls."""

    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return False
    message = getattr(choices[0], "message", None)
    if message is None:
        return False
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    return not (isinstance(content, str) and content.strip()) and not tool_calls


def _is_generated_user_content_invalid(response: Any) -> bool:
    """Recognize the same empty generation at the simulated-user boundary."""

    return _is_generated_agent_content_invalid(response)


def _is_gpt_oss_harmony_parse_failure(error: Exception, *, model_name: str) -> bool:
    """Recognize only the local GPT-OSS Harmony parser's HTTP-500 failures."""

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


def _has_unadvertised_agent_tool_call(
    response: Any,
    *,
    advertised_tool_names: set[str],
) -> bool:
    """Recognize a generated tool name that the official turn did not expose."""

    calls = _response_tool_calls(response)
    return bool(calls) and any(
        not isinstance(name, str) or name not in advertised_tool_names for name, _ in calls
    )


def _empty_usage() -> dict[str, int | float]:
    return {
        "model_calls": 0,
        "retrieval_calls": 0,
        "tool_calls": 0,
        "embedding_calls": 0,
        "unpriced_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cpu_seconds": 0.0,
        "accelerator_seconds": 0.0,
        "cost_usd": 0.0,
        "compute_cost_usd": 0.0,
    }


def _record_openai_usage(usage: dict[str, int | float], response: Any) -> None:
    usage["model_calls"] += 1
    reported = getattr(response, "usage", None)
    if reported is None:
        usage["unpriced_model_calls"] += 1
        return
    usage["input_tokens"] += int(getattr(reported, "prompt_tokens", 0) or 0)
    usage["output_tokens"] += int(getattr(reported, "completion_tokens", 0) or 0)


def _merge_tracked_usage(
    left: Mapping[str, int | float],
    right: Mapping[str, int | float],
) -> dict[str, int | float]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in _empty_usage()}


def _agent_available_tools(context: Any) -> Mapping[str, Any]:
    """Return the exact tool names exposed to ToolSandbox's agent role."""
    return context.get_available_tools(scrambling_allowed=True)


def _deterministic_scenarios(
    factory: Any,
    *,
    seed_material: Mapping[str, Any],
) -> Any:
    """Build upstream scenarios deterministically without leaking global RNG state."""
    previous_state = random.getstate()
    try:
        seed = int(content_hash(seed_material), 16)
        random.seed(seed)
        return factory()
    finally:
        random.setstate(previous_state)


def factory() -> ToolSandboxProvider:
    return ToolSandboxProvider()
