"""ACE tool-plan feedback with native ToolSandbox Agent tools and scoring.

This is noninteractive fitting feedback, not the native closed-loop heldout
agent. Caller-defined fit groups are not an official benchmark training split.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from agentinputlab.api import _state
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable

from .ace import ACEExample
from .ace_native_plan import (
    NativePlanFeedback,
    _save,
    _sha,
    native_process_scope,
    source_file_pins,
    verify_native_checkout,
)

SCHEMA = "agentinputlab.ace.ToolSandbox.native_feedback_input.v1"
PLAN_INSTRUCTION = (
    "Solve the supplied task as one noninteractive tool plan. Preserve all visible task "
    "rules and function schemas. Put exactly one JSON object in your final_answer string: "
    '{"actions":[{"name":"visible_function_name","arguments":{}}],"response":"message to user"}. '
    "Use only the advertised Agent-role functions with their required arguments. Actions execute "
    "in order without another generation or access to their results while planning. The "
    "end_conversation tool belongs to the simulated user and must never be emitted by the agent. "
    "Do not output Python, code fences, unknown functions or extra fields.\n"
)


def _provider():
    from workers.official_arenas.providers.toolsandbox import ToolSandboxProvider

    return ToolSandboxProvider()


def render_toolsandbox_agent_visible(state, schemas):
    """Retain task fields; substitute only the native Agent-role tool catalog."""
    value = to_jsonable(state)
    names = [tool.get("function", tool)["name"] for tool in schemas]
    if not names or len(names) != len(set(names)) or "end_conversation" in names:
        raise ValidationError("ToolSandbox needs native Agent-role functions, not user tools")
    visible = {
        key: copy.deepcopy(value[key]) for key in ("query", "instruction", "messages", "context")
    }
    visible["tool_names"] = names
    for block in visible["context"]:
        if block["block_id"] == "official_tool_catalog":
            catalog = json.loads(block["text"])
            catalog["available_tools"] = names
            catalog["tool_schemas"] = copy.deepcopy(schemas)
            block["text"] = canonical_json(catalog)
    return visible


def _native_inputs(checkout, configuration, rows, provider):
    from tool_sandbox.common.execution_context import (
        ExecutionContext,
        get_current_context,
        set_current_context,
    )
    from tool_sandbox.common.tool_conversion import convert_to_openai_tools
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

    files = [
        checkout / "tool_sandbox" / name
        for name in (
            "common/execution_context.py",
            "common/evaluation.py",
            "common/scenario.py",
            "common/tool_conversion.py",
            "common/message_conversion.py",
            "roles/openai_api_agent.py",
            "roles/execution_environment.py",
            "scenarios/__init__.py",
            "scenarios/single_tool_call_scenarios.py",
            "scenarios/multiple_tool_call_scenarios.py",
            "scenarios/multiple_user_turn_scenarios.py",
            "scenarios/insufficient_information_scenarios.py",
        )
    ]
    files.append(Path(inspect.getfile(OpenAIAPIAgent.get_available_tools.__func__)))
    names = [r["case_id"] for r in rows]
    schemas = {}
    previous = get_current_context()
    try:
        scenarios = provider._scenarios(dict(configuration, scenario_names=names, limit=len(names)))
        if set(scenarios) != set(names):
            raise ValidationError("ToolSandbox fitting scenario is absent from the native runtime")
        for row in rows:
            key = row["case_id"]
            protected = row["protected_checkpoint"]["protected"]
            state = _state(row["state"])
            context = protected["starting_context"]
            if (
                key not in configuration["scenario_names"]
                or protected["scenario_name"] != key
                or state.task_id != key
                or context["interactive_console"] is not None
                or content_hash(context) != protected["starting_context_hash"]
                or protected["locked_dependency_hashes"]
                != configuration["locked_dependency_hashes"]
            ):
                raise ValidationError("ToolSandbox frozen context, task or dependencies differ")
            set_current_context(ExecutionContext.from_dict(copy.deepcopy(context)))
            tools = OpenAIAPIAgent.get_available_tools()
            if not tools or not set(tools) <= set(state.tool_names) or "end_conversation" in tools:
                raise ValidationError("ToolSandbox native Agent-role tool catalog differs")
            schemas[key] = to_jsonable(convert_to_openai_tools(tools))
            render_toolsandbox_agent_visible(state, schemas[key])
            files.extend(
                Path(inspect.getfile(inspect.unwrap(function))) for function in tools.values()
            )
    finally:
        set_current_context(previous)
    return schemas, source_file_pins(checkout, files)


def build_ace_toolsandbox_feedback_input(
    *,
    checkout,
    captures: Sequence[Mapping],
    native_configuration: Mapping,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    heldout_ids: Sequence[str],
    partition_label: str,
    target_model: Mapping,
    environment_model: Mapping,
    output_path,
):
    """Freeze existing captures with explicit caller-declared fitting groups."""
    provider = _provider()
    root = verify_native_checkout(checkout, revision=provider.source_revision)
    config = provider._validate_config(to_jsonable(native_configuration))
    ids = [*train_ids, *validation_ids]
    if (
        not train_ids
        or not validation_ids
        or len(ids) != len(set(ids))
        or set(ids) & set(heldout_ids)
        or not isinstance(partition_label, str)
        or not partition_label.strip()
        or any(not isinstance(k, str) or not k for k in ids)
    ):
        raise ValidationError(
            "ToolSandbox needs explicit disjoint fitting groups and a partition label"
        )
    selected = {r["checkpoint_id"]: r for r in captures if r["checkpoint_id"] in ids}
    if len(selected) != len(ids) or len([r for r in captures if r["checkpoint_id"] in ids]) != len(
        ids
    ):
        raise ValidationError("ToolSandbox capture IDs are missing or duplicated")
    rows = []
    for role, group in (("train", train_ids), ("validation", validation_ids)):
        for key in group:
            capture = to_jsonable(selected[key])
            state = _state(capture["visible_state"])
            rows.append(
                dict(
                    case_id=key,
                    role=role,
                    state=to_jsonable(state),
                    state_hash=state.state_hash,
                    protected_checkpoint=capture,
                    caller_declared_fit_partition_NOT_official_split=True,
                )
            )
    with native_process_scope(root, package="tool_sandbox"):
        schemas, pins = _native_inputs(root, config, rows, provider)
    if (
        not isinstance(target_model, Mapping)
        or not isinstance(environment_model, Mapping)
        or target_model.get("model_id") != config["agent_model"]
        or environment_model.get("model_id") != config["user_model"]
    ):
        raise ValidationError("ToolSandbox model fingerprints differ from the native configuration")
    value = dict(
        schema=SCHEMA,
        benchmark=provider.benchmark_id,
        native_source_revision=provider.source_revision,
        source_files=pins,
        native_configuration=config,
        fitting_rows=rows,
        heldout_ids=list(heldout_ids),
        source_official_split=None,
        partition_source=partition_label,
        target_model=to_jsonable(target_model),
        environment_model=to_jsonable(environment_model),
        Agent_role_schemas=schemas,
        builder_model_calls=0,
        ToolTalk_not_used=True,
    )
    path = Path(output_path).expanduser().absolute()
    if path.resolve().is_relative_to(root) or path.exists() or path.is_symlink():
        raise ValidationError("ToolSandbox input must be a new file outside the clean checkout")
    _save(path, value)
    return dict(path=str(path), sha256=_sha(path))


class ACEToolSandboxFeedback(NativePlanFeedback):
    processor_id = "agentinputlab.ace.ToolSandbox.native_Agent_role_plan_feedback.v1"
    metric = "official_similarity"

    def __init__(self, *, checkout, source_input, output_root, mode="offline"):
        self._initialize(
            checkout=checkout, source_input=source_input, output_root=output_root, mode=mode
        )
        self.provider = _provider()
        if (
            self.source.get("schema") != SCHEMA
            or self.source.get("benchmark") != self.provider.benchmark_id
        ):
            raise ValidationError("ToolSandbox ACE feedback schema or benchmark differs")
        self.checkout = verify_native_checkout(checkout, revision=self.provider.source_revision)
        self._verify_source()
        self.configuration = self.provider._validate_config(self.source["native_configuration"])
        if (
            self.source["native_source_revision"] != self.provider.source_revision
            or self.source["source_official_split"] is not None
            or not isinstance(self.source.get("partition_source"), str)
            or not self.source["partition_source"].strip()
        ):
            raise ValidationError(
                "ToolSandbox fitting groups are caller-defined, not an official train split"
            )
        with native_process_scope(self.checkout, package="tool_sandbox"):
            self.schemas, pins = _native_inputs(
                self.checkout, self.configuration, list(self.rows.values()), self.provider
            )
        if pins != self.source["source_files"] or self.schemas != self.source["Agent_role_schemas"]:
            raise ValidationError("ToolSandbox native source or Agent-role schemas changed")
        if (
            self.source["target_model"]["model_id"] != self.configuration["agent_model"]
            or self.source["environment_model"]["model_id"] != self.configuration["user_model"]
        ):
            raise ValidationError("ToolSandbox deployment model identity differs")
        self.tool_names = {
            k: {t.get("function", t)["name"] for t in v} for k, v in self.schemas.items()
        }
        self._create_output()

    def fingerprint(self):
        self._verify_source()
        return dict(
            processor_id=self.processor_id,
            mode=self.mode,
            source_input=self.source_input,
            source_files=self.source["source_files"],
            source_revision=self.provider.source_revision,
            scorer_id=self.provider.scorer_id,
            scorer_revision=self.provider.scorer_revision,
            target_model=self.source["target_model"],
            environment_model=self.source["environment_model"],
            provider_source_sha256=_sha(inspect.getfile(type(self.provider))),
            task_adapter_source_sha256=_sha(__file__),
            feedback_metric=self.metric,
            source_official_split=None,
            partition_source=self.source["partition_source"],
            groups={
                role: [
                    dict(
                        case_id=k,
                        state_hash=r["state_hash"],
                        protected_starting_context_hash=r["protected_checkpoint"]["protected"][
                            "starting_context_hash"
                        ],
                    )
                    for k, r in self.rows.items()
                    if r["role"] == role
                ]
                for role in ("train", "validation")
            },
            native_Agent_role_tools={k: sorted(v) for k, v in self.tool_names.items()},
            visible_role_schemas_sha256={k: content_hash(v) for k, v in self.schemas.items()},
            role_tool_renderer="native_Agent_role_tool_plan_v2",
            user_simulator_tools_NOT_advertised_to_plan_generator=True,
            noninteractive_text_plan_NOT_native_closed_loop_agent=True,
            heldout_feedback_NOT_used=True,
            transductive_NOT_independent_heldout=self.mode == "online",
            native_user_simulator_calls=0,
            native_judge_model_calls=0,
            automatic_retry=False,
        )

    def _example(self, key, split):
        state = self.states[key]
        visible = render_toolsandbox_agent_visible(state, self.schemas[key])
        return ACEExample(
            example_id=key,
            input=dict(question=PLAN_INSTRUCTION + canonical_json(visible)),
            expected_output=key,
            split=split,
            metadata=dict(
                source_state_hash=state.state_hash,
                evaluator_target_is_opaque_case_id=True,
                task_processor=self.processor_id,
                native_Agent_role_tool_catalog=True,
            ),
        )

    def _native_score(self, key, plan, folder):
        with native_process_scope(self.checkout, package="tool_sandbox"):
            from openai.types.chat import ChatCompletion
            from tool_sandbox.common.execution_context import (
                ExecutionContext,
                RoleType,
                get_current_context,
                set_current_context,
            )
            from tool_sandbox.common.message_conversion import Message
            from tool_sandbox.roles.execution_environment import ExecutionEnvironment
            from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

            model = self.source["target_model"]["model_id"]

            class FrozenPlanAgent(OpenAIAPIAgent):
                model_name = model

                def __init__(self):
                    self.next_action = 0

                def respond(self, ending_index=None):
                    if self.next_action < len(plan["actions"]):
                        return super().respond(ending_index=ending_index)
                    self.add_messages(
                        [
                            Message(
                                sender=RoleType.AGENT,
                                recipient=RoleType.USER,
                                content=plan["response"],
                                conversation_active=False,
                            )
                        ]
                    )

                def model_inference(self, openai_messages, openai_tools):
                    action = plan["actions"][self.next_action]
                    self.next_action += 1
                    return ChatCompletion.model_validate(
                        dict(
                            id="frozen-plan-" + str(self.next_action),
                            created=0,
                            object="chat.completion",
                            model=self.model_name,
                            choices=[
                                dict(
                                    index=0,
                                    finish_reason="tool_calls",
                                    message=dict(
                                        role="assistant",
                                        content=None,
                                        tool_calls=[
                                            dict(
                                                id="frozen_plan_" + str(self.next_action),
                                                type="function",
                                                function=dict(
                                                    name=action["name"],
                                                    arguments=canonical_json(action["arguments"]),
                                                ),
                                            )
                                        ],
                                    ),
                                )
                            ],
                        )
                    )

            class NoSimulatedUser:
                def respond(self, *args, **kwargs):
                    raise ValidationError(
                        "Noninteractive ACE tool-plan feedback cannot invoke a user simulator"
                    )

            protected = self.rows[key]["protected_checkpoint"]["protected"]
            previous = get_current_context()
            try:
                scenario = self.provider._scenarios(
                    dict(self.configuration, scenario_names=[key], limit=1)
                )[key]
                scenario.starting_context = ExecutionContext.from_dict(
                    copy.deepcopy(protected["starting_context"])
                )
                result = scenario.play_and_evaluate(
                    roles={
                        RoleType.AGENT: FrozenPlanAgent(),
                        RoleType.USER: NoSimulatedUser(),
                        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
                    },
                    output_directory=folder,
                    scenario_name=key,
                )
                ending = result.ending_context.to_dict(serialize_console=False)
                evaluation = result.evaluation_result
            finally:
                set_current_context(previous)
            value = dict(
                status="native_ToolSandbox_tool_plan_scored",
                case_id=key,
                source_state_hash=self.states[key].state_hash,
                protected_starting_context_hash=protected["starting_context_hash"],
                plan=plan,
                native_metrics=dict(
                    official_similarity=float(evaluation.similarity),
                    milestone_similarity=float(evaluation.milestone_similarity),
                    minefield_similarity=float(evaluation.minefield_similarity),
                    turn_count=float(evaluation.turn_count),
                ),
                native_ending_context=ending,
                evaluated_agent_model_calls=0,
                native_user_simulator_calls=0,
                new_environment_model_calls=0,
                ordered_action_count=len(plan["actions"]),
                native_closed_loop_primary_parity_NOT_claimed=True,
            )
            _save(folder / "native-score.json", value)
            return value


def create_toolsandbox_feedback(**configuration):
    """Trusted startup factory accepting the constructor's data-only fields."""
    return ACEToolSandboxFeedback(**configuration)
