"""Source-locked LLMCompiler planner for the three common-support arenas.

This profile preserves the pinned planner prompt, plan grammar, dependency
references, and DAG materialization.  It deliberately stops before scheduling:
the official arena remains the sole owner of tool execution and final answers.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend
from agentinputlab.connectors.tools import ToolConnector, ToolSpec
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
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
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .bridge import LLMCOMPILER_UPSTREAM_COMMIT, CompilerTask, _validate_dag

LLMCOMPILER_COMMON_SUPPORT_RELEASE = "plan.llm_compiler.common_support_planner"
LLMCOMPILER_PLANNER_SOURCE_SHA256 = (
    "d597890d8a6aee9fc7576400bd443eb2090201632ac2f64dcef349057aed9e05"
)
LLMCOMPILER_PARSER_SOURCE_SHA256 = (
    "fb187ec2ce4aa3f8cd1b83c0576da4de382fd5154935497f401830085f50f275"
)
LLMCOMPILER_END_OF_PLAN = "<END_OF_PLAN>"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOOL = re.compile(r"^\w+$")
_ACTION_PATTERN = re.compile(r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?")
_ID_PATTERN = re.compile(r"\$\{?(\d+)\}?")
_JOIN_DESCRIPTION = (
    "join():\n"
    " - Collects and combines results from prior actions.\n"
    " - A LLM agent is called upon invoking join to either finalize the user query or wait until the plans are executed.\n"
    " - join should always be the last action in the plan, and will be called in two scenarios:\n"
    "   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.\n"
    "   (b) if the answer cannot be determined in the planning phase before you execute the plans. "
)
_CHAT_COMPATIBILITY_SUFFIX = (
    "\nCompatibility requirement for this chat backend:\n"
    "Return only the source planner grammar as plain text, never JSON, Markdown, "
    "code fences, or prose. Use one action per line in exactly this form: "
    "`1. tool_name(positional_python_literals)`. Arguments must follow the declared "
    "schema order. Inside each call pass only positional values: never repeat argument "
    "names and never use `name: value` or `name=value`. Number every line with "
    "consecutive decimal integers; end with "
    "the actual next integer and `join()` (for example, `2. join()`), followed by "
    "`<END_OF_PLAN>`.\n"
)


@dataclass(frozen=True, slots=True)
class LLMCompilerCommonSupportConfig:
    """Bounded controls for the planner-only comparison profile."""

    max_tasks: int = 32
    max_tokens: int = 512
    temperature: float = 0.0
    allow_connector_only_tool_inventory: bool = False

    def __post_init__(self) -> None:
        for label, value in (("max_tasks", self.max_tasks), ("max_tokens", self.max_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"LLMCompiler {label} must be a positive integer")
        if self.max_tasks > 128:
            raise ValidationError("LLMCompiler max_tasks must not exceed 128")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValidationError("LLMCompiler temperature must be in [0, 2]")
        if not isinstance(self.allow_connector_only_tool_inventory, bool):
            raise ValidationError("LLMCompiler allow_connector_only_tool_inventory must be boolean")


@dataclass(slots=True)
class LLMCompilerCommonSupportPlanner:
    """Generate one source-grammar task DAG over frozen official tool schemas."""

    config: LLMCompilerCommonSupportConfig = LLMCompilerCommonSupportConfig()
    method_id: str = "llm_compiler"
    release_id: str = LLMCOMPILER_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.DECOMPOSE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("LLMCompiler common support requires a non-empty query")
        backend = context.backend
        tools = context.tools
        if not isinstance(backend, LLMBackend):
            raise ExternalDependencyError("LLMCompiler common support requires an LLMBackend")
        if not isinstance(tools, ToolConnector):
            raise ExternalDependencyError(
                "LLMCompiler common support requires a frozen ToolConnector"
            )
        tool_specs = tools.list_tools()
        _validate_tool_specs(
            tool_specs,
            state.tool_names,
            allow_connector_only=self.config.allow_connector_only_tool_inventory,
        )
        if state.budget.max_model_calls < 1:
            raise BudgetExceededError("LLMCompiler has no remaining model-call budget")
        if state.budget.max_output_tokens < 1:
            raise BudgetExceededError("LLMCompiler has no remaining output-token budget")

        source_prompt = render_llmcompiler_planner_prompt(tool_specs)
        system_prompt = source_prompt + _CHAT_COMPATIBILITY_SUFFIX
        request = GenerationRequest(
            prompt=f"Question: {state.query}",
            system=system_prompt,
            temperature=self.config.temperature,
            max_tokens=min(self.config.max_tokens, state.budget.max_output_tokens),
            seed=context.seed,
            stop=(LLMCOMPILER_END_OF_PLAN,),
            metadata={
                "method_id": self.method_id,
                "release_id": self.release_id,
                "execution_mode": "planner_component",
                "upstream_commit": LLMCOMPILER_UPSTREAM_COMMIT,
            },
        )
        response = backend.generate(request)
        if not isinstance(response.text, str) or not response.text.strip():
            raise AlgorithmOutputError("LLMCompiler planner returned empty output")
        usage = response.usage
        if usage.output_tokens > state.budget.max_output_tokens:
            raise BudgetExceededError("LLMCompiler exceeded max_output_tokens")
        if (
            state.budget.max_context_tokens is not None
            and usage.input_tokens > state.budget.max_context_tokens
        ):
            raise BudgetExceededError("LLMCompiler exceeded max_context_tokens")
        if (
            state.budget.max_cost_usd is not None
            and usage.cost_usd is not None
            and usage.cost_usd > state.budget.max_cost_usd
        ):
            raise BudgetExceededError("LLMCompiler exceeded max_cost_usd")

        tasks, join_task_id = parse_llmcompiler_plan(
            response.text,
            tool_specs,
            max_tasks=self.config.max_tasks,
        )
        task_rows = tuple(
            {
                "task_id": task.task_id,
                "tool_name": task.tool_name,
                "arguments": dict(task.arguments),
                "dependencies": task.dependencies,
            }
            for task in tasks
        )
        canonical_plan = json.dumps(
            task_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        raw_plan_hash = content_hash(response.text)
        tool_schema_hash = content_hash(tool_specs)
        block = ContextBlock(
            block_id=f"{state.task_id or state.state_hash}:llm-compiler-plan",
            text=canonical_plan,
            source="llm_compiler_task_dag",
            metadata={
                "raw_plan_hash": raw_plan_hash,
                "task_count": len(tasks),
                "join_task_id": join_task_id,
                "full_system": False,
                "tool_schema_hash": tool_schema_hash,
            },
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "source_locked_common_support_planner",
                    {
                        "upstream_commit": LLMCOMPILER_UPSTREAM_COMMIT,
                        "planner_source_sha256": LLMCOMPILER_PLANNER_SOURCE_SHA256,
                        "parser_source_sha256": LLMCOMPILER_PARSER_SOURCE_SHA256,
                        "source_prompt_sha256": content_hash(source_prompt),
                        "chat_compatibility_suffix_sha256": content_hash(
                            _CHAT_COMPATIBILITY_SUFFIX
                        ),
                        "effective_prompt_sha256": content_hash(system_prompt),
                        "tool_schema_hash": tool_schema_hash,
                        "tool_names": tuple(spec.name for spec in tool_specs),
                        "demonstration_policy": "zero_shot_cross_arena",
                        "tool_inventory_policy": (
                            "connector_only_controlled_static.v1"
                            if self.config.allow_connector_only_tool_inventory
                            else "checkpoint_exact.v1"
                        ),
                        "stop": LLMCOMPILER_END_OF_PLAN,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "task_dag",
                    {
                        "tasks": task_rows,
                        "join_task_id": join_task_id,
                        "raw_plan_hash": raw_plan_hash,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "planner_component_complete",
                    {
                        "downstream_execution_required": True,
                        "optimizer_tool_calls": 0,
                        "final_answer_injected": False,
                    },
                ),
            ),
            counters={
                "model_calls": 1,
                "tool_calls": 0,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "planned_tasks": len(tasks),
            },
            warnings=(
                "controlled_zero_shot_cross_arena_prompt_not_upstream_benchmark_prompt",
                "planner_component_not_full_parallel_execution_system",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    value=block,
                    target="execution_plan",
                    rationale="materialize the source-grammar LLMCompiler dependency DAG",
                ),
                PatchOperation(
                    PatchOp.SET_METADATA,
                    value={
                        "llm_compiler_tasks": task_rows,
                        "llm_compiler_join_task_id": join_task_id,
                        "llm_compiler_termination": "planner_complete",
                    },
                    target="metadata",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"controlled_source_locked:SqueezeAILab/LLMCompiler@{LLMCOMPILER_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "controlled_source_locked",
                "profile_attestation": "controlled_common_support_planner",
                "upstream_prompt_template_preserved": True,
                "upstream_plan_grammar_preserved": True,
                "upstream_dependency_reference_semantics_preserved": True,
                "dependency_parser_repair": "known_prior_task_ids_only.v1",
                "named_argument_parser_repair": "single_colon_or_python_keyword.v1",
                "schema_rendering_adapter": "openai_json_schema_to_source_signature.v1",
                "chat_compatibility_wrapper": True,
                "demonstration_policy": "zero_shot_cross_arena",
                "tool_inventory_policy": (
                    "connector_only_controlled_static.v1"
                    if self.config.allow_connector_only_tool_inventory
                    else "checkpoint_exact.v1"
                ),
                "full_system": False,
                "final_answer_injected": False,
                "optimizer_tool_calls": 0,
                "backend_fingerprint": dict(backend.fingerprint()),
                "tools_fingerprint": dict(tools.fingerprint()),
                "tool_schema_hash": tool_schema_hash,
                "request_hash": request.request_hash,
            },
        )


def render_llmcompiler_planner_prompt(tool_specs: Sequence[ToolSpec]) -> str:
    """Render the pinned ``generate_llm_compiler_prompt`` with no task examples."""

    specs = tuple(tool_specs)
    prefix = (
        "Given a user query, create a plan to solve it with the utmost parallelizability. "
        f"Each plan should comprise an action from the following {len(specs) + 1} types:\n"
    )
    for index, spec in enumerate(specs):
        prefix += f"{index + 1}. {_source_tool_description(spec)}\n"
    prefix += f"{len(specs) + 1}. {_JOIN_DESCRIPTION}\n\n"
    prefix += (
        "Guidelines:\n"
        " - Each action described above contains input/output types and description.\n"
        "    - You must strictly adhere to the input and output types for each action.\n"
        "    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.\n"
        " - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.\n"
        " - Each action MUST have a unique ID, which is strictly increasing.\n"
        " - Inputs for actions can either be constants or outputs from preceding actions. "
        "In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.\n"
        f" - Always call join as the last action in the plan. Say '{LLMCOMPILER_END_OF_PLAN}' after you call join\n"
        " - Ensure the plan maximizes parallelizability.\n"
        " - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.\n"
        " - Never explain the plan with comments (e.g. #).\n"
        " - Never introduce new actions other than the ones provided.\n\n"
        "Here are some examples:\n\n"
    )
    return prefix


def parse_llmcompiler_plan(
    raw: str,
    tool_specs: Sequence[ToolSpec],
    *,
    max_tasks: int,
) -> tuple[tuple[CompilerTask, ...], int]:
    """Parse the pinned action grammar and omit the non-executable join node."""

    if not isinstance(raw, str) or not raw.strip():
        raise AlgorithmOutputError("LLMCompiler plan must be non-empty text")
    specs = {spec.name: spec for spec in tool_specs}
    tasks: list[CompilerTask] = []
    join_task_id: int | None = None
    for match in _ACTION_PATTERN.finditer(raw):
        task_id = int(match.group(1))
        tool_name = match.group(2)
        raw_arguments = match.group(3)
        if tool_name == "join":
            join_task_id = task_id
            break
        spec = specs.get(tool_name)
        if spec is None:
            raise AlgorithmOutputError(f"LLMCompiler planned unknown tool {tool_name!r}")
        arguments = _normalized_arguments(raw_arguments, spec)
        dependencies = _dependency_ids(arguments, {task.task_id for task in tasks})
        tasks.append(
            CompilerTask(
                task_id=task_id,
                tool_name=tool_name,
                arguments=arguments,
                dependencies=dependencies,
            )
        )
    if join_task_id is None:
        raise AlgorithmOutputError("LLMCompiler plan must end with join")
    if not tasks:
        raise AlgorithmOutputError("LLMCompiler plan contains no executable task")
    if join_task_id <= tasks[-1].task_id:
        raise AlgorithmOutputError("LLMCompiler join ID must follow executable tasks")
    frozen_tasks = tuple(tasks)
    _validate_dag(frozen_tasks, max_tasks=max_tasks, allowed_tools=set(specs))
    return frozen_tasks, join_task_id


def _normalized_arguments(raw: str, spec: ToolSpec) -> Mapping[str, Any]:
    properties = spec.input_schema.get("properties", {})
    names = tuple(properties) if isinstance(properties, Mapping) else ()
    if raw == "":
        values: tuple[Any, ...] = ()
    else:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            keyword_values = _python_keyword_arguments(raw, names)
            if keyword_values is not None:
                return keyword_values
            if len(names) == 1:
                colon = re.fullmatch(rf"\s*{re.escape(names[0])}\s*:\s*(.+)", raw)
                if colon is not None:
                    try:
                        value = ast.literal_eval(colon.group(1))
                    except (SyntaxError, ValueError):
                        value = colon.group(1).strip()
                    return {names[0]: value}
            parsed = raw
        values = tuple(parsed) if isinstance(parsed, (list, tuple)) else (parsed,)
    if len(values) == 1 and isinstance(values[0], Mapping):
        return dict(values[0])
    if len(names) == len(values):
        return dict(zip(names, values, strict=True))
    return {"__args__": values}


def _python_keyword_arguments(raw: str, names: tuple[str, ...]) -> Mapping[str, Any] | None:
    try:
        expression = ast.parse(f"_tool({raw})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or expression.args or not expression.keywords:
        return None
    values: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg not in names or keyword.arg in values:
            return None
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (TypeError, ValueError):
            return None
    return values if set(values) == set(names) else None


def _dependency_ids(arguments: Mapping[str, Any], known_ids: set[int]) -> tuple[int, ...]:
    found: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            found.update(
                task_id
                for task_id in (int(item) for item in _ID_PATTERN.findall(value))
                if task_id in known_ids
            )
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)

    visit(arguments)
    return tuple(sorted(found))


def _source_tool_description(spec: ToolSpec) -> str:
    properties = spec.input_schema.get("properties", {})
    required = spec.input_schema.get("required", ())
    required_names = set(required) if isinstance(required, Sequence) else set()
    rendered: list[str] = []
    if isinstance(properties, Mapping):
        for name, schema in properties.items():
            schema_type = schema.get("type", "Any") if isinstance(schema, Mapping) else "Any"
            optional = "" if name in required_names else " = None"
            rendered.append(f"{name}: {schema_type}{optional}")
    signature = f"{spec.name}({', '.join(rendered)}) -> Any:"
    compact_schema = json.dumps(
        to_jsonable(spec.input_schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{signature}\n - {spec.description.strip()}\n - Input JSON schema: {compact_schema}"


def _validate_tool_specs(
    specs: Sequence[ToolSpec],
    state_names: Sequence[str],
    *,
    allow_connector_only: bool = False,
) -> None:
    values = tuple(specs)
    if not values:
        raise ExternalDependencyError("LLMCompiler requires at least one tool schema")
    if any(not isinstance(spec, ToolSpec) for spec in values):
        raise TypeError("LLMCompiler connector returned a non-ToolSpec")
    names = tuple(spec.name for spec in values)
    if len(names) != len(set(names)) or any(_SAFE_TOOL.fullmatch(name) is None for name in names):
        raise ValidationError("LLMCompiler tool names must be unique Python identifiers")
    if set(names) != set(state_names) and not (allow_connector_only and not tuple(state_names)):
        raise ValidationError(
            "LLMCompiler frozen tool schemas differ from the checkpoint tool inventory"
        )
    if any(not spec.description.strip() for spec in values):
        raise ValidationError("LLMCompiler tool descriptions must be non-empty")
    for digest in (LLMCOMPILER_PLANNER_SOURCE_SHA256, LLMCOMPILER_PARSER_SOURCE_SHA256):
        if _HEX64.fullmatch(digest) is None:  # pragma: no cover - source-lock invariant
            raise RuntimeError("LLMCompiler source SHA constant is invalid")


__all__ = [
    "LLMCOMPILER_COMMON_SUPPORT_RELEASE",
    "LLMCOMPILER_PARSER_SOURCE_SHA256",
    "LLMCOMPILER_PLANNER_SOURCE_SHA256",
    "LLMCompilerCommonSupportConfig",
    "LLMCompilerCommonSupportPlanner",
    "parse_llmcompiler_plan",
    "render_llmcompiler_planner_prompt",
]
