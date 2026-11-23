from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.algorithms.decompose.llm_compiler import (
    LLMCOMPILER_COMMON_SUPPORT_RELEASE,
    LLMCompilerCommonSupportConfig,
    LLMCompilerCommonSupportPlanner,
    parse_llmcompiler_plan,
    render_llmcompiler_planner_prompt,
)
from agentinputlab.backends.llm import (
    GenerationRequest,
    GenerationResponse,
    ModelCapability,
    Usage,
)
from agentinputlab.connectors.tools import (
    SchemaOnlyToolConnector,
    ToolCall,
    ToolSpec,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.experiments.drivers.static_production import (
    StaticBackendBuildRequest,
    static_runtime_product_sha256,
)
from agentinputlab.experiments.production_builders import ProductionBackendBuilder


class FixtureBackend:
    model_id = "fixture-compiler"
    capabilities = frozenset({ModelCapability.COMPLETION, ModelCapability.SEED})

    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        return GenerationResponse(
            text=self.response,
            model=self.model_id,
            finish_reason="stop",
            usage=Usage(input_tokens=91, output_tokens=19, cost_usd=0.002),
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {"backend": "fixture-compiler", "revision": "v1"}


def _specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="lookup",
            description="Look up a customer record.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            read_only=False,
        ),
        ToolSpec(
            name="update_record",
            description="Update a customer record.",
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["record_id", "value"],
            },
            read_only=False,
        ),
    )


def _check_prompt_preserves_source_guidelines_and_renders_frozen_schema() -> None:
    prompt = render_llmcompiler_planner_prompt(_specs())
    assert prompt.startswith(
        "Given a user query, create a plan to solve it with the utmost parallelizability."
    )
    assert "Each plan should comprise an action from the following 3 types:" in prompt
    assert "lookup(query: string) -> Any:" in prompt
    assert "Always call join as the last action in the plan" in prompt
    assert "Say '<END_OF_PLAN>' after you call join" in prompt
    assert prompt.endswith("Here are some examples:\n\n")


def _check_parser_materializes_dependency_dag_and_omits_join() -> None:
    tasks, join_id = parse_llmcompiler_plan(
        'Thought: parallel lookups\n1. lookup("customer 7")\n'
        '2. update_record("$1", "verified")\n3. join()',
        _specs(),
        max_tasks=8,
    )
    assert join_id == 3
    assert [task.task_id for task in tasks] == [1, 2]
    assert tasks[0].arguments == {"query": "customer 7"}
    assert tasks[1].arguments == {"record_id": "$1", "value": "verified"}
    assert tasks[1].dependencies == (1,)


def _check_parser_does_not_treat_currency_as_a_task_dependency() -> None:
    tasks, join_id = parse_llmcompiler_plan(
        '1. lookup(query: "income is $100,000")\n2. join()',
        _specs(),
        max_tasks=8,
    )
    assert join_id == 2
    assert tasks[0].arguments == {"query": "income is $100,000"}
    assert tasks[0].dependencies == ()


def _check_parser_rejects_invalid_source_plan(raw: str) -> None:
    try:
        parse_llmcompiler_plan(raw, _specs(), max_tasks=8)
    except AlgorithmOutputError:
        return
    raise AssertionError("invalid LLMCompiler plan was accepted")


def _check_common_support_runs_one_model_call_and_no_tools() -> None:
    backend = FixtureBackend(
        'Thought: parallel lookups\n1. lookup("customer 7")\n'
        '2. update_record("$1", "verified")\n3. join()'
    )
    connector = SchemaOnlyToolConnector(_specs(), connector_id="arena-tools")
    state = FrozenAgentState(
        query="Verify customer 7",
        task_id="task-7",
        tool_names=("lookup", "update_record"),
    )
    patch = LLMCompilerCommonSupportPlanner(
        LLMCompilerCommonSupportConfig(max_tasks=8, max_tokens=128)
    ).propose(
        state,
        InterventionContext(backend=backend, tools=connector, seed=13),
    )
    assert patch.release_id == LLMCOMPILER_COMMON_SUPPORT_RELEASE
    assert patch.trace.counters["model_calls"] == 1
    assert patch.trace.counters["tool_calls"] == 0
    assert patch.metadata["full_system"] is False
    assert patch.metadata["final_answer_injected"] is False
    updated = patch.apply(state)
    assert updated.context[-1].source == "llm_compiler_task_dag"
    assert updated.metadata["llm_compiler_join_task_id"] == 3
    assert len(updated.metadata["llm_compiler_tasks"]) == 2
    assert len(backend.requests) == 1
    assert backend.requests[0].prompt == "Question: Verify customer 7"
    assert backend.requests[0].stop == ("<END_OF_PLAN>",)
    assert "Return only the source planner grammar as plain text" in backend.requests[0].system


def _check_controlled_static_profile_accepts_connector_only_inventory() -> None:
    backend = FixtureBackend('1. lookup("customer 7")\n2. join()')
    connector = SchemaOnlyToolConnector(_specs()[:1], connector_id="static-retrieve")
    state = FrozenAgentState(query="Verify customer 7")
    with unittest.TestCase().assertRaisesRegex(ValidationError, "tool schemas differ"):
        LLMCompilerCommonSupportPlanner().propose(
            state,
            InterventionContext(backend=backend, tools=connector, seed=13),
        )

    controlled_backend = FixtureBackend('1. lookup("customer 7")\n2. join()')
    patch = LLMCompilerCommonSupportPlanner(
        LLMCompilerCommonSupportConfig(
            allow_connector_only_tool_inventory=True,
        )
    ).propose(
        state,
        InterventionContext(backend=controlled_backend, tools=connector, seed=13),
    )
    assert patch.profile_kind == "controlled_adapter"
    assert patch.metadata["tool_inventory_policy"] == ("connector_only_controlled_static.v1")


def _check_schema_only_connector_normalizes_and_denies_execution() -> None:
    connector = SchemaOnlyToolConnector.from_openai_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        connector_id="official-schema:fixture",
    )
    assert connector.list_tools()[0].name == "lookup"
    result = connector.call(ToolCall("lookup", {"query": "x"}, call_id="call-1"))
    assert result.error.startswith("schema-only connector cannot execute")
    assert result.metadata["denied"] is True
    assert connector.fingerprint()["execution_policy"] == "deny_all.v1"


def _check_production_builder_hashes_schema_only_connector(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "responses": [
                    {
                        "text": "ok",
                        "model": "fixture",
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parameters = {
        "schema_version": 1,
        "backend": {
            "kind": "replay",
            "model_id": "fixture",
            "fixture_path": str(replay),
            "fixture_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
            "capabilities": ["completion", "seed"],
        },
        "required_capabilities": ["completion", "seed"],
        "tools": {
            "kind": "schema_only",
            "connector_id": "official-schema:fixture",
            "definitions": [
                {
                    "name": "lookup",
                    "description": "Look up a record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
        },
    }
    builder = ProductionBackendBuilder()
    assert builder.preflight(parameters) == ()
    runtime = builder.build(StaticBackendBuildRequest("fixture", 0, parameters))
    assert isinstance(runtime.tools, SchemaOnlyToolConnector)
    original_hash = static_runtime_product_sha256(runtime)
    changed = json.loads(json.dumps(parameters))
    changed["tools"]["definitions"][0]["description"] = "Different schema text."
    changed_runtime = builder.build(StaticBackendBuildRequest("fixture", 0, changed))
    assert static_runtime_product_sha256(changed_runtime) != original_hash


class LLMCompilerCommonSupportTests(unittest.TestCase):
    """Core-only tests intentionally runnable without the pytest extra."""

    def test_prompt_preserves_source_guidelines_and_renders_frozen_schema(self) -> None:
        _check_prompt_preserves_source_guidelines_and_renders_frozen_schema()

    def test_parser_materializes_dependency_dag_and_omits_join(self) -> None:
        _check_parser_materializes_dependency_dag_and_omits_join()

    def test_parser_does_not_treat_currency_as_a_task_dependency(self) -> None:
        _check_parser_does_not_treat_currency_as_a_task_dependency()

    def test_parser_rejects_invalid_source_plans(self) -> None:
        invalid_plans = (
            '1. invented("x")\n2. join()',
            '1. lookup("x")',
            "1. join()",
        )
        for raw in invalid_plans:
            with self.subTest(raw=raw):
                _check_parser_rejects_invalid_source_plan(raw)

    def test_common_support_runs_one_model_call_and_no_tools(self) -> None:
        _check_common_support_runs_one_model_call_and_no_tools()

    def test_controlled_static_profile_accepts_connector_only_inventory(self) -> None:
        _check_controlled_static_profile_accepts_connector_only_inventory()

    def test_schema_only_connector_normalizes_and_denies_execution(self) -> None:
        _check_schema_only_connector_normalizes_and_denies_execution()

    def test_production_builder_hashes_schema_only_connector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _check_production_builder_hashes_schema_only_connector(Path(directory))
