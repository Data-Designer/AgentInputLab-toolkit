from dataclasses import replace

import pytest
from run_llmcompiler_planner_acceptance import materialize_for_reader, remaining_budget

from agentinputlab import ExecutionBudget, FrozenAgentState
from agentinputlab.contracts import (
    BudgetExceededError,
    ContextBlock,
    ExecutionTrace,
    InputPatch,
    Message,
    PatchOp,
    PatchOperation,
    ValidationError,
)


def test_reader_receives_only_actual_remaining_budget():
    original = ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
    actual = remaining_budget(original, [{"response": {"usage": {"output_tokens": 400}}}])
    assert actual.max_model_calls == 7 and actual.max_output_tokens == 3696
    assert original.max_model_calls == 8 and original.max_output_tokens == 4096


@pytest.mark.parametrize("calls", [[{}], [{"response": {"usage": {"output_tokens": 4096}}}],
                                  [{"response": {"usage": {"output_tokens": 0}}}] * 8])
def test_unknown_or_exhausted_usage_cannot_fund_reader(calls):
    with pytest.raises(BudgetExceededError):
        remaining_budget(ExecutionBudget(), calls)


def test_query_only_planner_dag_is_added_then_reader_protocol_restored():
    original = FrozenAgentState(query="Synthetic case", task_id="case-1", instruction="Reader contract",
        messages=(Message("user", "Visible history"),),
        context=(ContextBlock("Original visible evidence", block_id="evidence"),))
    projected = replace(original, instruction="", messages=())
    patch = InputPatch(method_id="llm_compiler", release_id="plan.llm_compiler.planner_component",
        source_state_hash=projected.state_hash,
        trace=ExecutionTrace(method_id="llm_compiler", release_id="plan.llm_compiler.planner_component"),
        implementation_identity="synthetic_tool_test", profile_kind="controlled_adapter",
        operations=(PatchOperation(PatchOp.ADD_CONTEXT, ContextBlock("Synthetic DAG", block_id="plan")),))
    result = materialize_for_reader(original, projected, patch)
    assert result.instruction == original.instruction and result.messages == original.messages
    assert result.query == original.query and result.task_id == original.task_id
    assert result.context == (*original.context, ContextBlock("Synthetic DAG", block_id="plan"))
    assert original.context == (ContextBlock("Original visible evidence", block_id="evidence"),)
    with pytest.raises(ValidationError):
        materialize_for_reader(original, original, patch)
