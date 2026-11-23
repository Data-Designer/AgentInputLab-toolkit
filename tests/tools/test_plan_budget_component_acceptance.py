import pytest
from run_plan_budget_component_acceptance import remaining_budget

from agentinputlab import ExecutionBudget
from agentinputlab.contracts import BudgetExceededError


def test_reader_is_charged_actual_component_output():
    original = ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
    result = remaining_budget(original, [{"response": {"usage": {"output_tokens": 123}}}])
    assert result.max_model_calls == 7 and result.max_output_tokens == 3973
    assert original.max_output_tokens == 4096


@pytest.mark.parametrize("calls", [[{}], [{"response": {"usage": {"output_tokens": 4096}}}],
                                  [{"response": {"usage": {"output_tokens": 0}}}] * 8])
def test_unknown_or_exhausted_usage_cannot_dispatch_separate_reader(calls):
    with pytest.raises(BudgetExceededError):
        remaining_budget(ExecutionBudget(), calls)
