from types import SimpleNamespace

from run_ace_online_acceptance import SOURCE_INDICES, stream_example

from agentinputlab.contracts import ContextBlock, ExecutionBudget, FrozenAgentState


def test_online_feedback_not_added_to_visible_input_and_cases_are_fresh():
    state = FrozenAgentState(
        task_id="fixture",
        query="visible question",
        context=(ContextBlock("v", "visible passage"),),
        budget=ExecutionBudget(),
    )
    example = SimpleNamespace(
        example_id="fixture",
        visible_state=lambda **kwargs: state,
        evaluator_gold=dict(
            reference_groups=[dict(answers=["feedback-secret-fixture", "second-alias"])]
        ),
    )
    item = stream_example(example)
    assert item["expected_output"] == "feedback-secret-fixture"
    assert "feedback-secret-fixture" not in str(item["input"])
    assert item["metadata"]["online_feedback_not_independent_heldout"]
    assert SOURCE_INDICES == [216, 217]
