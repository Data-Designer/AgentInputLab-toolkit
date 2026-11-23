from dataclasses import replace

import pytest

from agentinputlab.algorithms.curate.recomp import prepare_recomp_sentence_state
from agentinputlab.contracts import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    Message,
    ValidationError,
)


def test_preparation_preserves_control_fields_and_exact_source_substrings():
    state = FrozenAgentState(query="question", instruction="instruction", task_id="case",
        messages=(Message(role="user", content="prior"),), tool_names=("tool",),
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096), metadata={"custom": "value"},
        context=(ContextBlock(block_id="doc", text="  First sentence.\n Second?  Third!  "),))
    prepared = prepare_recomp_sentence_state(state)
    assert prepared.query == state.query and prepared.messages == state.messages
    assert prepared.instruction == state.instruction and prepared.tool_names == state.tool_names
    assert prepared.budget == state.budget and prepared.task_id == state.task_id
    assert prepared.metadata["custom"] == "value"
    assert [b.text for b in prepared.context] == ["First sentence.", "Second?", "Third!"]
    for block in prepared.context:
        metadata = block.metadata
        original = state.context[metadata["source_context_block_index"]]
        assert original.text[metadata["source_char_start"]:metadata["source_char_end"]] == block.text
        assert metadata["source_captured_state_hash"] == state.state_hash
    assert state.metadata == {"custom": "value"} and len(state.context) == 1
    assert prepared.metadata["recomp_sentence_preparation"]["paper_native_preprocessing_claimed"] is False
    with pytest.raises(ValidationError, match="already has"):
        prepare_recomp_sentence_state(prepared)


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.5])
def test_invalid_limits_fail_without_preparing(limit):
    with pytest.raises(ValidationError, match="max_candidates"):
        prepare_recomp_sentence_state(FrozenAgentState(query="q"), max_candidates=limit)


def test_limit_is_enforced_without_returning_silently_truncated_evidence():
    state = FrozenAgentState(query="q", context=(ContextBlock(text="One. Two."),))
    assert len(prepare_recomp_sentence_state(state, max_candidates=2).context) == 2
    with pytest.raises(ValidationError, match="limit exceeded"):
        prepare_recomp_sentence_state(state, max_candidates=1)
    assert state.context[0].text == "One. Two."


def test_empty_or_ambiguous_context_is_rejected():
    state = FrozenAgentState(query="q", context=(ContextBlock(text=" "),))
    with pytest.raises(ValidationError, match="non-empty"):
        prepare_recomp_sentence_state(state)
    repeated = replace(state, context=(ContextBlock(block_id="same", text="One."),
                                       ContextBlock(block_id="same", text="Two.")))
    with pytest.raises(ValidationError, match="ambiguous"):
        prepare_recomp_sentence_state(repeated)
