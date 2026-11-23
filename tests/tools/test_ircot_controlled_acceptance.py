from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ContextBlock, FrozenAgentState
from tools.run_ircot_controlled_acceptance import (
    CONTROLS,
    build_jobs,
    select_cases,
    source_documents,
    state_for,
)


class Example:
    def __init__(self, number):
        self.example_id = f"case-{number:04d}"

    def visible_state(self):
        return FrozenAgentState(
            query="visible question",
            task_id=self.example_id,
            context=(
                ContextBlock("first complete paragraph", block_id="a", metadata={"title": "First"}),
                ContextBlock(
                    "second complete paragraph", block_id="b", metadata={"title": "Second"}
                ),
            ),
        )


def test_selection_is_fixed_source_order_not_model_or_reference_score():
    adapter = SimpleNamespace(examples=lambda split: [Example(n) for n in reversed(range(205))])
    assert [e.example_id for e in select_cases(adapter)] == ["case-0200", "case-0201"]
    with pytest.raises(ValueError, match="fixed real"):
        select_cases(SimpleNamespace(examples=lambda split: [Example(n) for n in range(201)]))


def test_source_view_and_corpus_are_separate_and_no_reference_is_injected():
    example = Example(200)
    state = state_for(example)
    assert state.context == () and state.query == example.visible_state().query
    assert state.budget.max_model_calls == 8 and state.budget.max_output_tokens == 4096
    documents = source_documents(example)
    assert [d.text for d in documents] == [b.text for b in example.visible_state().context]
    assert [d.title for d in documents] == ["First", "Second"]


def test_original_four_models_share_explicit_seven_plus_one_budget():
    cases = [Example(200), Example(201)]
    models = [{"id": m} for m in ("q4", "gemma", "o20", "q27")]
    jobs = build_jobs(models, cases, {"fixed": "source-proof"})
    assert len(jobs) == len({j["id"] for j in jobs}) == 16
    for job in jobs:
        assert job["seed"] == 17 and job["per_call_output_cap"] == 512
        assert job["budget"]["max_model_calls"] == 8
        assert job["native_controls"] == CONTROLS
        assert job["native_controls"]["max_steps"] + 1 == 8
        assert job["native_controls"]["max_output_tokens"] + 512 == 4096
    for model in models:
        for case in cases:
            pair = [j for j in jobs if j["model"] == model and j["case_id"] == case.example_id]
            assert len(pair) == 2 and pair[0]["input_state_hash"] == pair[1]["input_state_hash"]
