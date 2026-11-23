"""Synthetic boundary checks; real source/response acceptance is separate."""
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ContextBlock, Message, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.ircot_configuration import (
    HOTPOT_INPUT, HOTPOT_INSTRUCTION, prepare_evaluation_input, validate_evaluation_input,
)
from tests.unit.test_ircot_file_configuration import configuration


@pytest.fixture
def bound(tmp_path):
    state = FrozenAgentState(query="Synthetic question", task_id="case",
        instruction=HOTPOT_INSTRUCTION,
        context=(ContextBlock("Visible paragraph", block_id="p", metadata={"title": "Title"}),),
        metadata={"benchmark_cell": "hotpotqa.distractor", "source_split": "validation"})
    corpus = content_hash(state.context)
    documents = [Document("p", "Visible paragraph", title="Title", corpus_revision=corpus)]
    retriever = InMemoryBM25Retriever(documents)
    cfg = configuration(tmp_path, True)
    cfg = replace(cfg, retriever={"kind": "bm25", "documents": to_jsonable(documents)},
        runtime={**cfg.runtime, "corpus_revision": corpus,
                 "index_revision": content_hash(retriever.fingerprint())})
    return state, cfg, retriever


def test_explicit_contract_preserves_question_budget_and_all_paragraphs(bound):
    state, cfg, retriever = bound
    validate_evaluation_input({"projection": HOTPOT_INPUT, "acknowledge_answer_only": True}, cfg,
                              benchmark_id="hotpotqa.distractor", split="validation")
    actual, receipt = prepare_evaluation_input(state, cfg, retriever)
    assert actual == FrozenAgentState(query=state.query, task_id=state.task_id, budget=state.budget)
    assert receipt["source_input_state_hash"] == state.state_hash
    assert receipt["original_instruction"] == state.instruction
    assert receipt["paragraphs"] == 1 and not receipt["support_predictions_synthesized"]
    assert receipt["model_calls"] == receipt["retrieval_calls"] == 0


@pytest.mark.parametrize("value", [None, {}, {"projection": HOTPOT_INPUT},
    {"projection": HOTPOT_INPUT, "acknowledge_answer_only": 1},
    {"projection": HOTPOT_INPUT, "acknowledge_answer_only": True, "ignore_policy": True}])
def test_projection_requires_exact_explicit_options(bound, value):
    with pytest.raises(ValidationError):
        validate_evaluation_input(value, bound[1], benchmark_id="hotpotqa.distractor", split="validation")


@pytest.mark.parametrize("change", [
    {"instruction": HOTPOT_INSTRUCTION + " Respect extra policy."},
    {"messages": (Message("user", "Extra policy"),)}, {"tool_names": ("tool",)},
    {"metadata": {"benchmark_cell": "other", "source_split": "validation"}},
    {"context": ()}, {"context": (ContextBlock("Changed paragraph", block_id="p"),)},
])
def test_changed_visible_inputs_cannot_be_dropped(bound, change):
    state, cfg, retriever = bound
    with pytest.raises(ValidationError):
        prepare_evaluation_input(replace(state, **change), cfg, retriever)


@pytest.mark.parametrize("field", ["corpus_revision", "index_revision"])
def test_pinned_runtime_index_must_match(bound, field):
    state, cfg, retriever = bound
    with pytest.raises(ValidationError):
        prepare_evaluation_input(state, replace(cfg, runtime={**cfg.runtime, field: "wrong"}), retriever)


def test_external_index_cannot_omit_or_inject_paragraphs(bound):
    state, cfg, _ = bound
    with pytest.raises(ValidationError):
        prepare_evaluation_input(state, cfg, InMemoryBM25Retriever([Document("x", "other corpus")]))


@pytest.mark.parametrize("benchmark,split", [("musique.answerable", "validation"),
                                          ("hotpotqa.distractor", "train")])
def test_projection_does_not_expand_other_benchmarks(bound, benchmark, split):
    with pytest.raises(ValidationError):
        validate_evaluation_input({"projection": HOTPOT_INPUT, "acknowledge_answer_only": True}, bound[1],
                                  benchmark_id=benchmark, split=split)
