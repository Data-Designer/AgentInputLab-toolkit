"""Synthetic fixtures for explicit input preparation, not model acceptance."""
import json

import pytest

from agentinputlab import ContextBlock, FrozenAgentState, RunConfig
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.longbench_configuration import prepare_qrag_context
from tests.unit.test_qrag_file_configuration import wired  # noqa: F401


class Adapter:
    benchmark_id = "longbench.qmsum"
    template = "Read {context}\nQuestion: {input}\nGive the answer."
    pairing_mode = "stateless_score_twice"

    def examples(self, split):
        assert split == "test"
        return (BenchmarkExample(example_id="case", query="needle",
            visible_context=(ContextBlock(block_id="long_context", source="qmsum",
                text="  " + " ".join(["needle"] * 351 + ["evidence"] * 351) + "\n"),),
            evaluator_gold={"answers": ["NEVER_SEND_GOLD"]}),)

    def score(self, example, prediction):
        assert prediction == "answer"
        return {"rouge": 0.25}


def make(wired, monkeypatch):  # noqa: F811
    from agentinputlab import qrag_configuration
    from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
        QRAG_INSTRUCTION_PROMPT_SHA256,
        QRAG_INSTRUCTION_READER_PROFILE,
    )

    value, backend, selections, invocations = wired
    identity = value['parameters'].pop('artifacts')
    identity.update(reader_profile=QRAG_INSTRUCTION_READER_PROFILE,
                    reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256)
    monkeypatch.setattr(qrag_configuration, '_read_batch_assets', lambda *_: identity)
    backend.responses = [(json.dumps(dict(answer="answer", support_ids=[], evidence_ids=[], evidence_texts=[])), 4)]
    reader = dict(repository='fixture', revision='fixture', profile=QRAG_INSTRUCTION_READER_PROFILE,
                  longbench_context={'kind': 'lossless_words350_bm25_top6.v1'})
    return value, reader, backend, selections, invocations


def test_public_evaluation_prepares_original_visible_context_and_charges_cpu_retrieval(wired, monkeypatch):  # noqa: F811
    value, reader, backend, selections, invocations = make(wired, monkeypatch)
    result = RunConfig.from_dict(value).evaluate('longbench.qmsum', split='test',
        example_ids=['case'], adapter=Adapter(), qrag_reader=reader,
        predictor_parameters={'schema_version': 1, 'native_text': {'composite_output_format': 'structured_qa_json'}})
    assert result['scored_count'] == 1, result['records']
    row = result['records'][0]
    assert row['usage']['retrieval_calls'] == row['usage']['model_calls'] == 1
    assert row['usage']['output_tokens'] == 4 and len(backend.requests) == 1
    assert row['candidate_preparation']['visible_chunk_count'] == 3
    assert row['candidate_preparation']['model_and_output_token_budget_unchanged'] is True
    assert row['qrag_binding']['mode'] == 'longbench_visible_projection'
    assert row['input_state_hash'] == row['candidate_preparation']['prepared_state_sha256']
    assert row['source_input_state_hash'] != row['input_state_hash']
    assert len(selections) == len(invocations) == 1
    assert 'NEVER_SEND_GOLD' not in repr(backend.requests)
    assert 'Give the answer.' in backend.requests[0].prompt
    assert backend.requests[0].max_tokens == value['budget']['max_output_tokens']


def test_projection_is_lossless_and_keeps_exact_offsets_and_budget():
    adapter = Adapter()
    state = adapter.examples('test')[0].visible_state()
    seen = []
    def observe(retriever):
        seen.append(retriever)
        return retriever
    prepared, audit = prepare_qrag_context(state, adapter=adapter, observe_retriever=observe)
    ordered = sorted(prepared.context, key=lambda b: b.metadata['source_char_start'])
    assert ''.join(b.text for b in ordered) == state.context[0].text
    assert prepared.budget == state.budget and prepared.query == state.query
    assert prepared.instruction.replace('[AGENTINPUTLAB_SELECTED_VISIBLE_PASSAGES]', state.context[0].text) == adapter.template.format(context=state.context[0].text, input=state.query)
    assert audit['native_template_sha256'] == content_hash(adapter.template)
    for b in ordered:
        m = b.metadata
        assert b.text == state.context[0].text[m['source_char_start']:m['source_char_end']]
    assert len(seen) == 1


def test_too_few_real_candidates_are_not_padded():
    with pytest.raises(ValidationError, match='at least two'):
        prepare_qrag_context(FrozenAgentState(query='q', context=(ContextBlock(text='short'),)),
                             adapter=Adapter(), observe_retriever=lambda r: r)


def test_selection_is_not_claimed_to_be_lossless_when_more_than_six_chunks():
    state = FrozenAgentState(query='needle', context=(ContextBlock(block_id='doc',
        text=' '.join(['needle'] * 2500)),))
    prepared, audit = prepare_qrag_context(state, adapter=Adapter(), observe_retriever=lambda r: r)
    assert audit['visible_chunk_count'] == 8 and len(prepared.context) == 6
    assert audit['corpus_lossless'] and not audit['selection_is_full_context']


def test_reserved_marker_is_not_interpreted_as_a_template_slot():
    state = FrozenAgentState(query='[AGENTINPUTLAB_SELECTED_VISIBLE_PASSAGES]',
        context=(ContextBlock(text=' '.join(['word'] * 701)),))
    with pytest.raises(ValidationError, match='reserved'):
        prepare_qrag_context(state, adapter=Adapter(), observe_retriever=lambda r: r)


def test_no_retrieval_budget_stops_before_policy_or_reader(wired, monkeypatch):  # noqa: F811
    value, reader, backend, selections, invocations = make(wired, monkeypatch)
    value['budget']['max_retrieval_calls'] = 0
    result = RunConfig.from_dict(value).evaluate('longbench.qmsum', split='test', limit=1,
                                               adapter=Adapter(), qrag_reader=reader)
    assert result['failed_count'] == 1
    assert not selections and not invocations and not backend.requests


@pytest.mark.parametrize('change', [
    {'longbench_context': {}}, {'longbench_context': {'kind': 'invented'}},
    {'longbench_context': {'kind': 'lossless_words350_bm25_top6.v1', 'extra': True}},
    {'profile': 'factorial_reader_transfer'},
    {'candidate_source': {'kind': 'retriever', 'top_k': 6, 'max_document_chars': 100}},
])
def test_projection_configuration_is_explicit(wired, monkeypatch, change):  # noqa: F811
    value, reader, backend, selections, invocations = make(wired, monkeypatch)
    reader.update(change)
    with pytest.raises(ValidationError):
        RunConfig.from_dict(value).evaluate('longbench.qmsum', split='test', limit=1,
                                           adapter=Adapter(), qrag_reader=reader)
    assert not selections and not invocations and not backend.requests


def test_projection_rejects_other_benchmark(wired, monkeypatch):  # noqa: F811
    value, reader, backend, selections, invocations = make(wired, monkeypatch)
    with pytest.raises(ValidationError, match='only to LongBench'):
        RunConfig.from_dict(value).evaluate('2wiki.multihop', split='test', limit=1,
                                           adapter=Adapter(), qrag_reader=reader)
    assert not selections and not invocations and not backend.requests
