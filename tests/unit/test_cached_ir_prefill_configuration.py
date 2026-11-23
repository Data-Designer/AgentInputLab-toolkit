"""Synthetic cache preparation tests, not model-acceptance evidence."""
import hashlib
import json
from dataclasses import replace

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.algorithms.reformulate.infocqr.cache_replay import conversation_view_hash
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ContextBlock, ValidationError
from agentinputlab.retrieval_prefill_configuration import prepare
from tests.unit.test_composite_ir_evaluate_configuration import IRAdapter, evaluate


def setup(tmp_path, *, limit=8):
    budget = ExecutionBudget(max_retrieval_calls=limit)
    retriever = InMemoryBM25Retriever([Document('d', 'Visible question a savings', corpus_revision='frozen')])
    state = IRAdapter().examples('test')[0].visible_state(budget=budget)
    settings = dict(top_k=1, max_document_chars=10000)
    prepared, _ = prepare(state, retriever, settings)
    path = tmp_path/'cache.jsonl'
    path.write_text(json.dumps(dict(record_id='a', split='test', mode='rw_zsl',
        source_query=state.query, conversation_hash=conversation_view_hash(prepared), rewrite='savings'))+'\n')
    config = RunConfig.from_dict(dict(release_id='rewrite.infocqr.cached_rw_zsl',
        budget=dict(max_retrieval_calls=limit), parameters=dict(split='test'),
        artifacts=dict(cache=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            artifact_id='synthetic-fixture', dataset_revision='frozen'))))
    params = dict(schema_version=1, retrieval=dict(top_k=1,query_mode='query_only',source_mode='retriever'),
                  retrieval_prefill=settings)
    return config, InterventionContext(retriever=retriever), params


def test_cache_context_preparation_scores_without_backend_or_generation(tmp_path):
    config, context, params = setup(tmp_path)
    row = evaluate(config, context, params)['records'][0]
    assert row['status'] == 'scored'
    assert row['usage']['retrieval_calls'] == 2
    assert row['usage']['model_calls'] == row['usage']['output_tokens'] == 0
    assert row['evaluation']['metrics'] == {'ndcg_at_10':1.0}
    assert row['context_prefill']['document_ids'] == ['d']
    assert row['context_prefill']['prepared_state_sha256'] != row['input_state_hash']
    assert row['intervention']['patch']['operations'][0]['value'] == 'savings'
    assert 'patch_query_reader' not in row['downstream_metadata']


@pytest.mark.parametrize('limit,stage,calls', [(0,'context_prefill',0),(1,'downstream',1)])
def test_prefill_respects_original_retrieval_budget(tmp_path, limit, stage, calls):
    config, context, params = setup(tmp_path, limit=limit)
    row = evaluate(config, context, params)['records'][0]
    assert row['status'] == 'failed' and row['failure']['stage'] == stage
    assert row['usage']['retrieval_calls'] == calls and row['usage']['model_calls'] == 0


@pytest.mark.parametrize('key', ['patch_retrieval','composite_retrieval'])
def test_conflicting_preparation_rejected(tmp_path,key):
    config, context, params = setup(tmp_path)
    params[key] = {'output_format':'json_query'}
    with pytest.raises(ValidationError,match='retrieval_prefill'):
        evaluate(config,context,params)


@pytest.mark.parametrize('override', [dict(query_mode='instruction_plus_query'),dict(source_mode='context_or_retriever')])
def test_cannot_ignore_prepared_context_or_change_query_contract(tmp_path,override):
    config,context,params=setup(tmp_path)
    params['retrieval'].update(override)
    with pytest.raises(ValidationError):
        evaluate(config,context,params)


def test_different_visible_context_remains_a_cache_failure(tmp_path):
    config, context, params=setup(tmp_path)
    class DifferentConversation(IRAdapter):
        def examples(self,split):
            return (replace(super().examples(split)[0],visible_metadata={'new_context':'changed conversation'}),)
    row=config.evaluate('beir.arguana',split='test',example_ids=['a'],adapter=DifferentConversation(),
        context=context,predictor_parameters=params)['records'][0]
    assert row['status']=='failed' and row['usage']['model_calls']==0
    assert row['usage']['retrieval_calls']==1
    assert row['failure']['type']=='ArtifactError' and row['failure']['stage']=='intervention'


def test_default_does_not_prepare_or_regenerate_context(tmp_path):
    config,context,params=setup(tmp_path)
    del params['retrieval_prefill']
    row=evaluate(config,context,params)['records'][0]
    # InfoCQR hashes conversational messages/new_context, not retrieval passages.
    assert row['status']=='scored' and row['usage']['retrieval_calls']==1
    assert row['usage']['model_calls']==0 and 'context_prefill' not in row


def test_existing_context_is_never_overwritten(tmp_path):
    config,context,params=setup(tmp_path)
    class WithContext(IRAdapter):
        def examples(self,split):
            return (replace(super().examples(split)[0],visible_context=(ContextBlock(block_id='existing',text='keep'),)),)
    row=config.evaluate('beir.arguana',split='test',example_ids=['a'],adapter=WithContext(),
        context=context,predictor_parameters=params)['records'][0]
    assert row['status']=='failed' and row['failure']['stage']=='context_prefill'
    assert row['usage']['retrieval_calls']==0
