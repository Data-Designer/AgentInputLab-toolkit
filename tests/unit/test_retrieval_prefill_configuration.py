"""Synthetic prefill boundary tests; real saved-response validation is separate."""
from dataclasses import replace

import pytest

from agentinputlab import ExecutionBudget
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ContextBlock, FrozenAgentState, ValidationError
from agentinputlab.retrieval_prefill_configuration import options, prepare
from tests.unit.test_composite_ir_evaluate_configuration import evaluate, setup
from tests.unit.test_composite_ir_evaluate_configuration import options as ir_options


def settings():
    return {"top_k": 1, "max_document_chars": 10000}


def test_pipeline_prefills_visible_query_and_accounts_both_retrievals(tmp_path):
    config, backend, context = setup(tmp_path)
    result = evaluate(config, context, ir_options(context_prefill=settings()))
    row = result["records"][0]
    assert row["status"] == "scored" and len(backend.requests) == 1
    assert row["usage"]["retrieval_calls"] == 2
    assert row["usage"]["model_calls"] == 1 and row["usage"]["output_tokens"] == 4
    receipt = row["context_prefill"]
    assert receipt["document_ids"] == ["d"]
    assert receipt["source_state_sha256"] == row["source_input_state_hash"]
    assert receipt["execution_state_sha256"] == row["input_state_hash"]
    assert receipt["prepared_state_sha256"] != row["input_state_hash"]  # retrieval budget consumed
    assert "NEVER_SEND_GOLD" not in str(backend.requests)


@pytest.mark.parametrize("limit,stage,requests,retrievals", [(0,"context_prefill",0,0),(1,"downstream",1,1)])
def test_no_budget_increase_or_retry(tmp_path, limit, stage, requests, retrievals):
    config, backend, context = setup(tmp_path, budget=ExecutionBudget(max_retrieval_calls=limit))
    row = evaluate(config, context, ir_options(context_prefill=settings()))["records"][0]
    assert row["status"] == "failed" and row["failure"]["stage"] == stage
    assert len(backend.requests) == requests and row["usage"]["retrieval_calls"] == retrievals


@pytest.mark.parametrize("bad", [None, {}, {"top_k":0,"max_document_chars":1},
    {"top_k":True,"max_document_chars":1},{"top_k":1,"max_document_chars":1.5},
    {"top_k":1,"max_document_chars":10,"extra":True}])
def test_invalid_options_rejected_before_model(tmp_path, bad):
    config, backend, context = setup(tmp_path)
    with pytest.raises(ValidationError, match="context_prefill"):
        evaluate(config, context, ir_options(context_prefill=bad))
    assert not backend.requests


def test_text_only_projection_clipping_and_original_state_unchanged():
    state = FrozenAgentState(task_id="a", query="hello")
    retriever = InMemoryBM25Retriever([Document("d", "hello world", title="not injected",
        corpus_revision="pinned", metadata={"not_injected":True})])
    prepared, receipt = prepare(state, retriever, options({"top_k":1,"max_document_chars":5}))
    assert not state.context
    assert prepared.context == (ContextBlock(block_id="d", text="hello", source="pinned"),)
    assert receipt["clipped_documents"] == 1
    with pytest.raises(ValidationError, match="no replacement"):
        prepare(prepared, retriever, settings())


def test_empty_results_never_fall_back_to_original_query(tmp_path):
    config, backend, context = setup(tmp_path)
    context = replace(context, retriever=InMemoryBM25Retriever([]))
    row = evaluate(config, context, ir_options(context_prefill=settings()))["records"][0]
    assert row["status"] == "failed" and row["failure"]["stage"] == "context_prefill"
    assert row["usage"]["retrieval_calls"] == 1 and not backend.requests


def test_explicit_native_projection_preserves_scores_and_rank_provenance_only():
    state = FrozenAgentState(task_id='a', query='hello', metadata={'benchmark_cell':'bright.stackoverflow'})
    retriever = InMemoryBM25Retriever([Document('d', 'hello world', title='not injected',
        corpus_revision='pinned', metadata={'not_injected':True})])
    result = retriever.retrieve('hello', top_k=1)[0]
    prepared, receipt = prepare(state, retriever, options(dict(top_k=1,max_document_chars=5,
        projection='native_ir_context.v1')))
    assert prepared.context == (ContextBlock(block_id='d',text='hello',source='bright.stackoverflow',
        score=result.score,metadata={'document_id':'d','initial_capture_rank':1,
                                     'initial_capture_corpus_revision':'pinned'}),)
    assert prepared.metadata['initial_capture_retrieval_calls'] == 1
    assert receipt['mode']=='native_ir_context.v1'
    assert not state.context and 'initial_capture_retrieval_calls' not in state.metadata
    assert 'not_injected' not in repr(prepared)


def test_native_projection_requires_visible_dataset_identity_before_retrieval():
    class Never:
        def retrieve(self,*args,**kwargs): raise AssertionError('must not call')
    with pytest.raises(ValidationError,match='benchmark_cell'):
        prepare(FrozenAgentState(task_id='a',query='q'),Never(),
                options(dict(top_k=1,max_document_chars=5,projection='native_ir_context.v1')))
    with pytest.raises(ValidationError,match='projection'):
        options(dict(top_k=1,max_document_chars=5,projection='unknown'))
