"""Public preparation/entrypoint tests; fixture inference is not model acceptance."""
import json
from dataclasses import asdict, replace

import pytest

from agentinputlab import ExecutionBudget, RunConfig
from agentinputlab.contracts import ValidationError
from tests.workers import test_adaptive_rag_worker_driver as fixture_module


def test_existing_controlled_profile_accepts_explicit_file_runtime_without_io():
    fixture = fixture_module.AdaptiveRAGWorkerDriverTests()
    fixture.setUp()
    try:
        config = RunConfig('acquire.adaptive_rag.controlled_reimplementation',
            parameters={'artifacts': asdict(fixture.artifacts)},
            backend={'model_id': 'test', 'base_url': 'http://127.0.0.1:9/v1',
                     'deployment_identity': {'revision': 'fixture-v1'}},
            retriever={'kind': 'bm25', 'documents': [{'doc_id': 'd', 'text': 'test'}]},
            runtime={'kind': 'adaptive_rag_controlled', 'source_root': '/missing/source',
                     'router_snapshot_path': '/missing/router', 'router_weights_path': '/missing/weights',
                     'router_tokenizer_path': '/missing/tokenizer', 'prompt_root': '/missing/prompts',
                     'deployment_path': '/missing/deployment.json', 'deployment_sha256': 'a'*64,
                     'input_token_upper_bound': 65536})
        assert config.validate()['valid']
    finally:
        fixture.tearDown()


def test_preparation_command_is_exposed_without_loading_models(capsys):
    from agentinputlab.cli.main import main
    with pytest.raises(SystemExit) as exc:
        main(['config', 'prepare-adaptive', '--help'])
    assert exc.value.code == 0
    assert 'output-dir' in capsys.readouterr().out


def test_runtime_paths_resolve_against_configuration_file(tmp_path):
    path = tmp_path/'template.json'
    path.write_text(json.dumps({'release_id': 'acquire.adaptive_rag.controlled_reimplementation',
        'runtime': {'kind': 'adaptive_rag_controlled', 'router_snapshot_path': 'router',
            'router_weights_path': 'router/model.safetensors', 'router_tokenizer_path': 'router/tokenizer.json',
            'prompt_root': 'prompts', 'deployment_path': 'prepared/deployment.json'}}))
    config = RunConfig.from_file(path)
    for key in ('router_snapshot_path', 'router_weights_path', 'router_tokenizer_path', 'prompt_root', 'deployment_path'):
        assert config.runtime[key].startswith(str(tmp_path)+'/')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from agentinputlab.adaptive_configuration import prepare
    from agentinputlab.algorithms.acquire.adaptive_rag.runtime import (
        ADAPTIVE_RAG_SOURCE_CLOSURE_SHA256,
        ADAPTIVE_RAG_UPSTREAM_COMMIT,
        ADAPTIVE_RAG_UPSTREAM_TREE,
    )
    from tests.unit.test_benchmark_configuration import Backend
    from workers.adaptive_rag import driver, runtime

    for directory in ('source', 'router', 'prompts'):
        (tmp_path/directory).mkdir()
    for file in ('config.json', 'model.safetensors', 'tokenizer.json'):
        (tmp_path/'router'/file).write_text('explicit unit fixture')
    for name in ('no_retrieval', 'single_retrieval', 'ircot'):
        (tmp_path/'prompts'/(name+'.txt')).write_text('test '+name)
    # Only native source and classifier inference are fixtures. The original
    # driver still validates all generated closures, budgets, events and output.
    monkeypatch.setattr(driver.AdaptiveRAGWorkerDriver, '_verify_source', lambda self: {
        'commit': ADAPTIVE_RAG_UPSTREAM_COMMIT, 'tree': ADAPTIVE_RAG_UPSTREAM_TREE,
        'state': 'clean_git_checkout', 'closure_sha256': ADAPTIVE_RAG_SOURCE_CLOSURE_SHA256})
    loads, classifications = [], []
    class Router:
        def __init__(self, **kwargs):
            loads.append(kwargs)
        def estimate_input_tokens(self, question):
            return 10
        def first_decoder_step(self, question, *, seed):
            classifications.append((question, seed))
            label = 'B' if question.endswith('b') else 'A'
            return driver.RouterResult(logits={k: float(k == label) for k in ('A','B','C')},
                                       input_tokens=10, output_tokens=1)
    monkeypatch.setattr(runtime, 'LocalTransformersRouter', Router)
    backend = Backend([('answer', 3), ('wrong', 2)])
    monkeypatch.setattr(RunConfig, '_backend', lambda self: backend)
    template = RunConfig('acquire.adaptive_rag.controlled_reimplementation',
        parameters={'controls': {'max_output_tokens': 32, 'ircot_max_steps': 1}},
        backend={'model_id': 'fixture'},
        retriever={'kind': 'bm25', 'documents': [{'doc_id': 'd', 'text': 'visible evidence'}]},
        budget=ExecutionBudget(max_output_tokens=100, max_model_calls=8),
        runtime={'kind': 'adaptive_rag_controlled', 'source_root': str(tmp_path/'source'),
            'router_snapshot_path': str(tmp_path/'router'),
            'router_weights_path': str(tmp_path/'router/model.safetensors'),
            'router_tokenizer_path': str(tmp_path/'router/tokenizer.json'),
            'prompt_root': str(tmp_path/'prompts'), 'input_token_upper_bound': 100})
    proof = prepare(template, tmp_path/'prepared', router_checkpoint='unit-fixture-not-real-model')
    assert not loads and not classifications and not backend.requests
    assert proof['model_calls'] == 0 and not proof['router_weights_loaded']
    return RunConfig.from_file(proof['run_path']), backend, loads, classifications, template


def test_prepared_original_driver_runs_selected_arms_and_native_score(prepared):
    from tests.unit.test_benchmark_configuration import Adapter
    config, backend, loads, classifications, _ = prepared
    result = config.evaluate('2wiki.multihop', split='dev', example_ids=['a','b'], adapter=Adapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}})
    assert result['scored_count'] == 2 and result['failed_count'] == 0
    assert len(backend.requests) == len(classifications) == 2
    assert all(row['usage']['model_calls'] == 2 for row in result['records'])
    assert result['records'][0]['usage']['retrieval_calls'] == 0
    assert result['records'][1]['usage']['retrieval_calls'] == 1
    assert all('NEVER_SEND_GOLD' not in request.prompt for request in backend.requests)


def test_changed_reader_or_retriever_rejected_before_inference(prepared):
    config, backend, loads, classifications, _ = prepared
    changed = replace(config, retriever={'kind': 'bm25', 'documents': [{'doc_id': 'other', 'text': 'changed'}]})
    with pytest.raises(ValidationError, match='deployment differs'):
        changed.run('question')
    assert not loads and not classifications and not backend.requests
    backend.fingerprint = lambda: {'model_id':'changed'}
    with pytest.raises(ValidationError, match='deployment differs'):
        config.run('question')
    assert not loads and not classifications and not backend.requests


def test_rich_state_is_not_silently_reduced_to_question(prepared):
    from agentinputlab import FrozenAgentState
    config, backend, loads, classifications, _ = prepared
    with pytest.raises(ValidationError, match='question-only'):
        config.run(FrozenAgentState(query='question', instruction='Required evidence'))
    assert not loads and not classifications and not backend.requests


def test_changed_checkpoint_rejected_before_any_model_call(prepared):
    from pathlib import Path

    from agentinputlab.contracts import ExternalDependencyError
    config, backend, loads, classifications, _ = prepared
    Path(config.runtime['router_weights_path']).write_text('changed fixture')
    with pytest.raises(ExternalDependencyError, match='ROUTER_MISMATCH'):
        config.run('question')
    assert not loads and not classifications and not backend.requests


def test_failed_generation_remains_failed_without_extra_reader(prepared):
    from tests.unit.test_benchmark_configuration import Adapter
    config, backend, _, classifications, _ = prepared
    backend.responses = [('',0), ('answer',3)]
    result = config.evaluate('2wiki.multihop', split='dev', example_ids=['a','b'], adapter=Adapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}})
    assert result['selected_count'] == 2 and result['failed_count'] == result['scored_count'] == 1
    assert len(backend.requests) == len(classifications) == 2
    assert all(row['usage']['model_calls'] == 2 for row in result['records'])


@pytest.mark.parametrize('value', [None, 0, True, -1, '100'])
def test_invalid_input_reservation_rejected_without_io(value):
    from agentinputlab.adaptive_configuration import validate_runtime
    config = RunConfig('acquire.adaptive_rag.controlled_reimplementation',
        backend={'model_id':'fixture'}, retriever={'kind':'bm25'}, runtime={
            'kind':'adaptive_rag_controlled', 'source_root':'source', 'router_snapshot_path':'router',
            'router_weights_path':'weights','router_tokenizer_path':'tokenizer','prompt_root':'prompts',
            'input_token_upper_bound': value})
    with pytest.raises(ValidationError, match='positive integer'):
        validate_runtime(config, prepared=False)
