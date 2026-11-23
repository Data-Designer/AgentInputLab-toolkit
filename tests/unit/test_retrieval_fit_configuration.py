"""Public retrieval fit binding; fixture data is not real-model evidence."""
import hashlib
import json

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.fit_configuration import FitConfig
from agentinputlab.fit_preparation import _build, _configuration
from tests.unit.test_ace_retrieval_feedback import Adapter
from tests.unit.test_fit_configuration import ace


def inputs(tmp_path, monkeypatch):
    import agentinputlab.benchmarks as benchmarks

    adapter = Adapter()
    def example(split, key):
        for item in adapter.cases:
            if item.example_id == key:
                return item
        raise ValidationError('unknown source ID')
    adapter.example = example
    monkeypatch.setattr(benchmarks, 'load_production_benchmark_adapter', lambda *a: adapter)
    run = tmp_path / 'retriever.json'
    run.write_text(json.dumps(dict(release_id='baseline.no_op', retriever=dict(kind='bm25',
        documents=[dict(doc_id='d1', text='alpha'), dict(doc_id='d2', text='beta')]))))
    source = dict(dataset=dict(kind='production', benchmark_id=adapter.benchmark_id,
        prepared_root='prepared'), retriever_config=dict(path='retriever.json',
        sha256=hashlib.sha256(run.read_bytes()).hexdigest()),
        groups=dict(train=dict(split='test', ids=['train-id']),
                    validation=dict(split='test', ids=['validation-id']),
                    heldout=dict(split='test', ids=[])),
        metric='ndcg_at_10', top_k=15, allow_json_fence=False,
        partition_label='explicit_compatibility_not_official_train')
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback = dict(kind='retrieval', source_input=dict(path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return source, feedback, adapter


def test_public_retrieval_preparation_and_original_native_scoring(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    _, feedback, adapter = inputs(tmp_path, monkeypatch)
    template = ace()
    del template['examples']
    path = tmp_path / 'prepare.json'
    path.write_text(json.dumps(dict(fit=template, feedback=feedback, timeout_seconds=30)))
    out = tmp_path / 'prepared-fit'
    out.mkdir()
    result = _build(_configuration(path), out)
    config = FitConfig.from_file(result['fit_configuration']['path'])
    assert config.validate()['validated'] and result['model_calls'] == 0
    assert result['scoring_executed'] and result['scoring_scope'] == 'empty_ranking_preflight_only'
    assert 'secret-qrel-doc' not in json.dumps(dict(config.value['examples']['train'][0]['input']))
    processor = create_retrieval_feedback(source_input=feedback['source_input'],
        output_root=tmp_path / 'unused', mode='offline')
    assert processor.score_query('{"query":"alpha"}', 'train-id') == 0.6
    assert adapter.calls[-1] == ('train-id', ['d1', 'd2'])


@pytest.mark.parametrize('change', ['overlap', 'missing-heldout', 'unknown-id', 'metric',
                                  'retriever-hash', 'online', 'unknown-loader', 'unknown-field'])
def test_retrieval_source_rejects_invalid_bindings(tmp_path, monkeypatch, change):
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    source, feedback, _ = inputs(tmp_path, monkeypatch)
    if change == 'overlap':
        source['groups']['heldout']['ids'] = ['train-id']
    elif change == 'missing-heldout':
        del source['groups']['heldout']
    elif change == 'unknown-id':
        source['groups']['train']['ids'] = ['absent']
    elif change == 'metric':
        source['metric'] = 'invented_metric'
    elif change == 'retriever-hash':
        source['retriever_config']['sha256'] = '0' * 64
    elif change == 'unknown-loader':
        source['dataset']['kind'] = 'import_my_factory'
    elif change == 'unknown-field':
        source['factory'] = 'user.py'
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValidationError, ArtifactError)):
        create_retrieval_feedback(source_input=feedback['source_input'],
            output_root=tmp_path / 'unused', mode='online' if change == 'online' else 'offline')


def test_fit_selects_only_builtin_retrieval_factory(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace import ACEOfflineRecipe
    from tests.unit.test_ace_common_support import _artifact

    _, feedback, _ = inputs(tmp_path, monkeypatch)
    value = ace()
    value['feedback'] = feedback
    captured = {}
    def fit(self, *args, **kwargs):
        captured['command'] = self.client.command
        return _artifact()
    monkeypatch.setattr(ACEOfflineRecipe, 'fit', fit)
    FitConfig.from_dict(value).execute(tmp_path / 'fit')
    command = captured['command']
    assert command[command.index('--task-processor-factory') + 1] == (
        'agentinputlab.recipes.ace_retrieval:create_retrieval_feedback')


@pytest.mark.parametrize('kind,class_name', [('trec_dl_sample', 'TRECDLSampleAdapter'),
                                          ('beir_sample', 'BEIRSampleAdapter')])
def test_sample_binding_uses_original_loader_and_locked_manifest(tmp_path, monkeypatch, kind, class_name):
    import agentinputlab.benchmarks as benchmarks
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    source, feedback, adapter = inputs(tmp_path, monkeypatch)
    called = []
    def load(root, **kwargs):
        called.append((root, kwargs))
        return adapter
    monkeypatch.setattr(benchmarks, class_name, load)
    source['dataset'] = dict(kind=kind, benchmark_id=adapter.benchmark_id,
        root='native-sample', manifest_sha256='b' * 64, license_acknowledged=True)
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    processor = create_retrieval_feedback(source_input=feedback['source_input'], output_root=tmp_path)
    assert called == [(tmp_path / 'native-sample', dict(manifest_sha256='b' * 64, license_acknowledged=True))]
    assert processor.adapter is adapter


def captured_inputs(tmp_path, monkeypatch, change=None):
    from agentinputlab.api import _state
    from agentinputlab.contracts.serialization import content_hash, to_jsonable

    source, feedback, adapter = inputs(tmp_path, monkeypatch)
    records = {}
    for example in adapter.cases:
        state = to_jsonable(example.visible_state())
        state['context'] = [dict(block_id='d1', text='saved visible evidence', source='fixture')]
        state['metadata']['initial_capture_retrieval_calls'] = 1
        state['budget']['max_retrieval_calls'] -= 1
        records[example.example_id] = dict(source_state_sha256=example.visible_state().state_hash,
            state=state, state_sha256=_state(state).state_hash)
    first = records['train-id']
    if change == 'query':
        first['state']['query'] = 'different task'
    elif change == 'metadata':
        first['state']['metadata']['invented_gold'] = 'answer'
    elif change == 'source':
        first['source_state_sha256'] = '0' * 64
    elif change == 'budget':
        first['state']['budget']['max_output_tokens'] *= 2
    elif change == 'extra':
        records['unselected-case'] = first
    if change != 'hash':
        first['state_sha256'] = _state(first['state']).state_hash
    else:
        first['state_sha256'] = '0' * 64
    captured = tmp_path / 'captured.json'
    captured.write_text(json.dumps(dict(dataset_sha256=content_hash(adapter.fingerprint()), records=records)))
    source['captured_states'] = dict(path='captured.json', sha256=hashlib.sha256(captured.read_bytes()).hexdigest())
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return feedback, adapter


def test_pinned_captured_context_preserves_native_owned_scorer(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    feedback, original = captured_inputs(tmp_path, monkeypatch)
    processor = create_retrieval_feedback(source_input=feedback['source_input'], output_root=tmp_path)
    assert 'saved visible evidence' in processor.fit_examples('train')[0].input['context']
    assert processor.train[0] is not original.cases[0]
    assert processor.train[0].evaluator_gold is original.cases[0].evaluator_gold
    assert processor.score_query('{"query":"alpha"}', 'train-id') == 0.6
    # Fixture's score asserts exact ownership: the wrapper must use original examples.
    assert original.calls[-1] == ('train-id', ['d1', 'd2'])


@pytest.mark.parametrize('change', ['query', 'metadata', 'source', 'budget', 'extra', 'hash'])
def test_captures_cannot_change_task_source_budget_or_group(tmp_path, monkeypatch, change):
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    feedback, _ = captured_inputs(tmp_path, monkeypatch, change)
    with pytest.raises(ValidationError, match='captur'):
        create_retrieval_feedback(source_input=feedback['source_input'], output_root=tmp_path)


def test_archive_loader_uses_native_locked_archive_without_downloading(tmp_path, monkeypatch):
    from agentinputlab.benchmarks import beir_archive
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    source, feedback, adapter = inputs(tmp_path, monkeypatch)
    called = []
    def load(**kwargs):
        called.append(kwargs)
        return adapter
    monkeypatch.setattr(beir_archive, 'BEIRArchiveAdapter', load)
    source['dataset'] = dict(kind='beir_archive', benchmark_id=adapter.benchmark_id,
                            source_archive='source.zip', license_acknowledged=True)
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    create_retrieval_feedback(source_input=feedback['source_input'], output_root=tmp_path)
    assert called == [dict(benchmark_id=adapter.benchmark_id, source_archive=tmp_path / 'source.zip',
                           license_acknowledged=True)]


def test_separate_heldout_source_preserves_native_ownership(tmp_path, monkeypatch):
    from dataclasses import replace

    import agentinputlab.benchmarks as benchmarks
    from agentinputlab.benchmarks.retrieval import RetrievalPrediction
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    source, feedback, fit = inputs(tmp_path, monkeypatch)
    heldout = Adapter()
    heldout.revision = 'different-heldout-corpus'
    heldout.cases = (replace(heldout.cases[0], example_id='heldout-id', query='heldout query'),)
    heldout.example = lambda split, key: next(e for e in heldout.cases if e.example_id == key)
    monkeypatch.setattr(benchmarks, 'load_production_benchmark_adapter',
                        lambda benchmark, root: heldout if root.name == 'heldout' else fit)
    source['heldout_dataset'] = dict(source['dataset'], prepared_root='heldout')
    source['groups']['heldout']['ids'] = ['heldout-id']
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    processor = create_retrieval_feedback(source_input=feedback['source_input'], output_root=tmp_path)
    assert not heldout.calls  # No heldout feedback during preparation or fitting.
    assert processor.score_query('{"query":"alpha"}', 'train-id') == 0.6
    assert not heldout.calls
    processor.adapter.score(heldout.cases[0], RetrievalPrediction({}))
    assert heldout.calls == [('heldout-id', [])]
    with pytest.raises(ValidationError, match='owned'):
        processor.adapter.score(replace(heldout.cases[0]), RetrievalPrediction({}))
    assert processor.adapter.fingerprint()['heldout_dataset'] == heldout.fingerprint()
    with pytest.raises(ValidationError):
        processor.score_query('{"query":"alpha"}', 'heldout-id')


@pytest.mark.parametrize('change', ['benchmark', 'empty-heldout', 'same-input', 'wrong-owner'])
def test_separate_heldout_source_rejects_role_leakage(tmp_path, monkeypatch, change):
    from dataclasses import replace

    import agentinputlab.benchmarks as benchmarks
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback

    source, feedback, fit = inputs(tmp_path, monkeypatch)
    heldout = Adapter()
    heldout.cases = (replace(heldout.cases[0], example_id='heldout-id',
                            query='heldout query' if change != 'same-input' else fit.cases[0].query),)
    heldout.example = lambda split, key: next(e for e in heldout.cases if e.example_id == key)
    monkeypatch.setattr(benchmarks, 'load_production_benchmark_adapter',
                        lambda benchmark, root: heldout if root.name == 'heldout' else fit)
    source['heldout_dataset'] = dict(source['dataset'], prepared_root='heldout')
    source['groups']['heldout']['ids'] = ['heldout-id']
    if change == 'benchmark':
        heldout.benchmark_id = 'other.benchmark'
        source['heldout_dataset']['benchmark_id'] = heldout.benchmark_id
    elif change == 'empty-heldout':
        source['groups']['heldout']['ids'] = []
    elif change == 'wrong-owner':
        source['groups']['heldout']['ids'] = ['validation-id']
        source['groups']['validation']['ids'] = ['heldout-id']
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValidationError):
        create_retrieval_feedback(source_input=feedback['source_input'], output_root=tmp_path)
    assert not heldout.calls
