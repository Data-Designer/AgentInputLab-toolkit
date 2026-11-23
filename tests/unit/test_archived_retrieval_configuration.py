"""Synthetic contract tests, not acceptance of real model/dataset combinations."""

from dataclasses import replace
import json

import pytest

from agentinputlab import ExecutionBudget, RunConfig
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from tests.unit.test_ircot_prediction_replay import bundle as replay_bundle, save


class Adapter:
    benchmark_id = 'beir.arguana'
    pairing_mode = 'stateless_score_twice'

    def fingerprint(self):
        return {'source': 'synthetic-corpus-v1', 'scorer': 'synthetic-rank-v1'}

    def examples(self, split):
        assert split == 'test'
        return (BenchmarkExample(example_id='case', query='SYNTHETIC QUESTION',
            instruction='Retrieve original documents', evaluator_gold={'relevant': 'good', 'secret': 'GOLD_SECRET'}),)

    def score(self, example, prediction):
        # A rank-sensitive synthetic scorer; never invoked by the algorithm.
        expected = example.evaluator_gold['relevant']
        ids = sorted(prediction.scores, key=prediction.scores.get, reverse=True)
        return {'reciprocal_rank': 1 / (ids.index(expected) + 1) if expected in ids else 0.0}


@pytest.fixture
def archive(tmp_path):
    bundle = replay_bundle.__wrapped__(tmp_path)
    root, state, provider, predictions, inputs, manifest, _ = bundle
    manifest.update(benchmark='beir.arguana', split='test')
    manifest_pin = save(root/'manifest.json', manifest)
    run = RunConfig('acquire.ircot.prediction_replay', parameters={
        'manifest_path': str(root/'manifest.json'), 'expected_manifest_sha256': manifest_pin,
        'expected_generator_fingerprint_sha256': content_hash(provider.fingerprint()),
        'benchmark': 'beir.arguana', 'split': 'test'})
    receipt = dict(schema='agentinputlab.ircot.archived_retrieval.v1', benchmark='beir.arguana', split='test',
        prediction_manifest_sha256=manifest_pin, generator_fingerprint_sha256=content_hash(provider.fingerprint()),
        dataset_fingerprint_sha256=content_hash(Adapter().fingerprint()),
        records={'case': dict(query=state.query, input_state_hash=state.state_hash,
            origin_receipt_sha256=inputs['case']['origin_receipt_sha256'],
            answer_sha256=content_hash(predictions['case']), ranked_ids=['wrong', 'good'])})
    return root, run, receipt, provider, bundle


def predictor(archive):
    root, _, receipt, _, _ = archive
    return dict(schema_version=1, archived_input={'projection': 'ircot_question_only.v1'},
        archived_retrieval=dict(path=str(root/'rank.json'), sha256=save(root/'rank.json', receipt)))


def evaluate(archive, **kwargs):
    options = kwargs.pop('options') if 'options' in kwargs else predictor(archive)
    return kwargs.pop('run', archive[1]).evaluate('beir.arguana', split='test', example_ids=['case'],
        adapter=kwargs.pop('adapter', Adapter()), predictor_parameters=options, **kwargs)


def test_original_rank_scored_without_query_generation_or_retrieval(archive):
    result = evaluate(archive)
    row, = result['records']
    assert result['scored_count'] == 1 and row['evaluation']['metrics'] == {'reciprocal_rank': 0.5}
    assert row['usage']['model_calls'] == row['usage']['retrieval_calls'] == 0
    assert row['intervention']['result']['evidence'] == []
    assert row['intervention']['result']['final_answer'] == 'SYNTHETIC PREDICTION'
    assert row['downstream_metadata']['original_ranked_ids'] == ['wrong', 'good']
    assert not row['downstream_metadata']['answer_used_as_query_or_evidence']
    assert not archive[3].calls and 'GOLD_SECRET' not in json.dumps(row['intervention'])
    assert row['intervention']['result']['metadata']['historical_usage']['model_calls'] == 2


@pytest.mark.parametrize('field', ['schema', 'benchmark', 'split', 'prediction_manifest_sha256', 'generator_fingerprint_sha256'])
def test_scope_binding_rejected(archive, field):
    archive[2][field] = 'wrong'
    with pytest.raises(ArtifactError):
        evaluate(archive)


@pytest.mark.parametrize('field', ['query', 'input_state_hash', 'origin_receipt_sha256', 'answer_sha256'])
def test_case_identity_rejected(archive, field):
    archive[2]['records']['case'][field] = 'wrong'
    with pytest.raises(ArtifactError):
        evaluate(archive)


@pytest.mark.parametrize('ids', [['a','a'], 'a', [1], [''], [' '], ['x']*10001])
def test_invalid_ranking_rejected(archive, ids):
    archive[2]['records']['case']['ranked_ids'] = ids
    with pytest.raises(ArtifactError):
        evaluate(archive)


def test_empty_original_rank_is_honest_zero(archive):
    archive[2]['records']['case']['ranked_ids'] = []
    assert evaluate(archive)['records'][0]['evaluation']['metrics'] == {'reciprocal_rank': 0.0}


@pytest.mark.parametrize('level', ['top', 'case'])
def test_no_scores_or_gold_accepted_in_receipt(archive, level):
    (archive[2] if level == 'top' else archive[2]['records']['case'])['metrics'] = {'score': 1}
    with pytest.raises(ArtifactError):
        evaluate(archive)


def test_changed_dataset_or_scorer_fails_before_scoring(archive):
    archive[2]['dataset_fingerprint_sha256'] = 'a' * 64
    row, = evaluate(archive)['records']
    assert row['status'] == 'failed' and row['failure']['stage'] == 'downstream'
    assert 'evaluation' not in row and row['usage']['model_calls'] == 0


def test_changed_budget_fails_before_replay(archive):
    row, = evaluate(archive, run=replace(archive[1], budget=ExecutionBudget(max_model_calls=7)))['records']
    assert row['status'] == 'failed' and row['failure']['stage'] == 'archived_input'


def test_answer_only_archive_is_not_a_retrieval_run_without_opt_in(archive):
    options = predictor(archive)
    options.pop('archived_retrieval')
    with pytest.raises(ValidationError, match='QA prediction/scorer'):
        evaluate(archive, options=options)


@pytest.mark.parametrize('options', [None, {}, {'path':'relative.json','sha256':'a'*64}, {'path':'/tmp/x','sha256':'a'*64,'ignore_hash':True}])
def test_invalid_options_rejected(archive, options):
    with pytest.raises(ValidationError):
        evaluate(archive, options={'schema_version':1, 'archived_retrieval':options})


def test_tampered_file_rejected(archive):
    options = predictor(archive)
    (archive[0]/'rank.json').write_text('{}')
    with pytest.raises(ArtifactError):
        evaluate(archive, options=options)


def test_empty_answer_does_not_turn_into_success_via_saved_ranks(archive):
    root, run, receipt, provider, bundle = archive
    bundle[3]['case'] = ''
    bundle[5]['predictions']['sha256'] = save(root/'predictions.json', bundle[3])
    pin = save(root/'manifest.json', bundle[5])
    run = replace(run, parameters={**run.parameters, 'expected_manifest_sha256':pin})
    receipt['prediction_manifest_sha256'] = pin
    receipt['records']['case']['answer_sha256'] = content_hash('')
    row, = evaluate(archive, run=run)['records']
    assert row['status'] == 'failed' and row['failure']['type'] == 'AlgorithmOutputError'
    assert 'evaluation' not in row and row['usage']['model_calls'] == 0


@pytest.mark.parametrize('relative', [False, True])
def test_public_cli_archived_ranking(archive, monkeypatch, capsys, relative):
    from agentinputlab.cli.main import main
    root, run, _, _, _ = archive
    (root/'run.json').write_text(json.dumps(run.to_dict()))
    settings = predictor(archive)
    if relative:
        settings['archived_retrieval']['path'] = 'rank.json'
    (root/'predictor.json').write_text(json.dumps(settings))
    monkeypatch.chdir(root.parent)
    monkeypatch.setattr('agentinputlab.benchmarks.load_production_benchmark_adapter', lambda *a, **k: Adapter())
    assert main(['config','evaluate',str(root/'run.json'),'--benchmark','beir.arguana',
        '--split','test','--case-id','case','--prepared-root',str(root), '--predictor-config',str(root/'predictor.json'),
        '--offline','--output',str(root/'records.jsonl')]) == 0
    assert json.loads(capsys.readouterr().out)['scored_count'] == 1


def test_score_only_retrieval_does_not_enable_prediction():
    from agentinputlab import FrozenAgentState
    from agentinputlab.contracts import CapabilityError
    from agentinputlab.experiments.drivers.static_production import StaticPredictorBuildRequest, StaticRuntimeComponents
    from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
    request = StaticPredictorBuildRequest(benchmark_id='beir.arguana', split='test', model_slot='configured', seed=17,
        builder_parameters={'schema_version':1}, runtime=StaticRuntimeComponents(backend=None, score_only=True))
    builder = ProductionPredictorBuilder()
    with pytest.raises(CapabilityError, match='retriever'):
        builder.build(request)
    with pytest.raises(CapabilityError, match='score-only'):
        builder.build(request, prediction_required=False).predict(FrozenAgentState(query='query'))
