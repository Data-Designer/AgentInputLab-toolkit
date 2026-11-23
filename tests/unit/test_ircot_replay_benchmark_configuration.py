"""Synthetic entrypoint regression, separate from saved real EEZ acceptance."""
import json

import pytest

from agentinputlab import ExecutionBudget, RunConfig
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts import CapabilityError, ContextBlock, ValidationError
from agentinputlab.contracts.serialization import content_hash
from tests.unit import test_ircot_prediction_replay as replay_fixtures


@pytest.fixture
def bundle(tmp_path):
    return replay_fixtures.bundle.__wrapped__(tmp_path)


class Adapter:
    benchmark_id = 'hotpotqa.distractor'
    pairing_mode = 'stateless_score_twice'

    def examples(self, split):
        assert split == 'validation'
        return (BenchmarkExample(example_id='case', query='SYNTHETIC QUESTION',
            instruction='Original dataset instruction',
            visible_context=(ContextBlock('Visible dataset context', block_id='d'),),
            evaluator_gold={'secret': 'DO_NOT_COPY_GOLD'}),)

    def score(self, example, prediction):
        assert example.instruction == 'Original dataset instruction'
        return {'answer_em': float(prediction.answer == 'SYNTHETIC PREDICTION')}


def config(bundle):
    root, _, provider, _, _, _, build = bundle
    method = build()
    return RunConfig('acquire.ircot.prediction_replay', parameters={
        'manifest_path': str(root/'manifest.json'),
        'expected_manifest_sha256': method.expected_manifest_sha256,
        'expected_generator_fingerprint_sha256': content_hash(provider.fingerprint()),
        'benchmark': 'hotpotqa.distractor', 'split': 'validation'})


def predictor(projection=True):
    return {'schema_version': 1, 'qa': {'output_format': 'answer_only'},
        **({'archived_input': {'projection': 'ircot_question_only.v1'}} if projection else {})}


def evaluate(run, **kwargs):
    return run.evaluate('hotpotqa.distractor', split='validation', example_ids=['case'],
        adapter=Adapter(), predictor_parameters=kwargs.pop('predictor_parameters', predictor()), **kwargs)


def test_captured_question_only_score_needs_no_model_backend(bundle):
    run = config(bundle)
    result = evaluate(run)
    row, = result['records']
    assert result['scored_count'] == 1 and row['evaluation']['metrics']['answer_em'] == 1
    assert row['usage']['model_calls'] == 0 and row['model_responses'] == []
    assert row['input_state_hash'] == bundle[1].state_hash
    assert row['source_input_state_hash'] != row['input_state_hash']
    assert row['archived_input']['original_state_hash_verified']
    assert 'DO_NOT_COPY_GOLD' not in json.dumps(row['intervention'])
    assert row['intervention']['result']['metadata']['historical_usage']['model_calls'] == 2
    assert not bundle[2].calls


def test_projection_is_not_implicit(bundle):
    result = evaluate(config(bundle), predictor_parameters=predictor(False))
    assert result['failed_count'] == 1
    assert result['records'][0]['failure']['type'] == 'ArtifactError'
    assert result['records'][0]['failure']['stage'] == 'intervention'


@pytest.mark.parametrize('fault', ['budget', 'query', 'hash', 'missing'])
def test_original_capture_identity_cannot_be_relaxed(bundle, fault):
    from dataclasses import replace
    if fault == 'budget':
        run = replace(config(bundle), budget=ExecutionBudget(max_model_calls=7))
    else:
        inputs = bundle[4]
        if fault == 'query':
            inputs['case']['query'] = 'Different source question'
        elif fault == 'hash':
            inputs['case']['input_state_hash'] = 'a' * 64
        else:
            inputs['case']['input_state_hash'] = None
        run = config(bundle)
    result = evaluate(run)
    assert result['failed_count'] == 1
    row = result['records'][0]
    assert row['failure']['stage'] == 'archived_input'
    assert row['usage']['model_calls'] == 0 and 'evaluation' not in row


@pytest.mark.parametrize('value', [None, {}, 'ircot_question_only.v1',
    {'projection': 'anything'}, {'projection': 'ircot_question_only.v1', 'ignore_hash': True}])
def test_projection_configuration_rejects_unknown_fields(bundle, value):
    params = predictor()
    params['archived_input'] = value
    with pytest.raises(ValidationError):
        evaluate(config(bundle), predictor_parameters=params)


def test_author_archive_no_hash_is_not_controlled_capture(bundle):
    bundle[5]['origin_kind'] = 'author_release'
    bundle[4]['case'].update(input_state_hash=None, historical_usage=None, origin_receipt_sha256=None)
    run = config(bundle)
    with pytest.raises(ValidationError, match='controlled capture'):
        evaluate(run)
    # No input projection is needed for a genuine source archive with query binding only.
    result = evaluate(run, predictor_parameters=predictor(False))
    assert result['scored_count'] == 1
    metadata = result['records'][0]['intervention']['result']['metadata']
    assert metadata['historical_usage_unknown'] and metadata['historical_usage'] is None


def test_other_profile_cannot_use_projection():
    with pytest.raises(ValidationError, match='matching IRCoT'):
        evaluate(RunConfig('baseline.no_op'))


def test_public_cli_scores_saved_capture_without_backend(bundle, monkeypatch, capsys):
    from agentinputlab.cli.main import main
    root = bundle[0]
    run = config(bundle)
    (root/'run.json').write_text(json.dumps(run.to_dict()))
    (root/'predictor.json').write_text(json.dumps(predictor()))
    monkeypatch.setattr('agentinputlab.benchmarks.load_production_benchmark_adapter', lambda *a, **k: Adapter())
    assert main(['config','evaluate',str(root/'run.json'),'--benchmark','hotpotqa.distractor',
        '--split','validation','--case-id','case','--prepared-root',str(root),
        '--predictor-config',str(root/'predictor.json'),'--offline', '--output', str(root/'results.jsonl')]) == 0
    assert json.loads(capsys.readouterr().out)['scored_count'] == 1


def test_score_only_builder_never_generates_or_changes_default_requirements():
    from agentinputlab import FrozenAgentState
    from agentinputlab.experiments.drivers.static_production import (
        StaticPredictorBuildRequest,
        StaticRuntimeComponents,
    )
    from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
    request = StaticPredictorBuildRequest(benchmark_id='hotpotqa.distractor', split='validation',
        model_slot='configured', seed=17, builder_parameters={'schema_version': 1},
        runtime=StaticRuntimeComponents(backend=None, score_only=True))
    builder = ProductionPredictorBuilder()
    with pytest.raises(CapabilityError, match='LLMBackend'):
        builder.build(request)
    score_only = builder.build(request, prediction_required=False)
    assert score_only.fingerprint()['prediction_enabled'] is False
    with pytest.raises(CapabilityError, match='score-only'):
        score_only.predict(FrozenAgentState(query='question'))
    with pytest.raises(ValidationError, match='boolean'):
        builder.build(request, prediction_required=0)


@pytest.fixture
def longbench_capture(bundle, tmp_path, monkeypatch):
    from dataclasses import replace
    from agentinputlab import FrozenAgentState
    from tests.unit.test_longbench_adapter import fixture_adapter

    adapter = fixture_adapter(tmp_path, monkeypatch)
    example = replace(adapter.examples('test')[0], instruction='Retain the original task instruction.')
    adapter._examples = (example,)
    adapter._by_id = {example.example_id: example}
    marker = '[ORIGINAL_EXTERNAL_CONTEXT]'
    state = FrozenAgentState(query=adapter.template.format(context=marker, input=example.query) + '\n\n' + example.instruction,
        task_id=example.example_id)
    inputs, manifest = bundle[4:6]
    original = inputs.pop('case')
    inputs[example.example_id] = dict(original, query=state.query, input_state_hash=state.state_hash)
    bundle[3][example.example_id] = bundle[3].pop('case')
    manifest.update(benchmark='longbench.qasper', split='test')
    manifest['inputs']['sha256'] = replay_fixtures.save(tmp_path/'inputs.json', inputs)
    manifest['predictions']['sha256'] = replay_fixtures.save(tmp_path/'predictions.json', bundle[3])
    pin = replay_fixtures.save(tmp_path/'manifest.json', manifest)
    run = RunConfig('acquire.ircot.prediction_replay', parameters={
        'manifest_path': str(tmp_path/'manifest.json'), 'expected_manifest_sha256': pin,
        'expected_generator_fingerprint_sha256': content_hash(bundle[2].fingerprint()),
        'benchmark': adapter.benchmark_id, 'split': 'test'})
    options = dict(projection='ircot_longbench_task.v1', context_marker=marker)
    return run, adapter, options, state


def evaluate_longbench(capture):
    run, adapter, options, _ = capture
    return run.evaluate(adapter.benchmark_id, split='test', adapter=adapter,
        example_ids=[adapter._examples[0].example_id],
        predictor_parameters={'schema_version': 1, 'archived_input': options})


def test_longbench_native_template_projection_scores_without_generation(longbench_capture):
    result = evaluate_longbench(longbench_capture)
    row, = result['records']
    assert result['scored_count'] == 1
    assert row['evaluation']['metrics'] == {'answer_f1': 0.0}  # Preserve wrong saved answers.
    assert row['input_state_hash'] == longbench_capture[3].state_hash
    assert row['archived_input']['projection'] == 'ircot_longbench_task.v1'
    assert row['model_responses'] == [] and row['usage']['model_calls'] == 0
    assert 'gold-only-answer' not in json.dumps(row['intervention'])
    assert row['intervention']['result']['metadata']['historical_usage']['model_calls'] == 2


@pytest.mark.parametrize('fault', ['marker', 'template', 'budget', 'instruction', 'query', 'ambiguous_marker', 'render'])
def test_longbench_projection_preserves_capture_and_native_prompt_identity(longbench_capture, fault, monkeypatch):
    from dataclasses import replace
    run, adapter, options, state = longbench_capture
    if fault == 'marker':
        options['context_marker'] = 'DIFFERENT_MARKER'
    elif fault == 'template':
        adapter.template += '\nDifferent instruction'
    elif fault == 'budget':
        run = replace(run, budget=ExecutionBudget(max_model_calls=7))
    elif fault in ('instruction', 'query', 'ambiguous_marker'):
        example = adapter._examples[0]
        field = 'instruction' if fault == 'instruction' else 'query'
        replacement = options['context_marker'] if fault == 'ambiguous_marker' else 'Changed source'
        adapter._examples = (replace(example, **{field: replacement}),)
        adapter._by_id = {example.example_id: adapter._examples[0]}
    else:
        monkeypatch.setattr(adapter, 'render', lambda s: 'Not the original prompt')
    result = evaluate_longbench((run, adapter, options, state))
    row, = result['records']
    assert result['failed_count'] == 1 and row['failure']['stage'] == 'archived_input'
    assert row['usage']['model_calls'] == 0 and 'evaluation' not in row


@pytest.mark.parametrize('marker', ['', '  ', None, [], 17, 'x' * 4097])
def test_longbench_invalid_marker_rejected(longbench_capture, marker):
    longbench_capture[2]['context_marker'] = marker
    with pytest.raises(ValidationError):
        evaluate_longbench(longbench_capture)


def test_longbench_unknown_projection_fields_rejected(longbench_capture):
    longbench_capture[2]['ignore_hash'] = True
    with pytest.raises(ValidationError):
        evaluate_longbench(longbench_capture)


def test_longbench_projection_rejects_non_native_adapter(longbench_capture):
    from agentinputlab.ircot_replay_configuration import prepare
    from agentinputlab.api import create_intervention
    from agentinputlab.contracts import ArtifactError

    run, adapter, options, _ = longbench_capture
    method = create_intervention(run.release_id, **run.parameters)
    with pytest.raises(ArtifactError, match='native adapter'):
        prepare(adapter._examples[0].visible_state(), method, options=options, adapter=Adapter())


def test_longbench_option_not_allowed_on_qa(bundle):
    params = predictor()
    params['archived_input'] = dict(projection='ircot_longbench_task.v1', context_marker='marker')
    with pytest.raises(ValidationError, match='LongBench'):
        evaluate(config(bundle), predictor_parameters=params)


def test_longbench_projection_public_cli(longbench_capture, monkeypatch, tmp_path, capsys):
    from agentinputlab.cli.main import main
    run, adapter, options, _ = longbench_capture
    (tmp_path/'run.json').write_text(json.dumps(run.to_dict()))
    (tmp_path/'predictor.json').write_text(json.dumps({'schema_version': 1, 'archived_input': options}))
    monkeypatch.setattr('agentinputlab.benchmarks.load_benchmark', lambda *a, **k: adapter)
    assert main(['config','evaluate',str(tmp_path/'run.json'),'--benchmark',adapter.benchmark_id,
        '--split','test','--case-id',adapter._examples[0].example_id,
        '--predictor-config',str(tmp_path/'predictor.json'), '--offline',
        '--output',str(tmp_path/'records.jsonl')]) == 0
    assert json.loads(capsys.readouterr().out)['scored_count'] == 1
