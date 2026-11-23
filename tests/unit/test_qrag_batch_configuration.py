"""Multiple visible states must not share a stale Q-RAG candidate identity."""
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from tests.unit.test_benchmark_configuration import Adapter
from tests.unit.test_qrag_file_configuration import wired  # noqa: F401


class VisibleAdapter(Adapter):
    def examples(self, split):
        from agentinputlab import ContextBlock
        from agentinputlab.benchmarks import BenchmarkExample
        return tuple(BenchmarkExample(example_id=i, query="Visible question " + i,
            visible_context=(ContextBlock(block_id=i + "-d", text="document " + i),),
            evaluator_gold={"secret": "NEVER_SEND_GOLD"}) for i in self.ids)


@pytest.mark.parametrize("second_answer", ["wrong", ""])
def test_batch_derives_each_identity_and_keeps_failed_cases(wired, monkeypatch, second_answer):  # noqa: F811
    value, backend, selections, invocations = wired
    config = RunConfig.from_dict(value)
    import json
    from dataclasses import asdict

    from agentinputlab import qrag_configuration
    from agentinputlab.algorithms.acquire.qrag.bridge import QRAGArtifactIdentity
    # Fixture asset checks only; all case binding, original driver and scoring run.
    identity = QRAGArtifactIdentity(**dict(config.parameters['artifacts']))
    monkeypatch.setattr(qrag_configuration, '_read_batch_assets', lambda *_: asdict(identity), raising=False)
    parameters = dict(config.parameters)
    parameters.pop('artifacts')
    config = replace(config, parameters=parameters)
    backend.responses = [('answer', 3), (second_answer, 2)]
    result = config.evaluate('2wiki.multihop', split='dev', example_ids=['a', 'b'], adapter=VisibleAdapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}},
        qrag_reader={'repository': 'fixture', 'revision': 'fixture', 'profile': 'factorial_reader_transfer'})
    assert result['selected_count'] == 2
    assert result['failed_count'] == int(not second_answer)
    assert result['scored_count'] == 2 - int(not second_answer)
    assert len(invocations) == len(backend.requests) == 2
    for row, (_, invocation) in zip(result['records'], invocations, strict=False):
        bound = invocation['configuration']
        assert bound['runtime']['prepared_state_sha256'] == row['input_state_hash']
        assert row['qrag_binding']['artifact_identity_sha256'] == content_hash(bound['parameters']['artifacts'])
        assert 'NEVER_SEND_GOLD' not in json.dumps(bound)
        assert row['usage']['model_calls'] == 1
    assert invocations[0][1]['configuration']['runtime']['prepared_state_sha256'] != invocations[1][1]['configuration']['runtime']['prepared_state_sha256']
    assert invocations[0][1]['configuration']['parameters']['artifacts']['corpus_sha256'] != invocations[1][1]['configuration']['parameters']['artifacts']['corpus_sha256']


@pytest.mark.parametrize('reader', [{}, {'repository': 'x', 'revision': 'r', 'profile': 'unknown'},
                                  {'repository': 'x', 'revision': 'r', 'profile': 'factorial_reader_transfer', 'extra': True}])
def test_batch_rejects_invalid_reader_before_dataset_or_model(wired, reader):  # noqa: F811
    value, backend, _, invocations = wired
    config = RunConfig.from_dict(value)
    with pytest.raises(ValidationError):
        config.evaluate('2wiki.multihop', split='dev', limit=2, adapter=Adapter(), qrag_reader=reader)
    assert not invocations and not backend.requests


def test_explicit_batch_replay_miss_does_not_generate(wired, monkeypatch, tmp_path):  # noqa: F811
    import hashlib
    import json

    from agentinputlab import qrag_configuration
    from tests.unit.test_qrag_file_configuration import policy_response

    value, backend, selections, invocations = wired
    original = value['parameters'].pop('artifacts')
    config = RunConfig.from_dict(value)
    monkeypatch.setattr(qrag_configuration, '_read_batch_assets', lambda *_: original)
    reader = {'repository': 'fixture', 'revision': 'fixture', 'profile': 'factorial_reader_transfer'}
    state = VisibleAdapter().examples('dev')[0].visible_state(budget=config.budget)
    bound = qrag_configuration.QRAGBatchBinder(config, reader).bind(state)
    candidates = [{'document_id': b.block_id, 'text': b.text} for b in state.context]
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'schema': 'agentinputlab.qrag.policy-replay.v1',
        'request': {'question': state.query, 'candidates': candidates, 'seed': config.seed, 'max_steps': 2},
        'artifact_identity_sha256': content_hash(bound.parameters['artifacts']),
        'response': policy_response(candidates)}))
    reader['policy_replays'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
    backend.responses = [('answer', 3)]
    result = config.evaluate('2wiki.multihop', split='dev', example_ids=['a', 'b'], adapter=VisibleAdapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}}, qrag_reader=reader)
    assert result['selected_count'] == 2 and result['scored_count'] == result['failed_count'] == 1
    assert len(invocations) == len(backend.requests) == 1 and not selections
    assert result['records'][1]['usage']['model_calls'] == result['records'][1]['usage']['embedding_calls'] == 0
    assert not result['records'][1]['model_responses']
    with pytest.raises(ValidationError, match='duplicate'):
        qrag_configuration.QRAGBatchBinder(config, {**reader, 'policy_replays': reader['policy_replays'] * 2})


def test_cli_resolves_batch_replay_paths_relative_to_reader_file(tmp_path, monkeypatch):
    import json

    from agentinputlab.cli.main import main
    from tests.unit.test_qrag_file_configuration import configuration

    (tmp_path / 'run.json').write_text(json.dumps(configuration()))
    (tmp_path / 'reader.json').write_text(json.dumps({'repository': 'fixture', 'revision': 'fixture',
        'profile': 'factorial_reader_transfer', 'policy_replays': [{'path': 'policy.json', 'sha256': 'a' * 64}]}))
    observed = []
    def evaluate(self, benchmark, **options):
        observed.append(options['qrag_reader'])
        return {'records': [], 'failed_count': 0, 'awaiting_clarification_count': 0}
    monkeypatch.setattr(RunConfig, 'evaluate', evaluate)
    monkeypatch.chdir(tmp_path.parent)
    assert main(['config', 'evaluate', str(tmp_path / 'run.json'), '--benchmark', '2wiki.multihop',
        '--split', 'dev', '--limit', '2', '--qrag-reader-config', str(tmp_path / 'reader.json'),
        '--output', str(tmp_path / 'out.jsonl')]) == 0
    assert observed[0]['policy_replays'][0]['path'] == str(tmp_path / 'policy.json')


def candidate_config(wired, monkeypatch):  # noqa: F811
    from agentinputlab import qrag_configuration

    value, backend, selections, invocations = wired
    identity = value['parameters'].pop('artifacts')
    monkeypatch.setattr(qrag_configuration, '_read_batch_assets', lambda *_: identity)
    value['retriever'] = {'kind': 'bm25', 'documents': [
        {'doc_id': 'd', 'text': 'Visible question evidence for both cases'}]}
    reader = {'repository': 'fixture', 'revision': 'fixture', 'profile': 'factorial_reader_transfer',
              'candidate_source': {'kind': 'retriever', 'top_k': 1, 'max_document_chars': 20}}
    return RunConfig.from_dict(value), reader, backend, selections, invocations


def test_explicit_candidate_retrieval_binds_each_state_and_charges_calls(wired, monkeypatch):  # noqa: F811
    import json

    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    backend.responses = [('answer', 3), ('answer', 3)]
    result = config.evaluate('2wiki.multihop', split='dev', example_ids=['a', 'b'], adapter=Adapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}}, qrag_reader=reader)
    assert result['scored_count'] == 2, result['records']
    assert len(backend.requests) == len(invocations) == len(selections) == 2
    for row in result['records']:
        assert row['usage']['retrieval_calls'] == row['usage']['model_calls'] == 1
        prep = row['candidate_preparation']
        assert prep['candidate_ids'] == ['d'] and prep['clipped_documents'] == 1
        assert prep['execution_state_sha256'] == row['input_state_hash']
        assert prep['source_state_sha256'] == row['source_input_state_hash']
        assert row['input_state_hash'] != row['source_input_state_hash']
        assert row['qrag_binding']['mode'] == 'retrieved_visible_candidates'
    assert all(r.candidates[0].document.text == 'Visible question evi' for r in selections)
    assert 'NEVER_SEND_GOLD' not in json.dumps([r.prompt for r in backend.requests])


def test_candidate_retrieval_does_not_replace_visible_passages(wired, monkeypatch):  # noqa: F811
    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    result = config.evaluate('2wiki.multihop', split='dev', limit=1, adapter=VisibleAdapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}}, qrag_reader=reader)
    assert result['failed_count'] == 1
    assert result['records'][0]['failure']['stage'] == 'candidate_retrieval'
    assert result['records'][0]['usage']['retrieval_calls'] == 0
    assert not backend.requests and not selections and not invocations


@pytest.mark.parametrize('document_format', ['text', 'title_text'])
def test_candidate_document_format_is_explicit_and_preserves_policy_text(wired, monkeypatch, document_format):  # noqa: F811
    from agentinputlab import qrag_configuration
    from agentinputlab.algorithms.acquire.qrag.common_support import _policy_candidate_text
    from agentinputlab.backends.retrieval import Document, RetrievalResult

    config, reader, _, _, _ = candidate_config(wired, monkeypatch)
    reader['candidate_source'].update(document_format=document_format, max_document_chars=1000)
    item = RetrievalResult(Document('d', '  Original passage body  ', title='  Original title  '), 1.0, 1)
    class Source:
        def retrieve(self, query, *, top_k):
            return (item,)
        def fingerprint(self):
            return {'source': 'fixture'}
    binder = qrag_configuration.QRAGBatchBinder(config, reader)
    state = Adapter().examples('dev')[0].visible_state(budget=config.budget)
    prepared, receipt = binder.prepare_candidates(state, Source())
    expected = _policy_candidate_text(item) if document_format == 'title_text' else item.document.text
    assert prepared.context[0].text == expected
    assert content_hash([prepared.context[0].text.strip()]) == content_hash([expected.strip()])
    assert receipt['configuration']['document_format'] == document_format
    assert receipt['clipped_documents'] == 0
    assert not state.context


def test_candidate_title_prefix_counts_toward_declared_character_limit(wired, monkeypatch):  # noqa: F811
    from agentinputlab import qrag_configuration
    from agentinputlab.backends.retrieval import Document, RetrievalResult

    config, reader, _, _, _ = candidate_config(wired, monkeypatch)
    reader['candidate_source'].update(document_format='title_text', max_document_chars=12)
    class Source:
        def retrieve(self, query, *, top_k):
            return (RetrievalResult(Document('d', 'body', title='A long title'), 1.0, 1),)
        def fingerprint(self):
            return {'source': 'fixture'}
    state = Adapter().examples('dev')[0].visible_state(budget=config.budget)
    prepared, receipt = qrag_configuration.QRAGBatchBinder(config, reader).prepare_candidates(state, Source())
    assert prepared.context[0].text == 'Title: A lon'
    assert receipt['clipped_documents'] == 1


@pytest.mark.parametrize('source', [None, {}, {'kind': 'retriever', 'top_k': True, 'max_document_chars': 5},
    {'kind': 'retriever', 'top_k': 1, 'max_document_chars': 0},
    {'kind': 'retriever', 'top_k': 1, 'max_document_chars': 20, 'document_format': 'guess'},
    {'kind': 'retriever', 'top_k': 1000000, 'max_document_chars': 20}])
def test_invalid_candidate_source_fails_before_model(wired, monkeypatch, source):  # noqa: F811
    from agentinputlab.qrag_configuration import QRAGBatchBinder

    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    with pytest.raises(ValidationError):
        QRAGBatchBinder(config, {**reader, 'candidate_source': source})
    assert not backend.requests and not selections and not invocations


def test_missing_candidate_retriever_rejected_before_dataset(wired, monkeypatch):  # noqa: F811
    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    with pytest.raises(ValidationError, match='configured corpus retriever'):
        replace(config, retriever={}).evaluate('2wiki.multihop', split='dev', limit=1,
            adapter=Adapter(), qrag_reader=reader)
    assert not backend.requests and not selections and not invocations


def test_candidate_preparation_respects_retrieval_budget_without_retry(wired, monkeypatch):  # noqa: F811
    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    config = replace(config, budget=replace(config.budget, max_retrieval_calls=0))
    result = config.evaluate('2wiki.multihop', split='dev', limit=1, adapter=Adapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}}, qrag_reader=reader)
    assert result['failed_count'] == 1
    assert result['records'][0]['failure']['type'] == 'BudgetExceededError'
    assert result['records'][0]['usage']['retrieval_calls'] == 0
    assert not backend.requests and not selections and not invocations


def test_candidate_preparation_keeps_cost_when_reader_fails(wired, monkeypatch):  # noqa: F811
    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    backend.responses = [('', 17)]
    result = config.evaluate('2wiki.multihop', split='dev', limit=1, adapter=Adapter(),
        predictor_parameters={'schema_version': 1, 'qa': {'output_format': 'answer_only'}}, qrag_reader=reader)
    assert result['failed_count'] == 1 and result['scored_count'] == 0
    assert result['records'][0]['usage']['retrieval_calls'] == 1
    assert result['records'][0]['usage']['output_tokens'] == 17
    assert len(backend.requests) == len(selections) == len(invocations) == 1


@pytest.mark.parametrize('retrieval_limit,scored', [(2, True), (1, False)])
def test_query_batch_prepares_and_scores_from_one_corpus_with_shared_budget(wired, monkeypatch, retrieval_limit, scored):  # noqa: F811
    from agentinputlab import qrag_configuration
    from agentinputlab.algorithms.acquire.qrag import (
        QRAG_QUERY_PROMPT_SHA256,
        QRAG_QUERY_READER_PROFILE,
    )
    from tests.unit.test_composite_ir_evaluate_configuration import IRAdapter, options

    value, _, _, _ = wired
    identity = dict(value['parameters']['artifacts'])
    config, reader, backend, selections, invocations = candidate_config(wired, monkeypatch)
    identity.update(reader_profile=QRAG_QUERY_READER_PROFILE, reader_prompt_sha256=QRAG_QUERY_PROMPT_SHA256)
    monkeypatch.setattr(qrag_configuration, '_read_batch_assets', lambda *_: identity)
    reader['profile'] = QRAG_QUERY_READER_PROFILE
    config = replace(config, budget=replace(config.budget, max_retrieval_calls=retrieval_limit))
    backend.responses = [('{"query":"Visible question"}', 7)]
    result = config.evaluate('beir.arguana', split='test', limit=1, adapter=IRAdapter(),
        predictor_parameters=options(), qrag_reader=reader)
    assert result['scored_count'] == int(scored) and result['failed_count'] == int(not scored)
    assert result['records'][0]['usage']['retrieval_calls'] == retrieval_limit
    assert result['records'][0]['usage']['model_calls'] == 1
    assert result['records'][0]['usage']['output_tokens'] == 7
    assert len(backend.requests) == len(selections) == len(invocations) == 1
    if not scored:
        # Predictor refuses its zero remaining retrieval allowance before I/O.
        assert result['records'][0]['failure']['type'] == 'CapabilityError'
