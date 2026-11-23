"""Public file binding; synthetic fixtures are not model acceptance."""
import hashlib
import json
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState, RunConfig
from agentinputlab.algorithms.reformulate.infocqr.cache_replay import conversation_view_hash
from agentinputlab.contracts.errors import ArtifactError, ValidationError


def configuration(tmp_path, mode='rw_fsl'):
    state = FrozenAgentState(task_id='case-1', query='Where is it?')
    value = dict(record_id=state.task_id, split='test', mode=mode,
                 source_query=state.query, conversation_hash=conversation_view_hash(state),
                 rewrite='Where is Paris?')
    path = tmp_path / 'cache.jsonl'
    path.write_text(json.dumps(value) + '\n')
    release = 'rewrite.infocqr.cached_' + mode
    config = dict(release_id=release, parameters={'split': 'test'}, artifacts={'cache': {
        'path': 'cache.jsonl', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'artifact_id': 'fixture-cache', 'dataset_revision': 'fixture-v1'}})
    target = tmp_path / 'run.json'
    target.write_text(json.dumps(config))
    return RunConfig.from_file(target), state


@pytest.mark.parametrize('mode', ['rw_zsl', 'rw_fsl', 'edit'])
def test_file_cache_runs_without_runtime_or_backend(tmp_path, monkeypatch, mode):
    config, state = configuration(tmp_path, mode)
    monkeypatch.chdir(tmp_path.parent)
    assert config.validate()['runtime_ready'] is None
    result = config.run(state)
    assert result['state']['query'] == 'Where is Paris?'
    assert result['record']['usage']['model_calls'] == 0
    assert result['scored'] is False


def test_cache_keeps_native_identity_and_miss_gates(tmp_path):
    config, state = configuration(tmp_path)
    with pytest.raises(ArtifactError, match='source query mismatch'):
        config.run(replace(state, query='different'))
    with pytest.raises(ArtifactError, match='no rw_fsl'):
        config.run(replace(state, task_id='unknown'))
    with pytest.raises(ArtifactError, match='conversation view mismatch'):
        config.run(replace(state, metadata={'new_context': 'changed'}))
    (tmp_path / 'cache.jsonl').write_text('{}\n')
    with pytest.raises(ArtifactError, match='checksum mismatch'):
        config.run(state)


@pytest.mark.parametrize('key,value', [('artifact_id', ''), ('dataset_revision', None),
                                     ('factory', 'os.system')])
def test_cache_references_are_explicit_and_allowlisted(tmp_path, key, value):
    config, _ = configuration(tmp_path)
    data = config.to_dict()
    data['artifacts']['cache'][key] = value
    with pytest.raises(ValidationError):
        RunConfig.from_dict(data).validate()


def test_cache_shape_validation_does_not_read_files(tmp_path):
    config, state = configuration(tmp_path)
    (tmp_path / 'cache.jsonl').unlink()
    assert config.validate()['runtime_ready'] is None
    with pytest.raises((ArtifactError, OSError)):
        config.run(state)


def test_missing_cache_identity_is_rejected(tmp_path):
    config, _ = configuration(tmp_path)
    data = config.to_dict()
    del data['artifacts']['cache']['dataset_revision']
    with pytest.raises(ValidationError):
        RunConfig.from_dict(data).validate()
