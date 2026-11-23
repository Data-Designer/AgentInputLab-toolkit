"""Public BFCL fit binding; fixture scorers are not real model evidence."""
import hashlib
import json

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.fit_configuration import FitConfig
from agentinputlab.fit_preparation import _build, _configuration
from tests.unit.test_ace_bfcl_feedback import Adapter
from tests.unit.test_fit_configuration import ace


def inputs(tmp_path, monkeypatch):
    import agentinputlab.benchmarks as benchmarks

    adapter = Adapter()
    adapter.example = lambda split, key: next(e for e in adapter.cases if e.example_id == key)
    monkeypatch.setattr(benchmarks, 'SourceVerifiedBFCLV3ASTBridge', lambda **kw: kw)
    monkeypatch.setattr(benchmarks, 'load_bfcl_v3_simple_adapter', lambda *a, **kw: adapter)
    source = dict(source_archive='official.tar.gz', normalized_root='normalized',
                  prepared_file_hashes=dict(adapter.prepared.file_sha256),
                  train_ids=['case-0'], validation_ids=['case-1'], heldout_ids=['case-2'])
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback = dict(kind='bfcl', source_input=dict(path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()), model_id='m',
        partition_label='caller_declared_compatibility_partitions')
    return source, feedback, adapter


def test_bfcl_public_prepare_file_uses_original_ast_feedback(tmp_path, monkeypatch):
    _, feedback, adapter = inputs(tmp_path, monkeypatch)
    template = ace()
    del template['examples']
    path = tmp_path / 'prepare.json'
    path.write_text(json.dumps(dict(fit=template, feedback=feedback, timeout_seconds=30)))
    value = _configuration(path)
    out = tmp_path / 'prepared'
    out.mkdir()
    result = _build(value, out)
    config = FitConfig.from_file(result['fit_configuration']['path'])
    assert config.validate()['validated']
    assert result['model_calls'] == 0 and not result['scoring_executed']
    assert config.value['feedback']['model_id'] == 'm'
    assert config._ace_examples()['train'][0].expected_output == 'case-0'
    assert not adapter.calls


def test_bfcl_file_factory_preserves_native_scoring(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace_bfcl import create_bfcl_feedback

    _, feedback, adapter = inputs(tmp_path, monkeypatch)
    processor = create_bfcl_feedback(**{k: v for k, v in feedback.items() if k != 'kind'},
        mode='offline', output_root=tmp_path / 'feedback')
    assert processor.score_answer('[{"lookup":{"x":3}}]', 'case-0') == 1.0
    assert adapter.calls == [('case-0', [{'lookup': '{"x":3}'}])]
    with pytest.raises(ValidationError, match='outside'):
        processor.score_answer('[{"lookup":{"x":3}}]', 'case-2')


@pytest.mark.parametrize('change', ['overlap', 'duplicate', 'missing', 'hash', 'online'])
def test_bfcl_binding_rejects_invalid_source_before_scoring(tmp_path, monkeypatch, change):
    from agentinputlab.recipes.ace_bfcl import create_bfcl_feedback

    source, feedback, adapter = inputs(tmp_path, monkeypatch)
    if change == 'overlap':
        source['heldout_ids'] = ['case-0']
    elif change == 'duplicate':
        source['validation_ids'] *= 2
    elif change == 'missing':
        del source['heldout_ids']
    elif change == 'hash':
        source['prepared_file_hashes'] = {'fixture': 'other'}
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    feedback['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValidationError):
        create_bfcl_feedback(**{k: v for k, v in feedback.items() if k != 'kind'},
            mode='online' if change == 'online' else 'offline', output_root=tmp_path / 'feedback')
    assert not adapter.calls


def test_bfcl_fit_command_binds_builtin_factory_and_target_identity(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace import ACEOfflineRecipe
    from tests.unit.test_ace_common_support import _artifact

    _, feedback, _ = inputs(tmp_path, monkeypatch)
    value = ace()
    value['feedback'] = feedback
    captured = {}
    def fit(self, *a, **kw):
        captured['command'] = self.client.command
        return _artifact()
    monkeypatch.setattr(ACEOfflineRecipe, 'fit', fit)
    FitConfig.from_dict(value).execute(tmp_path / 'fit')
    cmd = captured['command']
    assert cmd[cmd.index('--task-processor-factory') + 1] == 'agentinputlab.recipes.ace_bfcl:create_bfcl_feedback'
    value['feedback']['model_id'] = 'another-model'
    with pytest.raises(ValidationError, match='target'):
        FitConfig.from_dict(value)
