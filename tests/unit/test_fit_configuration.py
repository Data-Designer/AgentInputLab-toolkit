"""File-based fitting delegates to native recipes; fixtures are not model evidence."""
import hashlib
import json
import sys

import pytest

from agentinputlab.cli.main import main
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.fit_configuration import FitConfig


def ace():
    return dict(release_id='recipe.ace.offline', checkout='upstream/ace',
        python_executable=sys.executable, timeout_seconds=1200,
        controls=dict(max_rounds=1, max_model_calls=8, max_tokens=4096, max_cost_usd=0, seed=17),
        deployment=dict(api_provider='openai', max_tokens_per_call=512, json_mode=True,
            base_url='http://localhost:8000/v1', api_key_env='TEST_ACE_KEY',
            roles={r: dict(model='m', fingerprint=dict(model='m', revision='r'),
                          input_cost_per_million=0, output_cost_per_million=0)
                   for r in ('target', 'reflector', 'curator')}),
        examples=dict(train=[dict(example_id='train-1', input='train question', expected_output='a', split='train')],
                      validation=[dict(example_id='dev-1', input='dev question', expected_output='b', split='validation')]))


def skill(tmp_path):
    path = tmp_path / 'native.yaml'
    path.write_text('evaluation:\n  eval_test: false\n')
    return dict(release_id='recipe.skillopt.offline', checkout='upstream/skillopt',
        python_executable=sys.executable,
        controls=dict(num_epochs=1, batch_size=1, learning_rate=4, seed=17,
                      max_steps=1, timeout_seconds=1200),
        native_config=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        model_fingerprints={role: dict(model='m', revision='r') for role in ('target', 'optimizer')},
        environment_names=['TEST_SKILLOPT_KEY'])


def test_missing_public_fit_command_is_replaced_with_nonexecuting_validation(tmp_path, capsys):
    path = tmp_path / 'fit.json'
    path.write_text(json.dumps(ace()))
    assert main(['config', 'fit', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['validated'] and not result['executed']
    assert not (tmp_path / 'artifact.json').exists()


@pytest.mark.parametrize('split', ['test', 'validation'])
def test_ace_never_relabels_training_split(split):
    value = ace()
    value['examples']['train'][0]['split'] = split
    with pytest.raises(ValidationError, match='split'):
        FitConfig.from_dict(value).validate()


def test_ace_duplicate_inputs_rejected_before_worker():
    value = ace()
    value['examples']['validation'][0]['input'] = 'train question'
    with pytest.raises(ValidationError, match='overlap'):
        FitConfig.from_dict(value).validate()


def test_ace_no_large_implicit_fit_budget():
    value = ace()
    del value['controls']['max_tokens']
    with pytest.raises(ValidationError, match='controls'):
        FitConfig.from_dict(value).validate()


def sandbox_feedback(tmp_path):
    source = tmp_path / 'feedback.json'
    source.write_text(json.dumps(dict(native_configuration={
        'agent_key_env':'TEST_NATIVE_SANDBOX_KEY', 'user_key_env':'TEST_NATIVE_SANDBOX_KEY',
        'unrelated_field':'DO_NOT_FORWARD_THIS',
    })))
    return dict(kind='toolsandbox', checkout='native/ToolSandbox',
                source_input=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))


def test_native_sandbox_missing_credential_before_fit_attempt(tmp_path, monkeypatch):
    monkeypatch.delenv('TEST_NATIVE_SANDBOX_KEY', raising=False)
    v=ace()
    v['feedback']=sandbox_feedback(tmp_path)
    with pytest.raises(ExternalDependencyError,match='missing fit environment variable: TEST_NATIVE_SANDBOX_KEY'):
        FitConfig.from_dict(v).execute(tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_pinned_sandbox_environment_forwarded_once_without_values_in_configuration(tmp_path, monkeypatch):
    from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker
    from agentinputlab.recipes import ACEOfflineRecipe
    from tests.unit.test_ace_common_support import _artifact

    captured=[]
    original=JsonLineSubprocessWorker.__init__
    def initialize(self,*args,**kwargs):
        captured.append(kwargs['pass_environment'])
        original(self,*args,**kwargs)
    monkeypatch.setattr(JsonLineSubprocessWorker,'__init__',initialize)
    monkeypatch.setattr(ACEOfflineRecipe,'fit',lambda *a,**k:_artifact())
    monkeypatch.setenv('TEST_NATIVE_SANDBOX_KEY','private-native-value')
    monkeypatch.setenv('DO_NOT_FORWARD_THIS','private-unrelated-value')
    v=ace()
    v['feedback']=sandbox_feedback(tmp_path)
    FitConfig.from_dict(v).execute(tmp_path/'result')
    assert captured==[('TEST_ACE_KEY','TEST_NATIVE_SANDBOX_KEY')]
    assert 'private-native-value' not in json.dumps(v)


@pytest.mark.parametrize('bad_name',[None,42,'not-a-name','secret/value'])
def test_native_environment_names_are_not_values(bad_name):
    config=FitConfig.from_dict(ace())
    with pytest.raises(ValidationError,match='must name an environment variable'):
        config._feedback_environment_names({'kind':'toolsandbox'},
            {'native_configuration':{'agent_key_env':bad_name}})


def test_ace_native_arguments_and_artifact_roundtrip(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace import ACEOfflineRecipe
    from tests.unit.test_ace_common_support import _artifact
    captured = {}
    def fit(self, train_examples, *, validation_examples, context):
        captured.update(command=self.client.command, controls=self.controls, context=context,
                        train=train_examples, validation=validation_examples)
        return _artifact()
    monkeypatch.setattr(ACEOfflineRecipe, 'fit', fit)
    config = FitConfig.from_dict(ace())
    result = config.execute(tmp_path / 'fit')
    assert captured['controls'].max_model_calls == 8
    assert captured['command'][1].endswith('/workers/ace/driver.py')
    assert captured['context'].seed == 17
    assert result['status'] == 'fit_completed'
    assert json.loads((tmp_path / 'fit/artifact.json').read_text()) == _artifact().as_dict()
    assert result['evaluation_release_id'] == 'recipe.ace.eval_only'
    with pytest.raises(ValidationError, match='new directory'):
        config.execute(tmp_path / 'fit')


def test_skillopt_native_recipe_and_environment_names(tmp_path, monkeypatch):
    from agentinputlab.recipes.skillopt import SkillOptOfflineRecipe
    from tests.unit.test_skillopt_common_support import _artifact
    captured = {}
    def fit(self, **kwargs):
        captured.update(kwargs, controls=self.controls, python=self.python_executable)
        return _artifact()
    monkeypatch.setattr(SkillOptOfflineRecipe, 'fit', fit)
    monkeypatch.setenv('TEST_SKILLOPT_KEY', 'private-test-value')
    result = FitConfig.from_dict(skill(tmp_path)).execute(tmp_path / 'fit')
    assert captured['environment'] == {'TEST_SKILLOPT_KEY': 'private-test-value'}
    assert captured['controls'].num_epochs == 1
    assert captured['context'].seed == 17
    assert captured['output_root'] == tmp_path / 'fit/native'
    assert result['status'] == 'fit_completed'
    assert 'private-test-value' not in (tmp_path / 'fit/terminal.json').read_text()


def test_cli_completed_ace_fit_prints_frozen_usage_as_json(tmp_path, monkeypatch, capsys):
    from agentinputlab.recipes.ace import ACEOfflineRecipe
    from tests.unit.test_ace_common_support import _artifact

    calls = []
    def fit(self, *args, **kwargs):
        calls.append(1)
        return _artifact()
    monkeypatch.setattr(ACEOfflineRecipe, 'fit', fit)
    path = tmp_path / 'fit.json'
    path.write_text(json.dumps(ace()))
    output = tmp_path / 'completed'
    assert main(['config', 'fit', str(path), '--execute', '--output-dir', str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / 'terminal.json').read_text())
    assert result['optimization_usage'] == dict(_artifact().usage)
    assert calls == [1]
    assert main(['config', 'fit', str(path), '--execute', '--output-dir', str(output)]) == 2
    assert calls == [1]


@pytest.mark.parametrize('kind,module', [('statebench', 'ace_statebench'),
                                      ('toolsandbox', 'ace_toolsandbox'), ('tau', 'ace_tau')])
def test_ace_native_feedback_uses_original_worker_factory(tmp_path, monkeypatch, kind, module):
    from agentinputlab.recipes.ace import ACEOfflineRecipe
    from tests.unit.test_ace_common_support import _artifact

    source = tmp_path / 'source.json'
    source.write_text('{"fixture": true}')
    value = ace()
    value['feedback'] = dict(kind=kind, checkout='./native',
        source_input=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        environment_names=['NATIVE_JUDGE_KEY'])
    captured = {}
    def fit(self, *args, **kwargs):
        captured.update(command=self.client.command)
        return _artifact()
    monkeypatch.setattr(ACEOfflineRecipe, 'fit', fit)
    monkeypatch.setenv('NATIVE_JUDGE_KEY', 'local-test-only')
    path = tmp_path / 'fit.json'
    path.write_text(json.dumps(value))
    FitConfig.from_file(path).execute(tmp_path / 'result')
    command = captured['command']
    factory = command[command.index('--task-processor-factory') + 1]
    assert factory == f'agentinputlab.recipes.{module}:create_{kind}_feedback'
    binding = json.loads(command[command.index('--task-processor-config-json') + 1])
    assert binding['checkout'] == str(tmp_path / 'native')
    assert binding['source_input'] == value['feedback']['source_input']
    assert binding['output_root'] == str(tmp_path / 'result/native-feedback')
    assert binding['mode'] == 'offline'
    assert 'local-test-only' not in json.dumps(command)


def test_ace_native_feedback_hash_and_environment_checked_before_attempt(tmp_path, monkeypatch):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    value = ace()
    value['feedback'] = dict(kind='statebench', checkout='./native',
        source_input=dict(path=str(source), sha256='a' * 64), environment_names=['NO_SUCH_FIT_KEY'])
    with pytest.raises(Exception, match='checksum'):
        FitConfig.from_dict(value).execute(tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    value['feedback']['source_input']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.delenv('NO_SUCH_FIT_KEY', raising=False)
    with pytest.raises(Exception, match='missing fit environment'):
        FitConfig.from_dict(value).execute(tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('change', [{'kind': 'tooltalk'}, {'factory': 'custom:execute'},
    {'mode': 'online'}, {'output_root': '/tmp/existing'}, {'environment_base_url': 'https://user:secret@example.com'}])
def test_native_feedback_cannot_inject_factory_mode_or_output(change):
    value = ace()
    value['feedback'] = dict(kind='statebench', checkout='./native',
        source_input=dict(path='input.json', sha256='a' * 64))
    value['feedback'].update(change)
    with pytest.raises(ValidationError):
        FitConfig.from_dict(value)


def test_skillopt_wrong_native_config_hash_before_attempt(tmp_path):
    value = skill(tmp_path)
    value['native_config']['sha256'] = 'a' * 64
    with pytest.raises(Exception, match='checksum'):
        FitConfig.from_dict(value).execute(tmp_path / 'fit')
    assert not (tmp_path / 'fit').exists()


def test_failed_fit_terminal_survives_and_cannot_retry(tmp_path, monkeypatch):
    from agentinputlab.recipes.ace import ACEOfflineRecipe
    def fail(*args, **kwargs):
        raise RuntimeError('sensitive-provider-text')
    monkeypatch.setattr(ACEOfflineRecipe, 'fit', fail)
    config = FitConfig.from_dict(ace())
    with pytest.raises(Exception, match='fitting failed'):
        config.execute(tmp_path / 'fit')
    receipt = json.loads((tmp_path / 'fit/terminal.json').read_text())
    assert receipt['status'] == 'failed_fit'
    assert not receipt['automatic_retry']
    assert 'sensitive-provider-text' not in json.dumps(receipt)
    with pytest.raises(ValidationError, match='new directory'):
        config.execute(tmp_path / 'fit')


def test_relative_paths_keep_selected_python_environment(tmp_path):
    value = ace()
    value['python_executable'] = 'env/bin/python'
    path = tmp_path / 'fit.json'
    path.write_text(json.dumps(value))
    config = FitConfig.from_file(path)
    assert config.value['checkout'] == str(tmp_path / 'upstream/ace')
    assert config.value['python_executable'] == str(tmp_path / 'env/bin/python')


def test_unknown_configuration_and_secret_values_rejected(tmp_path):
    value = skill(tmp_path)
    value['environment'] = {'API_KEY': 'secret'}
    with pytest.raises(ValidationError, match='unknown'):
        FitConfig.from_dict(value).validate()


def test_missing_environment_is_rejected_before_attempt(tmp_path, monkeypatch):
    monkeypatch.delenv('TEST_SKILLOPT_KEY', raising=False)
    with pytest.raises(Exception, match='missing fit environment'):
        FitConfig.from_dict(skill(tmp_path)).execute(tmp_path / 'fit')
    assert not (tmp_path / 'fit').exists()


def test_custom_environment_is_never_implicitly_enabled(tmp_path):
    value = skill(tmp_path)
    value['environment_binding'] = dict(factory_path='custom.py', factory_sha256='a' * 64,
        configuration_path='custom.json', configuration_sha256='b' * 64)
    with pytest.raises(ValidationError, match='allow_custom_environment'):
        FitConfig.from_dict(value)
    value['allow_custom_environment'] = True
    assert FitConfig.from_dict(value).validate()['executed'] is False


def test_cli_execute_requires_new_output_path(tmp_path, capsys):
    path = tmp_path / 'fit.json'
    path.write_text(json.dumps(ace()))
    assert main(['config', 'fit', str(path), '--execute']) == 2
    assert '--output-dir' in capsys.readouterr().err


@pytest.mark.parametrize('raw', ['{"release_id":"x","release_id":"y"}', '{"controls":{"seed":NaN}}'])
def test_fit_file_rejects_duplicate_or_nonfinite_values(tmp_path, raw):
    path = tmp_path / 'fit.json'
    path.write_text(raw)
    with pytest.raises(ValidationError):
        FitConfig.from_file(path)
