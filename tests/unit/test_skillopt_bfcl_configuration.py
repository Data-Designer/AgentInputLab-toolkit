"""Fixture tests for public SkillOpt BFCL, not real-model evidence."""
import hashlib
import json
import sys
import types
from dataclasses import replace

import pytest

from agentinputlab.backends import ReplayBackend
from agentinputlab.backends.llm import GenerationResponse, Usage
from agentinputlab.contracts import ValidationError
from agentinputlab.fit_configuration import FitConfig
from tests.unit.test_bfcl_fit_configuration import inputs
from tests.unit.test_skillopt_retrieval_configuration import configured


def setup(tmp_path, monkeypatch):
    value, _ = configured(tmp_path, monkeypatch)
    _, feedback, adapter = inputs(tmp_path, monkeypatch)
    value['environment'].update(kind='bfcl', source_input=feedback['source_input'])
    base = types.ModuleType('skillopt.envs.base')
    base.EnvAdapter = type('EnvAdapter', (), {})
    monkeypatch.setitem(sys.modules, 'skillopt.envs.base', base)
    return value, adapter


def test_public_fit_accepts_bfcl_without_custom_python(tmp_path, monkeypatch):
    value, _ = setup(tmp_path, monkeypatch)
    assert FitConfig.from_dict(value).validate()['validated']


def build(tmp_path, monkeypatch, *, calls=None, tokens=8):
    import agentinputlab.recipes.skillopt_builtin_environment as builtin

    value, adapter = setup(tmp_path, monkeypatch)
    message = dict(role='assistant', content=None, tool_calls=calls if calls is not None else [
        dict(type='function', function=dict(name='lookup', arguments='{"x":3}'))])
    backend = ReplayBackend([GenerationResponse('', usage=Usage(19, tokens),
        raw={'choices': [{'message': message}]})])
    seen = []
    original = backend.generate
    # ReplayBackend is immutable; a delegating backend preserves its protocol.
    class Recording:
        model_id, capabilities = backend.model_id, backend.capabilities
        def fingerprint(self): return backend.fingerprint()
        def generate(self, request):
            seen.append(request)
            return original(request)
    monkeypatch.setattr(builtin, 'validate_environment', lambda c: (c, Recording()))
    env = builtin.create(cfg={'minibatch_size': 1}, configuration=value['environment'])
    return env, adapter, seen


def test_native_tools_ast_and_skill_preserved(tmp_path, monkeypatch):
    env, adapter, seen = build(tmp_path, monkeypatch)
    batch = env.build_train_env(1, 17)
    row = env.rollout(batch, 'candidate skill', tmp_path / 'rollout')[0]
    assert row['soft'] == 1 and row['hard'] == 1
    assert adapter.calls == [('case-0', [{'lookup': '{"x":3}'}])]
    request = seen[0]
    assert request.tools == adapter.function_tools(adapter.cases[0])
    assert request.prompt == adapter.cases[0].query
    assert request.max_tokens == 512 and request.seed == 17
    assert 'candidate skill' in request.system and 'Use typed arguments.' in request.system
    assert 'secret-reference' not in request.system
    assert row['usage'] == dict(model_calls=1, input_tokens=19, output_tokens=8)
    assert env.fingerprint()['heldout_not_exposed']
    assert (tmp_path / 'rollout/results.jsonl').is_file()


@pytest.mark.parametrize('change', ['test', 'duplicate', 'mutated', 'unknown', 'diagnostic', 'source'])
def test_invalid_batch_cannot_consume_model(tmp_path, monkeypatch, change):
    env, adapter, seen = build(tmp_path, monkeypatch)
    batch = env.build_train_env(1, 17)
    kwargs = {}
    with pytest.raises(ValidationError):
        if change == 'test':
            env.build_eval_env(1, 'test', 17)
        elif change == 'duplicate':
            batch *= 2
        elif change == 'mutated':
            batch[0]['question'] = 'wrong'
        elif change == 'unknown':
            batch[0]['id'] = 'heldout'
        elif change == 'diagnostic':
            kwargs['diagnostic_mode'] = True
        elif change == 'source':
            adapter.prepared.file_sha256['fixture'] = 'changed'
        env.rollout(batch, 'skill', tmp_path / 'rollout', **kwargs)
    assert not seen and not adapter.calls


def test_native_mismatch_remains_zero_without_retry(tmp_path, monkeypatch):
    env, adapter, seen = build(tmp_path, monkeypatch, calls=[])
    row = env.rollout(env.build_eval_env(1, 'val', 17), 'skill', tmp_path / 'rollout')[0]
    assert row['soft'] == 0 and row['agent_ok']
    assert len(seen) == len(adapter.calls) == 1


@pytest.mark.parametrize('failure', ['scorer', 'tokens'])
def test_execution_errors_not_converted_to_zero(tmp_path, monkeypatch, failure):
    env, adapter, seen = build(tmp_path, monkeypatch, tokens=513 if failure == 'tokens' else 8)
    if failure == 'scorer':
        adapter.score = lambda *a: (_ for _ in ()).throw(OSError('scorer unavailable'))
    batch = env.build_train_env(1, 17)
    with pytest.raises((OSError, ValidationError)):
        env.rollout(batch, 'skill', tmp_path / 'rollout')
    assert len(seen) == 1
    error = tmp_path / 'rollout/predictions' / batch[0]['id'] / 'consumed-execution-error.json'
    assert not json.loads(error.read_text())['automatic_retry']


def test_existing_output_prevents_duplicate_attempt(tmp_path, monkeypatch):
    env, _, seen = build(tmp_path, monkeypatch)
    batch = env.build_train_env(1, 17)
    env.rollout(batch, 'skill', tmp_path / 'rollout')
    with pytest.raises(FileExistsError):
        env.rollout(batch, 'skill', tmp_path / 'rollout')
    assert len(seen) == 1


@pytest.mark.parametrize('change', ['none', 'budget', 'prompt', 'tools', 'duplicate'])
def test_saved_source_rows_must_match_native_inputs(tmp_path, monkeypatch, change):
    from pathlib import Path

    from agentinputlab.benchmarks.bfcl_v3 import BFCL_V3_OFFICIAL_SOURCE_REVISION
    from agentinputlab.contracts import ContextBlock
    from agentinputlab.contracts.serialization import to_jsonable
    from agentinputlab.recipes.skillopt_builtin_environment import create

    value, adapter = setup(tmp_path, monkeypatch)
    env = value['environment']
    path = Path(env['source_input']['path'])
    source = json.loads(path.read_text())
    source['rows'] = []
    for role, e in zip(('train', 'validation'), adapter.cases[:2], strict=True):
        functions = to_jsonable(adapter.function_tools(e))
        state = replace(e.visible_state(), context=(ContextBlock(
            block_id='native_function_catalog', text=json.dumps(functions, sort_keys=True),
            source=BFCL_V3_OFFICIAL_SOURCE_REVISION),))
        state = replace(state, budget=replace(state.budget, max_context_tokens=262144,
            max_output_tokens=128 if change == 'budget' else 4096))
        source['rows'].append(dict(role=role, case_id=e.example_id, state=to_jsonable(state),
            state_hash=state.state_hash, functions=functions))
    if change == 'prompt':
        source['rows'][0]['state']['query'] = 'changed'
    elif change == 'tools':
        source['rows'][0]['functions'] = []
    elif change == 'duplicate':
        source['rows'] = source['rows'][:1] * 2
    path.write_text(json.dumps(source))
    env['source_input']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if change == 'none':
        assert create(cfg={}, configuration=env).fingerprint()['heldout_not_exposed']
    else:
        with pytest.raises(ValidationError, match='captured|budget'):
            create(cfg={}, configuration=env)
    assert not adapter.calls
