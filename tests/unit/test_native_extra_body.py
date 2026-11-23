"""Transport extensions reach the request without changing task/budget ownership."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from workers.official_arenas.protocol import OfficialArenaError, native_extra_body
from workers.official_arenas.providers import statebench, tooltalk


@pytest.mark.parametrize('key', ['model', 'messages', 'input', 'instructions', 'tools',
    'tool_choice', 'parallel_tool_calls', 'temperature', 'seed', 'n', 'stream', 'stop',
    'max_tokens', 'max_completion_tokens', 'max_output_tokens', 'max_length',
    'response_format', 'text', 'reasoning', 'previous_response_id', 'truncation'])
def test_extra_body_cannot_override_native_request_owned_fields(key):
    with pytest.raises(OfficialArenaError, match='must not override'):
        native_extra_body({'extra': {key: 'override'}}, 'extra')


@pytest.mark.parametrize('value', [None, [], 'invalid', {'top_p': float('nan')},
    {'top_p': float('inf')}, {1: 'not-string'}, {'nested': object()}])
def test_extra_body_rejects_non_json_or_nonfinite_values(value):
    with pytest.raises(OfficialArenaError):
        native_extra_body({'extra': value}, 'extra')


def test_explicit_options_are_copied_and_omission_preserves_default():
    default = {'chat_template_kwargs': {'enable_thinking': False}}
    custom = {'chat_template_kwargs': {'enable_thinking': True}, 'top_p': 0.9}
    result = native_extra_body({'extra': custom}, 'extra', defaults=default)
    result['chat_template_kwargs']['enable_thinking'] = 'mutated'
    assert custom['chat_template_kwargs']['enable_thinking'] is True
    assert native_extra_body({}, 'extra', defaults=default) == default
    assert native_extra_body({}, 'extra') == {}


@pytest.mark.parametrize('value',[{}, {'thinking':{'type':'disabled'}}])
def test_explicit_replace_omits_only_inherited_extensions(value):
    defaults={'chat_template_kwargs':{'enable_thinking':False}}
    config={'extra':value,'extra_mode':'replace'}
    assert native_extra_body(config,'extra',defaults=defaults)==value
    assert defaults=={'chat_template_kwargs':{'enable_thinking':False}}
    with pytest.raises(OfficialArenaError,match='must not override'):
        native_extra_body({'extra':{'max_tokens':1},'extra_mode':'replace'},'extra',defaults=defaults)


@pytest.mark.parametrize('mode',[None,True,[],{},'auto',''])
def test_invalid_merge_mode_fails_before_transport(mode):
    with pytest.raises(OfficialArenaError,match='merge or replace'):
        native_extra_body({'extra':{},'extra_mode':mode},'extra')


def test_state_response_transport_adds_only_explicit_extensions(monkeypatch):
    calls = []
    module = ModuleType('openai')
    def create(**kwargs):
        calls.append(kwargs)
        return 'original response'
    module.OpenAI = lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=create))
    native = ModuleType('state_bench.client')
    native.LLMClient = type('LLMClient', (), {'complete_with_tools': lambda *args, **kwargs: None})
    monkeypatch.setitem(sys.modules, 'openai', module)
    monkeypatch.setitem(sys.modules, 'state_bench.client', native)
    monkeypatch.setenv('LOCAL_TEST_KEY', 'fixture')
    custom = {'top_p': 0.8, 'chat_template_kwargs': {'enable_thinking': True}}
    client = statebench._openai_compatible_client(model='fixture', base_url='http://localhost:1/v1',
        api_key_var='LOCAL_TEST_KEY', extra_body=custom)
    custom['chat_template_kwargs']['enable_thinking'] = False
    assert client._client.responses.create(model='fixture', input=[], max_output_tokens=17) == 'original response'
    assert calls == [{'model': 'fixture', 'input': [], 'max_output_tokens': 17,
        'extra_body': {'top_p': 0.8, 'chat_template_kwargs': {'enable_thinking': True}}}]


@pytest.mark.parametrize('custom', [{}, {'thinking': {'type': 'disabled'}}])
@pytest.mark.parametrize('mode', ['merge', 'replace'])
def test_state_extension_mode_keeps_native_request_fields(monkeypatch, custom, mode):
    calls = []
    module = ModuleType('openai')
    module.OpenAI = lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or 'response'))
    native = ModuleType('state_bench.client')
    native.LLMClient = type('LLMClient', (), {'complete_with_tools': lambda *args, **kwargs: None})
    monkeypatch.setitem(sys.modules, 'openai', module)
    monkeypatch.setitem(sys.modules, 'state_bench.client', native)
    monkeypatch.setenv('LOCAL_TEST_KEY', 'fixture')
    client = statebench._openai_compatible_client(model='fixture', base_url='http://localhost:1/v1',
        api_key_var='LOCAL_TEST_KEY', extra_body=custom, extra_body_mode=mode)
    request = {'model': 'fixture', 'input': [], 'max_output_tokens': 512,
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    assert client._client.responses.create(**request) == 'response'
    expected = request['extra_body'] if mode == 'merge' else {}
    assert calls == [{**request, 'extra_body': {**expected, **custom}}]
    assert request['extra_body'] == {'chat_template_kwargs': {'enable_thinking': False}}


def test_state_explicit_nonthinking_keeps_temperature_and_history(monkeypatch):
    calls = []
    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or "response"))
    native = ModuleType("state_bench.client")
    native.LLMClient = type("LLMClient", (), {"complete_with_tools": lambda *args, **kwargs: None})
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setitem(sys.modules, "state_bench.client", native)
    monkeypatch.setenv("LOCAL_TEST_KEY", "fixture")
    client = statebench._openai_compatible_client(model="fixture", base_url="http://localhost:1",
        api_key_var="LOCAL_TEST_KEY", responses_reasoning_disabled=True)
    request = dict(model="fixture", temperature=0, max_output_tokens=512, tools=[],
                   input=[{"role": "assistant", "type": "message", "status": "incomplete",
                           "phase": "final_answer", "content": []}])
    assert client._client.responses.create(**request) == "response"
    assert calls == [{**request, "reasoning": {"effort": "none"}}]
    with pytest.raises(OfficialArenaError, match="conflicts"):
        client._client.responses.create(**request, reasoning={"effort": "high"})
    assert len(calls) == 1
    helper = dict(model="fixture", input=[], reasoning={"effort": "high"}, max_output_tokens=16384)
    assert client._client.responses.create(**helper) == "response"
    assert calls[-1] == helper


@pytest.mark.parametrize('mode',['merge','replace'])
@pytest.mark.parametrize('custom', [None, {'top_p': 0.8},
    {'chat_template_kwargs': {'enable_thinking': True}}])
def test_tooltalk_generation_keeps_budget_and_applies_extensions(monkeypatch, custom, mode):
    calls = []
    module = ModuleType('openai')
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=None, content='synthetic'))], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    module.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', module)
    monkeypatch.setattr(tooltalk, '_tool_schemas', lambda: [])
    predictor = tooltalk._Predictor(model='fixture', base_url='http://localhost:1/v1', api_key='fixture',
        materialized={}, budget={'max_model_calls': 8, 'max_tool_calls': 8, 'max_output_tokens': 4096},
        optimizer_usage={}, seed=17, max_output_tokens_per_call=512, extra_body=custom,extra_body_mode=mode)
    monkeypatch.setattr(predictor, '_messages', lambda *args: [{'role': 'user', 'content': 'synthetic'}])
    assert predictor({}, []) == {'role': 'assistant', 'text': 'synthetic'}
    inherited={'chat_template_kwargs':{'enable_thinking':False}} if mode=='merge' else {}
    assert calls[0]['extra_body'] == {**inherited,**(custom or {})}
    assert calls[0]['max_tokens'] == 512 and calls[0]['seed'] == 17
    assert predictor.model_calls == 1 and predictor.output_tokens == 2
