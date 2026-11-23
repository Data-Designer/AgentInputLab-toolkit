"""Local provider bindings must not initialize a second public API client."""
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from workers.official_arenas.providers.toolsandbox import _intervened_agent, _tracked_user


@pytest.mark.parametrize("timeouts", [{}, {"agent_timeout_seconds": 31, "environment_timeout_seconds": 73.5}])
def test_local_agent_and_simulator_do_not_need_global_openai_credentials(timeouts):
    class DefaultClientRole:
        def __init__(self):
            raise AssertionError('unused upstream default client must not initialize')

    clients=[]
    def client(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(**kwargs)
    modules={name:ModuleType(name) for name in (
        'openai','tool_sandbox','tool_sandbox.common','tool_sandbox.roles',
        'tool_sandbox.common.execution_context','tool_sandbox.common.message_conversion',
        'tool_sandbox.roles.openai_api_agent','tool_sandbox.roles.openai_api_user')}
    modules['openai'].OpenAI=client
    modules['tool_sandbox.common.execution_context'].RoleType=SimpleNamespace()
    modules['tool_sandbox.common.message_conversion'].Message=object
    modules['tool_sandbox.roles.openai_api_agent'].OpenAIAPIAgent=DefaultClientRole
    modules['tool_sandbox.roles.openai_api_user'].OpenAIAPIUser=DefaultClientRole
    config={'agent_model':'local-agent','agent_base_url':'http://127.0.0.1:9001/v1',
        'agent_key_env':'LOCAL_ONLY_KEY','user_model':'local-user',
        'user_base_url':'http://127.0.0.1:9002/v1','user_key_env':'LOCAL_ONLY_KEY', **timeouts}
    with patch.dict('sys.modules',modules), patch.dict('os.environ',{'LOCAL_ONLY_KEY':'local-dummy'},clear=True):
        agent=_intervened_agent({},config,seed=17,budget={})
        user=_tracked_user(config,seed=17,budget={})
    assert len(clients)==2
    assert [c['base_url'] for c in clients]==[config['agent_base_url'],config['user_base_url']]
    assert all(c['max_retries']==0 and c['api_key']=='local-dummy' for c in clients)
    assert agent.openai_client.base_url!=user.openai_client.base_url
    if timeouts:
        assert [c['timeout'] for c in clients] == [31, 73.5]
    else:
        assert all('timeout' not in c for c in clients)


def test_live_history_optimizer_and_agent_share_one_budget_and_usage():
    from copy import deepcopy

    requests = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    def create(**kwargs):
        requests.append(kwargs)
        return response

    class Transform:
        def transform(self, messages, **kwargs):
            assert kwargs["remaining_budget"] == {"max_model_calls": 8, "max_output_tokens": 4096}
            return {"messages": messages, "usage": {"model_calls": 1, "output_tokens": 4}}

    modules = {name: ModuleType(name) for name in (
        'openai', 'tool_sandbox', 'tool_sandbox.common', 'tool_sandbox.roles',
        'tool_sandbox.common.execution_context', 'tool_sandbox.common.message_conversion',
        'tool_sandbox.roles.openai_api_agent')}
    modules['openai'].OpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    modules['tool_sandbox.common.execution_context'].RoleType = SimpleNamespace()
    modules['tool_sandbox.common.message_conversion'].Message = object
    modules['tool_sandbox.roles.openai_api_agent'].OpenAIAPIAgent = object
    config = {'agent_model': 'local-agent', 'agent_base_url': 'http://127.0.0.1:9001/v1',
              'agent_key_env': 'LOCAL_ONLY_KEY'}
    budget = {'model_calls': 0, 'output_tokens': 0, 'max_model_calls': 8,
              'max_output_tokens': 4096, 'max_output_tokens_per_call': 512,
              'max_tool_calls': 8, 'tool_calls': 0, 'hard_stops': 0}
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'query'}]
    original = deepcopy(messages)
    with patch.dict('sys.modules', modules), patch.dict('os.environ', {'LOCAL_ONLY_KEY': 'fixture'}, clear=True):
        agent = _intervened_agent({'query': 'query'}, config, seed=17, budget=budget, history_transform=Transform())
        assert agent.model_inference(messages, []) is response
    assert messages == original
    assert budget['model_calls'] == agent.usage['model_calls'] == 2
    assert budget['output_tokens'] == agent.usage['output_tokens'] == 9
    assert len(requests) == 1 and requests[0]['max_tokens'] == 512


@pytest.mark.parametrize('options', [{}, {'agent_extra_body': {'top_p': 0.9},
    'environment_extra_body': {'top_p': 0.8, 'chat_template_kwargs': {'enable_thinking': True}}},
    {'agent_extra_body_mode': 'replace', 'environment_extra_body_mode': 'replace'},
    {'agent_extra_body_mode': 'replace', 'agent_extra_body': {'thinking': {'type': 'disabled'}},
     'environment_extra_body_mode': 'replace', 'environment_extra_body': {'top_p': 0.8}}])
def test_agent_and_user_normalize_opaque_ids_without_new_calls_or_budget_changes(options):
    from copy import deepcopy

    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[SimpleNamespace(id='chatcmpl-tool-abc', type='function',
                function=SimpleNamespace(name='get_cellular_service_status', arguments='{}'))]))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    modules = {name: ModuleType(name) for name in (
        'openai', 'tool_sandbox', 'tool_sandbox.common', 'tool_sandbox.roles',
        'tool_sandbox.common.execution_context', 'tool_sandbox.common.message_conversion',
        'tool_sandbox.roles.openai_api_agent', 'tool_sandbox.roles.openai_api_user')}
    modules['openai'].OpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    modules['tool_sandbox.common.execution_context'].RoleType = SimpleNamespace()
    modules['tool_sandbox.common.message_conversion'].Message = object
    modules['tool_sandbox.roles.openai_api_agent'].OpenAIAPIAgent = object
    modules['tool_sandbox.roles.openai_api_user'].OpenAIAPIUser = object
    config = {'agent_model': 'agent', 'agent_base_url': 'http://127.0.0.1:9001/v1',
        'agent_key_env': 'LOCAL_ONLY_KEY', 'user_model': 'user',
        'user_base_url': 'http://127.0.0.1:9002/v1', 'user_key_env': 'LOCAL_ONLY_KEY', **options}
    original_budget = {'model_calls': 0, 'output_tokens': 0, 'max_model_calls': 8,
        'max_output_tokens': 4096, 'max_output_tokens_per_call': 512,
        'max_tool_calls': 8, 'tool_calls': 0, 'hard_stops': 0}
    agent_budget = deepcopy(original_budget)
    user_budget = deepcopy(original_budget)
    tools = [{'type': 'function', 'function': {'name': 'get_cellular_service_status'}}]
    with patch.dict('sys.modules', modules), patch.dict('os.environ', {'LOCAL_ONLY_KEY': 'fixture'}, clear=True):
        agent = _intervened_agent({'query': 'query'}, config, seed=17, budget=agent_budget)
        user = _tracked_user(config, seed=1000017, budget=user_budget)
        for role in (agent, user):
            response = role.model_inference([{'role': 'user', 'content': 'query'}], tools)
            assert response.choices[0].message.tool_calls[0].id.isidentifier()
            assert role.compatibility_repairs[0]['original_id'] == 'chatcmpl-tool-abc'
            assert role.usage['model_calls'] == 1 and role.usage['output_tokens'] == 5
    assert len(requests) == 2
    assert [request['seed'] for request in requests] == [17, 1000017]
    assert all(request['max_tokens'] == 512 for request in requests)
    for request, field in zip(requests, ('agent_extra_body', 'environment_extra_body'), strict=True):
        inherited = {} if options.get(field + '_mode') == 'replace' else {
            'chat_template_kwargs': {'enable_thinking': False}}
        assert request['extra_body'] == {**inherited, **options.get(field, {})}
    assert agent_budget['model_calls'] == user_budget['model_calls'] == 1
    assert agent_budget['output_tokens'] == user_budget['output_tokens'] == 5
    assert agent_budget['tool_calls'] == 1 and user_budget['tool_calls'] == 0
