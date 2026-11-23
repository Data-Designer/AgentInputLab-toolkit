"""Synthetic transport conformance; never substitute fixtures for model results."""
import copy

import pytest

from agentinputlab.backends import FINAL_CONTENT_STOP_POLICY_ID, FinalContentStopBackend
from agentinputlab.backends.llm import (
    GenerationRequest,
    ModelCapability,
    OpenAICompatibleBackend,
    ReplayBackend,
)
from agentinputlab.contracts import ValidationError


def configured(*, model='qwen3.8-27b-fp8', content='Yes.\nFollow up: next?\nIntermediate answer: unused',
               reason='stop', tokens=445, extra_body=None):
    calls = []
    data = {'model': model, 'choices': [{'message': {'content': content,
        'reasoning': 'analysis\nIntermediate answer: reasoning only'}, 'finish_reason': reason}],
        'usage': {'prompt_tokens': 199, 'completion_tokens': tokens}}
    def transport(url, headers, payload, timeout):
        calls.append(copy.deepcopy(payload))
        return copy.deepcopy(data)
    inner = OpenAICompatibleBackend(model_id=model, base_url='http://127.0.0.1:9000/v1',
        deployment_identity={'revision': 'synthetic-deployment', 'server': 'vllm'}, api_key='',
        transport=transport, extra_body=extra_body or {})
    return inner, calls


@pytest.mark.parametrize('model', ['qwen3.8-27b-fp8', 'qwen3-4b', 'gpt-oss-20b', 'caller-model'])
def test_stops_are_final_content_only_not_model_name_specific(model):
    inner, calls = configured(model=model)
    wrapped = FinalContentStopBackend(inner)
    request = GenerationRequest(prompt='synthetic', system='system', seed=17, max_tokens=512,
        stop=('\nIntermediate answer:',), temperature=.25, metadata={'stage': 'initial'})
    response = wrapped.generate(request)
    assert response.text == 'Yes.\nFollow up: next?'
    assert len(calls) == 1 and 'stop' not in calls[0]
    assert (calls[0]['max_tokens'], calls[0]['seed'], calls[0]['temperature']) == (512, 17, .25)
    assert response.usage.input_tokens == 199 and response.usage.output_tokens == 445
    assert response.raw['choices'][0]['message']['content'].endswith('unused')
    assert response.raw['agentinputlab_final_content_stop']['policy_id'] == FINAL_CONTENT_STOP_POLICY_ID
    assert request.stop == ('\nIntermediate answer:',)


@pytest.mark.parametrize('reason,tokens', [('length', 512), ('stop', 295)])
def test_saved_empty_final_cannot_be_recovered_by_promoting_reasoning(reason, tokens):
    inner, calls = configured(content=None, reason=reason, tokens=tokens)
    response = FinalContentStopBackend(inner).generate(GenerationRequest(
        prompt='synthetic', stop=('\nIntermediate answer:',), max_tokens=512))
    assert response.text == '' and response.finish_reason == reason
    assert response.usage.output_tokens == tokens and len(calls) == 1


def test_first_logical_match_preserves_all_tail_usage_and_provider_finish():
    inner, calls = configured(content='answer\nrest**tail', reason='length', tokens=512)
    response = FinalContentStopBackend(inner).generate(GenerationRequest(prompt='synthetic', stop=('**', '\n')))
    assert response.text == 'answer' and response.finish_reason == 'stop'
    assert response.usage.output_tokens == 512 and len(calls) == 1
    assert response.raw['agentinputlab_final_content_stop']['provider_finish_reason'] == 'length'


def test_no_stops_pass_through_and_default_backend_keeps_native_stops():
    inner, calls = configured(content='keep\nall')
    wrapped = FinalContentStopBackend(inner)
    response = wrapped.generate(GenerationRequest(prompt='synthetic'))
    assert response.text == 'keep\nall' and 'agentinputlab_final_content_stop' not in response.raw
    inner.generate(GenerationRequest(prompt='synthetic', stop=('\n',)))
    assert calls[-1]['stop'] == ['\n']


def test_streaming_not_advertised_without_stream_implementation():
    inner, _ = configured()
    inner.capabilities = inner.capabilities | {ModelCapability.STREAMING}
    wrapped = FinalContentStopBackend(inner)
    assert ModelCapability.STREAMING not in wrapped.capabilities
    assert wrapped.model_id == inner.model_id
    assert wrapped.fingerprint()['inner'] == inner.fingerprint()
    assert wrapped.fingerprint() != inner.fingerprint()


def test_rejects_empty_stop_unseparated_generic_decoder_and_extra_body_stop():
    inner, calls = configured()
    with pytest.raises(ValidationError, match='non-empty'):
        FinalContentStopBackend(inner).generate(GenerationRequest(prompt='synthetic', stop=('',)))
    assert calls == []
    safe, safe_calls = configured()
    wrapped = FinalContentStopBackend(safe)
    safe.extra_body = {'stop': ['provider stop reintroduced after construction']}
    with pytest.raises(ValidationError, match='extra_body'):
        wrapped.generate(GenerationRequest(prompt='synthetic', stop=('logical',)))
    assert safe_calls == []
    with pytest.raises(ValidationError, match='parser'):
        FinalContentStopBackend(ReplayBackend(('synthetic',)))
    inner, calls = configured(extra_body={'stop': ['hidden provider stop']})
    with pytest.raises(ValidationError, match='extra_body'):
        FinalContentStopBackend(inner)
    assert calls == []
