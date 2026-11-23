import io
import json
import os
from unittest.mock import patch

import pytest

from agentinputlab.backends.llm import GenerationRequest, ModelCapability, OpenAICompatibleBackend, PerCallOutputCapBackend
from agentinputlab.configuration import RunConfig
from agentinputlab.contracts import CapabilityError, ValidationError
from workers.reformer import driver as reformer
from workers.infocqr import driver as infocqr
from workers.acon import driver as acon

MODEL='deepseek-v4-1-flash-261017'
FAMILY='deepseek-v4-1-flash'
BASE=dict(model_id=MODEL,base_url='https://example.test/v1',deployment_identity={'revision':'261017'})


def response(model=FAMILY):
    return dict(model=model,choices=[{'message':{'content':'answer'},'finish_reason':'stop'}],
        usage={'prompt_tokens':12,'completion_tokens':3,'total_tokens':15})


def test_aliases_are_explicit_fingerprinted_and_raw_response_identity_is_retained():
    backend=OpenAICompatibleBackend(**BASE,response_model_aliases=(FAMILY,),transport=lambda *a:response())
    result=backend.generate(GenerationRequest(prompt='q'))
    assert result.model==FAMILY
    assert backend.fingerprint()['response_model_aliases']==[FAMILY]
    assert 'response_model_aliases' not in OpenAICompatibleBackend(**BASE).fingerprint()
    backend.transport=lambda *a:response('unrelated-model')
    with pytest.raises(RuntimeError,match='outside'):backend.generate(GenerationRequest(prompt='q'))


@pytest.mark.parametrize('aliases',['family',[FAMILY,FAMILY],[''],[True],[{}]])
def test_invalid_aliases_rejected_before_transport(aliases):
    with pytest.raises(ValidationError):OpenAICompatibleBackend(**BASE,response_model_aliases=aliases)


def test_schema_capability_fails_before_generation_and_never_silently_downgrades():
    calls=[]
    backend=OpenAICompatibleBackend(**BASE,transport=lambda *a:calls.append(a) or response(MODEL))
    request=GenerationRequest(prompt='q',response_format={'type':'json_schema','json_schema':{'schema':{'type':'object'}}})
    with pytest.raises(CapabilityError,match='json_schema'):backend.generate(request)
    assert calls==[]
    backend.capabilities=backend.capabilities|{ModelCapability.JSON_SCHEMA}
    backend.generate(request)
    assert calls[0][2]['response_format']['type']=='json_schema'


def test_run_config_stage_cap_reaches_transport_and_preserves_model_identity():
    config=RunConfig('baseline.no_op',backend={**BASE,'response_model_aliases':[FAMILY],
        'max_output_tokens_per_call':512,'stage_output_caps':{'method:joint':1024}})
    backend=config._backend();assert isinstance(backend,PerCallOutputCapBackend)
    seen=[];backend.inner.transport=lambda *a:seen.append(a[2]) or response()
    backend.generate(GenerationRequest(prompt='q',max_tokens=2048,metadata={'method_id':'method','phase':'joint'}))
    backend.generate(GenerationRequest(prompt='q',max_tokens=700,metadata={'method_id':'method','phase':'other'}))
    backend.generate(GenerationRequest(prompt='q',max_tokens=128))
    assert [r['max_tokens'] for r in seen]==[1024,512,128]
    assert backend.fingerprint()['model']==MODEL
    assert backend.fingerprint()['stage_output_caps']=={'method:joint':1024}


def test_production_backend_fingerprint_carries_alias_to_nested_native_workers():
    from agentinputlab.experiments import production_builders as builders
    config=builders._parse_backend_block({'kind':'openai_compatible',**BASE,'response_model_aliases':[FAMILY]})
    backend=builders._build_llm_backend(config)
    assert reformer.fingerprint_aliases(backend.fingerprint())==(FAMILY,)
    assert infocqr.fingerprint_aliases(backend.fingerprint())==(FAMILY,)
    default=builders._parse_backend_block({'kind':'openai_compatible',**BASE})
    assert 'response_model_aliases' not in default


@pytest.mark.parametrize('settings',[
    {'stage_output_caps':{'method:joint':1024}},
    {'stage_output_caps':[]},
    {'max_output_tokens_per_call':True},
    {'max_output_tokens_per_call':512,'stage_output_caps':{':':1024}},
])
def test_invalid_stage_budget_settings_fail_without_transport(settings):
    with pytest.raises(ValidationError):RunConfig('baseline.no_op',backend={**BASE,**settings})._backend()


@pytest.mark.parametrize('worker',[reformer,infocqr])
def test_native_worker_replays_model_family_only_with_declared_alias(worker):
    fp={'model_id':MODEL,'response_model_aliases':[FAMILY]}
    config=worker.BackendConfig(base_url='https://example.test/v1',api_key_env='TEST_API_KEY',model=MODEL,
        fingerprint=fp,extra_body={},timeout_seconds=3,max_input_tokens_per_call=100,max_output_tokens=512,
        temperature=0,input_cost_per_million=0,output_cost_per_million=0)
    backend=worker.OpenAICompatibleBackend(config)
    def invoke(model):
        with patch.dict(os.environ,{'TEST_API_KEY':'test-key'}),patch.object(worker.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(response(model)).encode())):
            return backend.complete(messages=[{'role':'user','content':'q'}],seed=17,temperature=0,max_output_tokens=512)
    assert invoke(FAMILY).text=='answer'
    with pytest.raises(worker.DriverError):invoke('foreign-model')
    config.fingerprint.pop('response_model_aliases')
    with pytest.raises(worker.DriverError):invoke(FAMILY)


def test_acon_alias_is_explicit_and_bound_to_optimizer_fingerprint():
    llm=acon.OpenAICompatibleLLM(base_url='https://example.test/v1',api_key='key',model=MODEL,seed=17,
        default_temperature=0,timeout_seconds=3,max_tokens=512,response_model_aliases=[FAMILY])
    assert llm.fingerprint()['response_model_aliases']==[FAMILY]
    with patch.object(llm._opener,'open',return_value=io.BytesIO(json.dumps(response()).encode())):
        assert llm.generate('q')=='answer'
    assert llm.total_model_calls==1 and llm.total_output_tokens==3
    with pytest.raises(acon.DriverError):acon._parse_model_response(response(),expected_model=MODEL)
