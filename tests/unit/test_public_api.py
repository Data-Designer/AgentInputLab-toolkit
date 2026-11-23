import json
import threading
import urllib.request
import urllib.error
import pytest
from agentinputlab import FrozenAgentState
from agentinputlab.api import ToolkitService, catalog, create_intervention, make_server, run_intervention

TOKEN='synthetic-unit-test-token-not-a-secret'

def test_registry_api_and_real_prompt_patch():
    c=catalog();assert len(c['methods'])==27;assert len(c['benchmarks'])==31
    assert "tooltalk.full" in {cell["cell_id"] for cell in c["benchmarks"]}
    result=run_intervention(create_intervention('plan.plan_and_solve.ps_prompt'),FrozenAgentState(query='Solve this task'))
    assert 'devise a plan' in result['state']['instruction']
    assert result['scored'] is False and result['benchmark_score'] is None

@pytest.fixture
def server():
    s=make_server(ToolkitService(),token=TOKEN,port=0)
    t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
    yield 'http://127.0.0.1:'+str(s.server_port)
    s.shutdown();s.server_close();t.join()

def request(base,path,body=None,token=TOKEN):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base+path,data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=5) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)

def test_http_auth_catalog_run(server):
    assert request(server,'/v1/health',token='wrong')[0]==401
    code,body=request(server,'/v1/catalog');assert code==200 and len(body['data']['methods'])==27
    code,body=request(server,'/v1/interventions',{'release_id':'baseline.no_op','state':{'query':'hello'},'seed':17})
    assert code==200 and body['data']['state']['query']=='hello'
    assert body['data']['record']['status']=='no_op'

@pytest.mark.parametrize('body',[
    {'release_id':[], 'state':{'query':'x'}},
    {'release_id':'baseline.no_op','state':{'query':'x'},'seed':True},
    {'release_id':'baseline.no_op','state':{'query':'x','messages':[{'role':3}]}},
    {'release_id':'baseline.no_op','state':{'query':'x','budget':{'max_model_calls':-1}}},
    {'release_id':'baseline.no_op','state':{'query':'x'},'backend_url':'http://untrusted'},
])
def test_invalid_requests_rejected_before_execution(server,body):
    assert request(server,'/v1/interventions',body)[0]==400

def test_unconfigured_release_has_no_fallback(server):
    code,body=request(server,'/v1/interventions',{'release_id':'unknown','state':{'query':'x'}})
    assert code==422 and body['error']['code']=='unavailable'

def test_no_unauthenticated_server():
    with pytest.raises(ValueError):make_server(ToolkitService(),token='')


def test_info_cqr_common_support_constructor_uses_the_requested_profile():
    method=create_intervention('rewrite.infocqr.common_support_rw_zsl',
        dataset_id='beir.arguana',dataset_revision='dataset-source-lock',retrieval_corpus_id='beir.arguana')
    assert method.release_id=='rewrite.infocqr.common_support_rw_zsl'
    assert method.prompt_profile=='rw_zsl'


def test_api_rejects_a_factory_returning_a_different_release(monkeypatch):
    from types import SimpleNamespace
    from agentinputlab.registry import AlgorithmRegistry
    monkeypatch.setattr(AlgorithmRegistry,'resolve_entrypoint',lambda self, release:lambda:SimpleNamespace(release_id='wrong'))
    with pytest.raises(Exception,match='different release'):
        create_intervention('plan.plan_and_solve.ps_prompt')


def test_native_compiler_planner_factory_accepts_context_bound_runtime_configuration():
    from agentinputlab.algorithms.decompose.llm_compiler import LLMCompilerControls, LLMCompilerPlannerBridge
    controls = LLMCompilerControls(max_replans=0, streaming=False)
    method = create_intervention('plan.llm_compiler.planner_component',
                                 prompt_hash='a' * 64, controls=controls)
    assert isinstance(method, LLMCompilerPlannerBridge)
    assert method.client is None and method.controls == controls
    assert method.release_id == 'plan.llm_compiler.planner_component'
