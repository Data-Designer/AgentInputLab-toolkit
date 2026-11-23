"""Synthetic source/runtime fixtures; not native-source or model acceptance."""
import hashlib
import json

import pytest

from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.acquire import HYDE_EXTERNAL_RELEASE, HyDENativeConfig
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.backends import Document
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, Usage
from agentinputlab.backends.retrieval import RetrievalResult
from agentinputlab.contracts import ArtifactError, BudgetExceededError, ValidationError
from agentinputlab.integrations.full_architecture import hyde as native_module
from agentinputlab.integrations.full_architecture.hyde import (
    HyDEPatchedSourceRuntime,
    write_hyde_patch_manifest,
)
from agentinputlab.runner import InterventionRunner

# Independently authored test double, deliberately NOT a copy of author source.
# Only the reviewed replace-site expression is shared with the external module.
SYNTHETIC_ALGORITHM='''import numpy as np
class HyDE:
    def __init__(self, promptor, generator, encoder, searcher):
        self.components = promptor, generator, encoder, searcher
    def e2e_search(self, question, k):
        p, g, e, s = self.components
        generated = g.generate(p.build_prompt(question))
        rows = []
        for snippet in (question, *generated):
            c_emb = e.encode(snippet)
            rows.append(np.array(c_emb))
        vector = np.mean(np.array(rows), axis=0)
        return s.search(vector.reshape(1, len(vector)), k)
'''
SYNTHETIC_PROMPTOR='''class Promptor:
    def __init__(self, task, language):
        self.task, self.language = task, language
    def build_prompt(self, question):
        return "SYNTHETIC " + self.task + ": " + question
'''


@pytest.fixture
def source(tmp_path,monkeypatch):
    pytest.importorskip('numpy')
    root=tmp_path/'mount'
    pins={}
    for name,text in (('src/hyde/hyde.py',SYNTHETIC_ALGORITHM),('src/hyde/promptor.py',SYNTHETIC_PROMPTOR)):
        p=root/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(text)
        pins[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(native_module,'HYDE_SOURCE_SHA256',pins)
    receipt=write_hyde_patch_manifest(root,tmp_path/'patch.json',accept_source_review=True)
    return root,HyDEPatchedSourceRuntime(source_root=root,patch_manifest=receipt['path'],expected_manifest_sha256=receipt['sha256'])


class EmbeddingFixture:
    embedder_id='synthetic-embedder'

    def __init__(self):
        self.inputs=[]

    def fingerprint(self):
        return {'kind':'synthetic','model':'fixture-v1'}

    def embed(self,texts):
        self.inputs.extend(texts)
        return [(float(len(t)),1.0) for t in texts]


class SearchFixture:
    searcher_id='synthetic-index'

    def __init__(self):
        self.inputs=[]

    def fingerprint(self):
        return {'kind':'synthetic','corpus':'fixture-v1'}

    def search(self,vector,*,top_k):
        self.inputs.append(tuple(vector))
        return [RetrievalResult(Document(doc_id='doc',text='Synthetic real-corpus fixture text',corpus_revision='fixture-v1'),score=1.0,rank=1)]


def backend(*texts):
    values=iter(texts)
    return CallableBackend(lambda r:GenerationResponse(next(values),usage=Usage(7,3)),
        revision='fixture',callable_id='synthetic-generation')


def test_source_acknowledgement_and_patch_create_only(source,tmp_path):
    root,runtime=source
    with pytest.raises(ArtifactError,match='source review'):
        write_hyde_patch_manifest(root,tmp_path/'absent.json')
    with pytest.raises(FileExistsError):
        write_hyde_patch_manifest(root,runtime.patch_manifest,accept_source_review=True)
    assert b'np.asarray' not in (root/'src/hyde/hyde.py').read_bytes()
    assert runtime.fingerprint()['source_redistributed'] is False


def test_native_pipeline_patch_nested_observability_and_mean(source):
    root,runtime=source
    generator=backend('aaa','bb')
    encoder,searcher=EmbeddingFixture(),SearchFixture()
    state=FrozenAgentState(query='Q',metadata={'gold':'PRIVATE'})
    method=create_intervention(HYDE_EXTERNAL_RELEASE,config=HyDENativeConfig(num_hypotheses=2))
    record=InterventionRunner().run(method,state,InterventionContext(backend=generator,
        dense_embedder=encoder,vector_searcher=searcher,external_runtime=runtime,seed=17))
    assert record.usage.model_calls==2 and record.usage.embedding_calls==3 and record.usage.retrieval_calls==1
    assert record.usage.input_tokens==14 and record.usage.output_tokens==6
    assert encoder.inputs==['Q','aaa','bb'] and searcher.inputs==[(2.0,1.0)]
    assert record.patch.metadata['source_algorithm_executed']
    assert record.patch.apply(state).context[0].block_id=='doc'
    assert [r.seed for r in generator.calls]==[17,18]
    assert all(r.messages[0].content=='SYNTHETIC web search: Q' for r in generator.calls)
    assert 'PRIVATE' not in repr(generator.calls)


def test_native_empty_hypotheses_are_retained_not_clean_room_filtered(source):
    _,runtime=source
    encoder=EmbeddingFixture()
    result=InterventionRunner().run(create_intervention(HYDE_EXTERNAL_RELEASE,config={'num_hypotheses':2}),
        FrozenAgentState(query='Q'),InterventionContext(backend=backend('',''),dense_embedder=encoder,
        vector_searcher=SearchFixture(),external_runtime=runtime))
    assert encoder.inputs==['Q','',''] and result.patch.trace.events[1].payload['empty_indices']==(0,1)
    assert result.usage.model_calls==2


def test_nested_cpu_diagnostic_is_not_an_additional_monitor_charge(source,monkeypatch):
    _,runtime=source
    clock=iter((10.0,10.25))
    monkeypatch.setattr(native_module.time,'thread_time',lambda:next(clock))
    from agentinputlab.integrations.full_architecture.hyde import HyDEPatchedExecutionRequest
    result=runtime.run(HyDEPatchedExecutionRequest('Q',HyDENativeConfig(num_hypotheses=1),17,
        ExecutionBudget(),backend('a'),EmbeddingFixture(),SearchFixture()))
    assert result['calling_thread_cpu_seconds']==0.25
    assert result['usage']['cpu_seconds']==0.0
    assert result['process_cpu_window_is_not_additive_across_concurrent_requests']


@pytest.mark.parametrize('budget',[ExecutionBudget(max_model_calls=1),ExecutionBudget(max_retrieval_calls=0),ExecutionBudget(max_output_tokens=1)])
def test_insufficient_budget_fails_before_generation(source,budget):
    _,runtime=source
    generator=backend()
    with pytest.raises(BudgetExceededError):
        InterventionRunner().run(create_intervention(HYDE_EXTERNAL_RELEASE,config={'num_hypotheses':2}),
            FrozenAgentState(query='Q',budget=budget),InterventionContext(backend=generator,dense_embedder=EmbeddingFixture(),
            vector_searcher=SearchFixture(),external_runtime=runtime))
    assert not generator.calls


def test_source_or_manifest_mutation_fails_before_models(source):
    root,runtime=source
    with (root/'src/hyde/hyde.py').open('a') as f:
        f.write('\n# mutation')
    with pytest.raises(ArtifactError,match='SHA-256 differs'):
        runtime.fingerprint()


def test_patch_manifest_checksum_and_patch_set_are_bound(source):
    _,runtime=source
    value=json.loads(runtime.patch_manifest.read_text())
    value['patch']['change']='unreviewed'
    runtime.patch_manifest.write_text(json.dumps(value))
    with pytest.raises(ArtifactError):
        runtime.fingerprint()


def test_operator_sdk_factory_can_execute_the_native_patch_profile(source):
    _,runtime=source
    service=ToolkitService(factories={HYDE_EXTERNAL_RELEASE:lambda:create_intervention(HYDE_EXTERNAL_RELEASE,config={'num_hypotheses':1})},
        context_factory=lambda seed:InterventionContext(seed=seed,backend=backend('a'),dense_embedder=EmbeddingFixture(),
            vector_searcher=SearchFixture(),external_runtime=runtime))
    value=service.execute({'release_id':HYDE_EXTERNAL_RELEASE,'state':{'query':'Q'},'seed':17})
    assert value['record']['usage']['model_calls']==1 and value['scored'] is False
    assert value['state']['context'][0]['block_id']=='doc'


@pytest.mark.parametrize('config',[{'task':'absent'},{'task':[]},{'language':''},{'num_hypotheses':0},
    {'num_hypotheses':True},{'max_tokens':0},{'temperature':float('nan')},{'top_k':True},{'stop':'string'},{'unknown':1}])
def test_invalid_native_controls_fail(config):
    with pytest.raises(ValidationError):
        create_intervention(HYDE_EXTERNAL_RELEASE,config=config)
