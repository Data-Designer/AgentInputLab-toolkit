"""Fixed real QA acceptance for source-bound planning and retrieval methods."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import time

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.connectors.tools import SchemaOnlyToolConnector, ToolSpec
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.experiments.drivers.static_production import StaticPredictorBuildRequest, StaticRuntimeComponents
from agentinputlab.runner import InterventionRunner
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

RELEASES=('plan.llm_compiler.common_support_planner','plan.self_ask.full_loop',
          'plan.decomp.common_support_odqa','acquire.ircot.common_support')


def checked(path, expected):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(f'upstream source differs: {path.name}')
    return raw.decode()


def demonstrations(root):
    def ordered(checkout, prompt, prompt_hash, config, config_hash):
        raw=checked(checkout/prompt,prompt_hash);cfg=checked(checkout/config,config_hash)
        match=re.search(r'"hotpotqa"\s*:\s*(\[[^]]+\])',cfg)
        if not match:raise ValueError('HotpotQA source demonstration order is absent')
        ids=json.loads(match.group(1));examples={}
        for chunk in re.split(r'(?=^# METADATA: )',raw.strip(),flags=re.MULTILINE):
            lines=chunk.strip().splitlines()
            if lines and lines[0].startswith('# METADATA: '):
                examples[json.loads(lines[0][12:])['qid']]='\n'.join(lines[1:]).strip()
        if len(ids)!=20 or len(set(ids))!=20:raise ValueError('source demonstration count differs')
        return '\n\n\n'.join(examples[i] for i in ids)
    decomp=ordered(root/'DecomP-ODQA','prompts/hotpotqa/question_decomposition.txt',
        '66abd147d9cb2f20db734a2cc817ceda282cd109a8346b3395de5a367f0cdc2f',
        'base_configs/decomp_context_direct_qa_flan_t5_large_hotpotqa.jsonnet',
        'e3de47295c31d95d11f59869b17ccd562bcdda2382e9d0f0344327737ae13cda')
    ircot=ordered(root/'ircot','prompts/hotpotqa/gold_with_2_distractors_context_cot_qa_flan_t5.txt',
        'bcc08b7b4d2158b625be0825d5852d5ca013e65832882056931b93d322055db1',
        'base_configs/ircot_flan_t5_large_hotpotqa.jsonnet',
        '9fc5f365852382726b49803556a1b49a02850d12111458c5637a6ad4b66368e9')
    notebook=json.loads(checked(root/'self-ask/self-ask_plus_search-engine_demo.ipynb',
        '9645cde930a24d943d6fd0237c2c158dc37dc04a17a30aca29017896d7e1c3c2'))
    pairs=[]
    for cell in notebook['cells']:
        if cell.get('cell_type')!='code':continue
        try:tree=ast.parse(''.join(cell.get('source',[])))
        except SyntaxError:continue
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='prompt':
                pairs.append(ast.literal_eval(node.value))
    if len(pairs)!=1 or not pairs[0][0].endswith('\n\nQuestion: '):raise ValueError('Self-Ask prompt changed')
    self_ask=pairs[0][0][:-len('\n\nQuestion: ')]
    return decomp,ircot,self_ask


def execute(config, output, upstream):
    from agentinputlab.algorithms.decompose.decomp import DecompCommonSupportConfig
    from agentinputlab.algorithms.decompose.self_ask import SelfAskConfig
    from agentinputlab.algorithms.acquire import IRCoTCommonSupportConfig
    demo_decomp,demo_ircot,demo_self=demonstrations(upstream)
    constructors={
        RELEASES[0]:{},
        RELEASES[1]:{'config':SelfAskConfig(demonstrations=demo_self,max_tokens_per_call=512)},
        RELEASES[2]:{'config':DecompCommonSupportConfig(demonstrations=demo_decomp,
            prompt_source_sha256='66abd147d9cb2f20db734a2cc817ceda282cd109a8346b3395de5a367f0cdc2f',
            demonstrations_sha256=content_hash(demo_decomp),max_decomposer_tokens=512,max_handler_tokens=256)},
        RELEASES[3]:{'config':IRCoTCommonSupportConfig(demonstrations=demo_ircot,
            prompt_sha256=hashlib.sha256(demo_ircot.encode()).hexdigest(),max_tokens_per_call=512)}}
    adapters={cell:load_production_benchmark_adapter(cell,config['prepared_root']) for cell,_ in QA_CELLS}
    offset=config.get('case_offset',4)
    examples={cell:tuple(sorted(a.examples(split),key=lambda e:e.example_id)[offset:offset+2])
              for (cell,split),a in zip(QA_CELLS,adapters.values())}
    jobs=[]
    for model in config['models']:
        for cell,split in QA_CELLS:
            for example in examples[cell]:
                for release in config.get('releases',RELEASES):
                    if release not in RELEASES:raise ValueError('unknown QA acceptance release')
                    job={'model':model,'benchmark':cell,'split':split,'case_id':example.example_id,
                        'release':release,'seed':17,'budget':{'max_model_calls':8,'max_output_tokens':4096},
                        'constructor':to_jsonable(constructors[release]),'downstream':'production_structured_json',
                        'retriever_scope':'original_visible_paragraphs_only','max_tokens_per_call':512,
                        'stop_policy':'gpt-oss.final-channel-stop.v1' if model['id']=='gpt-oss-20b' else 'native'}
                    job['id']=digest(job);jobs.append(job)
    for name in ('claims','terminals'):(output/name).mkdir()
    save(output/'manifest.json',{'jobs':jobs,'automatic_retry':False,'paper_results_changed':False,
        'demonstration_hashes':{'decomp':content_hash(demo_decomp),'ircot':content_hash(demo_ircot),'self_ask':content_hash(demo_self)}})

    def one(job):
        save(output/'claims'/(job['id']+'.json'),{'job':job,'started':time.time()})
        model=job['model'];calls=[];record=None;stage='intervention';started=time.monotonic()
        source=json.loads((Path(model['service_directory'])/'started.json').read_text())
        binding={'revision':digest(source['model_json_hashes']),'server':'vllm'}
        inner=RecordingBackend(calls=calls,model_id=model['id'],base_url=model['base_url'],
            deployment_identity=binding,timeout_seconds=180,extra_body=model.get('extra_body',{}),transport=local_transport)
        backend=GPTOSSFinalChannelStopBackend(inner) if model['id']=='gpt-oss-20b' else inner
        adapter=adapters[job['benchmark']]
        example=next(e for e in examples[job['benchmark']] if e.example_id==job['case_id'])
        state=replace(example.visible_state(budget=ExecutionBudget(**job['budget'])),tool_names=('search',))
        retriever=InMemoryBM25Retriever(documents=tuple(Document(doc_id=b.block_id or str(i),text=b.text,
            corpus_revision=adapter.prepared.source_revision)
            for i,b in enumerate(state.context)))
        tools=SchemaOnlyToolConnector((ToolSpec(name='search',description='Search the supplied visible evidence passages.',
            input_schema={'type':'object','properties':{'query':{'type':'string'}},'required':['query']},read_only=True),))
        result={'job':job,'package_location':__import__('agentinputlab').__file__,
            'model_binding':source,'input_state_hash':state.state_hash,'additional_attempts':1,'automatic_retry':False,
            'prepared_file_hashes':dict(adapter.prepared.file_sha256),'retriever':retriever.fingerprint()}
        try:
            method=create_intervention(job['release'],**constructors[job['release']])
            record=InterventionRunner().run(method,state,InterventionContext(backend=backend,retriever=retriever,tools=tools,seed=17))
            state=record.patch.apply(state);stage='downstream_generation'
            used=sum(c.get('response',{}).get('usage',{}).get('output_tokens',0) for c in calls)
            if len(calls)>=8 or used>=4096:raise RuntimeError('shared budget exhausted before downstream')
            predictor=ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                benchmark_id=job['benchmark'],split=job['split'],model_slot=model['id'],seed=17,
                builder_parameters={'schema_version':1,'task':'qa','qa':{'output_format':'structured_json',
                    'max_tokens':512,'use_seed':True}},runtime=StaticRuntimeComponents(backend=backend)))
            prediction=predictor.predict(replace(state,budget=replace(state.budget,
                max_model_calls=8-len(calls),max_output_tokens=4096-used)))
            stage='scoring'
            result.update(status='scored_terminal',metrics=dict(adapter.score(example,prediction.prediction)),
                prediction=to_jsonable(prediction))
        except Exception as exc:result.update(status='failed_terminal',stage=stage,error_type=type(exc).__name__,error=str(exc))
        result.update(seconds=time.monotonic()-started,calls=calls,model_calls=len(calls),
            intervention=to_jsonable(record) if record else None)
        save(output/'terminals'/(job['id']+'.json'),result);(output.parent/'keep-services-alive').touch()
    def lane(model):
        with ThreadPoolExecutor(5) as pool:list(pool.map(one,[j for j in jobs if j['model']['id']==model['id']]))
    with ThreadPoolExecutor(4) as pool:list(pool.map(lane,config['models']))
    terms=[json.loads(p.read_text()) for p in (output/'terminals').glob('*.json')]
    save(output/'complete.json',{'expected':len(jobs),'terminal':len(terms),
        'scored':sum(t['status']=='scored_terminal' for t in terms),'full_toolkit_acceptance_complete':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream-root',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True);execute(json.loads(a.config.read_text()),a.output,a.upstream_root)
if __name__=='__main__':main()
