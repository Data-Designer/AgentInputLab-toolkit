"""Run fixed real QA cases through an installed toolkit and real model services.

Configuration and all receipts belong outside the public source tree. This first
execution surface covers the packaged QA adapters and two planning profiles;
the full compatibility matrix remains a separate release requirement.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.backends import GenerationRequest, OpenAICompatibleBackend
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.runner import InterventionRunner
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.experiments.drivers.static_production import StaticPredictorBuildRequest, StaticRuntimeComponents

RELEASES = ('baseline.no_op', 'plan.plan_and_solve.ps_prompt',
            'plan.plan_and_budget.native_regenerated_soft_budget')
QA_CELLS = (('musique.answerable','dev'), ('2wiki.multihop','dev'),
            ('hotpotqa.distractor','validation'), ('qasper.v0_3_dev','dev'))


def selected_releases(config):
    """Select exact supported profiles without changing the frozen default batch."""
    releases = config.get('releases', RELEASES)
    allowed = {*RELEASES, 'plan.plan_and_solve.ps_plus_prompt', 'plan.plan_and_solve.plan_component'}
    if (not isinstance(releases, (list, tuple)) or not releases
            or any(not isinstance(release, str) or release not in allowed for release in releases)
            or len(releases) != len(set(releases))):
        raise ValueError('acceptance releases must be unique supported release identities')
    return tuple(releases)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path, value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(to_jsonable(value),handle,indent=2,sort_keys=True)


class RecordingBackend(OpenAICompatibleBackend):
    def __init__(self, *, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls=calls

    def generate(self, request):
        row={'request':to_jsonable(request),'request_hash':request.request_hash,
             'started':datetime.now(timezone.utc).isoformat()}
        self.calls.append(row)
        try:
            response=super().generate(request)
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc))
            raise
        row['response']=to_jsonable(response)
        return response


def execute(config, root, repair_zero_call_root=None):
    releases = selected_releases(config)
    adapters={cell:load_production_benchmark_adapter(cell,config['prepared_root']) for cell,_ in QA_CELLS}
    jobs=[]
    for model in config['models']:
        for cell,split in QA_CELLS:
            adapter=adapters[cell]
            # Fixed lexicographic IDs depend only on dataset identity, never scores.
            offset=config.get('case_offset',0)
            examples=sorted(adapter.examples(split),key=lambda e:e.example_id)[offset:offset+2]
            for example in examples:
                for release in releases:
                    row={'model':model,'benchmark':cell,'split':split,
                         'case_id':example.example_id,'release':release,'seed':17,
                         'budget':{'max_model_calls':8,'max_output_tokens':4096},
                         'per_call_tokens':512}
                    row['predictor']=config.get('predictor','answer_only')
                    row['constructor_parameters']=({'decomposition_max_tokens':512,'assessment_max_tokens':512}
                        if release=='plan.plan_and_budget.native_regenerated_soft_budget' else {})
                    row['id']=digest(row);jobs.append(row)
    repair_sources={}
    if repair_zero_call_root is not None:
        selected=set()
        for path in (repair_zero_call_root/'terminals').glob('*.json'):
            prior=json.loads(path.read_text())
            if (prior['status']=='failed_terminal' and prior.get('model_calls')==0
                    and 'deadlock detected by _ModuleLock' in prior.get('error','')):
                prior_job=prior['job']
                selected.add((prior_job['model']['id'],prior_job['benchmark'],prior_job['case_id'],prior_job['release']))
                repair_sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        jobs=[job for job in jobs if (job['model']['id'],job['benchmark'],job['case_id'],job['release']) in selected]
        if not jobs:raise ValueError('no verified zero-call cold-import failure to supplement')
    (root/'claims').mkdir(exist_ok=True);(root/'terminals').mkdir(exist_ok=True)
    manifest=root/'manifest.json'
    expected={'schema_version':1,'phase':'installed_qa_real_models_v1','jobs':jobs,
              'config_hash':digest(config),'automatic_retry':False,'paper_results_changed':False,
              'zero_call_repair_sources':repair_sources}
    if manifest.exists():
        if json.loads(manifest.read_text())!=expected:raise ValueError('saved manifest differs')
    else:save(manifest,expected)

    def one(job):
        terminal=root/'terminals'/(job['id']+'.json')
        if terminal.exists():return
        claim=root/'claims'/(job['id']+'.json')
        if claim.exists():return  # unresolved consumed attempt requires review, never automatic retry
        save(claim,{'job':job,'pid':os.getpid(),'started':time.time()})
        model=job['model'];service=Path(model['service_directory'])
        while not (service/'ready.json').exists():
            if (service/'attention.json').exists():
                save(terminal,{'job':job,'status':'service_unavailable_no_inference','model_calls':0});return
            (root.parent/'keep-services-alive').touch()
            time.sleep(5)
        calls=[]
        source=json.loads((service/'started.json').read_text())
        binding={'revision':digest(source['model_json_hashes']),
                 'model_json_hashes':source['model_json_hashes'],'service_identity':source['identity']}
        backend=RecordingBackend(calls=calls,model_id=model['id'],base_url=model['base_url'],
            deployment_identity=binding,timeout_seconds=180,
            extra_body=model.get('extra_body',{}),
            # An explicitly local transport avoids inherited user HTTP proxy configuration.
            transport=local_transport)
        adapter=adapters[job['benchmark']];example=adapter.example(job['split'],job['case_id'])
        state=example.visible_state(budget=ExecutionBudget(**job['budget']))
        record=None;answer=None;score=None;stage='intervention';started=time.monotonic()
        result={'job':job,'package_location':__import__('agentinputlab').__file__,
                'model_binding':binding,'input_state_hash':state.state_hash,
                'prepared_file_hashes':dict(adapter.prepared.file_sha256),
                'additional_attempts':1,'automatic_retry':False}
        try:
            if job['release']!='baseline.no_op':
                method=create_intervention(job['release'],**job['constructor_parameters'])
                record=InterventionRunner().run(method,state,InterventionContext(backend=backend,seed=17))
                state=record.patch.apply(state)
            stage='downstream_generation'
            used_tokens=sum(c.get('response',{}).get('usage',{}).get('output_tokens',0) for c in calls)
            if len(calls)>=8 or used_tokens>=4096:raise RuntimeError('shared run budget exhausted before downstream')
            if job['predictor']=='production_structured_json':
                runtime=StaticRuntimeComponents(backend=backend)
                predictor=ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                    benchmark_id=job['benchmark'],split=job['split'],model_slot=model['id'],seed=17,
                    builder_parameters={'schema_version':1,'task':'qa','qa':{
                        'output_format':'structured_json','max_tokens':512,'use_seed':True}},runtime=runtime))
                prediction=predictor.predict(replace(state,budget=replace(state.budget,
                    max_model_calls=8-len(calls),max_output_tokens=4096-used_tokens)))
                answer=prediction.prediction.answer;stage='scoring'
                score=dict(adapter.score(example,prediction.prediction))
                result['downstream_prediction']=to_jsonable(prediction)
                finish_reason=calls[-1]['response']['finish_reason']
            else:
                context='\n\n'.join(f'[{b.block_id}] {b.text}' for b in state.context)
                prompt=f'{state.instruction}\n\nContext:\n{context}\n\nQuestion: {state.query}\nAnswer:'
                response=backend.generate(GenerationRequest(prompt=prompt,max_tokens=min(512,4096-used_tokens),seed=17))
                answer=response.text;stage='scoring'
                score=dict(adapter.score(example,QAPrediction(answer=answer)))
                finish_reason=response.finish_reason
            result.update(status='scored_terminal',metrics=score,answer=answer,
                          empty_final=not answer.strip(),finish_reason=finish_reason)
        except Exception as exc:
            result.update(status='failed_terminal',stage=stage,error_type=type(exc).__name__,error=str(exc))
        result.update(seconds=time.monotonic()-started,calls=calls,
                      model_calls=len(calls),output_tokens=sum(c.get('response',{}).get('usage',{}).get('output_tokens',0) for c in calls),
                      intervention=to_jsonable(record) if record else None)
        save(terminal,result)
        (root.parent/'keep-services-alive').touch()

    # Three callers per model; independent model pools keep all four GPUs busy.
    def model_jobs(model):
        selected=[j for j in jobs if j['model']['id']==model['id']]
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,selected))
    with ThreadPoolExecutor(max_workers=len(config['models'])) as pool:
        list(pool.map(model_jobs,config['models']))
    terminals=[json.loads(p.read_text()) for p in (root/'terminals').glob('*.json')]
    save(root/'complete.json',{'expected':len(jobs),'terminal':len(terminals),
        'scored':sum(r['status']=='scored_terminal' for r in terminals),
        'nonempty_scored':sum(r['status']=='scored_terminal' and not r['empty_final'] for r in terminals),
        'failed':sum(r['status']=='failed_terminal' for r in terminals),
        'full_toolkit_acceptance_complete':False,'paper_results_changed':False})


def local_transport(url,headers,payload,timeout):
    request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=dict(headers))
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=timeout) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repair-zero-call-root',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    execute(json.loads(args.config.read_text()),args.output,args.repair_zero_call_root)

if __name__=='__main__':main()
