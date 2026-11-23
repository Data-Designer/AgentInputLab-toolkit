"""Exercise six pinned LongBench adapters with installed models and compression.

All data, helper checkpoints, model bindings and receipts are operator mounts.
Deterministic backbone-independent compression is prepared once per visible
state, then each downstream backbone runs independently. No dataset references
are presented to either the compressor or the downstream model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.backends import GenerationRequest
from agentinputlab.benchmarks import LongBenchAdapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.runner import InterventionRunner
from run_static_acceptance import RecordingBackend, digest, local_transport, save

SUBSETS=('longbench.qasper','longbench.multifieldqa_en','longbench.govreport',
         'longbench.qmsum','longbench.passage_retrieval_en','longbench.repobench_p')
COMPRESSOR='curate.llmlingua2.xlm_roberta_official'


def execute(config, root, repair_root=None):
    from transformers import AutoTokenizer
    import torch
    torch.set_num_threads(config.get('cpu_threads',4))
    assets=Path(config['source_root'])
    adapters={name:LongBenchAdapter(benchmark_id=name,data_zip=assets/'data.zip',
        scorer_path=assets/'metrics.py',prompts_path=assets/'dataset2prompt.json') for name in SUBSETS}
    examples={name:tuple(sorted(adapter.examples('test'),key=lambda e:e.example_id)[:2])
              for name,adapter in adapters.items()}
    models=config['models'];jobs=[]
    for model in models:
        for name in SUBSETS:
            for example in examples[name]:
                for release in ('baseline.no_op',COMPRESSOR):
                    job={'model':model,'benchmark':name,'case_id':example.example_id,
                         'release':release,'seed':17,'output_tokens_per_call':512,
                         'total_output_tokens':4096,'max_calls':8,
                         'max_input_tokens':config.get('max_input_tokens',24000)}
                    job['id']=digest(job);jobs.append(job)
    repairs={}
    if repair_root is not None:
        for path in (repair_root/'terminals').glob('*.json'):
            row=json.loads(path.read_text())
            if (row['status']=='failed_terminal' and row.get('model_calls')==0
                    and row.get('stage')=='compression'
                    and row.get('error')=='ValidationError: runtime-dependency inspection exceeded 4096 objects'):
                repairs[row['job']['id']]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        jobs=[job for job in jobs if job['id'] in repairs]
        if not jobs:raise ValueError('no verified pre-compression/pre-model guard failure to resume')
    for directory in ('claims','terminals','compression'):(root/directory).mkdir(exist_ok=True)
    save(root/'manifest.json',{'jobs':jobs,'data':{n:a.fingerprint() for n,a in adapters.items()},
        'paper_results_changed':False,'automatic_retry':False,
        'preprocessing':'official_middle_truncation_with_each_downstream_tokenizer',
        'compression_reuse':'same_visible_state_and_pinned_backbone_independent_classifier',
        'zero_call_repair_sources':repairs})
    tokenizers={};tokenizer_locks={};binding={}
    for model in models:
        source=json.loads((Path(model['service_directory'])/'started.json').read_text())
        binding[model['id']]={'revision':digest(source['model_json_hashes']),
            'model_json_hashes':source['model_json_hashes'],'service_identity':source['identity']}
        tokenizers[model['id']]=AutoTokenizer.from_pretrained(model['tokenizer_path'],
            local_files_only=True,trust_remote_code=False)
        tokenizer_locks[model['id']]=threading.Lock()
    helper=None

    def one(job, prepared=None):
        target=root/'terminals'/(job['id']+'.json');claim=root/'claims'/(job['id']+'.json')
        if target.exists() or claim.exists():return
        save(claim,{'job':job,'started':time.time()})
        model=job['model'];adapter=adapters[job['benchmark']]
        example=next(e for e in examples[job['benchmark']] if e.example_id==job['case_id'])
        state=example.visible_state(budget=ExecutionBudget(max_output_tokens=4096,max_model_calls=8))
        result={'job':job,'package_location':__import__('agentinputlab').__file__,
            'data_binding':adapter.fingerprint(),'model_binding':binding[model['id']],
            'input_state_hash':state.state_hash,'automatic_retry':False,'additional_attempts':1}
        calls=[];started=time.monotonic();stage='compression'
        try:
            if prepared:
                result['compression_receipt']=prepared['receipt']
                if prepared.get('error'):raise RuntimeError(prepared['error'])
                # The actual returned patch is applied to the exact same state,
                # not reconstructed from lossy text or a different backbone.
                state=prepared['record'].patch.apply(state)
            stage='native_prompt_and_middle_truncation'
            prompt=adapter.render(state);original_hash=hashlib.sha256(prompt.encode()).hexdigest()
            with tokenizer_locks[model['id']]:
                tokenizer=tokenizers[model['id']]
                tokens=tokenizer.encode(prompt,add_special_tokens=False)
                before=len(tokens);limit=job['max_input_tokens']
                if before>limit:
                    half=limit//2
                    prompt=tokenizer.decode(tokens[:half],skip_special_tokens=True)+tokenizer.decode(tokens[-half:],skip_special_tokens=True)
            result['prompt']={'before_sha256':original_hash,'after_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'tokens_before':before,'max_input_tokens':limit,'truncated':before>limit}
            backend=RecordingBackend(calls=calls,model_id=model['id'],base_url=model['base_url'],
                deployment_identity=binding[model['id']],timeout_seconds=300,
                extra_body=model.get('extra_body',{}),transport=local_transport)
            stage='downstream_generation'
            response=backend.generate(GenerationRequest(prompt=prompt,max_tokens=512,seed=17))
            stage='native_scoring'
            result.update(status='scored_terminal',metrics=dict(adapter.score(example,response.text)),
                prediction=response.text,empty_final=not response.text.strip(),finish_reason=response.finish_reason)
        except Exception as exc:
            result.update(status='failed_terminal',stage=stage,error_type=type(exc).__name__,error=str(exc))
        result.update(seconds=time.monotonic()-started,calls=calls,model_calls=len(calls),
            output_tokens=sum(c.get('response',{}).get('usage',{}).get('output_tokens',0) for c in calls))
        save(target,result);(root.parent/'keep-services-alive').touch()

    # Queue native baselines immediately. CPU compressor preparation proceeds
    # while these GPU requests execute; compression never holds a GPU pool.
    pools={m['id']:ThreadPoolExecutor(3) for m in models};futures=[]
    try:
        for job in jobs:
            if job['release']=='baseline.no_op':futures.append(pools[job['model']['id']].submit(one,job))
        for name in SUBSETS:
            for example in examples[name]:
                if not any(j['release']==COMPRESSOR and j['benchmark']==name and j['case_id']==example.example_id for j in jobs):continue
                state=example.visible_state(budget=ExecutionBudget(max_output_tokens=4096,max_model_calls=8))
                path=root/'compression'/(digest({'state':state.state_hash,'release':COMPRESSOR})+'.json')
                started=time.monotonic();prepared={'receipt':str(path)}
                receipt={'input_state_hash':state.state_hash,'release':COMPRESSOR,
                    'helper_path':config['compressor_path'],'seed':17,'backbone_independent':True,
                    'reference_exposed':False,'shared_preparation_count':1}
                try:
                    if helper is None:helper=create_intervention(COMPRESSOR,model_name=config['compressor_path'],
                        device_map='cpu',model_config={'local_files_only':True})
                    record=InterventionRunner().run(helper,state,InterventionContext(seed=17))
                    prepared['record']=record;receipt['record']=to_jsonable(record)
                except Exception as exc:
                    prepared['error']=f'{type(exc).__name__}: {exc}';receipt['error']=prepared['error']
                receipt['seconds']=time.monotonic()-started;save(path,receipt)
                for job in jobs:
                    if job['release']==COMPRESSOR and job['benchmark']==name and job['case_id']==example.example_id:
                        futures.append(pools[job['model']['id']].submit(one,job,prepared))
        for future in futures:future.result()
    finally:
        for pool in pools.values():pool.shutdown(wait=True)
    terminals=[json.loads(p.read_text()) for p in (root/'terminals').glob('*.json')]
    save(root/'complete.json',{'expected':len(jobs),'terminal':len(terminals),
        'scored':sum(r['status']=='scored_terminal' for r in terminals),
        'failed':sum(r['status']=='failed_terminal' for r in terminals),
        'full_toolkit_acceptance_complete':False,'paper_results_changed':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repair-root',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True)
    execute(json.loads(args.config.read_text()),args.output,args.repair_root)


if __name__=='__main__':main()
