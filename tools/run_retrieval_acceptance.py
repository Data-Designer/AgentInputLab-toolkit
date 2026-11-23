"""Fixed real retrieval cases with native QueryGym and source-locked RRR."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from run_static_acceptance import RecordingBackend, digest, local_transport, save
from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.benchmarks.retrieval import RetrievalPrediction
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.runner import InterventionRunner
from agentinputlab.algorithms.reformulate.query2doc import PinnedQueryGymRuntime
from agentinputlab.algorithms.reformulate.rrr import RRRCommonSupportConfig
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from acceptance_method_bindings import EXTERNAL_REWRITERS, bind_external_rewriter
from agentinputlab.observability import RunMonitor

RELEASES=('baseline.no_op','rewrite.query2doc.querygym_live','rewrite.rrr.common_support_prompt')


def run(config, output, cell, upstream):
    adapter=load_production_benchmark_adapter(cell,config['prepared_root'])
    documents=[]
    with adapter.corpus_path.open() as handle:
        for line in handle:
            row=json.loads(line)
            documents.append(Document(doc_id=row['id'],text=row['text'],title=row.get('title',''),
                                      corpus_revision=adapter.corpus_prepared.source_revision))
    retriever=InMemoryBM25Retriever(documents=tuple(documents))
    prompt_path=upstream/'upstream-paper/RAG-query-rewriting/generate/inprompts/myprompt.jsonl'
    raw=prompt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='1483b80e5b81a26b97641cc2a5c4535c045eb39394d2b88651aade0d4afa5fb4':
        raise ValueError('RRR source prompt differs')
    prompts=[json.loads(line) for line in raw.splitlines()]
    rewrite=next(r['prompt'] for r in prompts if r.get('task')=='rewrite' and r.get('pid')==2)
    reader=next(r['prompt'] for r in prompts if r.get('task')=='step2' and r.get('pid')==1)
    rrr_config=RRRCommonSupportConfig(rewrite_prompt=rewrite,reader_prompt=reader,
        prompt_source_sha256=hashlib.sha256(raw).hexdigest(),
        rewrite_prompt_sha256=content_hash(rewrite),reader_prompt_sha256=content_hash(reader))
    output.mkdir(parents=True,exist_ok=True)
    (output/'claims').mkdir();(output/'terminals').mkdir()
    jobs=[]
    examples=sorted(adapter.examples('test'),key=lambda e:e.example_id)[:2]
    for model in config['models']:
        for example in examples:
            for release in config.get('retrieval_releases',RELEASES):
                if release not in (*RELEASES,*EXTERNAL_REWRITERS):raise ValueError('unknown retrieval acceptance release')
                job={'model':model,'release':release,'case_id':example.example_id,'benchmark':cell,
                     'seed':17,'budget':{'max_model_calls':8,'max_output_tokens':4096},
                     'stop_policy':'gpt-oss.final-channel-stop.v1' if model['id']=='gpt-oss-20b' else 'native'}
                job['id']=digest(job);jobs.append(job)
    save(output/'manifest.json',{'jobs':jobs,'corpus_file_hashes':dict(adapter.corpus_prepared.file_sha256),
        'query_file_hashes':dict(adapter.prepared.file_sha256),'index':retriever.fingerprint(),
        'source_prompt_sha256':hashlib.sha256(raw).hexdigest(),'automatic_retry':False})

    def one(job):
        save(output/'claims'/(job['id']+'.json'),job)
        calls=[];model=job['model'];service=Path(model['service_directory'])
        while not (service/'ready.json').exists():
            if (service/'attention.json').exists():raise RuntimeError('service not available')
            time.sleep(5)
        source=json.loads((service/'started.json').read_text())
        binding={'revision':digest(source['model_json_hashes']),'server':'vllm'}
        inner=RecordingBackend(calls=calls,model_id=model['id'],base_url=model['base_url'],
            deployment_identity=binding,timeout_seconds=180,extra_body=model.get('extra_body',{}),transport=local_transport)
        backend=GPTOSSFinalChannelStopBackend(inner) if model['id']=='gpt-oss-20b' else inner
        example=next(e for e in examples if e.example_id==job['case_id'])
        state=example.visible_state(budget=ExecutionBudget(**job['budget']))
        result={'job':job,'package_location':__import__('agentinputlab').__file__,'model_binding':source,
                'additional_attempts':1,'automatic_retry':False};record=None
        started=time.monotonic();stage='intervention';monitor=RunMonitor()
        try:
            if job['release']!='baseline.no_op':
                if job['release']=='rewrite.query2doc.querygym_live':
                    method=create_intervention(job['release'])
                    runtime=PinnedQueryGymRuntime(upstream/'upstream/QueryGym',backend)
                elif job['release'] in EXTERNAL_REWRITERS:
                    # These external workers have their own bounded HTTP
                    # transport and must bind the exact unwrapped fingerprint.
                    backend=inner
                    method,runtime=bind_external_rewriter(job['release'],upstream=upstream,
                        backend=backend,model=model,dataset_revision=adapter.prepared.source_revision,benchmark=cell)
                    initial=retriever.retrieve(state.query,top_k=3)
                    from dataclasses import replace
                    from agentinputlab.contracts import ContextBlock
                    state=replace(state,context=tuple(ContextBlock(text=r.document.text,block_id=r.document.doc_id,
                        source=r.document.corpus_revision) for r in initial))
                else:
                    method=create_intervention(job['release'],config=rrr_config);runtime=None
                record=InterventionRunner(monitor=monitor).run(method,state,InterventionContext(
                    backend=backend,retriever=retriever,external_runtime=runtime,seed=17))
                state=record.patch.apply(state)
            stage='retrieval_and_scoring'
            ranked=retriever.retrieve(state.query,top_k=100)
            metrics=dict(adapter.score(example,RetrievalPrediction.from_ranked_ids([r.document.doc_id for r in ranked])))
            result.update(status='scored_terminal',metrics=metrics,ranked_ids=[r.document.doc_id for r in ranked])
        except Exception as exc:
            result.update(status='failed_terminal',stage=stage,error_type=type(exc).__name__,error=str(exc))
        snapshots=monitor.list_snapshots()
        result.update(seconds=time.monotonic()-started,calls=calls,recorded_backend_calls=len(calls),
                      model_calls=snapshots[-1].usage.model_calls if snapshots else len(calls),
                      monitor_snapshots=to_jsonable(snapshots),
                      intervention=to_jsonable(record) if record else None)
        save(output/'terminals'/(job['id']+'.json'),result)
        (output.parent/'keep-services-alive').touch()

    def per_model(model):
        with ThreadPoolExecutor(config.get('concurrency_per_model',3)) as pool:list(pool.map(one,[j for j in jobs if j['model']['id']==model['id']]))
    with ThreadPoolExecutor(4) as pool:list(pool.map(per_model,config['models']))
    terms=[json.loads(p.read_text()) for p in (output/'terminals').glob('*.json')]
    save(output/'complete.json',{'expected':len(jobs),'terminal':len(terms),
         'scored':sum(t['status']=='scored_terminal' for t in terms),'full_toolkit_acceptance_complete':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',required=True);parser.add_argument('--upstream-root',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.config.read_text()),args.output,args.benchmark,args.upstream_root)
if __name__=='__main__':main()
