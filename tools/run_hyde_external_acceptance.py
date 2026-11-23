"""Two fixed native ArgUana retrieval endpoints per reader, eight draws each.

All source, data, E5 weights and one shared corpus index are operator mounts.
Native qrels never cross the method boundary; no LLM judge or answer call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.backends import Document
from agentinputlab.backends.retrieval import RetrievalResult
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.retrieval import RetrievalPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.integrations.full_architecture.hyde import HyDEPatchedSourceRuntime
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASE="acquire.hyde.official_external_patched"


def file_sha(path):
    digestor=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digestor.update(block)
    return digestor.hexdigest()


class FrozenE5Index:
    """Actual CPU query encoder against once-built, pinned GPU corpus vectors."""
    embedder_id="operator.e5.native-hyde.query"
    searcher_id="operator.e5.native-hyde.arguana-index"

    def __init__(self,preparation,adapter):
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        self.receipt=json.loads(Path(preparation).read_text())
        r=self.receipt
        assert r['document_count']==8674 and r['dimensions']==1024 and r['normalization']=='l2-float64-after-model-float32'
        assert dict(adapter.corpus_prepared.file_sha256)==r['corpus_files']
        self.documents=[]
        with adapter.corpus_path.open() as f:
            for line in f:
                row=json.loads(line)
                self.documents.append(Document(doc_id=row['id'],text=row['text'],title=row.get('title',''),
                    corpus_revision=adapter.corpus_prepared.source_revision))
        assert [d.doc_id for d in self.documents]==r['ordered_document_ids']
        path=Path(r['vectors_path'])
        assert file_sha(path)==r['vectors_sha256']
        self.vectors=np.load(path,mmap_mode='r',allow_pickle=False)
        assert self.vectors.shape==(8674,1024) and self.vectors.dtype==np.float64
        mount=Path(r['encoder_path'])
        for name,old in r['encoder_files'].items():
            p=mount/name
            stat=p.stat()
            assert [stat.st_ino,stat.st_size,stat.st_mtime_ns]==old['stat'],name
            # Full weight SHA was verified in the shared preparation. Rehash
            # small loader/tokenizer files, not 2.2GB four times unchanged.
            if stat.st_size<30_000_000:
                assert file_sha(p)==old['sha256'],name
        torch.set_num_threads(1)
        self.model=SentenceTransformer(str(mount),device='cpu',local_files_only=True,trust_remote_code=False)
        self.lock=threading.Lock()
        self.identity=dict(implementation='pinned-e5-cosine-native-hyde.v1',helper_model=r['encoder_model'],
            helper_revision=r['encoder_revision'],helper_files={k:v['sha256'] for k,v in r['encoder_files'].items()},
            query_prefix='query: ',document_prefix='passage: ',vector_sha256=r['vectors_sha256'],
            ordered_corpus_ids_sha256=digest(r['ordered_document_ids']),corpus_files=r['corpus_files'],
            metric='cosine-float64',dimensions=1024,query_device='cpu',index_encoding_device=r['selected_gpu'],
            author_contriever_checkpoint_parity_claimed=False)

    def fingerprint(self):
        return self.identity

    def embed(self,texts):
        import numpy as np
        with self.lock:
            raw=self.model.encode(['query: '+t for t in texts],batch_size=len(texts),
                normalize_embeddings=False,show_progress_bar=False,convert_to_numpy=True)
        values=np.asarray(raw,dtype=np.float64)
        norms=np.linalg.norm(values,axis=1,keepdims=True)
        if not np.isfinite(values).all() or (norms<=0).any():
            raise ValueError('invalid real E5 query vectors')
        return tuple(tuple(float(x) for x in row) for row in values/norms)

    def search(self,vector,*,top_k):
        import numpy as np
        query=np.asarray(vector,dtype=np.float64)
        if query.shape!=(1024,) or not np.isfinite(query).all() or np.linalg.norm(query)<=0:
            raise ValueError('invalid native HyDE mean vector')
        scores=self.vectors@(query/np.linalg.norm(query))
        ordered=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.documents[i].doc_id))[:top_k]
        return tuple(RetrievalResult(self.documents[i],score=float(scores[i]),rank=rank,
            features={'cosine_score':float(scores[i])}) for rank,i in enumerate(ordered,1))


def run(config,output):
    assert len(config['models'])==1
    model=config['models'][0]
    binding=verify_services([model])[model['id']]
    adapter=load_production_benchmark_adapter('beir.arguana',config['prepared_root'])
    index=FrozenE5Index(config['index_preparation'],adapter)
    runtime=HyDEPatchedSourceRuntime(source_root=config['hyde_source_root'],patch_manifest=config['patch_manifest'],
        expected_manifest_sha256=config['patch_manifest_sha256'])
    native=runtime.fingerprint()
    selected=sorted(adapter.examples('test'),key=lambda e:e.example_id)[84:86]
    assert len(selected)==2
    jobs=[]
    for ex in selected:
        state=ex.visible_state(budget=ExecutionBudget(max_model_calls=8,max_output_tokens=4096))
        job=dict(model=model,release=RELEASE,benchmark='beir.arguana',case_id=ex.example_id,seed=17,
            budget={'max_model_calls':8,'max_output_tokens':4096},per_draw_output_cap=512,
            # One extra hit preserves 100 after native BEIR removes a possible
            # identical query/document ID. This is fixed before any outputs.
            config={'task':'arguana','num_hypotheses':8,'max_tokens':512,'temperature':0.7,'top_k':101,'stop':['\n\n\n']},
            input_state_hash=state.state_hash,source_native_runtime=native,index_fingerprint=index.fingerprint())
        job['id']=digest(job)
        jobs.append(job)
    output.mkdir(exist_ok=False)
    for folder in ('claims','terminals','model-outcomes'):
        (output/folder).mkdir()
    save(output/'manifest.json',dict(jobs=jobs,model_binding=binding,corpus_files=dict(adapter.corpus_prepared.file_sha256),
        query_files=dict(adapter.prepared.file_sha256),source_selection='lexicographic_source_indices84_85_fixed_before_outputs',
        shared_index_preparation_sha256=file_sha(config['index_preparation']),native_promptor_task='arguana',
        native_source_e2e_entrypoint=True,source_redistributed=False,automatic_retry=False,paper_results_changed=False,
        backend_sampling_defaults_are_operator_owned=True,no_legacy_logprob_reordering=True))

    def one(job):
        save(output/'claims'/(job['id']+'.json'),dict(job=job,pid=os.getpid(),started=time.time()))
        calls=[]
        record=None
        monitor=RunMonitor()
        started=time.monotonic()
        backend=RecordingBackend(calls=calls,model_id=model['id'],base_url=model['base_url'],
            extra_body=model.get('extra_body',{}),timeout_seconds=240,transport=local_transport,
            deployment_identity={'revision':digest(binding['model_json_hashes']),'server':'vllm'})
        ex=next(e for e in selected if e.example_id==job['case_id'])
        row=dict(job=job,model_binding=binding,package_location=__import__('agentinputlab').__file__,
            additional_attempts=1,automatic_retry=False)
        try:
            state=ex.visible_state(budget=ExecutionBudget(**job['budget']))
            assert state.state_hash==job['input_state_hash']
            record=InterventionRunner(monitor=monitor).run(create_intervention(RELEASE,config=job['config']),state,
                InterventionContext(backend=backend,dense_embedder=index,vector_searcher=index,external_runtime=runtime,seed=17))
            ids=list(record.patch.trace.events[-1].payload['document_ids'])
            row.update(status='retrieval_returned',record=to_jsonable(record),ranked_ids=ids)
        except Exception as exc:
            row.update(status='failed_terminal',stage='patched_native_hyde',error_type=type(exc).__name__,error=str(exc))
        save(output/'model-outcomes'/(job['id']+'.json'),{**row,'calls':calls,
            'monitor_snapshots':to_jsonable(monitor.list_snapshots()),'native_scoring_started':False})
        if record is not None:
            try:
                row.update(status='scored_terminal',metrics=dict(adapter.score(ex,RetrievalPrediction.from_ranked_ids(row['ranked_ids']))))
            except Exception as exc:
                row.update(status='failed_terminal',stage='source_native_IR_scoring',error_type=type(exc).__name__,error=str(exc))
        row.update(calls=calls,provider_request_attempts=len(calls),monitor_snapshots=to_jsonable(monitor.list_snapshots()),
            seconds=time.monotonic()-started)
        assert len(calls)<=8
        save(output/'terminals'/(job['id']+'.json'),row)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one,jobs))
    rows=[json.loads(p.read_text()) for p in (output/'terminals').glob('*.json')]
    assert len(rows)==2
    save(output/'complete.json',dict(expected_positions=2,terminal_positions=2,
        scored=sum(r['status']=='scored_terminal' for r in rows),failed=sum(r['status']=='failed_terminal' for r in rows),
        physical_provider_requests=sum(r['provider_request_attempts'] for r in rows),
        original_source_native_e2e_executed=True,official_IR_scorer=True,automatic_retry=False,publication_ready=False))
    print(model['id'],'two native HyDE retrieval positions closed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.output)
