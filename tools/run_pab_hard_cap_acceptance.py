"""Once-only fixed QA/LongBench endpoints for the explicit hard-cap extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_longbench_acceptance import SUBSETS
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ContextBlock, ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks import LongBenchAdapter, load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import CompositeInterventionRunner

RELEASE="plan.plan_and_budget.hard_cap_extension"
OUTPUT_FORMAT="Return only the concise final answer to the overall question, without explanation."


def bounded_state(example,tokenizer,limit=24000):
    source=example.visible_state(budget=ExecutionBudget(max_model_calls=8,max_output_tokens=4096))
    text="\n\n".join((f"[{b.block_id}] " if b.block_id else "")+b.text
        +(f"\nSource: {b.source}" if b.source else "") for b in source.context)
    encoded=tokenizer.encode(text,add_special_tokens=False).ids
    if len(encoded)<=limit:
        return source,source,dict(context_tokens_before=len(encoded),middle_truncation=False,limit=limit)
    half=limit//2
    reduced=tokenizer.decode(encoded[:half],skip_special_tokens=True)+tokenizer.decode(encoded[-half:],skip_special_tokens=True)
    state=replace(source,context=(ContextBlock(reduced,block_id="bounded_visible_reference",
        source="source_first_and_last_context_tokens"),))
    return source,state,dict(context_tokens_before=len(encoded),middle_truncation=True,limit=limit,
        resulting_text_sha256=hashlib.sha256(reduced.encode()).hexdigest(),
        protocol="method-visible-context-middle-truncation.v1",native_prompt_truncation_parity_claimed=False)


def run(config,output,*,preflight=False):
    from tokenizers import Tokenizer
    if len(config["models"])!=1:
        raise ValueError("one exact source-bound model per worker")
    model=config["models"][0]
    tokenizer_path=Path(model["tokenizer_path"])/"tokenizer.json"
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    adapters={cell:load_production_benchmark_adapter(cell,config["prepared_root"]) for cell,_ in QA_CELLS}
    assets=Path(config["longbench_source_root"])
    adapters.update({cell:LongBenchAdapter(benchmark_id=cell,data_zip=assets/"data.zip",
        scorer_path=assets/"metrics.py",prompts_path=assets/"dataset2prompt.json") for cell in SUBSETS})
    cases=(*QA_CELLS,*((cell,"test") for cell in SUBSETS))
    jobs=[]
    states={}
    examples={}
    offset=config["case_offset"]
    if type(offset) is not int or offset<0:
        raise ValueError("predeclared lexicographic case offset required")
    for cell,split in cases:
        selected=sorted(adapters[cell].examples(split),key=lambda e:e.example_id)[offset:offset+2]
        if len(selected)!=2:
            raise ValueError("two source cases required for every selected benchmark")
        for ex in selected:
            source,state,window=bounded_state(ex,tokenizer)
            job=dict(model=model,benchmark=cell,split=split,case_id=ex.example_id,release=RELEASE,seed=17,
                budget={"max_model_calls":8,"max_output_tokens":4096},per_call_output_cap=512,
                constructor_parameters={"total_step_tokens":512,"per_step_max_tokens":512,"final_max_tokens":512,
                    "decomposition_max_tokens":512,"assessment_max_tokens":512,"schedule":"constant"},
                source_state_hash=source.state_hash,method_state_hash=state.state_hash,
                visible_context_window=window,tokenizer_json_sha256=tokenizer_sha256,
                output_format=OUTPUT_FORMAT,prediction="final_answer_only_no_extra_reader_call")
            job["id"]=digest(job)
            jobs.append(job)
            states[job["id"]]=state
            examples[job["id"]]=ex
    assert len(jobs)==20
    if preflight:
        print("20 source-visible states and original scorers ready; zero model calls",flush=True)
        return
    binding=verify_services([model])[model["id"]]
    assert model["tokenizer_json_sha256"]==tokenizer_sha256
    assert Path(binding["command"][binding["command"].index("--model")+1]).resolve()==tokenizer_path.parent.resolve()
    output.mkdir(exist_ok=False)
    for folder in ("claims","terminals","model-outcomes"):
        (output/folder).mkdir()
    save(output/"manifest.json",dict(jobs=jobs,model_binding=binding,
        sources={k:(dict(a.prepared.file_sha256) if hasattr(a,"prepared") else dict(a.fingerprint())) for k,a in adapters.items()},
        selection=f"lexicographic_indices_{offset}_{offset+1}_fixed_before_outputs",
        helper_models_used=False,hidden_labels_visible=False,automatic_retry=False,paper_results_changed=False,
        endpoint_scope="explicit_hard_cap_extension_not_original_soft_word_solver",
        context_window_scope="source-visible context only; not native full-prompt truncation parity"))

    def one(job):
        save(output/"claims"/(job["id"]+".json"),dict(job=job,pid=os.getpid(),started=time.time()))
        calls=[]
        record=None
        monitor=RunMonitor()
        started=time.monotonic()
        backend=RecordingBackend(calls=calls,model_id=model["id"],base_url=model["base_url"],
            extra_body=model.get("extra_body",{}),timeout_seconds=240,transport=local_transport,
            deployment_identity={"revision":digest(binding["model_json_hashes"]),"server":"vllm"})
        row=dict(job=job,model_binding=binding,package_location=__import__("agentinputlab").__file__,
            automatic_retry=False,additional_attempts=1)
        try:
            method=create_intervention(RELEASE,output_format=OUTPUT_FORMAT,decomposition_max_tokens=512,
                assessment_max_tokens=512,total_step_tokens=512,per_step_max_tokens=512,final_max_tokens=512)
            record=CompositeInterventionRunner(monitor=monitor).run(method,states[job["id"]],
                InterventionContext(backend=backend,seed=17))
            row.update(status="answer_returned",prediction=record.result.final_answer,composite=to_jsonable(record))
        except Exception as exc:
            row.update(status="failed_terminal",stage="hard_cap_workflow",error_type=type(exc).__name__,error=str(exc))
        # Raw outcomes/cost are durable before any scorer import/execution error.
        save(output/"model-outcomes"/(job["id"]+".json"),{**row,"calls":calls,
            "monitor_snapshots":to_jsonable(monitor.list_snapshots()),"native_scoring_started":False})
        if record is not None:
            try:
                prediction=record.result.final_answer
                qa_prediction=QAPrediction(answer=prediction.strip()) if job["benchmark"] not in SUBSETS else prediction
                row.update(status="scored_terminal",metrics=dict(adapters[job["benchmark"]].score(examples[job["id"]],qa_prediction)))
            except Exception as exc:
                row.update(status="failed_terminal",stage="source_scoring",error_type=type(exc).__name__,error=str(exc))
        row.update(calls=calls,model_calls=len(calls),monitor_snapshots=to_jsonable(monitor.list_snapshots()),seconds=time.monotonic()-started)
        assert len(calls)<=8 and all(c["request"]["max_tokens"]<=512 for c in calls)
        save(output/"terminals"/(job["id"]+".json"),row)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one,jobs))
    rows=[json.loads(p.read_text()) for p in (output/"terminals").glob("*.json")]
    assert len(rows)==20
    save(output/"complete.json",dict(expected_positions=20,terminal_positions=20,
        statuses=dict(Counter(r["status"] for r in rows)),physical_model_calls=sum(r["model_calls"] for r in rows),
        all_failures_and_consumed_calls_retained=True,whole_release_certified=False,publication_ready=False,automatic_retry=False))
    print(model["id"],"20 real hard-cap positions closed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if not args.preflight and args.output is None:
        parser.error("--output is required for execution")
    run(json.loads(args.config.read_text()),args.output,preflight=args.preflight)
