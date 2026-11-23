"""Fixed real native-source P&B answer workflow; no synthetic model fallback."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.integrations.full_architecture.plan_and_budget import (
    PlanAndBudgetNativePromptRuntime,
)
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import CompositeInterventionRunner

RELEASE = "plan.plan_and_budget.full"
OUTPUT_FORMAT = "Return only the concise final answer to the overall question, without explanation."


def run(config, output):
    if len(config["models"]) != 1:
        raise ValueError("one source-bound model per worker")
    bindings = verify_services(config["models"])
    runtime = PlanAndBudgetNativePromptRuntime(source_root=config["pab_source_root"])
    source_identity = runtime.fingerprint()
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"])
                for cell, _ in QA_CELLS}
    offset = config["case_offset"]
    if type(offset) is not int or offset < 0:
        raise ValueError("fixed source offset required")
    jobs = []
    for cell, split in QA_CELLS:
        examples = sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[offset:offset+2]
        if len(examples) != 2:
            raise ValueError("two predeclared source cases required")
        for example in examples:
            job = dict(model=config["models"][0], benchmark=cell, split=split,
                case_id=example.example_id, release=RELEASE, seed=17,
                budget={"max_model_calls": 8, "max_output_tokens": 4096},
                per_call_output_cap=512, output_format=OUTPUT_FORMAT,
                prediction_protocol="native_answer_only_no_second_reader_call",
                original_support_and_evidence_metrics_not_separately_elicited=True)
            job["id"] = digest(job)
            jobs.append(job)
    output.mkdir(exist_ok=False)
    for folder in ("claims", "terminals"):
        (output/folder).mkdir()
    save(output/"manifest.json", dict(jobs=jobs, native_prompt_runtime=source_identity,
        model_bindings=bindings, source_hashes={k: dict(a.prepared.file_sha256) for k,a in adapters.items()},
        source_selection=f"lexicographic_indices_{offset}_{offset+1}_fixed_before_outputs",
        paper_results_changed=False, automatic_retry=False, new_assets_downloaded=False,
        endpoint_scope="full_controlled_P_and_B_answer_workflow_not_quality_gain_or_blanket_certification"))

    def one(job):
        save(output/"claims"/(job["id"]+".json"), dict(job=job,pid=os.getpid(),started=time.time()))
        model, calls, record = job["model"], [], None
        adapter = adapters[job["benchmark"]]
        example = adapter.example(job["split"], job["case_id"])
        state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        backend = RecordingBackend(calls=calls,model_id=model["id"],base_url=model["base_url"],
            extra_body=model.get("extra_body",{}),timeout_seconds=180,transport=local_transport,
            deployment_identity={"revision":digest(bindings[model["id"]]["model_json_hashes"]),"server":"vllm"})
        monitor = RunMonitor()
        row = dict(job=job,input_state_hash=state.state_hash,model_binding=bindings[model["id"]],
            native_prompt_runtime=source_identity,additional_attempts=1,automatic_retry=False,
            package_location=__import__("agentinputlab").__file__)
        started, stage = time.monotonic(), "full_pab_workflow"
        try:
            method = create_intervention(RELEASE,output_format=OUTPUT_FORMAT,
                decomposition_max_tokens=512,assessment_max_tokens=512,max_solver_tokens=512)
            record = CompositeInterventionRunner(monitor=monitor).run(method,state,
                InterventionContext(backend=backend,external_runtime=runtime,seed=17))
            stage = "source_backed_QA_scoring"
            prediction = QAPrediction(answer=record.result.final_answer.strip())
            row.update(status="scored_terminal",prediction=to_jsonable(prediction),
                       metrics=dict(adapter.score(example,prediction)))
        except Exception as exc:
            row.update(status="failed_terminal",stage=stage,error_type=type(exc).__name__,error=str(exc))
        row.update(calls=calls,model_calls=len(calls),composite=to_jsonable(record) if record else None,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),seconds=time.monotonic()-started)
        save(output/"terminals"/(job["id"]+".json"),row)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(one,jobs))
    rows = [json.loads(p.read_text()) for p in (output/"terminals").glob("*.json")]
    save(output/"complete.json",dict(expected=len(jobs),terminal=len(rows),
        statuses=dict(Counter(r["status"] for r in rows)),physical_model_calls=sum(r["model_calls"] for r in rows),
        whole_release_certified=False,publication_ready=False,automatic_retry=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.output)
