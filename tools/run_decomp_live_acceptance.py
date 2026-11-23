"""Fixed real native-DecomP compatibility probes, isolated from paper results.

This explicitly applies the source CommaQA config to caller-supplied QA capture
cases. It is controlled cross-domain compatibility, not original GPT-3/CommaQA
task performance or a substitute for the separate common-support ODQA release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.decompose.decomp import DecompFullBridge
from agentinputlab.backends import DecompCompletionBackend
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import DecompLiveRuntime
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import CompositeInterventionRunner

RELEASE = "plan.decomp.full"


class NativeWire:
    """Retain sanitized wire evidence and unknown consumption; never retry."""
    def __init__(self):
        self.calls = []

    def transport(self, url, headers, payload, timeout):
        call = {"url":url,"payload":payload,"started":time.time()}
        self.calls.append(call)
        try:
            result = local_transport(url, headers, payload, timeout)
        except Exception as exc:
            call.update(error_type=type(exc).__name__, response_received=False, usage_unknown=True)
            status = getattr(exc, "code", None)
            if type(status) is int and 400 <= status <= 599:
                call["http_status"] = status
            raise
        call.update(response=result, response_received=True)
        return result


def run(config, output, worker_python, checkout, nltk_data, wheel):
    bindings = verify_services(config["models"])
    adapters = {cell:load_production_benchmark_adapter(cell, config["prepared_root"]) for cell, _ in QA_CELLS}
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            cases = sorted(adapters[cell].examples(split), key=lambda e:e.example_id)[74:75]
            if len(cases) != 1:
                raise ValueError("fixed source index74 must exist")
            for release in ("baseline.no_op", RELEASE):
                job = {"model":model,"benchmark":cell,"split":split,"case_id":cases[0].example_id,
                    "release":release,"seed":17,"budget":{"max_model_calls":8,"max_output_tokens":4096},
                    "native_config_artifact_id":"decomp.commaqa.coarse.p1", "max_steps":20,
                    "per_call_ceiling":512,"native_output_caps_preserved":True,
                    "native_input_projection":"query_and_exact_visible_context_empty_instruction_history",
                    "scoring_projection":"answer_only_no_inferred_support_or_evidence_labels"}
                job["id"] = digest(job)
                jobs.append(job)
    if len(jobs) != 32:
        raise ValueError("four backbones byfour QAcaptures bytwo arms required")
    output.mkdir(exist_ok=False)
    for name in ("claims","terminals"):
        (output/name).mkdir()
    process = Path("/proc")/str(os.getpid())
    stat = (process/"stat").read_text().rsplit(")",1)[1].split()
    save(output/"execution-start.json", {"pid":os.getpid(),"uid":os.getuid(),"parent":int(stat[1]),
        "start":stat[19],"argv_sha256":hashlib.sha256((process/"cmdline").read_bytes()).hexdigest(),
        "operator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"started":time.time()})
    save(output/"manifest.json", {"jobs":jobs,"model_bindings":bindings,
        "source_hashes":{c:dict(a.prepared.file_sha256) for c,a in adapters.items()},
        "selection":"fixed_source_sorted_index74_before_outcomes", "automatic_retry":False,
        "eligibility":"controlled_cross_domain_native_CommaQA_participants_on_visible_QA_captures",
        "not_original_gpt3_checkpoint_or_native_CommaQA_task_parity":True,
        "source_release_not_common_support_odqa":True,"paper_results_changed":False,
        "new_benchmark_or_model_downloads":0,"installed_wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest()})

    def one(job):
        save(output/"claims"/(job["id"]+".json"),{"job":job,"pid":os.getpid(),"started":time.time()})
        model = job["model"]
        adapter = adapters[job["benchmark"]]
        example = adapter.example(job["split"],job["case_id"])
        original = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        state = replace(original,instruction="",messages=())
        wire, baseline_calls, runtime, record = NativeWire(), [], None, None
        monitor, started, stage = RunMonitor(), time.monotonic(), "generation"
        row = {"job":job,"input_state_hash":original.state_hash,"native_projected_state_hash":state.state_hash,
            "model_binding":bindings[model["id"]],"package_location":__import__("agentinputlab").__file__,
            "automatic_retry":False,"additional_attempts":1}
        identity = {"revision":digest(bindings[model["id"]]["model_json_hashes"]),"server":"vllm"}
        try:
            if job["release"] == RELEASE:
                backend = DecompCompletionBackend(model_id=model["id"],base_url=model["base_url"],
                    api_key_env="AGENTINPUTLAB_TOOLKIT_LOCAL_NO_AUTH",deployment_identity=identity,
                    transport=wire.transport,timeout_seconds=180)
                runtime = DecompLiveRuntime(python_executable=str(worker_python),checkout=checkout,
                    nltk_data=nltk_data,backend=backend,per_call_output_cap=512,timeout_seconds=300)
                record = CompositeInterventionRunner(monitor=monitor).run(
                    DecompFullBridge(None,config_artifact_id=job["native_config_artifact_id"],max_steps=20),
                    state,InterventionContext(backend=backend,external_runtime=runtime,seed=17))
                prediction = QAPrediction(answer=record.result.final_answer,support_ids=(),evidence_ids=(),evidence_texts=())
            else:
                backend = RecordingBackend(calls=baseline_calls,model_id=model["id"],base_url=model["base_url"],
                    deployment_identity=identity,timeout_seconds=180,extra_body=model.get("extra_body",{}),transport=local_transport)
                predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                    benchmark_id=job["benchmark"],split=job["split"],model_slot=model["id"],seed=17,
                    builder_parameters={"schema_version":1,"task":"qa","qa":{
                        "output_format":"answer_only","max_tokens":512,"use_seed":True}},
                    runtime=StaticRuntimeComponents(backend=backend)))
                prediction = predictor.predict(original).prediction
            stage = "native_scoring"
            row.update(status="scored_terminal",prediction=to_jsonable(prediction),metrics=dict(adapter.score(example,prediction)))
        except Exception as exc:
            row.update(status="failed_terminal",stage=stage,error_type=type(exc).__name__,error=str(exc))
        calls = wire.calls if job["release"] == RELEASE else baseline_calls
        row.update(calls=calls,model_calls=len(calls),intervention=to_jsonable(record) if record else None,
            callback_events=runtime.callback_events if runtime else [],
            worker_response=runtime.last_response if runtime else None,
            worker_identity=runtime.fingerprint() if runtime else None,
            resource_receipt=to_jsonable(runtime.last_resource_receipt) if runtime else None,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),seconds=time.monotonic()-started)
        save(output/"terminals"/(job["id"]+".json"),row)
        (output.parent/"keep-services-alive").touch()

    limits = {m["id"]:threading.BoundedSemaphore(2) for m in config["models"]}
    def bounded(job):
        with limits[job["model"]["id"]]:
            one(job)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(bounded,sorted(jobs,key=lambda j:(j["benchmark"],j["release"],j["model"]["id"]))))
    rows = [json.loads(p.read_text()) for p in (output/"terminals").glob("*.json")]
    save(output/"complete.json", {"expected":32,"terminal":len(rows),
        "statuses":dict(Counter(r["status"] for r in rows)),
        "model_generation_attempts":sum(r["model_calls"] for r in rows),"full_toolkit_acceptance_complete":False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","output","worker-python","checkout","nltk-data","wheel"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.output,args.worker_python,args.checkout,args.nltk_data,args.wheel)


if __name__=="__main__":
    main()
