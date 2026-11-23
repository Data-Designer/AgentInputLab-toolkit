"""Fixed ClariQ compatibility endpoints for the paper-step UDCS router.

No LLM judge, fabricated user reply, ordinal calibration or implicit retry.
The shared BM25 ranking is a carrier for native RQ2 scoring, not a claimed
UDCS question-ranking improvement. Source queries and hidden labels are intact.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.clarify import UDCS_DETERMINISTIC_RELEASE
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks.clariq import ClariQDevBundle, ClariQPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

PROJECTION = "udcs.clariq.initial_request_need.v1"
ROUTING_INSTRUCTION = "Assess whether the initial user request is underspecified before taking an action."


def project(example, budget):
    """Separate benchmark predictor JSON instructions from the policy's XML IO."""
    source = example.visible_state(budget=budget)
    state = replace(source, instruction=ROUTING_INSTRUCTION, tool_names=("continue_task",))
    assert state.query == source.query and state.context == source.context and state.metadata == source.metadata
    return source, state


def run(config, output):
    bundle = ClariQDevBundle(config["clariq_source_root"], accept_license_review=True)
    models = config["models"]
    if len(models) != 1:
        raise ValueError("one exact source-bound model per worker")
    model = models[0]
    binding = verify_services(models)[model["id"]]
    examples = bundle.examples()[:2]
    assert len(examples) == 2
    jobs = []
    for ex in examples:
        source, state = project(ex, ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
        job = dict(model=model, benchmark="clariq.main", split="dev", case_id=ex.example_id,
            release=UDCS_DETERMINISTIC_RELEASE, seed=17, per_call_output_tokens=512,
            budget={"max_model_calls":8,"max_output_tokens":4096},
            source_visible_state_hash=source.state_hash, method_visible_state_hash=state.state_hash,
            input_projection=PROJECTION, threshold=0.5, clarification_need_mapping="1=no_request,2=any_request")
        job["id"] = digest(job)
        jobs.append(job)
    output.mkdir(exist_ok=False)
    for name in ("claims", "terminals", "model-outcomes"):
        (output/name).mkdir()
    save(output/"manifest.json", dict(jobs=jobs, source=bundle.fingerprint(), model_binding=binding,
        selection="lexicographic first two source dev topics fixed before outputs",
        input_projection=PROJECTION, projection_removes_only_predictor_output_format_instruction=True,
        replacement_instruction=ROUTING_INSTRUCTION, no_gold_exposed=True,
        clarification_need_mapping="1=no request;2=any request;higher ordinal grades not inferred",
        question_ranker="unchanged shared initial-query BM25 over all3941source questions,top30",
        RQ2_ranking_effect_claimed=False, no_user_or_environment_reply_generated=True,
        original_paper_benchmark_parity_claimed=False, automatic_retry=False,paper_results_changed=False))
    mutex = threading.Lock()
    predictions = {}

    def one(job):
        save(output/"claims"/(job["id"]+".json"), dict(job=job,pid=os.getpid(),started=time.time()))
        ex = next(e for e in examples if e.example_id==job["case_id"])
        source,state = project(ex, ExecutionBudget(**job["budget"]))
        assert source.state_hash==job["source_visible_state_hash"] and state.state_hash==job["method_visible_state_hash"]
        calls = []
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            deployment_identity={"revision":digest(binding["model_json_hashes"]),"server":"vllm"},
            extra_body=model.get("extra_body",{}),timeout_seconds=180,transport=local_transport)
        monitor=RunMonitor()
        result=dict(job=job,model_binding=binding,package_location=__import__('agentinputlab').__file__,
            automatic_retry=False,additional_attempts=1)
        start=time.monotonic()
        prediction=record=None
        try:
            method=create_intervention(UDCS_DETERMINISTIC_RELEASE, max_tokens=512)
            record=InterventionRunner(monitor=monitor).run(method,state,InterventionContext(backend=backend,seed=17))
            need=2 if record.patch.metadata['requests_clarification'] else 1
            prediction=ClariQPrediction(need,bundle.rank_questions(source.query,top_k=30))
            result.update(status='scored_terminal',prediction=to_jsonable(prediction),record=to_jsonable(record))
        except Exception as exc:
            result.update(status='failed_terminal',error_type=type(exc).__name__,error=str(exc))
        save(output/'model-outcomes'/(job['id']+'.json'),{**result,'calls':calls,
            'record':to_jsonable(record) if record is not None else None,
            'monitor_snapshots':to_jsonable(monitor.list_snapshots()),
            'native_scoring_started':False})
        with mutex:
            # Original scorer's missing-prediction semantics retain failed cases;
            # no surrogate score/label is inserted into a failed model result.
            native=bundle.evaluate({ex.example_id:prediction} if prediction else {},
                topic_ids=[ex.example_id],output=output/'scores'/job['id'])
            if prediction is not None:
                predictions[ex.example_id]=prediction
        result.update(metrics=native['metrics'],native_score_receipt=str(output/'scores'/job['id']/'native-evaluation.json'),
            native_missing_prediction=prediction is None,seconds=time.monotonic()-start,
            calls=calls,provider_request_attempts=len(calls),monitor_snapshots=to_jsonable(monitor.list_snapshots()),
            native_scorer_sha256=native['native_scorer_sha256'])
        assert len(calls)<=1
        save(output/'terminals'/(job['id']+'.json'),result)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one,jobs))
    rows=[json.loads(p.read_text()) for p in (output/'terminals').glob('*.json')]
    assert len(rows)==2
    aggregate=bundle.evaluate(predictions,topic_ids=[e.example_id for e in examples],output=output/'scores'/'all-selected-topics')
    save(output/'complete.json',dict(expected_positions=2,terminal_positions=2,
        scored=sum(r['status']=='scored_terminal' for r in rows),failed=sum(r['status']=='failed_terminal' for r in rows),
        provider_request_attempts=sum(r['provider_request_attempts'] for r in rows),
        source_topic_denominator=2,missing_predictions=aggregate['missing_predictions'],native_metrics=aggregate['metrics'],
        native_aggregate_receipt=str(output/'scores'/'all-selected-topics'/'native-evaluation.json'),
        full_dev_evaluated=False,automatic_retry=False,publication_ready=False))
    print(model['id'],'two UDCS/ClariQ endpoints closed including failed denominators',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mode',choices=('preflight','run'),required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.mode=='preflight':
        bundle=ClariQDevBundle(config['clariq_source_root'],accept_license_review=True)
        for ex in bundle.examples()[:2]:
            project(ex,ExecutionBudget(max_model_calls=8,max_output_tokens=4096))
        print('two source-projected ClariQ states; zero model calls')
    elif args.output is None:
        parser.error('--output is required for execution')
    else:
        run(config,args.output)


if __name__=='__main__':
    main()
