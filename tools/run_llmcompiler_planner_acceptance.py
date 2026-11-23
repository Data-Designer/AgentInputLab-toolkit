"""Fixed real QA acceptance for the source-native LLMCompiler planner component.

The native planner sees the query and read-only tool schemas, not arbitrary
instruction/history. Its DAG is added to the original visible QA state for a
separate reader. No planned task or join is executed. All reader and planner
requests share the declared budget and every genuine failure is retained.
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
from run_llmcompiler_live_acceptance import VisibleSearch
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.decompose.llm_compiler import (
    LLMCompilerControls,
    LLMCompilerPlannerBridge,
)
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts import BudgetExceededError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import LLMCompilerLiveWorker
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASE = "plan.llm_compiler.planner_component"


def remaining_budget(budget, calls):
    """Do not fund a reader from unknown, uncharged or already exhausted usage."""
    used = 0
    for call in calls:
        if "response" not in call:
            raise BudgetExceededError("unknown consumed usage; no downstream dispatch")
        value = call["response"]["usage"]["output_tokens"]
        if type(value) is not int or value < 0:
            raise ValidationError("invalid saved generation usage")
        used += value
    if len(calls) >= budget.max_model_calls or used >= budget.max_output_tokens:
        raise BudgetExceededError("shared planner/reader budget exhausted")
    return replace(budget, max_model_calls=budget.max_model_calls - len(calls),
                   max_output_tokens=budget.max_output_tokens - used)


def materialize_for_reader(original, projected, patch):
    """Apply only to its bound state, then restore the declared reader protocol."""
    if (projected != replace(original, instruction="", messages=())
            or patch.source_state_hash != projected.state_hash):
        raise ValidationError("planner query-only projection differs from the frozen case")
    result = patch.apply(projected)
    if result.query != original.query or result.task_id != original.task_id:
        raise ValidationError("planner component changed the question or task identity")
    return replace(result, instruction=original.instruction, messages=original.messages)


def run(config, output, worker_python, checkout, prompt_bundle):
    bindings = verify_services(config["models"])
    prompts = json.loads(prompt_bundle.read_text())["prompts"]
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"])
                for cell, _ in QA_CELLS}
    offset = config.get("planner_case_offset", 84)
    if type(offset) is not int or offset < 0:
        raise ValueError("planner source case offset must be non-negative")
    cases = {cell: tuple(sorted(adapters[cell].examples(split),
                  key=lambda e: e.example_id)[offset:offset + 2]) for cell, split in QA_CELLS}
    if len(config["models"]) != 4 or any(len(v) != 2 for v in cases.values()):
        raise ValueError("four model bindings and two fixed source cases per QA required")
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in cases[cell]:
                for release in ("baseline.no_op", RELEASE):
                    job = {"model": model, "benchmark": cell, "split": split,
                        "case_id": example.example_id, "release": release, "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512, "compiler_streaming": False,
                        "max_tasks": 8, "max_replans": 0, "concurrency": 4,
                        "prompt_bundle_hash": content_hash(prompts),
                        "native_input_projection": "query_only_empty_instruction_history" if release == RELEASE else "none",
                        "downstream_protocol": "original_visible_state_plus_task_DAG_answer_only",
                        "support_evidence_labels_inferred_from_plan": False,
                        "search_scope": "schema_only_no_planned_tasks_executed"}
                    job["id"] = digest(job)
                    jobs.append(job)
    output.mkdir(exist_ok=False)
    for name in ("claims", "terminals"):
        (output / name).mkdir()
    save(output / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
        "selection": f"fixed_source_lexicographic_indices{offset}:{offset + 2}_before_outputs",
        "source_hashes": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
        "prompt_bundle_sha256": hashlib.sha256(prompt_bundle.read_bytes()).hexdigest(),
        "eligibility": "query_only_native_planner_component_then_separate_QA_reader_not_full_agent",
        "automatic_retry": False, "new_fit": 0, "asset_acquisition_bytes": 0,
        "paper_results_changed": False})

    def one(job):
        save(output / "claims" / (job["id"] + ".json"),
             {"job": job, "pid": os.getpid(), "started": time.time()})
        model, adapter = job["model"], adapters[job["benchmark"]]
        example = adapter.example(job["split"], job["case_id"])
        original = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        calls, record, worker, tools = [], None, None, None
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            extra_body=model.get("extra_body", {}), timeout_seconds=180, transport=local_transport,
            deployment_identity={"revision": digest(bindings[model["id"]]["model_json_hashes"]), "server": "vllm"})
        monitor, started = RunMonitor(), time.monotonic()
        row = {"job": job, "input_state_hash": original.state_hash,
            "model_binding": bindings[model["id"]], "automatic_retry": False,
            "package_location": __import__("agentinputlab").__file__, "additional_attempts": 1}
        stage = "native_planner_component"
        try:
            state = original
            if job["release"] == RELEASE:
                projected = replace(original, instruction="", messages=())
                row["native_projected_state_hash"] = projected.state_hash
                tools = VisibleSearch(projected, adapter.prepared.source_revision)
                worker = LLMCompilerLiveWorker(python_executable=str(worker_python), checkout=checkout,
                    prompt_bundle=prompt_bundle, backend=backend, tools=tools, timeout_seconds=400)
                record = InterventionRunner(monitor=monitor).run(
                    LLMCompilerPlannerBridge(prompt_hash=content_hash(prompts),
                        controls=LLMCompilerControls(max_tasks=8, max_replans=0, concurrency=4,
                                                    timeout_seconds=300, streaming=False)),
                    projected, InterventionContext(backend=backend, tools=tools,
                                                   external_runtime=worker, seed=17))
                if tools.calls:
                    raise ValidationError("planner component must not execute a planned tool")
                state = materialize_for_reader(original, projected, record.patch)
            stage = "downstream_reader"
            state = replace(state, budget=remaining_budget(original.budget, calls))
            predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                benchmark_id=job["benchmark"], split=job["split"], model_slot=model["id"], seed=17,
                builder_parameters={"schema_version": 1, "task": "qa", "qa": {
                    "output_format": "answer_only", "max_tokens": 512, "use_seed": True}},
                runtime=StaticRuntimeComponents(backend=backend)))
            prediction = predictor.predict(state)
            stage = "native_scoring"
            row.update(status="scored_terminal", prediction=to_jsonable(prediction),
                       metrics=dict(adapter.score(example, prediction.prediction)))
        except Exception as exc:
            row.update(status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc))
        row.update(calls=calls, model_calls=len(calls),
            callback_events=worker.callback_events if worker else [],
            worker_fingerprint=worker.fingerprint() if worker else None,
            physical_retrieval_calls=len(tools.calls) if tools else 0,
            search_calls=tools.calls if tools else [],
            resource_receipt=to_jsonable(worker.last_resource_receipt) if worker else None,
            intervention=to_jsonable(record) if record else None,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()), seconds=time.monotonic() - started)
        save(output / "terminals" / (job["id"] + ".json"), row)
        (output.parent / "keep-services-alive").touch()

    limits = {m["id"]: threading.BoundedSemaphore(2) for m in config["models"]}

    def bounded(job):
        with limits[job["model"]["id"]]:
            one(job)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(bounded, sorted(jobs, key=lambda j: (j["benchmark"], j["case_id"], j["release"], j["model"]["id"]))))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    save(output / "complete.json", {"expected": 64, "terminal": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "observed_backend_requests": sum(r["model_calls"] for r in rows),
        "physical_retrieval_calls": sum(r["physical_retrieval_calls"] for r in rows),
        "full_toolkit_acceptance_complete": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "output", "worker-python", "checkout", "prompt-bundle"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.worker_python, args.checkout, args.prompt_bundle)


if __name__ == "__main__":
    main()
