"""Fixed real QA acceptance for the controlled P&B decomposition-only component.

Shared first-stage prompt/parser, no credit assessor or budget allocation. A
separate QA reader sees its original visible state plus the suggested plan.
Failures and original budgets remain in the manifest and actual cost ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts import BudgetExceededError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASE = "plan.plan_and_budget.decomposer_component"


def remaining_budget(budget, calls):
    used = 0
    for call in calls:
        if "response" not in call:
            raise BudgetExceededError("unknown consumed usage; no downstream dispatch")
        value = call["response"]["usage"]["output_tokens"]
        if type(value) is not int or value < 0:
            raise ValidationError("invalid known generation usage")
        used += value
    if len(calls) >= budget.max_model_calls or used >= budget.max_output_tokens:
        raise BudgetExceededError("shared component/reader budget exhausted")
    return replace(budget, max_model_calls=budget.max_model_calls - len(calls),
                   max_output_tokens=budget.max_output_tokens - used)


def run(config, output):
    bindings = verify_services(config["models"])
    offset = config.get("component_case_offset", 88)
    if type(offset) is not int or offset < 0:
        raise ValueError("component source case offset must be non-negative")
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"])
                for cell, _ in QA_CELLS}
    cases = {cell: tuple(sorted(adapters[cell].examples(split),
                   key=lambda e: e.example_id)[offset:offset + 2]) for cell, split in QA_CELLS}
    if len(config["models"]) != 4 or any(len(values) != 2 for values in cases.values()):
        raise ValueError("four original models and two fixed cases per QA required")
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in cases[cell]:
                for release in ("baseline.no_op", RELEASE):
                    job = {"model": model, "benchmark": cell, "split": split,
                        "case_id": example.example_id, "release": release, "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512,
                        "constructor_parameters": {"decomposition_max_tokens": 512} if release == RELEASE else {},
                        "optimizer_input_projection": "query_only" if release == RELEASE else "none",
                        "credit_assessment": False, "budget_allocation": False,
                        "separate_reader_protocol": "original_visible_state_plus_suggested_subquestions",
                        "support_evidence_labels_inferred_from_plan": False}
                    job["id"] = digest(job)
                    jobs.append(job)
    output.mkdir(exist_ok=False)
    for directory in ("claims", "terminals"):
        (output / directory).mkdir()
    save(output / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
        "selection": f"fixed_source_lexicographic_indices{offset}:{offset + 2}_before_outputs",
        "source_hashes": {cell: dict(a.prepared.file_sha256) for cell, a in adapters.items()},
        "source_revisions": {cell: a.prepared.source_revision for cell, a in adapters.items()},
        "eligibility": "controlled_generalized_decomposition_only_component_not_full_P_and_B",
        "shared_prompt_parser": "native_regenerated_soft_budget_first_stage",
        "automatic_retry": False, "new_fit": 0, "asset_acquisition_bytes": 0,
        "paper_results_changed": False})

    def one(job):
        save(output / "claims" / (job["id"] + ".json"),
             {"job": job, "pid": os.getpid(), "started": time.time()})
        model, adapter = job["model"], adapters[job["benchmark"]]
        example = adapter.example(job["split"], job["case_id"])
        original = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        calls, record = [], None
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            extra_body=model.get("extra_body", {}), timeout_seconds=180, transport=local_transport,
            deployment_identity={"revision": digest(bindings[model["id"]]["model_json_hashes"]), "server": "vllm"})
        monitor, started = RunMonitor(), time.monotonic()
        row = {"job": job, "input_state_hash": original.state_hash,
            "model_binding": bindings[model["id"]], "automatic_retry": False,
            "package_location": __import__("agentinputlab").__file__, "additional_attempts": 1}
        stage = "controlled_decomposer_component"
        try:
            state = original
            if job["release"] == RELEASE:
                record = InterventionRunner(monitor=monitor).run(
                    create_intervention(RELEASE, **job["constructor_parameters"]), original,
                    InterventionContext(backend=backend, seed=17))
                state = record.patch.apply(original)
                if (state.query != original.query or state.context != original.context
                        or state.messages != original.messages):
                    raise ValidationError("component changed reader query/context/history")
            stage = "separate_downstream_reader"
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
        row.update(calls=calls, model_calls=len(calls), intervention=to_jsonable(record) if record else None,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()), seconds=time.monotonic() - started,
            physical_tool_or_retrieval_calls=0)
        save(output / "terminals" / (job["id"] + ".json"), row)
        (output.parent / "keep-services-alive").touch()

    def lane(model):
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(one, (j for j in jobs if j["model"]["id"] == model["id"])))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lane, config["models"]))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    save(output / "complete.json", {"expected": len(jobs), "terminal": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "full_toolkit_acceptance_complete": False, "paper_results_changed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
