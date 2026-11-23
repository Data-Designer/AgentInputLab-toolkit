"""Separate fixed real QA JSON-schema configuration check, no scientific retries."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder


def parameters():
    return {"schema_version": 1, "task": "qa", "qa": {
        "output_format": "structured_json", "structured_output_mode": "json_schema",
        "max_tokens": 512, "use_seed": True}}


def run(config, output):
    bindings = verify_services(config["models"])
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"]) for cell, _ in QA_CELLS}
    examples = {cell: sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[40:42]
                for cell, split in QA_CELLS}
    if any(len(selected) != 2 for selected in examples.values()):
        raise ValueError("fixed QA schema case selection is incomplete")
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in examples[cell]:
                job = {"model": model, "release": "baseline.no_op", "benchmark": cell,
                       "split": split, "case_id": example.example_id, "seed": 17,
                       "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                       "builder_parameters": parameters()}
                job["id"] = digest(job)
                jobs.append(job)
    output.mkdir(parents=True, exist_ok=False)
    (output / "claims").mkdir()
    (output / "terminals").mkdir()
    save(output / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
         "selection": "fixed lexicographic source indices 40/41 per cell",
         "prepared_sha256": {cell: dict(adapter.prepared.file_sha256) for cell, adapter in adapters.items()},
         "scope": "new explicit json_schema configuration, not replacements for json_object failures",
         "automatic_retry": False, "paper_results_changed": False})

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), job)
        model, calls = job["model"], []
        binding = bindings[model["id"]]
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            deployment_identity={"revision": digest(binding["model_json_hashes"]), "server": "vllm"},
            timeout_seconds=180, extra_body=model.get("extra_body", {}), transport=local_transport)
        example = next(e for e in examples[job["benchmark"]] if e.example_id == job["case_id"])
        state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        result = {"job": job, "package_location": __import__("agentinputlab").__file__,
                  "input_state_hash": state.state_hash, "model_binding": binding,
                  "additional_attempts": 1, "automatic_retry": False}
        stage, started = "predictor_build", time.monotonic()
        try:
            predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                benchmark_id=job["benchmark"], split=job["split"], model_slot=model["id"], seed=17,
                builder_parameters=job["builder_parameters"], runtime=StaticRuntimeComponents(backend=backend)))
            stage = "downstream_generation"
            prediction = predictor.predict(state)
            stage = "native_scoring"
            metrics = dict(adapters[job["benchmark"]].score(example, prediction.prediction))
            result.update(status="scored_terminal", metrics=metrics, prediction=to_jsonable(prediction))
        except Exception as exc:
            result.update(status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc))
        result.update(seconds=time.monotonic() - started, calls=calls, model_calls=len(calls))
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    def per_model(model):
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(one, [job for job in jobs if job["model"]["id"] == model["id"]]))
    with ThreadPoolExecutor(len(config["models"])) as pool:
        list(pool.map(per_model, config["models"]))
    terms = [json.loads(path.read_text()) for path in (output / "terminals").glob("*.json")]
    save(output / "complete.json", {"expected": len(jobs), "terminal": len(terms),
         "terminal_statuses": dict(Counter(t["status"] for t in terms)),
         "full_toolkit_acceptance_complete": False, "paper_results_changed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
