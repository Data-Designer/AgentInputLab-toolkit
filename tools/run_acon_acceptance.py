"""Real source-bound ACON component/common-support across four targets and four QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from acceptance_method_bindings import bind_acon
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASES = ("curate.acon.common_support_observation", "curate.acon.observation_runtime")


class RecordedWorker:
    worker_id = "acon-native-real-acceptance"

    def __init__(self, delegate, root):
        self.delegate, self.root = delegate, root
        self.responses = []

    def fingerprint(self):
        return self.delegate.fingerprint()

    def call(self, payload):
        number = len(self.responses) + 1
        save(self.root / f"worker-{number:02d}-request.json", payload)
        response = self.delegate.call(payload)
        self.responses.append(response)
        save(self.root / f"worker-{number:02d}-response.json", response)
        save(
            self.root / f"worker-{number:02d}-resource.json",
            to_jsonable(self.delegate.last_resource_receipt),
        )
        return response


def execute(config, root, upstream, entry_repair_root=None):
    repair_sources = {}
    if entry_repair_root is not None:
        prior = list(entry_repair_root.glob("*/terminal.json"))
        if len(prior) != len(config["models"]):
            raise ValueError("entry repair requires exactly one pre-call guard failure per model")
        for p in prior:
            d = json.loads(p.read_text())
            if not (
                d["status"] == "failed_terminal"
                and d["error_type"] == "ValidationError"
                and "hidden on intervention: intervention.client" in d["error"]
                and d["worker_responses"] == []
                and d["calls"] == []
                and d["intervention"] is None
            ):
                raise ValueError("entry repair must not repeat any consumed optimizer/model call")
            repair_sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[
                14:16
            ]:
                for release in RELEASES:
                    j = {
                        "model": model,
                        "benchmark": cell,
                        "split": split,
                        "case_id": example.example_id,
                        "release": release,
                        "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512,
                        "prompt_profile": "smolagents.observation.v1",
                    }
                    j["id"] = digest(j)
                    jobs.append(j)
    save(
        root / "manifest.json",
        {
            "jobs": jobs,
            "automatic_retry": False,
            "paper_results_changed": False,
            "native_tokenizer": "exact upstream gpt-4o tiktoken counting; not the target model",
            "references_exposed": False,
            "zero_call_entry_repair_sources": repair_sources,
        },
    )

    def one(job):
        out = root / job["id"]
        out.mkdir()
        save(out / "claim.json", {"job": job, "started": time.time()})
        model = job["model"]
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        calls, workers = [], []
        record, stage = None, "source_and_dependency_binding"
        monitor = RunMonitor()
        result = {
            "job": job,
            "package_location": __import__("agentinputlab").__file__,
            "model_binding": binding,
        }
        started = time.monotonic()
        try:
            backend = RecordingBackend(
                calls=calls,
                model_id=model["id"],
                base_url=model["base_url"],
                deployment_identity={
                    "revision": digest(binding["model_json_hashes"]),
                    "server": "vllm",
                },
                extra_body=model.get("extra_body", {}),
                timeout_seconds=300,
                transport=local_transport,
            )

            def wrap(client):
                worker = RecordedWorker(client, out)
                workers.append(worker)
                return worker

            method, worker = bind_acon(
                job["release"], upstream=upstream, backend=backend, model=model, client_wrapper=wrap
            )
            adapter = adapters[job["benchmark"]]
            ex = adapter.example(job["split"], job["case_id"])
            state = ex.visible_state(budget=ExecutionBudget(**job["budget"]))
            result["input_state_hash"] = state.state_hash
            stage = "native_preflight_then_optimizer"
            record = InterventionRunner(monitor=monitor).run(
                method,
                state,
                InterventionContext(backend=backend, external_runtime=worker, seed=17),
            )
            patched = record.patch.apply(state)
            patched = replace(
                patched,
                budget=replace(
                    state.budget,
                    max_model_calls=8 - record.usage.model_calls,
                    max_output_tokens=4096 - record.usage.output_tokens,
                ),
            )
            stage = "structured_target_and_native_score"
            predictor = ProductionPredictorBuilder().build(
                StaticPredictorBuildRequest(
                    benchmark_id=job["benchmark"],
                    split=job["split"],
                    model_slot=model["id"],
                    seed=17,
                    builder_parameters={
                        "schema_version": 1,
                        "task": "qa",
                        "qa": {
                            "output_format": "structured_json",
                            "max_tokens": 512,
                            "use_seed": True,
                        },
                    },
                    runtime=StaticRuntimeComponents(backend=backend),
                )
            )
            prediction = predictor.predict(patched)
            result.update(
                status="scored_terminal",
                metrics=dict(adapter.score(ex, prediction.prediction)),
                prediction=to_jsonable(prediction),
            )
        except Exception as exc:
            result.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        result.update(
            seconds=time.monotonic() - started,
            calls=calls,
            intervention=to_jsonable(record) if record else None,
            worker_responses=[r for w in workers for r in w.responses],
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),
        )
        save(out / "terminal.json", result)
        (root.parent / "keep-services-alive").touch()

    def lane(model):
        selected = [j for j in jobs if j["model"]["id"] == model["id"]]
        # Check the first actual native result before fanning out the same lane.
        one(selected[0])
        first = json.loads((root / selected[0]["id"] / "terminal.json").read_text())
        consumed = sum(r.get("usage", {}).get("model_calls", 0) for r in first["worker_responses"])
        if first["status"] == "failed_terminal" and consumed == 0:
            save(
                root / (model["id"] + "-attention.json"),
                {"entry_failure": selected[0]["id"], "other_cases_unstarted": True},
            )
            return
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(one, selected[1:]))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lane, config["models"]))
    rows = [json.loads(p.read_text()) for p in root.glob("*/terminal.json")]
    save(
        root / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--entry-repair-root", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True)
    execute(
        json.loads(args.config.read_text()), args.output, args.upstream_root, args.entry_repair_root
    )


if __name__ == "__main__":
    main()
