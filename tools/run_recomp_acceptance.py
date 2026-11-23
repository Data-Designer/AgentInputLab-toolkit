"""Real pinned RECOMP extraction and independent four-backbone QA acceptance.

The helper runs in its isolated legacy interpreter. Its native sentence scores
are shared only for the exact same visible capture, never across different
inputs. This controlled top-five profile is not paper-result parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker
from agentinputlab.runner import InterventionRunner

RELEASE = "curate.recomp.common_support_extractive"


class RecordedWorker:
    worker_id = "recomp-native-real-acceptance"

    def __init__(self, delegate, root):
        self.delegate, self.root = delegate, root
        self.responses = []
        self.attempts = 0

    def fingerprint(self):
        return self.delegate.fingerprint()

    def call(self, payload):
        self.attempts += 1
        number = self.attempts
        save(self.root / f"worker-{number:02d}-request.json", payload)
        try:
            response = self.delegate.call(payload)
        except Exception as exc:
            save(
                self.root / f"worker-{number:02d}-transport-error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "consumption": "unknown_if_process_failed_after_start",
                },
            )
            raise
        else:
            self.responses.append(response)
            save(self.root / f"worker-{number:02d}-response.json", response)
            return response
        finally:
            save(
                self.root / f"worker-{number:02d}-resource.json",
                to_jsonable(self.delegate.last_resource_receipt),
            )


@dataclass(frozen=True)
class CapturedContextSource:
    benchmark: str
    case_id: str
    context_hash: str

    def fingerprint(self):
        return {
            "provider": "benchmark_visible_context_capture",
            "benchmark": self.benchmark,
            "case_id": self.case_id,
            "context_hash": self.context_hash,
            "live_retrieval_performed": False,
        }


def execute(config, root, upstream, worker_python, worker_directory):
    driver = worker_directory / "driver.py"
    compatibility = worker_directory / "compatibility_manifest.json"
    checkpoint = upstream / "upstream-paper/recomp-assets/nq_extractive_compressor_frozen"
    checkpoint_manifest = (
        upstream / "upstream-paper/recomp-assets/nq_extractive_compressor_manifest.json"
    )
    asset = json.loads(checkpoint_manifest.read_text())
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    }
    common = (
        str(worker_python),
        str(driver),
        "--checkout",
        str(upstream / "upstream-paper/recomp"),
    )
    smoke = subprocess.run(
        (*common, "--smoke-import", "extractive", "--device", "cpu"),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    runtime = json.loads(smoke.stdout)
    save(root / "native-import.json", {"response": runtime, "stderr": smoke.stderr})
    if runtime.get("status") != "ok" or runtime["execution_evidence"] != "pinned_upstream_source":
        raise RuntimeError("native RECOMP import failed; no inference dispatched")
    identity = {
        "compatibility_manifest_sha256": hashlib.sha256(compatibility.read_bytes()).hexdigest(),
        "execution_evidence": runtime["execution_evidence"],
        "runtime_fingerprint_hash": runtime["runtime_fingerprint_hash"],
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "checkpoint_manifest_sha256": hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest(),
    }
    if runtime["compatibility_manifest_sha256"] != identity["compatibility_manifest_sha256"]:
        raise ValueError("RECOMP import used another compatibility policy")
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    examples = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda e: e.example_id)[16:18])
        for c, s in QA_CELLS
    }
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in examples[cell]:
                job = {
                    "model": model,
                    "benchmark": cell,
                    "split": split,
                    "case_id": example.example_id,
                    "release": RELEASE,
                    "assembly": "controlled_top5_source_order",
                    "seed": 17,
                    "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                    "per_call_output_cap": 512,
                    "helper_identity": identity,
                }
                job["id"] = digest(job)
                jobs.append(job)
    for name in ("claims", "terminals", "preparation"):
        (root / name).mkdir()
    save(
        root / "manifest.json",
        {
            "jobs": jobs,
            "native_runtime": runtime,
            "reference_exposed": False,
            "automatic_retry": False,
            "paper_results_changed": False,
            "controlled_profile_not_paper_result_parity": True,
            "physical_helper_cost_counted_once_per_exact_visible_capture": True,
            "source_context_is_captured_not_live_retrieved": True,
        },
    )

    def target(job, state, record, receipt):
        save(root / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        calls = []
        started = time.monotonic()
        model = job["model"]
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        result = {
            "job": job,
            "model_binding": binding,
            "shared_preparation_receipt": str(receipt),
            "input_state_hash": state.state_hash,
            "package_location": __import__("agentinputlab").__file__,
            "additional_target_attempts": 1,
        }
        try:
            patched = record.patch.apply(state)
            patched = replace(
                patched,
                budget=replace(
                    state.budget,
                    max_model_calls=state.budget.max_model_calls - record.usage.model_calls,
                    max_output_tokens=state.budget.max_output_tokens - record.usage.output_tokens,
                ),
            )
            backend = RecordingBackend(
                calls=calls,
                model_id=model["id"],
                base_url=model["base_url"],
                timeout_seconds=300,
                extra_body=model.get("extra_body", {}),
                deployment_identity={
                    "revision": digest(binding["model_json_hashes"]),
                    "server": "vllm",
                },
                transport=local_transport,
            )
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
            adapter = adapters[job["benchmark"]]
            example = adapter.example(job["split"], job["case_id"])
            result.update(
                status="scored_terminal",
                metrics=dict(adapter.score(example, prediction.prediction)),
                prediction=to_jsonable(prediction),
            )
        except Exception as exc:
            result.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        result.update(seconds=time.monotonic() - started, calls=calls, automatic_retry=False)
        save(root / "terminals" / (job["id"] + ".json"), result)
        (root.parent / "keep-services-alive").touch()

    pools = {m["id"]: ThreadPoolExecutor(2) for m in config["models"]}
    futures = []
    prepared = 0
    try:
        for cell, values in examples.items():
            for example in values:
                out = (
                    root
                    / "preparation"
                    / digest({"benchmark": cell, "case_id": example.example_id})
                )
                out.mkdir()
                state = example.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                worker = RecordedWorker(
                    JsonLineSubprocessWorker(
                        command=(
                            *common,
                            "--checkpoint",
                            str(checkpoint),
                            "--checkpoint-manifest",
                            str(checkpoint_manifest),
                            "--device",
                            "cpu",
                            "--acknowledge-noassertion-checkpoint",
                        ),
                        worker_id="toolkit-native-recomp-extractive",
                        identity=identity,
                        timeout_seconds=600,
                        environment=environment,
                    ),
                    out,
                )
                method = create_intervention(
                    RELEASE,
                    checkpoint_revision=asset["checkpoint_revision"],
                    tokenizer_revision=asset["tokenizer_revision"],
                    asset_manifest_hash=identity["checkpoint_manifest_sha256"],
                    asset_license_acknowledged=True,
                )
                source = CapturedContextSource(
                    cell, example.example_id, content_hash(state.context)
                )
                started = time.monotonic()
                try:
                    record = InterventionRunner().run(
                        method,
                        state,
                        InterventionContext(
                            seed=17,
                            retriever=source,
                            external_runtime=worker,
                        ),
                    )
                except Exception as exc:
                    save(
                        out / "terminal.json",
                        {
                            "status": "failed_preparation",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "worker_responses": worker.responses,
                            "seconds": time.monotonic() - started,
                            "automatic_retry": False,
                        },
                    )
                    save(
                        root / "attention.json",
                        {
                            "failed_preparation": str(out),
                            "completed_preparations": prepared,
                            "remaining_cases_unstarted": True,
                        },
                    )
                    return
                receipt = out / "terminal.json"
                save(
                    receipt,
                    {
                        "status": "native_compressed",
                        "record": to_jsonable(record),
                        "seconds": time.monotonic() - started,
                        "worker_responses": worker.responses,
                        "physical_model_forwards": 1,
                        "reference_exposed": False,
                    },
                )
                prepared += 1
                for job in jobs:
                    if (job["benchmark"], job["case_id"]) == (cell, example.example_id):
                        futures.append(
                            pools[job["model"]["id"]].submit(target, job, state, record, receipt)
                        )
                (root.parent / "keep-services-alive").touch()
        for future in futures:
            future.result()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    rows = [json.loads(p.read_text()) for p in (root / "terminals").glob("*.json")]
    save(
        root / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "native_preparation_count": prepared,
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "upstream-root", "worker-python", "worker-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    execute(
        json.loads(args.config.read_text()),
        args.output,
        args.upstream_root,
        args.worker_python,
        args.worker_directory,
    )


if __name__ == "__main__":
    main()
