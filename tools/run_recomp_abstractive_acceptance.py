"""Fixed real QA pairs with the released RECOMP seq2seq compressor, no training.

The exact visible capture is compressed once and shared across four reader arms.
Every reader's logical budget includes that compression; physical helper cost is
counted once. Failed preparations retain dependent terminals and do not stop the
other fixed cases. Neither real model failures nor empty summaries are retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_recomp_acceptance import CapturedContextSource, RecordedWorker
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.curate.recomp import RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker
from agentinputlab.runner import InterventionRunner

RELEASE = "curate.recomp.abstractive"


def select_cases(adapter, split):
    selected = sorted(adapter.examples(split), key=lambda item: item.example_id)[64:66]
    if len(selected) != 2:
        raise ValueError("two frozen source cases at sorted indices64:66 are required")
    return selected


def run(
    config,
    output,
    upstream,
    worker_python,
    worker_directory,
    checkpoint_root,
    checkpoint_manifest,
    conversion_proof,
    worker_addons,
):
    bindings = verify_services(config["models"])
    asset = json.loads(checkpoint_manifest.read_text())
    proof = json.loads(conversion_proof.read_text())
    manifest_sha = hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest()
    if (
        asset["mode"] != "abstractive"
        or proof["checkpoint_manifest_sha256"] != manifest_sha
        or proof["all_tensor_keys_shapes_dtypes_values_equal"] is not True
        or proof["quantization_or_parameter_change"] is not False
    ):
        raise ValueError("verified value-preserving source conversion proof is required")
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(worker_addons),
    }
    command = (
        str(worker_python),
        str(worker_directory / "driver.py"),
        "--checkout",
        str(upstream / "upstream-paper/recomp"),
        "--device",
        "cpu",
    )
    smoke = subprocess.run(
        (*command, "--smoke-import", "abstractive"),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    native = json.loads(smoke.stdout)
    if native.get("status") != "ok" or native.get("execution_evidence") != "pinned_upstream_source":
        raise ValueError("pinned native RECOMP import failed before inference")
    identity = {
        "driver_sha256": hashlib.sha256((worker_directory / "driver.py").read_bytes()).hexdigest(),
        "compatibility_manifest_sha256": hashlib.sha256(
            (worker_directory / "compatibility_manifest.json").read_bytes()
        ).hexdigest(),
        "runtime_fingerprint_hash": native["runtime_fingerprint_hash"],
        "checkpoint_manifest_sha256": manifest_sha,
        "conversion_proof_sha256": hashlib.sha256(conversion_proof.read_bytes()).hexdigest(),
        "source_weight_sha256": proof["source_weight_sha256"],
        "execution_evidence": native["execution_evidence"],
    }
    if identity["compatibility_manifest_sha256"] != native["compatibility_manifest_sha256"]:
        raise ValueError("worker compatibility policy differs")
    adapters = {
        cell: load_production_benchmark_adapter(cell, config["prepared_root"])
        for cell, _ in QA_CELLS
    }
    selected = {cell: select_cases(adapters[cell], split) for cell, split in QA_CELLS}
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in selected[cell]:
                for release in ("baseline.no_op", RELEASE):
                    job = {
                        "model": model,
                        "benchmark": cell,
                        "split": split,
                        "case_id": example.example_id,
                        "release": release,
                        "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512,
                        "compressor": identity,
                        "compressor_controls": {
                            "max_source_tokens": 1024,
                            "max_target_tokens": 512,
                            "num_beams": 1,
                        },
                        "predictor": "structured_json_json_object",
                    }
                    job["id"] = digest(job)
                    jobs.append(job)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("claims", "terminals", "preparation"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        {
            "jobs": jobs,
            "model_bindings": bindings,
            "native_import": native,
            "native_import_stderr": smoke.stderr,
            "case_selection": "sorted_source_indices64:66_before_outcomes",
            "automatic_retry": False,
            "paper_results_changed": False,
            "source_context_is_visible_capture_not_live_retrieval": True,
            "physical_helper_cost_once_per_capture_logical_cost_in_each_arm": True,
        },
    )

    def target(job, state, record=None, receipt=None):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model, calls = job["model"], []
        result = {
            "job": job,
            "package_location": __import__("agentinputlab").__file__,
            "model_binding": bindings[model["id"]],
            "input_state_hash": state.state_hash,
            "shared_preparation_receipt": str(receipt) if receipt else None,
            "additional_target_attempts": 1,
            "automatic_retry": False,
        }
        started = time.monotonic()
        try:
            if record is not None:
                state = record.patch.apply(state)
                state = replace(
                    state,
                    budget=replace(
                        state.budget,
                        max_model_calls=state.budget.max_model_calls - record.usage.model_calls,
                        max_output_tokens=state.budget.max_output_tokens
                        - record.usage.output_tokens,
                    ),
                )
            backend = RecordingBackend(
                calls=calls,
                model_id=model["id"],
                base_url=model["base_url"],
                extra_body=model.get("extra_body", {}),
                timeout_seconds=180,
                transport=local_transport,
                deployment_identity={
                    "revision": digest(bindings[model["id"]]["model_json_hashes"]),
                    "server": "vllm",
                },
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
            prediction = predictor.predict(state)
            example = adapters[job["benchmark"]].example(job["split"], job["case_id"])
            result.update(
                status="scored_terminal",
                metrics=dict(adapters[job["benchmark"]].score(example, prediction.prediction)),
            )
        except Exception as exc:
            result.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        result.update(
            calls=calls,
            intervention=to_jsonable(record) if record else None,
            seconds=time.monotonic() - started,
        )
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    pools = {model["id"]: ThreadPoolExecutor(2) for model in config["models"]}
    futures = []
    try:
        for cell, _split in QA_CELLS:
            for example in selected[cell]:
                state = example.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                matching = [
                    job
                    for job in jobs
                    if job["benchmark"] == cell and job["case_id"] == example.example_id
                ]
                for job in matching:
                    if job["release"] == "baseline.no_op":
                        futures.append(pools[job["model"]["id"]].submit(target, job, state))
                prep = (
                    output / "preparation" / digest({"cell": cell, "case_id": example.example_id})
                )
                prep.mkdir()
                worker = RecordedWorker(
                    JsonLineSubprocessWorker(
                        command=(
                            *command,
                            "--checkpoint",
                            str(checkpoint_root),
                            "--checkpoint-manifest",
                            str(checkpoint_manifest),
                            "--acknowledge-noassertion-checkpoint",
                        ),
                        worker_id="toolkit-native-recomp-abstractive",
                        identity=identity,
                        timeout_seconds=900,
                        environment=environment,
                    ),
                    prep,
                )
                record = None
                try:
                    method = create_intervention(
                        RELEASE,
                        checkpoint_revision=asset["checkpoint_revision"],
                        tokenizer_revision=asset["tokenizer_revision"],
                        asset_manifest_hash=manifest_sha,
                        asset_license_acknowledged=True,
                        input_template_hash=RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
                    )
                    record = InterventionRunner().run(
                        method,
                        state,
                        InterventionContext(
                            seed=17,
                            retriever=CapturedContextSource(
                                cell, example.example_id, content_hash(state.context)
                            ),
                            external_runtime=worker,
                        ),
                    )
                    save(
                        prep / "terminal.json",
                        {
                            "status": "native_compressed",
                            "record": to_jsonable(record),
                            "worker_responses": worker.responses,
                            "physical_helper_model_calls": record.usage.model_calls,
                            "reference_exposed": False,
                            "automatic_retry": False,
                        },
                    )
                except Exception as exc:
                    save(
                        prep / "terminal.json",
                        {
                            "status": "failed_preparation",
                            "worker_responses": worker.responses,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "automatic_retry": False,
                        },
                    )
                for job in matching:
                    if job["release"] == "baseline.no_op":
                        continue
                    if record is None:
                        save(
                            output / "terminals" / (job["id"] + ".json"),
                            {
                                "job": job,
                                "status": "capture_dependency_failed",
                                "additional_target_attempts": 0,
                                "calls": [],
                                "preparation_receipt": str(prep / "terminal.json"),
                                "automatic_retry": False,
                            },
                        )
                    else:
                        futures.append(
                            pools[job["model"]["id"]].submit(
                                target, job, state, record, prep / "terminal.json"
                            )
                        )
        for future in futures:
            future.result()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    rows = [json.loads(path.read_text()) for path in (output / "terminals").glob("*.json")]
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "output",
        "upstream-root",
        "worker-python",
        "worker-directory",
        "checkpoint-root",
        "checkpoint-manifest",
        "conversion-proof",
        "worker-addons",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.output,
        args.upstream_root,
        args.worker_python,
        args.worker_directory,
        args.checkpoint_root,
        args.checkpoint_manifest,
        args.conversion_proof,
        args.worker_addons,
    )


if __name__ == "__main__":
    main()
