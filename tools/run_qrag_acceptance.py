"""Real frozen Q-RAG policy with paired NO_OP/four-backbone QA acceptance.

One isolated CPU process owns the native policy and its global device state.
Only an exact visible-state/corpus policy preparation is reused across target
backbones. Target calls use their own serving identities and fixed budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.acquire.qrag import LocalQRAGFrozenPolicyRuntime, QRAGCommonSupport
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.runner import InterventionRunner


def configure_runtime_caches(root):
    """Keep generated compiler files out of shared research caches."""
    paths = {
        "TORCHINDUCTOR_CACHE_DIR": str(root / "runtime-cache/inductor"),
        "TRITON_CACHE_DIR": str(root / "runtime-cache/triton"),
        "TORCH_EXTENSIONS_DIR": str(root / "runtime-cache/extensions"),
    }
    os.environ.update(paths)
    return paths


def execute(config, root):
    configure_runtime_caches(root)
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (root.parent / "keep-services-alive").touch()

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()
    try:
        return execute_jobs(config, root)
    finally:
        stop.set()
        thread.join()


def execute_jobs(config, root):
    import torch

    if (
        config.get("acknowledge_noassertion_source") is not True
        or config.get("acknowledge_checkpoint_license") is not True
    ):
        raise ValueError("mounted source/checkpoint require separate explicit acknowledgements")
    torch.set_num_threads(24)
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda x: x.example_id)[20:22])
        for c, s in QA_CELLS
    }
    compatibility = Path(
        str(resources.files("workers.qrag").joinpath("compatibility_manifest.json"))
    )
    embedder_manifest = Path(config["embedder_root"]) / "agentinputlab-manifest.json"
    policy = LocalQRAGFrozenPolicyRuntime(
        source_root=Path(config["source_root"]),
        checkpoint_root=Path(config["checkpoint_root"]),
        embedder_root=Path(config["embedder_root"]),
        compatibility_manifest_path=compatibility,
        compatibility_manifest_sha256=content_hash(json.loads(compatibility.read_text())),
        embedder_manifest_sha256=content_hash(json.loads(embedder_manifest.read_text())),
        device="cpu",
    )
    method = QRAGCommonSupport()
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in selected[cell]:
                for release in ("baseline.no_op", method.release_id):
                    job = {
                        "model": model,
                        "benchmark": cell,
                        "split": split,
                        "case_id": example.example_id,
                        "release": release,
                        "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512,
                        "helper_device": "cpu",
                        "profile": "augment_captured_visible_context_from_same_visible_passage_corpus",
                    }
                    job["id"] = digest(job)
                    jobs.append(job)
    for name in ("claims", "terminals", "preparation"):
        (root / name).mkdir()
    save(
        root / "manifest.json",
        {
            "jobs": jobs,
            "declared_policy_identity_not_yet_executed": dict(policy.fingerprint()),
            "actual_dependency_versions": {
                x: importlib.metadata.version(x)
                for x in ("torch", "transformers", "numpy", "omegaconf")
            },
            "acknowledge_noassertion_source": True,
            "acknowledge_checkpoint_license": True,
            "paper_results_changed": False,
            "test_references_exposed_to_policy": False,
            "checkpoint_or_model_downloaded": False,
            "automatic_retry": False,
            "existing_captured_context_preserved": True,
            "runtime_cache_environment": {
                key: os.environ.get(key)
                for key in (
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "TORCH_EXTENSIONS_DIR",
                )
            },
            "native_reader_replaced_by_each_explicit_target": True,
            "preparation_reuse": "only_exact_visible_state_candidate_pool_and_frozen_checkpoint",
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "installation_receipt": config["installation_receipt"],
            "installation_receipt_sha256": hashlib.sha256(
                Path(config["installation_receipt"]).read_bytes()
            ).hexdigest(),
        },
    )

    def target(job, state, record=None, receipt=None):
        save(root / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        calls = []
        result = {
            "job": job,
            "model_binding": binding,
            "input_state_hash": state.state_hash,
            "shared_native_preparation_receipt": str(receipt) if receipt else None,
            "package_location": __import__("agentinputlab").__file__,
        }
        started = time.monotonic()
        try:
            patched = record.patch.apply(state) if record else state
            backend = RecordingBackend(
                calls=calls,
                model_id=model["id"],
                base_url=model["base_url"],
                deployment_identity={
                    "revision": digest(binding["model_json_hashes"]),
                    "server": "vllm",
                },
                timeout_seconds=300,
                extra_body=model.get("extra_body", {}),
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
            example = adapters[job["benchmark"]].example(job["split"], job["case_id"])
            result.update(
                status="scored_terminal",
                metrics=dict(adapters[job["benchmark"]].score(example, prediction.prediction)),
                prediction=to_jsonable(prediction),
            )
        except Exception as exc:
            result.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        save(
            root / "terminals" / (job["id"] + ".json"),
            {
                **result,
                "calls": calls,
                "seconds": time.monotonic() - started,
                "additional_target_attempts": 1,
                "automatic_retry": False,
            },
        )
        (root.parent / "keep-services-alive").touch()

    pools = {m["id"]: ThreadPoolExecutor(3) for m in config["models"]}
    futures = []
    prepared = 0
    try:
        for job in jobs:
            if job["release"] == "baseline.no_op":
                example = adapters[job["benchmark"]].example(job["split"], job["case_id"])
                state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
                futures.append(pools[job["model"]["id"]].submit(target, job, state))
        started = time.monotonic()
        save(
            root / "helper-startup-claim.json",
            {
                "started": time.time(),
                "device": "cpu",
                "native_policy_calls_dispatched": False,
                "source_and_weight_integrity_checks_once": True,
            },
        )
        try:
            policy._load()
        except Exception as exc:
            save(
                root / "helper-startup-terminal.json",
                {
                    "status": "failed_startup",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seconds": time.monotonic() - started,
                    "native_selection_unstarted": True,
                },
            )
            save(
                root / "attention.json",
                {
                    "failed_stage": "native_policy_startup",
                    "qrag_selection_cases_unstarted": True,
                    "baseline_jobs_may_complete": True,
                    "automatic_retry": False,
                },
            )
            return
        save(
            root / "helper-startup-terminal.json",
            {
                "status": "native_loaded",
                "seconds": time.monotonic() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "device": "cpu",
                "policy_identity": dict(policy.fingerprint()),
                "state_dicts_strict": True,
                "safe_weights_only": True,
            },
        )
        for cell, values in selected.items():
            for example in values:
                state = example.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                docs = tuple(
                    Document(
                        doc_id=b.block_id or str(i),
                        text=b.text,
                        corpus_revision=adapters[cell].prepared.source_revision,
                    )
                    for i, b in enumerate(state.context)
                )
                retriever = InMemoryBM25Retriever(documents=docs)
                out = (
                    root
                    / "preparation"
                    / digest({"benchmark": cell, "case_id": example.example_id})
                )
                out.mkdir()
                save(
                    out / "claim.json",
                    {
                        "input_state_hash": state.state_hash,
                        "candidate_retriever": retriever.fingerprint(),
                        "started": time.time(),
                        "native_reader_used": False,
                    },
                )
                started = time.monotonic()
                try:
                    record = InterventionRunner().run(
                        method,
                        state,
                        InterventionContext(seed=17, retriever=retriever, external_runtime=policy),
                    )
                except Exception as exc:
                    save(
                        out / "terminal.json",
                        {
                            "status": "failed_preparation",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "seconds": time.monotonic() - started,
                            "partial_native_selection_consumption": "unknown_if_failure_after_dispatch",
                            "automatic_retry": False,
                        },
                    )
                    save(
                        root / "attention.json",
                        {
                            "failed_preparation": str(out),
                            "completed_preparations": prepared,
                            "remaining_native_cases_unstarted": True,
                        },
                    )
                    return
                receipt = out / "terminal.json"
                save(
                    receipt,
                    {
                        "status": "native_selected",
                        "record": to_jsonable(record),
                        "seconds": time.monotonic() - started,
                        "physical_policy_attempts": 1,
                        "same_visible_capture_shared_across_target_backbones": True,
                    },
                )
                prepared += 1
                for job in jobs:
                    if (job["benchmark"], job["case_id"], job["release"]) == (
                        cell,
                        example.example_id,
                        method.release_id,
                    ):
                        futures.append(
                            pools[job["model"]["id"]].submit(target, job, state, record, receipt)
                        )
        for future in futures:
            future.result()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    rows = [json.loads(f.read_text()) for f in (root / "terminals").glob("*.json")]
    save(
        root / "complete.json",
        {
            "expected_downstream": len(jobs),
            "downstream_terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "physical_native_preparations": prepared,
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    execute(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
