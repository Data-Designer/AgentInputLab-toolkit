"""Real frozen T5-base Adaptive-RAG transfer, with strict saved NO_OP reuse.

The source-declared FLAN-label/T5-base artifact is not retrained or relabelled
as four different model-specific classifiers. Only its question-only neural
route may be cached. Each target executes its selected arm independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from run_qa_method_acceptance import demonstrations
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.acquire import (
    AdaptiveRAGCommonSupport,
    AdaptiveRAGCommonSupportConfig,
    IRCoTCommonSupportConfig,
    T5AdaptiveRAGRouter,
)
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASE = "acquire.adaptive_rag.common_support"


def baseline_execution_code_unchanged(old_wheel, new_wheel):
    """Allow only a change in the router module unused by the NO_OP path."""
    with zipfile.ZipFile(old_wheel) as old, zipfile.ZipFile(new_wheel) as new:
        names = {
            n for n in old.namelist() if n.startswith("agentinputlab/") and not n.endswith("/")
        }
        if names != {
            n for n in new.namelist() if n.startswith("agentinputlab/") and not n.endswith("/")
        }:
            raise ValueError("package execution source file set changed before baseline reuse")
        changed = sorted(n for n in names if old.read(n) != new.read(n))
    allowed = {"agentinputlab/algorithms/acquire/adaptive_rag_common_support.py"}
    if set(changed) - allowed:
        raise ValueError("non-router package bytes changed; saved baseline cannot be reused")
    return changed


@dataclass(frozen=True)
class FrozenQuestionRoute:
    question: str
    route_value: object
    identity: dict

    def route(self, question):
        if question != self.question:
            raise ValueError("cached native route belongs to a different visible question")
        return self.route_value

    def fingerprint(self):
        return self.identity


def execute(config, root):
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

    torch.set_num_threads(8)
    changed = baseline_execution_code_unchanged(
        Path(config["baseline_wheel"]), Path(config["installed_wheel"])
    )
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda e: e.example_id)[20:22])
        for c, s in QA_CELLS
    }
    baseline_rows = {}
    for path in (Path(config["baseline_root"]) / "terminals").glob("*.json"):
        row = json.loads(path.read_text())
        j = row["job"]
        if j["release"] == "baseline.no_op":
            key = (j["model"]["id"], j["benchmark"], j["case_id"])
            if key in baseline_rows:
                raise ValueError(
                    "ambiguous repeated saved baseline; do not pick a favourable attempt"
                )
            baseline_rows[key] = (path, row)
    reuses = []
    jobs = []
    for model in config["models"]:
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        for cell, split in QA_CELLS:
            for ex in selected[cell]:
                path, baseline = baseline_rows[(model["id"], cell, ex.example_id)]
                j = baseline["job"]
                state = ex.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                if not (
                    j["model"] == model
                    and j["seed"] == 17
                    and j["split"] == split
                    and j["budget"] == {"max_model_calls": 8, "max_output_tokens": 4096}
                    and j["per_call_output_cap"] == 512
                    and baseline["input_state_hash"] == state.state_hash
                    and baseline["model_binding"] == binding
                    and len(baseline["calls"]) == 1
                    and baseline["calls"][0]["request"]["seed"] == 17
                    and baseline["calls"][0]["request"]["max_tokens"] == 512
                    and baseline["status"] in {"scored_terminal", "failed_terminal"}
                ):
                    raise ValueError(
                        "saved baseline execution identity differs; no inference dispatched"
                    )
                reuse = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": baseline["status"],
                    "case_id": ex.example_id,
                    "benchmark": cell,
                    "model": model["id"],
                    "new_baseline_attempts": 0,
                }
                reuses.append(reuse)
                job = {
                    "model": model,
                    "benchmark": cell,
                    "split": split,
                    "case_id": ex.example_id,
                    "release": RELEASE,
                    "seed": 17,
                    "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                    "per_call_output_cap": 512,
                    "router_profile": "frozen_flan_label_t5_base_transfer",
                    "baseline_reuse": reuse,
                }
                job["id"] = digest(job)
                jobs.append(job)
    _, ircot_prompt, _ = demonstrations(Path(config["upstream_root"]) / "upstream-paper")
    ircot_config = IRCoTCommonSupportConfig(
        demonstrations=ircot_prompt,
        prompt_sha256=hashlib.sha256(ircot_prompt.encode()).hexdigest(),
        max_tokens_per_call=512,
    )
    artifact_root = Path(config["router_artifact"])
    manifest_path = artifact_root / "artifact-manifest.json"
    router_manifest = json.loads(manifest_path.read_text())
    router = T5AdaptiveRAGRouter(
        artifact_root=artifact_root,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        device="cpu",
    )
    for folder in ("preparation", "claims", "terminals"):
        (root / folder).mkdir()
    save(
        root / "manifest.json",
        {
            "jobs": jobs,
            "saved_no_op_reuses": reuses,
            "package_changes_unused_by_no_op": changed,
            "router_manifest": router_manifest,
            "router_transfer_not_target_specific_training": True,
            "automatic_retry": False,
            "paper_results_changed": False,
            "target_and_router_cost_separate": True,
            "source_question_only_classifier": True,
            "data_bindings": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "installed_wheel_sha256": hashlib.sha256(
                Path(config["installed_wheel"]).read_bytes()
            ).hexdigest(),
            "baseline_wheel_sha256": hashlib.sha256(
                Path(config["baseline_wheel"]).read_bytes()
            ).hexdigest(),
        },
    )

    def target(job, frozen_router, receipt):
        save(root / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        adapter = adapters[job["benchmark"]]
        ex = adapter.example(job["split"], job["case_id"])
        state = ex.visible_state(budget=ExecutionBudget(**job["budget"]))
        docs = tuple(
            Document(
                doc_id=b.block_id or str(i),
                text=b.text,
                corpus_revision=adapter.prepared.source_revision,
            )
            for i, b in enumerate(state.context)
        )
        retriever = InMemoryBM25Retriever(documents=docs)
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        calls, monitor, record = [], RunMonitor(), None
        inner = RecordingBackend(
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
        backend = GPTOSSFinalChannelStopBackend(inner) if model["id"] == "gpt-oss-20b" else inner
        result = {
            "job": job,
            "model_binding": binding,
            "shared_router_receipt": str(receipt),
            "input_state_hash": state.state_hash,
            "package_location": __import__("agentinputlab").__file__,
        }
        stage = "selected_native_controlled_arm"
        started = time.monotonic()
        try:
            method = AdaptiveRAGCommonSupport(
                router=frozen_router, config=AdaptiveRAGCommonSupportConfig(ircot=ircot_config)
            )
            record = InterventionRunner(monitor=monitor).run(
                method, state, InterventionContext(backend=backend, retriever=retriever, seed=17)
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
                    runtime=StaticRuntimeComponents(backend=inner),
                )
            )
            prediction = predictor.predict(patched)
            result.update(
                status="scored_terminal",
                prediction=to_jsonable(prediction),
                metrics=dict(adapter.score(ex, prediction.prediction)),
            )
        except Exception as exc:
            result.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        save(
            root / "terminals" / (job["id"] + ".json"),
            {
                **result,
                "seconds": time.monotonic() - started,
                "calls": calls,
                "intervention": to_jsonable(record) if record else None,
                "monitor_snapshots": to_jsonable(monitor.list_snapshots()),
                "automatic_retry": False,
            },
        )

    pools = {m["id"]: ThreadPoolExecutor(2) for m in config["models"]}
    futures, routes = [], []
    try:
        for cell, examples in selected.items():
            for ex in examples:
                receipt = (
                    root / "preparation" / (digest({"cell": cell, "id": ex.example_id}) + ".json")
                )
                started = time.monotonic()
                try:
                    route = router.route(ex.query)
                except Exception as exc:
                    save(
                        receipt,
                        {
                            "status": "failed_router",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "consumption": "unknown_if_failure_after_forward",
                            "seconds": time.monotonic() - started,
                        },
                    )
                    save(
                        root / "attention.json",
                        {"failed_router": str(receipt), "other_questions_unstarted": True},
                    )
                    return
                save(
                    receipt,
                    {
                        "status": "actual_neural_route",
                        "question": ex.query,
                        "route": to_jsonable(route),
                        "router_identity": dict(router.fingerprint()),
                        "seconds": time.monotonic() - started,
                        "physical_router_forwards": 1,
                        "actual_target_specific_router_fit": False,
                    },
                )
                frozen = FrozenQuestionRoute(
                    ex.query,
                    route,
                    {
                        **dict(router.fingerprint()),
                        "saved_route_receipt_sha256": hashlib.sha256(
                            receipt.read_bytes()
                        ).hexdigest(),
                    },
                )
                routes.append(route.label)
                for job in jobs:
                    if (job["benchmark"], job["case_id"]) == (cell, ex.example_id):
                        futures.append(
                            pools[job["model"]["id"]].submit(target, job, frozen, receipt)
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
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "physical_router_forwards": len(routes),
            "observed_route_labels": dict(Counter(routes)),
            "baseline_reuses": len(reuses),
            "new_baseline_attempts": 0,
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
