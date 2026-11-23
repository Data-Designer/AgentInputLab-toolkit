"""Real controlled RQ-RAG action trees with model-derived confidence.

The depth-one tree has seven calls and three retrievals. This is the declared
backend-neutral profile, not the separately licensed released 7B checkpoint.
Exact saved NO_OP attempts are reused without selecting favourable outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_adaptive_acceptance import baseline_execution_code_unchanged
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.backends import Document, InMemoryBM25Retriever, ModelCapability
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.recipes import RQRAGCommonSupportQueries, RQRAGMethodControls
from agentinputlab.runner import InterventionRunner

RELEASE = "recipe.rqrag.common_support_queries"


def execute(config, root):
    changes = baseline_execution_code_unchanged(
        Path(config["installed_wheel"]), Path(config["baseline_wheel"])
    )
    if changes:
        raise ValueError("RQ-RAG baseline reuse requires identical package bytes")
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda e: e.example_id)[24:26])
        for c, s in QA_CELLS
    }
    prior = {}
    for path in Path(config["baseline_root"]).glob("*/*/terminal.json"):
        row = json.loads(path.read_text())
        j = row["job"]
        if j["release"] != "baseline.no_op":
            continue
        key = (j["model"]["id"], j["benchmark"], j["case_id"])
        if key in prior:
            raise ValueError("ambiguous repeated baseline attempts")
        prior[key] = (path, row)
    controls = RQRAGMethodControls(
        max_depth=1,
        max_nodes=8,
        max_documents=15,
        retrieval_top_k=5,
        max_output_tokens_per_call=64,
        query_temperature=1.0,
        answer_temperature=0.0,
        selection_mode="confidence",
        allow_partial_tree=False,
    )
    jobs = []
    for model in config["models"]:
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        for c, s in QA_CELLS:
            for ex in selected[c]:
                path, old = prior[(model["id"], c, ex.example_id)]
                j = old["job"]
                state = ex.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                if not (
                    j["model"] == model
                    and j["seed"] == 17
                    and j["split"] == s
                    and j["budget"] == {"max_model_calls": 8, "max_output_tokens": 4096}
                    and j["per_call_output_cap"] == 512
                    and old["input_state_hash"] == state.state_hash
                    and old["model_binding"] == binding
                    and len(old["calls"]) == 1
                    and old["calls"][0]["request"]["seed"] == 17
                    and old["calls"][0]["request"]["max_tokens"] == 512
                    and old["status"] in {"scored_terminal", "failed_terminal"}
                ):
                    raise ValueError("saved baseline execution identity differs; no dispatch")
                job = {
                    "model": model,
                    "benchmark": c,
                    "split": s,
                    "case_id": ex.example_id,
                    "release": RELEASE,
                    "seed": 17,
                    "budget": j["budget"],
                    "per_call_output_cap": 512,
                    "baseline_reuse": {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "status": old["status"],
                        "new_baseline_attempts": 0,
                    },
                    "controls": to_jsonable(controls),
                }
                job["id"] = digest(job)
                jobs.append(job)
    (root / "claims").mkdir()
    (root / "terminals").mkdir()
    save(
        root / "manifest.json",
        {
            "jobs": jobs,
            "data_bindings": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
            "profile": "declared_controlled_method_tree_not_released_7b",
            "expected_complete_tree_calls": 7,
            "expected_retrieval_calls": 3,
            "expected_tree_nodes": 8,
            "scorer_labels_used_by_tree": False,
            "baseline_reuses": 32,
            "new_baseline_attempts": 0,
            "automatic_retry": False,
            "paper_results_changed": False,
            "installed_wheel_sha256": hashlib.sha256(
                Path(config["installed_wheel"]).read_bytes()
            ).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (root.parent / "keep-services-alive").touch()

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()

    def target(job):
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
        inner.capabilities = frozenset((*inner.capabilities, ModelCapability.LOGPROBS))
        backend = GPTOSSFinalChannelStopBackend(inner) if model["id"] == "gpt-oss-20b" else inner
        result = {
            "job": job,
            "model_binding": binding,
            "input_state_hash": state.state_hash,
            "package_location": __import__("agentinputlab").__file__,
        }
        stage = "controlled_action_tree"
        started = time.monotonic()
        try:
            record = InterventionRunner(monitor=monitor).run(
                RQRAGCommonSupportQueries(controls=controls),
                state,
                InterventionContext(backend=backend, retriever=retriever, seed=17),
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
            stage = "downstream_and_native_scoring"
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
                "calls": calls,
                "intervention": to_jsonable(record) if record else None,
                "monitor_snapshots": to_jsonable(monitor.list_snapshots()),
                "seconds": time.monotonic() - started,
                "automatic_retry": False,
            },
        )

    def lane(model):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(target, [j for j in jobs if j["model"]["id"] == model["id"]]))

    try:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lane, config["models"]))
    finally:
        stop.set()
        thread.join()
    rows = [json.loads(p.read_text()) for p in (root / "terminals").glob("*.json")]
    save(
        root / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(x["status"] == "scored_terminal" for x in rows),
            "actual_provider_requests": sum(len(x["calls"]) for x in rows),
            "new_baseline_attempts": 0,
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True)
    execute(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
