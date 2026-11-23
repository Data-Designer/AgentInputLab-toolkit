"""Fixed live native LLMCompiler QA cohort, separate from research results.

The original query-only native contract is explicit: benchmark formatting
instructions/history are projected away for the compiler, not silently claimed
supported. Read-only search executes BM25 on the exact visible capture, not a
result replay table or the full Wikipedia corpus. All genuine failures remain.
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
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.decompose.llm_compiler import (
    LLMCompilerControls,
    LLMCompilerFullBridge,
)
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.connectors.tools import ToolResult, ToolSpec
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import LLMCompilerLiveWorker
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import CompositeInterventionRunner

RELEASE = "plan.llm_compiler.full"


class VisibleSearch:
    def __init__(self, state, source_revision):
        self.state, self.calls, self.selected = state, [], []
        self.retriever = InMemoryBM25Retriever(tuple(Document(b.block_id, b.text,
            corpus_revision=source_revision) for b in state.context))

    def list_tools(self):
        return (ToolSpec("search", "search(query: string) -> string: search the supplied source-visible document capture; returns up to five passages",
            {"type": "object", "properties": {"query": {"type": "string"}},
             "required": ["query"], "additionalProperties": False}),)

    def fingerprint(self):
        return {"kind": "visible_capture_bm25_search", "capture_sha256": content_hash(self.state.context),
            "retriever": dict(self.retriever.fingerprint()), "top_k": 5,
            "full_corpus_or_live_wikipedia": False}

    def call(self, call):
        rows = self.retriever.retrieve(call.arguments["query"], top_k=5)
        self.calls.append({"query": call.arguments["query"], "results": to_jsonable(rows)})
        self.selected.extend(r.document.doc_id for r in rows)
        return ToolResult(call.call_id, output="\n\n".join(f"[{r.document.doc_id}] {r.document.text}" for r in rows))


def run(config, output, worker_python, checkout, prompt_bundle):
    bindings = verify_services(config["models"])
    prompts = json.loads(prompt_bundle.read_text())["prompts"]
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"])
                for cell, _ in QA_CELLS}
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            cases = sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[72:74]
            if len(cases) != 2:
                raise ValueError("two source-fixed cases72:74 are required")
            for example in cases:
                for release in ("baseline.no_op", RELEASE):
                    job = {"model": model, "benchmark": cell, "split": split,
                        "case_id": example.example_id, "release": release, "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512, "compiler_streaming": False,
                        "max_replans": 1, "max_tasks": 8, "concurrency": 4,
                        "native_input_projection": "question_only_empty_instruction_and_history" if release == RELEASE else "benchmark_visible_state",
                        "prompt_bundle_hash": content_hash(prompts),
                        "search_scope": "source_visible_capture_bm25_top5_not_full_corpus"}
                    job["id"] = digest(job)
                    jobs.append(job)
    if len(jobs) != 64:
        raise ValueError("four backbones and64fixed positions required")
    output.mkdir(exist_ok=False)
    for name in ("claims", "terminals"):
        (output / name).mkdir()
    save(output / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
        "source_hashes": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
        "selection": "fixed_sorted_source_indices72:74_before_outcomes",
        "eligibility": "controlled_single_query_readonly_search_QA_agent_not_conversational_native_parity",
        "prompt_bundle_sha256": hashlib.sha256(prompt_bundle.read_bytes()).hexdigest(),
        "prompt_hash": content_hash(prompts), "automatic_retry": False,
        "new_model_or_dataset_downloads": 0, "paper_results_changed": False})

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "pid": os.getpid(), "started": time.time()})
        adapter = adapters[job["benchmark"]]
        example = adapter.example(job["split"], job["case_id"])
        original = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        calls, record, worker, tools = [], None, None, None
        model = job["model"]
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            extra_body=model.get("extra_body", {}), timeout_seconds=180, transport=local_transport,
            deployment_identity={"revision": digest(bindings[model["id"]]["model_json_hashes"]), "server": "vllm"})
        monitor, started = RunMonitor(), time.monotonic()
        row = {"job": job, "input_state_hash": original.state_hash,
            "model_binding": bindings[model["id"]], "automatic_retry": False,
            "package_location": __import__("agentinputlab").__file__, "additional_attempts": 1}
        stage = "generation"
        try:
            if job["release"] == RELEASE:
                state = replace(original, instruction="", messages=())
                row["native_projected_state_hash"] = state.state_hash
                tools = VisibleSearch(state, adapter.prepared.source_revision)
                worker = LLMCompilerLiveWorker(python_executable=str(worker_python), checkout=checkout,
                    prompt_bundle=prompt_bundle, backend=backend, tools=tools, timeout_seconds=400)
                controls = LLMCompilerControls(max_tasks=8, max_replans=1, concurrency=4,
                    timeout_seconds=300, streaming=False)
                record = CompositeInterventionRunner(monitor=monitor).run(
                    LLMCompilerFullBridge(prompt_hash=content_hash(prompts), controls=controls), state,
                    InterventionContext(backend=backend, tools=tools, external_runtime=worker, seed=17))
                visible = {b.block_id: b for b in state.context}
                selected = tuple(dict.fromkeys(tools.selected))
                prediction = QAPrediction(answer=record.result.final_answer,
                    support_ids=tuple(dict.fromkeys(visible[i].metadata.get("official_support_id", i) for i in selected)),
                    evidence_ids=(), evidence_texts=tuple(visible[i].text for i in selected))
            else:
                predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                    benchmark_id=job["benchmark"], split=job["split"], model_slot=model["id"], seed=17,
                    builder_parameters={"schema_version": 1, "task": "qa", "qa": {
                        "output_format": "answer_only", "max_tokens": 512, "use_seed": True}},
                    runtime=StaticRuntimeComponents(backend=backend)))
                prediction = predictor.predict(original).prediction
            stage = "native_scoring"
            row.update(status="scored_terminal", prediction=to_jsonable(prediction),
                       metrics=dict(adapter.score(example, prediction)))
        except Exception as exc:
            row.update(status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc))
        row.update(calls=calls, model_calls=len(calls), callback_events=worker.callback_events if worker else [],
            worker_fingerprint=worker.fingerprint() if worker else None,
            search_calls=tools.calls if tools else [], physical_retrieval_calls=len(tools.calls) if tools else 0,
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
        "observed_model_requests": sum(r["model_calls"] for r in rows),
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
