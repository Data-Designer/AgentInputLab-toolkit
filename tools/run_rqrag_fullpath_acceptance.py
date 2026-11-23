"""Real complete RQ-RAG trees; post-hoc leaf selectors reuse identical native calls.

The fixed cases and NO_OP receipts are taken from the complete prior cohort,
including its failures. No best-run selection, test-label training or LLM retry.
PPL/ensemble exports must replay byte-equivalent native HTTP payloads, with no
network fallback, and are not additional scientific execution attempts.
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
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.recipes import RQRAGCompletionBackend, RQRAGMethodControls, RQRAGMethodTreeSearch
from agentinputlab.runner import CompositeInterventionRunner

RELEASE = "recipe.rqrag.method_tree_search"
MODES = ("confidence", "ppl", "ensemble")


def composite_qa_prediction(result, state):
    """Map selected visible IDs, never gold labels, to the QA scorer contract."""
    blocks = {block.block_id: block for block in state.context}
    selected = tuple(result.metadata["selected_document_ids"])
    if any(identifier not in blocks for identifier in selected):
        raise ValueError("selected retrieval evidence is outside the original visible capture")
    supports = tuple(dict.fromkeys(
        blocks[identifier].metadata.get("official_support_id", identifier) for identifier in selected
    ))
    return QAPrediction(answer=result.final_answer, support_ids=supports,
        # Retrieval IDs are NOT inferred 2Wiki knowledge-graph triples.
        evidence_ids=(), evidence_texts=tuple(blocks[identifier].text for identifier in selected))


def baseline_code_check(previous_wheel, current_wheel):
    with zipfile.ZipFile(previous_wheel) as old, zipfile.ZipFile(current_wheel) as new:
        old_names = {n for n in old.namelist() if n.startswith("agentinputlab/")}
        new_names = {n for n in new.namelist() if n.startswith("agentinputlab/")}
        changed = {n for n in old_names & new_names if old.read(n) != new.read(n)}
        if (old_names - new_names or new_names - old_names != {"agentinputlab/recipes/rqrag_completion.py"}
                or changed != {"agentinputlab/recipes/__init__.py"}):
            raise ValueError("NO_OP execution bytes differ beyond the unused RQ-RAG addition")
    return {"added": sorted(new_names - old_names), "changed": sorted(changed),
            "baseline_execution_modules_unchanged": True}


class NativeCalls:
    def __init__(self, saved=None):
        self.saved, self.calls, self.index = saved, [], 0
        self.payload_mismatch = False

    def transport(self, url, headers, payload, timeout):
        if self.saved is not None:
            if self.index >= len(self.saved):
                raise ValueError("saved native tree exhausted; network forbidden")
            old = self.saved[self.index]
            self.index += 1
            if url != old["url"] or payload != old["payload"]:
                self.payload_mismatch = True
                raise ValueError("post-hoc selector changed a native generation payload")
            if "response" not in old:
                raise RuntimeError("preserved original native transport failure; no retry")
            return old["response"]
        row = {"url": url, "payload": payload, "started": time.time()}
        self.calls.append(row)
        try:
            value = local_transport(url, headers, payload, timeout)
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
            raise
        row["response"] = value
        return value


def run(config, output, prior, previous_wheel, current_wheel):
    reuse_check = baseline_code_check(previous_wheel, current_wheel)
    bindings = verify_services(config["models"])
    probes = json.loads((prior.parent / "rqrag-completion-probes-v1/complete.json").read_text())
    if probes["native_ready"] != 4:
        raise ValueError("all four native probability interfaces must be ready before dispatch")
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"])
                for cell, _ in QA_CELLS}
    baselines = {}
    for path in (prior / "terminals").glob("*.json"):
        old = json.loads(path.read_text())
        if old["job"]["release"] == "baseline.no_op":
            key = (old["job"]["model"]["id"], old["job"]["benchmark"], old["job"]["case_id"])
            if key in baselines:
                raise ValueError("ambiguous baseline source; no favourable-run selection")
            baselines[key] = (path, old)
    controls = RQRAGMethodControls(max_depth=1, max_nodes=8, max_documents=15,
        retrieval_top_k=5, max_output_tokens_per_call=64, allow_partial_tree=False)
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            selected = sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[64:66]
            if len(selected) != 2:
                raise ValueError("frozen cases at source indices64:66 missing")
            for example in selected:
                path, old = baselines[(model["id"], cell, example.example_id)]
                state = example.visible_state(budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
                if (old["model_binding"] != bindings[model["id"]]
                        or old["input_state_hash"] != state.state_hash
                        or old["job"]["model"] != model or old["job"]["seed"] != 17):
                    raise ValueError("saved NO_OP model/source/seed identity differs")
                base = {"model": model, "benchmark": cell, "split": split,
                        "case_id": example.example_id, "release": RELEASE, "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 64,
                        "baseline_reuse": {"path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "status": old["status"], "new_baseline_attempts": 0}}
                for mode in MODES:
                    job = {**base, "controls": to_jsonable(replace(controls, selection_mode=mode))}
                    job["id"] = digest(job)
                    jobs.append(job)
    if len(jobs) != 96 or len(baselines) != 32:
        raise ValueError("fixed cohort identities incomplete")
    output.mkdir(exist_ok=False)
    (output / "terminals").mkdir()
    (output / "claims").mkdir()
    save(output / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
        "data_bindings": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
        "baseline_reuse_check": reuse_check, "new_baseline_attempts": 0,
        "physical_generation_trees": 32, "derived_leaf_selection_exports": 64,
        "complete_depth_one_tree_calls": 7, "automatic_retry": False,
        "paper_results_changed": False, "released_7b_or_paper_parity": False,
        "selection": "all prior fixed source indices64:66; no outcome filtering",
        "installed_wheel_sha256": hashlib.sha256(current_wheel.read_bytes()).hexdigest()})

    def one(source_job):
        adapter = adapters[source_job["benchmark"]]
        example = adapter.example(source_job["split"], source_job["case_id"])
        state = example.visible_state(budget=ExecutionBudget(**source_job["budget"]))
        retriever = InMemoryBM25Retriever(documents=tuple(Document(
            doc_id=block.block_id, text=block.text, corpus_revision=adapter.prepared.source_revision)
            for block in state.context))
        original_calls, source_path = None, None
        for mode in MODES:
            job = next(j for j in jobs if j["model"] == source_job["model"]
                and j["benchmark"] == source_job["benchmark"] and j["case_id"] == source_job["case_id"]
                and j["controls"]["selection_mode"] == mode)
            save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
            native = NativeCalls(original_calls)
            monitor, record = RunMonitor(), None
            model = job["model"]
            backend = RQRAGCompletionBackend(model_id=model["id"], base_url=model["base_url"],
                api_key_env="AGENTINPUTLAB_TOOLKIT_LOCAL_NO_AUTH", timeout_seconds=180,
                deployment_identity={"revision": digest(bindings[model["id"]]["model_json_hashes"]),
                                     "server": "vllm"}, transport=native.transport)
            row = {"job": job, "input_state_hash": state.state_hash, "automatic_retry": False,
                   "model_binding": bindings[model["id"]],
                   "package_location": __import__("agentinputlab").__file__,
                   "additional_attempts": int(mode == "confidence"),
                   "evidence_role": "execution_terminal" if mode == "confidence" else "derived_leaf_selection",
                   "model_calls": 0 if mode != "confidence" else None}
            started = time.monotonic()
            try:
                record = CompositeInterventionRunner(monitor=monitor).run(
                    RQRAGMethodTreeSearch(controls=replace(controls, selection_mode=mode)), state,
                    InterventionContext(backend=backend, retriever=retriever, seed=17))
                if original_calls is not None and native.index != len(original_calls):
                    raise ValueError("derived selector did not consume exactly the saved tree")
                prediction = composite_qa_prediction(record.result, state)
                row.update(status="scored_terminal", prediction=to_jsonable(prediction),
                           metrics=dict(adapter.score(example, prediction)))
            except Exception as exc:
                row.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
            if source_path:
                row.update(original_terminal_path=str(source_path),
                    original_terminal_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    zero_network_payload_matched_replay=not native.payload_mismatch,
                    replayed_native_requests=native.index,
                    replay_usage_is_logical_not_additional_physical_cost=True)
            row.update(calls=native.calls, intervention=to_jsonable(record) if record else None,
                monitor_snapshots=to_jsonable(monitor.list_snapshots()), seconds=time.monotonic() - started)
            path = output / "terminals" / (job["id"] + ".json")
            save(path, row)
            if mode == "confidence":
                original_calls, source_path = native.calls, path
            (output.parent / "keep-services-alive").touch()
    sources = [job for job in jobs if job["controls"]["selection_mode"] == "confidence"]
    limits = {model["id"]: threading.BoundedSemaphore(2) for model in config["models"]}

    def bounded_one(job):
        with limits[job["model"]["id"]]:
            one(job)

    # Two independent cases per own service at a time, four service lanes.
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(bounded_one, sorted(sources, key=lambda j: (j["benchmark"], j["case_id"], j["model"]["id"]))))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    save(output / "complete.json", {"expected": 96, "terminal": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "physical_generation_trees": 32, "derived_leaf_selection_exports": 64,
        "full_toolkit_acceptance_complete": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "prior", "previous-wheel", "current-wheel"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.prior, args.previous_wheel, args.current_wheel)


if __name__ == "__main__":
    main()
