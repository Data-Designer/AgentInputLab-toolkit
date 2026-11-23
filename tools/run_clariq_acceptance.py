"""Fixed, small ClariQ compatibility cohort using caller-mounted native scorer.

Only two initial requests and the small question bank are needed. Source rows,
responses, run files and costs are saved outside the public source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acceptance_method_bindings import EXTERNAL_REWRITERS, bind_external_rewriter
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.backends import GenerationRequest
from agentinputlab.benchmarks.clariq import CLARIQ_REVISION, ClariQDevBundle, ClariQPrediction
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASES = ("baseline.no_op", *EXTERNAL_REWRITERS)


def verify_services(models):
    """Compare exact owned API identity and served alias, not PID alone."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    bindings = {}
    for model in models:
        directory = Path(model["service_directory"])
        if not (directory / "ready.json").exists() or any(
            (directory / name).exists() for name in ("attention.json", "release.json")
        ):
            raise RuntimeError("own service not ready")
        binding = json.loads((directory / "started.json").read_text())
        old = binding["identity"]
        proc = Path("/proc") / str(old["pid"])
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        actual = {"pid": old["pid"], "uid": proc.stat().st_uid, "parent": int(stat[1]),
                  "start": stat[19], "argv_sha256": hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest()}
        if actual["uid"] != os.getuid() or any(str(old[key]) != str(value) for key, value in actual.items()):
            raise RuntimeError("exact own API identity differs")
        with opener.open(model["base_url"] + "/models", timeout=20) as response:
            if model["id"] not in {row["id"] for row in json.load(response)["data"]}:
                raise RuntimeError("served alias differs")
        bindings[model["id"]] = binding
    return bindings


def select_zero_call_entry_repairs(jobs, previous, fingerprint):
    prior_manifest = json.loads((previous / "manifest.json").read_text())
    if prior_manifest["source"] != fingerprint or prior_manifest["jobs"] != jobs:
        raise ValueError("ClariQ repair source inputs or frozen jobs differ")
    selected, old_terms = {}, []
    for path in (previous / "terminals").glob("*.json"):
        term = json.loads(path.read_text())
        old_terms.append(term)
        if term.get("error") not in {
            "ReFormeR context retriever dependency is required",
            "InfoCQR retrieval corpus fingerprint is required",
        }:
            continue
        if (term["status"] != "failed_terminal" or term.get("stage") != "intervention"
                or term.get("calls") != [] or any(
                    snapshot["usage"]["model_calls"] != 0 for snapshot in term["monitor_snapshots"])):
            raise ValueError("ClariQ entry guard has consumed or ambiguous model usage")
        selected[term["job"]["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not selected:
        raise ValueError("no proven zero-call retriever binding guards to repair")
    return [job for job in jobs if job["id"] in selected], selected, old_terms


def run(config, output, source_root, upstream, *, accept_license_review=False, repair_zero_call_root=None):
    bundle = ClariQDevBundle(source_root, accept_license_review=accept_license_review)
    bindings = verify_services(config["models"])
    examples = bundle.examples()[:2]  # fixed lexicographic source IDs, before outcomes
    jobs = []
    for model in config["models"]:
        for example in examples:
            for release in RELEASES:
                job = {"model": model, "release": release, "benchmark": "clariq.main",
                       "case_id": example.example_id, "seed": 17,
                       "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                       "per_call_tokens": 512, "ranking": "full_small_question_bank_bm25_top30"}
                job["id"] = digest(job)
                jobs.append(job)
    repairs, old_terms = {}, []
    if repair_zero_call_root is not None:
        jobs, repairs, old_terms = select_zero_call_entry_repairs(jobs, repair_zero_call_root, bundle.fingerprint())
    output.mkdir(parents=True, exist_ok=False)
    (output / "terminals").mkdir()
    (output / "claims").mkdir()
    save(output / "manifest.json", {"jobs": jobs, "source": bundle.fingerprint(),
         "model_bindings": bindings, "selection": "two lexicographically first dev topic IDs",
         "scope": "RQ1/RQ2 fixed compatibility subset, not RQ3 document utility or full dev",
         "automatic_retry": False, "paper_results_changed": False,
         "zero_call_entry_repair_sources": repairs})
    score_lock = threading.Lock()

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "pid": os.getpid()})
        calls, record = [], None
        model, source = job["model"], bindings[job["model"]["id"]]
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            deployment_identity={"revision": digest(source["model_json_hashes"]), "server": "vllm"},
            timeout_seconds=180, extra_body=model.get("extra_body", {}), transport=local_transport)
        example = next(e for e in examples if e.example_id == job["case_id"])
        state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        result = {"job": job, "package_location": __import__("agentinputlab").__file__,
                  "input_state_hash": state.state_hash, "additional_attempts": 1,
                  "automatic_retry": False, "model_binding": source}
        monitor, started, stage = RunMonitor(), time.monotonic(), "intervention"
        try:
            if job["release"] != "baseline.no_op":
                method, runtime = bind_external_rewriter(job["release"], upstream=upstream,
                    backend=backend, model=model, dataset_revision=CLARIQ_REVISION, benchmark="clariq.main")
                record = InterventionRunner(monitor=monitor).run(method, state, InterventionContext(
                    backend=backend, external_runtime=runtime,
                    retriever=bundle.question_retriever, seed=17))
                state = record.patch.apply(state)
            stage = "clarification_need_and_question_ranking"
            ranked = bundle.rank_questions(state.query, top_k=30)
            used = monitor.list_snapshots()[-1].usage if monitor.list_snapshots() else None
            used_calls, used_tokens = (used.model_calls, used.output_tokens) if used else (0, 0)
            if used_calls >= 8 or used_tokens >= 4096:
                raise RuntimeError("shared intervention budget exhausted before downstream")
            response = backend.generate(GenerationRequest(
                prompt=state.instruction + "\n\nInitial request: " + state.query,
                max_tokens=min(512, 4096 - used_tokens), seed=17,
                response_format={"type": "json_object"}))
            parsed = json.loads(response.text)
            if not isinstance(parsed, dict) or set(parsed) != {"clarification_need"}:
                raise ValueError("ClariQ need response must have exactly one key")
            prediction = ClariQPrediction(parsed["clarification_need"], ranked)
            result["prediction"] = to_jsonable(prediction)
            stage = "native_scoring"
            with score_lock:
                score = bundle.evaluate({example.example_id: prediction},
                    topic_ids=[example.example_id], output=output / "scores" / job["id"])
            result.update(status="scored_terminal", metrics=score["metrics"],
                          native_score_receipt=str(output / "scores" / job["id"] / "native-evaluation.json"))
        except Exception as exc:
            result.update(status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc))
        result.update(seconds=time.monotonic() - started, calls=calls,
                      observed_downstream_http_calls=len(calls),
                      # Worker optimizer usage is retained in its intervention trace;
                      # it must not be misreported as zero just because it used IPC.
                      monitor_snapshots=to_jsonable(monitor.list_snapshots()),
                      intervention=to_jsonable(record) if record else None)
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    def per_model(model):
        selected = [job for job in jobs if job["model"]["id"] == model["id"]]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(one, selected))
    with ThreadPoolExecutor(len(config["models"])) as pool:
        list(pool.map(per_model, config["models"]))
    terminals = [json.loads(path.read_text()) for path in (output / "terminals").glob("*.json")]
    aggregate_terminals = terminals + [term for term in old_terms if term["job"]["id"] not in repairs]
    # Aggregate native F1 is computed over all selected topics, never averaged
    # from case F1 and never restricted to successful-only predictions.
    for model in config["models"]:
        for release in RELEASES:
            selected = [t for t in aggregate_terminals if t["job"]["model"]["id"] == model["id"]
                        and t["job"]["release"] == release]
            predictions = {t["job"]["case_id"]: ClariQPrediction(**t["prediction"])
                           for t in selected if t["status"] == "scored_terminal"}
            bundle.evaluate(predictions, topic_ids=[e.example_id for e in examples],
                            output=output / "aggregates" / model["id"] / release)
    save(output / "complete.json", {"expected": len(jobs), "terminal": len(terminals),
         "terminal_statuses": dict(Counter(t["status"] for t in terminals)),
         "covered_positions_with_prior_non_retried_terminals": len(aggregate_terminals),
         "aggregate_terminal_statuses": dict(Counter(t["status"] for t in aggregate_terminals)),
         "full_toolkit_acceptance_complete": False, "paper_results_changed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-license-review", action="store_true")
    parser.add_argument("--repair-zero-call-root", type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.source_root, args.upstream_root,
        accept_license_review=args.accept_license_review, repair_zero_call_root=args.repair_zero_call_root)


if __name__ == "__main__":
    main()
