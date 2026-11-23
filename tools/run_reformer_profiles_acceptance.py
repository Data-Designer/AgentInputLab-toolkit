"""Four explicit native ReFormeR inference profiles on fixed bounded QA cases.

Automatic and manually fixed pattern paths each generate once per target/case.
The corresponding q-concat-r arm executes the same pinned native source against
exact saved completion requests/results, with no network fallback. Its reader
is a new real generation; logical helper costs remain in each arm. Physical
helper costs are counted only once. These are compatibility samples, not full
benchmark scores or upstream-checkpoint/paper-result parity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from acceptance_method_bindings import REFORMER_RELEASES, bind_external_rewriter
from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner
from workers.reformer.driver import (
    BackendConfig,
    CompletionResult,
    DriverError,
    OpenAICompatibleBackend,
    ReFormeRRuntime,
    _failure_response,
)


class CompletionTape:
    """Exact native model-seam replay, not a fabricated worker response."""

    def __init__(self, backend, *, saved=None, output=None):
        self.backend, self.saved, self.output = backend, saved, output
        self.calls, self.index, self.mismatch = [], 0, False

    def fingerprint(self):
        return self.backend.fingerprint()

    def complete(self, **kwargs):
        request = to_jsonable(kwargs)
        if self.saved is not None:
            if self.index >= len(self.saved):
                self.mismatch = True
                raise DriverError("REPLAY_EXHAUSTED", "saved completions exhausted; network forbidden")
            row = self.saved[self.index]
            self.index += 1
            if row["request"] != request:
                self.mismatch = True
                raise DriverError("REPLAY_MISMATCH", "native completion request changed")
            if "failure" in row:
                error = row["failure"]
                raise DriverError(error["code"], error["message"], consumed=error["consumed"],
                    provider_usage=error["provider_usage"], diagnostics=error["diagnostics"])
            return CompletionResult(**row["completion"])
        row = {"request": request, "started": time.time()}
        self.calls.append(row)
        try:
            result = self.backend.complete(**kwargs)
            row["completion"] = asdict(result)
            return result
        except DriverError as exc:
            row["failure"] = {"code": exc.code, "message": str(exc), "consumed": exc.consumed,
                "provider_usage": exc.provider_usage, "diagnostics": dict(exc.diagnostics)}
            raise
        finally:
            if self.output is not None:
                save(self.output / (str(len(self.calls)) + ".json"), row)

    def require_complete_replay(self):
        if self.saved is not None and (self.mismatch or self.index != len(self.saved)):
            raise RuntimeError("native replay must consume exactly the saved source calls")


class NativeWorker:
    """One isolated per-case runtime using already loaded, hash-pinned source."""

    worker_id = "toolkit-native-reformer-four-profiles"

    def __init__(self, template, config, tape, identity):
        self.runtime = copy.copy(template)
        self.runtime.backend_config, self.runtime.backend = config, tape
        self.identity, self.responses = identity, []

    def fingerprint(self):
        return {**self.identity, "worker_id": self.worker_id,
            "execution_transport": "in_process_pinned_source_model_seam",
            "completion_source": "exact_saved_native_calls" if self.runtime.backend.saved is not None else "live"}

    def call(self, payload):
        try:
            response = self.runtime.run(payload)
        except DriverError as exc:
            response = _failure_response(payload["request_id"], exc)
        self.responses.append({"request": to_jsonable(payload), "response": response})
        return response


def target_backend(model, binding, calls):
    return RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
        extra_body=model.get("extra_body", {}), timeout_seconds=180, transport=local_transport,
        deployment_identity={"revision": digest(binding["model_json_hashes"]), "server": "vllm"})


def run(config, output, upstream):
    # Only this isolated operator process; no inherited proxy affects local APIs.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["AIL_TOOLKIT_WORKER_LOCAL_KEY"] = "local-service-only"
    bindings = verify_services(config["models"])
    adapters = {cell: load_production_benchmark_adapter(cell, config["prepared_root"])
                for cell, _ in QA_CELLS}
    pattern_path = upstream / "upstream-paper/ReFormeR-72e5245/reformer/patterns/extracted_patterns.json"
    # Source order, not model outcome: one fixed pattern for all manual profiles.
    manual_id = json.loads(pattern_path.read_text())[0]["pattern_name"]
    jobs = []
    configs, methods, identities = {}, {}, {}
    for model in config["models"]:
        backend = target_backend(model, bindings[model["id"]], [])
        for release in REFORMER_RELEASES:
            method, worker = bind_external_rewriter(release, upstream=upstream, backend=backend,
                model=model, dataset_revision="visible_prepared_source", benchmark="qa",
                manual_pattern_id=manual_id if release in REFORMER_RELEASES[2:] else "")
            methods[(model["id"], release)] = method
            raw = worker.command[worker.command.index("--backend-config-json") + 1]
            configs[model["id"]] = BackendConfig.from_json(raw)
            identities[model["id"]] = dict(worker.identity)
        for cell, split in QA_CELLS:
            examples = sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[68:70]
            if len(examples) != 2:
                raise ValueError("two fixed cases at source indices68:70 are required")
            for example in examples:
                for release in ("baseline.no_op", *REFORMER_RELEASES):
                    job = {"model": model, "benchmark": cell, "split": split,
                        "case_id": example.example_id, "release": release, "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output_cap": 512,
                        "manual_pattern_id": manual_id if release in REFORMER_RELEASES[2:] else ""}
                    job["id"] = digest(job)
                    jobs.append(job)
    if len(jobs) != 160 or len({m["id"] for m in config["models"]}) != 4:
        raise ValueError("exactly four distinct backbones and 160 fixed positions required")
    first = configs[config["models"][0]["id"]]
    # Load source modules only once before threads; each case gets a shallow
    # runtime copy and a new source Reformulator/PromptManager/usage tracker.
    template = ReFormeRRuntime(pattern_path.parents[2], first, OpenAICompatibleBackend(first),
        source_license_acknowledged=True)
    output.mkdir(exist_ok=False)
    for name in ("claims", "terminals", "preparation"):
        (output / name).mkdir()
    save(output / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
        "prepared_source_hashes": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
        "manual_pattern_selection": "first_source_order_before_outcomes",
        "pattern_file_sha256": hashlib.sha256(pattern_path.read_bytes()).hexdigest(),
        "case_selection": "fixed_sorted_indices68:70_before_outcomes",
        "native_runtime": identities, "physical_helper_paths": 64,
        "cpu_assembly_native_replays": 64, "new_dataset_or_model_downloads": 0,
        "automatic_retry": False, "paper_results_changed": False,
        "source_license_operator_acknowledged": True,
        "official_checkpoint_or_paper_result_parity_claimed": False})

    def one(source):
        model = source["model"]
        adapter = adapters[source["benchmark"]]
        example = adapter.example(source["split"], source["case_id"])
        state = example.visible_state(budget=ExecutionBudget(**source["budget"]))
        retriever = InMemoryBM25Retriever(documents=tuple(Document(doc_id=b.block_id, text=b.text,
            corpus_revision=adapter.prepared.source_revision) for b in state.context))
        saved_by_mode = {}
        for release in ("baseline.no_op", *REFORMER_RELEASES):
            job = next(j for j in jobs if j["model"] == model and j["benchmark"] == source["benchmark"]
                and j["case_id"] == source["case_id"] and j["release"] == release)
            save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time(), "pid": os.getpid()})
            calls, record, worker, tape = [], None, None, None
            monitor, stage, started = RunMonitor(), "intervention", time.monotonic()
            backend = target_backend(model, bindings[model["id"]], calls)
            row = {"job": job, "input_state_hash": state.state_hash,
                "model_binding": bindings[model["id"]], "automatic_retry": False,
                "package_location": __import__("agentinputlab").__file__}
            is_replay = release in (REFORMER_RELEASES[1], REFORMER_RELEASES[3])
            mode = "manual" if release in REFORMER_RELEASES[2:] else "automatic"
            try:
                if release != "baseline.no_op":
                    config_ = configs[model["id"]]
                    prep = output / "preparation" / job["id"]
                    prep.mkdir()
                    saved = saved_by_mode[mode][0] if is_replay else None
                    tape = CompletionTape(OpenAICompatibleBackend(config_), saved=saved, output=prep)
                    worker = NativeWorker(template, config_, tape, identities[model["id"]])
                    record = InterventionRunner(monitor=monitor).run(methods[(model["id"], release)], state,
                        InterventionContext(backend=backend, retriever=retriever, external_runtime=worker, seed=17))
                    tape.require_complete_replay()
                stage = "downstream_generation"
                used = monitor.list_snapshots()[-1].usage if monitor.list_snapshots() else None
                budget = replace(state.budget,
                    max_model_calls=8 - (used.model_calls if used else 0),
                    max_output_tokens=4096 - (used.output_tokens if used else 0))
                downstream = record.patch.apply(state) if record else state
                predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                    benchmark_id=job["benchmark"], split=job["split"], model_slot=model["id"], seed=17,
                    builder_parameters={"schema_version": 1, "task": "qa", "qa": {
                        "output_format": "structured_json", "max_tokens": 512, "use_seed": True}},
                    runtime=StaticRuntimeComponents(backend=backend)))
                prediction = predictor.predict(replace(downstream, budget=budget))
                stage = "native_scoring"
                row.update(status="scored_terminal", prediction=to_jsonable(prediction.prediction),
                    metrics=dict(adapter.score(example, prediction.prediction)))
            except Exception as exc:
                row.update(status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc))
            if tape is not None:
                # Validate even failed source/parser replay; never fall back to
                # another model call to obtain a more favourable result.
                tape.require_complete_replay()
                row.update(native_completions=tape.calls, worker_responses=worker.responses,
                    physical_helper_calls=len(tape.calls), replayed_helper_calls=tape.index,
                    exact_native_completion_replay=is_replay,
                    logical_helper_usage_not_additional_physical_cost=is_replay)
                if is_replay:
                    path = saved_by_mode[mode][1]
                    row.update(source_terminal_path=str(path),
                        source_terminal_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            row.update(calls=calls, observed_downstream_calls=len(calls),
                intervention=to_jsonable(record) if record else None,
                monitor_snapshots=to_jsonable(monitor.list_snapshots()), seconds=time.monotonic() - started)
            path = output / "terminals" / (job["id"] + ".json")
            save(path, row)
            if release in (REFORMER_RELEASES[0], REFORMER_RELEASES[2]):
                saved_by_mode[mode] = (tape.calls if tape else [], path)
            (output.parent / "keep-services-alive").touch()

    sources = [j for j in jobs if j["release"] == "baseline.no_op"]
    limits = {m["id"]: threading.BoundedSemaphore(2) for m in config["models"]}

    def bounded(source):
        with limits[source["model"]["id"]]:
            one(source)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(bounded, sorted(sources, key=lambda j: (j["benchmark"], j["case_id"], j["model"]["id"]))))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    save(output / "complete.json", {"expected": 160, "terminal": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "physical_helper_calls": sum(r.get("physical_helper_calls", 0) for r in rows),
        "physical_downstream_calls": sum(r["observed_downstream_calls"] for r in rows),
        "cpu_assembly_replays": sum(r.get("exact_native_completion_replay", False) for r in rows),
        "full_toolkit_acceptance_complete": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "output", "upstream-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.upstream_root)


if __name__ == "__main__":
    main()
