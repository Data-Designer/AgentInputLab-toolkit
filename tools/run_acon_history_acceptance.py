"""Real ToolSandbox episodes with live source-native ACON history compression.

All inputs/outputs are isolated from paper runs. Every episode has its own
process, native environment, live-history state and upstream worker. No native
task or simulator is replaced by a static prompt/scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acceptance_method_bindings import bind_acon
from run_acon_acceptance import RecordedWorker
from run_clariq_acceptance import verify_services
from run_native_arena_acceptance import load_captured_checkpoint, provider_config, setup
from run_static_acceptance import RecordingBackend, digest, local_transport, save

RELEASE = "curate.acon.history_runtime"


def record_native_calls():
    import openai

    original = openai.OpenAI
    calls = []

    class RecordedOpenAI(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            delegate = self.chat.completions.create

            def create(**values):
                visible = {key: value for key, value in values.items() if key in {
                    "model", "messages", "tools", "seed", "max_tokens", "temperature", "extra_body"}
                    and not isinstance(value, openai.NotGiven)}
                request_seed = visible.get("seed")
                row = {"request_hash": digest(visible), "request": visible,
                       "native_role": "user_simulator" if isinstance(request_seed, int) and request_seed >= 1000000 else "agent"}
                calls.append(row)
                try:
                    response = delegate(**values)
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc))
                    raise
                row["response"] = response.model_dump(mode="json", include={"model", "choices", "usage"})
                return response

            self.chat.completions.create = create

    openai.OpenAI = RecordedOpenAI
    return calls


def worker(job, output, checkout, upstream):
    from agentinputlab import InterventionContext
    from agentinputlab.api import _state
    from agentinputlab.contracts.serialization import to_jsonable
    from agentinputlab.integrations.live_history import ACONLiveHistory
    from workers.official_arenas.protocol import EXECUTABLE_PAYLOAD_PROTOCOL, content_hash
    from workers.official_arenas.providers.toolsandbox import ToolSandboxProvider

    setup(checkout)
    calls = record_native_calls()
    model = job["model"]
    provider, config, source, _ = provider_config("toolsandbox", model, model, checkout)
    config.update(scenario_names=[job["checkpoint_id"]], limit=1)
    result = {"job": job, "model_binding": source, "additional_attempts": 1,
              "package_location": __import__("agentinputlab").__file__, "automatic_retry": False}
    runtime, workers, optimizer_calls = None, [], []
    stage, started = "native_source_and_checkpoint_binding", time.monotonic()
    try:
        blockers = provider.doctor(config)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        selected = load_captured_checkpoint(job)
        if selected["protected"]["locked_dependency_hashes"] != config["locked_dependency_hashes"]:
            raise ValueError("native checkpoint dependency identities differ")
        frozen = _state(dict(selected["visible_state"]))
        if job["release"] == RELEASE:
            backend = RecordingBackend(calls=optimizer_calls, model_id=model["id"], base_url=model["base_url"],
                deployment_identity={"revision": digest(source["model_json_hashes"]), "server": "vllm"},
                extra_body=model.get("extra_body", {}), timeout_seconds=180, transport=local_transport)

            def wrap(client):
                worker = RecordedWorker(client, output)
                workers.append(worker)
                return worker

            method, native = bind_acon(RELEASE, upstream=upstream, backend=backend, model=model, client_wrapper=wrap)
            runtime = ACONLiveHistory(method, InterventionContext(backend=backend, external_runtime=native, seed=17))
            provider = ToolSandboxProvider(history_transform=runtime)
        original = frozen.state_hash
        materialized = to_jsonable(frozen)
        payload = {"schema_version": 1, "protocol": EXECUTABLE_PAYLOAD_PROTOCOL,
                   "visible_state_hash": original, "release_id": job["release"],
                   "intervention_execution": {"input_state_hash": original,
                       "output_state_hash": content_hash(materialized), "materialized_state": materialized,
                       "usage": {}}}
        stage = "live_history_native_episode_and_official_scoring"
        response = provider.run_arm(config, replay_record={"checkpoint_id": selected["checkpoint_id"],
            "protected": selected["protected"]}, arm_payload=payload, seed=17,
            role="baseline" if runtime is None else "treatment", work_dir=output)
        result.update(status="scored_terminal", metrics=to_jsonable(response.metrics),
                      official_trace=to_jsonable(response.trace), usage=to_jsonable(response.usage))
    except Exception as exc:
        import traceback
        result.update(status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result.update(seconds=time.monotonic() - started, native_calls=calls,
                  optimizer_frontend_calls=optimizer_calls,
                  worker_responses=[response for worker in workers for response in worker.responses],
                  live_history_records=runtime.records if runtime else [])
    save(output / "terminal.json", result)


def implementation_retest_jobs(previous, bindings):
    manifest = json.loads((previous / "manifest.json").read_text())
    if manifest["model_bindings"] != bindings:
        raise ValueError("implementation retest model/source identities differ")
    if not (previous / "complete.json").exists():
        raise ValueError("original implementation batch is not terminal")
    jobs, sources, untouched = [], {}, []
    for job in manifest["jobs"]:
        path = previous / job["id"] / "terminal.json"
        terminal = json.loads(path.read_text())
        reason = None
        if (terminal["status"] == "failed_terminal" and terminal.get("error") ==
                "live ACON requires the native session to start with system then user"
                and terminal["job"]["release"] == RELEASE
                and terminal.get("live_history_records")
                and all(row["status"] == "native_threshold_skip" for row in terminal["live_history_records"])):
            reason = "verified_callback_return_aliasing_defect"
        elif (terminal["status"] == "failed_terminal" and terminal.get("error_type") == "TypeError"
              and terminal.get("error") == "Object of type NotGiven is not JSON serializable"
              and "digest(visible)" in terminal.get("traceback", "")):
            reason = "verified_instrumentation_sentinel_serialization_defect"
        if reason is None:
            untouched.append(terminal)
            continue
        original_id = job["id"]
        job = {key: value for key, value in job.items() if key != "id"}
        job.update(prior_job_id=original_id, prior_terminal_path=str(path),
                   prior_terminal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   implementation_retest_reason=reason, attempt_ordinal=2)
        job["id"] = digest(job)
        jobs.append(job)
        sources[original_id] = {"path": str(path), "sha256": job["prior_terminal_sha256"], "reason": reason}
    if not jobs:
        raise ValueError("no verified implementation defect positions to retest")
    return jobs, sources, untouched


def run(config, root, checkout, upstream, previous=None):
    setup(checkout)
    bindings = verify_services(config["models"])
    root.mkdir(parents=True, exist_ok=False)
    if previous is not None:
        jobs, sources, untouched = implementation_retest_jobs(previous, bindings)
        return dispatch(config, root, checkout, upstream, jobs, bindings, sources, untouched)
    first = config["models"][0]
    provider, native, _, _ = provider_config("toolsandbox", first, first, checkout)
    native["limit"] = 10000
    blockers = provider.doctor(native)
    if blockers:
        raise RuntimeError("; ".join(blockers))
    scenarios = provider._scenarios(native)
    names = sorted(name for name, scenario in scenarios.items()
                   if "insufficient" in name.lower() or any("insufficient" in str(c).lower() for c in scenario.categories))[:2]
    if len(names) != 2:
        raise ValueError("two source-declared insufficient-information scenarios are required")
    jobs = []
    for model in config["models"]:
        provider, native, _, _ = provider_config("toolsandbox", model, model, checkout)
        native.update(scenario_names=names, limit=2)
        rows = provider.capture(native, output_root=root)
        capture_dir = root / "captures" / model["id"]
        capture_dir.mkdir(parents=True)
        for row in rows:
            path = capture_dir / (digest(row["checkpoint_id"]) + ".json")
            save(path, row)
            for release in ("baseline.no_op", RELEASE):
                job = {"model": model, "release": release, "benchmark": provider.benchmark_id,
                       "checkpoint_id": row["checkpoint_id"], "checkpoint_file": str(path),
                       "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": 17,
                       "budget": {"max_model_calls": 8, "max_output_tokens": 4096}, "per_call_tokens": 512,
                       "history_profile": "smolagents.history.v1" if release == RELEASE else None,
                       "integration": "native_user_turn_boundaries_shared_agent_optimizer_budget"}
                job["id"] = digest(job)
                jobs.append(job)
    return dispatch(config, root, checkout, upstream, jobs, bindings, {}, [])


def dispatch(config, root, checkout, upstream, jobs, bindings, sources, untouched):
    save(root / "manifest.json", {"jobs": jobs, "model_bindings": bindings,
         "selection": "two lexicographically first source-declared insufficient-information scenarios",
         "capture_model_calls": 0, "native_user_and_scorer_not_replaced": True,
         "automatic_retry": False, "paper_results_changed": False,
         "implementation_retest_sources": sources,
         "consumed_prior_attempts_retained_and_not_free": bool(sources)})

    def one(job):
        output = root / job["id"]
        output.mkdir()
        save(output / "job.json", job)
        command = [sys.executable, __file__, "--worker", "--job", str(output / "job.json"),
                   "--output", str(output), "--checkout", str(checkout), "--upstream-root", str(upstream)]
        with (output / "process.log").open("x") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1500)
        if completed.returncode or not (output / "terminal.json").exists():
            raise RuntimeError("native history worker ended without its terminal; inspect, do not retry")
        (root.parent / "keep-services-alive").touch()

    def per_model(model):
        selected = [job for job in jobs if job["model"]["id"] == model["id"]]
        # Independent model lanes, fresh process/environment per episode.
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(one, selected))
    with ThreadPoolExecutor(len(config["models"])) as pool:
        list(pool.map(per_model, config["models"]))
    terminals = [json.loads(path.read_text()) for path in root.glob("*/terminal.json")]
    combined = [*untouched, *terminals]
    save(root / "complete.json", {"expected": len(jobs), "terminal": len(terminals),
         "terminal_statuses": dict(Counter(row["status"] for row in terminals)),
         "live_history_decisions": dict(Counter(record["status"] for row in terminals for record in row["live_history_records"])),
         "covered_positions_with_prior_unchanged": len(combined),
         "combined_statuses": dict(Counter(row["status"] for row in combined)),
         "full_toolkit_acceptance_complete": False, "paper_results_changed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementation-retest-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.job.read_text()), args.output, args.checkout, args.upstream_root)
    else:
        run(json.loads(args.config.read_text()), args.output, args.checkout, args.upstream_root,
            args.implementation_retest_root)


if __name__ == "__main__":
    main()
