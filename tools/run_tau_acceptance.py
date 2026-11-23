"""Installed native tau cases with one retained capture per backbone and case.

All state, traces, and results must be stored outside the public checkout.
The provider's user/judge model policy is retained; environment cost is separate
from each evaluated agent's token/call budget. Consumed attempts are not retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from run_native_arena_acceptance import provider_config, setup, worker
from run_static_acceptance import digest, save
from run_tooltalk_acceptance import RELEASES


def record_native_calls():
    from tau2.utils import llm_utils

    calls = []
    original = llm_utils.completion

    def completion(*args, **kwargs):
        visible = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "model",
                "messages",
                "tools",
                "max_tokens",
                "seed",
                "temperature",
                "response_format",
                "api_base",
                "num_retries",
            }
        }
        row = {
            "request_hash": digest(visible),
            "model": kwargs.get("model"),
            "max_tokens": kwargs.get("max_tokens"),
            "seed": kwargs.get("seed"),
            "started": datetime.now(timezone.utc).isoformat(),
        }
        calls.append(row)
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
            raise
        data = response.model_dump(mode="json", include={"model", "choices", "usage"})
        row["response"] = {key: data.get(key) for key in ("model", "choices", "usage")}
        return response

    llm_utils.completion = completion
    return calls


def capture(model, arena, checkout, output):
    setup(checkout)
    calls = record_native_calls()
    provider, config, binding, _ = provider_config(arena, model, model, checkout)
    result = {
        "model": model,
        "benchmark": provider.benchmark_id,
        "model_binding": binding,
        "config": config,
        "automatic_retry": False,
        "package_location": __import__("agentinputlab").__file__,
    }
    try:
        blockers = provider.doctor(config)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        if arena == "retail":
            batch = provider.capture_batch(config, output_root=output)
            rows = batch["records"]
            result["dispositions"] = batch["dispositions"]
        else:
            from tau2.runner.helpers import get_tasks

            tasks = get_tasks("banking_knowledge", task_split_name="test", num_tasks=2)
            rows = []
            result["dispositions"] = []
            for task in tasks:
                # Independent task selection, not another attempt at a failed
                # simulator trajectory. Keep the same frozen agent/environment
                # controls and retain each failed capture in the denominator.
                case_config = {**config, "limit": 1, "capture_task_ids": [str(task.id)]}
                before = len(calls)
                try:
                    rows.extend(provider.capture(case_config, output_root=output))
                    disposition = {"task_id": str(task.id), "status": "captured"}
                except Exception as exc:
                    disposition = {
                        "task_id": str(task.id),
                        "status": "capture_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                result["dispositions"].append(
                    {
                        **disposition,
                        "retry_permitted": False,
                        "capture_config_hash": digest(case_config),
                        "native_call_start": before,
                        "native_call_end": len(calls),
                    }
                )
        result["checkpoints"] = []
        for row in rows:
            path = output / ("checkpoint-" + row["checkpoint_id"] + ".json")
            save(path, row)
            result["checkpoints"].append(
                {
                    "id": row["checkpoint_id"],
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        result["status"] = "capture_terminal"
    except Exception as exc:
        import traceback

        result.update(
            status="capture_failed_terminal",
            error_type=type(exc).__name__,
            error=str(exc),
            diagnostic_traceback=traceback.format_exc(),
        )
    result["native_calls"] = calls
    save(output / "capture-terminal.json", result)


def child(args):
    job = json.loads(args.job.read_text())
    if args.capture:
        capture(job["model"], args.arena, args.checkout, args.output)
        return
    setup(args.checkout)
    calls = record_native_calls()
    try:
        worker(job, args.output, args.checkout, args.upstream_root)
    finally:
        save(args.output / "native-calls.json", {"calls": calls})


def probe_allows_fanout(directory):
    """Real model-output failures are terminal outcomes, not harness gates."""
    row = json.loads((directory / "terminal.json").read_text())
    if row["status"] == "scored_terminal":
        return True
    if row.get("stage") != "native_episode_and_official_scorer":
        return False
    if not str(row.get("error", "")).startswith(
        (
            "AssistantMessage must have either content or tool_calls.",
            "UserMessage must have either content or tool_calls.",
        )
    ):
        return False
    calls_path = directory / "native-calls.json"
    if not calls_path.is_file():
        return False
    return any(
        isinstance(call.get("response", {}).get("choices"), list)
        for call in json.loads(calls_path.read_text())["calls"]
    )


def dispatch_model_shards(models, jobs, one, accept_probe):
    """Each target owns two slots and advances after its own retained probe."""

    def lane(model):
        assigned = [job for job in jobs if job["model"]["id"] == model["id"]]
        baselines = [job for job in assigned if job["release"] == "baseline.no_op"]
        probe = next((job for job in baselines if "checkpoint_file" in job), baselines[0])
        one(probe)
        allowed = accept_probe(probe)
        remaining = [job for job in assigned if job is not probe]
        runnable = [job for job in remaining if allowed or "checkpoint_file" not in job]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(one, runnable))
        return len(remaining) - len(runnable)

    with ThreadPoolExecutor(4) as pool:
        return sum(pool.map(lane, models))


def execute(args):
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            if args.output == parent or parent in args.output.parents:
                raise ValueError("save native acceptance data outside the public checkout")
            break
    if args.output.exists():
        raise RuntimeError("create-only batch exists; inspect saved calls, never relaunch")
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    setup(args.checkout)
    from tau2.runner.helpers import get_tasks

    domain = "retail" if args.arena == "retail" else "banking_knowledge"
    ids = [str(task.id) for task in get_tasks(domain, task_split_name="test", num_tasks=2)]
    assert len(ids) == 2
    releases = config.get("releases", RELEASES)
    if any(release not in RELEASES for release in releases):
        raise ValueError("unknown tau acceptance release")
    save(
        args.output / "selection.json",
        {
            "task_ids": ids,
            "rule": "first two official test tasks",
            "arena": args.arena,
            "releases": releases,
            "automatic_retry": False,
            "environment_policy": "same declared backbone for native user and assertion judge",
            "agent_budget": {"model_calls": 8, "output_tokens": 4096, "per_call": 512},
            "user_output_cap": 1024,
            "retail_judge_output_cap": 2048,
            "full_toolkit_acceptance_complete": False,
        },
    )

    def spawn(job, directory, *, capture_job=False):
        directory.mkdir()
        save(directory / "job.json", job)
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHON_DOTENV_DISABLED="1",
        )
        environment.pop("PYTHONPATH", None)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--output",
            str(directory),
            "--checkout",
            str(args.checkout),
            "--upstream-root",
            str(args.upstream_root),
            "--arena",
            args.arena,
            "--job",
            str(directory / "job.json"),
        ]
        if capture_job:
            command.append("--capture")
        with (directory / "worker.log").open("x") as log:
            try:
                code = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800
                ).returncode
            except subprocess.TimeoutExpired:
                code = "own_worker_timeout_consumption_unknown"
        path = directory / ("capture-terminal.json" if capture_job else "terminal.json")
        if not path.exists():
            save(
                path,
                {
                    "job": job,
                    "status": "entry_failure_terminal",
                    "returncode": code,
                    "model_calls": None,
                    "automatic_retry": False,
                },
            )
        (args.output.parent / "keep-services-alive").touch()

    def capture_one(model):
        directory = args.output / ("capture-" + model["id"])
        spawn({"model": model}, directory, capture_job=True)
        return model["id"], json.loads((directory / "capture-terminal.json").read_text())

    with ThreadPoolExecutor(4) as pool:
        captured = dict(pool.map(capture_one, config["models"]))
    jobs = []
    for model in config["models"]:
        checkpoints = {row["id"]: row for row in captured[model["id"]].get("checkpoints", [])}
        for case in ids:
            for release in releases:
                job = {
                    "model": model,
                    "environment_model": model,
                    "arena": args.arena,
                    "benchmark": "tau3.retail"
                    if args.arena == "retail"
                    else "tau3.banking_knowledge",
                    "checkpoint_id": case,
                    "release": release,
                    "seed": 17,
                    "capture_limit": 2,
                    "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                    "environment_policy": "backbone_conditioned_native_roles",
                }
                if case in checkpoints:
                    job.update(
                        checkpoint_file=checkpoints[case]["path"],
                        checkpoint_sha256=checkpoints[case]["sha256"],
                    )
                job["id"] = digest(job)
                jobs.append(job)
    save(
        args.output / "manifest.json",
        {
            "jobs": jobs,
            "capture_count": len(captured),
            "automatic_retry": False,
            "paper_results_changed": False,
        },
    )

    def one(job):
        directory = args.output / job["id"]
        if "checkpoint_file" not in job:
            directory.mkdir()
            save(
                directory / "terminal.json",
                {
                    "job": job,
                    "status": "capture_dependency_terminal",
                    "evaluated_agent_attempts": 0,
                    "model_calls": 0,
                    "automatic_retry": False,
                },
            )
        else:
            spawn(job, directory)

    # A retained NO_OP probe per model is part of the same manifest. A technical
    # entry failure gates fan-out; real scored failures/zero rewards do not retry.
    unstarted = dispatch_model_shards(
        config["models"], jobs, one, lambda job: probe_allows_fanout(args.output / job["id"])
    )
    rows = [json.loads(p.read_text()) for p in args.output.glob("*/terminal.json")]
    save(
        args.output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "unstarted_technical_gate": unstarted,
            "scored": sum(row["status"] == "scored_terminal" for row in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--arena", choices=("retail", "tau3"), required=True)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    for name in ("config", "output", "checkout", "upstream_root", "job"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.job:
        child(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
