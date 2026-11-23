"""Resume only source-path startup faults proven to precede backend construction.

Original terminal records are preserved. This command never selects model
output failures, scientific timeouts or requests with unknown consumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acceptance_method_bindings import ASK_OR_ASSUME
from run_static_acceptance import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prior-root", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--tooltalk-checkout", type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    a.output.mkdir(parents=True)
    jobs = []
    for prior in a.prior_root:
        for path in prior.glob("*/terminal.json"):
            row = json.loads(path.read_text())
            job = row["job"]
            if job["release"] != ASK_OR_ASSUME:
                continue
            if not (
                row["status"] == "failed_terminal"
                and row.get("stage") == "intervention"
                and row.get("intervention") is None
                and row.get("optimizer_calls") == []
                and row.get("error")
                == "external worker 'toolkit-source-mounted-ask-or-assume' exited with code 2"
            ):
                raise RuntimeError("refuse any non-entry or consumed failure")
            # The old fixed command pointed to a nonexistent checkout. The
            # verified driver's main() checks it before creating a backend or
            # reading the first JSONL request. This is stronger than empty
            # parent-process call logs, which cannot audit external inference.
            if (a.upstream_root / "upstream-paper/ask-or-assume").exists():
                raise RuntimeError("old missing-path proof no longer holds")
            jobs.append(
                {
                    "job": job,
                    "prior_terminal": str(path),
                    "prior_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if not jobs:
        raise RuntimeError("no proven source-path entry failures")
    save(
        a.output / "manifest.json",
        {
            "jobs": jobs,
            "automatic_retry": False,
            "scope": "exact_zero_model_call_entry_repair",
            "proof": "old nonexistent checkout rejected before backend construction by pinned Ask-or-Assume main",
            "paper_results_changed": False,
        },
    )

    def one(item):
        job = item["job"]
        d = a.output / job["id"]
        d.mkdir()
        save(d / "job.json", job)
        arena = job.get("arena", "tooltalk")
        script = (
            "run_tooltalk_acceptance.py"
            if arena == "tooltalk"
            else "run_native_arena_acceptance.py"
        )
        checkout = (
            a.tooltalk_checkout
            if arena == "tooltalk"
            else a.upstream_root
            / "upstream-paper"
            / ("STATE-Bench" if arena == "statebench" else "ToolSandbox")
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name(script)),
            "--config",
            str(a.config),
            "--output",
            str(d),
            "--checkout",
            str(checkout),
            "--upstream-root",
            str(a.upstream_root),
            "--job",
            str(d / "job.json"),
        ]
        if arena != "tooltalk":
            command += ["--arena", arena]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        environment.pop("PYTHONPATH", None)
        for name, value in config.get("runtime_environment", {}).items():
            if name not in {"HF_HOME", "HF_HUB_CACHE"}:
                raise ValueError("unsupported runtime environment setting")
            environment[name] = value
        with (d / "worker.log").open("x") as log:
            try:
                code = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800
                ).returncode
            except subprocess.TimeoutExpired:
                code = "own_worker_timeout_consumption_unknown"
        if not (d / "terminal.json").exists():
            save(
                d / "terminal.json",
                {
                    "job": job,
                    "status": "entry_failure_terminal",
                    "returncode": code,
                    "model_calls": None,
                    "automatic_retry": False,
                },
            )
        (a.output.parent / "keep-services-alive").touch()

    def lane(model):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(one, [x for x in jobs if x["job"]["model"]["id"] == model]))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lane, sorted({x["job"]["model"]["id"] for x in jobs})))
    rows = [json.loads(x.read_text()) for x in a.output.glob("*/terminal.json")]
    save(
        a.output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(x["status"] == "scored_terminal" for x in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


if __name__ == "__main__":
    main()
