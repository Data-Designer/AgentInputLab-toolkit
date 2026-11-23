"""Local, opt-in web bridge to the existing workflow executor.

The browser edits the same configuration used by the CLI. No algorithm,
preparation, scoring or retry policy is implemented by this bridge.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from agentinputlab import ExecutionBudget, RunConfig
from agentinputlab.api import describe_intervention
from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.experiments.secrets import require_secret_free
from agentinputlab.workflows import WorkflowConfig, inspect_workflow, launch_workflow
from agentinputlab.workflows.configuration import read_json, workflow_template
from agentinputlab.workflows.planning import safe_error
from agentinputlab.workflows.recipes import list_recipes

PRESETS = (
    {"id": "prompt", "label": "Prompt-only starter · no model calls", "benchmark_id": None},
    {"id": "bfcl", "label": "Tool calls · BFCL Simple Python", "benchmark_id": "bfcl.v3.simple_python"},
    {"id": "qa", "label": "Question answering · HotpotQA", "benchmark_id": "hotpotqa.distractor"},
    {"id": "tooltalk", "label": "Tool agent · ToolTalk", "benchmark_id": "tooltalk.full"},
    {"id": "state", "label": "Tool agent · STATE", "benchmark_id": "state_bench.common_support"},
    {"id": "sandbox", "label": "Tool agent · ToolSandbox", "benchmark_id": "toolsandbox.common_support"},
)
JOB_ID = re.compile(r"^[0-9a-f]{32}$")


def workflow_catalog():
    return {
        "presets": [dict(p, release_id="plan.plan_and_solve.ps_prompt") for p in PRESETS],
        "recipes": list_recipes(),
        "benchmarks": [{"id": b.cell_id, "label": f"{b.suite}: {b.task}"}
                       for b in BenchmarkRegistry.default().list()],
    }


def generate_workflow(payload):
    """Generate a draft, including native defaults and explicit missing inputs."""
    if not isinstance(payload, dict):
        raise ValidationError("template settings must be an object")
    require_secret_free(payload, label="workflow settings")
    allowed = {"run", "release_id", "benchmark_id", "query", "limit", "case_ids", "backend",
               "parameters", "budget", "retriever", "artifacts", "runtime", "preparation",
               "environment_backend", "seed", "max_workers", "cache_dir"}
    if set(payload) - allowed:
        raise ValidationError("unknown template fields: " + ", ".join(sorted(set(payload) - allowed)))
    options = {k: v for k, v in payload.items() if k not in {"seed", "max_workers", "budget"}}
    if "run" in options:
        if "release_id" in options:
            raise ValidationError("choose run or release_id")
        options["run"] = RunConfig.from_dict(options["run"])
    if "budget" in payload:
        try:
            options["budget"] = ExecutionBudget(**payload["budget"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("budget must contain valid ExecutionBudget fields") from exc
    workflow = workflow_template(**options)
    workflow = replace(workflow, run=replace(workflow.run, seed=payload.get("seed", workflow.run.seed)),
                       max_workers=payload.get("max_workers", 1))
    return {"workflow": workflow.to_dict(), "run": workflow.run.to_dict(),
            "parameters": describe_intervention(workflow.run.release_id)["parameters"]}


class WorkflowBusyError(ValidationError):
    pass


class WorkflowManager:
    def __init__(self, root):
        root = Path(root).expanduser().absolute()
        if root.is_symlink():
            raise ValidationError("workflow root must not be a symlink")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = root.resolve()
        self._lock = threading.Lock()
        self._active = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-workflow")

    def config(self, value):
        require_secret_free(value, label="workflow")
        return WorkflowConfig.from_dict(value, base_dir=self.root)

    def inspect(self, value, *, probe_endpoint=False):
        if type(probe_endpoint) is not bool:
            raise ValidationError("probe_endpoint must be boolean")
        return inspect_workflow(self.config(value), probe_endpoint=probe_endpoint)

    def _directory(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValidationError("invalid workflow run ID")
        path = self.root / job_id
        if not path.is_dir() or path.is_symlink():
            raise ValidationError("workflow run does not exist")
        for name in ("workflow.json", "status.json", "launch", "launch/run", ".studio-running"):
            if (path / name).is_symlink():
                raise ValidationError("workflow run files must not be symlinked")
        return path

    @staticmethod
    def _save(path, value):
        require_secret_free(value, label="workflow record")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def submit(self, value=None, *, allow_network=False, resume_id=None):
        if type(allow_network) is not bool:
            raise ValidationError("allow_network must be boolean")
        if resume_id is not None and value is not None:
            raise ValidationError("resume uses the saved configuration; it cannot be edited")
        workflow = self.config(value) if resume_id is None else None
        with self._lock:
            if self._active:
                raise WorkflowBusyError("A workflow is active. Wait for it to finish before starting another.")
            if resume_id is None:
                job_id = uuid.uuid4().hex
                directory = self.root / job_id
                directory.mkdir(mode=0o700)
                self._save(directory / "workflow.json", workflow.to_dict())
                record = {"id": job_id, "workflow_id": workflow.identity,
                          "created_at": datetime.now(timezone.utc).isoformat(),
                          "release_id": workflow.run.release_id, "target": workflow.to_dict()["target"]}
            else:
                job_id = resume_id
                directory = self._directory(job_id)
                record = read_json(directory / "status.json")
                workflow = WorkflowConfig.from_file(directory / "workflow.json")
                if workflow.identity != record["workflow_id"]:
                    raise ValidationError("saved workflow identity changed")
            try:
                with (directory / ".studio-running").open("x") as guard:
                    guard.write(str(os.getpid()))
            except FileExistsError:
                raise WorkflowBusyError("Run is active or interrupted. Reconcile its .studio-running guard before resuming.") from None
            record.update(status="queued", error=None, allow_network=allow_network,
                          output_dir=str(directory / "launch"))
            try:
                self._save(directory / "status.json", record)
                self._active.add(job_id)
                self._executor.submit(self._execute, job_id, workflow, record, allow_network)
            except BaseException:
                self._active.discard(job_id)
                (directory / ".studio-running").unlink()
                raise
            return copy.deepcopy(record)

    def _execute(self, job_id, workflow, record, allow_network):
        directory = self.root / job_id
        try:
            with self._lock:
                self._directory(job_id)
                record["status"] = "running"
                self._save(directory / "status.json", record)
            output = directory / "launch"
            result = launch_workflow(workflow, output_dir=output, execute=True,
                                     allow_network=allow_network, resume=output.exists())
            record["result"] = result
            record["status"] = ("blocked" if result.get("execution_blocked") else result.get("status", "finished"))
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = safe_error(exc, workflow)
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            with self._lock:
                try:
                    self._directory(job_id)
                    self._save(directory / "status.json", record)
                finally:
                    if not directory.is_symlink():
                        (directory / ".studio-running").unlink(missing_ok=True)
                    self._active.discard(job_id)

    def get(self, job_id):
        with self._lock:
            directory = self._directory(job_id)
            record = read_json(directory / "status.json")
            record["workflow"] = read_json(directory / "workflow.json")
            active = job_id in self._active
        if not active and record["status"] in {"queued", "running"}:
            record["status"] = "needs_reconciliation"
        record["active"] = active
        record["preview"] = self._preview(directory / "launch" / "run" / "events.jsonl")
        return record

    @staticmethod
    def _preview(path):
        """Bounded output preview; complete native records stay in the journal."""
        rows, consumed = [], 0
        if not path.exists() or path.is_symlink():
            return {"records": rows, "truncated": False}
        with path.open("rb") as stream:
            while len(rows) < 20 and consumed < 8 * 1024 * 1024:
                raw = stream.readline(1024 * 1024 + 1)
                if not raw:
                    return {"records": rows, "truncated": False}
                consumed += len(raw)
                if len(raw) > 1024 * 1024 or not raw.endswith(b"\n"):
                    break
                event = json.loads(raw)
                if content_hash(event["record"]) != event["sha256"]:
                    raise ValidationError("workflow preview journal hash mismatch")
                row = event["record"]
                if row.get("status") == "started":
                    continue
                require_secret_free(row, label="workflow preview")
                preview = {k: row[k] for k in ("workflow_case_id", "status", "scored", "evaluation",
                           "usage", "failure", "output", "metrics") if k in row}
                encoded = json.dumps(preview, ensure_ascii=False)
                rows.append(preview if len(encoded) <= 24000 else {
                    "workflow_case_id": row.get("workflow_case_id"), "status": row.get("status"),
                    "preview_note": "Large result: inspect the complete native journal at the displayed output path."})
        return {"records": rows, "truncated": True}

    def list_runs(self):
        directories = sorted((p for p in self.root.iterdir() if JOB_ID.fullmatch(p.name)
                              and p.is_dir() and not p.is_symlink()),
                             key=lambda p: p.stat().st_mtime, reverse=True)[:100]
        rows = []
        for directory in directories:
            try:
                record = read_json(self._directory(directory.name) / "status.json")
                rows.append({k: record.get(k) for k in ("id", "status", "created_at", "release_id", "target")})
            except ValidationError:
                continue
        return rows

    def shutdown(self):
        self._executor.shutdown(wait=True)
