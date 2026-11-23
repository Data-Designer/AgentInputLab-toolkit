"""Glue over existing executors, with an append-only attempt journal.

There is no new scheduler. Resume skips completed failures and uncertain started
attempts alike; only never-started cases are passed to the existing executor.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

from agentinputlab import __version__
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .configuration import WorkflowConfig, read_json
from .planning import inspect_workflow, load_adapter, safe_error

_MAX_EVENT = 64 * 1024 * 1024


def _events(root):
    started, terminal = {}, {}
    try:
        with (root / "events.jsonl").open("rb") as stream:
            for _ in range(4001):
                raw = stream.readline(_MAX_EVENT + 1)
                if not raw:
                    break
                if len(raw) > _MAX_EVENT or not raw.endswith(b"\n"):
                    raise ValidationError("workflow journal is truncated; reconcile before resuming")
                event = json.loads(raw)
                if set(event) != {"record", "sha256"} or content_hash(event["record"]) != event["sha256"]:
                    raise ValidationError("workflow journal integrity check failed")
                row = event["record"]
                case = row.get("workflow_case_id")
                if not isinstance(case, str) or not case:
                    raise ValidationError("workflow journal case identity is invalid")
                if row.get("status") == "started":
                    if case in started:
                        raise ValidationError("workflow journal records a repeated attempt")
                    started[case] = row
                else:
                    if case not in started or case in terminal:
                        raise ValidationError("workflow terminal contradicts its started record")
                    if row.get("status") not in {"succeeded", "scored", "failed", "awaiting_clarification"}:
                        raise ValidationError("unknown workflow terminal status")
                    terminal[case] = row
            else:
                raise ValidationError("workflow journal has too many events")
    except ValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        raise ValidationError("cannot safely read workflow journal; no attempts will be repeated") from None
    return started, terminal


def _summary(root, expected):
    started, terminal = _events(root)
    if set(started) - set(expected):
        raise ValidationError("journal contains an unselected case")
    uncertain = sorted(set(started) - set(terminal))
    pending = [case for case in expected if case not in started]
    counts = {name: sum(row["status"] == name for row in terminal.values())
              for name in ("succeeded", "scored", "failed", "awaiting_clarification")}
    complete = len(terminal) == len(expected)
    status = ("needs_reconciliation" if uncertain else "pending" if pending else
              "awaiting_user" if counts["awaiting_clarification"] else
              "completed_with_failures" if counts["failed"] else "completed")
    return {"status": status, "selected_count": len(expected), "terminal_count": len(terminal),
            "outcomes": counts, "uncertain_case_ids": uncertain, "unstarted_case_ids": pending,
            "all_attempts_terminal": complete,
            "all_succeeded": complete and not counts["failed"] and not counts["awaiting_clarification"],
            "automatic_retry": False, "uncertain_attempts_requeued": False,
            "journal_path": str(root / "events.jsonl")}


def workflow_status(output_dir):
    root = Path(output_dir).absolute()
    if root.is_symlink() or (root / "events.jsonl").is_symlink():
        raise ValidationError("workflow evidence must not be symlinked")
    intent = read_json(root / "intent.json")
    return {"workflow_id": intent["workflow_id"], **_summary(root, intent["case_ids"])}


def _selection(workflow, adapter):
    target = dict(workflow.target)
    requested = target.get("case_ids")
    if requested is None:
        examples = tuple(itertools.islice(adapter.examples(target["split"]), target["limit"]))
    else:
        wanted = set(requested)
        selected = [x for x in adapter.examples(target["split"]) if x.example_id in wanted]
        by_id = {x.example_id: x for x in selected}
        if set(by_id) != wanted or len(by_id) != len(selected):
            raise ValidationError("selected native cases are missing or duplicated")
        examples = tuple(by_id[x] for x in requested)
    if not examples or len({x.example_id for x in examples}) != len(examples):
        raise ValidationError("native case selection is empty or not unique")
    observed = {x.example_id: x.visible_state().state_hash for x in examples}
    if workflow.case_state_hashes and dict(workflow.case_state_hashes) != observed:
        raise ValidationError("visible native states changed since preparation")
    target.pop("limit", None)
    target["case_ids"] = [x.example_id for x in examples]
    return replace(workflow, target=target, case_state_hashes=observed)


def run_workflow(workflow, *, output_dir=None, execute=False, resume=False, context_factory=None):
    """Execute explicitly, using existing run/evaluate/native-arena paths.

    This initial glue is sequential. Existing campaign/Studio concurrency stays
    unchanged. A context factory is a trusted Python-only testing/integration seam.
    """
    if type(execute) is not bool or type(resume) is not bool:
        raise ValidationError("execute and resume must be booleans")
    requested_id = workflow.identity
    from .artifacts import discover_artifacts
    # Inspection may report a catalog mismatch without creating any output.
    # Resolve the same binding before actual execution, not just in its report.
    try:
        workflow, _ = discover_artifacts(workflow)
    except Exception:
        return {"executed": False, "execution_blocked": True, "inspection": inspect_workflow(workflow)}
    adapters = {}
    report = inspect_workflow(workflow, probe_endpoint=execute and not resume, _adapters=adapters)
    if not execute:
        return {"executed": False, "inspection": report}
    if not output_dir:
        raise ValidationError("workflow execution requires output_dir")
    if any(x["check"] == "bound_inputs" and x["status"] == "blocked" for x in report["checks"]):
        return {"executed": False, "execution_blocked": True, "inspection": report}
    if workflow.max_workers != 1:
        raise ValidationError("this wrapper uses max_workers=1; existing campaign execution provides concurrency")
    if context_factory is not None and not callable(context_factory):
        raise ValidationError("context_factory must supply a fresh context")
    original_id = requested_id
    workflow = workflow.bind_inputs()
    workflow.verify_bindings()
    root = Path(output_dir).absolute()
    if root.is_symlink() or (root / "events.jsonl").is_symlink():
        raise ValidationError("workflow evidence must not be symlinked")
    from agentinputlab.arena_configuration import _implementation_digest
    implementation = _implementation_digest()
    if root.exists():
        if not resume:
            raise ValidationError("workflow output exists; use --resume for the exact same workflow")
        intent = read_json(root / "intent.json")
        if intent.get("requested_workflow_id") != original_id or intent.get("implementation") != implementation:
            raise ValidationError("workflow or installed implementation changed; cannot resume")
        saved = WorkflowConfig.from_file(root / "workflow.json")
        saved.verify_bindings()
        if workflow.run.to_dict() != saved.run.to_dict() or workflow.input_bindings != saved.input_bindings:
            raise ValidationError("workflow runtime or bound files changed")
        workflow = saved
        if not _summary(root, intent["case_ids"])["unstarted_case_ids"]:
            return {"executed": False, "reused_existing_attempts": True, **workflow_status(root)}
    elif resume:
        raise ValidationError("no saved workflow to resume")
    # Completed journals can be read after services were released. Only new
    # dispatches need a live endpoint and the full local preflight again.
    if resume:
        from .planning import _probe_endpoint
        if workflow.run.backend:
            try:
                probe = _probe_endpoint(workflow)
            except Exception as error:
                probe = {"status": "blocked", "error": safe_error(error, workflow)}
            report["checks"].append({"check": "endpoint_models", **probe})
            report["endpoint_probed"] = True
            report["can_attempt"] = report["can_attempt"] and probe["status"] == "ready"
    if not report["can_attempt"]:
        return {"executed": False, "execution_blocked": True, "inspection": report}
    kind = workflow.target["kind"]
    adapter = None
    if kind == "benchmark":
        adapter = adapters.get("adapter") or load_adapter(workflow, allow_network=False)
        workflow = _selection(workflow, adapter)
    elif kind == "arena" and "case_ids" not in workflow.target:
        raise ValidationError("freeze arena case_ids first with config evaluate-arena --list-cases")
    case_ids = ([f"query-{i:06d}" for i in range(len(workflow.target["queries"]))]
                if kind == "query" else list(workflow.target["case_ids"]))
    if not root.exists():
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        for name, value in (("workflow.json", workflow.to_dict()), ("intent.json", {
                "workflow_id": workflow.identity, "requested_workflow_id": original_id,
                "implementation": implementation, "version": __version__, "case_ids": case_ids})):
            with (root / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
        (root / "events.jsonl").touch(mode=0o600, exist_ok=False)
    if case_ids != read_json(root / "intent.json")["case_ids"]:
        raise ValidationError("workflow case selection changed")
    try:
        guard = (root / ".running").open("x", encoding="utf-8")
    except FileExistsError:
        raise ValidationError("run is active or interrupted; reconcile its .running guard before resuming") from None
    try:
        guard.write(str(os.getpid()))
        guard.flush()
        os.fsync(guard.fileno())
        started, _ = _events(root)
        remaining = [case for case in case_ids if case not in started]
        with (root / "events.jsonl").open("a", encoding="utf-8") as journal:
            dispatched_first = False
            def persist(row):
                nonlocal dispatched_first
                row = to_jsonable(row)
                case = row.get("example_id") if kind == "benchmark" else row.get("workflow_case_id")
                if case not in remaining:
                    raise ValidationError("executor returned an unselected case")
                if kind == "benchmark" and row.get("status") == "started" and dispatched_first and case == remaining[0]:
                    # The first dispatch marker was persisted before entering
                    # native preflight/context construction. Consume exactly
                    # one corresponding native callback, not another attempt.
                    dispatched_first = False
                    return
                value = {**row, "workflow_case_id": case}
                raw = json.dumps({"record": value, "sha256": content_hash(value)},
                                 ensure_ascii=False, allow_nan=False) + "\n"
                if len(raw.encode("utf-8")) > _MAX_EVENT:
                    raise ValidationError("event exceeds 64 MiB; preserve the uncertain attempt")
                journal.write(raw)
                journal.flush()
                os.fsync(journal.fileno())

            if kind == "benchmark" and remaining:
                persist({"example_id": remaining[0], "status": "started", "stage": "native_dispatch"})
                dispatched_first = True
                context = context_factory() if context_factory else None
                # The file-configured BEIR retriever owns its archive and native
                # scorer. Reuse the inspected case IDs, not a second adapter.
                owns_archive = workflow.run.retriever.get("kind") == "beir_archive"
                workflow.run.evaluate(workflow.target["benchmark_id"], split=workflow.target["split"],
                    example_ids=remaining, adapter=None if owns_archive else adapter,
                    context=context, on_record=persist,
                    predictor_parameters=workflow.target.get("predictor_parameters"),
                    qrag_reader=workflow.target.get("qrag_reader"), **({"offline": True} if owns_archive else {}))
            elif kind != "benchmark":
                for case in remaining:
                    workflow.verify_bindings()
                    persist({"workflow_case_id": case, "status": "started"})
                    try:
                        if kind == "query":
                            context = context_factory() if context_factory else None
                            result = {"status": "succeeded", "scored": False,
                                      "output": workflow.run.run(workflow.target["queries"][case_ids.index(case)],
                                                                 context=context)}
                        else:
                            if context_factory:
                                raise ValidationError("arena execution owns its process context")
                            batch = workflow.run.evaluate_arena(workflow.target["arena_config"],
                                case_ids=[case], output_dir=root / "native" / f"case-{case_ids.index(case):06d}")
                            result = batch["cases"][0]
                    except Exception as error:
                        receipt = getattr(error, "resource_receipt", None)
                        result = {"status": "failed", "scored": False, "failure": safe_error(error, workflow),
                                  "resource_receipt": to_jsonable(receipt) if receipt is not None else None,
                                  "usage_policy": "absent usage is unknown, not zero"}
                    persist({**result, "workflow_case_id": case})
    finally:
        guard.close()
        (root / ".running").unlink()
    return {"executed": True, "inspection": report, **workflow_status(root)}
