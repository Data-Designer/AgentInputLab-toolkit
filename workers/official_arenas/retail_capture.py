#!/usr/bin/env python3
"""Approved, restart-safe Retail capture and partial-inventory projection.

No real scope or authorization is manufactured here. Before enabling a scope,
the operator must verify actual human scope and common-model choices. Successful
captures become optimizer-visible inventories; capture failures remain in the
full denominator, without invented checkpoints or automatic retries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                   str(Path(__file__).resolve().parents[2])]

from workers.official_arenas.capture_records import _records
from agentinputlab.experiments.builders import CHECKPOINT_INVENTORY_PROTOCOL
from workers.official_arenas.protocol import REPLAY_INVENTORY_PROTOCOL, OfficialArenaError, content_hash, sha256_file
from workers.official_arenas.providers.tau_retail import SOURCE_REVISION, _create_only, _saved_capture, factory
from workers.official_arenas.retail_driver import _read, _validate_authorizations

PROTOCOL = "agentinputlab.retail-capture-scope.v1"
_FIELDS = frozenset({"schema_version", "protocol", "source_revision", "phase", "execution_enabled",
                    "output_root", "provider_config", "provider_config_sha256", "source_sha256",
                    "authorizations", "expected_task_ids", "model_slot"})


def _source_pins() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {relative: sha256_file(root / relative) for relative in (
        "workers/official_arenas/retail_capture.py", "workers/official_arenas/retail_driver.py",
        "workers/official_arenas/driver.py", "workers/official_arenas/protocol.py",
        "workers/official_arenas/providers/tau_retail.py", "workers/official_arenas/providers/tau3.py",
        "workers/official_arenas/capture_records.py",
    )}


def _validated_scope(path: Path, checksum: str, provider):
    value = _read(path)
    if sha256_file(path) != checksum or set(value) != _FIELDS:
        raise OfficialArenaError("Retail capture scope fields or checksum differ")
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["protocol"] != PROTOCOL or value["source_revision"] != SOURCE_REVISION
            or value["execution_enabled"] is not True or value["phase"] not in {"engineering", "formal"}):
        raise OfficialArenaError("Retail capture scope is not enabled or valid")
    _validate_authorizations(value["authorizations"])
    parsed = provider._validate_config(value["provider_config"])
    if value["provider_config_sha256"] != content_hash(parsed) or value["source_sha256"] != _source_pins():
        raise OfficialArenaError("Retail capture source or effective config differs")
    if parsed["task_split_name"] != ("train" if value["phase"] == "engineering" else "test"):
        raise OfficialArenaError("Retail capture phase and official split differ")
    ids = value["expected_task_ids"]
    if (not isinstance(ids, list) or len(ids) != parsed["limit"]
            or any(not isinstance(task_id, str) or not task_id.isdigit() for task_id in ids)
            or len(ids) != len(set(ids))):
        raise OfficialArenaError("Retail capture expected task inventory is invalid")
    if not isinstance(value["model_slot"], str) or not value["model_slot"].strip():
        raise OfficialArenaError("Retail capture model slot is absent")
    root = Path(value["output_root"])
    if (not root.is_absolute() or any(part.is_symlink() for part in (root, *root.parents))
            or not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077):
        raise OfficialArenaError("Retail capture root must be an existing owned private directory")
    return value, parsed, root


def project_batch(provider, config: Mapping[str, Any], batch: Mapping[str, Any], *, root: Path,
                  phase: str, model_slot: str, scope_sha256: str) -> Mapping[str, Any]:
    """Read retained receipts, validate them, and export without generation."""
    binding = content_hash(provider._validate_config(config))
    ids, dispositions, captured = batch["expected_task_ids"], batch["dispositions"], batch["records"]
    if (batch["protocol"] != "retail-capture-batch.v1" or batch["config_binding"] != binding
            or len(ids) != len(set(ids)) or [row["task_id"] for row in dispositions] != ids
            or any(row["status"] not in {"captured", "capture_failed", "reserved_unresolved"}
                   or row["retry_permitted"] is not False for row in dispositions)):
        raise OfficialArenaError("Retail capture dispositions differ from the expected inventory")
    records_by_id = {row["checkpoint_id"]: row for row in captured}
    if len(records_by_id) != len(captured):
        raise OfficialArenaError("Retail capture repeats a checkpoint")
    receipt_files = {}
    for row in dispositions:
        task_id = row["task_id"]
        path = root / f"first-capture-{task_id}.json"
        if row["status"] == "reserved_unresolved":
            if task_id in records_by_id or path.exists() or path.is_symlink():
                raise OfficialArenaError("Retail unresolved capture changed during projection")
            start = root / f"first-capture-{task_id}-start.json"
            _read(start)
            receipt_files[start.name] = sha256_file(start)
            continue
        saved = _saved_capture(path, binding=binding, task_id=task_id)
        if row["status"] != saved["terminal"] or row["receipt_sha256"] != saved["receipt_sha256"]:
            raise OfficialArenaError("Retail capture terminal receipt differs")
        receipt_files[path.name] = sha256_file(path)
        if row["status"] == "captured":
            if records_by_id.get(task_id) != saved["record"]:
                raise OfficialArenaError("Retail projected checkpoint differs from its receipt")
        elif task_id in records_by_id:
            raise OfficialArenaError("Retail failed capture cannot have a fabricated checkpoint")
    successful = [row["task_id"] for row in dispositions if row["status"] == "captured"]
    if set(records_by_id) != set(successful):
        raise OfficialArenaError("Retail captured checkpoint inventory differs")
    terminal = all(row["status"] != "reserved_unresolved" for row in dispositions)
    if batch["all_tasks_terminal"] is not terminal:
        raise OfficialArenaError("Retail batch terminality differs")
    output = {"protocol": "retail-capture-inventory-disposition.v1", "scope_sha256": scope_sha256,
              "phase": phase, "benchmark_id": "tau3.retail", "source_revision": SOURCE_REVISION,
              "model_slot": model_slot, "config_binding": binding, "expected_task_ids": ids,
              "expected_denominator": len(ids), "captured_task_ids": successful,
              "dispositions": dispositions, "all_tasks_terminal": terminal,
              "retry_permitted": False, "receipt_files": receipt_files,
              "failure_endpoints": "capture failures are unresolved environment outcomes, not invented agent zeros"}
    artifacts = {}
    if terminal and captured:
        # Reuse only the existing projection/validation, never its capture or
        # replacing writer. Protected simulator history stays in its own file.
        proxy = SimpleNamespace(benchmark_id="tau3.retail", source_revision=SOURCE_REVISION,
                                doctor=lambda _config: (), capture=lambda _config, **kwargs: captured)
        visible, protected, units = _records(proxy, config, split=config["task_split_name"], output_root=root)
        artifacts = {
            "checkpoints.json": {"schema_version": 1, "protocol": CHECKPOINT_INVENTORY_PROTOCOL,
                                 "checkpoint_count": len(visible), "checkpoints": visible},
            "replays.json": {"schema_version": 1, "protocol": REPLAY_INVENTORY_PROTOCOL,
                             "benchmark_id": "tau3.retail", "source_revision": SOURCE_REVISION,
                             "record_count": len(protected), "records": protected},
            "units.json": units,
        }
    output["inventory_content_sha256"] = {name: content_hash(value) for name, value in artifacts.items()}
    snapshot = root / ("inventory-" + content_hash(output))
    try:
        snapshot.mkdir(mode=0o700)
    except FileExistsError:
        if snapshot.is_symlink() or _read(snapshot / "dispositions.json") != output:
            raise OfficialArenaError("Retail inventory snapshot changed or is interrupted")
        for name, value in artifacts.items():
            path = snapshot / name
            if path.is_symlink() or json.loads(path.read_bytes()) != value:
                raise OfficialArenaError("Retail retained inventory artifact differs")
        return {"snapshot": str(snapshot), "disposition": output}
    for name, value in artifacts.items():
        _create_only(snapshot / name, value)
    _create_only(snapshot / "dispositions.json", output)
    return {"snapshot": str(snapshot), "disposition": output}


def run(path: Path, checksum: str) -> Mapping[str, Any]:
    provider = factory()
    scope, config, root = _validated_scope(path, checksum, provider)
    blockers = provider.doctor(config)
    if blockers:
        raise OfficialArenaError("Retail native capture preflight did not pass")
    from tau2.runner.helpers import get_tasks
    ids = [str(task.id) for task in get_tasks("retail", task_split_name=config["task_split_name"], num_tasks=config["limit"])]
    if ids != scope["expected_task_ids"]:
        raise OfficialArenaError("Retail scoped official task order differs")
    batch = provider.capture_batch(config, output_root=root)
    return project_batch(provider, config, batch, root=root, phase=scope["phase"],
                         model_slot=scope["model_slot"], scope_sha256=checksum)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bound first-attempt Retail capture batch")
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    args = parser.parse_args()
    try:
        result = run(args.scope, args.scope_sha256)
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": "Retail capture rejected; inspect owned receipts"}), file=sys.stderr)
        return 2
    # No protected user history or visible request contents in the console.
    disposition = result["disposition"]
    print(json.dumps({"snapshot": result["snapshot"], "expected": disposition["expected_denominator"],
                      "captured": len(disposition["captured_task_ids"]),
                      "all_tasks_terminal": disposition["all_tasks_terminal"],
                      "formal_inference_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
