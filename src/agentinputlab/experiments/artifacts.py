"""Content-bound, resumable experiment artifact storage.

The store writes only after an operator has explicitly selected execution.  A
resume request is accepted only when the frozen manifest, plan, and driver
fingerprint match the existing run exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import (
    canonical_json,
    freeze_mapping,
    to_jsonable,
)

from .plan import ExperimentCell, ExperimentPlan, validate_experiment_plan
from .schema import ExperimentManifest, require_sha256


@dataclass(frozen=True, slots=True)
class ExperimentArtifactIdentity:
    study_id: str
    manifest_hash: str
    plan_hash: str
    driver_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValidationError("experiment artifact study_id must be non-empty")
        for label in ("manifest_hash", "plan_hash", "driver_hash"):
            require_sha256(getattr(self, label), label=label)

    @property
    def identity_hash(self) -> str:
        from agentinputlab.contracts.serialization import content_hash

        return content_hash(self)


class ExperimentArtifactStore:
    """Append-only cell results under one immutable run identity."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: ExperimentManifest,
        plan: ExperimentPlan,
        driver_fingerprint: Mapping[str, Any],
    ) -> None:
        from agentinputlab.contracts.serialization import content_hash

        if plan.manifest_hash != manifest.manifest_hash:
            raise ValidationError("experiment plan does not belong to manifest")
        validate_experiment_plan(manifest, plan)
        try:
            fingerprint = freeze_mapping(driver_fingerprint)
            driver_hash = content_hash(fingerprint)
        except (TypeError, ValueError) as exc:
            raise ValidationError("driver fingerprint must be canonical JSON") from exc
        self.manifest = manifest
        self.plan = plan
        self.driver_fingerprint = fingerprint
        self.identity = ExperimentArtifactIdentity(
            study_id=manifest.study_id,
            manifest_hash=manifest.manifest_hash,
            plan_hash=plan.plan_hash,
            driver_hash=driver_hash,
        )
        configured_root = Path(root).expanduser()
        if not configured_root.is_absolute():
            configured_root = Path(os.path.abspath(configured_root))
        if configured_root.is_symlink():
            raise ArtifactError("experiment output root must not be a symbolic link")
        self.root = configured_root.resolve()
        self.run_dir = (
            self.root / manifest.study_id / manifest.manifest_hash / plan.plan_hash / driver_hash
        )
        self.results_dir = self.run_dir / "results"
        self._lock = RLock()

    def initialize(self, *, resume: bool) -> Path:
        """Create a new run or verify the exact identity of an existing one."""

        self._assert_run_path_confined()
        identity_path = self.run_dir / "run_identity.json"
        if self.run_dir.exists():
            if not resume:
                raise ArtifactError(
                    "experiment run already exists; use --resume only for the same frozen run"
                )
            self._verify_control_file(
                identity_path,
                self._identity_payload(),
                label="run identity",
            )
            self._verify_control_file(
                self.run_dir / "manifest.json",
                self.manifest.to_dict(),
                label="experiment manifest",
            )
            self._verify_control_file(
                self.run_dir / "plan.json",
                self.plan.to_dict(),
                label="experiment plan",
            )
            self.verify()
            return self.run_dir

        if resume:
            raise ArtifactError("resume run identity is missing or invalid")
        self.results_dir.mkdir(parents=True, exist_ok=False)
        self._assert_run_path_confined()
        self._write_new(identity_path, self._identity_payload())
        self._write_new(self.run_dir / "manifest.json", self.manifest.to_dict())
        self._write_new(self.run_dir / "plan.json", self.plan.to_dict())
        self._write_new(self.run_dir / "result_index.json", {"schema_version": 1, "results": {}})
        return self.run_dir

    def completed_cell_ids(self) -> frozenset[str]:
        self._assert_run_path_confined()
        index = self._load_index()
        completed: set[str] = set()
        for cell_id, expected_hash in index.items():
            path = self.results_dir / f"{cell_id}.json"
            if path.is_symlink():
                raise ArtifactError(f"result path must not be a symbolic link: {cell_id}")
            if not path.is_file():
                raise ArtifactError(f"result index references a missing cell: {cell_id}")
            actual_hash = _sha256_bytes(path.read_bytes())
            if actual_hash != expected_hash:
                raise ArtifactError(f"result checksum mismatch for cell {cell_id}")
            completed.add(cell_id)
        return frozenset(completed)

    def result_status_counts(self) -> Mapping[str, int]:
        """Return terminal status counts across all append-only cell records."""

        counts = {"succeeded": 0, "failed": 0}
        for cell_id in self.completed_cell_ids():
            record = _read_json_object(
                self.results_dir / f"{cell_id}.json",
                label=f"result {cell_id}",
            )
            result = record.get("result")
            status = result.get("status") if isinstance(result, Mapping) else None
            if not isinstance(status, str) or status not in counts:
                raise ArtifactError(f"result {cell_id} has an invalid terminal status")
            counts[status] += 1
        return freeze_mapping(counts)

    def write_cell_result(self, cell_id: str, payload: Mapping[str, Any]) -> Path:
        if cell_id not in {cell.cell_id for cell in self.plan.cells}:
            raise ArtifactError("refusing result for a cell outside the frozen plan")
        try:
            serialized = (canonical_json(payload) + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArtifactError("cell result must be canonical JSON") from exc
        destination = self.results_dir / f"{cell_id}.json"
        with self._lock:
            self._assert_run_path_confined()
            index = self._load_index()
            digest = _sha256_bytes(serialized)
            if cell_id in index:
                if index[cell_id] != digest or destination.read_bytes() != serialized:
                    raise ArtifactError(
                        "append-only result store refuses different content for a completed cell"
                    )
                return destination
            _atomic_write(destination, serialized)
            index[cell_id] = digest
            _atomic_write(
                self.run_dir / "result_index.json",
                (canonical_json({"schema_version": 1, "results": index}) + "\n").encode("utf-8"),
            )
        return destination

    def append_event(self, payload: Mapping[str, Any]) -> None:
        try:
            line = (canonical_json(payload) + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArtifactError("experiment event must be canonical JSON") from exc
        path = self.run_dir / "events.jsonl"
        with self._lock:
            self._assert_run_path_confined()
            _append_no_follow(path, line)

    def verify(self) -> Mapping[str, Any]:
        verified = verify_experiment_run_directory(self.run_dir)
        if (
            verified["identity_hash"] != self.identity.identity_hash
            or verified["manifest_hash"] != self.manifest.manifest_hash
            or verified["plan_hash"] != self.plan.plan_hash
            or verified["driver_hash"] != self.identity.driver_hash
        ):
            raise ArtifactError("verified run differs from the active artifact store")
        return verified

    def _identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "identity": to_jsonable(self.identity),
            "identity_hash": self.identity.identity_hash,
            "driver_fingerprint": self.driver_fingerprint,
        }

    def _assert_run_path_confined(self) -> None:
        """Reject existing symlinks or resolved paths outside the output root."""

        try:
            relative = self.run_dir.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError("experiment run path escapes the output root") from exc
        current = self.root
        if current.is_symlink():
            raise ArtifactError("experiment output root must not be a symbolic link")
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ArtifactError(
                    f"experiment run path must not contain a symbolic link: {current}"
                )
        for label, path in (
            ("run", self.run_dir),
            ("results", self.results_dir),
        ):
            try:
                path.resolve().relative_to(self.root)
            except (OSError, ValueError) as exc:
                raise ArtifactError(f"experiment {label} path escapes the output root") from exc

    def _load_index(self) -> dict[str, str]:
        path = self.run_dir / "result_index.json"
        try:
            raw = _read_json_object(path, label="experiment result index")
        except ArtifactError as exc:
            raise ArtifactError("experiment result index is invalid") from exc
        if (
            not isinstance(raw, dict)
            or set(raw) != {"schema_version", "results"}
            or raw["schema_version"] != 1
            or not isinstance(raw["results"], dict)
        ):
            raise ArtifactError("experiment result index schema is invalid")
        result: dict[str, str] = {}
        for cell_id, digest in raw["results"].items():
            if not isinstance(cell_id, str) or not cell_id or not isinstance(digest, str):
                raise ArtifactError("experiment result index entry is invalid")
            require_sha256(digest, label="result checksum")
            result[cell_id] = digest
        return result

    def _write_new(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.exists() or path.is_symlink():
            raise ArtifactError(f"refusing to overwrite experiment control file: {path.name}")
        _atomic_write(path, (canonical_json(payload) + "\n").encode("utf-8"))

    @staticmethod
    def _verify_control_file(
        path: Path,
        expected: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        try:
            actual = _read_json_object(path, label=label)
        except ArtifactError as exc:
            raise ArtifactError(f"{label} is missing or invalid") from exc
        if canonical_json(actual) != canonical_json(expected):
            raise ArtifactError(f"{label} differs from the frozen run")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_no_follow(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ArtifactError("experiment event path must not be a symbolic link")
    if path.exists() and not path.is_file():
        raise ArtifactError("experiment event path must be a regular file")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ArtifactError("experiment event path cannot be opened safely") from exc
    try:
        handle = os.fdopen(descriptor, "ab")
    except Exception:
        os.close(descriptor)
        raise
    with handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_experiment_run_directory(path: str | Path) -> Mapping[str, Any]:
    """Verify run-artifact integrity without claiming execution authenticity.

    This validates paths, schemas, hashes, deterministic plan closure, and
    result-record identities without importing the original driver.  A valid
    directory is checksum-consistent evidence, not proof that an external
    worker, model, service, or official scorer actually executed as claimed.
    """

    from agentinputlab.contracts.serialization import content_hash

    supplied_root = Path(path).expanduser()
    if not supplied_root.is_absolute():
        supplied_root = Path(os.path.abspath(supplied_root))
    if supplied_root.is_symlink():
        raise ArtifactError("experiment run directory must not be a symbolic link")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ArtifactError("experiment run directory is missing or invalid")
    identity_path = _require_internal_file(
        root,
        Path("run_identity.json"),
        label="run identity",
    )
    manifest_path = _require_internal_file(
        root,
        Path("manifest.json"),
        label="experiment manifest",
    )
    plan_path = _require_internal_file(
        root,
        Path("plan.json"),
        label="experiment plan",
    )
    results_root = _require_internal_directory(
        root,
        Path("results"),
        label="experiment results directory",
    )
    index_path = _require_internal_file(
        root,
        Path("result_index.json"),
        label="result index",
    )
    identity_raw = _read_json_object(identity_path, label="run identity")
    manifest_raw = _read_json_object(manifest_path, label="experiment manifest")
    plan_raw = _read_json_object(plan_path, label="experiment plan")
    manifest = ExperimentManifest.from_mapping(manifest_raw)
    if set(plan_raw) != {
        "study_id",
        "manifest_hash",
        "cells",
        "unit_inventories",
        "unit_inventory_hashes",
        "parent_plan_hash",
        "parent_cell_ids",
        "shard_index",
        "shard_count",
        "plan_hash",
        "cell_count",
    }:
        raise ArtifactError("experiment plan control schema is invalid")
    raw_cells = plan_raw["cells"]
    if not isinstance(raw_cells, list):
        raise ArtifactError("experiment plan cells must be an array")
    cells: list[ExperimentCell] = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping) or set(raw) != {
            "study_id",
            "benchmark_id",
            "split",
            "unit_id",
            "model_slot",
            "seed",
            "arm_ids",
            "manifest_hash",
            "cell_id",
        }:
            raise ArtifactError("experiment plan cell schema is invalid")
        cell = ExperimentCell(
            study_id=raw["study_id"],
            benchmark_id=raw["benchmark_id"],
            split=raw["split"],
            unit_id=raw["unit_id"],
            model_slot=raw["model_slot"],
            seed=raw["seed"],
            arm_ids=tuple(raw["arm_ids"]),
            manifest_hash=raw["manifest_hash"],
        )
        if raw["cell_id"] != cell.cell_id:
            raise ArtifactError("experiment plan cell hash mismatch")
        cells.append(cell)
    raw_inventories = plan_raw["unit_inventories"]
    inventory_hashes = plan_raw["unit_inventory_hashes"]
    if not isinstance(raw_inventories, Mapping):
        raise ArtifactError("experiment plan unit inventories must be an object")
    if not isinstance(inventory_hashes, Mapping):
        raise ArtifactError("experiment plan inventory hashes must be an object")
    inventories: dict[str, tuple[str, ...]] = {}
    for key, raw_units in raw_inventories.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(raw_units, list):
            raise ArtifactError("experiment plan unit inventory entry is invalid")
        inventories[key] = tuple(raw_units)
    parent_cell_ids = plan_raw["parent_cell_ids"]
    if not isinstance(parent_cell_ids, list):
        raise ArtifactError("experiment plan parent_cell_ids must be an array")
    plan = ExperimentPlan(
        study_id=plan_raw["study_id"],
        manifest_hash=plan_raw["manifest_hash"],
        cells=tuple(cells),
        unit_inventories=inventories,
        unit_inventory_hashes=inventory_hashes,
        parent_plan_hash=plan_raw["parent_plan_hash"],
        parent_cell_ids=tuple(parent_cell_ids),
        shard_index=plan_raw["shard_index"],
        shard_count=plan_raw["shard_count"],
    )
    if (
        plan.study_id != manifest.study_id
        or plan.manifest_hash != manifest.manifest_hash
        or plan_raw["plan_hash"] != plan.plan_hash
        or plan_raw["cell_count"] != len(plan.cells)
    ):
        raise ArtifactError("experiment plan identity differs from its manifest")
    for target in manifest.benchmarks:
        key = f"{target.cell_id}:{target.split}"
        if target.unit_ids and tuple(sorted(target.unit_ids)) != tuple(
            plan.unit_inventories.get(key, ())
        ):
            raise ArtifactError(
                f"experiment plan unit inventory differs from inline manifest units for {key}"
            )
    try:
        validate_experiment_plan(manifest, plan)
    except (TypeError, ValidationError) as exc:
        raise ArtifactError(f"experiment plan verification failed: {exc}") from exc

    if (
        set(identity_raw)
        != {
            "schema_version",
            "identity",
            "identity_hash",
            "driver_fingerprint",
        }
        or identity_raw["schema_version"] != 1
    ):
        raise ArtifactError("run identity schema is invalid")
    raw_identity = identity_raw["identity"]
    fingerprint = identity_raw["driver_fingerprint"]
    if not isinstance(raw_identity, Mapping) or not isinstance(fingerprint, Mapping):
        raise ArtifactError("run identity payload is invalid")
    if set(raw_identity) != {
        "study_id",
        "manifest_hash",
        "plan_hash",
        "driver_hash",
    }:
        raise ArtifactError("run identity fields are invalid")
    identity = ExperimentArtifactIdentity(
        study_id=raw_identity.get("study_id"),
        manifest_hash=raw_identity.get("manifest_hash"),
        plan_hash=raw_identity.get("plan_hash"),
        driver_hash=raw_identity.get("driver_hash"),
    )
    if (
        identity.manifest_hash != manifest.manifest_hash
        or identity.study_id != manifest.study_id
        or identity.plan_hash != plan.plan_hash
        or identity.driver_hash != content_hash(fingerprint)
        or identity_raw["identity_hash"] != identity.identity_hash
    ):
        raise ArtifactError("run identity hash mismatch")

    index_raw = _read_json_object(index_path, label="result index")
    if (
        set(index_raw) != {"schema_version", "results"}
        or index_raw["schema_version"] != 1
        or not isinstance(index_raw["results"], Mapping)
    ):
        raise ArtifactError("result index schema is invalid")
    planned_by_id = {cell.cell_id: cell for cell in plan.cells}
    planned = set(planned_by_id)
    completed: set[str] = set()
    status_counts = {"succeeded": 0, "failed": 0}
    for cell_id, expected in index_raw["results"].items():
        if cell_id not in planned or not isinstance(expected, str):
            raise ArtifactError("result index references an unplanned cell")
        require_sha256(expected, label="result checksum")
        result_path = _require_internal_file(
            root,
            Path("results") / f"{cell_id}.json",
            label=f"result {cell_id}",
        )
        try:
            result_path.relative_to(results_root)
        except ValueError as exc:
            raise ArtifactError(f"result {cell_id} escapes the results directory") from exc
        try:
            content = result_path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"result index references missing cell {cell_id}") from exc
        if _sha256_bytes(content) != expected:
            raise ArtifactError(f"result checksum mismatch for cell {cell_id}")
        result_record = _decode_json_object(content, label=f"result {cell_id}")
        if set(result_record) != {
            "schema_version",
            "study_id",
            "manifest_hash",
            "plan_hash",
            "cell",
            "result",
            "finished_at_utc",
            "claims",
        }:
            raise ArtifactError(f"result {cell_id} record schema is invalid")
        if (
            result_record.get("schema_version") != 1
            or result_record.get("study_id") != manifest.study_id
            or result_record.get("manifest_hash") != manifest.manifest_hash
            or result_record.get("plan_hash") != plan.plan_hash
        ):
            raise ArtifactError(f"result {cell_id} control identity mismatch")
        raw_cell = result_record.get("cell")
        expected_cell = {
            **to_jsonable(planned_by_id[cell_id]),
            "cell_id": cell_id,
        }
        if not isinstance(raw_cell, Mapping) or canonical_json(raw_cell) != canonical_json(
            expected_cell
        ):
            raise ArtifactError(f"result {cell_id} frozen cell identity mismatch")
        _require_utc_timestamp(
            result_record.get("finished_at_utc"),
            label=f"result {cell_id} finished_at_utc",
        )
        try:
            from .executor import parse_experiment_cell_result

            result = parse_experiment_cell_result(
                planned_by_id[cell_id],
                result_record.get("result"),
            )
        except (TypeError, ValidationError) as exc:
            raise ArtifactError(f"result {cell_id} payload is invalid: {exc}") from exc
        status = result.status
        claims = result_record.get("claims")
        expected_claims = {
            "official_parity_verified": False,
            "confirmatory_candidate_record": (
                manifest.stage.value == "confirmatory" and status == "succeeded"
            ),
            "paper_claim_authorized": False,
            "resource_receipt_count": len(result.resource_receipts),
        }
        if not isinstance(claims, Mapping) or canonical_json(claims) != canonical_json(
            expected_claims
        ):
            raise ArtifactError(f"result {cell_id} claim metadata is invalid")
        status_counts[status] += 1
        completed.add(cell_id)
    return freeze_mapping(
        {
            "study_id": manifest.study_id,
            "identity_hash": identity.identity_hash,
            "manifest_hash": manifest.manifest_hash,
            "plan_hash": plan.plan_hash,
            "driver_hash": identity.driver_hash,
            "profile_kind": manifest.estimand_profile_kind.value,
            "comparison_profile_hash": content_hash(
                {
                    "schema_version": 1,
                    "manifest_hash": manifest.manifest_hash,
                    "driver_hash": identity.driver_hash,
                    "profile_kind": manifest.estimand_profile_kind.value,
                }
            ),
            "planned_cells": len(planned),
            "completed_cells": len(completed),
            "complete": completed == planned,
            "succeeded_cells": status_counts["succeeded"],
            "failed_cells": status_counts["failed"],
            "all_succeeded": completed == planned and status_counts["failed"] == 0,
            "is_shard": plan.parent_plan_hash is not None,
            "parent_plan_hash": plan.parent_plan_hash,
            "shard_index": plan.shard_index,
            "shard_count": plan.shard_count,
        }
    )


def _require_internal_path(
    root: Path,
    relative: Path,
    *,
    label: str,
) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactError(f"{label} path must remain under the experiment run directory")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ArtifactError(f"{label} path must not contain a symbolic link")
    resolved = current.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(f"{label} path escapes the experiment run directory") from exc
    return resolved


def _require_internal_file(root: Path, relative: Path, *, label: str) -> Path:
    path = _require_internal_path(root, relative, label=label)
    if not path.is_file():
        raise ArtifactError(f"{label} is missing or invalid")
    return path


def _require_internal_directory(root: Path, relative: Path, *, label: str) -> Path:
    path = _require_internal_path(root, relative, label=label)
    if not path.is_dir():
        raise ArtifactError(f"{label} is missing or invalid")
    return path


def _decode_json_object(content: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is missing or invalid") from exc
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be a JSON object")
    return value


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink():
        raise ArtifactError(f"{label} path must not be a symbolic link")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"{label} is missing or invalid") from exc
    return _decode_json_object(content, label=label)


def _require_utc_timestamp(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ArtifactError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ArtifactError(f"{label} must be an ISO-8601 UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ArtifactError(f"{label} must be an ISO-8601 UTC timestamp")


__all__ = [
    "ExperimentArtifactIdentity",
    "ExperimentArtifactStore",
    "verify_experiment_run_directory",
]
