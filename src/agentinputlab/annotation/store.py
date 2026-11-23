"""Create-only, versioned storage for non-blocking measurement audits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentinputlab.contracts.serialization import canonical_json, content_hash

ANNOTATION_VERSION_PROTOCOL = "agentinputlab.annotation-version.v1"
ANNOTATION_SELECTION_PROTOCOL = "agentinputlab.annotation-selection.v1"
ANNOTATION_KINDS = frozenset({"input_audit", "trajectory_audit"})


class AnnotationStoreError(ValueError):
    """Raised when an annotation version would be ambiguous or mutable."""


def validate_annotation_selection(
    root: str | Path,
    *,
    kind: str,
) -> Mapping[str, Any] | None:
    """Return the selected completed audit version, or fail on ambiguity.

    Automatic measurements do not require a human audit to exist.  Once at
    least one completed audit does exist, however, experiment and plotting
    entry points must bind an explicit selection receipt instead of silently
    choosing the newest file on disk.
    """

    if kind not in ANNOTATION_KINDS:
        raise AnnotationStoreError("annotation kind must be input_audit or trajectory_audit")
    root_path = Path(root).expanduser().resolve()
    if root_path.is_symlink():
        raise AnnotationStoreError("annotation root must not be a symbolic link")
    if not root_path.exists():
        return None
    if not root_path.is_dir():
        raise AnnotationStoreError("annotation root must be a directory")

    completed: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root_path.glob(f"{kind}-v[0-9][0-9][0-9]/manifest.json")):
        value = _read_json(path)
        items_hash = value.get("items_sha256")
        if not isinstance(items_hash, str) or not items_hash:
            raise AnnotationStoreError(f"annotation manifest has no items hash: {path}")
        _validate_manifest(value, kind=kind, items_hash=items_hash)
        if value["status"] == "complete":
            annotations_path = path.parent / "annotations.jsonl"
            if annotations_path.is_symlink() or not annotations_path.is_file():
                raise AnnotationStoreError(
                    f"completed annotation data is unavailable: {annotations_path}"
                )
            if _sha256(annotations_path) != value.get("annotations_sha256"):
                raise AnnotationStoreError(
                    f"completed annotation data hash does not match its manifest: {annotations_path}"
                )
            completed[str(value["version_id"])] = value
    if not completed:
        return None

    selection_path = root_path / f"{kind}-selection.json"
    if selection_path.is_symlink() or not selection_path.is_file():
        versions = ", ".join(sorted(completed))
        raise AnnotationStoreError(
            f"completed {kind} annotation exists ({versions}) but no version is selected; "
            "open agentinputlab-annotate or run select_annotation_version.py"
        )
    selection = _read_json(selection_path)
    expected = selection.get("selection_hash")
    unhashed = {key: item for key, item in selection.items() if key != "selection_hash"}
    version_id = selection.get("version_id")
    if (
        selection.get("schema_version") != 1
        or selection.get("protocol") != ANNOTATION_SELECTION_PROTOCOL
        or selection.get("kind") != kind
        or expected != content_hash(unhashed)
        or version_id not in completed
    ):
        raise AnnotationStoreError(f"selected {kind} annotation receipt is invalid")
    manifest = completed[str(version_id)]
    if (
        selection.get("version_manifest_hash") != manifest["manifest_hash"]
        or selection.get("items_sha256") != manifest["items_sha256"]
        or selection.get("annotations_sha256") != manifest["annotations_sha256"]
    ):
        raise AnnotationStoreError(f"selected {kind} annotation hashes do not match its version")
    return dict(selection)


class AnnotationStore:
    """Persist one local audit census in immutable, explicitly selected versions."""

    def __init__(self, *, items_path: str | Path, root: str | Path, kind: str) -> None:
        if kind not in ANNOTATION_KINDS:
            raise AnnotationStoreError("annotation kind must be input_audit or trajectory_audit")
        self.items_path = Path(items_path).expanduser().resolve()
        self.root = Path(root).expanduser().resolve()
        self.kind = kind
        if self.items_path.is_symlink() or not self.items_path.is_file():
            raise AnnotationStoreError(f"annotation item file is unavailable: {self.items_path}")
        self.items = _load_items(self.items_path)
        self.items_hash = _sha256(self.items_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        versions = self.list_versions()
        completed = [row for row in versions if row["status"] == "complete"]
        selection = self._selection()
        return {
            "schema_version": 1,
            "service": "agentinputlab-annotation",
            "kind": self.kind,
            "item_count": len(self.items),
            "items_sha256": self.items_hash,
            "versions": versions,
            "completed_versions": len(completed),
            "completed_version_exists": bool(completed),
            "requires_reuse_or_new_version_choice": bool(completed),
            "selected_version": selection["version_id"] if selection is not None else None,
            "selection_path": str(self.selection_path),
            "execution_depends_on_completion": False,
        }

    @property
    def selection_path(self) -> Path:
        return self.root / f"{self.kind}-selection.json"

    def list_versions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob(f"{self.kind}-v[0-9][0-9][0-9]/manifest.json")):
            try:
                manifest = _read_json(path)
                _validate_manifest(manifest, kind=self.kind, items_hash=self.items_hash)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            rows.append(_public_manifest(manifest))
        return rows

    def create_version(self, *, annotator_id: str) -> dict[str, Any]:
        annotator_id = _text(annotator_id, label="annotator_id")
        existing = self.list_versions()
        number = max((int(row["version_id"].rsplit("v", 1)[1]) for row in existing), default=0) + 1
        version_id = f"{self.kind}-v{number:03d}"
        directory = self.root / version_id
        directory.mkdir(parents=False, exist_ok=False)
        now = datetime.now(timezone.utc).isoformat()
        without_hash = {
            "schema_version": 1,
            "protocol": ANNOTATION_VERSION_PROTOCOL,
            "version_id": version_id,
            "kind": self.kind,
            "annotator_id": annotator_id,
            "items_path": str(self.items_path),
            "items_sha256": self.items_hash,
            "item_count": len(self.items),
            "annotation_count": 0,
            "status": "in_progress",
            "created_at": now,
            "completed_at": None,
            "annotations_sha256": None,
            "supersedes_completed_version": any(row["status"] == "complete" for row in existing),
        }
        manifest = {**without_hash, "manifest_hash": content_hash(without_hash)}
        _write_atomic(directory / "manifest.json", manifest)
        _write_atomic(directory / "annotations.jsonl", b"")
        return _public_manifest(manifest)

    def payload(self, version_id: str) -> dict[str, Any]:
        manifest = self._manifest(version_id)
        annotations = self._annotations(version_id)
        return {
            "manifest": _public_manifest(manifest),
            "items": self.items,
            "annotations": annotations,
        }

    def save(self, version_id: str, annotation: Mapping[str, Any]) -> dict[str, Any]:
        manifest = self._manifest(version_id)
        if manifest["status"] == "complete":
            raise AnnotationStoreError(
                "completed annotations are immutable; create a new version to re-annotate"
            )
        item_id = _text(annotation.get("item_id"), label="annotation item_id")
        if item_id not in {row["item_id"] for row in self.items}:
            raise AnnotationStoreError("annotation item_id is outside the frozen audit census")
        labels = annotation.get("labels")
        if not isinstance(labels, Mapping) or not labels:
            raise AnnotationStoreError("annotation labels must be a non-empty object")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, (str, bool, int))
            or (isinstance(value, str) and not value.strip())
            for key, value in labels.items()
        ):
            raise AnnotationStoreError("annotation labels contain invalid values")
        notes = annotation.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 4000:
            raise AnnotationStoreError("annotation notes must be a string of at most 4000 chars")
        rows = {row["item_id"]: row for row in self._annotations(version_id)}
        row_without_hash = {
            "schema_version": 1,
            "version_id": version_id,
            "kind": self.kind,
            "item_id": item_id,
            "annotator_id": manifest["annotator_id"],
            "labels": dict(sorted(labels.items())),
            "notes": notes,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        rows[item_id] = {**row_without_hash, "annotation_hash": content_hash(row_without_hash)}
        ordered = [rows[key] for key in sorted(rows)]
        annotations_path = self.root / version_id / "annotations.jsonl"
        _write_atomic(annotations_path, _jsonl_bytes(ordered))
        updated = self._updated_manifest(
            manifest,
            annotation_count=len(ordered),
            annotations_sha256=_sha256(annotations_path),
        )
        _write_atomic(self.root / version_id / "manifest.json", updated)
        return {
            "saved": True,
            "version_id": version_id,
            "item_id": item_id,
            "annotation_count": len(ordered),
            "item_count": len(self.items),
        }

    def complete(self, version_id: str) -> dict[str, Any]:
        manifest = self._manifest(version_id)
        if manifest["status"] == "complete":
            return _public_manifest(manifest)
        annotations = self._annotations(version_id)
        observed = {row["item_id"] for row in annotations}
        expected = {row["item_id"] for row in self.items}
        if observed != expected:
            raise AnnotationStoreError(
                f"cannot complete annotation: {len(expected - observed)} audit items remain"
            )
        annotations_path = self.root / version_id / "annotations.jsonl"
        updated = self._updated_manifest(
            manifest,
            annotation_count=len(annotations),
            annotations_sha256=_sha256(annotations_path),
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_atomic(self.root / version_id / "manifest.json", updated)
        return _public_manifest(updated)

    def select(self, version_id: str, *, output: str | Path | None = None) -> dict[str, Any]:
        manifest = self._manifest(version_id)
        if manifest["status"] != "complete":
            raise AnnotationStoreError("only a complete annotation version may be selected")
        output_path = (
            Path(output).expanduser().resolve() if output is not None else self.selection_path
        )
        without_hash = {
            "schema_version": 1,
            "protocol": ANNOTATION_SELECTION_PROTOCOL,
            "kind": self.kind,
            "version_id": version_id,
            "version_manifest_hash": manifest["manifest_hash"],
            "items_sha256": self.items_hash,
            "annotations_sha256": manifest["annotations_sha256"],
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
        selection = {**without_hash, "selection_hash": content_hash(without_hash)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(
            output_path, (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
        )
        return selection

    def _selection(self) -> Mapping[str, Any] | None:
        path = self.selection_path
        if path.is_symlink() or not path.is_file():
            return None
        value = _read_json(path)
        expected = value.get("selection_hash")
        unhashed = {key: item for key, item in value.items() if key != "selection_hash"}
        if (
            value.get("schema_version") != 1
            or value.get("protocol") != ANNOTATION_SELECTION_PROTOCOL
            or value.get("kind") != self.kind
            or value.get("items_sha256") != self.items_hash
            or expected != content_hash(unhashed)
        ):
            raise AnnotationStoreError("annotation selection receipt is invalid")
        manifest = self._manifest(str(value.get("version_id", "")))
        annotations_path = self.root / manifest["version_id"] / "annotations.jsonl"
        if (
            manifest["status"] != "complete"
            or value.get("version_manifest_hash") != manifest["manifest_hash"]
            or value.get("annotations_sha256") != manifest["annotations_sha256"]
            or annotations_path.is_symlink()
            or not annotations_path.is_file()
            or _sha256(annotations_path) != manifest["annotations_sha256"]
        ):
            raise AnnotationStoreError("selected annotation version is unavailable or modified")
        return value

    def _manifest(self, version_id: str) -> Mapping[str, Any]:
        version_id = _text(version_id, label="version_id")
        if not version_id.startswith(f"{self.kind}-v") or not version_id[-3:].isdigit():
            raise AnnotationStoreError("annotation version_id is invalid")
        path = self.root / version_id / "manifest.json"
        if path.is_symlink() or not path.is_file():
            raise AnnotationStoreError("annotation version does not exist")
        manifest = _read_json(path)
        _validate_manifest(manifest, kind=self.kind, items_hash=self.items_hash)
        return manifest

    def _annotations(self, version_id: str) -> list[dict[str, Any]]:
        path = self.root / version_id / "annotations.jsonl"
        if path.is_symlink() or not path.is_file():
            raise AnnotationStoreError("annotation version data is unavailable")
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping) or value.get("version_id") != version_id:
                raise AnnotationStoreError(f"invalid annotation row at line {line_number}")
            expected = value.get("annotation_hash")
            unhashed = {key: item for key, item in value.items() if key != "annotation_hash"}
            if expected != content_hash(unhashed):
                raise AnnotationStoreError(f"annotation hash mismatch at line {line_number}")
            rows.append(dict(value))
        if len({row["item_id"] for row in rows}) != len(rows):
            raise AnnotationStoreError("annotation version contains duplicate items")
        return rows

    @staticmethod
    def _updated_manifest(manifest: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
        value = {key: item for key, item in manifest.items() if key != "manifest_hash"}
        value.update(changes)
        return {**value, "manifest_hash": content_hash(value)}


def _load_items(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise AnnotationStoreError(f"annotation item {line_number} must be an object")
        item_id = value.get("audit_item_id", value.get("item_id"))
        item_id = _text(item_id, label=f"annotation item {line_number} ID")
        rows.append({"item_id": item_id, **dict(value)})
    if not rows or len({row["item_id"] for row in rows}) != len(rows):
        raise AnnotationStoreError("annotation items must be non-empty and unique")
    return rows


def _validate_manifest(value: Mapping[str, Any], *, kind: str, items_hash: str) -> None:
    expected = value.get("manifest_hash")
    unhashed = {key: item for key, item in value.items() if key != "manifest_hash"}
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != ANNOTATION_VERSION_PROTOCOL
        or value.get("kind") != kind
        or value.get("items_sha256") != items_hash
        or value.get("status") not in {"in_progress", "complete"}
        or expected != content_hash(unhashed)
    ):
        raise AnnotationStoreError("annotation version manifest is invalid")


def _public_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "version_id",
            "kind",
            "annotator_id",
            "item_count",
            "annotation_count",
            "status",
            "created_at",
            "completed_at",
            "supersedes_completed_version",
            "manifest_hash",
        )
    }


def _write_atomic(path: Path, data: bytes | Mapping[str, Any]) -> None:
    if isinstance(data, Mapping):
        data = (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AnnotationStoreError("annotation manifest must be an object")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationStoreError(f"{label} must be a non-empty string")
    return value.strip()


__all__ = [
    "ANNOTATION_KINDS",
    "ANNOTATION_SELECTION_PROTOCOL",
    "ANNOTATION_VERSION_PROTOCOL",
    "AnnotationStore",
    "AnnotationStoreError",
    "validate_annotation_selection",
]
