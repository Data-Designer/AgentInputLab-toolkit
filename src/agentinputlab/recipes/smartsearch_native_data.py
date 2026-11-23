"""Complete the pinned vendored framework's omitted data package, unchanged.

All 171 Python files actually vendored by SmartSearch match this official
LLaMA-Factory revision. Only its 17 missing data files are mounted separately;
no existing trainer, loss, model or configuration file is replaced.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .qrag import _sha
from .smartsearch_curriculum import verify_smartsearch_preparation_source

SMARTSEARCH_LLAMAFACTORY_COMMIT = "3f0c3dc84d95e0ebdb632fd427ad30f7217c32e2"
_NATIVE_BLOBS_SHA256 = "0b31b97e27fc1b8053bb600ef7afb0572ceeeaa8da898a6c1af477a8a01b6bc3"
_SCHEMA = "agentinputlab.smartsearch.native_data_source.v1"
_DATA_BLOBS = {
    "__init__.py": "11c8c9fcecd10e736e240196fde98f833c9df3dc",
    "collator.py": "162f432c9e5bf195ed4c6a821eb36d279bf3bac4",
    "converter.py": "7ec6f12becf8318404905cdb8070890214619bf7",
    "data_utils.py": "139425a22691a9579f80aff99e9788a327b1610a",
    "formatter.py": "1c080f8812d0d9feb320e949d39bfcb1a0e1e582",
    "loader.py": "d3d44e6f3c25de0fa46c333be6a3a39d5f1dcfee",
    "mm_plugin.py": "2ecf06a6600bb947f66314a383d4373f7a10dd76",
    "parser.py": "5209da64954a363c59d232010111139b7b37fdfe",
    "processor/__init__.py": "357ab7899f9eecbd29344482d109b89af274ea2e",
    "processor/feedback.py": "871615b9266e501f25f68e84e4536c0d24617803",
    "processor/pairwise.py": "94101deb8e75af73c1851720604994a11f2eb87d",
    "processor/pretrain.py": "3fa6b1ca58a8d59493cd4b43c51cb268080cc506",
    "processor/processor_utils.py": "db44b19cf6fc84d6551fb7cce82283774ae72030",
    "processor/supervised.py": "b5aba11b6535078f46bdf6aca743c6ae262e1fc6",
    "processor/unsupervised.py": "256174b6dd38696b5b180501102af40ff395d0a9",
    "template.py": "d196ae388d9a8ebdbd99dfb85349cff3aa9e38cf",
    "tool_utils.py": "48132af84610b4083388cb3cdb3af3835231a4db",
}


def _git(root, *args):
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
             "-C", str(root), *args], capture_output=True, check=True, timeout=60,
            env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
                 "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                 "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExternalDependencyError("SmartSearch framework Git objects are unavailable") from exc


def _blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _regular(root, relative):
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ExternalDependencyError("SmartSearch framework source contains a symlink")
    if not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ExternalDependencyError("SmartSearch framework source is missing or excessive")
    return path


def _native_files(checkout):
    binding = verify_smartsearch_preparation_source(checkout)
    root = Path(binding["root"])
    prefix = "LLaMA-Factory/src/llamafactory/"
    blobs = {}
    for line in _git(root, "ls-tree", "-r", "HEAD", "--", prefix).decode().splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, oid = metadata.split()
        if not path.endswith(".py"):
            continue
        if mode != "100644" or kind != "blob" or not path.startswith(prefix):
            raise ExternalDependencyError("SmartSearch vendored framework object type differs")
        blobs[path[len(prefix):]] = oid
    digest = hashlib.sha256(json.dumps(blobs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != _NATIVE_BLOBS_SHA256 or len(blobs) != 171:
        raise ExternalDependencyError("SmartSearch's 171-file native framework revision differs")
    files = {}
    for name, oid in blobs.items():
        data = _regular(root, prefix + name).read_bytes()
        if _blob(data) != oid:
            raise ExternalDependencyError("SmartSearch native framework file changed")
        files[name] = hashlib.sha256(data).hexdigest()
    return root, files


def _data_files(root):
    expected = {"manifest.json"} | {"data/" + name for name in _DATA_BLOBS}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ExternalDependencyError("SmartSearch data supplement contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ExternalDependencyError("SmartSearch data supplement file inventory differs")
    files = {}
    for name, oid in _DATA_BLOBS.items():
        data = _regular(root, "data/" + name).read_bytes()
        if _blob(data) != oid:
            raise ExternalDependencyError("SmartSearch official data source bytes differ")
        files[name] = hashlib.sha256(data).hexdigest()
    return files


def verify_smartsearch_native_data_source(checkout, supplement_root):
    """Verify all native and supplementary bytes without importing a trainer."""
    root = Path(supplement_root)
    if root.is_symlink() or not root.is_dir():
        raise ExternalDependencyError("SmartSearch requires its explicit native data supplement")
    root = root.resolve()
    source, native = _native_files(checkout)
    manifest = json.loads(_regular(root, "manifest.json").read_text())
    artifact_id = manifest.pop("artifact_id", None)
    if (manifest.get("schema") != _SCHEMA or artifact_id != content_hash(manifest)
            or manifest.get("official_commit") != SMARTSEARCH_LLAMAFACTORY_COMMIT
            or manifest.get("native_files") != native or manifest.get("data_files") != _data_files(root)):
        raise ExternalDependencyError("SmartSearch native data supplement manifest differs")
    return {"root": str(root), "manifest_sha256": _sha(root / "manifest.json"),
            "artifact_id": artifact_id, "official_commit": SMARTSEARCH_LLAMAFACTORY_COMMIT,
            "native_source_root": str(source), "matched_native_python_files": 171,
            "supplementary_data_python_files": 17, "existing_source_modified": False}


def prepare_smartsearch_native_data_source(*, checkout, official_checkout, output_root):
    """Copy only 17 exact official data files to a new mount; no model calls.

    ``official_checkout`` is a Git repository containing the pinned historical
    objects. Its working-tree revision is irrelevant; existing files are never
    checked out, overwritten or used as executable fallbacks.
    """
    source, native = _native_files(checkout)
    official = Path(official_checkout)
    if official.is_symlink() or not official.is_dir():
        raise ValidationError("SmartSearch requires a mounted official LLaMA-Factory Git repository")
    official = official.resolve()
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise ValidationError("SmartSearch native data supplement must be create-only")
    root = root.resolve(strict=False)
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (source, official)):
        raise ValidationError("SmartSearch data supplement cannot overlap a source checkout")
    rows = {}
    for name, oid in _DATA_BLOBS.items():
        data = _git(official, "show", f"{SMARTSEARCH_LLAMAFACTORY_COMMIT}:src/llamafactory/data/{name}")
        if len(data) > 2 * 1024**2 or _blob(data) != oid:
            raise ExternalDependencyError("SmartSearch official data Git object differs")
        rows[name] = data
    root.mkdir(parents=True, exist_ok=False)
    for name, data in rows.items():
        path = root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    manifest = {"schema": _SCHEMA, "official_commit": SMARTSEARCH_LLAMAFACTORY_COMMIT,
                "native_files": native, "data_files": {n: hashlib.sha256(v).hexdigest() for n, v in rows.items()},
                "existing_source_modified": False, "model_calls": 0, "optimizer_steps": 0}
    manifest["artifact_id"] = content_hash(manifest)
    with (root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, allow_nan=False)
    return verify_smartsearch_native_data_source(source, root)
