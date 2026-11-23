"""Bind installed process workers without executing source or contacting models."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from agentinputlab.contracts import ArtifactError, ValidationError

_WORKERS = {
    "reformer_worker": "reformer",
    "infocqr_worker": "infocqr",
    "ask_or_assume_worker": "ask_or_assume",
    "acon_observation_worker": "acon",
    "acon_history_worker": "acon",
}


def _sha(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"worker preparation cannot read {path}") from exc
    return digest.hexdigest()


def _installed_worker(name):
    return Path(str(files("workers").joinpath(name)))


def prepare(config, output_dir):
    """Pin transport files; keep explicit source, model, artifacts and budgets."""
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise ValidationError("worker preparation requires a new output directory")
    runtime = dict(config.runtime)
    kind = runtime.get("kind")
    if not isinstance(kind, str) or kind not in _WORKERS:
        raise ValidationError("prepare-worker supports ReFormeR, InfoCQR, Ask-or-Assume and ACON process workers")
    for key in ("checkout", "python_executable"):
        value = runtime.get(key)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValidationError(f"runtime.{key} requires an explicit local path")
        # In particular, retain the interpreter's virtualenv symlink path.
        runtime[key] = str(Path(value).absolute())
    checkout = Path(runtime["checkout"])
    python = Path(runtime["python_executable"])
    if not checkout.is_dir():
        raise ArtifactError("worker checkout is missing; prepare the pinned upstream source first")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ArtifactError("worker Python is missing or not executable; prepare its environment first")
    worker = _installed_worker(_WORKERS[kind])
    derived = {
        "driver_path": str((worker / "driver.py").resolve()),
        "compatibility_manifest_path": str((worker / "compatibility_manifest.json").resolve()),
    }
    for key, value in derived.items():
        if key in runtime:
            if not isinstance(runtime[key], str) or not runtime[key] or "\x00" in runtime[key]:
                raise ValidationError(f"runtime.{key} requires a local path")
            if Path(runtime[key]).resolve() != Path(value):
                raise ValidationError(f"runtime.{key} conflicts with this installation's worker")
        runtime[key] = value
    for path_key, hash_key in (("driver_path", "driver_sha256"),
                               ("compatibility_manifest_path", "compatibility_manifest_sha256"),
                               ("python_executable", "python_executable_sha256")):
        actual = _sha(Path(runtime[path_key]))
        if hash_key in runtime and runtime[hash_key] != actual:
            raise ValidationError(f"runtime.{hash_key} conflicts with the current file")
        runtime[hash_key] = actual
    # This checks the exact files named by the bundled manifest, not arbitrary
    # user Python. Runtime still verifies Git identity and dependency versions.
    manifest = json.loads(Path(runtime["compatibility_manifest_path"]).read_text())
    sources = manifest.get("source_files", {})
    if not isinstance(sources, dict) or not sources:
        raise ArtifactError("installed worker manifest has no source files")
    for name, expected in sources.items():
        target = (checkout / name).resolve()
        if not target.is_relative_to(checkout.resolve()) or _sha(target) != expected:
            raise ArtifactError(f"worker source checksum mismatch: {name}")
    if not config.backend:
        raise ValidationError("worker preparation requires an explicit backend identity")
    for key in ("model_id", "base_url"):
        expected = config.backend.get(key)
        if not isinstance(expected, str) or not expected:
            raise ValidationError(f"backend.{key} is required")
        if key in runtime and runtime[key] != expected:
            raise ValidationError(f"runtime.{key} conflicts with backend.{key}")
        runtime[key] = expected
    if kind not in {"acon_history_worker", "acon_observation_worker"}:
        runtime.setdefault("api_key_env", config.backend.get("api_key_env", "OPENAI_API_KEY"))
    if kind != "ask_or_assume_worker" and "extra_body" in config.backend:
        runtime.setdefault("extra_body", config.backend["extra_body"])
    ready = replace(config, runtime=runtime)
    ready.validate()
    output.mkdir(parents=True)
    run = output / "run.json"
    with run.open("x") as stream:
        json.dump(ready.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"release_id": ready.release_id, "run_path": str(run), "run_sha256": _sha(run),
             "source_files_verified": len(sources), "model_calls": 0, "fit_performed": False,
             "worker_executed": False, "worker_dependencies_verified": False,
             "method_artifacts_prepared": False, "scientific_parameters_changed": False,
             "scope": "installed_transport_and_source_files_only"}
    with (output / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof
