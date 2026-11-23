"""Reuse the package's preparation entrypoints and owned worker environments."""
from __future__ import annotations

import importlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

from agentinputlab.configuration import RunConfig
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable

from .configuration import file_sha, read_json
from .readiness import interpreter_readiness
from .recipes import AUTO_PREPARERS, PREPARERS, WORKER_KINDS


def validate_asset(value):
    from agentinputlab.artifacts import AcquisitionPolicy, ArtifactKind, ArtifactManifest
    from agentinputlab.configuration import _known, _object
    data = _object(value, "preparation asset")
    _known(data, {"manifest", "destination", "allowed_hosts"}, "preparation asset")
    raw = _object(data.get("manifest"), "asset manifest")
    try:
        manifest = ArtifactManifest(**{**raw, "kind": ArtifactKind(raw["kind"]),
                                      "acquisition": AcquisitionPolicy(raw["acquisition"])})
    except (KeyError, TypeError, ValueError):
        raise ValidationError("invalid preparation asset manifest") from None
    if not manifest.sha256 or manifest.size_bytes is None or manifest.size_bytes < 1:
        raise ValidationError("automatic assets require a pinned SHA-256 and positive size_bytes")
    destination = data.get("destination")
    if not isinstance(destination, str) or not destination.strip() or "\x00" in destination:
        raise ValidationError("preparation asset needs a destination path")
    hosts = data.get("allowed_hosts", [])
    if not isinstance(hosts, (tuple, list)) or any(not isinstance(host, str) or not host for host in hosts):
        raise ValidationError("asset allowed_hosts must be an array of hostnames")
    return {"manifest": to_jsonable(manifest), "destination": str(Path(destination).expanduser().absolute()),
            "allowed_hosts": list(hosts)}


def prepare_assets(workflow, output, *, allow_network):
    """Acquire missing exact files with the existing bounded HTTPS transport."""
    from urllib.parse import urlsplit

    from agentinputlab.artifacts import (
        AcquisitionPolicy,
        ArtifactKind,
        ArtifactManifest,
        ArtifactStore,
    )
    from agentinputlab.data import (
        DownloadPolicy,
        PinnedHTTPSDownloadTransport,
        download_artifact_streamed,
    )
    from agentinputlab.data.storage import require_storage_capacity
    pending = []
    for value in workflow.preparation.get("assets", ()):
        raw = value["manifest"]
        manifest = ArtifactManifest(**{**raw, "kind": ArtifactKind(raw["kind"]),
                                      "acquisition": AcquisitionPolicy(raw["acquisition"])})
        path = Path(value["destination"])
        if path.is_symlink():
            raise ValidationError("automatic asset destination must not be a symlink")
        if path.exists():
            if not path.is_file() or path.stat().st_size != manifest.size_bytes or file_sha(path) != manifest.sha256:
                raise ValidationError("existing prepared asset differs; original left unchanged")
        else:
            pending.append((value, manifest, path))
    total = sum(manifest.size_bytes for _, manifest, _ in pending)
    if total > workflow.preparation.get("max_download_bytes", 512 * 1024 * 1024):
        raise ValidationError("missing method assets exceed preparation.max_download_bytes")
    if pending and not allow_network:
        raise ValidationError("missing pinned method assets need --allow-network")
    require_storage_capacity([(path.parent, manifest.size_bytes) for _, manifest, path in pending])
    for value, manifest, destination in pending:
        if manifest.acquisition is not AcquisitionPolicy.DOWNLOAD:
            raise ValidationError("this asset requires a user-provided or gated file")
        store = ArtifactStore(output / "asset-cache")
        try:
            path = store.verify(manifest)
        except Exception:
            path = store.path_for(manifest)
            if path.exists():
                raise ValidationError("interrupted asset cache differs; preserve it for inspection") from None
            hosts = tuple(value["allowed_hosts"]) or (urlsplit(manifest.source_uri).hostname,)
            path = Path(download_artifact_streamed(manifest, store,
                policy=DownloadPolicy(allow_network=True, allowed_hosts=hosts, max_bytes=manifest.size_bytes),
                transport=PinnedHTTPSDownloadTransport()))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError as error:
            import errno
            if error.errno != errno.EXDEV:
                raise
            import shutil
            import tempfile
            fd, temporary = tempfile.mkstemp(prefix=".agentinputlab-", dir=destination.parent)
            try:
                with path.open("rb") as source, os.fdopen(fd, "wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if file_sha(temporary) != manifest.sha256:
                    raise ValidationError("materialized method asset checksum differs")
                os.link(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        path.unlink()  # Destination is the verified retained copy.
    return {"assets": len(workflow.preparation.get("assets", ())), "downloaded_bytes": total, "model_calls": 0}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def selected_preparer(workflow):
    requested = workflow.preparation.get("method")
    if requested:
        return dict(requested)
    run = workflow.run
    if run.runtime.get("kind") in WORKER_KINDS and "driver_path" not in run.runtime:
        return {"kind": "worker", "options": {}}
    kind = AUTO_PREPARERS.get(run.release_id)
    if kind and (("bindings" not in run.parameters and kind in {"hyde", "query2doc", "longllmlingua"})
                 or (kind == "adaptive" and "artifacts" not in run.parameters)
                 or (kind == "qrag" and "artifacts" not in run.parameters)):
        return {"kind": kind, "options": {}}
    return None


def validate_preparer_options(workflow):
    selected = selected_preparer(workflow)
    if not selected:
        return {"required": False}
    module, function = PREPARERS[selected["kind"]]
    operation = getattr(importlib.import_module(module), function)
    options = dict(selected.get("options", {}))
    if "state" in inspect.signature(operation).parameters:
        options["state"] = None  # Supplied from a selected visible case during preparation.
    try:
        inspect.signature(operation).bind(workflow.run, output_dir=Path("unused"), **options)
    except TypeError as error:
        raise ValidationError(f"preparation.method.options for {selected['kind']}: {error}") from None
    return {"required": True, "kind": selected["kind"], "model_calls": 0}


def prepare_sources(workflow, output, *, allow_network):
    """Fetch only registry-pinned method sources, preserving existing paths."""
    from agentinputlab.registry import AlgorithmRegistry
    from agentinputlab.runtime_bootstrap import _materialize_source, runtime_source_plan
    run = workflow.run
    if not workflow.preparation.get("sources"):
        return run
    method, _ = AlgorithmRegistry.default().get_release(run.release_id)
    root = Path(workflow.preparation.get("source_root", str(output / "sources")))
    sources = runtime_source_plan(root, method_ids=[method.method_id], arena_ids=())
    if len(sources) != 1:
        raise ValidationError("selected method does not expose one pinned source recipe")
    source = sources[0]
    runtime = dict(run.runtime)
    key = "source_root" if runtime.get("kind") == "plan_and_budget_native" else "checkout"
    if runtime.get(key):
        source = replace(source, checkout=Path(runtime[key]))
    result = _materialize_source(source, allow_network=allow_network and not source.checkout.exists())
    if not result["ready"]:
        raise ValidationError(result["reason"])
    runtime[key] = str(source.checkout)
    return replace(run, runtime=runtime)


def prepare_worker_environment(workflow, run, output, *, python, environment, allow_network):
    """Install an isolated worker, never change the user's active Python env."""
    if run.runtime.get("kind") not in WORKER_KINDS or not python:
        return run
    from agentinputlab import __version__
    extra, modules = WORKER_KINDS[run.runtime["kind"]]
    # A prepared run already names its exact usable worker. Moving its result
    # directory is not a request to reinstall that environment.
    if (not environment and all(run.runtime.get(key) for key in
            ("driver_path", "driver_sha256", "python_executable_sha256", "compatibility_manifest_path"))):
        worker = Path(run.runtime["python_executable"])
        if file_sha(worker) != run.runtime["python_executable_sha256"]:
            raise ValidationError("bound worker interpreter changed")
        interpreter_readiness(worker, modules)
        return run
    wheel = workflow.preparation.get("toolkit_wheel")
    intent = {"python": str(Path(python).absolute()), "kind": run.runtime["kind"],
              "toolkit_version": __version__, "extra": extra, "wheel_sha256": file_sha(wheel) if wheel else None}
    from agentinputlab.benchmarks.cache import benchmark_cache_dir
    from agentinputlab.contracts.serialization import content_hash
    cache = benchmark_cache_dir(workflow.target.get("cache_dir"))
    root = Path(environment or cache / "worker-environments" / content_hash(intent)[:24]).absolute()
    receipt_path = root / "receipt.json"
    worker = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if root.exists():
        if root.is_symlink() or not (root / "intent.json").is_file() or read_json(root / "intent.json") != intent:
            raise ValidationError("worker environment belongs to a different preparation; choose another path")
        if receipt_path.is_file():
            proof = read_json(receipt_path)
            if proof.get("python_sha256") != file_sha(worker):
                raise ValidationError("prepared worker interpreter changed")
            interpreter_readiness(worker, modules)
            return replace(run, runtime={**run.runtime, "python_executable": str(worker)})
    else:
        if not allow_network:
            raise ValidationError("worker environment installation needs --allow-network")
        interpreter_readiness(python, ())
        root.mkdir(parents=True, mode=0o700)
        save(root / "intent.json", intent)
    if not allow_network:
        raise ValidationError("incomplete worker installation needs --allow-network to continue")
    from agentinputlab.arena_preparation import _pip_python, _run
    if not worker.exists():
        _run([str(python), "-I", "-m", "venv", "--without-pip", str(root / "venv")])
    package = "agentinputlab" + (f"[{extra}]" if extra else "") + "==" + __version__
    if wheel:
        package = "agentinputlab" + (f"[{extra}]" if extra else "") + " @ " + Path(wheel).as_uri()
    _run([_pip_python(root), "-m", "pip", "--isolated", "--python", str(worker), "install",
          "--disable-pip-version-check", "--no-input", "--no-cache-dir", package])
    observed = interpreter_readiness(worker, modules)
    if not receipt_path.exists():
        save(receipt_path, {"python_sha256": file_sha(worker), "verification": observed,
                            "model_calls": 0, "fit_performed": False})
    return replace(run, runtime={**run.runtime, "python_executable": str(worker)})


def prepare_method(workflow, run, output, *, state=None):
    selected = selected_preparer(replace(workflow, run=run))
    if not selected:
        return run
    kind = selected["kind"]
    module, function = PREPARERS[kind]
    operation = getattr(importlib.import_module(module), function)
    options = dict(selected.get("options", {}))
    if "state" in inspect.signature(operation).parameters:
        if state is None:
            raise ValidationError("this method preparation requires a selected optimizer-visible state")
        options["state"] = state
    # Validate helper options before creating a directory or acquiring assets.
    try:
        inspect.signature(operation).bind(run, output_dir=output / "method", **options)
    except TypeError as error:
        raise ValidationError(f"preparation.method.options for {kind}: {error}") from None
    path = output / "method"
    proof_path = path / "prepared.json"
    if proof_path.is_file():
        proof = read_json(proof_path)
    else:
        proof = operation(run, output_dir=path, **options)
    ready_path = Path(proof["run_path"])
    if file_sha(ready_path) != proof["run_sha256"]:
        raise ValidationError("prepared method configuration checksum differs")
    ready = RunConfig.from_file(ready_path)
    if ready.release_id != run.release_id or ready.budget != run.budget or ready.seed != run.seed:
        raise ValidationError("preparation changed method, budget or seed")
    return ready
