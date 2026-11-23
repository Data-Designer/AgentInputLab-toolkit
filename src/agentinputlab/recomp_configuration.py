"""File binding for the existing isolated, source-pinned RECOMP compressor."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path

from agentinputlab.configuration import _known
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker


def prepare(config, output_dir):
    """Derive immutable method metadata from verified local assets, without fitting."""
    from workers.recomp import driver

    output = Path(output_dir).absolute()
    if output.exists():
        raise ValidationError("RECOMP preparation requires a new output directory")
    if config.runtime.get("kind") != "recomp_abstractive_worker":
        raise ValidationError("RECOMP preparation requires recomp_abstractive_worker")
    runtime(config, load=False)
    if not config.retriever:
        raise ValidationError("RECOMP preparation requires an explicit upstream retriever")
    if config.release_id == "curate.recomp.selective" and not config.backend:
        raise ValidationError("RECOMP selective preparation requires a reader backend")
    settings = dict(config.runtime)
    for key in ("checkout", "checkpoint_root", "checkpoint_manifest_path", "python_executable"):
        settings[key] = str(Path(settings[key]).absolute())
    try:
        source = driver.verify_checkout(Path(settings["checkout"]))
        checkpoint = driver.verify_checkpoint(Path(settings["checkpoint_root"]),
                                              Path(settings["checkpoint_manifest_path"]))
    except driver.DriverError as exc:
        raise ArtifactError(f"RECOMP asset preparation failed: {exc}") from exc
    if checkpoint.mode != "abstractive":
        raise ArtifactError("RECOMP preparation requires an abstractive checkpoint")
    derived = {"checkpoint_revision": checkpoint.checkpoint_revision,
               "tokenizer_revision": checkpoint.tokenizer_revision,
               "asset_manifest_hash": checkpoint.manifest_sha256}
    if config.release_id == "curate.recomp.abstractive":
        derived["input_template_hash"] = driver.ABSTRACTIVE_INPUT_TEMPLATE_SHA256
    parameters = dict(config.parameters)
    for key, value in derived.items():
        if key in parameters and parameters[key] != value:
            raise ValidationError(f"RECOMP preparation {key} conflicts with the verified asset")
        parameters[key] = value
    ready = replace(config, parameters=parameters, runtime=settings)
    ready.validate()
    ready.create_intervention()
    runtime(ready, load=True)  # Checks the executable and current manifest, but never calls it.
    output.mkdir(parents=True)
    run_path = output / "run.json"
    with run_path.open("x") as stream:
        json.dump(ready.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"release_id": ready.release_id, "run_path": str(run_path),
             "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
             "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
             "checkpoint_tree_sha256": checkpoint.tree_sha256,
             "checkpoint_file_count": checkpoint.file_count,
             "compatibility_manifest_sha256": source.manifest_sha256,
             "source_evidence": source.evidence_level,
             "model_calls": 0, "fit_performed": False,
             "worker_dependencies_verified": False}
    with (output / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof


def runtime(config, *, load):
    if config.release_id not in {"curate.recomp.abstractive", "curate.recomp.selective"}:
        raise ValidationError("recomp_abstractive_worker is supported only by RECOMP abstractive/selective")
    value = dict(config.runtime)
    _known(value, {"kind", "checkout", "checkpoint_root", "checkpoint_manifest_path",
                   "python_executable", "device", "timeout_seconds"}, "RECOMP runtime")
    for key in ("checkout", "checkpoint_root", "checkpoint_manifest_path", "python_executable"):
        if not isinstance(value.get(key), str) or not value[key].strip() or "\x00" in value[key]:
            raise ValidationError(f"RECOMP runtime.{key} requires a local path")
    device = value.get("device", "cpu")
    if not isinstance(device, str) or re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device) is None:
        raise ValidationError("RECOMP device must be cpu, mps or cuda with an optional index")
    timeout = value.get("timeout_seconds", 900)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValidationError("RECOMP timeout_seconds must be positive and finite")
    if config.parameters.get("asset_license_acknowledged") is not True:
        raise ValidationError("RECOMP requires explicit asset license acknowledgement")
    if not load:
        return None

    # Use this installation's driver, not a user-supplied shell or Python factory.
    # The worker itself verifies source, dependency versions and all checkpoint
    # files before preflight/inference; constructing this binding performs no fit.
    from workers.recomp import driver

    executable = Path(value["python_executable"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValidationError("RECOMP python_executable is missing or not executable")
    try:
        with Path(value["checkpoint_manifest_path"]).open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
    except OSError as exc:
        raise ArtifactError("RECOMP checkpoint manifest is not readable") from exc
    if len(raw) > 1024 * 1024:
        raise ArtifactError("RECOMP checkpoint manifest exceeds 1 MiB")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config.parameters.get("asset_manifest_hash"):
        raise ArtifactError("RECOMP checkpoint manifest checksum mismatch")
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ArtifactError("RECOMP checkpoint manifest is not JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("mode") != "abstractive":
        raise ArtifactError("RECOMP requires an abstractive checkpoint manifest")
    for key in ("checkpoint_revision", "tokenizer_revision"):
        if manifest.get(key) != config.parameters.get(key):
            raise ArtifactError(f"RECOMP manifest {key} differs from method parameters")
    policy = driver.DEFAULT_MANIFEST.read_bytes()
    policy_sha = hashlib.sha256(policy).hexdigest()
    if policy_sha != driver.DEFAULT_MANIFEST_SHA256:
        raise ArtifactError("bundled RECOMP compatibility manifest checksum mismatch")
    environment = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "TOKENIZERS_PARALLELISM": "false"}
    if device in {"cpu", "mps"}:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return JsonLineSubprocessWorker(
        command=(str(executable), str(Path(driver.__file__).resolve()), "--checkout", value["checkout"],
                 "--checkpoint", value["checkpoint_root"], "--checkpoint-manifest",
                 value["checkpoint_manifest_path"], "--device", device,
                 "--acknowledge-noassertion-checkpoint"),
        worker_id="recomp-abstractive-file",
        identity={"driver_sha256": hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),
                  "compatibility_manifest_sha256": policy_sha,
                  "checkpoint_manifest_sha256": digest,
                  "execution_evidence": "pinned_upstream_source"},
        timeout_seconds=timeout, environment=environment,
        max_input_bytes=driver.MAX_REQUEST_BYTES,
    )
