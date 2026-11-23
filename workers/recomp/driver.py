#!/usr/bin/env python3
"""Pinned, local-only JSONL inference worker for carriex/recomp.

The worker imports the upstream extractive scoring function directly and uses
the exact abstractive input/decode path recorded in the pinned summarization
script.  Checkpoints are user-mounted and hash-attested; no model or dataset is
downloaded, copied, or redistributed by AgentInputLab.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

UPSTREAM_COMMIT = "51d4432151efb3275257a9407dc71d1e5ec6634d"
EXTRACTIVE_RELEASE = "curate.recomp.extractive"
ABSTRACTIVE_RELEASE = "curate.recomp.abstractive"
EXTRACTIVE_PROTOCOL = "agentinputlab.recomp.extractive.v3"
ABSTRACTIVE_PROTOCOL = "agentinputlab.recomp.abstractive.v3"
MANIFEST_SCHEMA = "agentinputlab.recomp.compatibility.v1"
CHECKPOINT_SCHEMA = "agentinputlab.recomp.checkpoint.v1"
DRIVER_ID = "agentinputlab.recomp.pinned.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
DEFAULT_MANIFEST_SHA256 = "33c6773a062e61416b2b040f9239fc7e0b4150b11a14562353ea15338ce255be"
ABSTRACTIVE_INPUT_TEMPLATE = "Question: {question}\n Document: {passages}\n Summary: "
ABSTRACTIVE_INPUT_TEMPLATE_SHA256 = hashlib.sha256(
    ABSTRACTIVE_INPUT_TEMPLATE.encode("utf-8")
).hexdigest()
CANDIDATE_UNIT_KIND = "presegmented_sentence"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_TEXT_CHARS = 8 * 1024 * 1024
MAX_CANDIDATES = 4096
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DEVICE = re.compile(r"^(cpu|mps|cuda(?::[0-9]+)?)$")
_CONTAINER_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".safetensors")
_COMMON_PAYLOAD_FIELDS = frozenset(
    {
        "phase",
        "preflight_token",
        "checkpoint_license_acknowledged",
        "remaining_budget",
    }
)
_TOP_FIELDS = frozenset(
    {
        "method_id",
        "release_id",
        "upstream_commit",
        "state_hash",
        "payload",
        "dependency_fingerprints",
        "minimum_reservation",
        "protocol_version",
        "request_id",
    }
)
_EXTRACTIVE_FIELDS = _COMMON_PAYLOAD_FIELDS | frozenset(
    {
        "task_id",
        "query",
        "documents",
        "document_ids",
        "checkpoint_revision",
        "tokenizer_revision",
        "sentence_splitter_revision",
        "asset_manifest_hash",
        "retriever_artifact_hash",
        "input_artifact_hash",
        "candidate_unit_kind",
        "controls",
        "seed",
    }
)
_ABSTRACTIVE_FIELDS = _COMMON_PAYLOAD_FIELDS | frozenset(
    {
        "task_id",
        "query",
        "documents",
        "document_ids",
        "passages",
        "passages_hash",
        "checkpoint_revision",
        "tokenizer_revision",
        "input_template_hash",
        "asset_manifest_hash",
        "retriever_artifact_hash",
        "input_artifact_hash",
        "controls",
        "seed",
    }
)


class DriverError(RuntimeError):
    """Stable, secret-free worker failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.usage = dict(usage) if usage is not None else None


@dataclass(frozen=True, slots=True)
class Checkout:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    evidence_level: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    root: Path
    manifest_sha256: str
    mode: str
    checkpoint_revision: str
    tokenizer_revision: str
    tree_sha256: str
    file_count: int


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    request_id: str
    protocol_version: str
    mode: str
    payload: Mapping[str, Any]
    retriever_fingerprint: Mapping[str, Any]
    state_hash: str
    phase: str
    preflight_token: str
    remaining_budget: Mapping[str, Any]


ModelLoader = Callable[[str, Path, str], tuple[Any, Any]]


class _DiscardTextIO(io.TextIOBase):
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


class _CountingTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.input_tokens = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        encoded = self.tokenizer(*args, **kwargs)
        self.input_tokens += _token_count(encoded)
        return encoded

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _content_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _read_bytes(path: Path, *, label: str, limit: int = MAX_REQUEST_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise DriverError("ARTIFACT_INVALID", f"{label} must be a regular non-symlink file")
    try:
        if path.stat().st_size > limit:
            raise DriverError("ARTIFACT_INVALID", f"{label} exceeds its byte limit")
        return path.read_bytes()
    except DriverError:
        raise
    except OSError as exc:
        raise DriverError("ARTIFACT_INVALID", f"{label} is unreadable") from exc


def _load_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    raw = _read_bytes(path, label=label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("ARTIFACT_INVALID", f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise DriverError("ARTIFACT_INVALID", f"{label} must contain a JSON object")
    return value, _sha256_bytes(raw)


def _canonical_checkpoint_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DriverError("CHECKPOINT_INVALID", "checkpoint paths must be canonical POSIX text")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DriverError("CHECKPOINT_INVALID", "checkpoint paths must be canonical and relative")
    return value


def _checkpoint_tree(root: Path) -> tuple[tuple[str, Path], ...]:
    """Return the closed regular-file tree, rejecting links and special files."""

    if root.is_symlink() or not root.is_dir():
        raise DriverError(
            "CHECKPOINT_MISSING",
            "the user-mounted RECOMP checkpoint must be a regular directory, not a symlink",
        )
    rows: list[tuple[str, Path]] = []
    try:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in tuple(directories):
                candidate = current_path / name
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise DriverError(
                        "CHECKPOINT_TREE_NOT_CLOSED",
                        "RECOMP checkpoint tree contains a symlink",
                    )
                if not stat.S_ISDIR(mode):
                    raise DriverError(
                        "CHECKPOINT_TREE_NOT_CLOSED",
                        "RECOMP checkpoint tree contains a non-directory entry",
                    )
            for name in files:
                candidate = current_path / name
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise DriverError(
                        "CHECKPOINT_TREE_NOT_CLOSED",
                        "RECOMP checkpoint tree contains a symlink",
                    )
                if not stat.S_ISREG(mode):
                    raise DriverError(
                        "CHECKPOINT_TREE_NOT_CLOSED",
                        "RECOMP checkpoint tree contains a non-regular file",
                    )
                relative = candidate.relative_to(root).as_posix()
                _canonical_checkpoint_relative(relative)
                rows.append((relative, candidate))
    except DriverError:
        raise
    except OSError as exc:
        raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint tree is unreadable") from exc
    return tuple(sorted(rows, key=lambda item: item[0]))


def _hash_regular_file(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise DriverError("CHECKPOINT_TREE_NOT_CLOSED", "checkpoint file became a symlink")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise DriverError("CHECKPOINT_INVALID", "checkpoint file is unreadable") from exc
    if path.is_symlink() or not path.is_file():
        raise DriverError("CHECKPOINT_TREE_NOT_CLOSED", "checkpoint file identity changed")
    return digest.hexdigest(), size


def _sha_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise DriverError("INVALID_REQUEST", f"{label} must be a SHA-256 digest")
    return value


def _bounded_text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise DriverError("INVALID_REQUEST", f"{label} must be text")
    if len(value) > MAX_TEXT_CHARS:
        raise DriverError("INVALID_REQUEST", f"{label} exceeds its character limit")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("INVALID_REQUEST", f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result < 1:
        raise DriverError("INVALID_REQUEST", f"{label} must be positive")
    return result


def _git(root: Path, *arguments: str) -> str:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise DriverError("CHECKOUT_VERIFICATION_FAILED", "git verification failed") from exc
    if result.returncode != 0:
        raise DriverError("CHECKOUT_VERIFICATION_FAILED", "git verification failed")
    return result.stdout.strip()


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "evidence_level",
        "upstream",
        "source_files",
        "runtime_contract",
        "legacy_dependencies",
        "artifact_policy",
    }
    if set(manifest) != expected or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DriverError("MANIFEST_INVALID", "RECOMP compatibility manifest schema mismatch")
    if manifest.get("evidence_level") not in {
        "pinned_upstream_source",
        "fake_checkout_plumbing",
    }:
        raise DriverError("MANIFEST_INVALID", "RECOMP evidence level is invalid")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("commit") != UPSTREAM_COMMIT:
        raise DriverError("MANIFEST_INVALID", "RECOMP upstream identity mismatch")
    if upstream.get("license_spdx") != "MIT":
        raise DriverError("MANIFEST_INVALID", "RECOMP source license must be MIT")
    _sha_text(upstream.get("license_sha256"), label="upstream.license_sha256")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise DriverError("MANIFEST_INVALID", "RECOMP source hash mapping is empty")
    required_sources = {
        "README.md",
        "requirements.txt",
        "run_extractive_compressor.py",
        "train_hf_summarization_model.py",
    }
    if set(source_files) != required_sources:
        raise DriverError("MANIFEST_INVALID", "RECOMP source hash mapping is incomplete")
    if any(
        not isinstance(path, str) or not _HEX64.fullmatch(str(digest))
        for path, digest in source_files.items()
    ):
        raise DriverError("MANIFEST_INVALID", "RECOMP source hash entry is invalid")
    contract = manifest.get("runtime_contract")
    if not isinstance(contract, Mapping) or contract.get("extractive_symbol") != (
        "run_extractive_compressor.get_contriever_scores"
    ):
        raise DriverError("MANIFEST_INVALID", "RECOMP extractive contract is invalid")
    if (
        contract.get("extractive_top_k") != -1
        or contract.get("candidate_unit_kind") != CANDIDATE_UNIT_KIND
        or contract.get("abstractive_input_template") != ABSTRACTIVE_INPUT_TEMPLATE
        or contract.get("abstractive_decode") != "batch_decode_skip_special_tokens_strip"
    ):
        raise DriverError("MANIFEST_INVALID", "RECOMP runtime contract drifted")
    dependencies = manifest.get("legacy_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != {
        "required_versions",
        "extractive_unpinned_imports",
    }:
        raise DriverError("MANIFEST_INVALID", "RECOMP dependency contract is invalid")
    versions = dependencies.get("required_versions")
    if not isinstance(versions, Mapping):
        raise DriverError("MANIFEST_INVALID", "RECOMP version mapping is invalid")
    for name, version in versions.items():
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise DriverError("MANIFEST_INVALID", "RECOMP version entry is invalid")
    unpinned = dependencies.get("extractive_unpinned_imports")
    if not isinstance(unpinned, Sequence) or isinstance(unpinned, (str, bytes)):
        raise DriverError("MANIFEST_INVALID", "RECOMP unpinned import list is invalid")
    if not all(isinstance(name, str) and name for name in unpinned):
        raise DriverError("MANIFEST_INVALID", "RECOMP unpinned import entry is invalid")
    policy = manifest.get("artifact_policy")
    if not isinstance(policy, Mapping) or policy != {
        "checkpoint_license": "NOASSERTION",
        "bundle_checkpoint": False,
        "bundle_training_data": False,
        "local_files_only": True,
        "trust_remote_code": False,
    }:
        raise DriverError("MANIFEST_INVALID", "RECOMP artifact policy drifted")


def verify_checkout(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    allow_fixture_manifest: bool = False,
) -> Checkout:
    manifest, manifest_hash = _load_json(manifest_path, label="RECOMP compatibility manifest")
    if not allow_fixture_manifest and manifest_hash != DEFAULT_MANIFEST_SHA256:
        raise DriverError("WORKER_INTEGRITY_FAILURE", "RECOMP compatibility manifest hash mismatch")
    _validate_manifest(manifest)
    root = root.resolve()
    if not root.is_dir():
        raise DriverError("CHECKOUT_MISSING", "the pinned RECOMP checkout is missing")
    evidence = str(manifest["evidence_level"])
    if evidence == "pinned_upstream_source":
        if _git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
            raise DriverError("CHECKOUT_REVISION_MISMATCH", "RECOMP checkout revision mismatch")
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise DriverError("CHECKOUT_DIRTY", "RECOMP checkout must be clean")
    else:
        marker = _read_bytes(root / ".git" / "HEAD", label="fixture checkout marker")
        if marker.decode("utf-8").strip() != UPSTREAM_COMMIT:
            raise DriverError("CHECKOUT_REVISION_MISMATCH", "fixture checkout revision mismatch")
    upstream = manifest["upstream"]
    assert isinstance(upstream, Mapping)
    if (
        _sha256_bytes(_read_bytes(root / "LICENSE", label="upstream LICENSE"))
        != upstream["license_sha256"]
    ):
        raise DriverError("LICENSE_MISMATCH", "RECOMP MIT license hash mismatch")
    source_files = manifest["source_files"]
    assert isinstance(source_files, Mapping)
    for relative, expected in source_files.items():
        candidate = root / str(relative)
        try:
            candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise DriverError("MANIFEST_INVALID", "RECOMP source path escapes checkout") from exc
        if _sha256_bytes(_read_bytes(candidate, label=f"upstream source {relative}")) != expected:
            raise DriverError("SOURCE_MISMATCH", f"RECOMP source mismatch: {relative}")
    return Checkout(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        evidence_level=evidence,
    )


def verify_checkpoint(root: Path, manifest_path: Path) -> Checkpoint:
    manifest, digest = _load_json(manifest_path, label="RECOMP checkpoint manifest")
    if (
        set(manifest)
        != {
            "schema_version",
            "mode",
            "checkpoint_revision",
            "tokenizer_revision",
            "license",
            "files",
        }
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA
    ):
        raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint manifest schema mismatch")
    mode = manifest.get("mode")
    if mode not in {"extractive", "abstractive"}:
        raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint mode is invalid")
    if manifest.get("license") != "NOASSERTION":
        raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint license must remain NOASSERTION")
    checkpoint_revision = _bounded_text(
        manifest.get("checkpoint_revision"), label="checkpoint_revision"
    )
    tokenizer_revision = _bounded_text(
        manifest.get("tokenizer_revision"), label="tokenizer_revision"
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint file manifest is empty")
    canonical_files: dict[str, str] = {}
    for relative, expected in files.items():
        normalized = _canonical_checkpoint_relative(relative)
        if not _HEX64.fullmatch(str(expected)):
            raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint file entry is invalid")
        canonical_files[normalized] = str(expected)
    if not any(relative.lower().endswith(_WEIGHT_SUFFIXES) for relative in canonical_files):
        raise DriverError("CHECKPOINT_INVALID", "RECOMP checkpoint manifest omits model weights")
    if root.is_symlink():
        raise DriverError("CHECKPOINT_TREE_NOT_CLOSED", "checkpoint root must not be a symlink")
    root = root.resolve()
    tree = _checkpoint_tree(root)
    actual_paths = {relative for relative, _ in tree}
    declared_paths = set(canonical_files)
    if actual_paths != declared_paths:
        unlisted = sorted(actual_paths - declared_paths)
        missing = sorted(declared_paths - actual_paths)
        detail = "unlisted regular files" if unlisted else "manifest files missing from tree"
        if unlisted and missing:
            detail = "unlisted and missing checkpoint files"
        raise DriverError(
            "CHECKPOINT_TREE_NOT_CLOSED",
            f"RECOMP checkpoint tree closure failed: {detail}",
        )
    tree_records: list[Mapping[str, Any]] = []
    for relative, candidate in tree:
        actual_hash, size = _hash_regular_file(candidate)
        if actual_hash != canonical_files[relative]:
            raise DriverError("CHECKPOINT_MISMATCH", f"RECOMP checkpoint file mismatch: {relative}")
        tree_records.append({"path": relative, "sha256": actual_hash, "bytes": size})
    return Checkpoint(
        root=root,
        manifest_sha256=digest,
        mode=str(mode),
        checkpoint_revision=checkpoint_revision,
        tokenizer_revision=tokenizer_revision,
        tree_sha256=_content_hash(tree_records),
        file_count=len(tree_records),
    )


def _verify_dependencies(manifest: Mapping[str, Any], *, mode: str, device: str = "") -> None:
    legacy = manifest["legacy_dependencies"]
    assert isinstance(legacy, Mapping)
    versions = legacy["required_versions"]
    assert isinstance(versions, Mapping)
    for name, expected in versions.items():
        try:
            actual = importlib.metadata.version(str(name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise DriverError(
                "LEGACY_DEPENDENCY_MISSING",
                f"RECOMP requires {name}=={expected}",
            ) from exc
        # Official CPU/CUDA 11.7 wheels retain the pinned public version but
        # append a build tag. Keep this allowlist explicit: no newer patch or
        # arbitrary local suffix, and never admit a CPU-only wheel on CUDA.
        official_torch_build = name == "torch" and expected == "1.13.0" and (
            (actual == "1.13.0+cpu" and device == "cpu")
            or (
                actual == "1.13.0+cu117"
                and (device == "cpu" or re.fullmatch(r"cuda(?::[0-9]+)?", device) is not None)
            )
        )
        if actual != expected and not official_torch_build:
            raise DriverError(
                "LEGACY_DEPENDENCY_VERSION_MISMATCH",
                f"RECOMP requires {name}=={expected}",
            )
    if mode == "extractive":
        imports = legacy["extractive_unpinned_imports"]
        assert isinstance(imports, Sequence)
        for name in imports:
            try:
                importlib.import_module(str(name))
            except ModuleNotFoundError as exc:
                raise DriverError(
                    "LEGACY_DEPENDENCY_MISSING",
                    f"RECOMP extractive script requires the unpinned dependency {name}",
                ) from exc


def _installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _runtime_fingerprint(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    device: str,
    container_digest: str,
    custom_loader: bool,
) -> Mapping[str, Any]:
    legacy = manifest["legacy_dependencies"]
    assert isinstance(legacy, Mapping)
    names = {"torch", "transformers", "tokenizers", "safetensors"}
    if mode == "extractive":
        imports = legacy["extractive_unpinned_imports"]
        assert isinstance(imports, Sequence)
        names.update(str(name) for name in imports)
    dependencies = {name: _installed_version(name) for name in sorted(names)}
    return {
        "device": device,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "dependency_versions": dependencies,
        "container_digest": container_digest,
        "custom_model_loader": custom_loader,
    }


def _remaining_budget(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "model_calls",
        "context_tokens",
        "output_tokens",
        "cost_usd",
    }:
        raise DriverError("INVALID_REQUEST", "RECOMP remaining_budget schema mismatch")
    model_calls = _nonnegative_int(value.get("model_calls"), label="remaining model_calls")
    output_tokens = _nonnegative_int(value.get("output_tokens"), label="remaining output_tokens")
    context_tokens = value.get("context_tokens")
    if context_tokens is not None:
        context_tokens = _nonnegative_int(context_tokens, label="remaining context_tokens")
    cost = value.get("cost_usd")
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise DriverError("INVALID_REQUEST", "remaining cost_usd must be finite and non-negative")
    return {
        "model_calls": model_calls,
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "cost_usd": None if cost is None else float(cost),
    }


def _gate_known_budget(budget: Mapping[str, Any], *, output_tokens_required: int) -> None:
    if budget["model_calls"] < 1:
        raise DriverError("MODEL_BUDGET_EXHAUSTED", "RECOMP model-call budget is exhausted")
    if budget["output_tokens"] < output_tokens_required:
        raise DriverError(
            "OUTPUT_BUDGET_EXHAUSTED",
            "RECOMP output-token budget is below the declared generation maximum",
        )
    if budget["cost_usd"] is not None:
        raise DriverError(
            "COST_BUDGET_UNENFORCEABLE",
            "RECOMP local inference is unpriced; a cost cap must fail closed",
        )


def _gate_context_budget(budget: Mapping[str, Any], *, input_tokens: int) -> None:
    cap = budget["context_tokens"]
    if cap is not None and input_tokens > cap:
        raise DriverError(
            "CONTEXT_BUDGET_EXHAUSTED",
            "RECOMP tokenized input exceeds the remaining context-token budget",
        )


def _documents(value: Any, ids_value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("INVALID_REQUEST", "documents must be a sequence")
    if isinstance(ids_value, (str, bytes)) or not isinstance(ids_value, Sequence):
        raise DriverError("INVALID_REQUEST", "document_ids must be a sequence")
    ids = tuple(ids_value)
    if not all(isinstance(item, str) and item for item in ids):
        raise DriverError("INVALID_REQUEST", "document_ids must contain non-empty strings")
    if not ids or len(ids) > MAX_CANDIDATES or len(ids) != len(set(ids)):
        raise DriverError("INVALID_REQUEST", "document_ids are empty, duplicate, or excessive")
    documents: list[Mapping[str, Any]] = []
    expected_fields = {"text", "block_id", "source", "score", "metadata"}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise DriverError("INVALID_REQUEST", f"documents[{index}] has an invalid schema")
        text = _bounded_text(item.get("text"), label=f"documents[{index}].text")
        block_id = _bounded_text(item.get("block_id"), label=f"documents[{index}].block_id")
        if index >= len(ids) or block_id != ids[index]:
            raise DriverError("SOURCE_MAPPING_MISMATCH", "document IDs/order changed")
        if not isinstance(item.get("source"), str) or not isinstance(item.get("metadata"), Mapping):
            raise DriverError("INVALID_REQUEST", "document source/metadata is invalid")
        score = item.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise DriverError("INVALID_REQUEST", "document score is invalid")
        documents.append(
            {
                "text": text,
                "block_id": block_id,
                "source": item["source"],
                "score": score,
                "metadata": dict(item["metadata"]),
            }
        )
    if len(documents) != len(ids):
        raise DriverError("SOURCE_MAPPING_MISMATCH", "documents and IDs have different lengths")
    return tuple(documents)


def parse_request(value: Any) -> ParsedRequest:
    if not isinstance(value, Mapping) or set(value) != _TOP_FIELDS:
        raise DriverError("INVALID_REQUEST", "worker request fields do not match schema")
    if value.get("method_id") != "recomp" or value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise DriverError("IDENTITY_MISMATCH", "worker accepts only the pinned RECOMP identity")
    release = value.get("release_id")
    protocol = value.get("protocol_version")
    if release == EXTRACTIVE_RELEASE and protocol == EXTRACTIVE_PROTOCOL:
        mode = "extractive"
        fields = _EXTRACTIVE_FIELDS
    elif release == ABSTRACTIVE_RELEASE and protocol == ABSTRACTIVE_PROTOCOL:
        mode = "abstractive"
        fields = _ABSTRACTIVE_FIELDS
    else:
        raise DriverError("IDENTITY_MISMATCH", "unsupported RECOMP release/protocol")
    request_id = _sha_text(value.get("request_id"), label="request_id")
    state_hash = _sha_text(value.get("state_hash"), label="state_hash")
    payload = value.get("payload")
    dependencies = value.get("dependency_fingerprints")
    reservation = value.get("minimum_reservation")
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise DriverError("INVALID_REQUEST", "RECOMP payload fields do not match schema")
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies) != {"retriever"}
        or not isinstance(dependencies.get("retriever"), Mapping)
    ):
        raise DriverError("INVALID_REQUEST", "RECOMP requires one retriever fingerprint")
    if not isinstance(reservation, Mapping):
        raise DriverError("INVALID_REQUEST", "RECOMP reservation must be a mapping")
    try:
        expected_id = _content_hash({key: value[key] for key in value if key != "request_id"})
    except (TypeError, ValueError) as exc:
        raise DriverError("INVALID_REQUEST", "RECOMP request is not canonical JSON") from exc
    if request_id != expected_id:
        raise DriverError("REQUEST_ID_MISMATCH", "RECOMP request_id is not canonical")
    normalized = json.loads(_canonical_json(payload))
    retriever = json.loads(_canonical_json(dependencies["retriever"]))
    if normalized.get("checkpoint_license_acknowledged") is not True:
        raise DriverError(
            "CHECKPOINT_LICENSE_NOT_ACKNOWLEDGED",
            "RECOMP request must acknowledge the NOASSERTION checkpoint license",
        )
    phase = normalized.get("phase")
    preflight_token = normalized.get("preflight_token")
    if phase == "preflight":
        if preflight_token != "" or dict(reservation):
            raise DriverError(
                "INVALID_REQUEST",
                "RECOMP preflight cannot carry a token or reserve a model call",
            )
    elif phase == "execute":
        _sha_text(preflight_token, label="preflight_token")
        if dict(reservation) != {"model_calls": 1}:
            raise DriverError("INVALID_REQUEST", "RECOMP execute must reserve one model call")
    else:
        raise DriverError("INVALID_REQUEST", "RECOMP phase must be preflight or execute")
    budget = _remaining_budget(normalized.get("remaining_budget"))
    query = _bounded_text(normalized.get("query"), label="query")
    _bounded_text(normalized.get("task_id"), label="task_id", allow_empty=True)
    documents = _documents(normalized.get("documents"), normalized.get("document_ids"))
    _bounded_text(normalized.get("checkpoint_revision"), label="checkpoint_revision")
    _bounded_text(normalized.get("tokenizer_revision"), label="tokenizer_revision")
    _sha_text(normalized.get("asset_manifest_hash"), label="asset_manifest_hash")
    _sha_text(normalized.get("retriever_artifact_hash"), label="retriever_artifact_hash")
    _sha_text(normalized.get("input_artifact_hash"), label="input_artifact_hash")
    if normalized["retriever_artifact_hash"] != _content_hash(retriever):
        raise DriverError("DEPENDENCY_MISMATCH", "retriever artifact hash mismatch")
    _nonnegative_int(normalized.get("seed"), label="seed")
    controls = normalized.get("controls")
    if not isinstance(controls, Mapping):
        raise DriverError("INVALID_REQUEST", "controls must be an object")
    if mode == "extractive":
        if normalized.get("candidate_unit_kind") != CANDIDATE_UNIT_KIND:
            raise DriverError("SOURCE_MAPPING_MISMATCH", "candidate units are not fixed sentences")
        if set(controls) != {
            "max_sentences",
            "minimum_score",
            "output_order",
            "assembly_profile",
            "dataset",
        }:
            raise DriverError("INVALID_REQUEST", "extractive controls do not match schema")
        max_sentences = controls.get("max_sentences")
        if (
            isinstance(max_sentences, bool)
            or not isinstance(max_sentences, int)
            or (max_sentences != -1 and max_sentences < 1)
        ):
            raise DriverError("INVALID_REQUEST", "max_sentences must be -1 or positive")
        minimum_score = controls.get("minimum_score")
        if minimum_score is not None and (
            isinstance(minimum_score, bool)
            or not isinstance(minimum_score, (int, float))
            or not math.isfinite(float(minimum_score))
        ):
            raise DriverError("INVALID_REQUEST", "minimum_score must be finite or null")
        if controls.get("output_order") not in {"source_order", "score_descending"}:
            raise DriverError("INVALID_REQUEST", "output_order is invalid")
        profile = controls.get("assembly_profile")
        dataset = controls.get("dataset")
        if profile == "controlled":
            if dataset != "":
                raise DriverError("INVALID_REQUEST", "controlled assembly cannot claim a dataset")
        elif profile == "native_paper_qa":
            expected_top_n = {"nq": 1, "tqa": 1, "hotpotqa": 2}.get(dataset)
            if (
                expected_top_n is None
                or max_sentences != expected_top_n
                or minimum_score is not None
                or controls.get("output_order") != "score_descending"
            ):
                raise DriverError("INVALID_REQUEST", "paper-native top-N controls drifted")
        else:
            raise DriverError("INVALID_REQUEST", "assembly_profile is invalid")
        expected_input_hash = _content_hash(
            {
                "query": query,
                "documents": documents,
                "document_ids": normalized["document_ids"],
                "candidate_unit_kind": CANDIDATE_UNIT_KIND,
            }
        )
    else:
        if set(controls) != {
            "max_source_tokens",
            "max_target_tokens",
            "num_beams",
            "empty_output_policy",
            "input_assembly",
        }:
            raise DriverError("INVALID_REQUEST", "abstractive controls do not match schema")
        _positive_int(controls.get("max_source_tokens"), label="max_source_tokens")
        _positive_int(controls.get("max_target_tokens"), label="max_target_tokens")
        _positive_int(controls.get("num_beams"), label="num_beams")
        if controls.get("empty_output_policy") != "clear_context":
            raise DriverError("INVALID_REQUEST", "RECOMP empty-output policy drifted")
        assembly = controls.get("input_assembly")
        if assembly == "prejoined_official_artifact":
            if len(documents) != 1:
                raise DriverError(
                    "SOURCE_MAPPING_MISMATCH",
                    "prejoined official passages require exactly one context block",
                )
            expected_passages = str(documents[0]["text"])
        elif assembly == "controlled_concatenate_no_separator":
            expected_passages = "".join(str(item["text"]) for item in documents)
        else:
            raise DriverError("INVALID_REQUEST", "unsupported passages assembly")
        passages = _bounded_text(normalized.get("passages"), label="passages")
        if passages != expected_passages or normalized.get("passages_hash") != _content_hash(
            passages
        ):
            raise DriverError("SOURCE_MAPPING_MISMATCH", "abstractive passages binding mismatch")
        if normalized.get("input_template_hash") != ABSTRACTIVE_INPUT_TEMPLATE_SHA256:
            raise DriverError("SOURCE_MAPPING_MISMATCH", "abstractive input template drifted")
        expected_input_hash = _content_hash(
            {
                "query": query,
                "documents": documents,
                "document_ids": normalized["document_ids"],
                "passages": passages,
                "input_template_hash": ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
            }
        )
    if normalized["input_artifact_hash"] != expected_input_hash:
        raise DriverError("SOURCE_MAPPING_MISMATCH", "RECOMP input artifact hash mismatch")
    return ParsedRequest(
        request_id=request_id,
        protocol_version=str(protocol),
        mode=mode,
        payload=normalized,
        retriever_fingerprint=retriever,
        state_hash=state_hash,
        phase=str(phase),
        preflight_token=str(preflight_token),
        remaining_budget=budget,
    )


def _load_extractive_module(checkout: Checkout) -> ModuleType:
    path = checkout.root / "run_extractive_compressor.py"
    specification = importlib.util.spec_from_file_location(
        f"agentinputlab_recomp_{checkout.manifest_sha256}", path
    )
    if specification is None or specification.loader is None:
        raise DriverError("UPSTREAM_IMPORT_FAILURE", "cannot load RECOMP extractive script")
    module = importlib.util.module_from_spec(specification)
    discarded = _DiscardTextIO()
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        try:
            with contextlib.redirect_stdout(discarded):
                specification.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            raise DriverError(
                "LEGACY_DEPENDENCY_MISSING",
                f"RECOMP extractive script dependency is unavailable: {exc.name}",
            ) from exc
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_IMPORT_FAILURE",
                f"RECOMP extractive import failed: {type(exc).__name__}",
            ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    location = getattr(module, "__file__", None)
    if not isinstance(location, str) or Path(location).resolve() != path.resolve():
        raise DriverError("UPSTREAM_IMPORT_FAILURE", "RECOMP extractive module origin mismatch")
    if not callable(getattr(module, "get_contriever_scores", None)):
        raise DriverError("UPSTREAM_API_MISMATCH", "official RECOMP scoring API is missing")
    return module


def _default_model_loader(mode: str, checkpoint: Path, device: str) -> tuple[Any, Any]:
    # The source-pinned worker uses legacy Torch/Transformers. Their default bin
    # loader is not a weights-only boundary; never deserialize a mounted pickle
    # merely because its bytes have a checksum. Repacking must happen separately
    # with a modern restricted loader and a retained tensor-equality proof.
    safe_index = checkpoint / "model.safetensors.index.json"
    if not ((checkpoint / "model.safetensors").is_file() or safe_index.is_file()):
        raise DriverError(
            "CHECKPOINT_FORMAT_UNSAFE",
            "legacy RECOMP loading requires safetensors; convert and verify tensor equality outside the worker",
        )
    if any(
        path.is_file() and path.suffix.lower() in {".bin", ".pt", ".pth"}
        for path in checkpoint.rglob("*")
    ):
        raise DriverError(
            "CHECKPOINT_FORMAT_UNSAFE", "legacy worker mounts must exclude pickle weights"
        )
    if safe_index.is_file():
        index, _ = _load_json(safe_index, label="safetensors index")
        shards = index.get("weight_map")
        if not isinstance(shards, Mapping) or not shards:
            raise DriverError("CHECKPOINT_FORMAT_UNSAFE", "safetensors shard mapping is invalid")
        for name in shards.values():
            if (
                not isinstance(name, str)
                or not name.endswith(".safetensors")
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not (checkpoint / name).is_file()
            ):
                raise DriverError(
                    "CHECKPOINT_FORMAT_UNSAFE", "safetensors index references unsafe weights"
                )
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise DriverError("LEGACY_DEPENDENCY_MISSING", "transformers is unavailable") from exc
    try:
        # Transformers4.24 predates the use_safetensors keyword. Its actual
        # local loader selects safe weights iff this native availability gate
        # is true; the closed mount above makes pickle fallback impossible.
        if not transformers.utils.is_safetensors_available():
            raise DriverError(
                "LEGACY_DEPENDENCY_MISSING", "install safetensors in the isolated RECOMP worker"
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        model_type = (
            transformers.AutoModel if mode == "extractive" else transformers.AutoModelForSeq2SeqLM
        )
        model = model_type.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = model.to(device)
        # This worker performs frozen inference, never SFT/RL. The pinned
        # extractive function calls the model without a no_grad context;
        # eval() alone would still retain a gradient graph for its full batch.
        # Freezing parameters changes neither weights nor scoring/decoding.
        model.requires_grad_(False)
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
    except DriverError:
        raise
    except Exception as exc:
        raise DriverError(
            "CHECKPOINT_LOAD_FAILURE",
            f"RECOMP checkpoint load failed: {type(exc).__name__}",
        ) from exc
    return model, tokenizer


def _token_count(encoded: Any) -> int:
    mask = (
        encoded.get("attention_mask")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "attention_mask", None)
    )
    if mask is not None:
        total = getattr(mask, "sum", lambda: 0)()
        item = getattr(total, "item", None)
        return int(item() if callable(item) else total)
    input_ids = (
        encoded.get("input_ids")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "input_ids", None)
    )
    if input_ids is None:
        raise DriverError("USAGE_ACCOUNTING_FAILURE", "tokenizer omitted input_ids")
    try:
        shape = input_ids.shape
        return int(shape[0] * shape[1])
    except (AttributeError, IndexError, TypeError):
        if isinstance(input_ids, Sequence):
            return sum(len(row) if isinstance(row, Sequence) else 1 for row in input_ids)
    raise DriverError("USAGE_ACCOUNTING_FAILURE", "cannot count tokenizer input IDs")


def _output_token_count(outputs: Any, pad_token_id: int | None) -> int:
    try:
        row = outputs[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "generation returned no token IDs") from exc
    values = row.tolist() if callable(getattr(row, "tolist", None)) else list(row)
    return sum(1 for value in values if pad_token_id is None or value != pad_token_id)


def _usage(*, input_tokens: int, output_tokens: int = 0) -> dict[str, int]:
    return {
        "model_calls": 1,
        "retrieval_calls": 0,
        "unpriced_model_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _zero_usage() -> dict[str, int]:
    return {
        "model_calls": 0,
        "retrieval_calls": 0,
        "unpriced_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _preflight_input_tokens(request: ParsedRequest, tokenizer: Any) -> tuple[int, int]:
    payload = request.payload
    controls = payload["controls"]
    try:
        if request.mode == "extractive":
            encoded = tokenizer(
                [payload["query"]] + [item["text"] for item in payload["documents"]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            output_required = 0
        else:
            prompt = ABSTRACTIVE_INPUT_TEMPLATE.format(
                question=payload["query"], passages=payload["passages"]
            )
            encoded = tokenizer(
                [prompt],
                max_length=controls["max_source_tokens"],
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            output_required = int(controls["max_target_tokens"])
        input_tokens = _token_count(encoded)
    except DriverError:
        raise
    except Exception as exc:
        raise DriverError(
            "TOKEN_PREFLIGHT_FAILURE",
            f"RECOMP tokenizer preflight failed: {type(exc).__name__}",
        ) from exc
    return input_tokens, output_required


def _preflight_token(
    request: ParsedRequest,
    *,
    checkout: Checkout,
    checkpoint: Checkpoint,
    runtime_fingerprint_hash: str,
    input_tokens_required: int,
    output_tokens_required: int,
) -> str:
    stable_payload = {
        key: value
        for key, value in request.payload.items()
        if key not in {"phase", "preflight_token"}
    }
    return _content_hash(
        {
            "state_hash": request.state_hash,
            "mode": request.mode,
            "payload": stable_payload,
            "retriever_fingerprint": request.retriever_fingerprint,
            "compatibility_manifest_sha256": checkout.manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "checkpoint_tree_sha256": checkpoint.tree_sha256,
            "runtime_fingerprint_hash": runtime_fingerprint_hash,
            "input_tokens_required": input_tokens_required,
            "output_tokens_required": output_tokens_required,
        }
    )


class _ForwardBudgetGate:
    """Delegate the official scorer's model call only after its tokenizer is observed."""

    def __init__(
        self,
        model: Any,
        tokenizer: _CountingTokenizer,
        *,
        budget: Mapping[str, Any],
        expected_input_tokens: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._budget = budget
        self._expected_input_tokens = expected_input_tokens
        self.forward_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._tokenizer.input_tokens != self._expected_input_tokens:
            raise DriverError(
                "TOKEN_PREFLIGHT_MISMATCH",
                "RECOMP scorer token count changed after preflight",
            )
        _gate_context_budget(self._budget, input_tokens=self._tokenizer.input_tokens)
        self.forward_calls += 1
        if self.forward_calls > 1:
            raise DriverError("MODEL_CALL_BUDGET_EXCEEDED", "RECOMP scorer called the model twice")
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


@dataclass(slots=True)
class RECOMPDriver:
    checkout_path: Path
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    device: str = "cpu"
    manifest_path: Path = DEFAULT_MANIFEST
    model_loader: ModelLoader | None = None
    allow_fixture_manifest: bool = False
    checkpoint_license_acknowledged: bool = False
    container_digest: str = ""

    def __post_init__(self) -> None:
        if not _DEVICE.fullmatch(self.device):
            raise DriverError("CONFIGURATION_INVALID", "RECOMP device is invalid")
        if self.checkpoint_license_acknowledged is not True:
            raise DriverError(
                "CHECKPOINT_LICENSE_NOT_ACKNOWLEDGED",
                "RECOMP worker startup must acknowledge the NOASSERTION checkpoint license",
            )
        if self.container_digest and not _CONTAINER_DIGEST.fullmatch(self.container_digest):
            raise DriverError(
                "CONFIGURATION_INVALID", "RECOMP container digest must be sha256:<64 hex>"
            )

    def fingerprint(self) -> Mapping[str, Any]:
        checkout = verify_checkout(
            self.checkout_path,
            manifest_path=self.manifest_path,
            allow_fixture_manifest=self.allow_fixture_manifest,
        )
        checkpoint = verify_checkpoint(self.checkpoint_path, self.checkpoint_manifest_path)
        runtime = _runtime_fingerprint(
            checkout.manifest,
            mode=checkpoint.mode,
            device=self.device,
            container_digest=self.container_digest,
            custom_loader=self.model_loader is not None,
        )
        return {
            "worker_id": DRIVER_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "compatibility_manifest_sha256": checkout.manifest_sha256,
            "asset_manifest_hash": checkpoint.manifest_sha256,
            "checkpoint_revision": checkpoint.checkpoint_revision,
            "tokenizer_revision": checkpoint.tokenizer_revision,
            "mode": checkpoint.mode,
            "execution_evidence": checkout.evidence_level,
            "checkpoint_license": "NOASSERTION",
            "checkpoint_license_acknowledged_at_startup": True,
            "checkpoint_tree_sha256": checkpoint.tree_sha256,
            "checkpoint_file_count": checkpoint.file_count,
            "runtime_fingerprint": runtime,
            "runtime_fingerprint_hash": _content_hash(runtime),
        }

    def call(self, raw_request: Mapping[str, Any]) -> Mapping[str, Any]:
        request = parse_request(raw_request)
        checkout = verify_checkout(
            self.checkout_path,
            manifest_path=self.manifest_path,
            allow_fixture_manifest=self.allow_fixture_manifest,
        )
        checkpoint = verify_checkpoint(self.checkpoint_path, self.checkpoint_manifest_path)
        payload = request.payload
        if checkpoint.mode != request.mode:
            raise DriverError("DEPENDENCY_MISMATCH", "checkpoint mode does not match release")
        if (
            checkpoint.manifest_sha256 != payload["asset_manifest_hash"]
            or checkpoint.checkpoint_revision != payload["checkpoint_revision"]
            or checkpoint.tokenizer_revision != payload["tokenizer_revision"]
        ):
            raise DriverError("DEPENDENCY_MISMATCH", "checkpoint identity does not match request")
        output_required = (
            int(payload["controls"]["max_target_tokens"]) if request.mode == "abstractive" else 0
        )
        _gate_known_budget(request.remaining_budget, output_tokens_required=output_required)
        loader = self.model_loader
        if loader is None:
            _verify_dependencies(checkout.manifest, mode=request.mode, device=self.device)
            loader = _default_model_loader
        runtime_fingerprint = _runtime_fingerprint(
            checkout.manifest,
            mode=request.mode,
            device=self.device,
            container_digest=self.container_digest,
            custom_loader=self.model_loader is not None,
        )
        runtime_fingerprint_hash = _content_hash(runtime_fingerprint)
        try:
            model, tokenizer = loader(request.mode, checkpoint.root, self.device)
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "CHECKPOINT_LOAD_FAILURE",
                f"RECOMP model loader failed: {type(exc).__name__}",
            ) from exc
        input_required, preflight_output_required = _preflight_input_tokens(request, tokenizer)
        if preflight_output_required != output_required:
            raise DriverError("WORKER_INTEGRITY_FAILURE", "RECOMP output reservation drifted")
        _gate_context_budget(request.remaining_budget, input_tokens=input_required)
        decision_token = _preflight_token(
            request,
            checkout=checkout,
            checkpoint=checkpoint,
            runtime_fingerprint_hash=runtime_fingerprint_hash,
            input_tokens_required=input_required,
            output_tokens_required=output_required,
        )
        common_payload = {
            "checkpoint_revision": checkpoint.checkpoint_revision,
            "tokenizer_revision": checkpoint.tokenizer_revision,
            "asset_manifest_hash": checkpoint.manifest_sha256,
            "checkpoint_tree_sha256": checkpoint.tree_sha256,
            "checkpoint_file_count": checkpoint.file_count,
            "checkpoint_license": "NOASSERTION",
            "checkpoint_license_acknowledged": True,
            "source_document_ids": list(payload["document_ids"]),
            "retriever_artifact_hash": payload["retriever_artifact_hash"],
            "input_artifact_hash": payload["input_artifact_hash"],
            "compatibility_manifest_sha256": checkout.manifest_sha256,
            "execution_evidence": checkout.evidence_level,
            "remaining_budget_hash": _content_hash(request.remaining_budget),
            "input_tokens_required": input_required,
            "output_tokens_required": output_required,
            "preflight_token": decision_token,
            "runtime_fingerprint": runtime_fingerprint,
            "runtime_fingerprint_hash": runtime_fingerprint_hash,
        }
        if request.phase == "preflight":
            return {
                "status": "ok",
                "request_id": request.request_id,
                "upstream_commit": UPSTREAM_COMMIT,
                "protocol_version": request.protocol_version,
                "payload": common_payload,
                "usage": _zero_usage(),
            }
        if request.preflight_token != decision_token:
            raise DriverError(
                "PREFLIGHT_TOKEN_MISMATCH",
                "RECOMP execute request does not match the frozen preflight",
            )
        if request.mode == "extractive":
            result_payload, usage = self._run_extractive(
                request,
                checkout,
                checkpoint,
                model,
                tokenizer,
                expected_input_tokens=input_required,
                budget=request.remaining_budget,
            )
        else:
            result_payload, usage = self._run_abstractive(
                request,
                checkout,
                checkpoint,
                model,
                tokenizer,
                expected_input_tokens=input_required,
                budget=request.remaining_budget,
            )
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": request.protocol_version,
            "payload": {
                **result_payload,
                **common_payload,
            },
            "usage": usage,
        }

    def _run_extractive(
        self,
        request: ParsedRequest,
        checkout: Checkout,
        checkpoint: Checkpoint,
        model: Any,
        tokenizer: Any,
        *,
        expected_input_tokens: int,
        budget: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        module = _load_extractive_module(checkout)
        payload = request.payload
        documents = payload["documents"]
        counting = _CountingTokenizer(tokenizer)
        gated_model = _ForwardBudgetGate(
            model,
            counting,
            budget=budget,
            expected_input_tokens=expected_input_tokens,
        )
        data_row = {
            "query": payload["query"],
            "retrieved_docs": [{"text": item["text"]} for item in documents],
        }
        discarded = _DiscardTextIO()
        try:
            with contextlib.redirect_stdout(discarded):
                scores = module.get_contriever_scores(
                    gated_model,
                    counting,
                    data_row,
                    self.device,
                    top_k=-1,
                )
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_RUNTIME_FAILURE",
                f"official RECOMP scorer failed: {type(exc).__name__}",
                consumed=True,
                usage=_usage(input_tokens=counting.input_tokens),
            ) from exc
        if counting.input_tokens != expected_input_tokens:
            raise DriverError(
                "TOKEN_PREFLIGHT_MISMATCH",
                "RECOMP scorer token count changed after preflight",
                consumed=gated_model.forward_calls > 0,
                usage=(
                    _usage(input_tokens=counting.input_tokens)
                    if gated_model.forward_calls > 0
                    else None
                ),
            )
        _gate_context_budget(budget, input_tokens=counting.input_tokens)
        if gated_model.forward_calls != 1:
            raise DriverError(
                "USAGE_ACCOUNTING_MISMATCH",
                "RECOMP scorer must execute exactly one model forward",
            )
        if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "RECOMP scorer returned non-sequence")
        if len(scores) != len(documents):
            raise DriverError(
                "SCORE_LENGTH_MISMATCH",
                "RECOMP scorer output length differs from fixed candidate units",
                consumed=True,
                usage=_usage(input_tokens=counting.input_tokens),
            )
        rows: list[dict[str, Any]] = []
        for index, (document, score) in enumerate(zip(documents, scores, strict=True)):
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise DriverError("UPSTREAM_OUTPUT_INVALID", "RECOMP scorer emitted invalid score")
            text = str(document["text"])
            rows.append(
                {
                    "sentence_id": document["block_id"],
                    "document_id": document["block_id"],
                    "document_index": index,
                    "sentence_index": 0,
                    "char_start": 0,
                    "char_end": len(text),
                    "text": text,
                    "score": float(score),
                }
            )
        return (
            {
                "sentence_splitter_revision": payload["sentence_splitter_revision"],
                "candidate_unit_kind": CANDIDATE_UNIT_KIND,
                "sentence_scores": rows,
                "scoring_api": "run_extractive_compressor.get_contriever_scores(top_k=-1)",
            },
            _usage(input_tokens=counting.input_tokens),
        )

    def _run_abstractive(
        self,
        request: ParsedRequest,
        checkout: Checkout,
        checkpoint: Checkpoint,
        model: Any,
        tokenizer: Any,
        *,
        expected_input_tokens: int,
        budget: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        del checkout, checkpoint
        payload = request.payload
        controls = payload["controls"]
        prompt = ABSTRACTIVE_INPUT_TEMPLATE.format(
            question=payload["query"],
            passages=payload["passages"],
        )
        counting = _CountingTokenizer(tokenizer)
        try:
            encoded = counting(
                [prompt],
                max_length=controls["max_source_tokens"],
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            to_device = getattr(encoded, "to", None)
            if callable(to_device):
                encoded = to_device(self.device)
            if counting.input_tokens != expected_input_tokens:
                raise DriverError(
                    "TOKEN_PREFLIGHT_MISMATCH",
                    "RECOMP generation token count changed after preflight",
                )
            _gate_context_budget(budget, input_tokens=counting.input_tokens)
            _gate_known_budget(
                budget,
                output_tokens_required=int(controls["max_target_tokens"]),
            )
            model_inputs = (
                dict(encoded)
                if isinstance(encoded, Mapping)
                else {
                    "input_ids": encoded.input_ids,
                    "attention_mask": encoded.attention_mask,
                }
            )
            outputs = model.generate(
                **model_inputs,
                max_length=controls["max_target_tokens"],
                num_beams=controls["num_beams"],
            )
            decoded = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_RUNTIME_FAILURE",
                f"official RECOMP generation path failed: {type(exc).__name__}",
                consumed=True,
                usage=_usage(input_tokens=counting.input_tokens),
            ) from exc
        if (
            not isinstance(decoded, Sequence)
            or isinstance(decoded, (str, bytes))
            or len(decoded) != 1
        ):
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "RECOMP generation returned invalid batch")
        raw_generation = decoded[0]
        if not isinstance(raw_generation, str):
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "RECOMP generation is not text")
        summary = raw_generation.strip()
        output_tokens = _output_token_count(outputs, getattr(tokenizer, "pad_token_id", None))
        return (
            {
                "input_template_hash": ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
                "passages_hash": payload["passages_hash"],
                "prompt_hash": _sha256_bytes(prompt.encode("utf-8")),
                "applied": True,
                "summary": summary,
                "raw_generation": raw_generation,
                "selective_augmentation": not summary,
                "parse_status": "generated_empty" if not summary else "generated_text",
                "generation_api": "AutoModelForSeq2SeqLM.generate+batch_decode(strip)",
            },
            _usage(input_tokens=counting.input_tokens, output_tokens=output_tokens),
        )


def _failure_response(exc: DriverError, *, request_id: str = "") -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "status": "error",
        "failure_code": exc.code,
        "message": exc.public_message,
        "consumed": exc.consumed,
    }
    if request_id:
        result["request_id"] = request_id
    if exc.usage is not None:
        result["usage"] = exc.usage
    return result


def _read_request() -> Any:
    raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise DriverError("REQUEST_TOO_LARGE", "RECOMP request exceeds byte limit")
    if not raw or sys.stdin.buffer.read(1):
        raise DriverError("INVALID_REQUEST", "RECOMP worker accepts exactly one JSON line")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("INVALID_REQUEST", "RECOMP request is not valid JSON") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--container-digest", default="")
    parser.add_argument(
        "--acknowledge-noassertion-checkpoint",
        action="store_true",
        help="explicitly acknowledge that mounted checkpoint assets have NOASSERTION license",
    )
    parser.add_argument("--smoke-import", choices=("extractive", "abstractive"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_id = ""
    try:
        checkout = verify_checkout(args.checkout)
        if args.smoke_import:
            if not _DEVICE.fullmatch(args.device):
                raise DriverError("CONFIGURATION_INVALID", "RECOMP device is invalid")
            if args.container_digest and not _CONTAINER_DIGEST.fullmatch(args.container_digest):
                raise DriverError(
                    "CONFIGURATION_INVALID", "RECOMP container digest must be sha256:<64 hex>"
                )
            _verify_dependencies(checkout.manifest, mode=args.smoke_import, device=args.device)
            importlib.import_module("torch")
            importlib.import_module("tokenizers")
            importlib.import_module("transformers")
            if args.smoke_import == "extractive":
                _load_extractive_module(checkout)
            runtime = _runtime_fingerprint(
                checkout.manifest,
                mode=args.smoke_import,
                device=args.device,
                container_digest=args.container_digest,
                custom_loader=False,
            )
            response: Mapping[str, Any] = {
                "status": "ok",
                "upstream_commit": UPSTREAM_COMMIT,
                "mode": args.smoke_import,
                "execution_evidence": checkout.evidence_level,
                "compatibility_manifest_sha256": checkout.manifest_sha256,
                "runtime_fingerprint": runtime,
                "runtime_fingerprint_hash": _content_hash(runtime),
            }
        else:
            if args.checkpoint is None or args.checkpoint_manifest is None:
                raise DriverError(
                    "CONFIGURATION_MISSING",
                    "normal inference requires --checkpoint and --checkpoint-manifest",
                )
            if not args.acknowledge_noassertion_checkpoint:
                raise DriverError(
                    "CHECKPOINT_LICENSE_NOT_ACKNOWLEDGED",
                    "normal inference requires --acknowledge-noassertion-checkpoint",
                )
            raw_request = _read_request()
            if isinstance(raw_request, Mapping) and isinstance(raw_request.get("request_id"), str):
                request_id = str(raw_request["request_id"])
            response = RECOMPDriver(
                checkout_path=args.checkout,
                checkpoint_path=args.checkpoint,
                checkpoint_manifest_path=args.checkpoint_manifest,
                device=args.device,
                container_digest=args.container_digest,
                checkpoint_license_acknowledged=True,
            ).call(raw_request)
    except DriverError as exc:
        response = _failure_response(exc, request_id=request_id)
    except Exception as exc:  # pragma: no cover - last-resort JSON containment
        response = _failure_response(
            DriverError("WORKER_INTERNAL_FAILURE", f"RECOMP failed closed: {type(exc).__name__}"),
            request_id=request_id,
        )
    sys.stdout.write(_canonical_json(response) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
