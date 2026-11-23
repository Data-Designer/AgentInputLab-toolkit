#!/usr/bin/env python3
"""Quarantined pinned-source inference worker for Rewrite-Retrieve-Read.

The unlicensed upstream tarball is verified and security-scanned but never
extracted or imported.  Inference follows the audited rewrite -> retrieve ->
snippet -> frozen-reader trajectory using user-mounted prompts, search artifacts,
and checkpoints.  PPO checkpoints are inference-only; this worker cannot train.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

UPSTREAM_COMMIT = "6bab6243e60c7f3e526b6ee44f95381b1024d471"
METHOD_ID = "rrr"
RELEASE_IDS = {
    "prompt": "rewrite.rrr.prompt_external",
    "warmup_checkpoint": "rewrite.rrr.warmup_external",
    "ppo_checkpoint": "rewrite.rrr.ppo_external",
}
PROTOCOL_VERSION = "agentinputlab.rrr.inference.v2"
MANIFEST_SCHEMA = "agentinputlab.rrr.compatibility.v1"
CHECKPOINT_SCHEMA = "agentinputlab.rrr.checkpoint.v1"
SEARCH_SCHEMA = "agentinputlab.rrr.search-cache.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
DEFAULT_MANIFEST_SHA256 = "83cb80359eadbed1b8d115e60f22a7dd26f16424aa71ff1f37350ba40ef953ff"
DRIVER_ID = "agentinputlab.rrr.quarantined.v1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^(?:|[0-9]{4}-[0-9]{2}-[0-9]{2})$")
_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?i)(?:api[_-]?key|subscription[_-]?key|authorization|bearer|azure|"
    rb"client[_-]?secret|password)\s*[:=]"
)
_ROOT_FIELDS = frozenset(
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
_PAYLOAD_FIELDS = frozenset(
    {
        "task_id",
        "query",
        "instruction",
        "messages",
        "controls",
        "prompt_hash",
        "reader_prompt_hash",
        "search_artifact_hash",
        "corpus_revision",
        "index_revision",
        "rewriter_checkpoint_revision",
        "reader_checkpoint_revision",
        "source_license_acknowledged",
        "checkpoint_license_acknowledged",
        "backend_artifact_hash",
        "retriever_artifact_hash",
        "input_artifact_hash",
        "seed",
    }
)
_CONTROL_FIELDS = frozenset(
    {
        "max_queries",
        "top_k",
        "query_parser",
        "search_mode",
        "rewriter_profile",
        "search_window_start",
        "search_window_end",
        "max_rewriter_tokens",
        "max_reader_tokens",
        "max_input_chars",
    }
)
_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".safetensors")


class DriverError(RuntimeError):
    """Stable error that never includes source or credential contents."""

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
class SourceEvidence:
    manifest_sha256: str
    evidence_level: str
    credential_risk_count: int


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class SearchDocument:
    document_id: str
    title: str
    text: str
    score: float
    metadata: Mapping[str, Any]


class Generator(Protocol):
    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> Generation: ...

    def fingerprint(self) -> Mapping[str, Any]: ...


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        window_start: str,
        window_end: str,
    ) -> tuple[SearchDocument, ...]: ...

    def fingerprint(self) -> Mapping[str, Any]: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(value: Any) -> str:
    return _sha_bytes(_canonical_json(value).encode("utf-8"))


def _sha_file(path: Path, *, limit: int) -> str:
    if path.is_symlink() or not path.is_file():
        raise DriverError("ARTIFACT_INVALID", "artifact must be a regular non-symlink file")
    if path.stat().st_size > limit:
        raise DriverError("ARTIFACT_TOO_LARGE", "artifact exceeds its byte limit")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bytes(path: Path, *, label: str, limit: int = MAX_MEMBER_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise DriverError("ARTIFACT_INVALID", f"{label} must be a regular non-symlink file")
    if path.stat().st_size > limit:
        raise DriverError("ARTIFACT_TOO_LARGE", f"{label} exceeds its byte limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DriverError("ARTIFACT_INVALID", f"{label} is unreadable") from exc


def _load_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    raw = _read_bytes(path, label=label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("ARTIFACT_INVALID", f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise DriverError("ARTIFACT_INVALID", f"{label} must contain an object")
    return value, _sha_bytes(raw)


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise DriverError("INVALID_REQUEST", f"{label} must be a SHA-256 digest")
    return value


def _text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise DriverError("INVALID_REQUEST", f"{label} must be text")
    if len(value) > MAX_TEXT_CHARS:
        raise DriverError("INVALID_REQUEST", f"{label} exceeds its character limit")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverError("INVALID_REQUEST", f"{label} must be positive")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("INVALID_REQUEST", f"{label} must be non-negative")
    return value


def _safe_member(name: str, root: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
        raise DriverError("SOURCE_ARCHIVE_UNSAFE", "source archive contains an unsafe path")
    relative = PurePosixPath(*path.parts[1:])
    if not relative.parts:
        return ""
    return str(relative)


def _validate_manifest(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "evidence_level",
        "upstream",
        "source_files",
        "required_symbols",
        "credential_risk_paths",
        "runtime_contract",
        "artifact_policy",
    }
    if set(value) != expected or value.get("schema_version") != MANIFEST_SCHEMA:
        raise DriverError("MANIFEST_INVALID", "RRR compatibility manifest schema mismatch")
    if value.get("evidence_level") not in {
        "pinned_tarball_compatible_chain",
        "fake_checkout_plumbing",
    }:
        raise DriverError("MANIFEST_INVALID", "RRR evidence level is invalid")
    upstream = value.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("commit") != UPSTREAM_COMMIT:
        raise DriverError("MANIFEST_INVALID", "RRR upstream revision mismatch")
    if (
        upstream.get("license_spdx") != "NOASSERTION"
        or upstream.get("license_file_present") is not False
    ):
        raise DriverError("MANIFEST_INVALID", "RRR license must remain absent/NOASSERTION")
    _sha(upstream.get("tarball_sha256"), label="upstream tarball hash")
    root = _text(upstream.get("archive_root"), label="archive_root")
    if "/" in root or root in {".", ".."}:
        raise DriverError("MANIFEST_INVALID", "RRR archive root is invalid")
    files = value.get("source_files")
    symbols = value.get("required_symbols")
    risks = value.get("credential_risk_paths")
    if not isinstance(files, Mapping) or not isinstance(symbols, Mapping):
        raise DriverError("MANIFEST_INVALID", "RRR source evidence is invalid")
    for relative, digest in files.items():
        if not isinstance(relative, str) or not relative or ".." in PurePosixPath(relative).parts:
            raise DriverError("MANIFEST_INVALID", "RRR source path is invalid")
        _sha(digest, label="source hash")
    if not set(symbols) <= set(files):
        raise DriverError("MANIFEST_INVALID", "RRR symbol source is not hash-bound")
    if not isinstance(risks, Sequence) or isinstance(risks, (str, bytes)):
        raise DriverError("MANIFEST_INVALID", "RRR credential-risk paths are invalid")
    if not all(isinstance(path, str) and path.endswith(".py") for path in risks):
        raise DriverError("MANIFEST_INVALID", "RRR credential-risk path is invalid")
    contract = value.get("runtime_contract")
    if not isinstance(contract, Mapping) or contract != {
        "stage_order": ["rewrite", "retrieve", "snippet", "reader"],
        "query_separator": ";",
        "generation_terminator": "**",
        "search_wrapper": "search[{query}]",
        "snippet_joiner": " ",
        "supported_rewriter_profiles": [
            "prompt",
            "warmup_checkpoint",
            "ppo_checkpoint",
        ],
        "supported_search_modes": ["bing_cache", "controlled_static"],
        "imports_upstream_runtime": False,
        "trains_ppo": False,
    }:
        raise DriverError("MANIFEST_INVALID", "RRR runtime contract drifted")
    if value.get("artifact_policy") != {
        "checkpoint_license": "NOASSERTION",
        "bundle_source": False,
        "bundle_checkpoints": False,
        "bundle_search_cache": False,
        "local_files_only": True,
        "trust_remote_code": False,
    }:
        raise DriverError("MANIFEST_INVALID", "RRR artifact policy drifted")


def verify_source_tarball(
    path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    allow_fixture_manifest: bool = False,
) -> SourceEvidence:
    manifest, manifest_hash = _load_json(manifest_path, label="RRR compatibility manifest")
    if not allow_fixture_manifest and manifest_hash != DEFAULT_MANIFEST_SHA256:
        raise DriverError("WORKER_INTEGRITY_FAILURE", "RRR compatibility manifest hash mismatch")
    _validate_manifest(manifest)
    upstream = manifest["upstream"]
    assert isinstance(upstream, Mapping)
    if _sha_file(path, limit=MAX_SOURCE_BYTES) != upstream["tarball_sha256"]:
        raise DriverError("SOURCE_ARCHIVE_MISMATCH", "RRR source tarball hash mismatch")
    archive_root = str(upstream["archive_root"])
    files = manifest["source_files"]
    symbols = manifest["required_symbols"]
    expected_risks = set(manifest["credential_risk_paths"])
    assert isinstance(files, Mapping) and isinstance(symbols, Mapping)
    seen: set[str] = set()
    risks: set[str] = set()
    license_files: set[str] = set()
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                relative = _safe_member(member.name, archive_root)
                if member.issym() or member.islnk() or member.isdev():
                    raise DriverError(
                        "SOURCE_ARCHIVE_UNSAFE", "RRR source archive contains a link/device"
                    )
                if relative and PurePosixPath(relative).name.lower() in {
                    "license",
                    "license.md",
                    "license.txt",
                    "copying",
                    "notice",
                }:
                    license_files.add(relative)
                if not member.isfile():
                    continue
                if relative in seen:
                    raise DriverError("SOURCE_ARCHIVE_UNSAFE", "RRR archive has duplicate paths")
                seen.add(relative)
                if member.size > MAX_MEMBER_BYTES and (
                    relative in files or relative.endswith(".py")
                ):
                    raise DriverError("SOURCE_ARCHIVE_UNSAFE", "RRR source member is too large")
                if relative in files or relative.endswith(".py"):
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise DriverError(
                            "SOURCE_ARCHIVE_INVALID", "RRR source member is unreadable"
                        )
                    raw = handle.read(MAX_MEMBER_BYTES + 1)
                    if len(raw) > MAX_MEMBER_BYTES:
                        raise DriverError("SOURCE_ARCHIVE_UNSAFE", "RRR source member is too large")
                    if relative in files and _sha_bytes(raw) != files[relative]:
                        raise DriverError(
                            "SOURCE_FILE_MISMATCH", f"RRR source hash mismatch: {relative}"
                        )
                    if relative.endswith(".py") and _CREDENTIAL_ASSIGNMENT.search(raw):
                        risks.add(relative)
                    if relative in symbols:
                        try:
                            tree = ast.parse(raw.decode("utf-8"), filename=relative)
                        except (UnicodeDecodeError, SyntaxError) as exc:
                            raise DriverError(
                                "SOURCE_SYMBOL_MISMATCH", "RRR source AST is invalid"
                            ) from exc
                        names = {
                            node.name
                            for node in tree.body
                            if isinstance(
                                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                            )
                        }
                        required = symbols[relative]
                        if not isinstance(required, Sequence) or not set(required) <= names:
                            raise DriverError(
                                "SOURCE_SYMBOL_MISMATCH", "RRR required source symbol is absent"
                            )
    except DriverError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise DriverError("SOURCE_ARCHIVE_INVALID", "RRR source tarball is invalid") from exc
    if license_files:
        raise DriverError("LICENSE_STATUS_DRIFT", "RRR source unexpectedly contains a license file")
    if not set(files) <= seen:
        raise DriverError("SOURCE_FILE_MISSING", "RRR source tarball omits audited files")
    if risks != expected_risks:
        raise DriverError(
            "CREDENTIAL_SCAN_DRIFT",
            "RRR credential-risk scan changed; values are intentionally redacted",
        )
    return SourceEvidence(
        manifest_sha256=manifest_hash,
        evidence_level=str(manifest["evidence_level"]),
        credential_risk_count=len(risks),
    )


def _git_read(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "--no-pager", "--no-optional-locks", "--no-replace-objects", "-C", str(root), *arguments],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError("SOURCE_CHECKOUT_INVALID", "RRR pinned checkout metadata is unavailable") from exc
    if result.returncode or len(result.stdout.encode()) > 4 * 1024 * 1024:
        raise DriverError("SOURCE_CHECKOUT_INVALID", "RRR pinned checkout metadata is invalid")
    return result.stdout


def verify_source_checkout(
    root: Path, *, manifest_path: Path = DEFAULT_MANIFEST, allow_fixture_manifest: bool = False,
) -> SourceEvidence:
    """Verify pinned executable source without importing it or reading datasets.

    Required files retain the shipped SHA-256 locks; every tracked Python file
    matches its blob at the pinned commit and the credential-risk inventory.
    This explicit source mode does not claim the complete archive checksum.
    """
    if root.is_symlink() or not root.is_dir():
        raise DriverError("SOURCE_CHECKOUT_INVALID", "RRR source checkout must be a regular directory")
    root = root.resolve()
    manifest, manifest_hash = _load_json(manifest_path, label="RRR compatibility manifest")
    if not allow_fixture_manifest and manifest_hash != DEFAULT_MANIFEST_SHA256:
        raise DriverError("WORKER_INTEGRITY_FAILURE", "RRR compatibility manifest hash mismatch")
    _validate_manifest(manifest)
    if _git_read(root, "rev-parse", "HEAD").strip() != UPSTREAM_COMMIT or (
        Path(_git_read(root, "rev-parse", "--show-toplevel").strip()).resolve() != root
    ):
        raise DriverError("SOURCE_CHECKOUT_MISMATCH", "RRR checkout commit/root differs")
    entries = _git_read(root, "ls-tree", "-r", "--full-tree", "-z", "HEAD").split("\0")
    if len(entries) > 10000:
        raise DriverError("SOURCE_CHECKOUT_INVALID", "RRR source tree exceeds its bounded file inventory")
    seen, risks, total_bytes = set(), set(), 0
    required = manifest["source_files"]
    for entry in filter(None, entries):
        try:
            metadata, name = entry.split("\t", 1)
            mode, kind, blob = metadata.split()
        except ValueError as exc:
            raise DriverError("SOURCE_CHECKOUT_INVALID", "RRR Git source inventory is malformed") from exc
        relative = _safe_member("tree/" + name, "tree")
        if relative in seen or mode not in {"100644", "100755"} or kind != "blob":
            raise DriverError("SOURCE_CHECKOUT_INVALID", "RRR Git source inventory contains a link or duplicate")
        seen.add(relative)
        if PurePosixPath(relative).name.lower() in {"license", "license.md", "license.txt", "copying", "notice"}:
            raise DriverError("LICENSE_STATUS_DRIFT", "RRR source unexpectedly contains a license file")
        if relative not in required and not relative.endswith(".py"):
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MEMBER_BYTES:
            raise DriverError("SOURCE_FILE_MISMATCH", "RRR tracked source file is missing, linked or too large")
        if not path.resolve().is_relative_to(root):
            raise DriverError("SOURCE_FILE_MISMATCH", "RRR tracked source escaped its checkout")
        raw = path.read_bytes()
        total_bytes += len(raw)
        if total_bytes > 64 * 1024 * 1024:
            raise DriverError("SOURCE_ARCHIVE_UNSAFE", "RRR executable-source review exceeds its byte cap")
        if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != blob:
            raise DriverError("SOURCE_FILE_MISMATCH", "RRR tracked source differs from its pinned Git blob")
        if relative in required and _sha_bytes(raw) != required[relative]:
            raise DriverError("SOURCE_FILE_MISMATCH", f"RRR source hash mismatch: {relative}")
        if relative.endswith(".py") and _CREDENTIAL_ASSIGNMENT.search(raw):
            risks.add(relative)
        if relative in manifest["required_symbols"]:
            try:
                tree = ast.parse(raw.decode("utf-8"), filename=relative)
            except (UnicodeDecodeError, SyntaxError) as exc:
                raise DriverError("SOURCE_SYMBOL_MISMATCH", "RRR source AST is invalid") from exc
            names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            if not set(manifest["required_symbols"][relative]) <= names:
                raise DriverError("SOURCE_SYMBOL_MISMATCH", "RRR required source symbol is absent")
    if not set(required) <= seen or risks != set(manifest["credential_risk_paths"]):
        raise DriverError("CREDENTIAL_SCAN_DRIFT", "RRR source/risk inventory changed; values are redacted")
    return SourceEvidence(manifest_hash,
        "fake_checkout_plumbing" if allow_fixture_manifest else "pinned_checkout_compatible_chain", len(risks))


def verify_checkpoint(root: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, digest = _load_json(manifest_path, label="RRR checkpoint manifest")
    expected = {
        "schema_version",
        "role",
        "profile",
        "revision",
        "model_type",
        "license",
        "files",
    }
    if set(manifest) != expected or manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint manifest schema mismatch")
    if manifest.get("role") not in {"rewriter", "reader"}:
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint role is invalid")
    if manifest.get("profile") not in {
        "prompt",
        "warmup_checkpoint",
        "ppo_checkpoint",
        "frozen_reader",
    }:
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint profile is invalid")
    if manifest.get("model_type") not in {"seq2seq", "causal"}:
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint model_type is invalid")
    if manifest.get("license") != "NOASSERTION":
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint license must remain NOASSERTION")
    _text(manifest.get("revision"), label="checkpoint revision")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint files are empty")
    if not any(str(name).lower().endswith(_WEIGHT_SUFFIXES) for name in files):
        raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint omits model weights")
    root = root.resolve()
    if not root.is_dir():
        raise DriverError("CHECKPOINT_MISSING", "RRR checkpoint root is missing")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str):
            raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint path is invalid")
        _sha(expected_hash, label="checkpoint file hash")
        candidate = root / relative
        try:
            candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise DriverError("CHECKPOINT_INVALID", "RRR checkpoint path escapes root") from exc
        if _sha_file(candidate, limit=2**40) != expected_hash:
            raise DriverError("CHECKPOINT_MISMATCH", "RRR checkpoint file hash mismatch")
    return {**dict(manifest), "manifest_sha256": digest, "root": root}


class LocalHFGenerator:
    """Local-only inference for prompt, warm-up, PPO, or frozen-reader checkpoints."""

    def __init__(self, checkpoint: Mapping[str, Any]) -> None:
        self.checkpoint = checkpoint
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise DriverError("DEPENDENCY_MISSING", "transformers is unavailable") from exc
        root = str(checkpoint["root"])
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                root, local_files_only=True, trust_remote_code=False
            )
            model_cls = (
                transformers.AutoModelForSeq2SeqLM
                if checkpoint["model_type"] == "seq2seq"
                else transformers.AutoModelForCausalLM
            )
            self.model = model_cls.from_pretrained(
                root, local_files_only=True, trust_remote_code=False
            )
            self.model.eval()
        except Exception as exc:
            raise DriverError(
                "CHECKPOINT_LOAD_FAILURE", "RRR local checkpoint failed to load"
            ) from exc

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "role": self.checkpoint["role"],
            "profile": self.checkpoint["profile"],
            "revision": self.checkpoint["revision"],
            "manifest_sha256": self.checkpoint["manifest_sha256"],
        }

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> Generation:
        del seed
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(**encoded, max_new_tokens=max_tokens, do_sample=False)
            input_ids = encoded["input_ids"]
            input_tokens = int(input_ids.shape[-1])
            row = outputs[0]
            if self.checkpoint["model_type"] == "causal":
                row = row[input_tokens:]
            text = self.tokenizer.decode(row, skip_special_tokens=True)
            output_tokens = int(row.shape[-1]) if hasattr(row, "shape") else len(row)
            if self.checkpoint["model_type"] == "seq2seq":
                # Encoder-decoder sequences contain a supplied decoder prefix;
                # max_new_tokens and metered output exclude that prefix.
                decoder_input = encoded.get("decoder_input_ids")
                decoder_prefix = int(decoder_input.shape[-1]) if decoder_input is not None else 1
                output_tokens -= decoder_prefix
                if output_tokens < 0:
                    raise ValueError("RRR sequence omits its decoder prefix")
        except Exception as exc:
            raise DriverError(
                "MODEL_RUNTIME_FAILURE",
                "RRR local generation failed",
                consumed=True,
            ) from exc
        return Generation(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


class SearchCache:
    def __init__(self, path: Path) -> None:
        manifest, digest = _load_json(path, label="RRR search cache")
        expected = {
            "schema_version",
            "mode",
            "corpus_revision",
            "index_revision",
            "window_start",
            "window_end",
            "entries",
        }
        if set(manifest) != expected or manifest.get("schema_version") != SEARCH_SCHEMA:
            raise DriverError("SEARCH_CACHE_INVALID", "RRR search cache schema mismatch")
        if manifest.get("mode") not in {"bing_cache", "controlled_static"}:
            raise DriverError("SEARCH_CACHE_INVALID", "RRR search cache mode is invalid")
        if not _DATE.fullmatch(str(manifest.get("window_start", ""))) or not _DATE.fullmatch(
            str(manifest.get("window_end", ""))
        ):
            raise DriverError("SEARCH_CACHE_INVALID", "RRR search cache window is invalid")
        entries = manifest.get("entries")
        if not isinstance(entries, Mapping):
            raise DriverError("SEARCH_CACHE_INVALID", "RRR search cache entries are invalid")
        self.manifest = manifest
        self.digest = digest

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "mode": self.manifest["mode"],
            "artifact_sha256": self.digest,
            "corpus_revision": self.manifest["corpus_revision"],
            "index_revision": self.manifest["index_revision"],
            "window_start": self.manifest["window_start"],
            "window_end": self.manifest["window_end"],
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        window_start: str,
        window_end: str,
    ) -> tuple[SearchDocument, ...]:
        if (
            window_start != self.manifest["window_start"]
            or window_end != self.manifest["window_end"]
        ):
            raise DriverError("SEARCH_IDENTITY_MISMATCH", "RRR search time window mismatch")
        rows = self.manifest["entries"].get(query)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise DriverError("SEARCH_CACHE_MISS", "RRR search cache has no bound query")
        if len(rows) > top_k:
            rows = rows[:top_k]
        result: list[SearchDocument] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "document_id",
                "title",
                "text",
                "score",
                "metadata",
            }:
                raise DriverError("SEARCH_CACHE_INVALID", "RRR search row schema is invalid")
            document_id = _text(row["document_id"], label="search document ID")
            text = _text(row["text"], label="search document text")
            score = row["score"]
            if (
                document_id in seen
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise DriverError("SEARCH_CACHE_INVALID", "RRR search row identity is invalid")
            seen.add(document_id)
            metadata = row["metadata"]
            if not isinstance(metadata, Mapping):
                raise DriverError("SEARCH_CACHE_INVALID", "RRR search metadata is invalid")
            result.append(
                SearchDocument(
                    document_id=document_id,
                    title=_text(row["title"], label="search title", allow_empty=True),
                    text=text,
                    score=float(score),
                    metadata=dict(metadata),
                )
            )
        return tuple(result)


def _validate_request(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROOT_FIELDS:
        raise DriverError("INVALID_REQUEST", "RRR request fields do not match schema")
    if (
        value.get("method_id") != METHOD_ID
        or value.get("release_id") not in set(RELEASE_IDS.values())
        or value.get("upstream_commit") != UPSTREAM_COMMIT
        or value.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise DriverError("IDENTITY_MISMATCH", "RRR request identity mismatch")
    request_id = _sha(value.get("request_id"), label="request_id")
    _sha(value.get("state_hash"), label="state_hash")
    try:
        expected_id = _content_hash({key: value[key] for key in value if key != "request_id"})
    except (TypeError, ValueError) as exc:
        raise DriverError("INVALID_REQUEST", "RRR request is not canonical JSON") from exc
    if request_id != expected_id:
        raise DriverError("REQUEST_ID_MISMATCH", "RRR request_id mismatch")
    payload = value.get("payload")
    dependencies = value.get("dependency_fingerprints")
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS:
        raise DriverError("INVALID_REQUEST", "RRR payload fields do not match schema")
    if not isinstance(dependencies, Mapping) or set(dependencies) != {"backend", "retriever"}:
        raise DriverError("INVALID_REQUEST", "RRR dependency fields do not match schema")
    if not isinstance(dependencies["backend"], Mapping) or not dependencies["backend"]:
        raise DriverError("INVALID_REQUEST", "RRR backend fingerprint is required")
    if not isinstance(dependencies["retriever"], Mapping):
        raise DriverError("INVALID_REQUEST", "RRR retriever fingerprint must be an object")
    if value.get("minimum_reservation") != {"model_calls": 2, "retrieval_calls": 1}:
        raise DriverError("INVALID_REQUEST", "RRR reservation is invalid")
    normalized = json.loads(_canonical_json(payload))
    controls = normalized.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != _CONTROL_FIELDS:
        raise DriverError("INVALID_REQUEST", "RRR controls do not match schema")
    if controls.get("max_queries") != 1:
        raise DriverError("INVALID_REQUEST", "RRR worker supports one query")
    top_k = _positive_int(controls.get("top_k"), label="top_k")
    if top_k > 100:
        raise DriverError("INVALID_REQUEST", "RRR top_k exceeds 100")
    if controls.get("query_parser") != "upstream_semicolon_terminator_single":
        raise DriverError("INVALID_REQUEST", "RRR parser identity drifted")
    if controls.get("search_mode") not in {"bing_cache", "controlled_static"}:
        raise DriverError("INVALID_REQUEST", "RRR search mode is invalid")
    if controls.get("rewriter_profile") not in {
        "prompt",
        "warmup_checkpoint",
        "ppo_checkpoint",
    }:
        raise DriverError("INVALID_REQUEST", "RRR rewriter profile is invalid")
    if value.get("release_id") != RELEASE_IDS[controls["rewriter_profile"]]:
        raise DriverError("IDENTITY_MISMATCH", "RRR release/profile identity mismatch")
    for key in ("max_rewriter_tokens", "max_reader_tokens", "max_input_chars"):
        _positive_int(controls.get(key), label=key)
    for key in ("search_window_start", "search_window_end"):
        if not isinstance(controls.get(key), str) or not _DATE.fullmatch(controls[key]):
            raise DriverError("INVALID_REQUEST", "RRR search time window is invalid")
    if controls["search_window_start"] > controls["search_window_end"]:
        raise DriverError("INVALID_REQUEST", "RRR search time window is reversed")
    _text(normalized.get("query"), label="query")
    _text(normalized.get("task_id"), label="task_id", allow_empty=True)
    _text(normalized.get("instruction"), label="instruction", allow_empty=True)
    if not isinstance(normalized.get("messages"), Sequence):
        raise DriverError("INVALID_REQUEST", "RRR messages must be a sequence")
    for key in (
        "prompt_hash",
        "reader_prompt_hash",
        "search_artifact_hash",
        "backend_artifact_hash",
        "retriever_artifact_hash",
        "input_artifact_hash",
    ):
        _sha(normalized.get(key), label=key)
    for key in (
        "corpus_revision",
        "index_revision",
        "rewriter_checkpoint_revision",
        "reader_checkpoint_revision",
    ):
        _text(
            normalized.get(key), label=key, allow_empty=key in {"corpus_revision", "index_revision"}
        )
    if (
        normalized.get("source_license_acknowledged") is not True
        or normalized.get("checkpoint_license_acknowledged") is not True
    ):
        raise DriverError("LICENSE_ACKNOWLEDGEMENT_REQUIRED", "RRR license acknowledgement missing")
    _nonnegative_int(normalized.get("seed"), label="seed")
    backend_hash = _content_hash(dependencies["backend"])
    retriever_hash = _content_hash(dependencies["retriever"])
    if (
        normalized["backend_artifact_hash"] != backend_hash
        or normalized["retriever_artifact_hash"] != retriever_hash
    ):
        raise DriverError("DEPENDENCY_MISMATCH", "RRR dependency fingerprint hash mismatch")
    expected_input = _content_hash(
        {
            "query": normalized["query"],
            "instruction": normalized["instruction"],
            "messages": normalized["messages"],
            "controls": controls,
        }
    )
    if normalized["input_artifact_hash"] != expected_input:
        raise DriverError("INPUT_IDENTITY_MISMATCH", "RRR input artifact hash mismatch")
    return {
        "request_id": request_id,
        "payload": normalized,
        "dependencies": json.loads(_canonical_json(dependencies)),
    }


def _parse_query(raw: str) -> str:
    value = raw.split("**", 1)[0].strip()
    if "Query:" in value:
        value = value.split("Query:", 1)[1].strip()
    queries = tuple(item.strip() for item in value.split(";") if item.strip())
    if len(queries) != 1:
        raise DriverError("QUERY_PARSE_FAILURE", "RRR single-query parser produced != 1 query")
    return queries[0]


def _usage(rewriter: Generation, reader: Generation) -> dict[str, int]:
    return {
        "model_calls": 2,
        "retrieval_calls": 1,
        "input_tokens": rewriter.input_tokens + reader.input_tokens,
        "output_tokens": rewriter.output_tokens + reader.output_tokens,
    }


@dataclass(slots=True)
class RRRRuntime:
    source: SourceEvidence
    prompt_path: Path
    reader_prompt_path: Path
    rewriter: Generator
    searcher: Searcher
    reader: Generator
    acknowledge_noassertion_source: bool = False
    acknowledge_noassertion_checkpoints: bool = False
    acknowledge_credential_quarantine: bool = False
    prompt: str = field(init=False, repr=False)
    reader_prompt: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.acknowledge_noassertion_source:
            raise DriverError(
                "LICENSE_ACKNOWLEDGEMENT_REQUIRED",
                "RRR source NOASSERTION acknowledgement required",
            )
        if not self.acknowledge_noassertion_checkpoints:
            raise DriverError(
                "LICENSE_ACKNOWLEDGEMENT_REQUIRED",
                "RRR checkpoint NOASSERTION acknowledgement required",
            )
        if self.source.credential_risk_count and not self.acknowledge_credential_quarantine:
            raise DriverError(
                "CREDENTIAL_QUARANTINE_REQUIRED",
                "RRR source contains redacted credential-risk findings and must remain quarantined",
            )
        self.prompt = _read_bytes(self.prompt_path, label="RRR rewrite prompt").decode("utf-8")
        self.reader_prompt = _read_bytes(self.reader_prompt_path, label="RRR reader prompt").decode(
            "utf-8"
        )
        if "{query}" not in self.prompt or not {"{query}", "{background}"} <= {
            token for token in ("{query}", "{background}") if token in self.reader_prompt
        }:
            raise DriverError("PROMPT_INVALID", "RRR mounted prompt placeholders are invalid")

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": DRIVER_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_manifest_sha256": self.source.manifest_sha256,
            "execution_evidence": self.source.evidence_level,
            "credential_risk_count": self.source.credential_risk_count,
            "upstream_imported": False,
            "ppo_training": False,
            "rewriter": dict(self.rewriter.fingerprint()),
            "searcher": dict(self.searcher.fingerprint()),
            "reader": dict(self.reader.fingerprint()),
        }

    def call(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        request = _validate_request(raw)
        payload = request["payload"]
        controls = payload["controls"]
        if (
            _sha_bytes(self.prompt.encode()) != payload["prompt_hash"]
            or _sha_bytes(self.reader_prompt.encode()) != payload["reader_prompt_hash"]
        ):
            raise DriverError("PROMPT_IDENTITY_MISMATCH", "RRR prompt hash mismatch")
        search_fp = self.searcher.fingerprint()
        if (
            search_fp.get("artifact_sha256") != payload["search_artifact_hash"]
            or search_fp.get("mode") != controls["search_mode"]
            or search_fp.get("corpus_revision") != payload["corpus_revision"]
            or search_fp.get("index_revision") != payload["index_revision"]
        ):
            raise DriverError("SEARCH_IDENTITY_MISMATCH", "RRR search artifact identity mismatch")
        rewriter_fp = self.rewriter.fingerprint()
        reader_fp = self.reader.fingerprint()
        expected_backend_hash = _content_hash({"rewriter": rewriter_fp, "reader": reader_fp})
        if payload["backend_artifact_hash"] != expected_backend_hash:
            raise DriverError(
                "DEPENDENCY_MISMATCH", "RRR rewriter/reader deployment identity mismatch"
            )
        expected_retriever_hash = (
            _content_hash(search_fp)
            if controls["search_mode"] == "controlled_static"
            else _content_hash({})
        )
        if payload["retriever_artifact_hash"] != expected_retriever_hash:
            raise DriverError("DEPENDENCY_MISMATCH", "RRR controlled retriever identity mismatch")
        if (
            rewriter_fp.get("profile") != controls["rewriter_profile"]
            or rewriter_fp.get("revision") != payload["rewriter_checkpoint_revision"]
            or reader_fp.get("revision") != payload["reader_checkpoint_revision"]
        ):
            raise DriverError("CHECKPOINT_IDENTITY_MISMATCH", "RRR model identity mismatch")
        rewrite_prompt = self.prompt.replace("{query}", payload["query"])
        if len(rewrite_prompt) > controls["max_input_chars"]:
            raise DriverError("BUDGET_EXCEEDED", "RRR rewrite prompt exceeds character budget")
        rewrite = self.rewriter.generate(
            rewrite_prompt,
            seed=payload["seed"],
            max_tokens=controls["max_rewriter_tokens"],
        )
        try:
            query = _parse_query(rewrite.text)
        except DriverError as exc:
            # Parsing happens after the paid rewrite. Keep the known first
            # call's cost instead of reporting an unconsumed input failure.
            exc.consumed = True
            exc.usage = dict(model_calls=1, retrieval_calls=0,
                input_tokens=rewrite.input_tokens, output_tokens=rewrite.output_tokens)
            raise
        documents = self.searcher.search(
            query,
            top_k=controls["top_k"],
            window_start=controls["search_window_start"],
            window_end=controls["search_window_end"],
        )
        snippet = " ".join(item.text for item in documents)
        reader_prompt = self.reader_prompt.replace("{background}", snippet).replace(
            "{query}", payload["query"]
        )
        if len(reader_prompt) > controls["max_input_chars"]:
            raise DriverError(
                "BUDGET_EXCEEDED",
                "RRR reader prompt exceeds character budget",
                consumed=True,
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 1,
                    "input_tokens": rewrite.input_tokens,
                    "output_tokens": rewrite.output_tokens,
                },
            )
        reader = self.reader.generate(
            reader_prompt,
            seed=payload["seed"],
            max_tokens=controls["max_reader_tokens"],
        )
        rows = [
            {
                "document_id": item.document_id,
                "title": item.title,
                "text": item.text,
                "corpus_revision": payload["corpus_revision"],
                "query_index": 0,
                "rank": index + 1,
                "score": item.score,
                "metadata": {
                    **dict(item.metadata),
                    "search_artifact_hash": payload["search_artifact_hash"],
                },
            }
            for index, item in enumerate(documents)
        ]
        trajectory = {
            "stage_order": ["rewrite", "retrieve", "snippet", "reader"],
            "rewrite_prompt_hash": _sha_bytes(rewrite_prompt.encode()),
            "raw_rewrite_hash": _sha_bytes(rewrite.text.encode()),
            "query": query,
            "search_artifact_hash": payload["search_artifact_hash"],
            "document_ids": [item.document_id for item in documents],
            "snippet_hash": _sha_bytes(snippet.encode()),
            "reader_prompt_hash": _sha_bytes(reader_prompt.encode()),
            "answer_hash": _sha_bytes(reader.text.encode()),
        }
        return {
            "status": "ok",
            "request_id": request["request_id"],
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "payload": {
                "queries": [query],
                "reader_query": payload["query"],
                "documents": rows,
                "answer": reader.text.strip().removesuffix("**").strip(),
                "parser_mode": controls["query_parser"],
                "stage_order": ["rewrite", "retrieve", "snippet", "reader"],
                "trajectory_hash": _content_hash(trajectory),
                "prompt_hash": payload["prompt_hash"],
                "reader_prompt_hash": payload["reader_prompt_hash"],
                "search_artifact_hash": payload["search_artifact_hash"],
                "rewriter_checkpoint_revision": payload["rewriter_checkpoint_revision"],
                "reader_checkpoint_revision": payload["reader_checkpoint_revision"],
                "backend_artifact_hash": payload["backend_artifact_hash"],
                "retriever_artifact_hash": payload["retriever_artifact_hash"],
                "input_artifact_hash": payload["input_artifact_hash"],
                "source_manifest_sha256": self.source.manifest_sha256,
                "execution_evidence": self.source.evidence_level,
            },
            "usage": _usage(rewrite, reader),
        }


def _failure(exc: DriverError, request_id: str = "") -> Mapping[str, Any]:
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
    if not raw or len(raw) > MAX_REQUEST_BYTES or sys.stdin.buffer.read(1):
        raise DriverError("INVALID_REQUEST", "RRR worker accepts one bounded JSON line")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("INVALID_REQUEST", "RRR request is invalid JSON") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-tarball", type=Path)
    source.add_argument("--source-checkout", type=Path)
    parser.add_argument("--smoke-source", action="store_true")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--reader-prompt", type=Path)
    parser.add_argument("--search-cache", type=Path)
    parser.add_argument("--rewriter-checkpoint", type=Path)
    parser.add_argument("--rewriter-manifest", type=Path)
    parser.add_argument("--reader-checkpoint", type=Path)
    parser.add_argument("--reader-manifest", type=Path)
    parser.add_argument("--acknowledge-noassertion-source", action="store_true")
    parser.add_argument("--acknowledge-noassertion-checkpoints", action="store_true")
    parser.add_argument("--acknowledge-credential-quarantine", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_id = ""
    try:
        source = (verify_source_checkout(args.source_checkout) if args.source_checkout is not None
                  else verify_source_tarball(args.source_tarball))
        if args.smoke_source:
            response: Mapping[str, Any] = {
                "status": "ok",
                "upstream_commit": UPSTREAM_COMMIT,
                "source_manifest_sha256": source.manifest_sha256,
                "execution_evidence": source.evidence_level,
                "credential_risk_count": source.credential_risk_count,
                "credential_values_redacted": True,
                "upstream_imported": False,
            }
        else:
            required = (
                args.prompt,
                args.reader_prompt,
                args.search_cache,
                args.rewriter_checkpoint,
                args.rewriter_manifest,
                args.reader_checkpoint,
                args.reader_manifest,
            )
            if any(value is None for value in required):
                raise DriverError("CONFIGURATION_MISSING", "RRR inference mounts are incomplete")
            rewriter_checkpoint = verify_checkpoint(
                args.rewriter_checkpoint, args.rewriter_manifest
            )
            reader_checkpoint = verify_checkpoint(args.reader_checkpoint, args.reader_manifest)
            runtime = RRRRuntime(
                source=source,
                prompt_path=args.prompt,
                reader_prompt_path=args.reader_prompt,
                rewriter=LocalHFGenerator(rewriter_checkpoint),
                searcher=SearchCache(args.search_cache),
                reader=LocalHFGenerator(reader_checkpoint),
                acknowledge_noassertion_source=args.acknowledge_noassertion_source,
                acknowledge_noassertion_checkpoints=args.acknowledge_noassertion_checkpoints,
                acknowledge_credential_quarantine=args.acknowledge_credential_quarantine,
            )
            raw = _read_request()
            if isinstance(raw, Mapping) and isinstance(raw.get("request_id"), str):
                request_id = raw["request_id"]
            response = runtime.call(raw)
    except DriverError as exc:
        response = _failure(exc, request_id)
    except Exception as exc:  # pragma: no cover - final secret-free containment
        response = _failure(
            DriverError("WORKER_INTERNAL_FAILURE", f"RRR failed closed: {type(exc).__name__}"),
            request_id,
        )
    sys.stdout.write(_canonical_json(response) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
