"""Content-bound receipts for assets admitted by experiment preflight.

The manifest deliberately stores environment-variable *names*, not machine-local
paths or secret values.  This module resolves those names at preflight time and
turns every admitted prerequisite into a canonical, secret-safe identity.  The
resulting receipt is embedded in the effective experiment-driver fingerprint.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import (
    content_hash,
    freeze_mapping,
    to_jsonable,
)

from .schema import ExperimentManifest, PrerequisiteKind, require_sha256

_GIT_OBJECT_ID_LENGTHS = frozenset({40, 64})
_CHECKOUT_SUFFIX = "_CHECKOUT"


@dataclass(frozen=True, slots=True)
class FrozenBindingEntry:
    """One prerequisite identity without its raw environment value."""

    name: str
    kind: str
    identity: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported frozen binding entry schema")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("frozen binding entry requires a name")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValidationError("frozen binding entry requires a kind")
        try:
            identity = freeze_mapping(self.identity)
        except (TypeError, ValueError) as exc:
            raise ValidationError("frozen binding identity must be canonical JSON") from exc
        if not identity:
            raise ValidationError("frozen binding identity must not be empty")
        object.__setattr__(self, "identity", identity)

    @property
    def binding_hash(self) -> str:
        return content_hash(
            {
                "schema_version": self.schema_version,
                "name": self.name,
                "kind": self.kind,
                "identity": self.identity,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "identity": to_jsonable(self.identity),
            "binding_hash": self.binding_hash,
        }


@dataclass(frozen=True, slots=True)
class FrozenBindingReceipt:
    """Canonical receipt for all manifest prerequisites accepted by preflight."""

    manifest_hash: str
    bindings: tuple[FrozenBindingEntry, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported frozen binding receipt schema")
        require_sha256(self.manifest_hash, label="binding receipt manifest_hash")
        bindings = tuple(self.bindings)
        if any(not isinstance(item, FrozenBindingEntry) for item in bindings):
            raise ValidationError("frozen binding receipt entries must be typed")
        identities = [(item.kind, item.name) for item in bindings]
        if len(identities) != len(set(identities)):
            raise ValidationError("frozen binding receipt entries must be unique")
        canonical = tuple(sorted(bindings, key=lambda item: (item.kind, item.name)))
        object.__setattr__(self, "bindings", canonical)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema_version": self.schema_version,
                "manifest_hash": self.manifest_hash,
                "bindings": [
                    {
                        "schema_version": item.schema_version,
                        "name": item.name,
                        "kind": item.kind,
                        "identity": item.identity,
                    }
                    for item in self.bindings
                ],
            }
        )

    def binding(self, name: str, *, kind: str | None = None) -> FrozenBindingEntry:
        matches = tuple(
            item
            for item in self.bindings
            if item.name == name and (kind is None or item.kind == kind)
        )
        if len(matches) != 1:
            raise ArtifactError(f"frozen binding receipt has no unique entry for {name!r}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "bindings": [item.to_dict() for item in self.bindings],
            "receipt_hash": self.receipt_hash,
        }


def resolve_frozen_binding_receipt(
    manifest: ExperimentManifest,
    *,
    environment: Mapping[str, str] | None = None,
) -> FrozenBindingReceipt:
    """Resolve and independently attest every declared prerequisite.

    Checkout directories are required to be clean Git roots.  Other directories
    receive a deterministic content-tree hash.  Raw paths and environment values
    never enter the receipt; only their SHA-256 identities do.
    """

    if not isinstance(manifest, ExperimentManifest):
        raise TypeError("binding resolution requires ExperimentManifest")
    env = dict(os.environ if environment is None else environment)
    expectations = _checkout_expectations(manifest, environment=env)
    declared_directories = {
        item.name for item in manifest.prerequisites if item.kind is PrerequisiteKind.DIRECTORY_ENV
    }
    unknown_expectations = sorted(set(expectations) - declared_directories)
    if unknown_expectations:
        raise ValidationError(
            "checkout_expected_revisions references undeclared directories: "
            + ", ".join(unknown_expectations)
        )

    bindings: list[FrozenBindingEntry] = []
    for prerequisite in manifest.prerequisites:
        if prerequisite.kind is PrerequisiteKind.ENV:
            raw = env.get(prerequisite.name)
            if not raw:
                raise ArtifactError(f"missing environment prerequisite {prerequisite.name}")
            encoded = raw.encode("utf-8")
            identity = {
                "identity_mode": "value_sha256",
                "value_sha256": hashlib.sha256(encoded).hexdigest(),
                "value_bytes": len(encoded),
            }
        elif prerequisite.kind is PrerequisiteKind.EXECUTABLE:
            executable = shutil.which(prerequisite.name, path=env.get("PATH"))
            if executable is None:
                raise ArtifactError(f"missing executable prerequisite {prerequisite.name}")
            candidate = Path(executable).expanduser().resolve()
            if not candidate.is_file():
                raise ArtifactError(f"executable prerequisite {prerequisite.name} is not a file")
            identity = {
                "identity_mode": "executable_sha256",
                "resolved_path_sha256": _sha256_text(str(candidate)),
                "content_sha256": _sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        else:
            raw = env.get(prerequisite.name)
            if not raw:
                raise ArtifactError(f"missing environment prerequisite {prerequisite.name}")
            candidate = Path(raw).expanduser().resolve()
            if prerequisite.kind is PrerequisiteKind.FILE_ENV:
                if not candidate.is_file():
                    raise ArtifactError(f"{prerequisite.name} does not identify an existing file")
                actual = _sha256_file(candidate)
                if prerequisite.sha256_env is not None:
                    expected = env.get(prerequisite.sha256_env)
                    if not expected:
                        raise ArtifactError(
                            f"missing checksum prerequisite {prerequisite.sha256_env}"
                        )
                    require_sha256(
                        expected,
                        label=f"environment {prerequisite.sha256_env}",
                    )
                    if actual != expected:
                        raise ArtifactError(
                            f"{prerequisite.name} checksum differs from {prerequisite.sha256_env}"
                        )
                identity = {
                    "identity_mode": "file_sha256",
                    "resolved_path_sha256": _sha256_text(str(candidate)),
                    "content_sha256": actual,
                    "size_bytes": candidate.stat().st_size,
                    "checksum_environment": prerequisite.sha256_env,
                }
            else:
                if not candidate.is_dir():
                    raise ArtifactError(
                        f"{prerequisite.name} does not identify an existing directory"
                    )
                expected_revision = expectations.get(prerequisite.name)
                if expected_revision is not None or prerequisite.name.endswith(_CHECKOUT_SUFFIX):
                    identity = _git_checkout_identity(
                        candidate,
                        expected_revision=expected_revision,
                    )
                else:
                    identity = {
                        "identity_mode": "directory_tree_sha256",
                        "resolved_path_sha256": _sha256_text(str(candidate)),
                        "tree_sha256": _sha256_directory_tree(candidate),
                    }
        bindings.append(
            FrozenBindingEntry(
                name=prerequisite.name,
                kind=prerequisite.kind.value,
                identity=identity,
            )
        )
    return FrozenBindingReceipt(
        manifest_hash=manifest.manifest_hash,
        bindings=tuple(bindings),
    )


def build_guarded_driver_fingerprint(
    operator_fingerprint: Mapping[str, Any],
    binding_receipt: FrozenBindingReceipt,
    *,
    parity_contract: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build the exact fingerprint persisted by the guarded runtime driver."""

    if not isinstance(binding_receipt, FrozenBindingReceipt):
        raise TypeError("guarded fingerprint requires FrozenBindingReceipt")
    try:
        operator = freeze_mapping(operator_fingerprint)
        contract = freeze_mapping(parity_contract or {})
    except (TypeError, ValueError) as exc:
        raise ValidationError("guarded driver identity must be canonical JSON") from exc
    if not operator:
        raise ValidationError("operator driver fingerprint must not be empty")
    return freeze_mapping(
        {
            "schema_version": 1,
            "guard": "agentinputlab.frozen_binding_driver_guard",
            "operator_driver_fingerprint": operator,
            "frozen_binding_receipt": binding_receipt.to_dict(),
            "frozen_binding_receipt_hash": binding_receipt.receipt_hash,
            "parity_contract": contract,
        }
    )


def _checkout_expectations(
    manifest: ExperimentManifest,
    *,
    environment: Mapping[str, str],
) -> dict[str, str]:
    raw = manifest.metadata.get("checkout_expected_revisions", {})
    if not isinstance(raw, Mapping):
        raise ValidationError("checkout_expected_revisions metadata must be an object")
    result: dict[str, str] = {}
    for name, revision in raw.items():
        if not isinstance(name, str) or not name:
            raise ValidationError("checkout expectation names must be non-empty strings")
        if (
            not isinstance(revision, str)
            or len(revision) not in _GIT_OBJECT_ID_LENGTHS
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValidationError(
                f"checkout expectation {name} must be a full lowercase Git object ID"
            )
        result[name] = revision

    dynamic = manifest.metadata.get("checkout_expected_revision_from_env", {})
    if not isinstance(dynamic, Mapping):
        raise ValidationError("checkout_expected_revision_from_env metadata must be an object")
    declared_value_environments = {
        item.name for item in manifest.prerequisites if item.kind is PrerequisiteKind.ENV
    }
    for checkout_environment, revision_environment in dynamic.items():
        if not isinstance(checkout_environment, str) or not checkout_environment:
            raise ValidationError("dynamic checkout expectation names must be non-empty strings")
        if not isinstance(revision_environment, str) or not revision_environment:
            raise ValidationError(
                "dynamic checkout revision environments must be non-empty strings"
            )
        if revision_environment not in declared_value_environments:
            raise ValidationError(
                "checkout_expected_revision_from_env references an undeclared "
                f"value prerequisite: {revision_environment}"
            )
        revision = environment.get(revision_environment)
        if not revision:
            raise ArtifactError(f"missing checkout revision prerequisite {revision_environment}")
        if (
            not isinstance(revision, str)
            or len(revision) not in _GIT_OBJECT_ID_LENGTHS
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValidationError(
                f"environment {revision_environment} must be a full lowercase Git object ID"
            )
        existing = result.get(checkout_environment)
        if existing is not None and existing != revision:
            raise ValidationError(
                f"dynamic checkout expectation for {checkout_environment} "
                "conflicts with the manifest-pinned revision"
            )
        result[checkout_environment] = revision
    return result


def _git_checkout_identity(
    path: Path,
    *,
    expected_revision: str | None,
) -> Mapping[str, Any]:
    if shutil.which("git") is None:
        raise ArtifactError("git is required to attest checkout prerequisites")
    top_level = Path(_git(path, "rev-parse", "--show-toplevel")).resolve()
    if top_level != path:
        raise ArtifactError("checkout prerequisite must identify the Git repository root")
    head = _git(path, "rev-parse", "HEAD")
    tree = _git(path, "rev-parse", "HEAD^{tree}")
    for label, value in (("HEAD", head), ("tree", tree)):
        if len(value) not in _GIT_OBJECT_ID_LENGTHS or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ArtifactError(f"checkout {label} is not a full Git object ID")
    if expected_revision is not None and head != expected_revision:
        raise ArtifactError("checkout HEAD differs from the manifest-pinned revision")
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ArtifactError("checkout prerequisite must have a clean working tree")
    return freeze_mapping(
        {
            "identity_mode": "clean_git_checkout",
            "resolved_path_sha256": _sha256_text(str(path)),
            "git_head": head,
            "git_tree": tree,
            "git_clean": True,
            "expected_revision": expected_revision,
        }
    )


def _git(path: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(path), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactError(f"cannot attest Git checkout with {' '.join(arguments)}") from exc
    return completed.stdout.strip()


def _sha256_directory_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode):
            digest.update(f"D\0{relative}\0".encode())
        elif stat.S_ISREG(mode):
            digest.update(f"F\0{relative}\0".encode())
            digest.update(_sha256_file(candidate).encode())
            digest.update(b"\0")
        elif stat.S_ISLNK(mode):
            target = candidate.resolve(strict=True)
            if not target.is_file():
                raise ArtifactError(f"directory binding symlink must resolve to a file: {relative}")
            digest.update(f"L\0{relative}\0".encode())
            digest.update(_sha256_text(os.readlink(candidate)).encode())
            digest.update(b"\0")
            digest.update(_sha256_file(target).encode())
            digest.update(b"\0")
        else:
            raise ArtifactError(
                f"directory binding contains unsupported filesystem entry: {relative}"
            )
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "FrozenBindingEntry",
    "FrozenBindingReceipt",
    "build_guarded_driver_fingerprint",
    "resolve_frozen_binding_receipt",
]
