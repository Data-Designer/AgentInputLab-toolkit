"""Load and validate source or packaged algorithm registries."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import RegistryError
from agentinputlab.contracts.serialization import canonical_json

from .schema import RegistryDocument


def _decode_json_compatible_yaml(text: str, source: str) -> Any:
    """Decode the registry with the standard library.

    JSON is a strict subset of YAML, so the editable ``registry.yaml`` remains
    valid YAML while minimal installations avoid a PyYAML dependency.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryError(
            f"{source} is not JSON-compatible YAML: line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def load_registry_text(text: str, *, source: str = "<memory>") -> RegistryDocument:
    """Parse and validate a registry document from text."""

    return RegistryDocument.from_mapping(_decode_json_compatible_yaml(text, source))


def load_registry_file(path: str | Path) -> RegistryDocument:
    """Parse and validate a registry document from a filesystem path."""

    resolved = Path(path).expanduser().resolve()
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"cannot read registry {resolved}: {exc}") from exc
    return load_registry_text(text, source=str(resolved))


def load_packaged_registry() -> RegistryDocument:
    """Load the immutable snapshot distributed inside the Python package."""

    resource = resources.files("agentinputlab.registry").joinpath("data", "algorithms.json")
    try:
        text = resource.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"cannot read packaged algorithm registry: {exc}") from exc
    return load_registry_text(text, source="agentinputlab.registry/data/algorithms.json")


def find_repository_registry(start: str | Path | None = None) -> Path | None:
    """Find ``manifests/algorithms/registry.yaml`` in a source checkout."""

    cursor = Path.cwd() if start is None else Path(start)
    cursor = cursor.expanduser().resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for directory in (cursor, *cursor.parents):
        candidate = directory / "manifests" / "algorithms" / "registry.yaml"
        if candidate.is_file():
            return candidate
    return None


def registry_documents_equal(left: RegistryDocument, right: RegistryDocument) -> bool:
    """Compare semantic content independent of JSON formatting."""

    return canonical_json(left.to_dict()) == canonical_json(right.to_dict())


def validate_snapshot_sync(source_path: str | Path) -> tuple[RegistryDocument, RegistryDocument]:
    """Validate a source registry and require equality with the package snapshot."""

    source = load_registry_file(source_path)
    packaged = load_packaged_registry()
    if not registry_documents_equal(source, packaged):
        raise RegistryError(
            "source registry and packaged snapshot differ; run "
            "`python scripts/check_registry.py --write`"
        )
    return source, packaged


__all__ = [
    "find_repository_registry",
    "load_packaged_registry",
    "load_registry_file",
    "load_registry_text",
    "registry_documents_equal",
    "validate_snapshot_sync",
]
