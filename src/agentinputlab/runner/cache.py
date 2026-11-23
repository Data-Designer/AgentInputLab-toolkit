"""Run cache identities include every declared scientific dependency."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.contracts.state import FrozenAgentState

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def run_cache_key(
    *,
    state: FrozenAgentState,
    release_id: str,
    profile_id: str,
    seed: int,
    dependency_hashes: Mapping[str, str] | None = None,
    backend_fingerprint: Mapping[str, Any] | None = None,
    retriever_fingerprint: Mapping[str, Any] | None = None,
    prompt_hash: str = "",
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Return a v2 cache identity over all declared execution dependencies.

    ``dependency_hashes`` accepts the output of
    :func:`context_dependency_fingerprints` directly.  The legacy backend and
    retriever fingerprint mappings remain supported during the alpha API, but
    a slot may not be supplied through both interfaces.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationError("cache seed must be a non-negative integer")

    normalized = _normalize_dependency_hashes(dependency_hashes, seed=seed)
    for slot, fingerprint in (
        ("backend", backend_fingerprint),
        ("retriever", retriever_fingerprint),
    ):
        if fingerprint is None:
            continue
        if slot in normalized:
            raise ValidationError(
                f"dependency slot supplied through both dependency_hashes and legacy API: {slot}"
            )
        if not isinstance(fingerprint, Mapping):
            raise ValidationError(f"legacy {slot} fingerprint must be a mapping")
        normalized[slot] = _canonical_mapping_hash(
            fingerprint,
            label=f"legacy {slot} fingerprint",
            require_non_empty=True,
        )

    normalized_extra = _normalize_extra(extra)

    return content_hash(
        {
            "schema": "agentinputlab-run-cache-v2",
            "state_hash": state.state_hash,
            "release_id": release_id,
            "profile_id": profile_id,
            "dependency_hashes": dict(sorted(normalized.items())),
            "seed": seed,
            "prompt_hash": prompt_hash,
            "extra": dict(normalized_extra),
        }
    )


def _normalize_dependency_hashes(
    dependency_hashes: Mapping[str, str] | None,
    *,
    seed: int,
) -> dict[str, str]:
    if dependency_hashes is None:
        return {}
    if not isinstance(dependency_hashes, Mapping):
        raise ValidationError("dependency_hashes must be a mapping")

    normalized: dict[str, str] = {}
    for slot, fingerprint in dependency_hashes.items():
        if not isinstance(slot, str) or not slot or slot != slot.strip():
            raise ValidationError(
                "dependency hash slots must be non-empty strings without surrounding whitespace"
            )
        if slot in normalized:
            raise ValidationError(f"duplicate dependency hash slot: {slot}")
        if not isinstance(fingerprint, str) or not _HASH_PATTERN.fullmatch(fingerprint):
            raise ValidationError(f"dependency hash {slot} must be a SHA-256 hex string")
        normalized[slot] = fingerprint

    # The context helper treats seed like any other runtime dependency, while
    # the cache schema stores its value explicitly.  Accept the derived hash as
    # a consistency assertion, then remove it so omission cannot create a
    # second key for the same seed.
    supplied_seed_hash = normalized.pop("seed", None)
    if supplied_seed_hash is not None and supplied_seed_hash != content_hash(seed):
        raise ValidationError("dependency seed hash does not match the explicit cache seed")
    return normalized


def _normalize_extra(extra: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if extra is None:
        return freeze_mapping(None)
    if not isinstance(extra, Mapping):
        raise ValidationError("cache extra must be a mapping or None")
    try:
        normalized = freeze_mapping(extra)
        # ``deep_freeze_jsonable`` rejects unsupported values, while canonical
        # serialization additionally rejects non-finite floats.
        content_hash(normalized)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"cache extra must be canonically serializable: {exc}") from exc
    return normalized


def _canonical_mapping_hash(
    value: Mapping[str, Any],
    *,
    label: str,
    require_non_empty: bool,
) -> str:
    try:
        normalized = freeze_mapping(value)
        if require_non_empty and not normalized:
            raise ValidationError(f"{label} must be a non-empty mapping")
        return content_hash(normalized)
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be canonically serializable: {exc}") from exc
