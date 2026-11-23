"""Fail-closed secret screening for persisted experiment artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from agentinputlab.contracts.errors import ArtifactError

_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|password|client[_-]?secret)"
        r"=)[^&#\s]+"
    ),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|x[_-]?api[_-]?key|authorization|access[_-]?token|"
        r"refresh[_-]?token|password|passwd|client[_-]?secret|secret[_-]?key)"
        r"\b\s*[:=]\s*)['\"]?(?:bearer\s+)?[^\s,;'\"]{4,}"
    ),
)
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "token",
        "password",
        "passwd",
        "client_secret",
        "secret",
        "secret_key",
    }
)
_SECRET_ENVIRONMENT_NAME = re.compile(
    r"(?i)(?:^|_)(?:api_?key|authorization|credential|password|passwd|"
    r"secret|token)(?:$|_)"
)


def redact_secret_text(
    value: str,
    *,
    secret_values: Sequence[str] = (),
    limit: int = 2048,
) -> str:
    """Redact known values and common credential forms before persistence.

    Pattern matching is only a fallback: opaque credentials do not necessarily
    have a recognizable prefix.  Process drivers therefore pass the exact
    values of their explicitly allowlisted environment variables here. Empty
    values are ignored. Short values are retained by
    :func:`environment_secret_values` only for secret-bearing variable names.
    """

    result = value
    known_values = {item for item in secret_values if isinstance(item, str) and item}
    for secret in sorted(known_values, key=len, reverse=True):
        result = result.replace(secret, "<redacted>")
    for pattern in _SECRET_VALUE_PATTERNS:
        result = pattern.sub(_redacted_match, result)
    return result[:limit]


def environment_secret_values(
    environment: Mapping[str, str],
    names: Sequence[str],
) -> tuple[str, ...]:
    """Return opaque values for only the explicitly passed worker variables."""

    return tuple(
        dict.fromkeys(
            value
            for name in names
            if isinstance(name, str)
            and isinstance((value := environment.get(name)), str)
            and value
            and (len(value) >= 4 or _SECRET_ENVIRONMENT_NAME.search(name))
        )
    )


def require_secret_free(value: Any, *, label: str) -> None:
    """Reject credentials before a fingerprint or result reaches disk."""

    _walk(value, path=label)


def require_secret_values_absent(
    value: Any,
    *,
    secret_values: Sequence[str],
    label: str,
) -> None:
    """Reject opaque passed-environment values before accepting worker output."""

    known_values = tuple(
        sorted(
            {item for item in secret_values if isinstance(item, str) and item},
            key=len,
            reverse=True,
        )
    )
    if known_values:
        _walk_known_values(
            value,
            path=label,
            known_values=known_values,
        )


def _walk(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized = key.casefold().replace("-", "_")
            if normalized in _SECRET_KEYS:
                raise ArtifactError(f"{path} contains forbidden secret field {key!r}")
            _walk(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _walk(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and redact_secret_text(value, limit=len(value)) != value:
        raise ArtifactError(f"{path} contains credential-shaped text")


def _walk_known_values(
    value: Any,
    *,
    path: str,
    known_values: tuple[str, ...],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _walk_known_values(
                    key,
                    path=f"{path}.<key>",
                    known_values=known_values,
                )
                _walk_known_values(
                    item,
                    path=f"{path}.{key}",
                    known_values=known_values,
                )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _walk_known_values(
                item,
                path=f"{path}[{index}]",
                known_values=known_values,
            )
        return
    if isinstance(value, str) and any(secret in value for secret in known_values):
        raise ArtifactError(f"{path} contains a value copied from the worker environment")


def _redacted_match(match: re.Match[str]) -> str:
    prefix = match.group(1) if match.lastindex else ""
    return f"{prefix}<redacted>"


__all__ = [
    "environment_secret_values",
    "redact_secret_text",
    "require_secret_free",
    "require_secret_values_absent",
]
