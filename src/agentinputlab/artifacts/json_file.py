"""Bounded checksum-pinned JSON loading without pickle or executable imports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import ArtifactError


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object keys")
        result[key] = value
    return result


def _finite_json(_value):
    raise ValueError("non-finite JSON numbers")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON numbers")
    return number


def load_pinned_json_mapping(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Read one regular UTF-8 JSON object under a caller-trusted byte checksum.

    The checksum must come from a trusted receipt, not be computed from an
    untrusted file immediately before loading. It proves byte identity, not
    authorship, source-license permission, or the truth of recorded results.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256.lower())
    ):
        raise ArtifactError("expected_sha256 must be a SHA-256 hex digest")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ArtifactError("artifact JSON max_bytes must be a positive integer")
    source = Path(path).expanduser()
    read_failed = False
    try:
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ArtifactError("artifact JSON must be a bounded regular file, not a symlink")
        descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ArtifactError("artifact JSON file changed while opening")
            raw = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
            if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ArtifactError("artifact JSON file changed while reading")
    except OSError:
        read_failed = True
    if read_failed:
        raise ArtifactError("artifact JSON file is unavailable")
    if len(raw) > max_bytes:
        raise ArtifactError("artifact JSON exceeds max_bytes")
    if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise ArtifactError("artifact JSON checksum mismatch")
    invalid_json = False
    value = None
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_finite_json,
            parse_float=_finite_float,
        )
    except (UnicodeError, ValueError, RecursionError):
        invalid_json = True
    if invalid_json or not isinstance(value, dict):
        raise ArtifactError(
            "artifact must contain a valid UTF-8 JSON object with unique keys and finite values"
        )
    return value
