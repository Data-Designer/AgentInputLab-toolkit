from __future__ import annotations

import hashlib
import os
import traceback

import pytest

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts.errors import ArtifactError


def _load(tmp_path, raw, **kwargs):
    source = tmp_path / "artifact.json"
    source.write_bytes(raw)
    return load_pinned_json_mapping(
        source, expected_sha256=hashlib.sha256(raw).hexdigest(), **kwargs
    )


def test_pinned_json_accepts_finite_utf8_object_and_uppercase_digest(tmp_path):
    raw = '{"text":"证据", "values":[0, 1.25, true, null]}'.encode()
    source = tmp_path / "artifact.json"
    source.write_bytes(raw)
    result = load_pinned_json_mapping(
        source, expected_sha256=hashlib.sha256(raw).hexdigest().upper(), max_bytes=len(raw)
    )
    assert result == {"text": "证据", "values": [0, 1.25, True, None]}


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"null",
        b'{"x":1,"x":2}',
        b'{"n":{"x":1,"x":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e999}',
        b'{"x":-1e999}',
        b'{"x":"\xff"}',
        b"{",
    ],
)
def test_pinned_json_rejects_ambiguous_nonfinite_or_invalid_json(tmp_path, raw):
    with pytest.raises(ArtifactError, match="valid UTF-8 JSON object") as error:
        _load(tmp_path, raw)
    assert error.value.__context__ is None


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, None])
def test_pinned_json_rejects_invalid_limits_before_io(tmp_path, limit):
    with pytest.raises(ArtifactError, match="positive integer"):
        load_pinned_json_mapping(tmp_path / "absent", expected_sha256="a" * 64, max_bytes=limit)


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "g" * 64])
def test_pinned_json_requires_explicit_trusted_digest(tmp_path, digest):
    with pytest.raises(ArtifactError, match="SHA-256 hex digest"):
        load_pinned_json_mapping(tmp_path / "absent", expected_sha256=digest)


def test_pinned_json_rejects_wrong_checksum_and_oversized_file(tmp_path):
    source = tmp_path / "artifact.json"
    source.write_text('{"ok":true}')
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_pinned_json_mapping(source, expected_sha256="a" * 64)
    with pytest.raises(ArtifactError, match="bounded regular file"):
        load_pinned_json_mapping(source, expected_sha256="a" * 64, max_bytes=2)


def test_pinned_json_rejects_symlink_directory_and_fifo_without_opening(tmp_path):
    source = tmp_path / "artifact.json"
    source.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(source)
    targets = [link, tmp_path]
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        targets.append(fifo)
    for target in targets:
        with pytest.raises(ArtifactError, match="bounded regular file"):
            load_pinned_json_mapping(target, expected_sha256="a" * 64)


def test_pinned_json_diagnostics_do_not_include_file_content_or_path(tmp_path):
    secret = "private-canary-do-not-log"
    source = tmp_path / secret
    with pytest.raises(ArtifactError) as error:
        load_pinned_json_mapping(source, expected_sha256="a" * 64)
    assert error.value.__context__ is None
    assert secret not in "".join(traceback.format_exception(error.value))
    with pytest.raises(ArtifactError) as error:
        _load(tmp_path, ('{"' + secret + '":1,"' + secret + '":2}').encode())
    assert error.value.__context__ is None
    assert secret not in "".join(traceback.format_exception(error.value))
