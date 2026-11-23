"""Acquisition limits must fail before any network or filesystem side effect."""

from unittest.mock import patch

import pytest

from agentinputlab.artifacts import AcquisitionPolicy, ArtifactKind, ArtifactManifest, ArtifactStore
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.data import DownloadPolicy, PinnedHTTPSDownloadTransport, ValidatedDownloadTarget


@pytest.mark.parametrize("field,value", [
    ("allow_network", "false"), ("allow_private_addresses", "false"),
    ("allowed_hosts", "example.com"), ("allowed_hosts", [123]),
    *[(key, value) for key in ("max_bytes", "max_redirects")
      for value in (True, 1.5, float("nan"), float("inf"), "10")],
    *[("timeout_seconds", value) for value in (True, float("nan"), float("inf"), "10")],
])
def test_invalid_download_policy_is_rejected(field, value):
    with pytest.raises(ArtifactError):
        DownloadPolicy(**{field: value})


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), "10", -1])
def test_artifact_size_is_an_integer(value):
    with pytest.raises(ValidationError):
        _manifest(size_bytes=value)


def _manifest(**extra):
    return ArtifactManifest(
        artifact_id="bounded", kind=ArtifactKind.DATASET, revision="v1", sha256="",
        acquisition=AcquisitionPolicy.USER_PROVIDED, license_id="MIT", **extra,
    )


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), "10", -1])
def test_invalid_stream_limit_does_not_consume_input(tmp_path, value):
    consumed = []

    def chunks():
        consumed.append(True)
        yield b"payload"

    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError):
        store.put_stream(_manifest(), chunks(), max_bytes=value)
    assert consumed == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    *[("max_bytes", value) for value in (True, 1.5, float("nan"), float("inf"), "10")],
    *[("timeout_seconds", value) for value in (True, float("nan"), float("inf"), "10", 0)],
])
def test_direct_transport_rejects_invalid_limits_before_connect(field, value):
    target = ValidatedDownloadTarget("https://example.com/x", "example.com", 443, ("93.184.216.34",))
    options = {"timeout_seconds": 10, "max_bytes": 10, field: value}
    with patch("agentinputlab.data.http._PinnedHTTPSConnection", side_effect=AssertionError("unexpected connect")) as connection:
        with pytest.raises(ArtifactError):
            PinnedHTTPSDownloadTransport().request(target, **options)
        connection.assert_not_called()


def test_valid_limits_keep_zero_size_and_zero_redirect_support(tmp_path):
    policy = DownloadPolicy(allowed_hosts=[" EXAMPLE.COM "], max_redirects=0, timeout_seconds=0.5)
    assert policy.allowed_hosts == ("example.com",)
    assert ArtifactStore(tmp_path).put_stream(_manifest(size_bytes=0), [], max_bytes=0).read_bytes() == b""
