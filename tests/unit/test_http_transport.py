from __future__ import annotations

import unittest
from unittest.mock import patch

from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.data import PinnedHTTPSDownloadTransport, ValidatedDownloadTarget


class _Socket:
    def getpeername(self):
        return ("93.184.216.34", 443)


class _Response:
    def __init__(self, body: bytes, *, headers=(), status=200):
        self._body = body
        self._read = False
        self._headers = list(headers)
        self.status = status

    def getheaders(self):
        return self._headers

    def read(self, _size):
        if self._read:
            return b""
        self._read = True
        return self._body


class _Connection:
    def __init__(self, response: _Response):
        self.response = response
        self.sock = _Socket()
        self.requests = []
        self.closed = False

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def _target() -> ValidatedDownloadTarget:
    return ValidatedDownloadTarget(
        url="https://example.com/data?q=1",
        hostname="example.com",
        port=443,
        resolved_addresses=("93.184.216.34",),
    )


class PinnedHTTPSTransportTests(unittest.TestCase):
    def test_stream_is_lazy_and_closes_when_not_consumed(self) -> None:
        body = _Response(b"verified", headers=(("Content-Length", "8"),))
        connection = _Connection(body)
        with (
            patch("agentinputlab.data.http._PinnedHTTPSConnection", return_value=connection),
            PinnedHTTPSDownloadTransport().stream(
                _target(), timeout_seconds=10, max_bytes=16
            ) as response,
        ):
            self.assertEqual(response.status_code, 200)
            self.assertFalse(body._read)
        self.assertTrue(connection.closed)
        self.assertFalse(body._read)

    def test_one_hop_request_reports_peer_and_disables_decompression(self) -> None:
        connection = _Connection(_Response(b"verified", headers=(("Content-Length", "8"),)))
        with patch("agentinputlab.data.http._PinnedHTTPSConnection", return_value=connection):
            response = PinnedHTTPSDownloadTransport().request(
                _target(), timeout_seconds=10, max_bytes=16
            )
        self.assertEqual(response.body, b"verified")
        self.assertEqual(response.peer_address, "93.184.216.34")
        self.assertEqual(connection.requests[0][0], ("GET", "/data?q=1"))
        headers = connection.requests[0][1]["headers"]
        self.assertEqual(headers["Accept-Encoding"], "identity")
        self.assertEqual(headers["Host"], "example.com")
        self.assertTrue(connection.closed)

    def test_encoded_response_is_rejected(self) -> None:
        connection = _Connection(_Response(b"compressed", headers=(("Content-Encoding", "gzip"),)))
        with (
            patch("agentinputlab.data.http._PinnedHTTPSConnection", return_value=connection),
            self.assertRaisesRegex(ArtifactError, "encoded response"),
        ):
            PinnedHTTPSDownloadTransport().request(_target(), timeout_seconds=10, max_bytes=32)

    def test_streamed_body_limit_is_enforced_without_content_length(self) -> None:
        connection = _Connection(_Response(b"too large"))
        with (
            patch("agentinputlab.data.http._PinnedHTTPSConnection", return_value=connection),
            self.assertRaisesRegex(ArtifactError, "size limit"),
        ):
            PinnedHTTPSDownloadTransport().request(_target(), timeout_seconds=10, max_bytes=4)


if __name__ == "__main__":
    unittest.main()
