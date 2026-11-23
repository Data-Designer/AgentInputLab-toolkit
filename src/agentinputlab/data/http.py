"""Pinned-address HTTPS transport for explicit benchmark acquisition.

This transport is opt-in.  It connects to one of the addresses validated by
``validate_download_url`` while retaining the original hostname for TLS SNI and
certificate verification.  It does not use environment proxies, follow
redirects, or decompress response bodies.
"""

from __future__ import annotations

import http.client
import socket
import ssl
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field

from agentinputlab.contracts.errors import ArtifactError

from .downloader import (
    DownloadResponse,
    StreamingDownloadResponse,
    ValidatedDownloadTarget,
    _validate_download_limits,
)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        target: ValidatedDownloadTarget,
        address: str,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(
            target.hostname,
            target.port,
            timeout=timeout,
            context=context,
        )
        self._validated_address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except BaseException:
            raw_socket.close()
            raise


@dataclass(slots=True)
class PinnedHTTPSDownloadTransport:
    """One-hop stdlib HTTPS client honoring the downloader's DNS pin."""

    user_agent: str = "AgentInputLab/0.1 benchmark-acquisition"
    ssl_context: ssl.SSLContext = field(default_factory=ssl.create_default_context)

    def request(
        self,
        target: ValidatedDownloadTarget,
        *,
        timeout_seconds: float,
        max_bytes: int,
    ) -> DownloadResponse:
        with self.stream(target, timeout_seconds=timeout_seconds, max_bytes=max_bytes) as response:
            return DownloadResponse(
                status_code=response.status_code,
                headers=response.headers,
                body=b"".join(response.chunks),
                peer_address=response.peer_address,
            )

    @contextmanager
    def stream(
        self,
        target: ValidatedDownloadTarget,
        *,
        timeout_seconds: float,
        max_bytes: int,
    ):
        """Stream one validated address, without redirect or body-read retries."""
        _validate_download_limits(max_bytes, timeout_seconds)
        parsed = urllib.parse.urlsplit(target.url)
        request_target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        host_header = target.hostname if target.port == 443 else f"{target.hostname}:{target.port}"
        failures: list[str] = []
        for address in target.resolved_addresses:
            connection = _PinnedHTTPSConnection(
                target,
                address,
                timeout=timeout_seconds,
                context=self.ssl_context,
            )
            try:
                connection.request(
                    "GET",
                    request_target,
                    headers={
                        "Accept": "*/*",
                        "Accept-Encoding": "identity",
                        "Connection": "close",
                        "Host": host_header,
                        "User-Agent": self.user_agent,
                    },
                )
                if connection.sock is None:  # pragma: no cover - http.client invariant
                    raise ArtifactError("HTTPS transport did not establish a socket")
                peer_address = str(connection.sock.getpeername()[0])
                response = connection.getresponse()
                headers = {key: value for key, value in response.getheaders()}
                encoding = _header(headers, "content-encoding")
                if encoding and encoding.lower() != "identity":
                    raise ArtifactError("download transport refuses encoded response bodies")
                declared = _header(headers, "content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as exc:
                        raise ArtifactError(
                            "download response has an invalid Content-Length"
                        ) from exc
                    if declared_size < 0 or declared_size > max_bytes:
                        raise ArtifactError("download response exceeds the transport size limit")
            except ArtifactError:
                connection.close()
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                connection.close()
                failures.append(f"{address}: {type(exc).__name__}")
                continue
            break
        else:
            detail = ", ".join(failures) if failures else "no validated addresses"
            raise ArtifactError(f"HTTPS transport could not connect ({detail})")
        try:
            yield StreamingDownloadResponse(
                status_code=response.status,
                headers=headers,
                chunks=_bounded_chunks(response, max_bytes=max_bytes),
                peer_address=peer_address,
            )
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise ArtifactError(f"artifact stream interrupted ({type(exc).__name__})") from exc
        finally:
            connection.close()


def _bounded_chunks(response: http.client.HTTPResponse, *, max_bytes: int):
    size = 0
    while True:
        chunk = response.read(min(1024 * 1024, max_bytes + 1 - size))
        if not chunk:
            return
        size += len(chunk)
        if size > max_bytes:
            raise ArtifactError("download response exceeds the transport size limit")
        yield chunk


def _header(headers: dict[str, str], name: str) -> str | None:
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected:
            return value
    return None
