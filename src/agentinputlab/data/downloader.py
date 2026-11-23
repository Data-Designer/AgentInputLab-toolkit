"""Fail-closed artifact acquisition with an injected hardened transport.

The core package deliberately does not ship a generic ``urlopen`` transport.
An injected transport receives already-resolved targets, must connect to one of
those addresses while validating TLS for the original hostname, and must not
follow redirects itself. AgentInputLab validates every redirect hop and the
reported peer before accepting bytes.
"""

from __future__ import annotations

import ipaddress
import math
import socket
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentinputlab.artifacts import ArtifactManifest, ArtifactStore
from agentinputlab.contracts.errors import ArtifactError


@dataclass(frozen=True, slots=True)
class DownloadPolicy:
    allow_network: bool = False
    allowed_hosts: tuple[str, ...] = ()
    allow_private_addresses: bool = False
    max_bytes: int = 2 * 1024 * 1024 * 1024
    timeout_seconds: float = 120.0
    max_redirects: int = 3

    def __post_init__(self) -> None:
        if type(self.allow_network) is not bool or type(self.allow_private_addresses) is not bool:
            raise ArtifactError("download network permissions must be booleans")
        if (isinstance(self.allowed_hosts, (str, bytes))
                or not isinstance(self.allowed_hosts, Sequence)
                or any(not isinstance(host, str) or not host.strip() for host in self.allowed_hosts)):
            raise ArtifactError("download allowed_hosts must be a sequence of non-empty host strings")
        normalized = tuple(sorted({host.strip().lower() for host in self.allowed_hosts if host}))
        object.__setattr__(self, "allowed_hosts", normalized)
        _validate_download_limits(self.max_bytes, self.timeout_seconds)
        if type(self.max_redirects) is not int or self.max_redirects < 0:
            raise ArtifactError("download max_redirects must be a non-negative integer")


def _validate_download_limits(max_bytes: int, timeout_seconds: float) -> None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ArtifactError("download max_bytes must be a positive integer")
    try:
        valid_timeout = (type(timeout_seconds) in (int, float)
                         and math.isfinite(timeout_seconds) and timeout_seconds > 0)
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ArtifactError("download timeout_seconds must be a positive finite number")


@dataclass(frozen=True, slots=True)
class ValidatedDownloadTarget:
    url: str
    hostname: str
    port: int
    resolved_addresses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DownloadResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    peer_address: str


@dataclass(frozen=True, slots=True)
class StreamingDownloadResponse:
    status_code: int
    headers: Mapping[str, str]
    chunks: Iterable[bytes]
    peer_address: str


class StreamingDownloadTransport(Protocol):
    def stream(
        self, target: ValidatedDownloadTarget, *, timeout_seconds: float, max_bytes: int
    ) -> AbstractContextManager[StreamingDownloadResponse]:
        """Open one DNS-pinned hop; close it even when the consumer stops early."""


@runtime_checkable
class DownloadTransport(Protocol):
    def request(
        self,
        target: ValidatedDownloadTarget,
        *,
        timeout_seconds: float,
        max_bytes: int,
    ) -> DownloadResponse:
        """Fetch one hop without redirects or decompression expansion."""


def validate_download_url(url: str, policy: DownloadPolicy) -> ValidatedDownloadTarget:
    if not policy.allow_network:
        raise ArtifactError("network acquisition is disabled")
    if not policy.allowed_hosts:
        raise ArtifactError("network acquisition requires a non-empty host allowlist")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ArtifactError("artifact URLs must use absolute HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ArtifactError("artifact URLs must not contain user information")
    hostname = parsed.hostname.lower()
    if hostname not in policy.allowed_hosts:
        raise ArtifactError(f"host is not allowlisted: {hostname}")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ArtifactError("artifact URL contains an invalid port") from exc
    try:
        addresses = {
            info[4][0].split("%", 1)[0]
            for info in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise ArtifactError(f"cannot resolve artifact host: {hostname}") from exc
    if not addresses:
        raise ArtifactError(f"artifact host resolved to no addresses: {hostname}")
    if not policy.allow_private_addresses:
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ArtifactError(f"artifact host resolves to disallowed address: {address}")
    return ValidatedDownloadTarget(
        url=urllib.parse.urlunsplit(parsed),
        hostname=hostname,
        port=port,
        resolved_addresses=tuple(sorted(addresses)),
    )


def download_artifact(
    manifest: ArtifactManifest,
    store: ArtifactStore,
    *,
    policy: DownloadPolicy,
    transport: DownloadTransport | None = None,
) -> str:
    if not manifest.sha256:
        raise ArtifactError("downloaded artifacts require a pinned SHA-256")
    if transport is None:
        raise ArtifactError("network acquisition requires an explicit hardened transport")
    target = validate_download_url(manifest.source_uri, policy)
    response: DownloadResponse | None = None
    for redirect_count in range(policy.max_redirects + 1):
        response = transport.request(
            target,
            timeout_seconds=policy.timeout_seconds,
            max_bytes=policy.max_bytes,
        )
        _validate_peer(response.peer_address, target)
        if 300 <= response.status_code < 400:
            if redirect_count >= policy.max_redirects:
                raise ArtifactError("artifact download exceeded redirect limit")
            location = _header(response.headers, "location")
            if not location:
                raise ArtifactError("artifact redirect is missing Location")
            target = validate_download_url(
                urllib.parse.urljoin(target.url, location),
                policy,
            )
            continue
        if not 200 <= response.status_code < 300:
            raise ArtifactError(f"artifact download returned HTTP {response.status_code}")
        break
    if response is None:  # pragma: no cover - loop invariant
        raise ArtifactError("artifact transport returned no response")
    content_length = _header(response.headers, "content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise ArtifactError("artifact Content-Length is invalid") from exc
        if declared_size < 0 or declared_size > policy.max_bytes:
            raise ArtifactError("artifact exceeds configured size limit")
    if len(response.body) > policy.max_bytes:
        raise ArtifactError("artifact exceeds configured size limit")
    return str(store.put_bytes(manifest, response.body))


def _validate_peer(peer_address: str, target: ValidatedDownloadTarget) -> None:
    try:
        normalized = str(ipaddress.ip_address(peer_address.split("%", 1)[0]))
    except ValueError as exc:
        raise ArtifactError("download transport reported an invalid peer address") from exc
    allowed = {str(ipaddress.ip_address(address)) for address in target.resolved_addresses}
    if normalized not in allowed:
        raise ArtifactError("download peer does not match the validated DNS result")


def download_artifact_streamed(
    manifest: ArtifactManifest,
    store: ArtifactStore,
    *,
    policy: DownloadPolicy,
    transport: StreamingDownloadTransport,
) -> str:
    """Acquire a pinned file with bounded RAM and validation at every hop."""
    if not manifest.sha256:
        raise ArtifactError("downloaded artifacts require a pinned SHA-256")
    if manifest.size_bytes is not None and manifest.size_bytes > policy.max_bytes:
        raise ArtifactError("artifact exceeds configured size limit before download")
    target = validate_download_url(manifest.source_uri, policy)
    for count in range(policy.max_redirects + 1):
        with transport.stream(
            target, timeout_seconds=policy.timeout_seconds, max_bytes=policy.max_bytes
        ) as response:
            _validate_peer(response.peer_address, target)
            if 300 <= response.status_code < 400:
                if count >= policy.max_redirects:
                    raise ArtifactError("artifact download exceeded redirect limit")
                location = _header(response.headers, "location")
                if not location:
                    raise ArtifactError("artifact redirect is missing Location")
                target = validate_download_url(urllib.parse.urljoin(target.url, location), policy)
                continue
            if not 200 <= response.status_code < 300:
                raise ArtifactError(f"artifact download returned HTTP {response.status_code}")
            encoding = _header(response.headers, "content-encoding")
            if encoding and encoding.lower() != "identity":
                raise ArtifactError("artifact stream refuses encoded response bodies")
            declared = _header(response.headers, "content-length")
            if declared is not None:
                try:
                    size = int(declared)
                except ValueError as exc:
                    raise ArtifactError("artifact Content-Length is invalid") from exc
                if size < 0 or size > policy.max_bytes:
                    raise ArtifactError("artifact exceeds configured size limit")
                if manifest.size_bytes is not None and size != manifest.size_bytes:
                    raise ArtifactError("artifact declared size differs from pinned size")
            return str(store.put_stream(manifest, response.chunks, max_bytes=policy.max_bytes))
    raise ArtifactError("artifact download exceeded redirect limit")  # pragma: no cover


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected:
            return str(value)
    return None
