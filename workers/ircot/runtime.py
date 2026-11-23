"""Core ``WorkerClient`` adapter for the pinned IRCoT driver."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.contracts import ExternalDependencyError
from agentinputlab.integrations.external_workers.ircot import IRCoTWorkerRequest

from .driver import (
    DEFAULT_MANIFEST,
    DEFAULT_PROFILE,
    PROTOCOL_VERSION,
    UPSTREAM_COMMIT,
    DriverError,
    IRCoTDriver,
)


class PinnedIRCoTWorker:
    """Expose the manifest-locked fixture profile through ``WorkerClient``.

    The direct adapter is intentionally suitable only for the frozen profile.
    Upstream-native profiles fail closed in :class:`IRCoTDriver` unless the
    driver is launched as its isolated JSONL process.
    """

    worker_id = "agentinputlab.ircot.pinned.in_process.v1"

    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_MANIFEST,
        profile_id: str = DEFAULT_PROFILE,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.driver = IRCoTDriver(
            manifest_path=manifest_path,
            profile_id=profile_id,
            environment=environment,
            isolated_process=False,
        )

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the driver's structured success or fail-closed response."""

        return self.driver.safe_call(payload)

    def call_controlled(self, request: IRCoTWorkerRequest) -> Mapping[str, Any]:
        """Call the driver with the bridge's immutable request contract."""

        if not isinstance(request, IRCoTWorkerRequest):
            raise TypeError("PinnedIRCoTWorker requires IRCoTWorkerRequest")
        return self.call(request.as_dict())

    def fingerprint(self) -> Mapping[str, Any]:
        """Return source, manifest, profile, and upstream execution identity."""

        try:
            identity = self.driver.identity()
            driver_sha256 = _sha256_file(Path(__file__).with_name("driver.py"))
            adapter_sha256 = _sha256_file(Path(__file__))
        except DriverError as exc:
            raise ExternalDependencyError(
                f"pinned IRCoT runtime identity failed with {exc.code}"
            ) from exc
        return {
            "worker_id": self.worker_id,
            "execution_mode": "in_process_external_environment",
            "protocol_version": PROTOCOL_VERSION,
            "upstream_revision": UPSTREAM_COMMIT,
            "profile_id": identity["profile_id"],
            "identity_kind": identity["identity_kind"],
            "compatibility_manifest_sha256": identity["manifest_sha256"],
            "config_sha256": identity["config_sha256"],
            "prompt_sha256": identity["prompt_sha256"],
            "binding_attestation": identity["binding_attestation"],
            "full_api_status": identity["full_api_status"],
            "driver_sha256": driver_sha256,
            "runtime_adapter_sha256": adapter_sha256,
        }


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DriverError(
            "RUNTIME_SOURCE_MISSING",
            "IRCoT runtime source file is unavailable",
        ) from exc


__all__ = ["PinnedIRCoTWorker"]
