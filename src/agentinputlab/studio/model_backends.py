"""Operator-owned model-backend gate for the ordinary Studio run surface.

The browser may select only a normalized backend ID.  Endpoint URLs, local
paths, credentials, environment-variable names, and builder parameters are
loaded exclusively from one checksum-locked server-side file.  The file must
itself be valid ``ProductionBackendBuilder`` parameters.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.backends.llm import (
    GenerationRequest,
    GenerationResponse,
    LLMBackend,
    ModelCapability,
)
from agentinputlab.contracts.errors import (
    AgentInputLabError,
    ArtifactError,
    CapabilityError,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.experiments.drivers.static_production import (
    StaticBackendBuildRequest,
    static_runtime_product_sha256,
)
from agentinputlab.experiments.production_builders import (
    ProductionBackendBuilder,
    builder_fingerprint_sha256,
)

STUDIO_MODEL_BACKEND_CONFIG_ENV = "AGENTINPUTLAB_STUDIO_MODEL_BACKEND_CONFIG"
STUDIO_MODEL_BACKEND_CONFIG_SHA256_ENV = "AGENTINPUTLAB_STUDIO_MODEL_BACKEND_CONFIG_SHA256"

_MAX_CONFIG_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_BACKEND_IDS = MappingProxyType(
    {
        "openai_compatible": "openai-compatible",
        "transformers": "transformers-local",
    }
)


class StudioModelBackendUnavailable(RuntimeError):
    """Raised without disclosing private operator configuration."""


class _PublicFingerprintBackend:
    """Delegate generation while exposing only a cryptographic public identity."""

    __slots__ = ("_deterministic_only", "_inner", "_public_fingerprint")

    def __init__(
        self,
        inner: LLMBackend,
        public_fingerprint: Mapping[str, Any],
        *,
        deterministic_only: bool,
    ) -> None:
        if not isinstance(inner, LLMBackend):
            raise TypeError("Studio model runtime requires an LLMBackend")
        self._inner = inner
        self._public_fingerprint = freeze_mapping(public_fingerprint)
        self._deterministic_only = deterministic_only

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        if not self._deterministic_only:
            return self._inner.capabilities
        return frozenset(
            capability
            for capability in self._inner.capabilities
            if capability is not ModelCapability.SEED
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._deterministic_only:
            if request.temperature != 0:
                raise CapabilityError("Studio local Transformers execution requires temperature=0")
            # The core Transformers backend correctly refuses a stochastic
            # seed it cannot enforce.  Studio permits only greedy decoding and
            # removes the irrelevant seed before delegation.
            request = replace(request, seed=None)
        return self._inner.generate(request)

    def fingerprint(self) -> Mapping[str, Any]:
        return self._public_fingerprint


@dataclass(frozen=True, slots=True)
class StudioModelRuntime:
    """A fresh runtime plus its path- and secret-free audit receipt."""

    backend_id: str
    backend: LLMBackend
    run_config: Mapping[str, Any]
    intent_contract: Any
    pricing_catalog: Any
    strict_intent: bool
    public_fingerprint: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.backend_id not in {"openai-compatible", "transformers-local"}:
            raise ValidationError("unsupported Studio model backend ID")
        if not isinstance(self.backend, LLMBackend):
            raise TypeError("Studio model runtime backend must implement LLMBackend")
        object.__setattr__(self, "run_config", freeze_mapping(self.run_config))
        object.__setattr__(
            self,
            "public_fingerprint",
            freeze_mapping(self.public_fingerprint),
        )


class StudioModelBackendGate:
    """Fail-closed server-owned gate for one production model deployment."""

    __slots__ = (
        "_backend_id",
        "_builder_fingerprint_sha256",
        "_config_path",
        "_config_sha256",
        "_expected_product_sha256",
        "_locked_reason",
        "_parameters",
    )

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        config_path: Path | None = None,
        config_sha256: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        builder_fingerprint: str | None = None,
        expected_product_sha256: str | None = None,
        locked_reason: str,
    ) -> None:
        self._backend_id = backend_id
        self._config_path = config_path
        self._config_sha256 = config_sha256
        self._parameters = (
            freeze_mapping(parameters) if parameters is not None else MappingProxyType({})
        )
        self._builder_fingerprint_sha256 = builder_fingerprint
        self._expected_product_sha256 = expected_product_sha256
        self._locked_reason = locked_reason

    @classmethod
    def from_operator_config(
        cls,
        path: str | Path | None,
        expected_sha256: str | None,
    ) -> StudioModelBackendGate:
        """Load, attest, preflight, and dry-build one operator configuration.

        Every failure becomes a generic catalog reason.  Private paths,
        endpoint values, environment-variable names, and parser details never
        cross the Studio API boundary.
        """

        if path is None and expected_sha256 is None:
            return cls(locked_reason="No operator-owned model backend is configured.")
        if path is None or expected_sha256 is None:
            return cls(
                locked_reason=(
                    "The operator model backend requires both a config path and SHA-256."
                )
            )
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            return cls(locked_reason="The operator model backend SHA-256 is invalid.")
        try:
            config_path = Path(path).expanduser()
            if not config_path.is_absolute():
                return cls(locked_reason="The operator model backend config path must be absolute.")
            config_path = config_path.resolve(strict=True)
            raw = _read_locked_file(config_path, expected_sha256)
            parameters = _decode_parameters(raw)
            # Ordinary Studio never resolves operator-supplied entrypoints.
            # Reject this field before ProductionBackendBuilder.preflight()
            # can inspect a nested retriever builder.
            if parameters.get("retrievers"):
                return cls(
                    locked_reason=(
                        "Ordinary Studio model configs cannot load retriever entrypoints."
                    )
                )
            builder = ProductionBackendBuilder()
            blockers = tuple(builder.preflight(parameters))
            if blockers:
                return cls(
                    locked_reason=(
                        "The operator model backend did not pass "
                        "ProductionBackendBuilder preflight."
                    )
                )
            backend_kind = _backend_kind(parameters)
            backend_id = _SUPPORTED_BACKEND_IDS.get(backend_kind)
            if backend_id is None:
                return cls(
                    locked_reason=(
                        "The operator model backend must be OpenAI-compatible "
                        "or local Transformers."
                    )
                )
            request = StaticBackendBuildRequest(
                model_slot=f"studio:{backend_id}",
                seed=0,
                builder_parameters=parameters,
            )
            runtime = builder.build(request)
            required_runtime_capabilities = {ModelCapability.CHAT}
            if backend_id == "openai-compatible":
                required_runtime_capabilities.add(ModelCapability.SEED)
            missing = required_runtime_capabilities - set(runtime.backend.capabilities)
            if missing:
                names = ", ".join(sorted(capability.value for capability in missing))
                raise CapabilityError(f"ordinary Studio model backend lacks capabilities: {names}")
            product_sha256 = static_runtime_product_sha256(runtime)
            return cls(
                backend_id=backend_id,
                config_path=config_path,
                config_sha256=expected_sha256,
                parameters=parameters,
                builder_fingerprint=builder_fingerprint_sha256(builder),
                expected_product_sha256=product_sha256,
                locked_reason="",
            )
        except (
            AgentInputLabError,
            json.JSONDecodeError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
        ):
            return cls(
                locked_reason=(
                    "The operator model backend config could not be attested or validated."
                )
            )

    @property
    def ready_backend_id(self) -> str | None:
        return self._backend_id

    @property
    def network_enabled(self) -> bool:
        return self._backend_id == "openai-compatible"

    @property
    def locked_reason(self) -> str:
        return self._locked_reason or "No attested deployment is configured for this backend."

    def catalog_entry(self, backend_id: str) -> Mapping[str, Any]:
        if backend_id not in {"openai-compatible", "transformers-local"}:
            raise ValueError(f"unknown gated Studio backend: {backend_id}")
        ready = self._backend_id == backend_id
        if ready:
            description = (
                "Operator-attested ProductionBackendBuilder deployment; "
                "private endpoint, paths, and credentials remain server-side."
            )
        elif self._backend_id is not None:
            description = (
                f"The configured operator deployment enables {self._backend_id}, not {backend_id}."
            )
        else:
            description = self.locked_reason
        return freeze_mapping(
            {
                "id": backend_id,
                "name": (
                    "OpenAI-compatible / vLLM"
                    if backend_id == "openai-compatible"
                    else "Transformers local"
                ),
                "ready": ready,
                "network": backend_id == "openai-compatible",
                "description": description,
                "operator_attested": ready,
            }
        )

    def build(self, backend_id: str, *, seed: int) -> StudioModelRuntime:
        """Re-attest and construct the exact runtime selected by normalized ID."""

        if backend_id != self._backend_id or self._config_path is None:
            raise StudioModelBackendUnavailable(
                "the selected operator-owned model backend is unavailable"
            )
        if (
            self._config_sha256 is None
            or self._builder_fingerprint_sha256 is None
            or self._expected_product_sha256 is None
        ):
            raise StudioModelBackendUnavailable(
                "the selected operator-owned model backend is unavailable"
            )
        try:
            raw = _read_locked_file(self._config_path, self._config_sha256)
            parameters = _decode_parameters(raw)
            if content_hash(parameters) != content_hash(self._parameters):
                raise ArtifactError("operator backend parameters drifted")
            builder = ProductionBackendBuilder()
            if builder_fingerprint_sha256(builder) != self._builder_fingerprint_sha256:
                raise ArtifactError("production backend builder fingerprint drifted")
            blockers = tuple(builder.preflight(parameters))
            if blockers:
                raise ArtifactError("production backend preflight failed")
            runtime = builder.build(
                StaticBackendBuildRequest(
                    model_slot=f"studio:{backend_id}",
                    seed=seed,
                    builder_parameters=parameters,
                )
            )
            runtime_product_sha256 = static_runtime_product_sha256(runtime)
            if runtime_product_sha256 != self._expected_product_sha256:
                raise ArtifactError("production backend runtime product drifted")
            actual_backend_fingerprint_sha256 = content_hash(runtime.backend.fingerprint())
            deterministic_only = backend_id == "transformers-local"
            public_capabilities = (
                set(runtime.backend.capabilities) - {ModelCapability.SEED}
                if deterministic_only
                else set(runtime.backend.capabilities)
            )
            public_fingerprint = freeze_mapping(
                {
                    "implementation": "agentinputlab.studio.operator_backend.v1",
                    "backend_id": backend_id,
                    "config_sha256": self._config_sha256,
                    "builder_fingerprint_sha256": self._builder_fingerprint_sha256,
                    "runtime_product_sha256": runtime_product_sha256,
                    "actual_backend_fingerprint_sha256": (actual_backend_fingerprint_sha256),
                    "model_id_sha256": hashlib.sha256(
                        runtime.backend.model_id.encode("utf-8")
                    ).hexdigest(),
                    "capabilities": sorted(capability.value for capability in public_capabilities),
                    "seed_policy": (
                        "deterministic_decoding_seed_not_forwarded.v1"
                        if deterministic_only
                        else "request_seed_forwarded.v1"
                    ),
                }
            )
            return StudioModelRuntime(
                backend_id=backend_id,
                backend=_PublicFingerprintBackend(
                    runtime.backend,
                    public_fingerprint,
                    deterministic_only=deterministic_only,
                ),
                run_config=runtime.run_config,
                intent_contract=runtime.intent_contract,
                pricing_catalog=runtime.pricing_catalog,
                strict_intent=runtime.strict_intent,
                public_fingerprint=public_fingerprint,
            )
        except (
            AgentInputLabError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise StudioModelBackendUnavailable(
                "the selected operator-owned model backend failed re-attestation"
            ) from exc


def _read_locked_file(path: Path, expected_sha256: str) -> bytes:
    try:
        if not path.is_file():
            raise ArtifactError("operator model backend config is not a file")
        size = path.stat().st_size
        if size < 2 or size > _MAX_CONFIG_BYTES:
            raise ArtifactError("operator model backend config size is invalid")
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactError("operator model backend config cannot be read") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ArtifactError("operator model backend config checksum mismatch")
    return raw


def _decode_parameters(raw: bytes) -> Mapping[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValidationError("operator model backend config must be a JSON object")
    return freeze_mapping(value)


def _backend_kind(parameters: Mapping[str, Any]) -> str:
    backend = parameters.get("backend")
    if not isinstance(backend, Mapping):
        raise ValidationError("operator model backend config requires backend")
    kind = backend.get("kind")
    if not isinstance(kind, str):
        raise ValidationError("operator model backend kind must be text")
    return kind


__all__ = [
    "STUDIO_MODEL_BACKEND_CONFIG_ENV",
    "STUDIO_MODEL_BACKEND_CONFIG_SHA256_ENV",
    "StudioModelBackendGate",
    "StudioModelBackendUnavailable",
    "StudioModelRuntime",
]
