"""Runtime wrapper that keeps preflight identities attached through persistence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import freeze_mapping

from .bindings import (
    FrozenBindingReceipt,
    build_guarded_driver_fingerprint,
    resolve_frozen_binding_receipt,
)
from .executor import ExperimentCellResult, ExperimentDriver
from .parity_identity import (
    HOST_EXECUTION_RECEIPT_PROTOCOL,
    ParityImplementationContract,
    validate_host_process_cell_result,
    validate_parity_cell_result,
)
from .plan import ExperimentCell
from .schema import ExperimentManifest

_GUARDED_ENTRYPOINT = "agentinputlab.experiments.driver_guard:active_guarded_driver_factory"


@dataclass(frozen=True, slots=True)
class _GuardConfiguration:
    manifest: ExperimentManifest
    operator_entrypoint: str
    operator_fingerprint: Mapping[str, Any]
    binding_receipt: FrozenBindingReceipt
    parity_contract: ParityImplementationContract | None

    def __post_init__(self) -> None:
        if self.operator_entrypoint == _GUARDED_ENTRYPOINT:
            raise ArtifactError("guarded driver cannot recursively wrap itself")
        try:
            fingerprint = freeze_mapping(self.operator_fingerprint)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("operator driver fingerprint is not canonical JSON") from exc
        if not fingerprint:
            raise ArtifactError("operator driver fingerprint must not be empty")
        object.__setattr__(self, "operator_fingerprint", fingerprint)


_ACTIVE_GUARD: ContextVar[_GuardConfiguration | None] = ContextVar(
    "agentinputlab_active_experiment_driver_guard",
    default=None,
)


class GuardedExperimentDriver:
    """Bind one operator driver to the exact receipt accepted by preflight."""

    def __init__(
        self,
        operator: ExperimentDriver,
        *,
        manifest: ExperimentManifest,
        expected_operator_fingerprint: Mapping[str, Any],
        binding_receipt: FrozenBindingReceipt,
        parity_contract: ParityImplementationContract | None = None,
    ) -> None:
        if not isinstance(operator, ExperimentDriver):
            raise ArtifactError("guard requires an ExperimentDriver implementation")
        self._operator = operator
        self._manifest = manifest
        self._expected_operator_fingerprint = freeze_mapping(expected_operator_fingerprint)
        self._binding_receipt = binding_receipt
        self._parity_contract = parity_contract
        self._require_stable_operator_fingerprint()
        self._fingerprint = build_guarded_driver_fingerprint(
            self._expected_operator_fingerprint,
            binding_receipt,
            parity_contract=(parity_contract.to_dict() if parity_contract is not None else None),
        )

    def fingerprint(self) -> Mapping[str, Any]:
        self._require_stable_operator_fingerprint()
        return self._fingerprint

    def preflight(self, manifest: ExperimentManifest) -> Sequence[str]:
        self._require_manifest(manifest)
        self._require_stable_bindings()
        self._require_stable_operator_fingerprint()
        blockers = self._operator.preflight(manifest)
        self._require_stable_operator_fingerprint()
        self._require_stable_bindings()
        return blockers

    def execute_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentCellResult:
        self._require_manifest(manifest)
        self._require_stable_bindings()
        self._require_stable_operator_fingerprint()
        result = self._operator.execute_cell(manifest, cell)
        self._require_stable_operator_fingerprint()
        self._require_stable_bindings()
        if self._parity_contract is not None and isinstance(result, ExperimentCellResult):
            validate_parity_cell_result(
                cell,
                result,
                binding_receipt=self._binding_receipt,
                contract=self._parity_contract,
                operator_fingerprint=self._expected_operator_fingerprint,
            )
        elif (
            isinstance(result, ExperimentCellResult)
            and self._expected_operator_fingerprint.get("execution_receipt_protocol")
            == HOST_EXECUTION_RECEIPT_PROTOCOL
        ):
            validate_host_process_cell_result(
                cell,
                result,
                binding_receipt=self._binding_receipt,
                operator_fingerprint=self._expected_operator_fingerprint,
            )
        return result

    def _require_manifest(self, manifest: ExperimentManifest) -> None:
        if manifest.manifest_hash != self._manifest.manifest_hash:
            raise ArtifactError("guarded driver received another experiment manifest")

    def _require_stable_operator_fingerprint(self) -> None:
        try:
            observed = freeze_mapping(self._operator.fingerprint())
        except (TypeError, ValueError) as exc:
            raise ArtifactError("operator driver fingerprint is not canonical JSON") from exc
        if observed != self._expected_operator_fingerprint:
            raise ArtifactError("operator driver fingerprint changed after frozen preflight")

    def _require_stable_bindings(self) -> None:
        """Re-attest every prerequisite immediately around each cell."""

        current = resolve_frozen_binding_receipt(self._manifest)
        if current.to_dict() != self._binding_receipt.to_dict():
            raise ArtifactError("experiment prerequisite bindings changed after frozen preflight")


@contextmanager
def activate_guarded_driver(
    *,
    manifest: ExperimentManifest,
    operator_entrypoint: str,
    operator_fingerprint: Mapping[str, Any],
    binding_receipt: FrozenBindingReceipt,
    parity_contract: ParityImplementationContract | None = None,
) -> Iterator[str]:
    """Install a process-local factory used only around one synchronous run."""

    configuration = _GuardConfiguration(
        manifest=manifest,
        operator_entrypoint=operator_entrypoint,
        operator_fingerprint=operator_fingerprint,
        binding_receipt=binding_receipt,
        parity_contract=parity_contract,
    )
    token = _ACTIVE_GUARD.set(configuration)
    try:
        yield _GUARDED_ENTRYPOINT
    finally:
        _ACTIVE_GUARD.reset(token)


def active_guarded_driver_factory() -> GuardedExperimentDriver:
    """Zero-argument factory consumed by the existing executor entrypoint API."""

    configuration = _ACTIVE_GUARD.get()
    if configuration is None:
        raise ArtifactError("guarded driver factory is inactive; use activate_guarded_driver()")
    current_receipt = resolve_frozen_binding_receipt(configuration.manifest)
    if current_receipt.to_dict() != configuration.binding_receipt.to_dict():
        raise ArtifactError("experiment prerequisite bindings changed after frozen preflight")
    from .loader import resolve_driver_entrypoint

    factory = resolve_driver_entrypoint(configuration.operator_entrypoint)
    if not callable(factory):
        raise ArtifactError("operator driver entrypoint must be a zero-argument factory")
    try:
        operator = factory()
    except TypeError as exc:
        raise ArtifactError("operator driver entrypoint must be a zero-argument factory") from exc
    return GuardedExperimentDriver(
        operator,
        manifest=configuration.manifest,
        expected_operator_fingerprint=configuration.operator_fingerprint,
        binding_receipt=configuration.binding_receipt,
        parity_contract=configuration.parity_contract,
    )


__all__ = [
    "GuardedExperimentDriver",
    "activate_guarded_driver",
    "active_guarded_driver_factory",
]
