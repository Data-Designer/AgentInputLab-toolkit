"""Execute a registered intervention before entering an official external arena.

The external-arena transport deliberately keeps the protected benchmark
runtime in a separate process.  A release name alone is not an intervention,
however: the selected algorithm must first run against the exact frozen,
optimizer-visible state.  This builder performs that missing production step.

It reuses the same checksum-locked backend and registry-intervention builders
as the static production driver.  The resulting payload contains the complete
``InputPatch``, its materialized post-intervention state, and measured
intervention usage.  The official arena worker can therefore inject the
materialized state without reimplementing or guessing any algorithm logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import (
    ArmExecutionError,
    ArtifactError,
    BudgetExceededError,
    IntentViolationError,
    ValidationError,
)
from agentinputlab.contracts.lifecycle import Intervention, InterventionContext
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    LockedBuilderSpec,
    StaticArmBuildRequest,
    StaticBackendBuildRequest,
    StaticRuntimeComponents,
    static_runtime_product_sha256,
)
from agentinputlab.experiments.loader import resolve_driver_entrypoint
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner
from agentinputlab.selectors import NoOpIntervention

from .external_process_arena import (
    ExternalArmPayload,
    ExternalArmPayloadBuildRequest,
)

EXECUTABLE_RELEASE_PAYLOAD_PROTOCOL = "agentinputlab.executable-release-payload.v1"

_SCHEMA_VERSION = 1
_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "allowed_release_ids",
        "runtime_by_model_slot",
        "intervention_builder",
    }
)


class ExecutableReleaseArmPayloadBuilder:
    """Run one unchanged registered release and serialize its materialized patch."""

    builder_kind = "arm_payload"

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": EXECUTABLE_RELEASE_PAYLOAD_PROTOCOL,
                "schema_version": _SCHEMA_VERSION,
                "runtime_source": "checksum_locked_backend_builder",
                "intervention_source": "checksum_locked_intervention_builder",
                "execution": "agentinputlab.InterventionRunner",
                "payload": "patch_plus_materialized_visible_state_plus_usage",
                "protected_runtime_access": False,
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            parsed = _parse_parameters(parameters)
            for model_slot, runtime_spec in parsed["runtime_by_model_slot"].items():
                builder = _resolve_builder(runtime_spec, expected_kind="backend")
                _require_clean_preflight(
                    builder,
                    runtime_spec.parameters,
                    label=f"runtime_by_model_slot.{model_slot}",
                )
            intervention_spec = parsed["intervention_builder"]
            intervention_builder = _resolve_builder(
                intervention_spec,
                expected_kind="intervention",
            )
            _require_clean_preflight(
                intervention_builder,
                intervention_spec.parameters,
                label="intervention_builder",
            )
        except (ArtifactError, ValidationError, TypeError, ValueError) as exc:
            return (str(exc),)
        return ()

    def build(self, request: ExternalArmPayloadBuildRequest) -> ExternalArmPayload:
        if not isinstance(request, ExternalArmPayloadBuildRequest):
            raise TypeError(
                "executable release payload builder requires ExternalArmPayloadBuildRequest"
            )
        parsed = _parse_parameters(request.builder_parameters)
        if request.release_id not in parsed["allowed_release_ids"]:
            raise ValidationError("release is outside the executable payload allowlist")
        if request.arm_kind.value != "release":
            raise ValidationError(
                "executable release payload builder does not construct selector policy arms"
            )

        state = request.checkpoint.visible_state
        if request.release_id == "baseline.no_op":
            if request.family is not SemanticFamily.NO_OP:
                raise ValidationError("baseline.no_op requires the no_op family")
            runtime = None
            intervention: Intervention = NoOpIntervention()
            runtime_product_sha256 = content_hash(
                {
                    "schema_version": 1,
                    "runtime": "not_used_by_no_op_intervention",
                    "model_slot": request.model_slot,
                }
            )
            context = InterventionContext(
                seed=request.seed,
                run_config={
                    "benchmark_id": request.checkpoint.benchmark_id,
                    "checkpoint_id": request.checkpoint.checkpoint_id,
                    "model_slot": request.model_slot,
                },
                monitor=RunMonitor(),
            )
        else:
            if request.family is SemanticFamily.NO_OP:
                raise ValidationError("non-baseline release cannot use the no_op family")
            runtime_spec = _runtime_spec(parsed, request.model_slot)
            runtime_builder = _resolve_builder(runtime_spec, expected_kind="backend")
            runtime = runtime_builder.build(
                StaticBackendBuildRequest(
                    model_slot=request.model_slot,
                    seed=request.seed,
                    builder_parameters=runtime_spec.parameters,
                )
            )
            if not isinstance(runtime, StaticRuntimeComponents):
                raise ValidationError(
                    "external payload backend builder must return StaticRuntimeComponents"
                )
            runtime_product_sha256 = static_runtime_product_sha256(runtime)
            if runtime_product_sha256 != runtime_spec.expected_product_sha256:
                raise ArtifactError("external payload runtime product checksum mismatch")
            _bind_benchmark_retriever(runtime, request.checkpoint.benchmark_id)

            intervention_spec = parsed["intervention_builder"]
            intervention_builder = _resolve_builder(
                intervention_spec,
                expected_kind="intervention",
            )
            intervention = intervention_builder.build(
                StaticArmBuildRequest(
                    arm_id=request.arm_id,
                    release_id=request.release_id,
                    family=request.family,
                    profile_kind=request.profile_kind,
                    model_slot=request.model_slot,
                    seed=request.seed,
                    builder_parameters=intervention_spec.parameters,
                    arm_parameters=request.arm_parameters,
                    runtime=runtime,
                )
            )
            if not isinstance(intervention, Intervention):
                raise ValidationError(
                    "external payload intervention builder did not return an Intervention"
                )
            if (
                intervention.release_id != request.release_id
                or intervention.family is not request.family
            ):
                raise ValidationError(
                    "external payload intervention identity differs from the requested arm"
                )
            context = InterventionContext(
                backend=runtime.backend,
                dense_embedder=runtime.dense_embedder,
                retriever=runtime.retriever,
                vector_searcher=runtime.vector_searcher,
                tools=runtime.tools,
                external_runtime=runtime.external_runtime,
                artifact_manager=runtime.artifact_manager,
                seed=request.seed,
                run_config={
                    **dict(runtime.run_config),
                    "benchmark_id": request.checkpoint.benchmark_id,
                    "checkpoint_id": request.checkpoint.checkpoint_id,
                    "model_slot": request.model_slot,
                    "runtime_product_sha256": runtime_product_sha256,
                },
                intent_contract=runtime.intent_contract,
                monitor=RunMonitor(),
                pricing_catalog=runtime.pricing_catalog,
                strict_intent=runtime.strict_intent,
            )

        monitor = context.monitor
        if not isinstance(monitor, RunMonitor):  # pragma: no cover - construction invariant
            raise TypeError("external payload context monitor must be a RunMonitor")
        intervention_builder_sha256 = (
            content_hash(parsed["intervention_builder"].binding_identity)
            if request.release_id != "baseline.no_op"
            else content_hash({"builder": "builtin.no_op"})
        )
        try:
            record = InterventionRunner().run(intervention, state, context)
        except (ArmExecutionError, BudgetExceededError, IntentViolationError):
            snapshots = monitor.list_snapshots()
            if len(snapshots) != 1 or snapshots[0].failure is None:
                raise ValidationError(
                    "arm-local intervention failure has no unique monitored failure receipt"
                ) from None
            snapshot = snapshots[0]
            execution = {
                "status": "failed",
                "input_state_hash": state.state_hash,
                "output_state_hash": state.state_hash,
                "patch": None,
                "patch_hash": None,
                # The official arena may still execute the unchanged state to
                # preserve paired runtime/scorer evidence.  Downstream
                # normalization assigns the preregistered algorithm-failure
                # utility; it never mistakes this continuation for success.
                "materialized_state": to_jsonable(state),
                "runtime_product_sha256": runtime_product_sha256,
                "intervention_builder_sha256": intervention_builder_sha256,
                "usage": to_jsonable(snapshot.usage),
                "duration_ms": snapshot.duration_ms,
                "intent_check": None,
                "failure": {
                    "scope": "algorithm",
                    **to_jsonable(snapshot.failure),
                    "monitor_run_id": snapshot.run_id,
                },
            }
        else:
            materialized = record.patch.apply(state)
            if materialized.state_hash != record.output_state_hash:
                raise ValidationError("materialized external payload state hash is inconsistent")
            execution = {
                "status": record.status.value,
                "input_state_hash": record.input_state_hash,
                "output_state_hash": record.output_state_hash,
                "patch": record.patch.as_dict(),
                "patch_hash": record.patch.patch_hash,
                "materialized_state": to_jsonable(materialized),
                "runtime_product_sha256": runtime_product_sha256,
                "intervention_builder_sha256": intervention_builder_sha256,
                "usage": to_jsonable(record.usage),
                "duration_ms": record.duration_ms,
                "intent_check": to_jsonable(record.intent_check),
                "failure": None,
            }
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "protocol": EXECUTABLE_RELEASE_PAYLOAD_PROTOCOL,
            "arm_id": request.arm_id,
            "release_id": request.release_id,
            "family": request.family.value,
            "profile_kind": request.profile_kind.value,
            "model_slot": request.model_slot,
            "seed": request.seed,
            "checkpoint_identity_hash": request.checkpoint.identity_hash,
            "visible_state_hash": state.state_hash,
            "pre_action": request.pre_action,
            "arm_parameters": request.arm_parameters,
            "intervention_execution": execution,
        }
        return ExternalArmPayload(
            arm_id=request.arm_id,
            release_id=request.release_id,
            family=request.family,
            profile_kind=request.profile_kind,
            payload=payload,
        )


def executable_release_arm_payload_builder_factory() -> ExecutableReleaseArmPayloadBuilder:
    """Return the zero-argument production payload builder."""

    return ExecutableReleaseArmPayloadBuilder()


def _bind_benchmark_retriever(
    runtime: StaticRuntimeComponents,
    benchmark_id: str,
) -> None:
    """Bind the production retriever router to the audited checkpoint cell.

    Static predictors perform this binding themselves.  External-arena payloads
    execute the intervention before a downstream predictor exists, so methods
    such as Self-Ask and IRCoT need the same benchmark-validated route here.
    The router fingerprint is intentionally stable across binding.
    """

    from agentinputlab.experiments.production_builders import ProductionRetrieverRouter

    retriever = runtime.retriever
    if isinstance(retriever, ProductionRetrieverRouter):
        retriever.bind(benchmark_id)


def _parse_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(parameters, Mapping) or set(parameters) != _FIELDS:
        raise ValidationError("executable release payload parameters are invalid")
    if (
        parameters["schema_version"] != _SCHEMA_VERSION
        or parameters["protocol"] != EXECUTABLE_RELEASE_PAYLOAD_PROTOCOL
    ):
        raise ValidationError("unsupported executable release payload protocol")
    raw_ids = parameters["allowed_release_ids"]
    if (
        isinstance(raw_ids, (str, bytes))
        or not isinstance(raw_ids, Sequence)
        or not raw_ids
        or len(raw_ids) != len(set(raw_ids))
        or any(not isinstance(item, str) or not item.strip() for item in raw_ids)
    ):
        raise ValidationError(
            "executable payload allowed_release_ids must be unique non-empty strings"
        )
    raw_runtimes = parameters["runtime_by_model_slot"]
    if not isinstance(raw_runtimes, Mapping) or not raw_runtimes:
        raise ValidationError("runtime_by_model_slot must be a non-empty object")
    runtimes: dict[str, LockedBuilderSpec] = {}
    for model_slot, value in raw_runtimes.items():
        if not isinstance(model_slot, str) or not model_slot.strip():
            raise ValidationError("runtime_by_model_slot keys must be non-empty strings")
        spec = LockedBuilderSpec.from_mapping(
            value,
            path=f"runtime_by_model_slot.{model_slot}",
        )
        if spec.expected_product_sha256 is None:
            raise ValidationError(
                f"runtime_by_model_slot.{model_slot} requires expected_product_sha256"
            )
        runtimes[model_slot] = spec
    intervention = LockedBuilderSpec.from_mapping(
        parameters["intervention_builder"],
        path="intervention_builder",
    )
    if intervention.expected_product_sha256 is not None:
        raise ValidationError(
            "executable payload intervention_builder must not declare a product checksum"
        )
    return MappingProxyType(
        {
            "schema_version": _SCHEMA_VERSION,
            "protocol": EXECUTABLE_RELEASE_PAYLOAD_PROTOCOL,
            "allowed_release_ids": tuple(raw_ids),
            "runtime_by_model_slot": MappingProxyType(runtimes),
            "intervention_builder": intervention,
        }
    )


def _runtime_spec(
    parsed: Mapping[str, Any],
    model_slot: str,
) -> LockedBuilderSpec:
    try:
        return parsed["runtime_by_model_slot"][model_slot]
    except KeyError as exc:
        raise ArtifactError(
            f"executable payload has no runtime binding for model slot {model_slot!r}"
        ) from exc


def _resolve_builder(spec: LockedBuilderSpec, *, expected_kind: str) -> Any:
    factory = resolve_driver_entrypoint(spec.entrypoint)
    if not callable(factory):
        raise ArtifactError("executable payload builder entrypoint is not callable")
    try:
        builder = factory()
    except TypeError as exc:
        raise ArtifactError(
            "executable payload builder entrypoint must be a zero-argument factory"
        ) from exc
    if getattr(builder, "builder_kind", None) != expected_kind:
        raise ArtifactError(f"executable payload component kind must be {expected_kind!r}")
    fingerprint = builder.fingerprint()
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        raise ArtifactError("executable payload component fingerprint is invalid")
    if content_hash(fingerprint) != spec.fingerprint_sha256:
        raise ArtifactError("executable payload component fingerprint checksum mismatch")
    return builder


def _require_clean_preflight(
    builder: Any,
    parameters: Mapping[str, Any],
    *,
    label: str,
) -> None:
    blockers = builder.preflight(parameters)
    if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
        raise ValidationError(f"{label} preflight must return a sequence")
    messages = tuple(blockers)
    if any(not isinstance(message, str) or not message.strip() for message in messages):
        raise ValidationError(f"{label} preflight returned an invalid blocker")
    if messages:
        raise ArtifactError(f"{label}: " + "; ".join(messages))


__all__ = [
    "EXECUTABLE_RELEASE_PAYLOAD_PROTOCOL",
    "ExecutableReleaseArmPayloadBuilder",
    "executable_release_arm_payload_builder_factory",
]
