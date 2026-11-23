"""Mechanism controls that preserve visible input while retaining intervention cost."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentinputlab.contracts.enums import PatchOp, SemanticFamily, TracePhase
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.lifecycle import Intervention, InterventionContext
from agentinputlab.contracts.patch import InputPatch, PatchOperation
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.contracts.trace import ExecutionTrace, TraceEvent

if TYPE_CHECKING:
    from agentinputlab.experiments.drivers.static_production import StaticArmBuildRequest

COST_MATCHED_NO_OP_RELEASE_ID = "baseline.no_op_cost_shadow"
COST_MATCHED_NO_OP_BUILDER_PROTOCOL = "agentinputlab.cost-matched-no-op-builder.v1"
COST_MATCHED_EXTERNAL_PAYLOAD_PROTOCOL = "agentinputlab.cost-matched-external-payload-builder.v1"


@dataclass(slots=True)
class CostMatchedNoOpIntervention:
    """Run a frozen intervention, discard its patch, and preserve its measured cost."""

    delegate: Intervention
    method_id: str = "no_op_cost_shadow"
    release_id: str = COST_MATCHED_NO_OP_RELEASE_ID
    family: SemanticFamily = SemanticFamily.NO_OP

    def __post_init__(self) -> None:
        if not isinstance(self.delegate, Intervention):
            raise TypeError("cost-matched NO_OP requires an Intervention delegate")
        if self.delegate.family is SemanticFamily.NO_OP:
            raise ValidationError("cost-matched NO_OP delegate must perform a real intervention")
        if self.release_id != COST_MATCHED_NO_OP_RELEASE_ID:
            raise ValidationError("cost-matched NO_OP release identity is fixed")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        candidate = self.delegate.propose(state, context)
        if not isinstance(candidate, InputPatch):
            raise ValidationError("cost-matched NO_OP delegate returned an invalid patch")
        if candidate.source_state_hash != state.state_hash:
            raise ValidationError("cost-matched NO_OP delegate patch belongs to another state")
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    phase=TracePhase.FINALIZE,
                    name="discard_intervention_patch",
                    payload={
                        "delegate_method_id": self.delegate.method_id,
                        "delegate_release_id": self.delegate.release_id,
                        "discarded_patch_hash": candidate.patch_hash,
                        "visible_input_unchanged": True,
                    },
                ),
            ),
            counters=dict(candidate.trace.counters),
            warnings=("delegate patch intentionally discarded for cost-only mechanism control",),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.NO_OP,
                    rationale="cost-matched shadow preserves the original visible input",
                ),
            ),
            trace=trace,
            implementation_identity="cost_shadow:"
            + content_hash(
                {
                    "delegate_method_id": self.delegate.method_id,
                    "delegate_release_id": self.delegate.release_id,
                }
            ),
            profile_kind="controlled_adapter",
            metadata={
                "control_kind": "optimizer_orchestration_only",
                "delegate_method_id": self.delegate.method_id,
                "delegate_release_id": self.delegate.release_id,
                "discarded_patch_hash": candidate.patch_hash,
                "visible_input_unchanged": True,
            },
        )


class CostMatchedNoOpBuilder:
    """Static-driver builder for an exact registry-backed shadow delegate."""

    builder_kind = "intervention"

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": COST_MATCHED_NO_OP_BUILDER_PROTOCOL,
                "schema_version": 1,
                "delegate_builder": "agentinputlab.registry-intervention-builder.v1",
                "output_release_id": COST_MATCHED_NO_OP_RELEASE_ID,
                "visible_input_policy": "discard_delegate_patch",
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        from agentinputlab.experiments.builders.registry_intervention import (
            RegistryInterventionBuilder,
        )

        try:
            parsed = _parameters(parameters)
        except ValidationError as exc:
            return (str(exc),)
        return RegistryInterventionBuilder().preflight(_delegate_parameters(parsed))

    def build(self, request: StaticArmBuildRequest) -> CostMatchedNoOpIntervention:
        from agentinputlab.experiments.builders.registry_intervention import (
            RegistryInterventionBuilder,
        )
        from agentinputlab.experiments.drivers.static_production import (
            StaticArmBuildRequest,
        )

        if not isinstance(request, StaticArmBuildRequest):
            raise TypeError("cost-matched NO_OP builder requires StaticArmBuildRequest")
        if (
            request.release_id != COST_MATCHED_NO_OP_RELEASE_ID
            or request.family is not SemanticFamily.NO_OP
        ):
            raise ValidationError("cost-matched NO_OP builder received another arm identity")
        parsed = _parameters(request.builder_parameters)
        delegate_request = StaticArmBuildRequest(
            arm_id=request.arm_id + ".delegate",
            release_id=parsed["delegate_release_id"],
            family=SemanticFamily(parsed["delegate_family"]),
            profile_kind=request.profile_kind,
            model_slot=request.model_slot,
            seed=request.seed,
            builder_parameters=_delegate_parameters(parsed),
            arm_parameters=request.arm_parameters,
            runtime=request.runtime,
        )
        delegate = RegistryInterventionBuilder().build(delegate_request)
        return CostMatchedNoOpIntervention(delegate=delegate)


def cost_matched_no_op_builder_factory() -> CostMatchedNoOpBuilder:
    return CostMatchedNoOpBuilder()


class CostMatchedExternalPayloadBuilder:
    """Build an atomic-arena instruction that discards a delegate's patch."""

    builder_kind = "arm_payload"

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": COST_MATCHED_EXTERNAL_PAYLOAD_PROTOCOL,
                "schema_version": 1,
                "output_release_id": COST_MATCHED_NO_OP_RELEASE_ID,
                "required_runtime_action": "execute_optimizer_discard_patch",
                "visible_input_policy": "preserve_original_state",
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            parsed = _external_parameters(parameters)
        except ValidationError as exc:
            return (str(exc),)
        if not parsed["runtime_support_attested"]:
            return ("external runtime has not attested execute_optimizer_discard_patch support",)
        return ()

    def build(self, request: Any) -> Any:
        from agentinputlab.experiments.drivers.external_arena import (
            ExternalArmPayload,
            ExternalArmPayloadBuildRequest,
        )

        if not isinstance(request, ExternalArmPayloadBuildRequest):
            raise TypeError(
                "cost-matched external payload builder requires ExternalArmPayloadBuildRequest"
            )
        if request.release_id != COST_MATCHED_NO_OP_RELEASE_ID or request.family not in {
            SemanticFamily.REFORMULATE,
            SemanticFamily.DECOMPOSE,
            SemanticFamily.ACQUIRE,
            SemanticFamily.CLARIFY,
            SemanticFamily.CURATE,
        }:
            raise ValidationError(
                "cost-matched external payload builder received another arm identity"
            )
        parsed = _external_parameters(request.builder_parameters)
        if not parsed["runtime_support_attested"]:
            raise ValidationError(
                "cost-matched external payload requires runtime support attestation"
            )
        shadow_family = request.arm_parameters.get("shadow_family")
        shadow_release = request.arm_parameters.get("shadow_release_id")
        if (
            shadow_family != parsed["delegate_family"]
            or shadow_release != parsed["delegate_release_id"]
            or request.family.value != parsed["delegate_family"]
        ):
            raise ValidationError(
                "cost-matched external arm parameters differ from its locked delegate"
            )
        payload = {
            "schema_version": 1,
            "protocol": COST_MATCHED_EXTERNAL_PAYLOAD_PROTOCOL,
            "action": "execute_optimizer_discard_patch",
            "delegate": {
                "family": parsed["delegate_family"],
                "release_id": parsed["delegate_release_id"],
                "configuration_sha256": parsed["delegate_configuration_sha256"],
            },
            "checkpoint_identity_sha256": request.checkpoint.identity_hash,
            "visible_state_sha256": request.checkpoint.visible_state.state_hash,
            "model_slot": request.model_slot,
            "seed": request.seed,
            "visible_input_unchanged": True,
            "discard_delegate_patch": True,
        }
        return ExternalArmPayload(
            arm_id=request.arm_id,
            release_id=request.release_id,
            family=request.family,
            profile_kind=request.profile_kind,
            payload=payload,
        )


def cost_matched_external_payload_builder_factory() -> CostMatchedExternalPayloadBuilder:
    return CostMatchedExternalPayloadBuilder()


def _parameters(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("cost-matched NO_OP parameters must be an object")
    required = {
        "schema_version",
        "delegate_release_id",
        "delegate_family",
        "constructor_kwargs",
        "arm_parameter_bindings",
        "required_environment",
    }
    if set(value) != required or value.get("schema_version") != 1:
        raise ValidationError("cost-matched NO_OP parameter schema is invalid")
    release_id = value.get("delegate_release_id")
    family = value.get("delegate_family")
    if not isinstance(release_id, str) or not release_id.strip():
        raise ValidationError("cost-matched NO_OP delegate_release_id must be non-empty")
    try:
        SemanticFamily(family)
    except (TypeError, ValueError) as exc:
        raise ValidationError("cost-matched NO_OP delegate_family is invalid") from exc
    if SemanticFamily(family) in {SemanticFamily.NO_OP, SemanticFamily.COMPOSITE}:
        raise ValidationError("cost-matched NO_OP delegate must be one prompt family")
    if not isinstance(value["constructor_kwargs"], Mapping):
        raise ValidationError("cost-matched NO_OP constructor_kwargs must be an object")
    if not isinstance(value["arm_parameter_bindings"], Mapping):
        raise ValidationError("cost-matched NO_OP arm_parameter_bindings must be an object")
    required_environment = value["required_environment"]
    if isinstance(required_environment, (str, bytes)) or not isinstance(
        required_environment, Sequence
    ):
        raise ValidationError("cost-matched NO_OP required_environment must be an array")
    try:
        return freeze_mapping(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("cost-matched NO_OP parameters must be canonical JSON") from exc


def _delegate_parameters(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return freeze_mapping(
        {
            "schema_version": 1,
            "release_id": value["delegate_release_id"],
            "constructor_kwargs": value["constructor_kwargs"],
            "runtime_bindings": {},
            "arm_parameter_bindings": value["arm_parameter_bindings"],
            "required_environment": value["required_environment"],
        }
    )


def _external_parameters(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("cost-matched external parameters must be an object")
    required = {
        "schema_version",
        "delegate_release_id",
        "delegate_family",
        "delegate_configuration_sha256",
        "runtime_support_attested",
    }
    if set(value) != required or value.get("schema_version") != 1:
        raise ValidationError("cost-matched external parameter schema is invalid")
    release_id = value.get("delegate_release_id")
    family = value.get("delegate_family")
    digest = value.get("delegate_configuration_sha256")
    if not isinstance(release_id, str) or not release_id.strip():
        raise ValidationError("cost-matched external delegate_release_id must be non-empty")
    try:
        typed_family = SemanticFamily(family)
    except (TypeError, ValueError) as exc:
        raise ValidationError("cost-matched external delegate_family is invalid") from exc
    if typed_family in {SemanticFamily.NO_OP, SemanticFamily.COMPOSITE}:
        raise ValidationError("cost-matched external delegate must be one prompt family")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValidationError("cost-matched external delegate_configuration_sha256 must be SHA-256")
    if not isinstance(value["runtime_support_attested"], bool):
        raise ValidationError("cost-matched external runtime_support_attested must be boolean")
    return freeze_mapping(value)


__all__ = [
    "COST_MATCHED_EXTERNAL_PAYLOAD_PROTOCOL",
    "COST_MATCHED_NO_OP_BUILDER_PROTOCOL",
    "COST_MATCHED_NO_OP_RELEASE_ID",
    "CostMatchedExternalPayloadBuilder",
    "CostMatchedNoOpBuilder",
    "CostMatchedNoOpIntervention",
    "cost_matched_external_payload_builder_factory",
    "cost_matched_no_op_builder_factory",
]
