"""Controlled ACON observation profile shared by the paper's main arenas.

The upstream ACON observation optimizer is unchanged.  This wrapper freezes
only the mapping from an AgentInputLab capture to ACON's single
``observation`` argument: context block zero is optimized and every other
captured block is retained verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers import WorkerClient

from .common import ACONArtifactIdentity, ACONObservationControls
from .observation_bridge import ACONObservationRuntimeExternal

ACON_COMMON_SUPPORT_RELEASE = "curate.acon.common_support_observation"
ACON_COMMON_SUPPORT_CONTEXT_RULE = "captured_context_block_zero.v1"


@dataclass(slots=True)
class ACONObservationCommonSupport:
    """Run the pinned ACON observation optimizer on a frozen context slot."""

    client: WorkerClient | None = None
    artifacts: ACONArtifactIdentity | None = None
    controls: ACONObservationControls = field(
        default_factory=lambda: ACONObservationControls(
            summarization_threshold=-1,
            context_index=0,
            add_previous_history_summary=False,
            observation_version=1,
        )
    )
    method_id: str = "acon"
    release_id: str = ACON_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, ACONArtifactIdentity):
            raise ValidationError("ACON common support requires frozen artifact identities")
        if not isinstance(self.controls, ACONObservationControls):
            raise ValidationError("ACON common-support controls have the wrong type")
        if self.controls.context_index != 0:
            raise ValidationError("ACON common support freezes context_index=0")
        if self.controls.add_previous_history_summary:
            raise ValidationError(
                "ACON common support forbids hidden previous-history summaries at turn zero"
            )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.context:
            raise ValidationError("ACON common support requires captured visible context")
        component = ACONObservationRuntimeExternal(
            client=self.client,
            artifacts=self.artifacts,
            controls=self.controls,
        ).propose(state, context)
        if len(component.operations) != 1:
            raise AlgorithmOutputError("ACON component emitted an unexpected operation count")
        operation = component.operations[0]
        if operation.op is not PatchOp.REPLACE_CONTEXT:
            raise AlgorithmOutputError("ACON component did not replace captured context")
        output_context = tuple(operation.value)
        if len(output_context) != len(state.context):
            raise AlgorithmOutputError("ACON common support changed the context block count")
        for index, (before, after) in enumerate(zip(state.context, output_context, strict=True)):
            if before.block_id != after.block_id:
                raise AlgorithmOutputError("ACON common support changed a source block identity")
            if index != 0 and before != after:
                raise AlgorithmOutputError("ACON common support modified an unselected block")

        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "acon_common_support_context_selection",
                    {
                        "selection_rule": ACON_COMMON_SUPPORT_CONTEXT_RULE,
                        "selected_context_index": 0,
                        "selected_context_id": state.context[0].block_id,
                        "selected_context_hash": content_hash(state.context[0]),
                        "source_context_blocks": len(state.context),
                    },
                ),
                *component.trace.events,
            ),
            counters={
                **dict(component.trace.counters),
                "source_context_blocks": len(state.context),
                "selected_context_blocks": 1,
            },
            warnings=(
                *component.trace.warnings,
                "controlled_common_support_context_mapping_not_paper_result_parity",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=component.operations,
            trace=trace,
            implementation_identity=("frozen_context_mapping+" + component.implementation_identity),
            profile_kind="controlled_adapter",
            metadata={
                **dict(component.metadata),
                "component_release_id": component.release_id,
                "common_support_context_rule": ACON_COMMON_SUPPORT_CONTEXT_RULE,
                "source_state_hash": state.state_hash,
                "paper_result_parity_claimed": False,
            },
        )


__all__ = [
    "ACON_COMMON_SUPPORT_CONTEXT_RULE",
    "ACON_COMMON_SUPPORT_RELEASE",
    "ACONObservationCommonSupport",
]
