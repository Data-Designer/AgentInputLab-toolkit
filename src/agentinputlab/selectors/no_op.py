"""Zero-cost baseline intervention."""

from __future__ import annotations

from dataclasses import dataclass

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.lifecycle import InterventionContext
from agentinputlab.contracts.patch import InputPatch
from agentinputlab.contracts.state import FrozenAgentState


@dataclass(frozen=True, slots=True)
class NoOpIntervention:
    method_id: str = "no_op"
    release_id: str = "baseline.no_op"
    family: SemanticFamily = SemanticFamily.NO_OP
    profile_kind: str = "controlled_adapter"

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        return InputPatch.no_op(
            state,
            method_id=self.method_id,
            release_id=self.release_id,
            rationale="explicit no-intervention baseline",
            profile_kind=self.profile_kind,
        )
