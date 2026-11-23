"""Dedicated planner-only native subprocess; cannot satisfy the full bridge."""

from __future__ import annotations

from agentinputlab.algorithms.decompose.decomp.planner_component import (
    DECOMP_PLANNER_MODE,
    DecompPlannerRequest,
)

from .decomp_live import DecompLiveRuntime


class DecompPlannerLiveRuntime(DecompLiveRuntime):
    request_type = DecompPlannerRequest
    worker_id = "agentinputlab.decomp.native-planner.v1"
    worker_flags = ("--planner-component",)

    def fingerprint(self):
        return {
            **super().fingerprint(),
            "backend_binding_mode": DECOMP_PLANNER_MODE,
            "component_only": True,
            "max_native_transitions": 1,
        }

    def _parse_payload(self, value):
        return value


__all__ = ["DecompPlannerLiveRuntime"]
