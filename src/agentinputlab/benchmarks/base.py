"""Benchmark adapter contracts that physically isolate evaluator gold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    Message,
)


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    example_id: str
    query: str
    instruction: str = ""
    visible_context: tuple[ContextBlock, ...] = ()
    visible_messages: tuple[Message, ...] = ()
    visible_metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluator_gold: Mapping[str, Any] = field(default_factory=dict, repr=False)
    visible_tool_names: tuple[str, ...] = ()
    optimizer_visible_id: str | None = None

    def visible_state(self, *, budget: ExecutionBudget | None = None) -> FrozenAgentState:
        """Create a state with no reference to evaluator-only fields."""

        return FrozenAgentState(
            task_id=self.optimizer_visible_id or self.example_id,
            query=self.query,
            instruction=self.instruction,
            context=self.visible_context,
            messages=self.visible_messages,
            tool_names=self.visible_tool_names,
            metadata=dict(self.visible_metadata),
            budget=budget or ExecutionBudget(),
        )


def opaque_checkpoint_state_id(
    benchmark_id: str,
    checkpoint_id: str,
    replay_identity: str,
) -> str:
    """Derive a stable predictor-visible ID without exposing protected labels."""

    return "ailcp_" + content_hash(
        {
            "benchmark_id": benchmark_id,
            "checkpoint_id": checkpoint_id,
            "replay_identity": replay_identity,
        }
    )


@runtime_checkable
class BenchmarkAdapter(Protocol):
    benchmark_id: str
    pairing_mode: str

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]: ...

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]: ...
