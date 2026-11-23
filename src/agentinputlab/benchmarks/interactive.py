"""Locked constructors for stateful, externally executed benchmark arenas.

The static production factory owns downloaded/prepared datasets.  This module
keeps a separate public seam for checkpoints captured from an official
interactive runtime.  The split is deliberate: an external evaluation bridge
can normalize official metrics, but it does not by itself prove that two
experimental arms started from identical hidden environment and simulator
state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError

from .appworld_ul import (
    AUDITED_APPWORLD_UPSTREAM_COMMIT,
    AppWorldULCheckpoint,
    AppWorldULCheckpointAdapter,
    AppWorldULEvaluationBridge,
)
from .registry import BenchmarkRegistry
from .statebench import (
    STATEBENCH_OFFICIAL_SOURCE_REVISION,
    StateBenchCheckpoint,
    StateBenchCheckpointAdapter,
    StateBenchEvaluationBridge,
)
from .tau3 import (
    TAU3_OFFICIAL_SOURCE_REVISION,
    PreparedTau3BankingBundle,
    Tau3BankingCheckpoint,
    Tau3BankingCheckpointAdapter,
    Tau3EvaluationBridge,
)
from .toolsandbox import (
    OFFICIAL_SOURCE_REVISION as TOOLSANDBOX_OFFICIAL_SOURCE_REVISION,
)
from .toolsandbox import (
    ToolSandboxCheckpoint,
    ToolSandboxCommonSupportAdapter,
    ToolSandboxEvaluationBridge,
    ToolSandboxScenarioInventory,
)

PAIRING_MODE_UNSUPPORTED_EXTERNAL_REPLAY = "unsupported_external_replay"
_FIVE_FAMILIES = frozenset(
    {
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)


@dataclass(frozen=True, slots=True)
class InteractiveBenchmarkSpec:
    """Registry lock and execution boundary for one external arena."""

    cell_id: str
    source_revision: str
    adapter_metric_names: tuple[str, ...]
    split_contract: str
    pairing_mode: str = PAIRING_MODE_UNSUPPORTED_EXTERNAL_REPLAY

    def __post_init__(self) -> None:
        for label in ("cell_id", "source_revision", "split_contract", "pairing_mode"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"interactive benchmark {label} must be trimmed and non-empty")
        metrics = tuple(self.adapter_metric_names)
        if not metrics or len(metrics) != len(set(metrics)):
            raise ValueError("interactive benchmark adapter metrics must be non-empty and unique")
        if self.pairing_mode != PAIRING_MODE_UNSUPPORTED_EXTERNAL_REPLAY:
            raise ValueError(
                "no interactive arena is yet certified for atomic external paired replay"
            )
        object.__setattr__(self, "adapter_metric_names", metrics)

    @property
    def structural_five_family_candidate(self) -> bool:
        """Whether the cell may admit all-five checkpoints after local gating."""

        return True

    @property
    def confirmatory_pairing_ready(self) -> bool:
        """Whether AgentInputLab can currently attest atomic two-arm restoration."""

        return False


_INTERACTIVE_SPECS = (
    InteractiveBenchmarkSpec(
        cell_id="tau3.banking_knowledge",
        source_revision=TAU3_OFFICIAL_SOURCE_REVISION,
        adapter_metric_names=Tau3BankingCheckpointAdapter.adapter_metric_names,
        split_contract=Tau3BankingCheckpointAdapter.split,
    ),
    InteractiveBenchmarkSpec(
        cell_id="appworld_ul.common_support",
        source_revision=AUDITED_APPWORLD_UPSTREAM_COMMIT,
        adapter_metric_names=AppWorldULCheckpointAdapter.adapter_metric_names,
        split_contract="checkpoint_declared",
    ),
    InteractiveBenchmarkSpec(
        cell_id="state_bench.common_support",
        source_revision=STATEBENCH_OFFICIAL_SOURCE_REVISION,
        adapter_metric_names=StateBenchCheckpointAdapter.adapter_metric_names,
        split_contract=StateBenchCheckpointAdapter.split,
    ),
    InteractiveBenchmarkSpec(
        cell_id="toolsandbox.common_support",
        source_revision=TOOLSANDBOX_OFFICIAL_SOURCE_REVISION,
        adapter_metric_names=ToolSandboxCommonSupportAdapter.adapter_metric_names,
        split_contract=ToolSandboxCommonSupportAdapter.split,
    ),
)
_INTERACTIVE_BY_CELL = {spec.cell_id: spec for spec in _INTERACTIVE_SPECS}


def interactive_benchmark_specs() -> tuple[InteractiveBenchmarkSpec, ...]:
    """Return the registry-checked external-arena inventory."""

    registry = BenchmarkRegistry.default()
    for spec in _INTERACTIVE_SPECS:
        cell = registry.get(spec.cell_id)
        if cell.source_revision != spec.source_revision:
            raise RuntimeError(
                f"interactive source lock differs from registry for {spec.cell_id!r}"
            )
        if cell.adapter_metrics != spec.adapter_metric_names:
            raise RuntimeError(
                f"interactive metric contract differs from registry for {spec.cell_id!r}"
            )
        if cell.family_support != "checkpoint_conditional":
            raise RuntimeError(
                f"interactive family support must be checkpoint_conditional for {spec.cell_id!r}"
            )
        if frozenset(cell.families) != _FIVE_FAMILIES:
            raise RuntimeError(
                f"interactive registry cell does not declare the five core families: "
                f"{spec.cell_id!r}"
            )
    return _INTERACTIVE_SPECS


def create_interactive_benchmark_adapter(
    cell_id: str,
    *,
    checkpoints: Iterable[
        Tau3BankingCheckpoint | AppWorldULCheckpoint | StateBenchCheckpoint | ToolSandboxCheckpoint
    ],
    evaluation_bridge: (
        Tau3EvaluationBridge
        | AppWorldULEvaluationBridge
        | StateBenchEvaluationBridge
        | ToolSandboxEvaluationBridge
    ),
    bundle: PreparedTau3BankingBundle | None = None,
    inventory: ToolSandboxScenarioInventory | None = None,
) -> (
    Tau3BankingCheckpointAdapter
    | AppWorldULCheckpointAdapter
    | StateBenchCheckpointAdapter
    | ToolSandboxCommonSupportAdapter
):
    """Construct one external arena without silently crossing runtime boundaries.

    ``bundle`` is mandatory only for tau3, whose prepared static task source is
    separately checksum-verified.  ``inventory`` is optional only for
    ToolSandbox.  Each concrete adapter validates checkpoint and bridge types.
    """

    if not isinstance(cell_id, str) or not cell_id.strip() or cell_id != cell_id.strip():
        raise ValidationError("interactive benchmark cell_id must be trimmed and non-empty")
    interactive_benchmark_specs()
    try:
        spec = _INTERACTIVE_BY_CELL[cell_id]
    except KeyError as exc:
        available = ", ".join(item.cell_id for item in _INTERACTIVE_SPECS)
        raise ValidationError(
            f"unknown interactive benchmark cell {cell_id!r}; available: {available}"
        ) from exc

    normalized_checkpoints = tuple(checkpoints)
    if spec.cell_id == "tau3.banking_knowledge":
        if bundle is None:
            raise ValidationError("tau3 interactive loading requires a prepared source bundle")
        if inventory is not None:
            raise ValidationError("ToolSandbox inventory is not valid for tau3")
        return Tau3BankingCheckpointAdapter(
            bundle,
            checkpoints=normalized_checkpoints,
            evaluation_bridge=evaluation_bridge,
        )

    if bundle is not None:
        raise ValidationError("prepared tau3 bundle is only valid for tau3.banking_knowledge")
    if spec.cell_id == "appworld_ul.common_support":
        if inventory is not None:
            raise ValidationError("ToolSandbox inventory is not valid for AppWorld-UL")
        return AppWorldULCheckpointAdapter(
            checkpoints=normalized_checkpoints,
            evaluation_bridge=evaluation_bridge,
        )
    if spec.cell_id == "state_bench.common_support":
        if inventory is not None:
            raise ValidationError("ToolSandbox inventory is not valid for STATE-Bench")
        return StateBenchCheckpointAdapter(
            checkpoints=normalized_checkpoints,
            evaluation_bridge=evaluation_bridge,
        )
    return ToolSandboxCommonSupportAdapter(
        checkpoints=normalized_checkpoints,
        evaluation_bridge=evaluation_bridge,
        inventory=inventory,
    )


def structural_full_block_checkpoint_ids(
    adapter: (
        Tau3BankingCheckpointAdapter
        | AppWorldULCheckpointAdapter
        | StateBenchCheckpointAdapter
        | ToolSandboxCommonSupportAdapter
    ),
    split: str,
) -> tuple[str, ...]:
    """Select checkpoints whose visible structure admits all five families.

    This is only the first inclusion gate.  It does not certify atomic paired
    replay and therefore must not be used alone to populate a confirmatory
    matched-arm table.
    """

    examples = adapter.examples(split)
    return tuple(
        example.example_id
        for example in examples
        if adapter.supports_full_block(example.example_id)
    )


__all__ = [
    "PAIRING_MODE_UNSUPPORTED_EXTERNAL_REPLAY",
    "TAU3_OFFICIAL_SOURCE_REVISION",
    "InteractiveBenchmarkSpec",
    "create_interactive_benchmark_adapter",
    "interactive_benchmark_specs",
    "structural_full_block_checkpoint_ids",
]
