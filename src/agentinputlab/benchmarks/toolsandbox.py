"""Fail-closed bridge for ToolSandbox common-support experiments.

ToolSandbox owns the scenario runtime and evaluator.  AgentInputLab only
normalizes an externally captured, optimizer-visible checkpoint and a typed
result returned after the official milestone/minefield evaluator has run.  It
does not reimplement ToolSandbox's state transition or matching logic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.contracts.state import ContextBlock, Message

from .base import BenchmarkExample, opaque_checkpoint_state_id

BENCHMARK_ID = "toolsandbox.common_support"
OFFICIAL_REPOSITORY = "https://github.com/apple/ToolSandbox"
OFFICIAL_SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
OFFICIAL_BASE_SCENARIO_COUNT = 129
OFFICIAL_VARIANT_SCENARIO_COUNT = 1032
OFFICIAL_SCENARIO_MEMBERSHIP_ASSET = "toolsandbox_165848b_scenario_names.json"
OFFICIAL_SCENARIO_MEMBERSHIP_SHA256 = (
    "4de99f83ae5972b4556308e793e57db3e34e0dbd8b3da60d4a30d4c484358673"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALLER_ASSERTED = "caller_asserted"
_EXTERNAL_OFFICIAL_SEMANTICS = "external_official_semantics"
_STRUCTURAL_CANDIDATE = "structural_candidate"
_EXPECTED_VARIANT_SUFFIXES = (
    "",
    "_3_distraction_tools",
    "_10_distraction_tools",
    "_all_tools",
    "_3_distraction_tools_tool_description_scrambled",
    "_3_distraction_tools_arg_type_scrambled",
    "_3_distraction_tools_arg_description_scrambled",
    "_3_distraction_tools_tool_name_scrambled",
)
_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)


def _load_official_scenario_membership(
    *,
    asset_name: str = OFFICIAL_SCENARIO_MEMBERSHIP_ASSET,
) -> tuple[tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Load and verify the scenario names extracted from the pinned source tree."""

    try:
        payload_bytes = (
            resources.files("agentinputlab.benchmarks").joinpath("data", asset_name).read_bytes()
        )
    except OSError as exc:
        raise RuntimeError("ToolSandbox scenario membership asset is unavailable") from exc
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != OFFICIAL_SCENARIO_MEMBERSHIP_SHA256:
        raise RuntimeError("ToolSandbox scenario membership asset failed its SHA-256 lock")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ToolSandbox scenario membership asset is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "base_names",
        "source_revision",
        "variant_suffixes",
    }:
        raise RuntimeError("ToolSandbox scenario membership asset has an invalid schema")
    if payload["source_revision"] != OFFICIAL_SOURCE_REVISION:
        raise RuntimeError("ToolSandbox scenario membership asset has the wrong source revision")

    raw_base_names = payload["base_names"]
    raw_suffixes = payload["variant_suffixes"]
    if not isinstance(raw_base_names, list) or not all(
        isinstance(name, str) and name and name == name.strip() for name in raw_base_names
    ):
        raise RuntimeError("ToolSandbox base scenario names must be trimmed strings")
    if not isinstance(raw_suffixes, list) or not all(
        isinstance(suffix, str) and suffix == suffix.strip() for suffix in raw_suffixes
    ):
        raise RuntimeError("ToolSandbox variant suffixes must be strings")

    base_names = tuple(raw_base_names)
    variant_suffixes = tuple(raw_suffixes)
    if len(base_names) != OFFICIAL_BASE_SCENARIO_COUNT or len(set(base_names)) != len(base_names):
        raise RuntimeError("ToolSandbox base scenario membership is not the audited 129-name set")
    if variant_suffixes != _EXPECTED_VARIANT_SUFFIXES:
        raise RuntimeError("ToolSandbox variant suffixes differ from the pinned generator")

    variant_names = tuple(
        f"{base_name}{suffix}" for base_name in base_names for suffix in variant_suffixes
    )
    variant_name_set = frozenset(variant_names)
    if len(variant_names) != OFFICIAL_VARIANT_SCENARIO_COUNT or len(variant_name_set) != len(
        variant_names
    ):
        raise RuntimeError(
            "ToolSandbox generated scenario membership is not the audited 1032-name set"
        )
    return base_names, variant_suffixes, variant_name_set


(
    _OFFICIAL_BASE_SCENARIO_NAMES,
    _OFFICIAL_VARIANT_SUFFIXES,
    _OFFICIAL_SCENARIO_NAMES,
) = _load_official_scenario_membership()


@dataclass(frozen=True, slots=True)
class ToolSandboxScenarioInventory:
    """Audited identity of the official repository's generated scenario set."""

    source_revision: str = OFFICIAL_SOURCE_REVISION
    base_scenario_count: int = OFFICIAL_BASE_SCENARIO_COUNT
    variant_scenario_count: int = OFFICIAL_VARIANT_SCENARIO_COUNT
    membership_revision: str = field(
        init=False,
        default=OFFICIAL_SCENARIO_MEMBERSHIP_SHA256,
    )

    def __post_init__(self) -> None:
        if self.source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValidationError("ToolSandbox inventory must use the audited source revision")
        if self.base_scenario_count != OFFICIAL_BASE_SCENARIO_COUNT:
            raise ValidationError("ToolSandbox base-scenario count differs from the source lock")
        if self.variant_scenario_count != OFFICIAL_VARIANT_SCENARIO_COUNT:
            raise ValidationError("ToolSandbox variant count differs from the source lock")

    @property
    def base_scenario_names(self) -> tuple[str, ...]:
        return _OFFICIAL_BASE_SCENARIO_NAMES

    @property
    def variant_suffixes(self) -> tuple[str, ...]:
        return _OFFICIAL_VARIANT_SUFFIXES

    @property
    def scenario_names(self) -> frozenset[str]:
        return _OFFICIAL_SCENARIO_NAMES

    def contains(self, scenario_name: str) -> bool:
        return scenario_name in _OFFICIAL_SCENARIO_NAMES


@dataclass(frozen=True, slots=True)
class ToolSandboxFamilyEligibility:
    """Checkpoint-conditional support, not an official benchmark certification."""

    checkpoint_id: str
    eligibility: Mapping[str, bool]
    reasons: Mapping[str, str]
    provenance_status: str = _CALLER_ASSERTED
    support_status: str = _STRUCTURAL_CANDIDATE

    def __post_init__(self) -> None:
        expected = {family.value for family in _FAMILIES}
        if set(self.eligibility) != expected or set(self.reasons) != expected:
            raise ValidationError("ToolSandbox eligibility must cover the five core families")
        if any(not isinstance(value, bool) for value in self.eligibility.values()):
            raise ValidationError("ToolSandbox eligibility values must be boolean")
        if any(not isinstance(reason, str) or not reason for reason in self.reasons.values()):
            raise ValidationError("ToolSandbox eligibility reasons must be non-empty strings")
        if self.provenance_status != _CALLER_ASSERTED:
            raise ValidationError("ToolSandbox eligibility provenance must remain caller_asserted")
        if self.support_status != _STRUCTURAL_CANDIDATE:
            raise ValidationError(
                "ToolSandbox eligibility support must remain structural_candidate"
            )
        object.__setattr__(self, "eligibility", MappingProxyType(dict(self.eligibility)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))

    @property
    def eligible_families(self) -> tuple[SemanticFamily, ...]:
        return tuple(family for family in _FAMILIES if self.eligibility[family.value])

    @property
    def supports_full_block(self) -> bool:
        return all(self.eligibility.values())


@dataclass(frozen=True, slots=True)
class ToolSandboxCheckpoint:
    """Optimizer-visible state bound to one external ToolSandbox snapshot.

    Snapshot and replay hashes are opaque.  They must not expose evaluator
    milestones, minefields, hidden simulator instructions, or database rows.
    """

    checkpoint_id: str
    scenario_name: str
    query: str
    instruction: str
    visible_messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    runtime_snapshot_id: str = field(repr=False)
    replay_identity: str = field(repr=False)
    source_revision: str = OFFICIAL_SOURCE_REVISION
    visible_context: tuple[ContextBlock, ...] = ()
    visible_evidence_tool_names: tuple[str, ...] = ()
    snapshot_restore_available: bool = False
    user_simulator_continuation_available: bool = False

    def __post_init__(self) -> None:
        for label in ("checkpoint_id", "scenario_name", "query", "instruction"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(
                    f"ToolSandbox checkpoint {label} must be a trimmed non-empty string"
                )
        if self.source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValidationError("ToolSandbox checkpoint source revision is not audited")
        if self.scenario_name not in _OFFICIAL_SCENARIO_NAMES:
            raise ValidationError(
                "ToolSandbox checkpoint scenario_name is not in the pinned official inventory"
            )
        for label in ("runtime_snapshot_id", "replay_identity"):
            value = getattr(self, label)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValidationError(f"ToolSandbox checkpoint {label} must be a lowercase SHA-256")
        messages = tuple(self.visible_messages)
        context = tuple(self.visible_context)
        tools = tuple(self.tool_names)
        evidence_tools = tuple(self.visible_evidence_tool_names)
        if not messages or not all(isinstance(message, Message) for message in messages):
            raise ValidationError("ToolSandbox checkpoint requires visible Message instances")
        if not all(isinstance(block, ContextBlock) for block in context):
            raise ValidationError("ToolSandbox context must contain ContextBlock instances")
        if not tools or not all(
            isinstance(name, str) and name.strip() and name == name.strip() for name in tools
        ):
            raise ValidationError("ToolSandbox checkpoint requires visible tool names")
        if len(tools) != len(set(tools)):
            raise ValidationError("ToolSandbox checkpoint tool names must be unique")
        if len(evidence_tools) != len(set(evidence_tools)):
            raise ValidationError("ToolSandbox evidence tool names must be unique")
        if any(name not in tools for name in evidence_tools):
            raise ValidationError(
                "ToolSandbox evidence tool names must be an audited subset of visible tools"
            )
        user_messages = [message for message in messages if message.role == "user"]
        if not user_messages or user_messages[-1].content != self.query:
            raise ValidationError("ToolSandbox query must equal the latest visible user message")
        for label in (
            "snapshot_restore_available",
            "user_simulator_continuation_available",
        ):
            if not isinstance(getattr(self, label), bool):
                raise ValidationError(f"ToolSandbox {label} must be boolean")
        object.__setattr__(self, "visible_messages", messages)
        object.__setattr__(self, "visible_context", context)
        object.__setattr__(self, "tool_names", tools)
        object.__setattr__(self, "visible_evidence_tool_names", evidence_tools)

    @property
    def family_eligibility(self) -> ToolSandboxFamilyEligibility:
        restore = self.snapshot_restore_available
        evidence_channel = bool(self.visible_evidence_tool_names)
        clarification = restore and self.user_simulator_continuation_available
        curation_material = len(self.visible_messages) >= 3 or len(self.visible_context) >= 2
        eligibility = {
            SemanticFamily.REFORMULATE.value: restore,
            SemanticFamily.DECOMPOSE.value: restore,
            SemanticFamily.ACQUIRE.value: restore and evidence_channel,
            SemanticFamily.CLARIFY.value: clarification,
            SemanticFamily.CURATE.value: restore and curation_material,
        }
        reasons = {
            SemanticFamily.REFORMULATE.value: (
                "explicit visible query and snapshot restoration are available"
                if restore
                else "requires an official-state restore path for matched arms"
            ),
            SemanticFamily.DECOMPOSE.value: (
                "explicit visible query and snapshot restoration are available"
                if restore
                else "requires an official-state restore path for matched arms"
            ),
            SemanticFamily.ACQUIRE.value: (
                "restorable checkpoint exposes a visible search/evidence tool"
                if restore and evidence_channel
                else "requires restoration plus a visible search/evidence tool"
            ),
            SemanticFamily.CLARIFY.value: (
                "restorable checkpoint exposes on-policy user-simulator continuation"
                if clarification
                else "requires restoration plus replayable user-simulator continuation"
            ),
            SemanticFamily.CURATE.value: (
                "restorable checkpoint has sufficient visible history/context"
                if restore and curation_material
                else "requires restoration plus three messages or two context blocks"
            ),
        }
        return ToolSandboxFamilyEligibility(self.checkpoint_id, eligibility, reasons)


@dataclass(frozen=True, slots=True)
class ToolSandboxEvaluationResult:
    """Official-shaped values produced by an external ToolSandbox evaluator."""

    milestone_similarity: float
    minefield_similarity: float
    similarity: float
    turn_count: int
    cost: float
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_OFFICIAL_SEMANTICS)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        for label in ("milestone_similarity", "minefield_similarity", "similarity"):
            value = _unit_interval(getattr(self, label), label=label)
            object.__setattr__(self, label, value)
        if isinstance(self.turn_count, bool) or not isinstance(self.turn_count, int):
            raise ValidationError("ToolSandbox turn_count must be an integer")
        if self.turn_count < 0:
            raise ValidationError("ToolSandbox turn_count must be non-negative")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
            raise ValidationError("ToolSandbox cost must be numeric")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValidationError("ToolSandbox cost must be finite and non-negative")
        object.__setattr__(self, "cost", cost)
        expected_similarity = self.milestone_similarity if self.minefield_similarity == 0.0 else 0.0
        if not math.isclose(self.similarity, expected_similarity, abs_tol=1e-12):
            raise ValidationError(
                "ToolSandbox similarity must follow official minefield-nullification semantics"
            )

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "cost": self.cost,
                "milestone_similarity": self.milestone_similarity,
                "minefield_similarity": self.minefield_similarity,
                "official_similarity": self.similarity,
                "turn_count": float(self.turn_count),
            }
        )


@runtime_checkable
class ToolSandboxEvaluationBridge(Protocol):
    """Typed bridge to the pinned official runtime and evaluator."""

    bridge_id: str
    bridge_revision: str
    cost_basis: str
    source_revision: str
    supports_snapshot_restore: bool
    supports_user_simulator_replay: bool
    provenance_status: str
    evaluation_scope: str
    official_runtime_verified: bool

    def evaluate(
        self,
        checkpoint: ToolSandboxCheckpoint,
        prediction: Any,
    ) -> ToolSandboxEvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CallerAssertedToolSandboxEvaluationBridge:
    """Explicit wrapper for an unverified external official-runtime callback."""

    bridge_id: str
    bridge_revision: str
    cost_basis: str
    supports_snapshot_restore: bool
    supports_user_simulator_replay: bool
    callback: Callable[
        [ToolSandboxCheckpoint, Any],
        ToolSandboxEvaluationResult,
    ] = field(repr=False)
    source_revision: str = field(init=False, default=OFFICIAL_SOURCE_REVISION)
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_OFFICIAL_SEMANTICS)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        for label in ("bridge_id", "cost_basis"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"ToolSandbox bridge {label} must be non-empty")
        if (
            not isinstance(self.bridge_revision, str)
            or _SHA256.fullmatch(self.bridge_revision) is None
        ):
            raise ValidationError("ToolSandbox bridge revision must be a lowercase SHA-256")
        if not callable(self.callback):
            raise TypeError("ToolSandbox bridge callback must be callable")
        for label in (
            "supports_snapshot_restore",
            "supports_user_simulator_replay",
        ):
            if not isinstance(getattr(self, label), bool):
                raise ValidationError(f"ToolSandbox bridge {label} must be boolean")

    def evaluate(
        self,
        checkpoint: ToolSandboxCheckpoint,
        prediction: Any,
    ) -> ToolSandboxEvaluationResult:
        result = self.callback(checkpoint, prediction)
        if not isinstance(result, ToolSandboxEvaluationResult):
            raise ValidationError("ToolSandbox callback must return ToolSandboxEvaluationResult")
        return result


class ToolSandboxCommonSupportAdapter:
    """Adapter for externally captured ToolSandbox checkpoint experiments."""

    benchmark_id = BENCHMARK_ID
    adapter_metric_names = (
        "milestone_similarity",
        "minefield_similarity",
        "official_similarity",
        "turn_count",
        "cost",
    )
    split = "external_checkpoints"
    pairing_mode = "unsupported_external_replay"

    def __init__(
        self,
        *,
        checkpoints: Iterable[ToolSandboxCheckpoint],
        evaluation_bridge: ToolSandboxEvaluationBridge,
        inventory: ToolSandboxScenarioInventory | None = None,
    ) -> None:
        if not isinstance(evaluation_bridge, ToolSandboxEvaluationBridge):
            raise TypeError(
                "evaluation_bridge must implement ToolSandboxEvaluationBridge; "
                "wrap callbacks with CallerAssertedToolSandboxEvaluationBridge"
            )
        _validate_bridge(evaluation_bridge)
        self.inventory = inventory or ToolSandboxScenarioInventory()
        if not isinstance(self.inventory, ToolSandboxScenarioInventory):
            raise TypeError("inventory must be ToolSandboxScenarioInventory")
        checkpoints_tuple = tuple(checkpoints)
        if not checkpoints_tuple:
            raise ValidationError("ToolSandbox adapter requires at least one checkpoint")
        if not all(isinstance(item, ToolSandboxCheckpoint) for item in checkpoints_tuple):
            raise TypeError("ToolSandbox checkpoints must be ToolSandboxCheckpoint instances")
        checkpoint_ids = [item.checkpoint_id for item in checkpoints_tuple]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValidationError("ToolSandbox checkpoint IDs must be unique")
        experimental_units = [
            (item.scenario_name, item.runtime_snapshot_id, item.replay_identity)
            for item in checkpoints_tuple
        ]
        if len(experimental_units) != len(set(experimental_units)):
            raise ValidationError("ToolSandbox experimental units must be unique")

        examples: list[BenchmarkExample] = []
        checkpoints_by_id: dict[str, ToolSandboxCheckpoint] = {}
        for checkpoint in sorted(checkpoints_tuple, key=lambda item: item.checkpoint_id):
            if (
                checkpoint.snapshot_restore_available
                and not evaluation_bridge.supports_snapshot_restore
            ):
                raise ValidationError(
                    "checkpoint claims snapshot restoration not supported by its bridge"
                )
            if (
                checkpoint.user_simulator_continuation_available
                and not evaluation_bridge.supports_user_simulator_replay
            ):
                raise ValidationError(
                    "checkpoint claims user-simulator continuation not supported by its bridge"
                )
            support = checkpoint.family_eligibility
            metadata = freeze_mapping(
                {
                    "benchmark_cell": self.benchmark_id,
                    "family_eligibility": dict(support.eligibility),
                    "family_eligibility_provenance": support.provenance_status,
                    "family_eligibility_reasons": dict(support.reasons),
                    "family_support_status": support.support_status,
                    "source_revision": OFFICIAL_SOURCE_REVISION,
                }
            )
            example = BenchmarkExample(
                example_id=checkpoint.checkpoint_id,
                query=checkpoint.query,
                instruction=checkpoint.instruction,
                visible_context=checkpoint.visible_context,
                visible_messages=checkpoint.visible_messages,
                visible_tool_names=checkpoint.tool_names,
                visible_metadata=metadata,
                evaluator_gold=MappingProxyType({}),
                optimizer_visible_id=opaque_checkpoint_state_id(
                    self.benchmark_id,
                    checkpoint.checkpoint_id,
                    checkpoint.replay_identity,
                ),
            )
            examples.append(example)
            checkpoints_by_id[checkpoint.checkpoint_id] = checkpoint

        self._bridge = evaluation_bridge
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._checkpoints = MappingProxyType(checkpoints_by_id)

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(
                f"ToolSandbox adapter exposes split {self.split!r}, not {split!r}"
            )
        return self._examples

    def family_eligibility(self, checkpoint_id: str) -> ToolSandboxFamilyEligibility:
        return self._checkpoint(checkpoint_id).family_eligibility

    def eligible_families(self, checkpoint_id: str) -> tuple[SemanticFamily, ...]:
        return self.family_eligibility(checkpoint_id).eligible_families

    def supports_full_block(self, checkpoint_id: str) -> bool:
        return self.family_eligibility(checkpoint_id).supports_full_block

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this adapter")
        result = self._bridge.evaluate(self._checkpoint(example.example_id), prediction)
        if not isinstance(result, ToolSandboxEvaluationResult):
            raise ValidationError("ToolSandbox bridge returned an invalid result type")
        return result.metrics

    @property
    def evaluation_provenance(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "bridge_id": self._bridge.bridge_id,
                "bridge_revision": self._bridge.bridge_revision,
                "cost_basis": self._bridge.cost_basis,
                "evaluation_scope": self._bridge.evaluation_scope,
                "official_runtime_verified": self._bridge.official_runtime_verified,
                "pairing_mode": self.pairing_mode,
                "provenance_status": self._bridge.provenance_status,
                "supports_snapshot_restore": self._bridge.supports_snapshot_restore,
                "source_revision": self._bridge.source_revision,
                "supports_user_simulator_replay": (self._bridge.supports_user_simulator_replay),
            }
        )

    def _checkpoint(self, checkpoint_id: str) -> ToolSandboxCheckpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise ValidationError(f"unknown ToolSandbox checkpoint {checkpoint_id!r}") from exc


def _unit_interval(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"ToolSandbox {label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValidationError(f"ToolSandbox {label} must be finite and in [0, 1]")
    return numeric


def _validate_bridge(bridge: ToolSandboxEvaluationBridge) -> None:
    if bridge.source_revision != OFFICIAL_SOURCE_REVISION:
        raise ValidationError("ToolSandbox bridge source revision is not audited")
    if bridge.provenance_status != _CALLER_ASSERTED:
        raise ValidationError("ToolSandbox bridge provenance must remain caller_asserted")
    if bridge.evaluation_scope != _EXTERNAL_OFFICIAL_SEMANTICS:
        raise ValidationError("ToolSandbox bridge scope must remain external_official_semantics")
    if bridge.official_runtime_verified is not False:
        raise ValidationError("current ToolSandbox bridge cannot claim runtime verification")
    for label in (
        "supports_snapshot_restore",
        "supports_user_simulator_replay",
    ):
        if not isinstance(getattr(bridge, label), bool):
            raise ValidationError(f"ToolSandbox bridge {label} must be boolean")
    for label in ("bridge_id", "cost_basis"):
        value = getattr(bridge, label)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValidationError(f"ToolSandbox bridge {label} must be non-empty")
    if (
        not isinstance(bridge.bridge_revision, str)
        or _SHA256.fullmatch(bridge.bridge_revision) is None
    ):
        raise ValidationError("ToolSandbox bridge revision must be a lowercase SHA-256")


__all__ = [
    "BENCHMARK_ID",
    "OFFICIAL_BASE_SCENARIO_COUNT",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_SCENARIO_MEMBERSHIP_ASSET",
    "OFFICIAL_SCENARIO_MEMBERSHIP_SHA256",
    "OFFICIAL_SOURCE_REVISION",
    "OFFICIAL_VARIANT_SCENARIO_COUNT",
    "CallerAssertedToolSandboxEvaluationBridge",
    "ToolSandboxCheckpoint",
    "ToolSandboxCommonSupportAdapter",
    "ToolSandboxEvaluationBridge",
    "ToolSandboxEvaluationResult",
    "ToolSandboxFamilyEligibility",
    "ToolSandboxScenarioInventory",
]
