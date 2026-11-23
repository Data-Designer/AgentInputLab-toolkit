"""Fail-closed AppWorld-UL checkpoint and external-evaluation contracts.

AppWorld-UL extends AppWorld with a user simulator and evaluator state that are
not part of the optimizer-visible prompt.  This module therefore does not copy,
reimplement, or infer those protected upstream components.  It binds a captured
pre-action checkpoint to opaque runtime identities and delegates continuation
and scoring to an explicitly caller-asserted external bridge.

Shared catalog wiring is owned by the repository's integration layer.  This
module becomes a runnable, production-attested benchmark only after a caller
supplies checkpoints captured from the pinned official runtime and a bridge
that restores and evaluates them with the upstream implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.contracts.state import ContextBlock, Message

from .base import BenchmarkExample, opaque_checkpoint_state_id

APPWORLD_UL_PROJECT_URL = "https://appworld.dev/appworld-ul/"
APPWORLD_UPSTREAM_REPOSITORY = "https://github.com/StonyBrookNLP/appworld"
AUDITED_APPWORLD_UPSTREAM_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"

_BENCHMARK_ID = "appworld_ul.common_support"
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALLER_ASSERTED = "caller_asserted"
_STRUCTURAL_CANDIDATE = "structural_candidate"
_EXTERNAL_BRIDGE = "external_bridge"
_STRUCTURAL_CAPABILITY_NAMES = frozenset(
    {
        "clarification_roundtrip_available",
        "state_restore_available",
    }
)
_INTERVENTION_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)


@dataclass(frozen=True, slots=True)
class AppWorldULFamilyEligibility:
    """Five-family support derived only from optimizer-visible structure."""

    checkpoint_id: str
    eligibility: Mapping[str, bool]
    reasons: Mapping[str, str]
    provenance_status: str = _CALLER_ASSERTED
    support_status: str = _STRUCTURAL_CANDIDATE

    def __post_init__(self) -> None:
        _require_trimmed_text(self.checkpoint_id, label="checkpoint_id")
        expected = {family.value for family in _INTERVENTION_FAMILIES}
        if set(self.eligibility) != expected or set(self.reasons) != expected:
            raise ValidationError(
                "AppWorld-UL family eligibility must cover exactly the five core families"
            )
        if any(not isinstance(value, bool) for value in self.eligibility.values()):
            raise ValidationError("AppWorld-UL family eligibility values must be boolean")
        if any(not isinstance(value, str) or not value.strip() for value in self.reasons.values()):
            raise ValidationError(
                "AppWorld-UL family eligibility reasons must be non-empty strings"
            )
        if self.provenance_status != _CALLER_ASSERTED:
            raise ValidationError("AppWorld-UL eligibility provenance must remain caller_asserted")
        if self.support_status != _STRUCTURAL_CANDIDATE:
            raise ValidationError("AppWorld-UL eligibility status must remain structural_candidate")
        object.__setattr__(self, "eligibility", MappingProxyType(dict(self.eligibility)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))

    @property
    def eligible_families(self) -> tuple[SemanticFamily, ...]:
        return tuple(family for family in _INTERVENTION_FAMILIES if self.eligibility[family.value])

    @property
    def supports_full_block(self) -> bool:
        """Whether all five transformations have visible structural support."""

        return all(self.eligibility.values())


@dataclass(frozen=True, slots=True)
class AppWorldULCheckpoint:
    """One optimizer-visible state bound to opaque official-runtime state.

    ``runtime_state_handle`` is passed only to the evaluator bridge.  The hash
    identities bind the protected runtime state, user-simulator initialization,
    data bundle, tool catalog, and evaluator without exposing their contents.
    ``visible_read_tool_names`` is a caller-audited subset of ``tool_names``;
    names are never guessed from hidden task labels.
    """

    checkpoint_id: str
    task_id: str = field(repr=False)
    split: str
    query: str
    instruction: str
    visible_messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    visible_read_tool_names: tuple[str, ...]
    runtime_state_handle: str = field(repr=False)
    runtime_state_identity: str = field(repr=False)
    user_simulator_identity: str = field(repr=False)
    data_revision: str = field(repr=False)
    tool_catalog_revision: str = field(repr=False)
    evaluator_identity: str = field(repr=False)
    replay_identity: str = field(repr=False)
    upstream_commit: str = AUDITED_APPWORLD_UPSTREAM_COMMIT
    visible_context: tuple[ContextBlock, ...] = ()
    structural_capabilities: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in (
            "checkpoint_id",
            "task_id",
            "split",
            "query",
            "instruction",
            "runtime_state_handle",
        ):
            _require_trimmed_text(getattr(self, label), label=label)
        if (
            not isinstance(self.upstream_commit, str)
            or _GIT_COMMIT.fullmatch(self.upstream_commit) is None
        ):
            raise ValidationError(
                "AppWorld-UL checkpoint upstream_commit must be a lowercase Git commit"
            )
        if self.upstream_commit != AUDITED_APPWORLD_UPSTREAM_COMMIT:
            raise ValidationError(
                "AppWorld-UL checkpoint does not match the audited upstream commit"
            )
        for label in (
            "runtime_state_identity",
            "user_simulator_identity",
            "data_revision",
            "tool_catalog_revision",
            "evaluator_identity",
            "replay_identity",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValidationError(f"AppWorld-UL checkpoint {label} must be a lowercase SHA-256")

        messages = tuple(self.visible_messages)
        context = tuple(self.visible_context)
        tool_names = tuple(self.tool_names)
        read_tool_names = tuple(self.visible_read_tool_names)
        if not messages or not all(isinstance(message, Message) for message in messages):
            raise ValidationError(
                "AppWorld-UL checkpoint requires captured visible Message instances"
            )
        if not all(isinstance(block, ContextBlock) for block in context):
            raise ValidationError(
                "AppWorld-UL checkpoint context must contain ContextBlock instances"
            )
        _validate_tool_names(tool_names, label="visible tool names", require_nonempty=True)
        _validate_tool_names(
            read_tool_names,
            label="visible read-tool names",
            require_nonempty=False,
        )
        if not set(read_tool_names) <= set(tool_names):
            raise ValidationError(
                "AppWorld-UL visible_read_tool_names must be a subset of tool_names"
            )
        user_messages = [message for message in messages if message.role == "user"]
        if not user_messages:
            raise ValidationError(
                "AppWorld-UL checkpoint requires at least one visible user message"
            )
        if user_messages[-1].content != self.query:
            raise ValidationError(
                "AppWorld-UL checkpoint query must equal the latest visible user message"
            )

        capabilities = dict(self.structural_capabilities)
        if not set(capabilities) <= _STRUCTURAL_CAPABILITY_NAMES:
            raise ValidationError(
                "AppWorld-UL checkpoint contains an unknown structural capability; "
                "task types and defect labels are not eligibility inputs"
            )
        if any(not isinstance(value, bool) for value in capabilities.values()):
            raise ValidationError("AppWorld-UL checkpoint structural capabilities must be boolean")
        object.__setattr__(self, "visible_messages", messages)
        object.__setattr__(self, "visible_context", context)
        object.__setattr__(self, "tool_names", tool_names)
        object.__setattr__(self, "visible_read_tool_names", read_tool_names)
        object.__setattr__(
            self,
            "structural_capabilities",
            MappingProxyType(capabilities),
        )

    @property
    def family_eligibility(self) -> AppWorldULFamilyEligibility:
        acquisition = bool(self.visible_read_tool_names)
        clarification = self.structural_capabilities.get(
            "clarification_roundtrip_available",
            False,
        )
        curation = len(self.visible_messages) >= 3 or len(self.visible_context) >= 2
        eligibility = {
            SemanticFamily.REFORMULATE.value: True,
            SemanticFamily.DECOMPOSE.value: True,
            SemanticFamily.ACQUIRE.value: acquisition,
            SemanticFamily.CLARIFY.value: clarification,
            SemanticFamily.CURATE.value: curation,
        }
        reasons = {
            SemanticFamily.REFORMULATE.value: (
                "captured checkpoint has an explicit visible user query"
            ),
            SemanticFamily.DECOMPOSE.value: (
                "captured checkpoint has an explicit visible query that can be decomposed"
            ),
            SemanticFamily.ACQUIRE.value: (
                "a caller-audited visible read-only tool channel is present"
                if acquisition
                else "requires at least one caller-audited visible read-only tool"
            ),
            SemanticFamily.CLARIFY.value: (
                "captured runtime exposes a user-simulator clarification round trip"
                if clarification
                else "requires clarification_roundtrip_available=true"
            ),
            SemanticFamily.CURATE.value: (
                "captured state has enough visible history or context to curate"
                if curation
                else "requires at least three visible messages or two visible context blocks"
            ),
        }
        return AppWorldULFamilyEligibility(self.checkpoint_id, eligibility, reasons)

    @property
    def structural_replay_candidate(self) -> bool:
        """Whether visible structure declares prerequisites for later replay audit."""

        return self.family_eligibility.supports_full_block and self.structural_capabilities.get(
            "state_restore_available",
            False,
        )


@dataclass(frozen=True, slots=True)
class AppWorldULEvaluationResult:
    """Typed task-level outcome from an external AppWorld-UL evaluator.

    AppWorld's scenario goal completion (SGC) is a grouped, offline metric and
    is deliberately not fabricated per checkpoint.  A complete official batch
    evaluator must compute it after all task variations have been executed.
    """

    task_success: float
    cost: float
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        metrics = _validated_bridge_metrics(
            {
                "cost": self.cost,
                "task_success": self.task_success,
            }
        )
        object.__setattr__(
            self,
            "task_success",
            metrics["task_success"],
        )
        object.__setattr__(self, "cost", metrics["cost"])

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "cost": self.cost,
                "task_success": self.task_success,
            }
        )


@runtime_checkable
class AppWorldULEvaluationBridge(Protocol):
    """Protocol for restoring and evaluating a protected upstream checkpoint."""

    appworld_ul_bridge_contract: str
    benchmark_id: str
    source_revision: str
    bridge_id: str
    bridge_revision: str
    cost_basis: str
    supports_state_restore: bool
    supports_user_simulator_replay: bool
    provenance_status: str
    evaluation_scope: str
    official_runtime_verified: bool

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        checkpoint: AppWorldULCheckpoint,
        prediction: Any,
    ) -> AppWorldULEvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CallerAssertedAppWorldULEvaluationBridge:
    """Explicit wrapper for a non-attested external AppWorld-UL callback."""

    bridge_id: str
    bridge_revision: str
    cost_basis: str
    supports_state_restore: bool
    supports_user_simulator_replay: bool
    callback: Callable[
        [Mapping[str, Any], AppWorldULCheckpoint, Any],
        AppWorldULEvaluationResult,
    ] = field(repr=False)
    appworld_ul_bridge_contract: str = field(
        init=False,
        default="appworld_ul_evaluation_bridge_v1",
    )
    benchmark_id: str = field(init=False, default=_BENCHMARK_ID)
    source_revision: str = field(init=False, default=AUDITED_APPWORLD_UPSTREAM_COMMIT)
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        for label in ("bridge_id", "cost_basis"):
            _require_trimmed_text(getattr(self, label), label=label)
        if (
            not isinstance(self.bridge_revision, str)
            or _SHA256.fullmatch(self.bridge_revision) is None
        ):
            raise ValidationError(
                "AppWorld-UL evaluation bridge revision must be a lowercase SHA-256"
            )
        if not callable(self.callback):
            raise TypeError("AppWorld-UL evaluation bridge callback must be callable")
        for label in (
            "supports_state_restore",
            "supports_user_simulator_replay",
        ):
            if not isinstance(getattr(self, label), bool):
                raise ValidationError(f"AppWorld-UL evaluation bridge {label} must be boolean")

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        checkpoint: AppWorldULCheckpoint,
        prediction: Any,
    ) -> AppWorldULEvaluationResult:
        result = self.callback(evaluator_payload, checkpoint, prediction)
        if not isinstance(result, AppWorldULEvaluationResult):
            raise ValidationError(
                "AppWorld-UL evaluation callback must return "
                "AppWorldULEvaluationResult; bare metric mappings are not accepted"
            )
        if (
            result.provenance_status != self.provenance_status
            or result.evaluation_scope != self.evaluation_scope
            or result.official_runtime_verified is not False
        ):
            raise ValidationError(
                "AppWorld-UL evaluation result provenance differs from its caller-asserted bridge"
            )
        return result


class AppWorldULCheckpointAdapter:
    """Evaluator-blind adapter over externally captured AppWorld-UL checkpoints."""

    benchmark_id = _BENCHMARK_ID
    adapter_metric_names = ("task_success", "cost")
    pairing_mode = "unsupported_external_replay"

    def __init__(
        self,
        *,
        checkpoints: Iterable[AppWorldULCheckpoint],
        evaluation_bridge: AppWorldULEvaluationBridge,
    ) -> None:
        if not isinstance(evaluation_bridge, AppWorldULEvaluationBridge):
            raise TypeError(
                "evaluation_bridge must implement AppWorldULEvaluationBridge; wrap "
                "callbacks with CallerAssertedAppWorldULEvaluationBridge"
            )
        _validate_evaluation_bridge(evaluation_bridge)
        normalized = tuple(checkpoints)
        if not normalized:
            raise ValidationError("AppWorld-UL requires at least one captured visible checkpoint")
        if not all(isinstance(item, AppWorldULCheckpoint) for item in normalized):
            raise TypeError("AppWorld-UL checkpoints must be AppWorldULCheckpoint instances")
        checkpoint_ids = [item.checkpoint_id for item in normalized]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValidationError("AppWorld-UL checkpoint IDs must be unique")
        experimental_units = [
            (
                item.task_id,
                item.runtime_state_identity,
                item.user_simulator_identity,
                item.data_revision,
                item.tool_catalog_revision,
                item.evaluator_identity,
                item.replay_identity,
            )
            for item in normalized
        ]
        if len(experimental_units) != len(set(experimental_units)):
            raise ValidationError("AppWorld-UL protected experimental units must be unique")

        examples: list[BenchmarkExample] = []
        examples_by_split: dict[str, list[BenchmarkExample]] = {}
        checkpoints_by_id: dict[str, AppWorldULCheckpoint] = {}
        for checkpoint in sorted(normalized, key=lambda item: item.checkpoint_id):
            support = checkpoint.family_eligibility
            clarification = support.eligibility[SemanticFamily.CLARIFY.value]
            state_restore = checkpoint.structural_capabilities.get(
                "state_restore_available",
                False,
            )
            if clarification and not evaluation_bridge.supports_user_simulator_replay:
                raise ValidationError(
                    "AppWorld-UL clarification checkpoint requires a bridge with "
                    "user-simulator replay support"
                )
            if state_restore and not evaluation_bridge.supports_state_restore:
                raise ValidationError(
                    "AppWorld-UL checkpoint advertises state restoration but the bridge "
                    "does not support it"
                )
            if support.supports_full_block and not evaluation_bridge.supports_state_restore:
                raise ValidationError(
                    "AppWorld-UL full-block checkpoint requires a bridge with state restore support"
                )
            visible_metadata = _frozen_mapping(
                {
                    "benchmark_cell": self.benchmark_id,
                    "family_eligibility": dict(support.eligibility),
                    "family_eligibility_provenance": support.provenance_status,
                    "family_eligibility_reasons": dict(support.reasons),
                    "family_support_status": support.support_status,
                    "source_split": checkpoint.split,
                },
                label="AppWorld-UL visible checkpoint metadata",
            )
            evaluator_payload = _frozen_mapping(
                {
                    "data_revision": checkpoint.data_revision,
                    "evaluator_identity": checkpoint.evaluator_identity,
                    "replay_identity": checkpoint.replay_identity,
                    "runtime_state_handle": checkpoint.runtime_state_handle,
                    "runtime_state_identity": checkpoint.runtime_state_identity,
                    "source_split": checkpoint.split,
                    "source_task_id": checkpoint.task_id,
                    "tool_catalog_revision": checkpoint.tool_catalog_revision,
                    "upstream_commit": checkpoint.upstream_commit,
                    "user_simulator_identity": checkpoint.user_simulator_identity,
                },
                label="AppWorld-UL evaluator payload",
            )
            example = BenchmarkExample(
                example_id=checkpoint.checkpoint_id,
                query=checkpoint.query,
                instruction=checkpoint.instruction,
                visible_context=checkpoint.visible_context,
                visible_messages=checkpoint.visible_messages,
                visible_tool_names=checkpoint.tool_names,
                visible_metadata=visible_metadata,
                evaluator_gold=evaluator_payload,
                optimizer_visible_id=opaque_checkpoint_state_id(
                    self.benchmark_id,
                    checkpoint.checkpoint_id,
                    checkpoint.replay_identity,
                ),
            )
            examples.append(example)
            examples_by_split.setdefault(checkpoint.split, []).append(example)
            checkpoints_by_id[checkpoint.checkpoint_id] = checkpoint

        self._evaluation_bridge = evaluation_bridge
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._examples_by_split = MappingProxyType(
            {
                split: tuple(split_examples)
                for split, split_examples in sorted(examples_by_split.items())
            }
        )
        self._checkpoints = MappingProxyType(checkpoints_by_id)

    @property
    def available_splits(self) -> tuple[str, ...]:
        return tuple(self._examples_by_split)

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        try:
            return self._examples_by_split[split]
        except KeyError as exc:
            raise ValidationError(f"AppWorld-UL adapter does not expose split {split!r}") from exc

    def family_eligibility(self, checkpoint_id: str) -> AppWorldULFamilyEligibility:
        return self._checkpoint(checkpoint_id).family_eligibility

    def eligible_families(self, checkpoint_id: str) -> tuple[SemanticFamily, ...]:
        return self.family_eligibility(checkpoint_id).eligible_families

    def supports_full_block(self, checkpoint_id: str) -> bool:
        return self.family_eligibility(checkpoint_id).supports_full_block

    def structural_replay_candidate(self, checkpoint_id: str) -> bool:
        checkpoint = self._checkpoint(checkpoint_id)
        return (
            checkpoint.structural_replay_candidate
            and self._evaluation_bridge.supports_state_restore
            and self._evaluation_bridge.supports_user_simulator_replay
        )

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this AppWorld-UL adapter")
        checkpoint = self._checkpoints[example.example_id]
        result = self._evaluation_bridge.evaluate(
            example.evaluator_gold,
            checkpoint,
            prediction,
        )
        if not isinstance(result, AppWorldULEvaluationResult):
            raise ValidationError("AppWorld-UL evaluation bridge returned an invalid result type")
        return result.metrics

    @property
    def evaluation_provenance(self) -> Mapping[str, Any]:
        return _frozen_mapping(
            {
                "bridge_id": self._evaluation_bridge.bridge_id,
                "bridge_revision": self._evaluation_bridge.bridge_revision,
                "cost_basis": self._evaluation_bridge.cost_basis,
                "evaluation_scope": self._evaluation_bridge.evaluation_scope,
                "official_runtime_verified": (self._evaluation_bridge.official_runtime_verified),
                "provenance_status": self._evaluation_bridge.provenance_status,
                "supports_state_restore": (self._evaluation_bridge.supports_state_restore),
                "supports_user_simulator_replay": (
                    self._evaluation_bridge.supports_user_simulator_replay
                ),
            },
            label="AppWorld-UL evaluation provenance",
        )

    @property
    def source_provenance(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "benchmark_id": self.benchmark_id,
                "project_url": APPWORLD_UL_PROJECT_URL,
                "repository_url": APPWORLD_UPSTREAM_REPOSITORY,
                "upstream_commit": AUDITED_APPWORLD_UPSTREAM_COMMIT,
            }
        )

    def _checkpoint(self, checkpoint_id: str) -> AppWorldULCheckpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise ValidationError(f"unknown AppWorld-UL checkpoint {checkpoint_id!r}") from exc


def _validated_bridge_metrics(value: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValidationError("AppWorld-UL evaluation bridge metrics must be a mapping")
    expected = {"cost", "task_success"}
    if set(value) != expected:
        raise ValidationError(
            "AppWorld-UL evaluation bridge must return exactly task_success and cost"
        )
    metrics: dict[str, float] = {}
    for key in sorted(expected):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError(f"AppWorld-UL evaluation bridge metric {key!r} must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValidationError(f"AppWorld-UL evaluation bridge metric {key!r} must be finite")
        metrics[key] = numeric
    if metrics["task_success"] not in (0.0, 1.0):
        raise ValidationError("AppWorld-UL task_success must be the official binary 0/1 bit")
    if metrics["cost"] < 0.0:
        raise ValidationError("AppWorld-UL cost must be non-negative")
    return MappingProxyType(metrics)


def _validate_evaluation_bridge(bridge: AppWorldULEvaluationBridge) -> None:
    if bridge.appworld_ul_bridge_contract != "appworld_ul_evaluation_bridge_v1":
        raise ValidationError("AppWorld-UL bridge contract identity does not match")
    if bridge.benchmark_id != _BENCHMARK_ID:
        raise ValidationError("AppWorld-UL bridge benchmark identity does not match")
    if bridge.source_revision != AUDITED_APPWORLD_UPSTREAM_COMMIT:
        raise ValidationError(
            "AppWorld-UL bridge source revision does not match the audited upstream commit"
        )
    if bridge.provenance_status != _CALLER_ASSERTED:
        raise ValidationError("AppWorld-UL bridge provenance must remain caller_asserted")
    if bridge.evaluation_scope != _EXTERNAL_BRIDGE:
        raise ValidationError("AppWorld-UL bridge evaluation scope must remain external_bridge")
    if bridge.official_runtime_verified is not False:
        raise ValidationError(
            "current AppWorld-UL bridge cannot claim official runtime verification"
        )
    for label in ("bridge_id", "cost_basis"):
        _require_trimmed_text(getattr(bridge, label), label=label)
    for label in (
        "supports_state_restore",
        "supports_user_simulator_replay",
    ):
        if not isinstance(getattr(bridge, label), bool):
            raise ValidationError(f"AppWorld-UL evaluation bridge {label} must be boolean")
    if (
        not isinstance(bridge.bridge_revision, str)
        or _SHA256.fullmatch(bridge.bridge_revision) is None
    ):
        raise ValidationError("AppWorld-UL evaluation bridge revision must be a lowercase SHA-256")


def _frozen_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        return freeze_mapping(value)
    except TypeError as exc:
        raise ValidationError(f"{label} must be canonically serializable: {exc}") from exc


def _require_trimmed_text(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"AppWorld-UL {label} must be a trimmed non-empty string")


def _validate_tool_names(
    values: tuple[str, ...],
    *,
    label: str,
    require_nonempty: bool,
) -> None:
    if require_nonempty and not values:
        raise ValidationError(f"AppWorld-UL checkpoint requires {label}")
    if not all(
        isinstance(value, str) and value.strip() and value == value.strip() for value in values
    ):
        raise ValidationError(f"AppWorld-UL checkpoint {label} must be trimmed non-empty strings")
    if len(values) != len(set(values)):
        raise ValidationError(f"AppWorld-UL checkpoint {label} must be unique")
