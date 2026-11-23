"""External checkpoint bridge for Microsoft STATE-Bench.

STATE-Bench owns the task environments, domain tools, user simulator,
continuation loop, and scorers.  This module does not copy or reimplement that
runtime.  It validates an optimizer-visible pre-action checkpoint, derives
five-family structural eligibility from visible state, and delegates reset,
continuation, and scoring to an explicitly typed external bridge.

The integration is pinned to STATE-Bench v0.8.1 at commit
``4efcbf2d4fe60df04878859b692d9391f3d5b33a``.  A bridge remains
``caller_asserted`` until a future direct runtime integration can attest the
official execution itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.contracts.state import ContextBlock, Message

from .base import BenchmarkExample, opaque_checkpoint_state_id

STATEBENCH_BENCHMARK_ID = "state_bench.common_support"
STATEBENCH_EXTERNAL_SPLIT = "external_checkpoints"
STATEBENCH_OFFICIAL_SOURCE_REVISION = "4efcbf2d4fe60df04878859b692d9391f3d5b33a"
STATEBENCH_OFFICIAL_VERSION = "0.8.1"
STATEBENCH_OFFICIAL_TASK_MEMBERSHIP_REVISION = (
    "1410974ccd3ea80e1d20425ba4b11539a0883eb2029b3b1023fa8627ed1b2870"
)
STATEBENCH_OFFICIAL_PROTOCOL_ID = "state_bench_v0.8.1_gpt54"
STATEBENCH_OFFICIAL_PROTOCOL_REVISION = (
    "e4a97cab2b2ed31ec180f671f4b5e5760c00cf30cc56d4cf717481e7a0d29a0c"
)
STATEBENCH_OFFICIAL_SOURCE_SPLIT = "test"
STATEBENCH_OFFICIAL_RUNS_PER_TASK = 5
STATEBENCH_OFFICIAL_MAX_AGENT_TURNS = 15

STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "customer_support": 150,
        "shopping_assistant": 150,
        "travel": 150,
    }
)

STATEBENCH_OFFICIAL_TOOL_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "travel": (
            "search_flights",
            "get_user_details",
            "get_user_reservations",
            "get_booking",
            "get_flight_status",
            "get_policies",
            "create_booking",
            "update_booking",
            "cancel_booking",
            "search_hotels",
            "book_hotel",
            "get_hotel_reservation",
            "cancel_hotel_reservation",
            "search_car_rentals",
            "book_car_rental",
            "get_car_rental",
            "cancel_car_rental",
        ),
        "customer_support": (
            "get_order",
            "get_customer",
            "search_products",
            "get_product_details",
            "get_policies",
            "get_warranty_status",
            "process_return",
            "process_refund",
            "cancel_order",
            "process_exchange",
            "process_warranty_claim",
        ),
        "shopping_assistant": (
            "search_products",
            "get_product_details",
            "get_variants",
            "get_customer_account",
            "get_cart",
            "check_compatibility",
            "get_promotions",
            "get_policies",
            "validate_promo",
            "add_to_cart",
            "update_cart_item",
            "remove_from_cart",
            "apply_promo",
            "remove_promo",
            "redeem_loyalty_points",
            "cancel_loyalty_redemption",
            "get_shipping_options",
            "set_shipping_option",
        ),
    }
)

_OFFICIAL_READ_TOOL_NAMES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "travel": frozenset(
            {
                "search_flights",
                "get_user_details",
                "get_user_reservations",
                "get_booking",
                "get_flight_status",
                "get_policies",
                "search_hotels",
                "get_hotel_reservation",
                "search_car_rentals",
                "get_car_rental",
            }
        ),
        "customer_support": frozenset(
            {
                "get_order",
                "get_customer",
                "search_products",
                "get_product_details",
                "get_policies",
                "get_warranty_status",
            }
        ),
        "shopping_assistant": frozenset(
            {
                "search_products",
                "get_product_details",
                "get_variants",
                "get_customer_account",
                "get_cart",
                "check_compatibility",
                "get_promotions",
                "get_policies",
                "validate_promo",
                "get_shipping_options",
            }
        ),
    }
)

STATEBENCH_OFFICIAL_TOOL_CATALOG_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "travel": "8578328f635d01a620f8112ff24b6c9e439dd10e6eb62c688b72d1d1f31162a6",
        "customer_support": ("ac36ecc06cf02c011b262648ed88a53eb00fa33adc99fdd6fb51ec7918c86cbc"),
        "shopping_assistant": ("eeece103dbd3b992d25f0a5dab4c6d9e2670b089f6ed1a4b9972a5b831674f0f"),
    }
)

STATEBENCH_OFFICIAL_TASK_CATALOG_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "travel": "b7d92174e35293f7dbcf0b70725caa65e2bc73363d013249caa27d7a4b725ddb",
        "customer_support": ("164c86c890ea26f6a4f942ae4fc4f05c8536619a7071bc588dedbdf98622c41b"),
        "shopping_assistant": ("15aa50a4a4a231d711cfce575827bc4e107894c94583631b2c2a803c86634faa"),
    }
)

STATEBENCH_OFFICIAL_ENVIRONMENT_CATALOG_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "travel": "2eaef6d775da6edd1fd2301aac556258abbb992d6e7a28963996ee409e36d5a2",
        "customer_support": ("df6c5fe16d7033f988dede7c6a1bd8f174a467ff8da68998a41eb250c793f152"),
        "shopping_assistant": ("d9007ed2aa16b8effc636b4210c67edaf93dfa9c335a6100f64e03ca42c9b217"),
    }
)

STATEBENCH_OFFICIAL_SPLIT_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "travel": "55ba8285139a7edfca25e589562ae72622d0b633c598cbb2963e103b3855cb9f",
        "customer_support": ("2988c8c87d426babf0daa3f342c3110caefb63fa0bac97650a38fc2dcfa7291e"),
        "shopping_assistant": ("bc9be8ae4afce8fc817c5a3e2c3d72373009742d726a853cbcb743d9b4a09938"),
    }
)

STATEBENCH_OFFICIAL_SIMULATOR_PROMPT_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "travel": "6a94abedb77e2ec3f3b94a149291bc42a6aeacd5691ef0cb69862b9af923ae3c",
        "customer_support": ("0b58e108ac16469926ee210774e6e3275049d898d35ea23ef40c13ddfbe829d8"),
        "shopping_assistant": ("70a546fce5108cb4381c6c6c6ea0adccd6e3d91b99b90a14f1c09d1b1f2bc5fc"),
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID = re.compile(r"^(?P<number>[1-9][0-9]{0,2})-[a-z0-9][a-z0-9_-]*$")
_CALLER_ASSERTED = "caller_asserted"
_STRUCTURAL_CANDIDATE = "structural_candidate"
_EXTERNAL_BRIDGE = "external_single_trajectory"
_OFFICIAL_COST_BASIS = "agent_reported_usd"
_ALLOWED_CAPABILITIES = frozenset(
    {
        "clarification_roundtrip_available",
        "continuation_available",
        "fresh_task_reset_available",
        "mid_dialogue_clone_available",
    }
)
_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)


def _load_official_task_ids() -> tuple[
    Mapping[str, frozenset[str]],
    Mapping[str, Mapping[str, frozenset[str]]],
]:
    resource = files("agentinputlab.benchmarks").joinpath(
        "data",
        "statebench_v0_8_1_task_ids.json",
    )
    payload_bytes = resource.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != STATEBENCH_OFFICIAL_TASK_MEMBERSHIP_REVISION:
        raise RuntimeError("STATE-Bench packaged task membership differs from the pinned revision")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("STATE-Bench packaged task membership is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("benchmark_version") != STATEBENCH_OFFICIAL_VERSION
        or payload.get("source_revision") != STATEBENCH_OFFICIAL_SOURCE_REVISION
        or payload.get("split_version") != "train_test"
    ):
        raise RuntimeError("STATE-Bench packaged task membership has the wrong source revision")
    domains = payload.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS):
        raise RuntimeError("STATE-Bench packaged task membership has the wrong domains")

    normalized: dict[str, frozenset[str]] = {}
    normalized_by_split: dict[str, dict[str, frozenset[str]]] = {
        "test": {},
        "train": {},
    }
    for domain, expected_count in STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS.items():
        domain_splits = domains[domain]
        if not isinstance(domain_splits, dict) or set(domain_splits) != {"test", "train"}:
            raise RuntimeError(
                f"STATE-Bench packaged task membership is invalid for domain {domain!r}"
            )
        split_task_ids: dict[str, frozenset[str]] = {}
        for split, split_count in (("train", 100), ("test", 50)):
            task_ids = domain_splits[split]
            if (
                not isinstance(task_ids, list)
                or any(not isinstance(task_id, str) for task_id in task_ids)
                or len(task_ids) != split_count
                or len(set(task_ids)) != split_count
                or any(_TASK_ID.fullmatch(task_id) is None for task_id in task_ids)
            ):
                raise RuntimeError(
                    f"STATE-Bench packaged {split} task membership is invalid for domain {domain!r}"
                )
            split_task_ids[split] = frozenset(task_ids)
            normalized_by_split[split][domain] = split_task_ids[split]
        all_task_ids = split_task_ids["train"] | split_task_ids["test"]
        numeric_prefixes = {int(task_id.split("-", 1)[0]) for task_id in all_task_ids}
        if (
            not split_task_ids["train"].isdisjoint(split_task_ids["test"])
            or len(all_task_ids) != expected_count
            or numeric_prefixes != set(range(1, expected_count + 1))
        ):
            raise RuntimeError(
                f"STATE-Bench packaged task membership is incomplete for domain {domain!r}"
            )
        normalized[domain] = frozenset(all_task_ids)
    return MappingProxyType(normalized), MappingProxyType(
        {
            split: MappingProxyType(task_ids_by_domain)
            for split, task_ids_by_domain in normalized_by_split.items()
        }
    )


STATEBENCH_OFFICIAL_TASK_IDS, STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT = _load_official_task_ids()


@dataclass(frozen=True, slots=True)
class StateBenchFamilyEligibility:
    """Visible-only, caller-asserted support for one STATE-Bench checkpoint."""

    checkpoint_id: str
    eligibility: Mapping[str, bool]
    reasons: Mapping[str, str]
    provenance_status: str = _CALLER_ASSERTED
    support_status: str = _STRUCTURAL_CANDIDATE

    def __post_init__(self) -> None:
        expected = {family.value for family in _FAMILIES}
        if set(self.eligibility) != expected or set(self.reasons) != expected:
            raise ValidationError(
                "STATE-Bench family eligibility must cover the five core families"
            )
        if any(not isinstance(value, bool) for value in self.eligibility.values()):
            raise ValidationError("STATE-Bench family eligibility values must be boolean")
        if any(not isinstance(value, str) or not value for value in self.reasons.values()):
            raise ValidationError(
                "STATE-Bench family eligibility reasons must be non-empty strings"
            )
        if self.provenance_status != _CALLER_ASSERTED:
            raise ValidationError("STATE-Bench eligibility provenance must remain caller_asserted")
        if self.support_status != _STRUCTURAL_CANDIDATE:
            raise ValidationError(
                "STATE-Bench eligibility support must remain structural_candidate"
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
class StateBenchCheckpoint:
    """Optimizer-visible state plus opaque bindings to the official runtime.

    Hidden task summaries, simulator instructions, state requirements, task
    requirements, task-local database rows, and scorer prompts are deliberately
    absent.  They remain inside the official external runtime.
    """

    checkpoint_id: str
    task_id: str = field(repr=False)
    domain: str
    query: str
    instruction: str
    visible_messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    turn_index: int
    run_index: int
    source_revision: str
    protocol_id: str
    protocol_revision: str
    task_catalog_revision: str
    environment_catalog_revision: str
    split_revision: str
    tool_catalog_revision: str
    simulator_prompt_revision: str
    runtime_snapshot_id: str = field(repr=False)
    initial_environment_revision: str = field(repr=False)
    active_environment_revision: str = field(repr=False)
    simulator_session_identity: str = field(repr=False)
    replay_identity: str = field(repr=False)
    continuation_identity: str = field(repr=False)
    visible_context: tuple[ContextBlock, ...] = ()
    structural_capabilities: Mapping[str, bool] = field(default_factory=dict)
    source_split: str = STATEBENCH_OFFICIAL_SOURCE_SPLIT

    def __post_init__(self) -> None:
        for label in ("checkpoint_id", "query", "instruction"):
            _require_trimmed_string(getattr(self, label), f"STATE-Bench checkpoint {label}")
        if self.domain not in STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS:
            raise ValidationError(
                "STATE-Bench checkpoint domain must be travel, customer_support, "
                "or shopping_assistant"
            )
        if (
            isinstance(self.turn_index, bool)
            or not isinstance(self.turn_index, int)
            or not 0 <= self.turn_index < STATEBENCH_OFFICIAL_MAX_AGENT_TURNS
        ):
            raise ValidationError("STATE-Bench turn_index must be an integer in [0, 14]")
        if (
            isinstance(self.run_index, bool)
            or not isinstance(self.run_index, int)
            or not 1 <= self.run_index <= STATEBENCH_OFFICIAL_RUNS_PER_TASK
        ):
            raise ValidationError("STATE-Bench run_index must be an integer in [1, 5]")

        expected_values = {
            "source_revision": STATEBENCH_OFFICIAL_SOURCE_REVISION,
            "protocol_id": STATEBENCH_OFFICIAL_PROTOCOL_ID,
            "protocol_revision": STATEBENCH_OFFICIAL_PROTOCOL_REVISION,
            "task_catalog_revision": (STATEBENCH_OFFICIAL_TASK_CATALOG_REVISIONS[self.domain]),
            "environment_catalog_revision": (
                STATEBENCH_OFFICIAL_ENVIRONMENT_CATALOG_REVISIONS[self.domain]
            ),
            "split_revision": STATEBENCH_OFFICIAL_SPLIT_REVISIONS[self.domain],
            "tool_catalog_revision": (STATEBENCH_OFFICIAL_TOOL_CATALOG_REVISIONS[self.domain]),
            "simulator_prompt_revision": (
                STATEBENCH_OFFICIAL_SIMULATOR_PROMPT_REVISIONS[self.domain]
            ),
            "source_split": STATEBENCH_OFFICIAL_SOURCE_SPLIT,
        }
        for label, expected in expected_values.items():
            if getattr(self, label) != expected:
                raise ValidationError(
                    f"STATE-Bench checkpoint {label} differs from the pinned v0.8.1 contract"
                )
        _validate_task_id(
            self.task_id,
            domain=self.domain,
            source_split=self.source_split,
        )

        for label in (
            "runtime_snapshot_id",
            "initial_environment_revision",
            "active_environment_revision",
            "simulator_session_identity",
            "replay_identity",
            "continuation_identity",
        ):
            _require_sha256(getattr(self, label), f"STATE-Bench checkpoint {label}")

        messages = tuple(self.visible_messages)
        context = tuple(self.visible_context)
        tool_names = tuple(self.tool_names)
        if not messages or not all(isinstance(message, Message) for message in messages):
            raise ValidationError(
                "STATE-Bench checkpoint requires captured visible Message instances"
            )
        if not all(isinstance(block, ContextBlock) for block in context):
            raise ValidationError(
                "STATE-Bench checkpoint visible_context must contain ContextBlock instances"
            )
        user_messages = [message for message in messages if message.role == "user"]
        if not user_messages:
            raise ValidationError("STATE-Bench checkpoint requires a visible user message")
        if user_messages[-1].content != self.query:
            raise ValidationError("STATE-Bench query must equal the latest visible user message")
        if tool_names != STATEBENCH_OFFICIAL_TOOL_NAMES[self.domain]:
            raise ValidationError(
                "STATE-Bench tool_names must preserve the pinned official domain tool order"
            )

        capabilities = dict(self.structural_capabilities)
        if not set(capabilities) <= _ALLOWED_CAPABILITIES:
            raise ValidationError(
                "STATE-Bench checkpoint contains unknown structural capabilities; "
                "gold, defect labels, and task requirements are not eligibility inputs"
            )
        if any(not isinstance(value, bool) for value in capabilities.values()):
            raise ValidationError("STATE-Bench checkpoint structural capabilities must be boolean")
        object.__setattr__(self, "visible_messages", messages)
        object.__setattr__(self, "visible_context", context)
        object.__setattr__(self, "tool_names", tool_names)
        object.__setattr__(
            self,
            "structural_capabilities",
            MappingProxyType(capabilities),
        )

    @property
    def replay_ready(self) -> bool:
        continuation = self.structural_capabilities.get("continuation_available", False)
        if self.turn_index == 0:
            reset = self.structural_capabilities.get("fresh_task_reset_available", False)
        else:
            reset = self.structural_capabilities.get("mid_dialogue_clone_available", False)
        return continuation and reset

    @property
    def family_eligibility(self) -> StateBenchFamilyEligibility:
        replay_ready = self.replay_ready
        read_tool_available = bool(set(self.tool_names) & _OFFICIAL_READ_TOOL_NAMES[self.domain])
        clarification = (
            replay_ready
            and self.turn_index < STATEBENCH_OFFICIAL_MAX_AGENT_TURNS - 1
            and self.structural_capabilities.get(
                "clarification_roundtrip_available",
                False,
            )
        )
        curation_material = len(self.visible_messages) >= 3 or len(self.visible_context) >= 2
        eligibility = {
            SemanticFamily.REFORMULATE.value: replay_ready,
            SemanticFamily.DECOMPOSE.value: replay_ready,
            SemanticFamily.ACQUIRE.value: replay_ready and read_tool_available,
            SemanticFamily.CLARIFY.value: clarification,
            SemanticFamily.CURATE.value: replay_ready and curation_material,
        }
        reset_requirement = (
            "fresh_task_reset_available=true"
            if self.turn_index == 0
            else "mid_dialogue_clone_available=true"
        )
        replay_reason = (
            "checkpoint has a bound continuation and reset/clone contract"
            if replay_ready
            else f"requires continuation_available=true and {reset_requirement}"
        )
        reasons = {
            SemanticFamily.REFORMULATE.value: replay_reason,
            SemanticFamily.DECOMPOSE.value: replay_reason,
            SemanticFamily.ACQUIRE.value: (
                "checkpoint is replayable and exposes official read/domain tools"
                if replay_ready and read_tool_available
                else "requires replayability and at least one pinned official read tool"
            ),
            SemanticFamily.CLARIFY.value: (
                "official user-simulator continuation is caller-asserted and a turn remains"
                if clarification
                else "requires replayability, clarification_roundtrip_available=true, "
                "and turn_index < 14"
            ),
            SemanticFamily.CURATE.value: (
                "checkpoint is replayable and has enough visible history/context to curate"
                if replay_ready and curation_material
                else "requires replayability plus at least three visible messages "
                "or two visible context blocks"
            ),
        }
        return StateBenchFamilyEligibility(self.checkpoint_id, eligibility, reasons)


@dataclass(frozen=True, slots=True)
class StateBenchEvaluationResult:
    """Official-shaped metrics for one externally evaluated trajectory."""

    task_completion_pass: int
    state_requirements_met: int
    task_requirements_met: int
    ux_score: float
    cost_usd: float
    turns: int
    tool_calls: int
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        state = _binary_metric(self.state_requirements_met, "state_requirements_met")
        task = _binary_metric(self.task_requirements_met, "task_requirements_met")
        completion = _binary_metric(self.task_completion_pass, "task_completion_pass")
        if completion != int(state == 1 and task == 1):
            raise ValidationError(
                "STATE-Bench task_completion_pass must be the conjunction of "
                "state_requirements_met and task_requirements_met"
            )
        ux = _finite_number(self.ux_score, "ux_score")
        if not 1.0 <= ux <= 5.0:
            raise ValidationError("STATE-Bench ux_score must be in [1, 5]")
        cost = _finite_number(self.cost_usd, "cost_usd")
        if cost < 0:
            raise ValidationError("STATE-Bench cost_usd must be non-negative")
        turns = _bounded_integer(
            self.turns,
            "turns",
            minimum=1,
            maximum=STATEBENCH_OFFICIAL_MAX_AGENT_TURNS,
        )
        tool_calls = _bounded_integer(self.tool_calls, "tool_calls", minimum=0)
        object.__setattr__(self, "task_completion_pass", completion)
        object.__setattr__(self, "state_requirements_met", state)
        object.__setattr__(self, "task_requirements_met", task)
        object.__setattr__(self, "ux_score", ux)
        object.__setattr__(self, "cost_usd", cost)
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "tool_calls", tool_calls)

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "cost_usd": self.cost_usd,
                "state_requirements_met": float(self.state_requirements_met),
                "task_completion_pass": float(self.task_completion_pass),
                "task_requirements_met": float(self.task_requirements_met),
                "tool_calls": float(self.tool_calls),
                "turns": float(self.turns),
                "ux_score": self.ux_score,
            }
        )


@runtime_checkable
class StateBenchEvaluationBridge(Protocol):
    """External owner of STATE-Bench reset, continuation, and official scoring."""

    bridge_id: str
    bridge_revision: str
    source_revision: str
    protocol_id: str
    protocol_revision: str
    cost_basis: str
    provenance_status: str
    evaluation_scope: str
    official_runtime_verified: bool
    preserves_domain_tools: bool
    preserves_user_simulator: bool
    preserves_scorer: bool
    preserves_turn_limit: bool
    supports_fresh_task_reset: bool
    supports_mid_dialogue_clone: bool
    supports_simulator_replay: bool

    def evaluate(
        self,
        checkpoint: StateBenchCheckpoint,
        prediction: Any,
    ) -> StateBenchEvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CallerAssertedStateBenchEvaluationBridge:
    """Typed wrapper for an external, not-yet-attested STATE-Bench runtime."""

    bridge_id: str
    bridge_revision: str
    callback: Callable[[StateBenchCheckpoint, Any], StateBenchEvaluationResult] = field(repr=False)
    supports_fresh_task_reset: bool
    supports_mid_dialogue_clone: bool
    supports_simulator_replay: bool
    preserves_domain_tools: bool
    preserves_user_simulator: bool
    preserves_scorer: bool
    preserves_turn_limit: bool
    source_revision: str = STATEBENCH_OFFICIAL_SOURCE_REVISION
    protocol_id: str = STATEBENCH_OFFICIAL_PROTOCOL_ID
    protocol_revision: str = STATEBENCH_OFFICIAL_PROTOCOL_REVISION
    cost_basis: str = _OFFICIAL_COST_BASIS
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        _validate_bridge(self)
        if not callable(self.callback):
            raise TypeError("STATE-Bench evaluation bridge callback must be callable")

    def evaluate(
        self,
        checkpoint: StateBenchCheckpoint,
        prediction: Any,
    ) -> StateBenchEvaluationResult:
        result = self.callback(checkpoint, prediction)
        if not isinstance(result, StateBenchEvaluationResult):
            raise ValidationError(
                "STATE-Bench callback must return StateBenchEvaluationResult; "
                "bare metric mappings are not accepted"
            )
        if (
            result.provenance_status != self.provenance_status
            or result.evaluation_scope != self.evaluation_scope
            or result.official_runtime_verified is not False
        ):
            raise ValidationError(
                "STATE-Bench result provenance differs from its caller-asserted bridge"
            )
        return result


class StateBenchCheckpointAdapter:
    """Adapter over externally captured STATE-Bench pre-action checkpoints."""

    benchmark_id = STATEBENCH_BENCHMARK_ID
    adapter_metric_names = (
        "task_completion_pass",
        "state_requirements_met",
        "task_requirements_met",
        "ux_score",
        "cost_usd",
        "turns",
        "tool_calls",
    )
    split = STATEBENCH_EXTERNAL_SPLIT
    pairing_mode = "unsupported_external_replay"

    def __init__(
        self,
        *,
        checkpoints: Iterable[StateBenchCheckpoint],
        evaluation_bridge: StateBenchEvaluationBridge,
    ) -> None:
        if not isinstance(evaluation_bridge, StateBenchEvaluationBridge):
            raise TypeError(
                "evaluation_bridge must implement StateBenchEvaluationBridge; wrap callbacks "
                "with CallerAssertedStateBenchEvaluationBridge"
            )
        _validate_bridge(evaluation_bridge)
        normalized = tuple(checkpoints)
        if not normalized:
            raise ValidationError(
                "STATE-Bench adapter requires at least one captured visible checkpoint"
            )
        if not all(isinstance(item, StateBenchCheckpoint) for item in normalized):
            raise TypeError("STATE-Bench checkpoints must be StateBenchCheckpoint instances")
        checkpoint_ids = [item.checkpoint_id for item in normalized]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValidationError("STATE-Bench checkpoint IDs must be unique")
        experimental_units = [
            (item.domain, item.task_id, item.run_index, item.turn_index) for item in normalized
        ]
        if len(experimental_units) != len(set(experimental_units)):
            raise ValidationError(
                "STATE-Bench experimental units "
                "(domain, task_id, run_index, turn_index) must be unique"
            )

        examples: list[BenchmarkExample] = []
        checkpoints_by_id: dict[str, StateBenchCheckpoint] = {}
        for checkpoint in sorted(normalized, key=lambda item: item.checkpoint_id):
            _validate_checkpoint_replay_contract(checkpoint, evaluation_bridge)
            support = checkpoint.family_eligibility
            visible_metadata = _frozen_mapping(
                {
                    "benchmark_cell": self.benchmark_id,
                    "domain": checkpoint.domain,
                    "family_eligibility": dict(support.eligibility),
                    "family_eligibility_provenance": support.provenance_status,
                    "family_eligibility_reasons": dict(support.reasons),
                    "family_support_status": support.support_status,
                    "official_protocol_id": checkpoint.protocol_id,
                    "official_source_revision": checkpoint.source_revision,
                    "run_index": checkpoint.run_index,
                    "source_split": checkpoint.source_split,
                    "turn_index": checkpoint.turn_index,
                },
                label="STATE-Bench visible checkpoint metadata",
            )
            examples.append(
                BenchmarkExample(
                    example_id=checkpoint.checkpoint_id,
                    query=checkpoint.query,
                    instruction=checkpoint.instruction,
                    visible_context=checkpoint.visible_context,
                    visible_messages=checkpoint.visible_messages,
                    visible_tool_names=checkpoint.tool_names,
                    visible_metadata=visible_metadata,
                    evaluator_gold=MappingProxyType({}),
                    optimizer_visible_id=opaque_checkpoint_state_id(
                        self.benchmark_id,
                        checkpoint.checkpoint_id,
                        checkpoint.replay_identity,
                    ),
                )
            )
            checkpoints_by_id[checkpoint.checkpoint_id] = checkpoint

        self._evaluation_bridge = evaluation_bridge
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._checkpoints = MappingProxyType(checkpoints_by_id)

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(
                f"STATE-Bench adapter exposes split {self.split!r}, not {split!r}"
            )
        return self._examples

    def family_eligibility(self, checkpoint_id: str) -> StateBenchFamilyEligibility:
        return self._checkpoint(checkpoint_id).family_eligibility

    def eligible_families(self, checkpoint_id: str) -> tuple[SemanticFamily, ...]:
        return self.family_eligibility(checkpoint_id).eligible_families

    def supports_full_block(self, checkpoint_id: str) -> bool:
        return self.family_eligibility(checkpoint_id).supports_full_block

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this STATE-Bench adapter")
        checkpoint = self._checkpoints[example.example_id]
        result = self._evaluation_bridge.evaluate(checkpoint, prediction)
        if not isinstance(result, StateBenchEvaluationResult):
            raise ValidationError("STATE-Bench evaluation bridge returned an invalid result type")
        return result.metrics

    @property
    def evaluation_provenance(self) -> Mapping[str, Any]:
        return _frozen_mapping(
            {
                "bridge_id": self._evaluation_bridge.bridge_id,
                "bridge_revision": self._evaluation_bridge.bridge_revision,
                "cost_basis": self._evaluation_bridge.cost_basis,
                "evaluation_scope": self._evaluation_bridge.evaluation_scope,
                "official_runtime_verified": self._evaluation_bridge.official_runtime_verified,
                "preserves_domain_tools": self._evaluation_bridge.preserves_domain_tools,
                "preserves_scorer": self._evaluation_bridge.preserves_scorer,
                "preserves_turn_limit": self._evaluation_bridge.preserves_turn_limit,
                "preserves_user_simulator": (self._evaluation_bridge.preserves_user_simulator),
                "protocol_id": self._evaluation_bridge.protocol_id,
                "protocol_revision": self._evaluation_bridge.protocol_revision,
                "provenance_status": self._evaluation_bridge.provenance_status,
                "source_revision": self._evaluation_bridge.source_revision,
                "supports_fresh_task_reset": (self._evaluation_bridge.supports_fresh_task_reset),
                "supports_mid_dialogue_clone": (
                    self._evaluation_bridge.supports_mid_dialogue_clone
                ),
                "supports_simulator_replay": (self._evaluation_bridge.supports_simulator_replay),
            },
            label="STATE-Bench evaluation provenance",
        )

    def _checkpoint(self, checkpoint_id: str) -> StateBenchCheckpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise ValidationError(f"unknown STATE-Bench checkpoint {checkpoint_id!r}") from exc


def _validate_checkpoint_replay_contract(
    checkpoint: StateBenchCheckpoint,
    bridge: StateBenchEvaluationBridge,
) -> None:
    if not checkpoint.replay_ready:
        raise ValidationError(
            "STATE-Bench checkpoint lacks a continuation plus reset/clone contract"
        )
    if checkpoint.turn_index == 0 and not bridge.supports_fresh_task_reset:
        raise ValidationError(
            "STATE-Bench turn-zero checkpoint requires bridge fresh-task reset support"
        )
    if checkpoint.turn_index > 0 and not bridge.supports_mid_dialogue_clone:
        raise ValidationError("STATE-Bench mid-dialogue checkpoint requires bridge clone support")
    if (
        checkpoint.turn_index < STATEBENCH_OFFICIAL_MAX_AGENT_TURNS - 1
        and not bridge.supports_simulator_replay
    ):
        raise ValidationError(
            "STATE-Bench checkpoint with a remaining user turn requires bridge simulator replay"
        )


def _validate_bridge(bridge: StateBenchEvaluationBridge) -> None:
    _require_trimmed_string(bridge.bridge_id, "STATE-Bench evaluation bridge bridge_id")
    _require_sha256(
        bridge.bridge_revision,
        "STATE-Bench evaluation bridge bridge_revision",
    )
    if bridge.source_revision != STATEBENCH_OFFICIAL_SOURCE_REVISION:
        raise ValidationError("STATE-Bench evaluation bridge source revision differs from the pin")
    if bridge.protocol_id != STATEBENCH_OFFICIAL_PROTOCOL_ID:
        raise ValidationError("STATE-Bench evaluation bridge protocol ID differs from the pin")
    if bridge.protocol_revision != STATEBENCH_OFFICIAL_PROTOCOL_REVISION:
        raise ValidationError(
            "STATE-Bench evaluation bridge protocol revision differs from the pin"
        )
    if bridge.cost_basis != _OFFICIAL_COST_BASIS:
        raise ValidationError("STATE-Bench evaluation bridge cost basis must be agent_reported_usd")
    if bridge.provenance_status != _CALLER_ASSERTED:
        raise ValidationError("STATE-Bench bridge provenance must remain caller_asserted")
    if bridge.evaluation_scope != _EXTERNAL_BRIDGE:
        raise ValidationError(
            "STATE-Bench bridge evaluation scope must remain external_single_trajectory"
        )
    if bridge.official_runtime_verified is not False:
        raise ValidationError(
            "the external STATE-Bench bridge cannot claim official runtime verification"
        )
    for label in (
        "preserves_domain_tools",
        "preserves_user_simulator",
        "preserves_scorer",
        "preserves_turn_limit",
    ):
        if getattr(bridge, label) is not True:
            raise ValidationError(f"STATE-Bench evaluation bridge must set {label}=true")
    for label in (
        "supports_fresh_task_reset",
        "supports_mid_dialogue_clone",
        "supports_simulator_replay",
    ):
        if not isinstance(getattr(bridge, label), bool):
            raise ValidationError(f"STATE-Bench evaluation bridge {label} must be boolean")


def _validate_task_id(value: Any, *, domain: str, source_split: str) -> None:
    if not isinstance(value, str) or value != value.strip():
        raise ValidationError("STATE-Bench task_id must be a trimmed official-style ID")
    match = _TASK_ID.fullmatch(value)
    if match is None or not 1 <= int(match.group("number")) <= 150:
        raise ValidationError("STATE-Bench task_id must use an active numeric prefix in [1, 150]")
    if value not in STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT[source_split][domain]:
        raise ValidationError(
            f"STATE-Bench task_id is not a pinned v0.8.1 {source_split} task for domain {domain!r}"
        )


def _require_trimmed_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")


def _binary_metric(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise ValidationError(f"STATE-Bench {label} must be integer 0 or 1")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"STATE-Bench {label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError(f"STATE-Bench {label} must be finite")
    return numeric


def _bounded_integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"STATE-Bench {label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(f"STATE-Bench {label} must be an integer <= {maximum}")
    return value


def _frozen_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        return freeze_mapping(value)
    except TypeError as exc:
        raise ValidationError(f"{label} must be canonically serializable: {exc}") from exc
