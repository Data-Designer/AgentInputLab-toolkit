"""Deterministic, outcome-free measurement for paper inputs and trajectories.

The paper protocol uses these rules as the executable measurement path.  Human
annotation is deliberately an audit of the rules, not a dependency of model
execution.  Rules may inspect the visible pre-action state and public benchmark
metadata, but never evaluator gold, another arm's outcome, or post-treatment
success.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import ContextBlock, ExecutionBudget, FrozenAgentState, Message
from agentinputlab.selectors.features import VisibleStateFeatureExtractor

AUTOMATIC_INPUT_LABEL_PROTOCOL = "agentinputlab.automatic-input-label.v1"
AUTOMATIC_TRAJECTORY_PROTOCOL = "agentinputlab.automatic-trajectory-validity.v1"
AUTOMATIC_REPAIR_QUALITY_PROTOCOL = "agentinputlab.automatic-proximal-repair-quality.v1"
AUTOMATIC_MEASUREMENT_RULE_REVISION = "deterministic-visible-contract-v2"

_FAMILIES = ("reformulate", "decompose", "acquire", "clarify", "curate")
_DEFECT_FIELDS = (
    "unclear_or_misaligned_request",
    "decomposition_need",
    "missing_external_evidence",
    "clarification_need",
    "context_overload_or_noise",
)
_KNOWN_MAIN_ARENAS = frozenset(
    {
        "tau3.banking_knowledge",
        "state_bench.common_support",
        "toolsandbox.common_support",
    }
)
_THRESHOLDS = {
    "unclear_or_misaligned_request": 0.35,
    "decomposition_need": 0.30,
    "missing_external_evidence": 0.30,
    "clarification_need": 0.30,
    "context_overload_or_noise": 0.35,
}
_FAMILY_TARGET_DEFECT = {
    "reformulate": "unclear_or_misaligned_request",
    "decompose": "decomposition_need",
    "acquire": "missing_external_evidence",
    "clarify": "clarification_need",
    "curate": "context_overload_or_noise",
}
_NUMBER = re.compile(
    r"(?<![\w.])(?:[$£€¥][\u0020\u00a0]*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_IDENTIFIER = re.compile(
    r"\b(?=[A-Z0-9_-]{4,}\b)(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*\d)[A-Z0-9_-]+\b"
)
_QUOTED = re.compile(
    r"[\"'\u201c\u201d\u2018\u2019\u300c\u300d]"
    r"([^\"'\u201c\u201d\u2018\u2019\u300c\u300d]{2,80})"
    r"[\"'\u201c\u201d\u2018\u2019\u300c\u300d]"
)
_CONSTRAINT_TERMS = re.compile(
    r"\b(?:must|mustn't|must not|should|shouldn't|should not|only|without|before|after|"
    r"at least|at most|no more than|do not|don't|cannot|can't|required|unless)\b|"
    r"(?:必须|不得|不要|只能|仅限|至少|至多|之前|之后|除非)",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"[\n.!?\u3002\uff01\uff1f;\uff1b]+")


def frozen_state_from_visible(value: Mapping[str, Any]) -> FrozenAgentState:
    """Parse the dataset-neutral visible checkpoint schema into a frozen state."""

    if not isinstance(value, Mapping):
        raise ValidationError("visible state must be an object")
    context: list[ContextBlock] = []
    for raw in _sequence(value.get("context", ()), label="visible state context"):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("text"), str):
            raise ValidationError("visible state context block is invalid")
        context.append(
            ContextBlock(
                text=raw["text"],
                block_id=str(raw.get("block_id", "")),
                source=str(raw.get("source", "")),
                score=raw.get("score"),
                metadata=_mapping(raw.get("metadata", {}), label="context metadata"),
            )
        )
    messages: list[Message] = []
    for raw in _sequence(value.get("messages", ()), label="visible state messages"):
        if not isinstance(raw, Mapping):
            raise ValidationError("visible state message is invalid")
        messages.append(
            Message(
                role=str(raw.get("role", "")),
                content=str(raw.get("content", "")),
                name=(str(raw["name"]) if raw.get("name") is not None else None),
            )
        )
    raw_budget = _mapping(value.get("budget", {}), label="visible state budget")
    budget_fields = {
        name: raw_budget[name]
        for name in (
            "max_model_calls",
            "max_retrieval_calls",
            "max_tool_calls",
            "max_output_tokens",
            "max_context_tokens",
            "max_cost_usd",
        )
        if name in raw_budget
    }
    tools = _sequence(value.get("tool_names", ()), label="visible state tool_names")
    if any(not isinstance(item, str) or not item for item in tools):
        raise ValidationError("visible state tool_names are invalid")
    return FrozenAgentState(
        query=str(value.get("query", "")),
        task_id=str(value.get("task_id", "")),
        instruction=str(value.get("instruction", "")),
        context=tuple(context),
        messages=tuple(messages),
        tool_names=tuple(tools),
        metadata=_mapping(value.get("metadata", {}), label="visible state metadata"),
        budget=ExecutionBudget(**budget_fields),
    )


def automatic_input_label(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Assign deterministic defect and family-eligibility labels to one input."""

    visible = checkpoint.get("visible_state")
    if not isinstance(visible, Mapping):
        raise ValidationError("checkpoint requires visible_state")
    benchmark_id = _text(checkpoint.get("benchmark_id"), label="benchmark_id")
    group_id = _text(checkpoint.get("group_id"), label="group_id")
    checkpoint_id = _text(checkpoint.get("checkpoint_id"), label="checkpoint_id")
    state = frozen_state_from_visible(visible)
    observation, scores = _visible_observation_and_scores(state)
    defects = {name: scores[name] >= _THRESHOLDS[name] for name in _DEFECT_FIELDS}
    defects["low_defect_state"] = not any(defects.values())

    metadata = _mapping(visible.get("metadata", {}), label="visible state metadata")
    declared = metadata.get("eligible_families")
    if declared is None:
        declared = checkpoint.get("eligible_families")
    declared_set = (
        frozenset(str(value) for value in _sequence(declared, label="eligible_families"))
        if declared is not None
        else None
    )
    if declared_set is not None and not declared_set <= set(_FAMILIES):
        raise ValidationError("checkpoint eligible_families contains an unknown family")
    arena_common_support = benchmark_id in _KNOWN_MAIN_ARENAS or benchmark_id.startswith(
        ("tau3.", "state_bench.", "toolsandbox.")
    )
    eligibility = {
        f"{family}_eligible": (
            family in declared_set if declared_set is not None else arena_common_support
        )
        for family in _FAMILIES
    }
    if not arena_common_support and declared_set is None:
        # Conservative dataset-neutral fallback.  It never reads gold or outcomes.
        eligibility.update(
            {
                "reformulate_eligible": bool(state.query or state.instruction),
                "decompose_eligible": bool(state.query or state.instruction),
                "acquire_eligible": bool(state.tool_names or state.context),
                "clarify_eligible": bool(state.messages or metadata.get("interactive", False)),
                "curate_eligible": bool(state.context or state.messages),
            }
        )
    primary = "low_defect_state"
    if not defects["low_defect_state"]:
        primary = max(_DEFECT_FIELDS, key=lambda name: (scores[name], name))
    record = {
        "schema_version": 1,
        "protocol": AUTOMATIC_INPUT_LABEL_PROTOCOL,
        "rule_revision": AUTOMATIC_MEASUREMENT_RULE_REVISION,
        "benchmark_id": benchmark_id,
        "checkpoint_id": checkpoint_id,
        "group_id": group_id,
        "visible_state_hash": state.state_hash,
        "feature_schema_id": observation.provenance["schema_id"],
        "feature_schema_hash": observation.provenance["schema_hash"],
        "feature_hash": observation.feature_hash,
        "defect_scores": {name: round(float(value), 8) for name, value in scores.items()},
        "defect_labels": defects,
        "primary_stratum": primary,
        "eligibility_labels": eligibility,
        "evidence": {
            "benchmark_common_support": arena_common_support,
            "declared_eligible_families_used": declared_set is not None,
            "visible_signal_counts": observation.provenance["signals"],
            "outcome_fields_accessed": False,
            "gold_fields_accessed": False,
        },
        "human_audit_required_for_execution": False,
    }
    record["label_hash"] = content_hash(record)
    return record


def visible_defect_scores(visible_state: Mapping[str, Any]) -> dict[str, float]:
    """Return the frozen outcome-free defect scores for one visible state.

    This is the public bridge used by runtime intervention measurement.  It is
    deliberately the same implementation used by :func:`automatic_input_label`
    so a paper exporter cannot silently redefine a proximal target.
    """

    _observation, scores = _visible_observation_and_scores(frozen_state_from_visible(visible_state))
    return {name: round(float(value), 8) for name, value in scores.items()}


def automatic_proximal_repair_quality(
    *,
    pre_state: Mapping[str, Any],
    post_state: Mapping[str, Any],
    candidate_family: str,
    contract_retention: Mapping[str, float | bool],
    action_evidence: float,
) -> dict[str, Any]:
    """Measure target-defect repair without reading an arena outcome.

    The score is not an activation flag.  It combines (i) reduction of the
    family-matched visible defect, (ii) an explicit artifact signal only for
    decomposition/clarification, whose artifacts are not represented by the
    generic feature extractor, and (iii) the worst protected-contract
    retention component.  Contract loss is subtracted, so an aggressive edit
    cannot receive positive repair quality merely because it activated.

    The returned value lies in ``[-1, 1]`` and is a *proximal*, visible-state
    construct.  It never claims official task success or semantic correctness.
    """

    if candidate_family not in _FAMILY_TARGET_DEFECT:
        raise ValidationError("repair quality candidate_family is invalid")
    if isinstance(action_evidence, bool) or not isinstance(action_evidence, (int, float)):
        raise ValidationError("repair quality action_evidence must be numeric")
    evidence = float(action_evidence)
    if not 0.0 <= evidence <= 1.0:
        raise ValidationError("repair quality action_evidence must lie in [0, 1]")
    required = {
        "instruction_exactly_preserved",
        "entity_retention",
        "numeric_retention",
        "constraint_retention",
    }
    if set(contract_retention) != required:
        raise ValidationError("repair quality contract_retention fields are invalid")
    instruction = contract_retention["instruction_exactly_preserved"]
    if not isinstance(instruction, bool):
        raise ValidationError("instruction_exactly_preserved must be boolean")
    components = [float(instruction)]
    for name in ("entity_retention", "numeric_retention", "constraint_retention"):
        value = contract_retention[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"repair quality {name} must be numeric")
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValidationError(f"repair quality {name} must lie in [0, 1]")
        components.append(number)

    before_scores = visible_defect_scores(pre_state)
    after_scores = visible_defect_scores(post_state)
    target = _FAMILY_TARGET_DEFECT[candidate_family]
    raw_reduction = before_scores[target] - after_scores[target]
    # A decomposition plan and a clarification request are intermediate
    # artifacts rather than a state in which the underlying need disappears.
    # For those two families only, credit the artifact in proportion to the
    # pre-action target need.  Merely editing/adding context in the other
    # families receives no such fallback credit.
    artifact_credit = (
        before_scores[target] * evidence if candidate_family in {"decompose", "clarify"} else 0.0
    )
    target_repair = min(1.0, max(0.0, raw_reduction, artifact_credit))
    retention_floor = min(components)
    contract_loss = 1.0 - retention_floor
    score = max(-1.0, min(1.0, target_repair * retention_floor - contract_loss))
    result = {
        "schema_version": 1,
        "protocol": AUTOMATIC_REPAIR_QUALITY_PROTOCOL,
        "rule_revision": AUTOMATIC_MEASUREMENT_RULE_REVISION,
        "claim_scope": "outcome_blind_visible_state_proximal",
        "candidate_family": candidate_family,
        "target_defect": target,
        "target_need_before": round(before_scores[target], 8),
        "target_need_after": round(after_scores[target], 8),
        "raw_target_defect_reduction": round(raw_reduction, 8),
        "artifact_credit": round(artifact_credit, 8),
        "action_evidence": round(evidence, 8),
        "contract_retention_floor": round(retention_floor, 8),
        "contract_loss": round(contract_loss, 8),
        "repair_quality": round(score, 8),
        "repair_success": score > 0.0,
        "outcome_fields_accessed": False,
        "gold_fields_accessed": False,
    }
    result["measurement_hash"] = content_hash(result)
    return result


def _visible_observation_and_scores(
    state: FrozenAgentState,
) -> tuple[Any, dict[str, float]]:
    observation = VisibleStateFeatureExtractor().extract(state)
    feature = observation.features
    return observation, {
        "unclear_or_misaligned_request": max(
            feature.reformulation_need_score,
            0.5 * feature.ambiguity_score,
        ),
        "decomposition_need": feature.multi_step_score,
        "missing_external_evidence": feature.retrieval_need_score,
        "clarification_need": feature.ambiguity_score,
        "context_overload_or_noise": feature.context_pressure,
    }


def infer_visible_contract(visible_state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract auditable entity, numeric, and constraint anchors from visible input."""

    state = frozen_state_from_visible(visible_state)
    # API catalogs describe tools, not the user's task.  Treating JSON schema
    # strings such as ``required`` or example IDs as intent anchors would make
    # legal context curation look like task drift.
    text = _contract_text(visible_state)
    metadata = _mapping(visible_state.get("metadata", {}), label="visible state metadata")
    explicit = metadata.get("intent_contract", {})
    if explicit is None:
        explicit = {}
    explicit = _mapping(explicit, label="visible intent_contract")
    entities = set(_strings(explicit.get("required_entities", ()), label="required_entities"))
    entities.update(_EMAIL.findall(text))
    entities.update(_IDENTIFIER.findall(text))
    entities.update(match.group(1).strip() for match in _QUOTED.finditer(text))
    constraints = set(
        _strings(explicit.get("required_constraints", ()), label="required_constraints")
    )
    constraints.update(
        clause.strip()
        for clause in _CLAUSE_SPLIT.split(text)
        if clause.strip() and _CONSTRAINT_TERMS.search(clause)
    )
    forbidden = set(
        _strings(explicit.get("forbidden_assumptions", ()), label="forbidden_assumptions")
    )
    numbers = {_normalize_number(value) for value in _NUMBER.findall(text)}
    contract = {
        "required_entities": sorted(entities, key=str.casefold),
        "required_numbers": sorted(numbers),
        "required_constraints": sorted(constraints, key=str.casefold),
        "forbidden_assumptions": sorted(forbidden, key=str.casefold),
        "may_request_clarification": bool(explicit.get("may_request_clarification", True)),
        "source_state_hash": state.state_hash,
        "extraction_revision": AUTOMATIC_MEASUREMENT_RULE_REVISION,
    }
    contract["contract_hash"] = content_hash(contract)
    return contract


def visible_contract_retention(
    *,
    pre_state: Mapping[str, Any],
    post_state: Mapping[str, Any],
) -> dict[str, float | bool | int | str]:
    """Measure protected task-anchor retention without reading gold or outcomes."""

    contract = infer_visible_contract(pre_state)
    output_text = _visible_text(post_state)
    output_numbers = {_normalize_number(value) for value in _NUMBER.findall(output_text)}

    def phrase_retention(values: Sequence[str]) -> float:
        if not values:
            return 1.0
        return sum(_contains_phrase(output_text, value) for value in values) / len(values)

    entities = tuple(contract["required_entities"])
    numbers = tuple(contract["required_numbers"])
    constraints = tuple(contract["required_constraints"])
    forbidden = tuple(contract["forbidden_assumptions"])
    introduced_forbidden = sum(_contains_phrase(output_text, value) for value in forbidden)
    constraint_retention = phrase_retention(constraints)
    if introduced_forbidden:
        constraint_retention = 0.0
    return {
        "instruction_exactly_preserved": str(pre_state.get("instruction", ""))
        == str(post_state.get("instruction", "")),
        "entity_retention": phrase_retention(entities),
        "numeric_retention": (
            1.0 if not numbers else sum(value in output_numbers for value in numbers) / len(numbers)
        ),
        "constraint_retention": constraint_retention,
        "required_entity_count": len(entities),
        "required_number_count": len(numbers),
        "required_constraint_count": len(constraints),
        "introduced_forbidden_assumption_count": introduced_forbidden,
        "protected_contract_hash": str(contract["contract_hash"]),
        "extraction_revision": AUTOMATIC_MEASUREMENT_RULE_REVISION,
    }


def automatic_trajectory_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Judge one observed trajectory with four deterministic contract checks."""

    for name in (
        "slot_id",
        "group_id",
        "benchmark_id",
        "checkpoint_id",
        "candidate_family",
    ):
        _text(record.get(name), label=f"trajectory {name}")
    if record["candidate_family"] not in _FAMILIES:
        raise ValidationError("trajectory candidate_family is invalid")
    pre = record.get("pre_state")
    if not isinstance(pre, Mapping):
        raise ValidationError("trajectory requires pre_state")
    raw_checkpoints = record.get("trajectory_checkpoints")
    if raw_checkpoints is None and isinstance(record.get("post_state"), Mapping):
        raw_checkpoints = [{"checkpoint_id": "final_state", "state": record["post_state"]}]
    checkpoints = _sequence(raw_checkpoints, label="trajectory_checkpoints")
    if not checkpoints:
        raise ValidationError("trajectory requires at least one observed checkpoint")
    contract = record.get("protected_contract")
    if contract is None:
        contract = infer_visible_contract(pre)
    contract = _validated_contract(contract)
    results = []
    for raw in checkpoints:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("state"), Mapping):
            raise ValidationError("trajectory checkpoint is invalid")
        checkpoint_name = _text(raw.get("checkpoint_id"), label="trajectory checkpoint_id")
        output_text = _visible_text(raw["state"])
        missing_entities = [
            value
            for value in contract["required_entities"]
            if not _contains_phrase(output_text, value)
        ]
        output_numbers = {_normalize_number(value) for value in _NUMBER.findall(output_text)}
        missing_numbers = [
            value for value in contract["required_numbers"] if value not in output_numbers
        ]
        missing_constraints = [
            value
            for value in contract["required_constraints"]
            if not _contains_phrase(output_text, value)
        ]
        introduced = [
            value
            for value in contract["forbidden_assumptions"]
            if _contains_phrase(output_text, value)
        ]
        contract_diff = {
            "missing_entities": missing_entities,
            "missing_numbers": missing_numbers,
            "missing_constraints": missing_constraints,
            "introduced_forbidden_assumptions": introduced,
        }
        checks = {
            "contract_diff": not any(contract_diff.values()),
            "entity_preservation": not missing_entities,
            "numeric_preservation": not missing_numbers,
            "constraint_preservation": not missing_constraints and not introduced,
        }
        results.append(
            {
                "checkpoint_id": checkpoint_name,
                "checks": checks,
                "diff": contract_diff,
                "valid": all(checks.values()),
            }
        )
    result = {
        "schema_version": 1,
        "protocol": AUTOMATIC_TRAJECTORY_PROTOCOL,
        "rule_revision": AUTOMATIC_MEASUREMENT_RULE_REVISION,
        "slot_id": record["slot_id"],
        "group_id": record["group_id"],
        "benchmark_id": record["benchmark_id"],
        "checkpoint_id": record["checkpoint_id"],
        "candidate_family": record["candidate_family"],
        "protected_contract": contract,
        "checkpoint_results": results,
        "auto_contract_valid": all(row["valid"] for row in results),
        "validity_stratum": (
            "valid" if all(row["valid"] for row in results) else "contract_violation"
        ),
        "contract_anchor_count": sum(
            len(contract[name])
            for name in (
                "required_entities",
                "required_numbers",
                "required_constraints",
                "forbidden_assumptions",
            )
        ),
        "human_audit_required_for_execution": False,
    }
    result["measurement_hash"] = content_hash(result)
    return result


def deterministic_stratified_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    strata: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Select an exact, reproducible, proportionally stratified audit sample."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("audit sample count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationError("audit sample seed must be a non-negative integer")
    if not strata:
        raise ValidationError("audit sample requires strata fields")
    values = list(records)
    if count > len(values):
        raise ValidationError("audit sample count exceeds the record census")
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in values:
        key = tuple(_text(row.get(field), label=f"audit stratum {field}") for field in strata)
        groups[key].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda row: _stable_order(seed, key, row))
    allocations = {key: 0 for key in groups}
    if count >= len(groups):
        for key in groups:
            allocations[key] = 1
    remaining = count - sum(allocations.values())
    population = len(values)
    while remaining:
        candidates = [key for key, rows in groups.items() if allocations[key] < len(rows)]
        if not candidates:  # pragma: no cover - count/census invariant
            raise AssertionError("audit allocation exhausted before reaching target")
        key = max(
            candidates,
            key=lambda item: (
                len(groups[item]) * count / population - allocations[item],
                -int(hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(), 16),
            ),
        )
        allocations[key] += 1
        remaining -= 1
    selected = [row for key, rows in groups.items() for row in rows[: allocations[key]]]
    return sorted(selected, key=lambda row: _stable_order(seed, ("final",), row))


def audit_summary(records: Sequence[Mapping[str, Any]], *, strata: Sequence[str]) -> dict[str, Any]:
    counts = Counter(tuple(str(row[field]) for field in strata) for row in records)
    return {"|".join(key): value for key, value in sorted(counts.items())}


def _validated_contract(value: Any) -> dict[str, Any]:
    raw = _mapping(value, label="protected_contract")
    required = {
        "required_entities",
        "required_numbers",
        "required_constraints",
        "forbidden_assumptions",
    }
    contract = {name: list(_strings(raw.get(name, ()), label=name)) for name in required}
    contract.update(
        {
            "may_request_clarification": bool(raw.get("may_request_clarification", True)),
            "source_state_hash": str(raw.get("source_state_hash", "")),
            "extraction_revision": str(
                raw.get("extraction_revision", AUTOMATIC_MEASUREMENT_RULE_REVISION)
            ),
        }
    )
    expected_hash = raw.get("contract_hash")
    contract["contract_hash"] = content_hash(contract)
    if expected_hash is not None and expected_hash != contract["contract_hash"]:
        raise ValidationError("protected_contract hash mismatch")
    return contract


def _visible_text(value: Mapping[str, Any]) -> str:
    parts = [str(value.get("instruction", "")), str(value.get("query", ""))]
    for block in _sequence(value.get("context", ()), label="state context"):
        if isinstance(block, Mapping):
            parts.append(str(block.get("text", "")))
    for message in _sequence(value.get("messages", ()), label="state messages"):
        if isinstance(message, Mapping):
            parts.append(str(message.get("content", "")))
    return "\n".join(part for part in parts if part)


def _contract_text(value: Mapping[str, Any]) -> str:
    """Return task-bearing visible text while excluding structural tool catalogs."""

    parts = [str(value.get("instruction", "")), str(value.get("query", ""))]
    for block in _sequence(value.get("context", ()), label="state context"):
        if not isinstance(block, Mapping):
            continue
        block_id = str(block.get("block_id", "")).casefold()
        source = str(block.get("source", "")).casefold()
        if block_id == "official_tool_catalog" or any(
            marker in source
            for marker in ("tool_schema", "tool_catalog", ".tools", "starting_context")
        ):
            continue
        parts.append(str(block.get("text", "")))
    for message in _sequence(value.get("messages", ()), label="state messages"):
        if isinstance(message, Mapping):
            parts.append(str(message.get("content", "")))
    return "\n".join(part for part in parts if part)


def _normalize_number(value: str) -> str:
    return value.replace(",", "").replace("\u00a0", "").strip().casefold()


def _contains_phrase(text: str, phrase: str) -> bool:
    tokens = tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
    phrase_tokens = tuple(re.findall(r"\w+", phrase.casefold(), flags=re.UNICODE))
    width = len(phrase_tokens)
    return bool(width) and any(
        tokens[index : index + width] == phrase_tokens for index in range(len(tokens) - width + 1)
    )


def _stable_order(seed: int, stratum: tuple[str, ...], row: Mapping[str, Any]) -> str:
    identity = row.get("label_hash", row.get("measurement_hash", row.get("group_id")))
    return hashlib.sha256(f"{seed}:{stratum}:{identity}".encode()).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    return value


def _strings(value: Any, *, label: str) -> tuple[str, ...]:
    values = _sequence(value, label=label)
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValidationError(f"{label} must contain non-empty strings")
    return tuple(item.strip() for item in values)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value.strip()


__all__ = [
    "AUTOMATIC_INPUT_LABEL_PROTOCOL",
    "AUTOMATIC_MEASUREMENT_RULE_REVISION",
    "AUTOMATIC_REPAIR_QUALITY_PROTOCOL",
    "AUTOMATIC_TRAJECTORY_PROTOCOL",
    "audit_summary",
    "automatic_input_label",
    "automatic_proximal_repair_quality",
    "automatic_trajectory_validity",
    "deterministic_stratified_sample",
    "frozen_state_from_visible",
    "infer_visible_contract",
    "visible_contract_retention",
    "visible_defect_scores",
]
