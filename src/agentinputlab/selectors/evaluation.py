"""Outcome-blind selector decisions and verified counterfactual result assembly.

This module intentionally separates two capabilities:

* :func:`build_selector_evaluation_input` exports only authenticated pre-action
  feature observations, eligibility masks, and content identities.
* :func:`assemble_selector_campaign_results` is the only function here that
  reads evaluator outcomes, and it accepts an already frozen decision artifact.

Consequently a selector policy never receives test outcomes through this API.
The diagnostic observed-action oracle is created only in the second phase and
is rejected from the outcome-blind decision artifact.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.experiments import (
    ExperimentManifest,
    ExperimentStage,
    load_experiment_manifest,
    load_verified_cell_results,
    verify_experiment_run_directory,
)

from .artifacts import SelectorArtifactBundle, SelectorArtifactStage
from .campaigns import (
    validate_representative_manifest,
    validate_selector_campaign_design,
    validate_selector_campaign_manifest,
    validate_selector_campaign_results,
)
from .discovery import (
    _convert_result_record,
    _observation_from_artifact,
    _observation_to_artifact,
    _state_from_artifact,
    _validate_pre_action_receipt,
)
from .features import VisibleStateFeatureExtractor
from .outcome_table import EvaluatedArmOutcome, UtilityDefinition

_INPUT_ARTIFACT = "agentinputlab.selector_evaluation_input"
_DECISION_ARTIFACT = "agentinputlab.selector_campaign_decisions"
_ZERO_SHOT_ARTIFACT = "agentinputlab.selector_zero_shot_replay"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CORE_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)
_POLICY_ARTIFACT_IDS = {
    "defect_rule",
    "zero_shot_router",
    "penalized_t_learner",
    "gbdt_t_learner",
    "eligibility_aware_selector",
}
_STANDARD_SELECTOR_KINDS = {
    "rule",
    "ridge_utility",
    "ridge_t_learner",
    "logistic_t_learner",
    "gbdt_t_learner",
    "knn_utility",
    "cost_risk_guardrail",
}


def build_selector_evaluation_input(
    run_dirs: Sequence[str | Path],
    *,
    design: Mapping[str, Any],
    representatives: Mapping[str, Any],
    extractor: VisibleStateFeatureExtractor,
    expected_partial_plan_hash: str,
) -> Mapping[str, Any]:
    """Export the frozen outcome-blind selector test input.

    The formal paper design is 526 three-arena common-support groups. The
    older 500-common + 100-partial layout remains accepted for non-paper
    library studies.
    """

    roots = tuple(Path(item).expanduser().resolve() for item in run_dirs)
    if not roots:
        raise ValidationError("selector evaluation input requires verified run directories")
    parsed_design = validate_selector_campaign_design(design)
    frozen_representatives = validate_representative_manifest(representatives)
    _require_sha256(expected_partial_plan_hash, label="partial-support plan hash")
    if not isinstance(extractor, VisibleStateFeatureExtractor):
        raise TypeError("selector evaluation requires VisibleStateFeatureExtractor")
    rank_one = _rank_one_releases(frozen_representatives)

    source_runs: list[dict[str, str]] = []
    groups: dict[str, dict[str, Any]] = {}
    seen_cells: set[tuple[str, str]] = set()
    common_run_count = 0
    partial_run_count = 0
    for root in roots:
        manifest, records, source = _verified_source(root)
        common_support = _validate_evaluation_manifest(
            manifest,
            representative_manifest_hash=frozen_representatives["manifest_hash"],
            rank_one=rank_one,
            expected_partial_plan_hash=expected_partial_plan_hash,
        )
        common_run_count += int(common_support)
        partial_run_count += int(not common_support)
        source_runs.append(source)
        arm_by_id = {arm.arm_id: arm for arm in manifest.arms}
        for record in records:
            pre_action = _pre_action_record(
                record,
                arm_by_id=arm_by_id,
                extractor=extractor,
                common_support=common_support,
            )
            group_id = pre_action["group_id"]
            cell_key = (group_id, pre_action["model_slot"])
            if cell_key in seen_cells:
                raise ValidationError("selector evaluation contains duplicate group/model cells")
            seen_cells.add(cell_key)
            existing = groups.get(group_id)
            if existing is None:
                groups[group_id] = {
                    "group_id": group_id,
                    "suite_id": pre_action["suite_id"],
                    "common_support": common_support,
                    "state_hash": pre_action["state_hash"],
                    "observation": pre_action["observation"],
                    "eligible_families": pre_action["eligible_families"],
                    "model_slots": [pre_action["model_slot"]],
                    "seed": pre_action["seed"],
                }
                continue
            comparable = {
                key: existing[key]
                for key in (
                    "suite_id",
                    "common_support",
                    "state_hash",
                    "observation",
                    "eligible_families",
                    "seed",
                )
            }
            if comparable != {
                "suite_id": pre_action["suite_id"],
                "common_support": common_support,
                "state_hash": pre_action["state_hash"],
                "observation": pre_action["observation"],
                "eligible_families": pre_action["eligible_families"],
                "seed": pre_action["seed"],
            }:
                raise ValidationError(
                    "selector group changed pre-action state, features, eligibility, or suite"
                )
            existing["model_slots"].append(pre_action["model_slot"])

    expected_groups = parsed_design["required_independent_groups"]
    expected_common = parsed_design["minimum_common_support_groups"]
    common_groups = sum(bool(group["common_support"]) for group in groups.values())
    if len(groups) != expected_groups or common_groups != expected_common:
        raise ValidationError(
            "selector evaluation support differs from the frozen design "
            f"({len(groups)} total/{common_groups} common)"
        )
    if set(group["suite_id"] for group in groups.values()) != set(parsed_design["suite_ids"]):
        raise ValidationError("selector evaluation suites differ from the frozen design")
    expected_model_slots = ["anchor_a", "anchor_b"]
    for group in groups.values():
        group["model_slots"] = sorted(group["model_slots"])
        if (
            group["model_slots"] != expected_model_slots
            or group["seed"] != 17
            or len(group["eligible_families"]) != (5 if group["common_support"] else 3)
        ):
            raise ValidationError(
                "selector evaluation group differs from the two-model, one-seed "
                "common/partial support protocol"
            )
    if expected_groups == 600 and expected_common == 500:
        if common_run_count != 1 or partial_run_count != 10:
            raise ValidationError(
                "600-group selector evaluation requires one common and ten partial runs"
            )
        partial_groups = expected_groups - expected_common
        if partial_groups != 100:
            raise ValidationError("selector partial-support block must contain 100 groups")

    design_payload = _jsonable_mapping(parsed_design)
    source_payload = sorted(source_runs, key=lambda item: item["run_identity_hash"])
    group_payload = [groups[group_id] for group_id in sorted(groups)]
    base = {
        "schema_version": 1,
        "artifact": _INPUT_ARTIFACT,
        "stage": "frozen",
        "data_status": "VERIFIED_PRE_ACTION_INPUTS",
        "is_synthetic": False,
        "campaign_design": design_payload,
        "campaign_design_hash": content_hash(design_payload),
        "representative_manifest_hash": frozen_representatives["manifest_hash"],
        "partial_support_plan_hash": expected_partial_plan_hash,
        "source_runs": source_payload,
        "extractor": {
            "id": extractor.extractor_id,
            "revision": extractor.extractor_revision,
            "sha256": extractor.extractor_hash,
            "feature_schema_id": extractor.feature_schema_id,
            "feature_schema_sha256": extractor.feature_schema_hash,
        },
        "groups": group_payload,
        "outcomes_included": False,
        "other_arm_outcomes_accessible": False,
    }
    artifact = {**base, "input_hash": content_hash(base)}
    return validate_selector_evaluation_input(artifact)


def validate_selector_evaluation_input(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate and freeze an outcome-free selector evaluation input."""

    required = {
        "schema_version",
        "artifact",
        "stage",
        "data_status",
        "is_synthetic",
        "campaign_design",
        "campaign_design_hash",
        "representative_manifest_hash",
        "partial_support_plan_hash",
        "source_runs",
        "extractor",
        "groups",
        "outcomes_included",
        "other_arm_outcomes_accessible",
        "input_hash",
    }
    data = _strict_mapping(value, required, label="selector evaluation input")
    if (
        data["schema_version"] != 1
        or data["artifact"] != _INPUT_ARTIFACT
        or data["stage"] != "frozen"
        or data["data_status"] != "VERIFIED_PRE_ACTION_INPUTS"
        or data["is_synthetic"] is not False
        or data["outcomes_included"] is not False
        or data["other_arm_outcomes_accessible"] is not False
    ):
        raise ValidationError("selector evaluation input identity or leakage boundary is invalid")
    design = validate_selector_campaign_design(
        _mapping(data["campaign_design"], label="campaign_design")
    )
    design_payload = _jsonable_mapping(design)
    if content_hash(design_payload) != data["campaign_design_hash"]:
        raise ValidationError("selector evaluation campaign design hash mismatch")
    for name in (
        "campaign_design_hash",
        "representative_manifest_hash",
        "partial_support_plan_hash",
        "input_hash",
    ):
        _require_sha256(data[name], label=name)
    extractor = _strict_mapping(
        data["extractor"],
        {
            "id",
            "revision",
            "sha256",
            "feature_schema_id",
            "feature_schema_sha256",
        },
        label="selector evaluation extractor",
    )
    for name in ("id", "revision", "feature_schema_id"):
        _text(extractor[name], label=f"extractor.{name}")
    for name in ("sha256", "feature_schema_sha256"):
        _require_sha256(extractor[name], label=f"extractor.{name}")
    sources = _validate_sources(data["source_runs"])
    if not sources:
        raise ValidationError("selector evaluation input requires verified source runs")
    groups_raw = _sequence(data["groups"], label="selector evaluation groups")
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    common_count = 0
    suites: set[str] = set()
    for index, raw in enumerate(groups_raw):
        group = _strict_mapping(
            raw,
            {
                "group_id",
                "suite_id",
                "common_support",
                "state_hash",
                "observation",
                "eligible_families",
                "model_slots",
                "seed",
            },
            label=f"selector evaluation group {index}",
        )
        group_id = _group_id(group["group_id"], label="selector evaluation group ID")
        if group_id in seen:
            raise ValidationError("selector evaluation group IDs must be unique")
        seen.add(group_id)
        suite_id = _text(group["suite_id"], label="selector evaluation suite")
        if suite_id not in design["suite_ids"]:
            raise ValidationError("selector evaluation group suite is undeclared")
        suites.add(suite_id)
        common_support = _boolean(
            group["common_support"], label="selector evaluation common_support"
        )
        common_count += int(common_support)
        _require_sha256(group["state_hash"], label="selector evaluation state_hash")
        observation = _observation_from_artifact(to_jsonable(group["observation"]))
        if observation.state_hash != group["state_hash"]:
            raise ValidationError("selector evaluation observation is bound to another state")
        if (
            observation.extractor_id != extractor["id"]
            or observation.extractor_revision != extractor["revision"]
        ):
            raise ValidationError("selector evaluation observation changed extractor identity")
        families = _family_sequence(group["eligible_families"])
        if len(families) != (5 if common_support else 3):
            raise ValidationError("selector evaluation eligibility cardinality is invalid")
        model_slots = tuple(
            _text(item, label="selector evaluation model slot")
            for item in _sequence(group["model_slots"], label="selector model slots")
        )
        if model_slots != ("anchor_a", "anchor_b"):
            raise ValidationError("selector evaluation requires exact anchor_a/anchor_b slots")
        if group["seed"] != 17:
            raise ValidationError("selector evaluation requires continuation seed 17")
        groups.append(
            {
                **group,
                "observation": _observation_to_artifact(observation),
                "eligible_families": [family.value for family in families],
                "model_slots": list(model_slots),
            }
        )
    if (
        len(groups) != design["required_independent_groups"]
        or common_count != design["minimum_common_support_groups"]
        or suites != set(design["suite_ids"])
    ):
        raise ValidationError("selector evaluation group support differs from campaign design")
    if (
        content_hash({key: item for key, item in data.items() if key != "input_hash"})
        != data["input_hash"]
    ):
        raise ValidationError("selector evaluation input hash differs from its content")
    return freeze_mapping(
        {
            **data,
            "campaign_design": design_payload,
            "source_runs": sources,
            "groups": tuple(groups),
        }
    )


def freeze_zero_shot_router_replay(
    evaluation_input: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    selector_id: str,
    revision: str,
    router_prompt_sha256: str,
    backend_sha256: str,
    routing_run_sha256: str,
) -> Mapping[str, Any]:
    """Freeze externally generated, outcome-blind zero-shot routing choices."""

    inputs = validate_selector_evaluation_input(evaluation_input)
    for name, digest in (
        ("router prompt", router_prompt_sha256),
        ("backend", backend_sha256),
        ("routing run", routing_run_sha256),
    ):
        _require_sha256(digest, label=f"zero-shot {name}")
    _text(selector_id, label="zero-shot selector ID")
    _text(revision, label="zero-shot selector revision")
    groups = {group["group_id"]: group for group in inputs["groups"]}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(decisions):
        row = _strict_mapping(
            raw,
            {"group_id", "selected_family", "selected_release_id"},
            label=f"zero-shot decision {index}",
        )
        group_id = _group_id(row["group_id"], label="zero-shot group ID")
        if group_id in seen or group_id not in groups:
            raise ValidationError("zero-shot decisions have duplicate or unknown groups")
        seen.add(group_id)
        family = _family(row["selected_family"])
        release_id = _text(row["selected_release_id"], label="zero-shot release")
        _validate_selected_action(group=groups[group_id], family=family, release_id=release_id)
        rows.append(
            {
                "group_id": group_id,
                "selected_family": family.value,
                "selected_release_id": release_id,
            }
        )
    if seen != set(groups):
        raise ValidationError("zero-shot replay must contain one choice for every input group")
    base = {
        "schema_version": 1,
        "artifact": _ZERO_SHOT_ARTIFACT,
        "stage": "frozen",
        "selector_kind": "zero_shot_router",
        "selector_id": selector_id,
        "revision": revision,
        "evaluation_input_hash": inputs["input_hash"],
        "attestation": {
            "router_prompt_sha256": router_prompt_sha256,
            "backend_sha256": backend_sha256,
            "routing_run_sha256": routing_run_sha256,
            "outcomes_accessed": False,
            "other_arm_outcomes_accessed": False,
        },
        "decisions": sorted(rows, key=lambda item: item["group_id"]),
    }
    return freeze_mapping({**base, "bundle_hash": content_hash(base)})


def validate_zero_shot_router_replay(
    evaluation_input: Mapping[str, Any],
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    inputs = validate_selector_evaluation_input(evaluation_input)
    required = {
        "schema_version",
        "artifact",
        "stage",
        "selector_kind",
        "selector_id",
        "revision",
        "evaluation_input_hash",
        "attestation",
        "decisions",
        "bundle_hash",
    }
    data = _strict_mapping(value, required, label="zero-shot replay")
    if (
        data["schema_version"] != 1
        or data["artifact"] != _ZERO_SHOT_ARTIFACT
        or data["stage"] != "frozen"
        or data["selector_kind"] != "zero_shot_router"
        or data["evaluation_input_hash"] != inputs["input_hash"]
    ):
        raise ValidationError("zero-shot replay identity differs from its evaluation input")
    _text(data["selector_id"], label="zero-shot selector ID")
    _text(data["revision"], label="zero-shot selector revision")
    attestation = _strict_mapping(
        data["attestation"],
        {
            "router_prompt_sha256",
            "backend_sha256",
            "routing_run_sha256",
            "outcomes_accessed",
            "other_arm_outcomes_accessed",
        },
        label="zero-shot attestation",
    )
    for name in ("router_prompt_sha256", "backend_sha256", "routing_run_sha256"):
        _require_sha256(attestation[name], label=f"zero-shot {name}")
    if (
        attestation["outcomes_accessed"] is not False
        or attestation["other_arm_outcomes_accessed"] is not False
    ):
        raise ValidationError("zero-shot replay may not access test outcomes")
    rebuilt = freeze_zero_shot_router_replay(
        inputs,
        _sequence(data["decisions"], label="zero-shot decisions"),
        selector_id=data["selector_id"],
        revision=data["revision"],
        router_prompt_sha256=attestation["router_prompt_sha256"],
        backend_sha256=attestation["backend_sha256"],
        routing_run_sha256=attestation["routing_run_sha256"],
    )
    if rebuilt["bundle_hash"] != data["bundle_hash"]:
        raise ValidationError("zero-shot replay hash differs from its content")
    return freeze_mapping(data)


def build_selector_campaign_decisions(
    campaign: Mapping[str, Any],
    evaluation_input: Mapping[str, Any],
    policy_artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Run 13 outcome-blind policies; the diagnostic oracle is deliberately absent."""

    manifest = validate_selector_campaign_manifest(campaign)
    inputs = validate_selector_evaluation_input(evaluation_input)
    _validate_campaign_input_binding(manifest, inputs)
    if set(policy_artifacts) != _POLICY_ARTIFACT_IDS:
        raise ValidationError(
            "selector decision builder requires exact artifacts for five frozen policies"
        )
    policies = {policy["policy_id"]: policy for policy in manifest["policies"]}
    rank_one = _campaign_rank_one_releases(manifest)
    loaded: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    for policy_id in sorted(_POLICY_ARTIFACT_IDS):
        policy = policies[policy_id]
        binding = policy["parameters"]["artifact_binding"]
        artifact = _mapping(
            policy_artifacts[policy_id],
            label=f"selector artifact {policy_id}",
        )
        if binding["selector_kind"] == "zero_shot_router":
            replay = validate_zero_shot_router_replay(inputs, artifact)
            if (
                replay["selector_id"] != binding["selector_id"]
                or replay["revision"] != binding["revision"]
                or replay["bundle_hash"] != binding["sha256"]
            ):
                raise ValidationError("zero-shot replay differs from its campaign binding")
            loaded[policy_id] = {row["group_id"]: row for row in replay["decisions"]}
            artifact_hashes[policy_id] = replay["bundle_hash"]
            continue
        bundle = SelectorArtifactBundle.from_artifact(artifact)
        selector_kind = bundle.model_payload.get("selector_kind")
        if (
            bundle.stage is not SelectorArtifactStage.FROZEN
            or selector_kind != binding["selector_kind"]
            or selector_kind not in _STANDARD_SELECTOR_KINDS
            or bundle.selector_card_id != binding["selector_id"]
            or bundle.revision != binding["revision"]
            or bundle.bundle_hash != binding["sha256"]
        ):
            raise ValidationError(f"selector artifact {policy_id} differs from its binding")
        input_extractor = inputs["extractor"]
        if (
            bundle.extractor_id != input_extractor["id"]
            or bundle.extractor_revision != input_extractor["revision"]
            or bundle.extractor_hash != input_extractor["sha256"]
            or bundle.feature_schema_id != input_extractor["feature_schema_id"]
            or bundle.feature_schema_hash != input_extractor["feature_schema_sha256"]
            or dict(bundle.representative_by_family) != rank_one
        ):
            raise ValidationError(
                f"selector artifact {policy_id} changed features or representatives"
            )
        loaded[policy_id] = bundle.selector()
        artifact_hashes[policy_id] = bundle.bundle_hash

    rows: list[dict[str, Any]] = []
    oracle_policy_ids = {
        policy_id
        for policy_id, policy in policies.items()
        if policy["kind"] == "observed_action_oracle"
    }
    if oracle_policy_ids != {"observed_action_oracle"}:
        raise ValidationError("selector campaign diagnostic oracle contract is invalid")
    for group in inputs["groups"]:
        group_id = group["group_id"]
        eligible = tuple(SemanticFamily(item) for item in group["eligible_families"])
        observation = _observation_from_artifact(to_jsonable(group["observation"]))
        state = FrozenAgentState(
            query="[outcome-blind selector feature evaluation]",
            task_id=group_id,
        )
        for policy_id, policy in policies.items():
            if policy_id in oracle_policy_ids:
                continue
            family, release_id = _select_policy_action(
                policy,
                group=group,
                state=state,
                features=observation.features,
                eligible=eligible,
                rank_one=rank_one,
                loaded=loaded,
                random_seed=manifest["metadata"]["random_seed"],
            )
            expected_release = _release_for(family, rank_one)[1]
            if release_id != expected_release:
                raise ValidationError(
                    f"selector policy {policy_id} changed the frozen representative release"
                )
            rows.append(
                {
                    "group_id": group_id,
                    "policy_id": policy_id,
                    "selected_family": family.value,
                    "selected_release_id": release_id,
                    "selection_eligible": True,
                    "diagnostic_only": policy["diagnostic_only"],
                }
            )
    base = {
        "schema_version": 1,
        "artifact": _DECISION_ARTIFACT,
        "stage": "frozen",
        "data_status": "OUTCOME_BLIND_DECISIONS",
        "is_synthetic": False,
        "campaign_manifest_hash": manifest["campaign_hash"],
        "evaluation_input_hash": inputs["input_hash"],
        "policy_artifact_hashes": artifact_hashes,
        "records": sorted(rows, key=lambda row: (row["group_id"], row["policy_id"])),
        "outcomes_accessed": False,
        "other_arm_outcomes_accessed": False,
        "diagnostic_oracle_included": False,
    }
    artifact = freeze_mapping({**base, "decision_hash": content_hash(base)})
    return validate_selector_campaign_decisions(manifest, inputs, artifact)


def validate_selector_campaign_decisions(
    campaign: Mapping[str, Any],
    evaluation_input: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest = validate_selector_campaign_manifest(campaign)
    inputs = validate_selector_evaluation_input(evaluation_input)
    _validate_campaign_input_binding(manifest, inputs)
    required = {
        "schema_version",
        "artifact",
        "stage",
        "data_status",
        "is_synthetic",
        "campaign_manifest_hash",
        "evaluation_input_hash",
        "policy_artifact_hashes",
        "records",
        "outcomes_accessed",
        "other_arm_outcomes_accessed",
        "diagnostic_oracle_included",
        "decision_hash",
    }
    data = _strict_mapping(decisions, required, label="selector campaign decisions")
    if (
        data["schema_version"] != 1
        or data["artifact"] != _DECISION_ARTIFACT
        or data["stage"] != "frozen"
        or data["data_status"] != "OUTCOME_BLIND_DECISIONS"
        or data["is_synthetic"] is not False
        or data["campaign_manifest_hash"] != manifest["campaign_hash"]
        or data["evaluation_input_hash"] != inputs["input_hash"]
        or data["outcomes_accessed"] is not False
        or data["other_arm_outcomes_accessed"] is not False
        or data["diagnostic_oracle_included"] is not False
    ):
        raise ValidationError("selector decision artifact violates the outcome-blind boundary")
    hashes = _mapping(data["policy_artifact_hashes"], label="selector policy hashes")
    if set(hashes) != _POLICY_ARTIFACT_IDS:
        raise ValidationError("selector decision artifact policy hashes are incomplete")
    for digest in hashes.values():
        _require_sha256(digest, label="selector policy artifact hash")
    policies = {
        policy["policy_id"]: policy
        for policy in manifest["policies"]
        if policy["kind"] != "observed_action_oracle"
    }
    groups = {group["group_id"]: group for group in inputs["groups"]}
    rank_one = _campaign_rank_one_releases(manifest)
    seen: set[tuple[str, str]] = set()
    parsed_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(data["records"], label="selector decisions")):
        row = _strict_mapping(
            raw,
            {
                "group_id",
                "policy_id",
                "selected_family",
                "selected_release_id",
                "selection_eligible",
                "diagnostic_only",
            },
            label=f"selector decision {index}",
        )
        group_id = _group_id(row["group_id"], label="selector decision group ID")
        policy_id = _text(row["policy_id"], label="selector decision policy ID")
        if group_id not in groups or policy_id not in policies:
            raise ValidationError("selector decision references an unknown group or policy")
        key = (group_id, policy_id)
        if key in seen:
            raise ValidationError("selector decisions contain duplicate group-policy rows")
        seen.add(key)
        family = _family(row["selected_family"])
        release_id = _text(row["selected_release_id"], label="selector decision release")
        _validate_selected_action(
            group=groups[group_id],
            family=family,
            release_id=release_id,
        )
        if release_id != _release_for(family, rank_one)[1]:
            raise ValidationError(
                "selector decision release differs from the campaign representative"
            )
        if (
            row["selection_eligible"] is not True
            or row["diagnostic_only"] is not policies[policy_id]["diagnostic_only"]
        ):
            raise ValidationError("selector decision eligibility or policy role is invalid")
        parsed_rows.append(
            {
                **row,
                "selected_family": family.value,
            }
        )
    expected = {(group_id, policy_id) for group_id in groups for policy_id in policies}
    if seen != expected:
        raise ValidationError("selector decision artifact lacks the complete non-oracle grid")
    _require_sha256(data["decision_hash"], label="selector decision hash")
    if (
        content_hash({key: item for key, item in data.items() if key != "decision_hash"})
        != data["decision_hash"]
    ):
        raise ValidationError("selector decision hash differs from its content")
    return freeze_mapping({**data, "records": tuple(parsed_rows)})


def assemble_selector_campaign_results(
    campaign: Mapping[str, Any],
    evaluation_input: Mapping[str, Any],
    decisions: Mapping[str, Any],
    run_dirs: Sequence[str | Path],
    *,
    extractor: VisibleStateFeatureExtractor,
    utility_definition: UtilityDefinition,
) -> Mapping[str, Any]:
    """Join frozen choices to verified outcomes and add the diagnostic oracle."""

    manifest = validate_selector_campaign_manifest(campaign)
    inputs = validate_selector_evaluation_input(evaluation_input)
    frozen_decisions = validate_selector_campaign_decisions(manifest, inputs, decisions)
    if utility_definition.utility_metric != manifest["primary_metric"]:
        raise ValidationError("selector result utility differs from the campaign primary metric")
    input_extractor = inputs["extractor"]
    if (
        extractor.extractor_id != input_extractor["id"]
        or extractor.extractor_revision != input_extractor["revision"]
        or extractor.extractor_hash != input_extractor["sha256"]
        or extractor.feature_schema_id != input_extractor["feature_schema_id"]
        or extractor.feature_schema_hash != input_extractor["feature_schema_sha256"]
    ):
        raise ValidationError("selector result assembler changed the frozen feature extractor")

    rank_one = _campaign_rank_one_releases(manifest)
    outcomes, sources = _verified_group_outcomes(
        run_dirs,
        inputs=inputs,
        extractor=extractor,
        utility_definition=utility_definition,
        rank_one=rank_one,
    )
    if sources != tuple(inputs["source_runs"]):
        raise ValidationError("selector result source runs differ from pre-action input")
    groups = {group["group_id"]: group for group in inputs["groups"]}
    result_rows: list[dict[str, Any]] = []
    for decision in frozen_decisions["records"]:
        group = groups[decision["group_id"]]
        family = SemanticFamily(decision["selected_family"])
        selected = outcomes[decision["group_id"]][family]
        baseline = outcomes[decision["group_id"]][SemanticFamily.NO_OP]
        intervened = family is not SemanticFamily.NO_OP
        result_rows.append(
            _campaign_result_row(
                group=group,
                policy_id=decision["policy_id"],
                family=family,
                release_id=decision["selected_release_id"],
                value=selected["value"],
                no_op_value=baseline["value"],
                cost=selected["cost"],
                diagnostic_only=decision["diagnostic_only"],
                hard_harm=intervened and bool(selected["hard_harm"]),
            )
        )
    oracle_policy = next(
        policy for policy in manifest["policies"] if policy["kind"] == "observed_action_oracle"
    )
    for group_id, group in groups.items():
        available = outcomes[group_id]
        selected_family = min(
            available,
            key=lambda family: (
                -available[family]["value"],
                0 if family is SemanticFamily.NO_OP else 1,
                family.value,
            ),
        )
        selected = available[selected_family]
        baseline = available[SemanticFamily.NO_OP]
        result_rows.append(
            _campaign_result_row(
                group=group,
                policy_id=oracle_policy["policy_id"],
                family=selected_family,
                release_id=selected["release_id"],
                value=selected["value"],
                no_op_value=baseline["value"],
                cost=selected["cost"],
                diagnostic_only=True,
                hard_harm=(
                    selected_family is not SemanticFamily.NO_OP and bool(selected["hard_harm"])
                ),
            )
        )
    result = {
        "schema_version": 1,
        "artifact": "agentinputlab.selector_campaign_results",
        "stage": "verified",
        "data_status": "VERIFIED_REAL_RESULTS",
        "is_synthetic": False,
        "campaign_manifest_hash": manifest["campaign_hash"],
        "source_runs": list(sources),
        "records": result_rows,
    }
    return validate_selector_campaign_results(manifest, result)


def _verified_group_outcomes(
    run_dirs: Sequence[str | Path],
    *,
    inputs: Mapping[str, Any],
    extractor: VisibleStateFeatureExtractor,
    utility_definition: UtilityDefinition,
    rank_one: Mapping[SemanticFamily, str],
) -> tuple[dict[str, dict[SemanticFamily, dict[str, Any]]], tuple[Mapping[str, Any], ...]]:
    expected_groups = {group["group_id"]: group for group in inputs["groups"]}
    cell_outcomes: dict[str, dict[SemanticFamily, list[EvaluatedArmOutcome]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sources: list[dict[str, str]] = []
    for root_value in run_dirs:
        root = Path(root_value).expanduser().resolve()
        manifest, records, source = _verified_source(root)
        common_support = _validate_evaluation_manifest(
            manifest,
            representative_manifest_hash=inputs["representative_manifest_hash"],
            rank_one=rank_one,
            expected_partial_plan_hash=inputs["partial_support_plan_hash"],
        )
        sources.append(source)
        arm_by_id = {arm.arm_id: arm for arm in manifest.arms}
        no_op_ids = [arm.arm_id for arm in manifest.arms if arm.family is SemanticFamily.NO_OP]
        if len(no_op_ids) != 1:
            raise ValidationError("selector source manifest requires one NO_OP arm")
        representative_arm_ids = {
            arm.family: arm.arm_id
            for arm in manifest.arms
            if arm.family is not SemanticFamily.NO_OP
        }
        for record in records:
            converted, observation, _profile, _model, _backend = _convert_result_record(
                record,
                arm_by_id=arm_by_id,
                no_op_arm_id=no_op_ids[0],
                representative_arm_ids=representative_arm_ids,
                representative_by_family=rank_one,
                extractor=extractor,
                selector_split_override="test",
            )
            group_id = converted[0].group_id
            group = expected_groups.get(group_id)
            if group is None:
                raise ValidationError("selector outcomes contain an unknown group")
            if (
                observation.feature_hash != group["observation"]["feature_hash"]
                or common_support is not group["common_support"]
            ):
                raise ValidationError("selector outcomes differ from the frozen pre-action input")
            for outcome in converted:
                cell_outcomes[group_id][outcome.family].append(outcome)
    if set(cell_outcomes) != set(expected_groups):
        raise ValidationError("selector outcomes do not cover every frozen group")
    aggregated: dict[str, dict[SemanticFamily, dict[str, Any]]] = {}
    for group_id, by_family in cell_outcomes.items():
        eligible = {
            SemanticFamily(value) for value in expected_groups[group_id]["eligible_families"]
        }
        expected_actions = {SemanticFamily.NO_OP, *eligible}
        if set(by_family) != expected_actions:
            raise ValidationError("selector outcomes differ from the frozen group eligibility mask")
        aggregated[group_id] = {}
        for family, rows in by_family.items():
            if len(rows) != 2:
                raise ValidationError(
                    "selector outcome action requires exactly two model-slot observations"
                )
            values = [utility_definition.value(row) for row in rows]
            release_ids = {row.release_id for row in rows}
            if len(release_ids) != 1:
                raise ValidationError("selector outcome action changed release identity")
            aggregated[group_id][family] = {
                "value": fmean(values),
                "cost": fmean(row.cost for row in rows),
                "hard_harm": any(row.risk > 0.0 for row in rows),
                "release_id": next(iter(release_ids)),
            }
    return aggregated, tuple(sorted(sources, key=lambda item: item["run_identity_hash"]))


def _pre_action_record(
    record: Mapping[str, Any],
    *,
    arm_by_id: Mapping[str, Any],
    extractor: VisibleStateFeatureExtractor,
    common_support: bool,
) -> dict[str, Any]:
    cell = _mapping(record.get("cell"), label="selector source cell")
    result = _mapping(record.get("result"), label="selector source result")
    if result.get("status") != "succeeded":
        raise ArtifactError("selector evaluation requires succeeded source cells")
    payload = _mapping(result.get("payload"), label="selector source payload")
    selector_record = _strict_mapping(
        payload.get("selector_record"),
        {"schema_version", "state", "group_id", "eligible_families"},
        label="selector source pre-action record",
    )
    if selector_record["schema_version"] != 1:
        raise ArtifactError("unsupported selector pre-action record schema")
    state = _state_from_artifact(selector_record["state"])
    observation = extractor.extract(state)
    _validate_pre_action_receipt(
        payload.get("pre_action"),
        observation=observation,
        extractor=extractor,
    )
    cell_arm_ids = tuple(
        _text(item, label="selector source arm ID")
        for item in _sequence(cell.get("arm_ids"), label="selector source arm IDs")
    )
    if set(cell_arm_ids) != set(arm_by_id):
        raise ValidationError("selector source cell arms differ from its manifest")
    observed_families = {
        arm_by_id[arm_id].family
        for arm_id in cell_arm_ids
        if arm_by_id[arm_id].family is not SemanticFamily.NO_OP
    }
    eligible = set(_family_sequence(selector_record["eligible_families"]))
    if eligible != observed_families:
        raise ValidationError("selector pre-action eligibility differs from the executed arm block")
    if len(eligible) != (5 if common_support else 3):
        raise ValidationError("selector source support cardinality is invalid")
    benchmark_id = _text(cell.get("benchmark_id"), label="selector source benchmark")
    raw_group_id = _text(selector_record["group_id"], label="selector source group ID")
    group_id = "group_" + content_hash({"benchmark_id": benchmark_id, "group_id": raw_group_id})
    return {
        "group_id": group_id,
        "suite_id": benchmark_id,
        "state_hash": state.state_hash,
        "observation": _observation_to_artifact(observation),
        "eligible_families": [family.value for family in _CORE_FAMILIES if family in eligible],
        "model_slot": _text(cell.get("model_slot"), label="selector source model slot"),
        "seed": _integer(cell.get("seed"), label="selector source seed"),
    }


def _verified_source(
    root: Path,
) -> tuple[ExperimentManifest, tuple[Mapping[str, Any], ...], dict[str, str]]:
    verification = verify_experiment_run_directory(root)
    if not verification["complete"] or not verification["all_succeeded"]:
        raise ArtifactError("selector evaluation requires complete, all-succeeded verified runs")
    manifest = load_experiment_manifest(root / "manifest.json")
    records = load_verified_cell_results(root)
    source = {
        "study_id": manifest.study_id,
        "run_identity_hash": _sha256(verification["identity_hash"], label="source run identity"),
        "manifest_hash": _sha256(verification["manifest_hash"], label="source manifest hash"),
        "plan_hash": _sha256(verification["plan_hash"], label="source plan hash"),
        "result_set_hash": content_hash(records),
    }
    return manifest, records, source


def _validate_evaluation_manifest(
    manifest: ExperimentManifest,
    *,
    representative_manifest_hash: str,
    rank_one: Mapping[SemanticFamily, str],
    expected_partial_plan_hash: str,
) -> bool:
    common_support = manifest.study_id == "primary.common_support"
    partial_support = manifest.study_id.startswith("selector.partial_support.")
    if not common_support and not partial_support:
        raise ValidationError("selector evaluation received an undeclared source study")
    if common_support:
        census = manifest.metadata.get("experiment_census", {})
        if (
            manifest.stage is not ExperimentStage.CONFIRMATORY
            or census.get("campaign_kind") != "common_support"
            or census.get("independent_group_target") != 526
            or census.get("expected_agent_episodes") != 6312
        ):
            raise ValidationError("selector common-support source is not confirmatory")
        expected_count = 5
    else:
        census = manifest.metadata.get("experiment_census", {})
        if (
            manifest.stage is not ExperimentStage.SELECTOR_CONFIRMATION
            or manifest.metadata.get("partial_support") is not True
            or manifest.metadata.get("common_support") is not False
            or manifest.metadata.get("fit_or_calibrate_allowed") is not False
            or manifest.metadata.get("test_outcomes_visible_to_selector") is not False
            or manifest.metadata.get("eligibility_plan_hash") != expected_partial_plan_hash
            or census.get("campaign_kind") != "selector_partial_support"
            or census.get("independent_group_target") != 10
            or census.get("expected_agent_episodes") != 80
        ):
            raise ValidationError("selector partial-support source is not frozen")
        expected_count = 3
    if (
        manifest.metadata.get("representative_manifest_hash") != representative_manifest_hash
        or manifest.model_slots != ("anchor_a", "anchor_b")
        or manifest.seeds != (17,)
    ):
        raise ValidationError("selector source changed representatives, models, or seed")
    no_ops = [arm for arm in manifest.arms if arm.family is SemanticFamily.NO_OP]
    interventions = [arm for arm in manifest.arms if arm.family is not SemanticFamily.NO_OP]
    if (
        len(no_ops) != 1
        or no_ops[0].release_id != "baseline.no_op"
        or len(interventions) != expected_count
        or len({arm.family for arm in interventions}) != expected_count
    ):
        raise ValidationError("selector source arm support is invalid")
    for arm in interventions:
        if (
            arm.release_id != rank_one.get(arm.family)
            or arm.parameters.get("representative_rank") != 1
            or arm.parameters.get("representative_manifest_hash") != representative_manifest_hash
        ):
            raise ValidationError("selector source arm is not a rank-1 representative")
    return common_support


def _select_policy_action(
    policy: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
    state: FrozenAgentState,
    features: Any,
    eligible: Sequence[SemanticFamily],
    rank_one: Mapping[SemanticFamily, str],
    loaded: Mapping[str, Any],
    random_seed: int,
) -> tuple[SemanticFamily, str]:
    kind = policy["kind"]
    policy_id = policy["policy_id"]
    parameters = policy["parameters"]
    if kind == "no_op":
        return SemanticFamily.NO_OP, "baseline.no_op"
    if kind == "random_eligible":
        actions = (SemanticFamily.NO_OP, *eligible)
        seed = int(
            content_hash(
                {
                    "campaign_seed": random_seed,
                    "group_id": group["group_id"],
                    "policy_id": policy_id,
                }
            ),
            16,
        )
        family = random.Random(seed).choice(actions)
        return _release_for(family, rank_one)
    if kind == "fixed_family":
        family = SemanticFamily(parameters["family"])
        return (
            _release_for(family, rank_one)
            if family in eligible
            else (SemanticFamily.NO_OP, "baseline.no_op")
        )
    if kind == "best_fixed_global":
        family = SemanticFamily(parameters["selected_family"])
        return (
            _release_for(family, rank_one)
            if family in eligible
            else (SemanticFamily.NO_OP, "baseline.no_op")
        )
    if kind != "frozen_artifact_policy":
        raise ValidationError(f"unsupported outcome-blind selector policy kind: {kind}")
    artifact = loaded[policy_id]
    if isinstance(artifact, Mapping):
        replay = artifact[group["group_id"]]
        return SemanticFamily(replay["selected_family"]), replay["selected_release_id"]
    decision = artifact.select(
        state,
        features,
        eligible_families=eligible,
    )
    if decision.family is SemanticFamily.NO_OP:
        return SemanticFamily.NO_OP, "baseline.no_op"
    if decision.family not in eligible:
        raise ValidationError(f"frozen selector {policy_id} selected an ineligible family")
    expected_release = rank_one[decision.family]
    if decision.release_id not in (None, expected_release):
        raise ValidationError(f"frozen selector {policy_id} changed representative release")
    return decision.family, expected_release


def _campaign_result_row(
    *,
    group: Mapping[str, Any],
    policy_id: str,
    family: SemanticFamily,
    release_id: str,
    value: float,
    no_op_value: float,
    cost: float,
    diagnostic_only: bool,
    hard_harm: bool,
) -> dict[str, Any]:
    return {
        "group_id": group["group_id"],
        "suite_id": group["suite_id"],
        "policy_id": policy_id,
        "selected_family": family.value,
        "selected_release_id": release_id,
        "value": value,
        "no_op_value": no_op_value,
        "hard_harm": hard_harm,
        "cost": cost,
        "intervened": family is not SemanticFamily.NO_OP,
        "selection_eligible": True,
        "common_support": group["common_support"],
        "diagnostic_only": diagnostic_only,
    }


def _validate_campaign_input_binding(
    campaign: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    design = inputs["campaign_design"]
    if (
        campaign["campaign_id"] != design["campaign_id"]
        or campaign["primary_metric"] != design["primary_metric"]
        or campaign["evaluation_split"] != design["evaluation_split"]
        or campaign["representative_manifest_hash"] != inputs["representative_manifest_hash"]
        or campaign["metadata"]["suite_ids"] != design["suite_ids"]
        or campaign["metadata"]["required_independent_groups"]
        != design["required_independent_groups"]
        or campaign["metadata"]["minimum_common_support_groups"]
        != design["minimum_common_support_groups"]
        or campaign["metadata"]["random_seed"] != design["random_seed"]
    ):
        raise ValidationError("selector campaign differs from its pre-action input design")


def _campaign_rank_one_releases(
    campaign: Mapping[str, Any],
) -> dict[SemanticFamily, str]:
    policies = {policy["policy_id"]: policy for policy in campaign["policies"]}
    releases = {
        family: policies[f"fixed_{family.value}"]["parameters"]["release_id"]
        for family in _CORE_FAMILIES
    }
    if any(not isinstance(value, str) or not value for value in releases.values()):
        raise ValidationError("selector campaign fixed-family releases are invalid")
    return releases


def _rank_one_releases(
    representatives: Mapping[str, Any],
) -> dict[SemanticFamily, str]:
    result = {
        SemanticFamily(row["family"]): row["release_id"]
        for row in representatives["selections"]
        if row["rank"] == 1
    }
    if set(result) != set(_CORE_FAMILIES):
        raise ValidationError("selector evaluation requires one rank-1 release per family")
    return result


def _validate_selected_action(
    *,
    group: Mapping[str, Any],
    family: SemanticFamily,
    release_id: str,
) -> None:
    if family is SemanticFamily.NO_OP:
        if release_id != "baseline.no_op":
            raise ValidationError("selector NO_OP decision requires baseline.no_op")
        return
    if family.value not in group["eligible_families"]:
        raise ValidationError("selector decision selected a structurally ineligible family")
    if release_id == "baseline.no_op":
        raise ValidationError("selector intervention decision requires an intervention release")


def _release_for(
    family: SemanticFamily,
    rank_one: Mapping[SemanticFamily, str],
) -> tuple[SemanticFamily, str]:
    if family is SemanticFamily.NO_OP:
        return family, "baseline.no_op"
    return family, rank_one[family]


def _validate_sources(value: Any) -> tuple[dict[str, str], ...]:
    rows = _sequence(value, label="selector evaluation source runs")
    parsed: list[dict[str, str]] = []
    identities: set[str] = set()
    for index, raw in enumerate(rows):
        row = _strict_mapping(
            raw,
            {
                "study_id",
                "run_identity_hash",
                "manifest_hash",
                "plan_hash",
                "result_set_hash",
            },
            label=f"selector evaluation source {index}",
        )
        parsed_row = {
            "study_id": _text(row["study_id"], label="selector source study"),
            "run_identity_hash": _sha256(
                row["run_identity_hash"], label="selector source identity"
            ),
            "manifest_hash": _sha256(row["manifest_hash"], label="selector source manifest"),
            "plan_hash": _sha256(row["plan_hash"], label="selector source plan"),
            "result_set_hash": _sha256(row["result_set_hash"], label="selector source results"),
        }
        if parsed_row["run_identity_hash"] in identities:
            raise ValidationError("selector source run identities must be unique")
        identities.add(parsed_row["run_identity_hash"])
        parsed.append(parsed_row)
    return tuple(sorted(parsed, key=lambda item: item["run_identity_hash"]))


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = to_jsonable(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("selector artifact mapping must be JSON serializable") from exc
    if not isinstance(normalized, dict):
        raise ValidationError("selector artifact mapping must remain an object")
    return normalized


def _family_sequence(value: Any) -> tuple[SemanticFamily, ...]:
    raw = _sequence(value, label="eligible families")
    try:
        families = tuple(SemanticFamily(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError("eligible families contain an unknown semantic family") from exc
    if (
        len(families) != len(set(families))
        or any(family not in _CORE_FAMILIES for family in families)
        or families != tuple(family for family in _CORE_FAMILIES if family in families)
    ):
        raise ValidationError("eligible families must be canonical unique core families")
    return families


def _family(value: Any) -> SemanticFamily:
    try:
        family = SemanticFamily(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("selector decision family is invalid") from exc
    if family not in {SemanticFamily.NO_OP, *_CORE_FAMILIES}:
        raise ValidationError("selector decision family is outside the campaign action set")
    return family


def _strict_mapping(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object")
    data = dict(value)
    if set(data) != keys:
        raise ValidationError(f"{label} fields differ from the frozen schema")
    return data


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    return tuple(value)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{label} must be boolean")
    return value


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")


def _sha256(value: Any, *, label: str) -> str:
    _require_sha256(value, label=label)
    return value


def _group_id(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if not text.startswith("group_") or _SHA256.fullmatch(text[6:]) is None:
        raise ValidationError(f"{label} must be a privacy-safe group hash")
    return text


__all__ = [
    "assemble_selector_campaign_results",
    "build_selector_campaign_decisions",
    "build_selector_evaluation_input",
    "freeze_zero_shot_router_replay",
    "validate_selector_campaign_decisions",
    "validate_selector_evaluation_input",
    "validate_zero_shot_router_replay",
]
