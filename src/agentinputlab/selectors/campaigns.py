"""Frozen selector-study manifests and full-information policy analysis.

The campaign layer is deliberately execution neutral.  It binds preregistered
selection rules, frozen policy artifacts, evaluation support, and diagnostic
roles without calling a model or benchmark.  Real outcomes enter only through
an explicitly verified, non-synthetic result artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_GROUP_ID = re.compile(r"^group_[0-9a-f]{64}$")
_CORE_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)
_REPRESENTATIVE_ARTIFACT = "agentinputlab.selector_representative_manifest"
_CAMPAIGN_ARTIFACT = "agentinputlab.selector_campaign_manifest"
_RESULT_ARTIFACT = "agentinputlab.selector_campaign_results"
_LOSO_ARTIFACT = "agentinputlab.selector_loso_manifest"
_DISCOVERY_SUMMARY_ARTIFACT = "agentinputlab.selector_discovery_summary"
_RULE_ARTIFACT = "agentinputlab.selector_pareto_rule"
_DESIGN_ARTIFACT = "agentinputlab.selector_campaign_design"
_BINDINGS_ARTIFACT = "agentinputlab.selector_campaign_bindings"
_ABLATION_BINDINGS_ARTIFACT = "agentinputlab.selector_ablation_bindings"
_ABLATION_MATRIX_ARTIFACT = "agentinputlab.selector_ablation_matrix"
_LOSO_DESIGN_ARTIFACT = "agentinputlab.selector_loso_design"


def freeze_pareto_representatives(
    summary: Mapping[str, Any],
    rule: Mapping[str, Any],
    *,
    source_file_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Select two deterministic representatives per family from verified discovery.

    Candidates are first filtered by the frozen support/safety rule, then
    ordered by successive Pareto fronts over benefit (higher), hard harm
    (lower), and normalized cost (lower).  The preregistered utility score and
    a stable release-ID tie break order candidates within a front.
    """

    parsed_summary = _validated_discovery_summary(summary)
    parsed_rule = _validated_pareto_rule(rule)
    if source_file_sha256 is not None:
        _require_sha256(source_file_sha256, label="discovery summary file SHA-256")
    candidates = parsed_summary["candidates"]
    by_family: dict[SemanticFamily, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate["family"]].append(candidate)

    selected_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    count = parsed_rule["representatives_per_family"]
    for family in _CORE_FAMILIES:
        family_rows = by_family.get(family, [])
        if not family_rows:
            raise ValidationError(f"discovery summary has no candidates for {family.value}")
        eligible: list[dict[str, Any]] = []
        for candidate in family_rows:
            reasons: list[str] = []
            if candidate["independent_groups"] < parsed_rule["minimum_independent_groups"]:
                reasons.append("insufficient_independent_groups")
            if candidate["support_fraction"] < parsed_rule["minimum_support_fraction"]:
                reasons.append("insufficient_support")
            if candidate["hard_harm_rate"] > parsed_rule["maximum_hard_harm_rate"]:
                reasons.append("hard_harm_gate")
            if candidate["mean_valid_success_gain"] < parsed_rule["minimum_valid_success_gain"]:
                reasons.append("minimum_gain_gate")
            if reasons:
                rejected_rows.append(
                    {
                        "family": family.value,
                        "release_id": candidate["release_id"],
                        "reason_codes": sorted(reasons),
                    }
                )
            else:
                eligible.append(candidate)
        if len(eligible) < count:
            raise ValidationError(
                f"Pareto freeze requires at least {count} eligible candidates for "
                f"{family.value}; found {len(eligible)}"
            )
        ranked = _pareto_ranked(eligible, parsed_rule)
        for rank, row in enumerate(ranked[:count], start=1):
            selected_rows.append(
                {
                    "family": family.value,
                    "rank": rank,
                    "release_id": row["release_id"],
                    "pareto_layer": row["pareto_layer"],
                    "utility_score": row["utility_score"],
                    "mean_valid_success_gain": row["mean_valid_success_gain"],
                    "hard_harm_rate": row["hard_harm_rate"],
                    "normalized_cost": row["normalized_cost"],
                    "independent_groups": row["independent_groups"],
                    "support_fraction": row["support_fraction"],
                    "source_run_identity_hash": row["source_run_identity_hash"],
                }
            )

    semantic_summary_hash = content_hash(_summary_for_hash(parsed_summary))
    base = {
        "schema_version": 1,
        "artifact": _REPRESENTATIVE_ARTIFACT,
        "stage": "frozen",
        "selection_scope": "two_per_core_family",
        "source_summary": {
            "revision": parsed_summary["revision"],
            "semantic_sha256": semantic_summary_hash,
            "file_sha256": source_file_sha256,
            "profile_kind": "controlled_adapter",
            "source_run_identity_hashes": sorted(
                item["run_identity_hash"] for item in parsed_summary["source_runs"]
            ),
        },
        "rule": {
            **{key: value for key, value in parsed_rule.items() if key != "rule_hash"},
            "rule_hash": parsed_rule["rule_hash"],
        },
        "selections": selected_rows,
        "rejections": sorted(
            rejected_rows,
            key=lambda item: (item["family"], item["release_id"]),
        ),
        "experiments_executed": False,
    }
    return freeze_mapping({**base, "manifest_hash": content_hash(base)})


def validate_representative_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and freeze one representative-selection artifact."""

    data = _strict_mapping(
        value,
        label="representative manifest",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "selection_scope",
            "source_summary",
            "rule",
            "selections",
            "rejections",
            "experiments_executed",
            "manifest_hash",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _REPRESENTATIVE_ARTIFACT
        or data["stage"] != "frozen"
        or data["selection_scope"] != "two_per_core_family"
        or data["experiments_executed"] is not False
    ):
        raise ValidationError("representative manifest identity or stage is invalid")
    expected_hash = data["manifest_hash"]
    _require_sha256(expected_hash, label="representative manifest hash")
    if content_hash({key: item for key, item in data.items() if key != "manifest_hash"}) != (
        expected_hash
    ):
        raise ValidationError("representative manifest hash differs from its content")
    source = _strict_mapping(
        data["source_summary"],
        label="representative source summary",
        keys={
            "revision",
            "semantic_sha256",
            "file_sha256",
            "profile_kind",
            "source_run_identity_hashes",
        },
    )
    _require_text(source["revision"], label="representative source revision")
    _require_sha256(source["semantic_sha256"], label="representative source semantic hash")
    if source["file_sha256"] is not None:
        _require_sha256(source["file_sha256"], label="representative source file hash")
    if source["profile_kind"] != "controlled_adapter":
        raise ValidationError("selector representatives require controlled discovery outcomes")
    identities = _string_sequence(
        source["source_run_identity_hashes"],
        label="representative source run identities",
    )
    if not identities or identities != tuple(sorted(set(identities))):
        raise ValidationError("representative source run identities must be sorted unique")
    for identity in identities:
        _require_sha256(identity, label="representative source run identity")
    rule = _strict_mapping(
        data["rule"],
        label="representative rule",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "representatives_per_family",
            "minimum_independent_groups",
            "minimum_support_fraction",
            "maximum_hard_harm_rate",
            "minimum_valid_success_gain",
            "benefit_weight",
            "harm_penalty",
            "cost_penalty",
            "tie_break",
            "rule_hash",
        },
    )
    _validated_pareto_rule({key: item for key, item in rule.items() if key != "rule_hash"})
    _require_sha256(rule["rule_hash"], label="representative rule hash")
    if (
        content_hash({key: item for key, item in rule.items() if key != "rule_hash"})
        != rule["rule_hash"]
    ):
        raise ValidationError("representative rule hash differs from its content")
    selections = _sequence(data["selections"], label="representative selections")
    expected_pairs = {(family.value, rank) for family in _CORE_FAMILIES for rank in (1, 2)}
    observed_pairs: set[tuple[str, int]] = set()
    releases: set[str] = set()
    for index, raw in enumerate(selections):
        row = _strict_mapping(
            raw,
            label=f"representative selection {index}",
            keys={
                "family",
                "rank",
                "release_id",
                "pareto_layer",
                "utility_score",
                "mean_valid_success_gain",
                "hard_harm_rate",
                "normalized_cost",
                "independent_groups",
                "support_fraction",
                "source_run_identity_hash",
            },
        )
        family = _core_family(row["family"], label=f"selection {index} family")
        rank = _positive_integer(row["rank"], label=f"selection {index} rank")
        observed_pairs.add((family.value, rank))
        release_id = _identifier(row["release_id"], label=f"selection {index} release")
        if release_id in releases:
            raise ValidationError("representative releases must be unique across families")
        releases.add(release_id)
        _nonnegative_integer(row["pareto_layer"], label=f"selection {index} Pareto layer")
        _finite(row["utility_score"], label=f"selection {index} utility score")
        _bounded(row["mean_valid_success_gain"], -1.0, 1.0, label="selection gain")
        _bounded(row["hard_harm_rate"], 0.0, 1.0, label="selection hard harm")
        _bounded(row["normalized_cost"], 0.0, 1.0, label="selection normalized cost")
        _positive_integer(row["independent_groups"], label="selection independent groups")
        _bounded(row["support_fraction"], 0.0, 1.0, label="selection support")
        _require_sha256(
            row["source_run_identity_hash"],
            label="selection source run identity",
        )
        if row["source_run_identity_hash"] not in identities:
            raise ValidationError(
                "representative selection source is absent from the source summary"
            )
    if observed_pairs != expected_pairs or len(selections) != 10:
        raise ValidationError("representative manifest requires exact ranks 1 and 2 per family")
    _sequence(data["rejections"], label="representative rejections")
    return freeze_mapping(data)


def build_selector_baseline_campaign(
    representatives: Mapping[str, Any],
    bindings: Mapping[str, Any],
    design: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the frozen multi-policy baseline campaign."""

    reps = validate_representative_manifest(representatives)
    parsed_bindings = _validated_campaign_bindings(bindings)
    parsed_design = _validated_campaign_design(design)
    if parsed_bindings["representative_manifest_hash"] != reps["manifest_hash"]:
        raise ValidationError("selector campaign bindings differ from the representative manifest")
    selected = _selected_releases(reps)
    best = parsed_bindings["best_fixed_global"]
    if best["selected_release_id"] != selected[best["selected_family"]][0]:
        raise ValidationError("best-fixed release is not the rank-1 selector representative")

    policies: list[dict[str, Any]] = [
        _policy(
            "no_op",
            kind="no_op",
            role="fixed_baseline",
            formal_inference=True,
            diagnostic_only=False,
            parameters={"release_id": "baseline.no_op"},
        ),
        _policy(
            "random_eligible",
            kind="random_eligible",
            role="frequency_control",
            formal_inference=True,
            diagnostic_only=False,
            parameters={"random_seed": parsed_design["random_seed"]},
        ),
    ]
    for family in _CORE_FAMILIES:
        policies.append(
            _policy(
                f"fixed_{family.value}",
                kind="fixed_family",
                role="family_fixed_baseline",
                formal_inference=True,
                diagnostic_only=False,
                parameters={
                    "family": family.value,
                    "release_id": selected[family][0],
                    "representative_rank": 1,
                },
            )
        )
    policies.extend(
        (
            _policy(
                "best_fixed_global",
                kind="best_fixed_global",
                role="calibration_frozen_baseline",
                formal_inference=True,
                diagnostic_only=False,
                parameters={
                    "selected_family": best["selected_family"].value,
                    "selected_release_id": best["selected_release_id"],
                    "policy_manifest_sha256": best["policy_manifest_sha256"],
                },
            ),
            _artifact_policy(
                "defect_rule",
                parsed_bindings["selectors"]["defect_rule"],
                role="interpretable_baseline",
                formal_inference=True,
            ),
            _artifact_policy(
                "zero_shot_router",
                parsed_bindings["selectors"]["zero_shot_router"],
                role="generic_router_comparator",
                formal_inference=True,
            ),
            _artifact_policy(
                "penalized_t_learner",
                parsed_bindings["selectors"]["penalized_t_learner"],
                role="confirmatory_learned_policy",
                formal_inference=True,
            ),
            _artifact_policy(
                "gbdt_t_learner",
                parsed_bindings["selectors"]["gbdt_t_learner"],
                role="exploratory_nonlinear_policy",
                formal_inference=False,
            ),
            _artifact_policy(
                "eligibility_aware_selector",
                parsed_bindings["selectors"]["eligibility_aware_selector"],
                role="locked_primary_selector",
                formal_inference=True,
            ),
            _policy(
                "observed_action_oracle",
                kind="observed_action_oracle",
                role="descriptive_upper_bound",
                formal_inference=False,
                diagnostic_only=True,
                parameters={
                    "outcome_access": "post_outcome_only",
                    "may_select_or_train": False,
                },
            ),
        )
    )
    metadata = {
        "suite_ids": list(parsed_design["suite_ids"]),
        "required_independent_groups": parsed_design["required_independent_groups"],
        "minimum_common_support_groups": parsed_design["minimum_common_support_groups"],
        "random_seed": parsed_design["random_seed"],
        "selector_representative_rank": parsed_bindings["representative_rank"],
        "oracle_policy_id": "observed_action_oracle",
        "oracle_diagnostic_only": True,
        "fit_or_calibrate_allowed": False,
        "preflight_first": True,
    }
    return _campaign(
        campaign_id=parsed_design["campaign_id"],
        campaign_kind="baseline",
        primary_metric=parsed_design["primary_metric"],
        evaluation_split=parsed_design["evaluation_split"],
        representative_manifest_hash=reps["manifest_hash"],
        policies=policies,
        metadata=metadata,
    )


def build_selector_ablation_campaign(
    baseline_campaign: Mapping[str, Any],
    bindings: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    campaign_id: str,
) -> Mapping[str, Any]:
    """Build refit-required feature and guardrail ablations."""

    baseline = validate_selector_campaign_manifest(baseline_campaign)
    if baseline["campaign_kind"] != "baseline":
        raise ValidationError("selector ablations require a baseline campaign")
    _identifier(campaign_id, label="selector ablation campaign ID")
    parsed = _validated_ablation_bindings(bindings)
    if parsed["baseline_campaign_hash"] != baseline["campaign_hash"]:
        raise ValidationError("selector ablation bindings differ from the baseline campaign")
    entries = _validated_ablation_matrix(matrix)
    policies: list[dict[str, Any]] = []
    no_op = next(item for item in baseline["policies"] if item["policy_id"] == "no_op")
    policies.append(dict(no_op))
    primary = next(
        item for item in baseline["policies"] if item["policy_id"] == "eligibility_aware_selector"
    )
    policies.append(
        {
            **dict(primary),
            "policy_id": "full_selector",
            "role": "ablation_reference",
            "formal_inference": False,
        }
    )
    unsafe = {entry["policy_id"] for entry in entries if entry["diagnostic_only"]}
    for entry in entries:
        policy_id = entry["policy_id"]
        artifact = parsed["artifacts"][policy_id]
        policies.append(
            _policy(
                policy_id,
                kind="frozen_ablation",
                role="prespecified_ablation",
                formal_inference=False,
                diagnostic_only=entry["diagnostic_only"],
                parameters={
                    "changed_component": entry["changed_component"],
                    "requires_independent_refit": entry["requires_independent_refit"],
                    "artifact_binding": artifact,
                },
            )
        )
    metadata = {
        **dict(baseline["metadata"]),
        "baseline_campaign_hash": baseline["campaign_hash"],
        "unsafe_ablation_policy_ids": sorted(unsafe),
        "ablation_results_are_secondary": True,
        "fit_or_calibrate_allowed_during_evaluation": False,
    }
    return _campaign(
        campaign_id=campaign_id,
        campaign_kind="ablation",
        primary_metric=baseline["primary_metric"],
        evaluation_split=baseline["evaluation_split"],
        representative_manifest_hash=baseline["representative_manifest_hash"],
        policies=policies,
        metadata=metadata,
    )


def build_selector_loso_manifest(design: Mapping[str, Any]) -> Mapping[str, Any]:
    """Generate leave-one-suite-out refit/freeze instructions."""

    data = _strict_mapping(
        design,
        label="selector LOSO design",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "campaign_id",
            "suite_ids",
            "selector_kind",
            "fit_config_sha256",
            "calibration_config_sha256",
            "representative_manifest_hash",
            "revision_prefix",
            "minimum_train_suites",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _LOSO_DESIGN_ARTIFACT
        or data["stage"] != "frozen"
    ):
        raise ValidationError("selector LOSO design identity or stage is invalid")
    campaign_id = _identifier(data["campaign_id"], label="selector LOSO campaign ID")
    suites = _identifier_sequence(data["suite_ids"], label="selector LOSO suites")
    minimum = _positive_integer(
        data["minimum_train_suites"],
        label="selector LOSO minimum train suites",
    )
    if len(suites) < minimum + 1:
        raise ValidationError("selector LOSO design has too few suites for its training floor")
    selector_kind = _identifier(data["selector_kind"], label="selector LOSO kind")
    for label in (
        "fit_config_sha256",
        "calibration_config_sha256",
        "representative_manifest_hash",
    ):
        _require_sha256(data[label], label=f"selector LOSO {label}")
    prefix = _identifier(data["revision_prefix"], label="selector LOSO revision prefix")
    folds = []
    for held_out in suites:
        training = tuple(item for item in suites if item != held_out)
        folds.append(
            {
                "fold_id": f"holdout.{held_out}",
                "held_out_suite": held_out,
                "training_suites": list(training),
                "calibration_source": "training_suites_group_disjoint",
                "evaluation_split": "untouched_held_out_suite",
                "selector_kind": selector_kind,
                "fit_config_sha256": data["fit_config_sha256"],
                "calibration_config_sha256": data["calibration_config_sha256"],
                "output_revision": f"{prefix}.{held_out}",
                "requires_fresh_fit": True,
                "test_outcomes_visible_during_fit": False,
                "test_outcomes_visible_during_calibration": False,
            }
        )
    base = {
        "schema_version": 1,
        "artifact": _LOSO_ARTIFACT,
        "stage": "frozen",
        "campaign_id": campaign_id,
        "representative_manifest_hash": data["representative_manifest_hash"],
        "minimum_train_suites": minimum,
        "folds": folds,
        "preflight_first": True,
        "experiments_executed": False,
    }
    return freeze_mapping({**base, "manifest_hash": content_hash(base)})


def validate_selector_loso_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a generated LOSO artifact and its disjoint folds."""

    data = _strict_mapping(
        value,
        label="selector LOSO manifest",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "campaign_id",
            "representative_manifest_hash",
            "minimum_train_suites",
            "folds",
            "preflight_first",
            "experiments_executed",
            "manifest_hash",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _LOSO_ARTIFACT
        or data["stage"] != "frozen"
        or data["preflight_first"] is not True
        or data["experiments_executed"] is not False
    ):
        raise ValidationError("selector LOSO manifest identity or stage is invalid")
    _identifier(data["campaign_id"], label="selector LOSO campaign ID")
    _require_sha256(
        data["representative_manifest_hash"],
        label="selector LOSO representative hash",
    )
    minimum = _positive_integer(
        data["minimum_train_suites"],
        label="selector LOSO minimum train suites",
    )
    folds = _sequence(data["folds"], label="selector LOSO folds")
    if len(folds) < minimum + 1:
        raise ValidationError("selector LOSO manifest has too few folds")
    held_out: set[str] = set()
    all_suites: set[str] = set()
    for index, raw in enumerate(folds):
        fold = _strict_mapping(
            raw,
            label=f"selector LOSO fold {index}",
            keys={
                "fold_id",
                "held_out_suite",
                "training_suites",
                "calibration_source",
                "evaluation_split",
                "selector_kind",
                "fit_config_sha256",
                "calibration_config_sha256",
                "output_revision",
                "requires_fresh_fit",
                "test_outcomes_visible_during_fit",
                "test_outcomes_visible_during_calibration",
            },
        )
        _identifier(fold["fold_id"], label=f"selector LOSO fold {index} ID")
        held = _identifier(fold["held_out_suite"], label=f"selector LOSO fold {index} holdout")
        training = set(
            _identifier_sequence(
                fold["training_suites"],
                label=f"selector LOSO fold {index} training suites",
            )
        )
        if held in training or len(training) < minimum:
            raise ValidationError("selector LOSO fold violates its suite separation")
        if held in held_out:
            raise ValidationError("selector LOSO holdout suites must be unique")
        held_out.add(held)
        all_suites.update(training)
        all_suites.add(held)
        if (
            fold["calibration_source"] != "training_suites_group_disjoint"
            or fold["evaluation_split"] != "untouched_held_out_suite"
            or fold["requires_fresh_fit"] is not True
            or fold["test_outcomes_visible_during_fit"] is not False
            or fold["test_outcomes_visible_during_calibration"] is not False
        ):
            raise ValidationError("selector LOSO fold permits held-out leakage")
        _identifier(fold["selector_kind"], label="selector LOSO selector kind")
        _identifier(fold["output_revision"], label="selector LOSO output revision")
        _require_sha256(fold["fit_config_sha256"], label="selector LOSO fit config hash")
        _require_sha256(
            fold["calibration_config_sha256"],
            label="selector LOSO calibration config hash",
        )
    if held_out != all_suites:
        raise ValidationError("selector LOSO folds do not hold out every declared suite")
    _require_sha256(data["manifest_hash"], label="selector LOSO manifest hash")
    if (
        content_hash({key: item for key, item in data.items() if key != "manifest_hash"})
        != data["manifest_hash"]
    ):
        raise ValidationError("selector LOSO manifest hash differs from its content")
    return freeze_mapping(data)


def validate_selector_campaign_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a frozen baseline or ablation campaign."""

    data = _strict_mapping(
        value,
        label="selector campaign manifest",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "campaign_id",
            "campaign_kind",
            "primary_metric",
            "evaluation_split",
            "representative_manifest_hash",
            "policies",
            "metadata",
            "experiments_executed",
            "campaign_hash",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _CAMPAIGN_ARTIFACT
        or data["stage"] != "frozen"
        or data["campaign_kind"] not in {"baseline", "ablation"}
        or data["experiments_executed"] is not False
    ):
        raise ValidationError("selector campaign identity or stage is invalid")
    _identifier(data["campaign_id"], label="selector campaign ID")
    _identifier(data["primary_metric"], label="selector campaign primary metric")
    _identifier(data["evaluation_split"], label="selector campaign evaluation split")
    _require_sha256(
        data["representative_manifest_hash"],
        label="selector campaign representative hash",
    )
    policies = _sequence(data["policies"], label="selector campaign policies")
    if len(policies) < 3:
        raise ValidationError("selector campaign requires at least three policies")
    policy_ids: set[str] = set()
    oracle_count = 0
    no_op_count = 0
    for index, raw in enumerate(policies):
        policy = _validated_policy(raw, label=f"selector campaign policy {index}")
        if policy["policy_id"] in policy_ids:
            raise ValidationError("selector campaign policy IDs must be unique")
        policy_ids.add(policy["policy_id"])
        no_op_count += policy["policy_id"] == "no_op"
        if policy["kind"] == "observed_action_oracle":
            oracle_count += 1
            if (
                policy["diagnostic_only"] is not True
                or policy["formal_inference"] is not False
                or policy["parameters"].get("outcome_access") != "post_outcome_only"
            ):
                raise ValidationError("observed-action oracle must remain diagnostic-only")
    if no_op_count != 1:
        raise ValidationError("selector campaign requires exactly one NO_OP policy")
    if data["campaign_kind"] == "baseline" and oracle_count != 1:
        raise ValidationError("baseline campaign requires one diagnostic observed-action oracle")
    if data["campaign_kind"] == "ablation" and oracle_count:
        raise ValidationError("ablation campaign must not introduce an outcome oracle")
    metadata = _mapping(data["metadata"], label="selector campaign metadata")
    _positive_integer(
        metadata.get("required_independent_groups"),
        label="selector campaign required independent groups",
    )
    minimum = _positive_integer(
        metadata.get("minimum_common_support_groups"),
        label="selector campaign minimum common-support groups",
    )
    if minimum > metadata["required_independent_groups"]:
        raise ValidationError("selector campaign common-support floor exceeds total groups")
    suites = _identifier_sequence(metadata.get("suite_ids"), label="selector campaign suites")
    if not suites:
        raise ValidationError("selector campaign requires suite IDs")
    if metadata.get("preflight_first") is not True:
        raise ValidationError("selector campaign must remain preflight-first")
    _require_sha256(data["campaign_hash"], label="selector campaign hash")
    if (
        content_hash({key: item for key, item in data.items() if key != "campaign_hash"})
        != data["campaign_hash"]
    ):
        raise ValidationError("selector campaign hash differs from its content")
    return freeze_mapping(data)


def validate_selector_campaign_design(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the outcome-blind design used before policy artifacts exist."""

    return freeze_mapping(_validated_campaign_design(value))


def validate_selector_campaign_results(
    campaign: Mapping[str, Any],
    results: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate complete, verified, non-synthetic multi-policy outcomes."""

    manifest = validate_selector_campaign_manifest(campaign)
    data = _strict_mapping(
        results,
        label="selector campaign results",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "data_status",
            "is_synthetic",
            "campaign_manifest_hash",
            "source_runs",
            "records",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _RESULT_ARTIFACT
        or data["stage"] != "verified"
        or data["data_status"] != "VERIFIED_REAL_RESULTS"
        or data["is_synthetic"] is not False
    ):
        raise ValidationError("selector results are not verified real outcomes")
    if data["campaign_manifest_hash"] != manifest["campaign_hash"]:
        raise ValidationError("selector result campaign hash differs from the manifest")
    source_runs = _validated_source_runs(data["source_runs"])
    if not source_runs:
        raise ValidationError("selector campaign results require verified source runs")
    policies = {
        item["policy_id"]: item
        for item in (
            _validated_policy(raw, label="selector result policy") for raw in manifest["policies"]
        )
    }
    records = _sequence(data["records"], label="selector campaign records")
    by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    suites = set(manifest["metadata"]["suite_ids"])
    common_groups: set[str] = set()
    for index, raw in enumerate(records):
        record = _validated_result_record(
            raw,
            label=f"selector result record {index}",
        )
        if record["suite_id"] not in suites:
            raise ValidationError("selector result references an undeclared suite")
        try:
            policy = policies[record["policy_id"]]
        except KeyError as exc:
            raise ValidationError("selector result references an undeclared policy") from exc
        if record["diagnostic_only"] is not policy["diagnostic_only"]:
            raise ValidationError("selector result diagnostic role differs from its policy")
        if record["policy_id"] in by_group[record["group_id"]]:
            raise ValidationError("selector results contain duplicate group-policy rows")
        if (
            not record["selection_eligible"]
            and record["intervened"]
            and not policy["diagnostic_only"]
        ):
            raise ValidationError("release policy selected a structurally ineligible action")
        by_group[record["group_id"]][record["policy_id"]] = record
        if record["common_support"]:
            common_groups.add(record["group_id"])

    required_groups = manifest["metadata"]["required_independent_groups"]
    if len(by_group) != required_groups:
        raise ValidationError(
            f"selector result group count is {len(by_group)}, expected {required_groups}"
        )
    if len(common_groups) < manifest["metadata"]["minimum_common_support_groups"]:
        raise ValidationError("selector results miss the frozen common-support floor")
    expected_policies = set(policies)
    fixed_policy_ids = {"no_op", *(f"fixed_{family.value}" for family in _CORE_FAMILIES)}
    for group_id, group_rows in by_group.items():
        if set(group_rows) != expected_policies:
            raise ValidationError(f"selector group {group_id} lacks the complete policy set")
        support_values = {row["common_support"] for row in group_rows.values()}
        suite_values = {row["suite_id"] for row in group_rows.values()}
        baseline_values = {row["no_op_value"] for row in group_rows.values()}
        if len(support_values) != 1 or len(suite_values) != 1 or len(baseline_values) != 1:
            raise ValidationError("selector group has inconsistent support, suite, or baseline")
        no_op = group_rows["no_op"]
        if (
            no_op["selected_family"] != SemanticFamily.NO_OP.value
            or no_op["selected_release_id"] != "baseline.no_op"
            or no_op["intervened"]
            or not no_op["selection_eligible"]
            or no_op["value"] != no_op["no_op_value"]
            or no_op["hard_harm"]
        ):
            raise ValidationError("selector NO_OP result violates the identity policy")
        if "observed_action_oracle" in group_rows:
            if not fixed_policy_ids.issubset(group_rows):
                raise ValidationError("oracle validation requires NO_OP and five fixed policies")
            fixed_values = [group_rows[policy_id]["value"] for policy_id in fixed_policy_ids]
            oracle = group_rows["observed_action_oracle"]
            maximum = max(fixed_values)
            if not math.isclose(oracle["value"], maximum, abs_tol=1e-12):
                raise ValidationError("observed-action oracle differs from fixed-action headroom")
            maximizing_actions = {
                (
                    group_rows[policy_id]["selected_family"],
                    group_rows[policy_id]["selected_release_id"],
                )
                for policy_id in fixed_policy_ids
                if math.isclose(group_rows[policy_id]["value"], maximum, abs_tol=1e-12)
            }
            if (
                oracle["selected_family"],
                oracle["selected_release_id"],
            ) not in maximizing_actions:
                raise ValidationError(
                    "observed-action oracle action differs from the maximizing fixed action"
                )
    return freeze_mapping(
        {
            **data,
            "source_runs": source_runs,
            "records": tuple(
                row
                for group_id in sorted(by_group)
                for _policy_id, row in sorted(by_group[group_id].items())
            ),
        }
    )


def analyze_selector_campaign_results(
    campaign: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    bootstrap_samples: int = 2_000,
    random_seed: int = 2027,
) -> Mapping[str, Any]:
    """Summarize multi-policy value, harm, cost, eligibility, and coverage."""

    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 100
    ):
        raise ValidationError("selector campaign bootstrap_samples must be an integer >= 100")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValidationError("selector campaign random_seed must be non-negative")
    manifest = validate_selector_campaign_manifest(campaign)
    verified = validate_selector_campaign_results(manifest, results)
    policy_specs = {item["policy_id"]: item for item in manifest["policies"]}
    rows_by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in verified["records"]:
        rows_by_policy[record["policy_id"]].append(record)
    rng = random.Random(random_seed)
    summaries: dict[str, Any] = {}
    for policy_id in sorted(rows_by_policy):
        rows = rows_by_policy[policy_id]
        effects = [row["value"] - row["no_op_value"] for row in rows]
        summaries[policy_id] = {
            "independent_groups": len(rows),
            "mean_value": fmean(row["value"] for row in rows),
            "mean_no_op_value": fmean(row["no_op_value"] for row in rows),
            "mean_gain_vs_no_op": fmean(effects),
            "gain_interval": _bootstrap_mean_interval(
                effects,
                samples=bootstrap_samples,
                rng=rng,
            ),
            "intervention_coverage": fmean(1.0 if row["intervened"] else 0.0 for row in rows),
            "hard_harm_rate": fmean(1.0 if row["hard_harm"] else 0.0 for row in rows),
            "ineligible_selection_rate": fmean(
                1.0 if row["intervened"] and not row["selection_eligible"] else 0.0 for row in rows
            ),
            "mean_cost": fmean(row["cost"] for row in rows),
            "diagnostic_only": policy_specs[policy_id]["diagnostic_only"],
            "formal_inference": policy_specs[policy_id]["formal_inference"],
        }
    formal = sorted(
        policy_id
        for policy_id, policy in policy_specs.items()
        if policy["formal_inference"] and not policy["diagnostic_only"]
    )
    diagnostic = sorted(
        policy_id for policy_id, policy in policy_specs.items() if policy["diagnostic_only"]
    )
    return freeze_mapping(
        {
            "schema_version": 1,
            "analysis": "selector_multi_policy_campaign",
            "campaign_id": manifest["campaign_id"],
            "campaign_hash": manifest["campaign_hash"],
            "primary_metric": manifest["primary_metric"],
            "evaluation_split": manifest["evaluation_split"],
            "independent_groups": manifest["metadata"]["required_independent_groups"],
            "common_support_groups": len(
                {row["group_id"] for row in verified["records"] if row["common_support"]}
            ),
            "policy_summaries": summaries,
            "formal_inference_policy_ids": formal,
            "diagnostic_policy_ids": diagnostic,
            "observed_action_oracle_used_for_training": False,
            "observed_action_oracle_used_for_inference": False,
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "paper_claim_authorized": False,
        }
    )


def load_json_mapping(path: str | Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    """Read one bounded JSON object and return its exact file digest."""

    source = Path(path).expanduser().resolve()
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read {label}: {source}") from exc
    if len(content) > 16 * 1024 * 1024:
        raise ArtifactError(f"{label} exceeds the 16 MiB JSON limit")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a JSON object")
    return value, hashlib.sha256(content).hexdigest()


def _validated_discovery_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label="selector discovery summary",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "revision",
            "data_status",
            "observed",
            "is_synthetic",
            "profile_kind",
            "source_runs",
            "candidates",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _DISCOVERY_SUMMARY_ARTIFACT
        or data["stage"] != "verified"
        or data["data_status"] != "VERIFIED_REAL_RESULTS"
        or data["observed"] is not True
        or data["is_synthetic"] is not False
        or data["profile_kind"] != "controlled_adapter"
    ):
        raise ValidationError("representative selection requires verified real controlled results")
    revision = _identifier(data["revision"], label="selector discovery summary revision")
    source_runs = _validated_source_runs(data["source_runs"])
    source_identities = {item["run_identity_hash"] for item in source_runs}
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[SemanticFamily, str]] = set()
    for index, raw in enumerate(_sequence(data["candidates"], label="discovery candidates")):
        row = _strict_mapping(
            raw,
            label=f"discovery candidate {index}",
            keys={
                "family",
                "release_id",
                "mean_valid_success_gain",
                "hard_harm_rate",
                "normalized_cost",
                "independent_groups",
                "support_fraction",
                "source_run_identity_hash",
            },
        )
        family = _core_family(row["family"], label=f"discovery candidate {index} family")
        release = _identifier(
            row["release_id"],
            label=f"discovery candidate {index} release",
        )
        key = (family, release)
        if key in seen:
            raise ValidationError("selector discovery candidates must be unique by family/release")
        seen.add(key)
        source_identity = row["source_run_identity_hash"]
        _require_sha256(source_identity, label="candidate source run identity")
        if source_identity not in source_identities:
            raise ValidationError("candidate source identity is absent from verified source runs")
        candidates.append(
            {
                "family": family,
                "release_id": release,
                "mean_valid_success_gain": _bounded(
                    row["mean_valid_success_gain"],
                    -1.0,
                    1.0,
                    label="candidate mean valid-success gain",
                ),
                "hard_harm_rate": _bounded(
                    row["hard_harm_rate"],
                    0.0,
                    1.0,
                    label="candidate hard-harm rate",
                ),
                "normalized_cost": _bounded(
                    row["normalized_cost"],
                    0.0,
                    1.0,
                    label="candidate normalized cost",
                ),
                "independent_groups": _positive_integer(
                    row["independent_groups"],
                    label="candidate independent groups",
                ),
                "support_fraction": _bounded(
                    row["support_fraction"],
                    0.0,
                    1.0,
                    label="candidate support fraction",
                ),
                "source_run_identity_hash": source_identity,
            }
        )
    if not candidates:
        raise ValidationError("selector discovery summary has no candidates")
    return {
        **data,
        "revision": revision,
        "source_runs": source_runs,
        "candidates": candidates,
    }


def _validated_pareto_rule(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label="selector Pareto rule",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "representatives_per_family",
            "minimum_independent_groups",
            "minimum_support_fraction",
            "maximum_hard_harm_rate",
            "minimum_valid_success_gain",
            "benefit_weight",
            "harm_penalty",
            "cost_penalty",
            "tie_break",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _RULE_ARTIFACT
        or data["stage"] != "frozen"
    ):
        raise ValidationError("selector Pareto rule must use frozen schema v1")
    count = _positive_integer(
        data["representatives_per_family"],
        label="representatives per family",
    )
    if count != 2:
        raise ValidationError("current selector protocol requires exactly two per family")
    parsed = {
        **data,
        "representatives_per_family": count,
        "minimum_independent_groups": _positive_integer(
            data["minimum_independent_groups"],
            label="Pareto minimum independent groups",
        ),
        "minimum_support_fraction": _bounded(
            data["minimum_support_fraction"],
            0.0,
            1.0,
            label="Pareto minimum support",
        ),
        "maximum_hard_harm_rate": _bounded(
            data["maximum_hard_harm_rate"],
            0.0,
            1.0,
            label="Pareto maximum hard harm",
        ),
        "minimum_valid_success_gain": _bounded(
            data["minimum_valid_success_gain"],
            -1.0,
            1.0,
            label="Pareto minimum gain",
        ),
        "benefit_weight": _positive_number(
            data["benefit_weight"],
            label="Pareto benefit weight",
        ),
        "harm_penalty": _nonnegative_number(
            data["harm_penalty"],
            label="Pareto harm penalty",
        ),
        "cost_penalty": _nonnegative_number(
            data["cost_penalty"],
            label="Pareto cost penalty",
        ),
    }
    if data["tie_break"] != "pareto_layer,-utility,-benefit,harm,cost,-support,release_id":
        raise ValidationError("unsupported selector Pareto tie-break")
    parsed["rule_hash"] = content_hash(parsed)
    return parsed


def _pareto_ranked(
    candidates: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    remaining = [dict(item) for item in candidates]
    ranked: list[dict[str, Any]] = []
    layer = 0
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                _dominates(other, candidate) for other in remaining if other is not candidate
            )
        ]
        if not front:  # pragma: no cover - finite partial order always has a front
            raise AssertionError("Pareto ordering produced no front")
        for candidate in front:
            candidate["pareto_layer"] = layer
            candidate["utility_score"] = (
                rule["benefit_weight"] * candidate["mean_valid_success_gain"]
                - rule["harm_penalty"] * candidate["hard_harm_rate"]
                - rule["cost_penalty"] * candidate["normalized_cost"]
            )
        front.sort(
            key=lambda item: (
                -item["utility_score"],
                -item["mean_valid_success_gain"],
                item["hard_harm_rate"],
                item["normalized_cost"],
                -item["support_fraction"],
                item["release_id"],
            )
        )
        ranked.extend(front)
        front_ids = {id(item) for item in front}
        remaining = [item for item in remaining if id(item) not in front_ids]
        layer += 1
    return ranked


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        left["mean_valid_success_gain"] >= right["mean_valid_success_gain"]
        and left["hard_harm_rate"] <= right["hard_harm_rate"]
        and left["normalized_cost"] <= right["normalized_cost"]
    )
    strict = (
        left["mean_valid_success_gain"] > right["mean_valid_success_gain"]
        or left["hard_harm_rate"] < right["hard_harm_rate"]
        or left["normalized_cost"] < right["normalized_cost"]
    )
    return no_worse and strict


def _summary_for_hash(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        **{key: item for key, item in summary.items() if key != "candidates"},
        "candidates": [
            {
                **candidate,
                "family": candidate["family"].value,
            }
            for candidate in summary["candidates"]
        ],
    }


def _validated_campaign_design(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label="selector campaign design",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "campaign_id",
            "primary_metric",
            "evaluation_split",
            "suite_ids",
            "required_independent_groups",
            "minimum_common_support_groups",
            "random_seed",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _DESIGN_ARTIFACT
        or data["stage"] != "frozen"
    ):
        raise ValidationError("selector campaign design must use frozen schema v1")
    parsed = {
        **data,
        "campaign_id": _identifier(data["campaign_id"], label="selector campaign ID"),
        "primary_metric": _identifier(
            data["primary_metric"],
            label="selector campaign primary metric",
        ),
        "evaluation_split": _identifier(
            data["evaluation_split"],
            label="selector campaign evaluation split",
        ),
        "suite_ids": _identifier_sequence(
            data["suite_ids"],
            label="selector campaign suites",
        ),
        "required_independent_groups": _positive_integer(
            data["required_independent_groups"],
            label="selector campaign required independent groups",
        ),
        "minimum_common_support_groups": _positive_integer(
            data["minimum_common_support_groups"],
            label="selector campaign minimum common-support groups",
        ),
        "random_seed": _nonnegative_integer(
            data["random_seed"],
            label="selector campaign random seed",
        ),
    }
    if (
        not parsed["suite_ids"]
        or parsed["minimum_common_support_groups"] > parsed["required_independent_groups"]
    ):
        raise ValidationError("selector campaign support design is invalid")
    return parsed


def _validated_campaign_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label="selector campaign bindings",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "representative_manifest_hash",
            "representative_rank",
            "best_fixed_global",
            "selectors",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _BINDINGS_ARTIFACT
        or data["stage"] != "frozen"
    ):
        raise ValidationError("selector campaign bindings must use frozen schema v1")
    representative_manifest_hash = _sha256_value(
        data["representative_manifest_hash"],
        label="selector campaign representative manifest",
    )
    representative_rank = _positive_integer(
        data["representative_rank"],
        label="selector campaign representative rank",
    )
    if representative_rank != 1:
        raise ValidationError("selector campaign currently requires rank-1 family policies")
    best = _strict_mapping(
        data["best_fixed_global"],
        label="selector best-fixed binding",
        keys={
            "selected_family",
            "selected_release_id",
            "policy_manifest_sha256",
        },
    )
    parsed_best = {
        "selected_family": _core_family(
            best["selected_family"],
            label="selector best-fixed family",
        ),
        "selected_release_id": _identifier(
            best["selected_release_id"],
            label="selector best-fixed release",
        ),
        "policy_manifest_sha256": _sha256_value(
            best["policy_manifest_sha256"],
            label="selector best-fixed policy manifest",
        ),
    }
    raw_selectors = _strict_mapping(
        data["selectors"],
        label="selector artifact bindings",
        keys={
            "defect_rule",
            "zero_shot_router",
            "penalized_t_learner",
            "gbdt_t_learner",
            "eligibility_aware_selector",
        },
    )
    selectors = {
        policy_id: _validated_artifact_binding(
            raw,
            label=f"selector artifact binding {policy_id}",
        )
        for policy_id, raw in raw_selectors.items()
    }
    expected_kinds = {
        "defect_rule": {"rule"},
        "zero_shot_router": {"zero_shot_router"},
        "penalized_t_learner": {"ridge_t_learner", "logistic_t_learner"},
        "gbdt_t_learner": {"gbdt_t_learner"},
        "eligibility_aware_selector": {
            "ridge_utility",
            "ridge_t_learner",
            "logistic_t_learner",
            "gbdt_t_learner",
            "knn_utility",
            "cost_risk_guardrail",
        },
    }
    for policy_id, allowed in expected_kinds.items():
        if selectors[policy_id]["selector_kind"] not in allowed:
            raise ValidationError(f"{policy_id} binding has an incompatible selector kind")
    return {
        **data,
        "representative_manifest_hash": representative_manifest_hash,
        "representative_rank": representative_rank,
        "best_fixed_global": parsed_best,
        "selectors": selectors,
    }


def _validated_ablation_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label="selector ablation bindings",
        keys={
            "schema_version",
            "artifact",
            "stage",
            "baseline_campaign_hash",
            "artifacts",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _ABLATION_BINDINGS_ARTIFACT
        or data["stage"] != "frozen"
    ):
        raise ValidationError("selector ablation bindings must use frozen schema v1")
    baseline_campaign_hash = _sha256_value(
        data["baseline_campaign_hash"],
        label="selector ablation baseline campaign",
    )
    artifacts = _strict_mapping(
        data["artifacts"],
        label="selector ablation artifacts",
        keys={
            "drop_ambiguity_features",
            "drop_multistep_features",
            "drop_retrieval_features",
            "drop_context_features",
            "drop_reformulation_features",
            "without_eligibility_mask",
            "without_cost_gate",
            "without_harm_calibration",
        },
    )
    return {
        **data,
        "baseline_campaign_hash": baseline_campaign_hash,
        "artifacts": {
            policy_id: _validated_artifact_binding(
                raw,
                label=f"selector ablation artifact {policy_id}",
            )
            for policy_id, raw in artifacts.items()
        },
    }


def _validated_ablation_matrix(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    data = _strict_mapping(
        value,
        label="selector ablation matrix",
        keys={"schema_version", "artifact", "stage", "entries"},
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != _ABLATION_MATRIX_ARTIFACT
        or data["stage"] != "frozen"
    ):
        raise ValidationError("selector ablation matrix must use frozen schema v1")
    expected = {
        "drop_ambiguity_features": ("feature_group.ambiguity", True, False),
        "drop_multistep_features": ("feature_group.multi_step", True, False),
        "drop_retrieval_features": ("feature_group.retrieval_need", True, False),
        "drop_context_features": ("feature_group.context_pressure", True, False),
        "drop_reformulation_features": (
            "feature_group.reformulation_need",
            True,
            False,
        ),
        "without_eligibility_mask": ("eligibility_mask", False, True),
        "without_cost_gate": ("cost_gate", False, False),
        "without_harm_calibration": ("harm_calibration", False, True),
    }
    parsed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_sequence(data["entries"], label="selector ablation entries")):
        row = _strict_mapping(
            raw,
            label=f"selector ablation entry {index}",
            keys={
                "policy_id",
                "changed_component",
                "requires_independent_refit",
                "diagnostic_only",
            },
        )
        policy_id = _identifier(row["policy_id"], label="selector ablation policy ID")
        if policy_id in parsed:
            raise ValidationError("selector ablation policy IDs must be unique")
        parsed[policy_id] = {
            "policy_id": policy_id,
            "changed_component": _identifier(
                row["changed_component"],
                label="selector ablation changed component",
            ),
            "requires_independent_refit": _boolean(
                row["requires_independent_refit"],
                label="selector ablation refit flag",
            ),
            "diagnostic_only": _boolean(
                row["diagnostic_only"],
                label="selector ablation diagnostic flag",
            ),
        }
    if set(parsed) != set(expected):
        raise ValidationError("selector ablation matrix differs from the frozen protocol")
    for policy_id, values in expected.items():
        row = parsed[policy_id]
        if (
            row["changed_component"],
            row["requires_independent_refit"],
            row["diagnostic_only"],
        ) != values:
            raise ValidationError(f"selector ablation entry {policy_id} differs from protocol")
    return tuple(parsed[policy_id] for policy_id in expected)


def _validated_artifact_binding(value: Any, *, label: str) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label=label,
        keys={"stage", "selector_kind", "selector_id", "revision", "sha256"},
    )
    if data["stage"] != "frozen":
        raise ValidationError(f"{label} must bind a frozen artifact")
    return {
        "stage": "frozen",
        "selector_kind": _identifier(data["selector_kind"], label=f"{label} kind"),
        "selector_id": _identifier(data["selector_id"], label=f"{label} ID"),
        "revision": _identifier(data["revision"], label=f"{label} revision"),
        "sha256": _sha256_value(data["sha256"], label=f"{label} SHA-256"),
    }


def _selected_releases(
    representatives: Mapping[str, Any],
) -> dict[SemanticFamily, tuple[str, str]]:
    selected: dict[SemanticFamily, list[tuple[int, str]]] = defaultdict(list)
    for row in representatives["selections"]:
        selected[SemanticFamily(row["family"])].append((row["rank"], row["release_id"]))
    return {
        family: tuple(release for _rank, release in sorted(rows))
        for family, rows in selected.items()
    }


def _campaign(
    *,
    campaign_id: str,
    campaign_kind: str,
    primary_metric: str,
    evaluation_split: str,
    representative_manifest_hash: str,
    policies: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = {
        "schema_version": 1,
        "artifact": _CAMPAIGN_ARTIFACT,
        "stage": "frozen",
        "campaign_id": campaign_id,
        "campaign_kind": campaign_kind,
        "primary_metric": primary_metric,
        "evaluation_split": evaluation_split,
        "representative_manifest_hash": representative_manifest_hash,
        "policies": list(policies),
        "metadata": dict(metadata),
        "experiments_executed": False,
    }
    result = freeze_mapping({**base, "campaign_hash": content_hash(base)})
    return validate_selector_campaign_manifest(result)


def _policy(
    policy_id: str,
    *,
    kind: str,
    role: str,
    formal_inference: bool,
    diagnostic_only: bool,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "kind": kind,
        "role": role,
        "formal_inference": formal_inference,
        "diagnostic_only": diagnostic_only,
        "parameters": dict(parameters),
    }


def _artifact_policy(
    policy_id: str,
    binding: Mapping[str, Any],
    *,
    role: str,
    formal_inference: bool,
) -> dict[str, Any]:
    return _policy(
        policy_id,
        kind="frozen_artifact_policy",
        role=role,
        formal_inference=formal_inference,
        diagnostic_only=False,
        parameters={"artifact_binding": dict(binding)},
    )


def _validated_policy(value: Any, *, label: str) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label=label,
        keys={
            "policy_id",
            "kind",
            "role",
            "formal_inference",
            "diagnostic_only",
            "parameters",
        },
    )
    policy_id = _identifier(data["policy_id"], label=f"{label} ID")
    kind = _identifier(data["kind"], label=f"{label} kind")
    role = _identifier(data["role"], label=f"{label} role")
    formal = _boolean(data["formal_inference"], label=f"{label} formal inference")
    diagnostic = _boolean(data["diagnostic_only"], label=f"{label} diagnostic flag")
    if formal and diagnostic:
        raise ValidationError(f"{label} cannot be both formal and diagnostic-only")
    return {
        "policy_id": policy_id,
        "kind": kind,
        "role": role,
        "formal_inference": formal,
        "diagnostic_only": diagnostic,
        "parameters": freeze_mapping(_mapping(data["parameters"], label=f"{label} parameters")),
    }


def _validated_result_record(value: Any, *, label: str) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        label=label,
        keys={
            "group_id",
            "suite_id",
            "policy_id",
            "selected_family",
            "selected_release_id",
            "value",
            "no_op_value",
            "hard_harm",
            "cost",
            "intervened",
            "selection_eligible",
            "common_support",
            "diagnostic_only",
        },
    )
    group_id = _require_text(data["group_id"], label=f"{label} group ID")
    if _GROUP_ID.fullmatch(group_id) is None:
        raise ValidationError(f"{label} group ID must be a privacy-safe hash")
    family = _family(data["selected_family"], label=f"{label} selected family")
    release = _require_text(data["selected_release_id"], label=f"{label} selected release")
    if family is SemanticFamily.NO_OP and release != "baseline.no_op":
        raise ValidationError(f"{label} NO_OP selection requires baseline.no_op")
    if family is not SemanticFamily.NO_OP and release == "baseline.no_op":
        raise ValidationError(f"{label} intervention selection cannot use baseline.no_op")
    intervened = _boolean(data["intervened"], label=f"{label} intervened")
    if intervened is (family is SemanticFamily.NO_OP):
        raise ValidationError(f"{label} intervention flag differs from its selected family")
    return {
        "group_id": group_id,
        "suite_id": _identifier(data["suite_id"], label=f"{label} suite"),
        "policy_id": _identifier(data["policy_id"], label=f"{label} policy"),
        "selected_family": family.value,
        "selected_release_id": release,
        "value": _bounded(data["value"], 0.0, 1.0, label=f"{label} value"),
        "no_op_value": _bounded(
            data["no_op_value"],
            0.0,
            1.0,
            label=f"{label} NO_OP value",
        ),
        "hard_harm": _boolean(data["hard_harm"], label=f"{label} hard harm"),
        "cost": _nonnegative_number(data["cost"], label=f"{label} cost"),
        "intervened": intervened,
        "selection_eligible": _boolean(
            data["selection_eligible"],
            label=f"{label} selection eligibility",
        ),
        "common_support": _boolean(
            data["common_support"],
            label=f"{label} common support",
        ),
        "diagnostic_only": _boolean(
            data["diagnostic_only"],
            label=f"{label} diagnostic flag",
        ),
    }


def _validated_source_runs(value: Any) -> tuple[dict[str, str], ...]:
    rows = _sequence(value, label="selector verified source runs")
    parsed: list[dict[str, str]] = []
    identities: set[str] = set()
    for index, raw in enumerate(rows):
        row = _strict_mapping(
            raw,
            label=f"selector source run {index}",
            keys={
                "study_id",
                "run_identity_hash",
                "manifest_hash",
                "plan_hash",
                "result_set_hash",
            },
        )
        parsed_row = {
            "study_id": _identifier(row["study_id"], label="selector source study ID"),
            "run_identity_hash": _sha256_value(
                row["run_identity_hash"],
                label="selector run identity",
            ),
            "manifest_hash": _sha256_value(
                row["manifest_hash"],
                label="selector source manifest",
            ),
            "plan_hash": _sha256_value(row["plan_hash"], label="selector source plan"),
            "result_set_hash": _sha256_value(
                row["result_set_hash"],
                label="selector result set",
            ),
        }
        if parsed_row["run_identity_hash"] in identities:
            raise ValidationError("selector source run identities must be unique")
        identities.add(parsed_row["run_identity_hash"])
        parsed.append(parsed_row)
    return tuple(sorted(parsed, key=lambda item: item["run_identity_hash"]))


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> Mapping[str, float]:
    if not values:
        raise ValidationError("selector bootstrap requires values")
    estimates = sorted(
        fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)
    )
    return {
        "estimate": fmean(values),
        "low": _quantile(estimates, 0.025),
        "high": _quantile(estimates, 0.975),
        "method": "independent_group_bootstrap",
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _strict_mapping(
    value: Any,
    *,
    label: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object")
    data = dict(value)
    if set(data) != keys:
        raise ValidationError(
            f"{label} fields differ: expected {sorted(keys)!r}, got {sorted(data)!r}"
        )
    return data


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    return tuple(value)


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    rows = _sequence(value, label=label)
    if any(not isinstance(item, str) or not item.strip() for item in rows):
        raise ValidationError(f"{label} must contain non-empty strings")
    return tuple(rows)


def _identifier_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    rows = tuple(_identifier(item, label=f"{label} item") for item in _sequence(value, label=label))
    if len(rows) != len(set(rows)):
        raise ValidationError(f"{label} must be unique")
    return rows


def _identifier(value: Any, *, label: str) -> str:
    text = _require_text(value, label=label)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{label} must be a lowercase stable identifier")
    return text


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _sha256_value(value: Any, *, label: str) -> str:
    _require_sha256(value, label=label)
    return value


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")


def _core_family(value: Any, *, label: str) -> SemanticFamily:
    family = _family(value, label=label)
    if family not in _CORE_FAMILIES:
        raise ValidationError(f"{label} must be a core intervention family")
    return family


def _family(value: Any, *, label: str) -> SemanticFamily:
    try:
        return SemanticFamily(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be a semantic family") from exc


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{label} must be boolean")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result < 1:
        raise ValidationError(f"{label} must be positive")
    return result


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{label} must be finite")
    return float(value)


def _bounded(value: Any, lower: float, upper: float, *, label: str) -> float:
    result = _finite(value, label=label)
    if not lower <= result <= upper:
        raise ValidationError(f"{label} must be in [{lower:g}, {upper:g}]")
    return result


def _positive_number(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0:
        raise ValidationError(f"{label} must be positive")
    return result


def _nonnegative_number(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0:
        raise ValidationError(f"{label} must be non-negative")
    return result


__all__ = [
    "analyze_selector_campaign_results",
    "build_selector_ablation_campaign",
    "build_selector_baseline_campaign",
    "build_selector_loso_manifest",
    "freeze_pareto_representatives",
    "load_json_mapping",
    "validate_representative_manifest",
    "validate_selector_campaign_design",
    "validate_selector_campaign_manifest",
    "validate_selector_campaign_results",
    "validate_selector_loso_manifest",
]
