"""One-command orchestration over strict AgentInputLab experiment jobs.

The runner composes existing experiment manifests and drivers.  It does not
weaken their prerequisite, binding, unit-inventory, guarded-driver, or result
verification contracts.  A complete campaign is preflighted before the first
job executes so missing assets are reported together instead of after partial
model spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import AgentInputLabError, ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.experiments import (
    ExperimentArmSpec,
    ExperimentBenchmarkTarget,
    ExperimentManifest,
    ExperimentPrerequisite,
    ExperimentStage,
    PrerequisiteKind,
    activate_guarded_driver,
    build_experiment_plan,
    execute_experiment_plan,
    load_experiment_manifest,
    preflight_experiment,
    verify_experiment_run_directory,
)
from agentinputlab.experiments.plan import ExperimentPlan
from agentinputlab.experiments.secrets import redact_secret_text, require_secret_free
from agentinputlab.registry import AlgorithmRegistry

ONE_CLICK_CAMPAIGN_PROTOCOL = "agentinputlab.one-click-campaign.v1"

_MAX_CONFIG_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_NAME = re.compile(
    r"(?i)(?:^|_)(?:api_?key|authorization|bearer|credential|password|secret|token)(?:_|$)"
)
_REQUIRED_DATA_CELLS = frozenset(
    {
        "beir.arguana",
        "bright.earth_science",
        "bright.economics",
        "musique.answerable",
        "2wiki.multihop",
        "hotpotqa.distractor",
        "qasper.v0_3_dev",
        "clariq.main",
        "tau3.banking_knowledge",
        "bfcl.v3.simple_python",
    }
)
_AUXILIARY_STATIC_CELLS = frozenset(
    {
        "beir.arguana",
        "bright.earth_science",
        "bright.economics",
        "musique.answerable",
        "2wiki.multihop",
        "hotpotqa.distractor",
        "qasper.v0_3_dev",
    }
)
_REPRESENTATIVE_COVERAGE = "representative_27x10"
_AUXILIARY_COVERAGE = "auxiliary_27x7_legal"
_TAU3_REVISION = "cf71a8070269883e38a365ffa85f78f46844c1f4"
_METHOD_TARGETS = MappingProxyType(
    {
        "rewrite.query2doc": ("bfcl.v3.simple_python", "test"),
        "rewrite.rrr": ("beir.arguana", "test"),
        "rewrite.reformer": ("clariq.main", "dev"),
        "rewrite.infocqr": ("bright.economics", "test"),
        "plan.plan_and_solve": ("musique.answerable", "dev"),
        "plan.decomp": ("2wiki.multihop", "dev"),
        "plan.self_ask": ("hotpotqa.distractor", "validation"),
        "plan.llm_compiler": ("musique.answerable", "dev"),
        "plan.plan_and_budget": ("2wiki.multihop", "dev"),
        "acquire.hyde": ("bright.earth_science", "test"),
        "acquire.ircot": ("musique.answerable", "dev"),
        "acquire.adaptive_rag": ("hotpotqa.distractor", "validation"),
        "acquire.r2ag": ("beir.arguana", "test"),
        "acquire.qrag": ("2wiki.multihop", "dev"),
        "clarify.ask_or_assume": ("tau3.banking_knowledge", "discovery_train"),
        "clarify.udcs": ("tau3.banking_knowledge", "discovery_train"),
        "clarify.future_turns": ("tau3.banking_knowledge", "discovery_train"),
        "clarify.active_task_disambiguation": (
            "tau3.banking_knowledge",
            "discovery_train",
        ),
        "curate.longllmlingua": ("qasper.v0_3_dev", "dev"),
        "curate.llmlingua2": ("qasper.v0_3_dev", "dev"),
        "curate.recomp": ("qasper.v0_3_dev", "dev"),
        "curate.acon": ("qasper.v0_3_dev", "dev"),
        "recipe.smartsearch": ("tau3.banking_knowledge", "discovery_train"),
        "recipe.rqrag": ("tau3.banking_knowledge", "discovery_train"),
        "recipe.ace": ("qasper.v0_3_dev", "dev"),
        "recipe.mce": ("qasper.v0_3_dev", "dev"),
        "recipe.skillopt": ("qasper.v0_3_dev", "dev"),
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "campaign_id",
        "prepared_data_root",
        "data_cells",
        "output_root",
        "resume",
        "fail_fast",
        "jobs",
        "campaign_hash",
    }
)
_CONFIG_FIELDS_WITH_COVERAGE = _CONFIG_FIELDS | {"coverage_design"}
_JOB_FIELDS = frozenset(
    {
        "job_id",
        "method_ids",
        "benchmark_cells",
        "manifest_path",
        "manifest_sha256",
        "driver_entrypoint",
        "environment",
    }
)


@dataclass(frozen=True, slots=True)
class OneClickCampaignJob:
    """One strict experiment manifest inside a larger campaign."""

    job_id: str
    method_ids: tuple[str, ...]
    benchmark_cells: tuple[str, ...]
    manifest_path: Path
    manifest_sha256: str
    driver_entrypoint: str
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        _identifier(self.job_id, label="campaign job_id")
        methods = _identifier_tuple(self.method_ids, label=f"{self.job_id}.method_ids")
        cells = _identifier_tuple(
            self.benchmark_cells,
            label=f"{self.job_id}.benchmark_cells",
        )
        source = self.manifest_path
        if not isinstance(source, Path) or not source.is_absolute():
            raise ValidationError(f"{self.job_id}.manifest_path must be absolute")
        _require_sha256(self.manifest_sha256, label=f"{self.job_id}.manifest_sha256")
        if (
            not isinstance(self.driver_entrypoint, str)
            or _ENTRYPOINT.fullmatch(self.driver_entrypoint) is None
        ):
            raise ValidationError(f"{self.job_id}.driver_entrypoint must use module:factory syntax")
        environment: dict[str, str] = {}
        for name, value in self.environment.items():
            if (
                not isinstance(name, str)
                or _ENVIRONMENT_NAME.fullmatch(name) is None
                or _SECRET_NAME.search(name) is not None
            ):
                raise ValidationError(
                    f"{self.job_id}.environment contains an invalid or secret-bearing name"
                )
            if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
                raise ValidationError(f"{self.job_id}.environment values must be non-empty strings")
            environment[name] = value
        require_secret_free(environment, label=f"{self.job_id}.environment")
        object.__setattr__(self, "method_ids", methods)
        object.__setattr__(self, "benchmark_cells", cells)
        object.__setattr__(self, "environment", MappingProxyType(environment))


@dataclass(frozen=True, slots=True)
class OneClickCampaign:
    """Checksum-bound, exact-scope collection of experiment jobs."""

    campaign_id: str
    prepared_data_root: Path
    data_cells: tuple[str, ...]
    output_root: Path
    resume: bool
    fail_fast: bool
    jobs: tuple[OneClickCampaignJob, ...]
    campaign_hash: str
    source_path: Path
    source_sha256: str
    coverage_design: str = _REPRESENTATIVE_COVERAGE

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="campaign_id")
        for label, path in (
            ("prepared_data_root", self.prepared_data_root),
            ("output_root", self.output_root),
            ("source_path", self.source_path),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValidationError(f"campaign {label} must be absolute")
        if not isinstance(self.resume, bool) or not isinstance(self.fail_fast, bool):
            raise ValidationError("campaign resume and fail_fast must be boolean")
        if self.coverage_design not in {
            _REPRESENTATIVE_COVERAGE,
            _AUXILIARY_COVERAGE,
        }:
            raise ValidationError("one-click campaign coverage_design is unsupported")
        cells = _identifier_tuple(self.data_cells, label="campaign data_cells")
        expected_cells = (
            _REQUIRED_DATA_CELLS
            if self.coverage_design == _REPRESENTATIVE_COVERAGE
            else _AUXILIARY_STATIC_CELLS
        )
        if set(cells) != expected_cells:
            raise ValidationError(
                "one-click campaign data_cells differ from the declared coverage design"
            )
        jobs = tuple(self.jobs)
        if not jobs or any(not isinstance(job, OneClickCampaignJob) for job in jobs):
            raise ValidationError("one-click campaign requires typed jobs")
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValidationError("one-click campaign job IDs must be unique")
        methods = {item.method_id for item in AlgorithmRegistry.default().list_methods()}
        observed_methods = {method_id for job in jobs for method_id in job.method_ids}
        observed_cells = {cell for job in jobs for cell in job.benchmark_cells}
        if self.coverage_design == _REPRESENTATIVE_COVERAGE:
            if observed_methods != methods:
                raise ValidationError(
                    "representative one-click jobs must cover the exact 27-method registry"
                )
            if observed_cells != _REQUIRED_DATA_CELLS:
                raise ValidationError(
                    "representative one-click jobs must cover the exact ten cells"
                )
        else:
            from agentinputlab.experiments.auxiliary_matrix import (
                build_auxiliary_method_cell_matrix,
            )

            expected_pairs = {
                (row["method_id"], row["benchmark_cell_id"])
                for row in build_auxiliary_method_cell_matrix()["rows"]
                if row["eligible"]
            }
            observed_pairs: set[tuple[str, str]] = set()
            for job in jobs:
                if len(job.method_ids) != 1 or len(job.benchmark_cells) != 1:
                    raise ValidationError(
                        "auxiliary matrix jobs must bind exactly one method and one cell"
                    )
                observed_pairs.add((job.method_ids[0], job.benchmark_cells[0]))
            if observed_pairs != expected_pairs or len(jobs) != 117:
                raise ValidationError(
                    "auxiliary one-click jobs must cover the exact 117 legal pairs"
                )
            if observed_cells != _AUXILIARY_STATIC_CELLS:
                raise ValidationError(
                    "auxiliary one-click jobs must cover the exact seven static cells"
                )
        _require_sha256(self.campaign_hash, label="campaign_hash")
        _require_sha256(self.source_sha256, label="campaign source_sha256")
        object.__setattr__(self, "data_cells", cells)
        object.__setattr__(self, "jobs", jobs)


@dataclass(frozen=True, slots=True)
class _PreparedJob:
    job: OneClickCampaignJob
    manifest: ExperimentManifest
    plan: ExperimentPlan | None
    preflight: Any
    environment: Mapping[str, str]


def campaign_bundle_hash(value: Mapping[str, Any]) -> str:
    """Return the semantic hash for a strict bundle payload."""

    if not isinstance(value, Mapping):
        raise TypeError("campaign bundle must be a mapping")
    return content_hash({key: item for key, item in value.items() if key != "campaign_hash"})


def one_click_method_targets() -> Mapping[str, tuple[str, str]]:
    """Return the exact 27-method coverage map over ten accepted-source cells."""

    methods = {item.method_id for item in AlgorithmRegistry.default().list_methods()}
    if set(_METHOD_TARGETS) != methods:
        raise ValidationError("one-click target map differs from the exact algorithm registry")
    if {cell for cell, _split in _METHOD_TARGETS.values()} != _REQUIRED_DATA_CELLS:
        raise ValidationError("one-click target map does not cover the exact ten data cells")
    return _METHOD_TARGETS


def build_one_click_method_manifest(
    *,
    method_id: str,
    release_id: str,
    family: str,
    profile_kind: str,
    upstream_commit: str,
    driver_entrypoint: str,
    seeds: Sequence[int],
    model_slots: Sequence[str],
    target_cell: str | None = None,
    target_split: str | None = None,
    study_id: str | None = None,
) -> ExperimentManifest:
    """Build one candidate-versus-NO_OP discovery manifest for the campaign."""

    targets = one_click_method_targets()
    try:
        default_cell, default_split = targets[method_id]
        method, release = AlgorithmRegistry.default().get_release(release_id)
        semantic_family = SemanticFamily(family)
        profile = ProfileKind(profile_kind)
    except (KeyError, ValueError) as exc:
        raise ValidationError("invalid one-click method manifest identity") from exc
    if (
        method.method_id != method_id
        or method.semantic_family is not semantic_family
        or method.upstream.commit != upstream_commit
        or profile not in release.profile_modes
    ):
        raise ValidationError("one-click method manifest differs from the algorithm registry")
    selected_cell = default_cell if target_cell is None else target_cell
    selected_split = default_split if target_split is None else target_split
    if not isinstance(selected_cell, str) or not selected_cell.strip():
        raise ValidationError("one-click target_cell must be non-empty")
    if not isinstance(selected_split, str) or not selected_split.strip():
        raise ValidationError("one-click target_split must be non-empty")
    external = semantic_family in {SemanticFamily.CLARIFY, SemanticFamily.COMPOSITE}
    prerequisites = [
        ExperimentPrerequisite(
            PrerequisiteKind.DIRECTORY_ENV,
            "AILAB_PREPARED_DATA_ROOT",
            "Checksum-verified prepared benchmark root",
        ),
        ExperimentPrerequisite(
            PrerequisiteKind.DIRECTORY_ENV,
            "AILAB_METHOD_CHECKOUT",
            "Clean checkout of the method source pinned by the catalog",
        ),
        ExperimentPrerequisite(
            PrerequisiteKind.DIRECTORY_ENV,
            "AILAB_CANDIDATE_CHECKOUT",
            "Clean AgentInputLab checkout used for the candidate runtime",
        ),
        ExperimentPrerequisite(
            PrerequisiteKind.FILE_ENV,
            "AILAB_METHOD_ASSET_MANIFEST",
            "Checksum-locked weights, prompts, worker, and auxiliary assets",
            sha256_env="AILAB_METHOD_ASSET_MANIFEST_SHA256",
        ),
        ExperimentPrerequisite(
            PrerequisiteKind.FILE_ENV,
            "AILAB_MODEL_CONFIG",
            "Frozen backend/model configuration",
            sha256_env="AILAB_MODEL_CONFIG_SHA256",
        ),
        ExperimentPrerequisite(
            PrerequisiteKind.FILE_ENV,
            "AILAB_RETRIEVER_CONFIG",
            "Frozen corpus/index/embedder/retriever/reranker configuration",
            sha256_env="AILAB_RETRIEVER_CONFIG_SHA256",
        ),
    ]
    if driver_entrypoint == "agentinputlab.experiments.drivers.static_production:factory":
        prerequisites.extend(
            (
                ExperimentPrerequisite(
                    PrerequisiteKind.FILE_ENV,
                    "AILAB_PREDICTOR_CONFIG",
                    "Frozen downstream predictor configuration",
                    sha256_env="AILAB_PREDICTOR_CONFIG_SHA256",
                ),
                ExperimentPrerequisite(
                    PrerequisiteKind.FILE_ENV,
                    "AILAB_STATIC_PRODUCTION_DRIVER_CONFIG",
                    "Generated checksum-locked static method-dataset driver",
                    sha256_env="AILAB_STATIC_PRODUCTION_DRIVER_CONFIG_SHA256",
                ),
            )
        )
    checkout_revisions = {"AILAB_METHOD_CHECKOUT": upstream_commit}
    if external:
        prerequisites.append(
            ExperimentPrerequisite(
                PrerequisiteKind.DIRECTORY_ENV,
                "AILAB_BENCHMARK_CHECKOUT",
                "Clean tau3 checkout used by the atomic discovery runtime",
            )
        )
        checkout_revisions["AILAB_BENCHMARK_CHECKOUT"] = _TAU3_REVISION
    return ExperimentManifest(
        schema_version=1,
        study_id=study_id or f"one_click.{method_id}",
        stage=ExperimentStage.DISCOVERY,
        estimand_profile_kind=profile,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id=selected_cell,
                split=selected_split,
                unit_inventory_env="AILAB_METHOD_DISCOVERY_UNITS",
                unit_inventory_sha256_env="AILAB_METHOD_DISCOVERY_UNITS_SHA256",
                external=external,
                source_revision=_TAU3_REVISION if external else None,
                arm_ids=("no_op", "candidate"),
            ),
        ),
        arms=(
            ExperimentArmSpec(
                arm_id="no_op",
                release_id="baseline.no_op",
                family=SemanticFamily.NO_OP,
                profile_kind=profile,
            ),
            ExperimentArmSpec(
                arm_id="candidate",
                release_id=release_id,
                family=semantic_family,
                profile_kind=profile,
                parameters={"method_id": method_id},
            ),
        ),
        seeds=tuple(seeds),
        model_slots=tuple(model_slots),
        prerequisites=tuple(prerequisites),
        driver_entrypoint=driver_entrypoint,
        metadata={
            "method_id": method_id,
            "candidate_release_id": release_id,
            "checkout_expected_revisions": checkout_revisions,
            "comparison_scope": "one_click_capability_specific_method_vs_no_op",
            "cross_method_raw_ranking_allowed": False,
            "confirmatory_claimed": False,
            "execution_default": "preflight_only",
            "require_driver_unit_preflight": True,
            "requires_atomic_external_replay": external,
        },
    )


def load_one_click_campaign(path: str | Path) -> OneClickCampaign:
    """Load and strictly validate one immutable one-click campaign bundle."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ArtifactError("one-click campaign config must not be a symlink")
    source = unresolved.resolve()
    try:
        if source.stat().st_size > _MAX_CONFIG_BYTES:
            raise ArtifactError("one-click campaign config exceeds the size limit")
        payload = source.read_bytes()
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("one-click campaign config is not strict UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("one-click campaign must be an object")
    fields = frozenset(raw)
    if fields == _CONFIG_FIELDS:
        coverage_design = _REPRESENTATIVE_COVERAGE
    elif fields == _CONFIG_FIELDS_WITH_COVERAGE:
        coverage_design = raw["coverage_design"]
    else:
        raise ValidationError(
            "one-click campaign fields must match the representative or auxiliary schema"
        )
    data = raw
    if data["schema_version"] != 1 or data["protocol"] != ONE_CLICK_CAMPAIGN_PROTOCOL:
        raise ValidationError("unsupported one-click campaign protocol")
    expected_hash = campaign_bundle_hash(data)
    if data["campaign_hash"] != expected_hash:
        raise ValidationError("one-click campaign hash differs from its content")
    jobs_raw = data["jobs"]
    if isinstance(jobs_raw, (str, bytes)) or not isinstance(jobs_raw, Sequence):
        raise ValidationError("one-click campaign jobs must be an array")
    jobs = tuple(_parse_job(item, index=index) for index, item in enumerate(jobs_raw))
    return OneClickCampaign(
        campaign_id=_identifier(data["campaign_id"], label="campaign_id"),
        prepared_data_root=_absolute_path(
            data["prepared_data_root"],
            label="prepared_data_root",
        ),
        data_cells=tuple(data["data_cells"]),
        output_root=_absolute_path(data["output_root"], label="output_root"),
        resume=_boolean(data["resume"], label="resume"),
        fail_fast=_boolean(data["fail_fast"], label="fail_fast"),
        jobs=jobs,
        campaign_hash=expected_hash,
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        coverage_design=coverage_design,
    )


def preflight_one_click_campaign(
    campaign: OneClickCampaign,
) -> tuple[Mapping[str, Any], tuple[_PreparedJob, ...]]:
    """Preflight data and every job before any model or external runtime call."""

    if not isinstance(campaign, OneClickCampaign):
        raise TypeError("preflight requires a OneClickCampaign")
    data_report: Mapping[str, Any] | None = None
    data_blockers: list[str] = []
    try:
        if campaign.coverage_design == _REPRESENTATIVE_COVERAGE:
            from agentinputlab.campaign import campaign_data_smoke

            data_report = campaign_data_smoke(campaign.prepared_data_root)
            observed = set(data_report["accepted_source_cells"])
            if observed != _REQUIRED_DATA_CELLS:
                data_blockers.append(
                    "prepared data smoke did not cover the exact ten accepted-source cells"
                )
        else:
            from agentinputlab.benchmarks import load_production_benchmark_adapter

            rows = []
            for cell_id in sorted(_AUXILIARY_STATIC_CELLS):
                adapter = load_production_benchmark_adapter(
                    cell_id,
                    campaign.prepared_data_root,
                )
                rows.append(
                    {
                        "cell_id": cell_id,
                        "split": adapter.split,
                        "unit_count": len(tuple(adapter.examples(adapter.split))),
                    }
                )
            data_report = freeze_mapping(
                {
                    "schema_version": 1,
                    "protocol": "agentinputlab.auxiliary-static-data-smoke.v1",
                    "accepted_source_cells": tuple(sorted(_AUXILIARY_STATIC_CELLS)),
                    "cells": tuple(rows),
                }
            )
    except (AgentInputLabError, OSError, TypeError, ValueError) as exc:
        data_blockers.append(f"data gate: {type(exc).__name__}: {redact_secret_text(str(exc))}")

    prepared_jobs: list[_PreparedJob] = []
    job_rows: list[dict[str, Any]] = []
    for job in campaign.jobs:
        blockers: list[str] = []
        manifest: ExperimentManifest | None = None
        plan: ExperimentPlan | None = None
        report: Any = None
        environment = _job_environment(campaign, job)
        try:
            _verify_file(job.manifest_path, expected_sha256=job.manifest_sha256)
            manifest = load_experiment_manifest(job.manifest_path)
            _validate_job_manifest(job, manifest)
            with _temporary_environment(environment):
                report, units = preflight_experiment(
                    manifest,
                    driver_entrypoint=job.driver_entrypoint,
                    environment=environment,
                    resolve_units=True,
                )
            blockers.extend(report.blockers)
            if report.ready:
                plan = build_experiment_plan(manifest, units)
        except (AgentInputLabError, OSError, TypeError, ValueError) as exc:
            blockers.append(
                f"{type(exc).__name__}: "
                f"{redact_secret_text(str(exc), secret_values=_secret_values(environment))}"
            )
        secrets = _secret_values(environment)
        blockers = sorted(
            {redact_secret_text(blocker, secret_values=secrets) for blocker in blockers}
        )
        if manifest is not None and report is not None:
            prepared_jobs.append(
                _PreparedJob(
                    job=job,
                    manifest=manifest,
                    plan=plan,
                    preflight=report,
                    environment=freeze_mapping(environment),
                )
            )
        job_rows.append(
            {
                "job_id": job.job_id,
                "method_ids": job.method_ids,
                "benchmark_cells": job.benchmark_cells,
                "manifest_sha256": job.manifest_sha256,
                "driver_entrypoint": job.driver_entrypoint,
                "ready": not blockers and plan is not None,
                "blockers": tuple(blockers),
                "plan_hash": plan.plan_hash if plan is not None else None,
                "planned_cells": len(plan.cells) if plan is not None else 0,
            }
        )
    ready_jobs = sum(row["ready"] for row in job_rows)
    ready = (
        not data_blockers
        and data_report is not None
        and ready_jobs == len(campaign.jobs)
        and len(prepared_jobs) == len(campaign.jobs)
    )
    result = {
        "schema_version": 1,
        "protocol": "agentinputlab.one-click-campaign-preflight.v1",
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash,
        "source_sha256": campaign.source_sha256,
        "method_count": len({method_id for job in campaign.jobs for method_id in job.method_ids}),
        "method_dataset_pair_count": sum(
            len(job.method_ids) * len(job.benchmark_cells) for job in campaign.jobs
        ),
        "coverage_design": campaign.coverage_design,
        "full_cartesian": False,
        "full_legal_matrix": campaign.coverage_design == _AUXILIARY_COVERAGE,
        "declared_data_cell_count": len(campaign.data_cells),
        "data_cells": campaign.data_cells,
        "data_ready": not data_blockers,
        "data_blockers": tuple(data_blockers),
        "data_smoke": data_report,
        "job_count": len(campaign.jobs),
        "ready_jobs": ready_jobs,
        "blocked_jobs": len(campaign.jobs) - ready_jobs,
        "planned_cells": sum(row["planned_cells"] for row in job_rows),
        "jobs": tuple(job_rows),
        "ready": ready,
        "experiment_executed": False,
    }
    return freeze_mapping(result), tuple(prepared_jobs)


def run_one_click_campaign(
    campaign: OneClickCampaign,
    *,
    execute: bool,
) -> Mapping[str, Any]:
    """Preflight the whole bundle, then optionally execute and verify every job."""

    if not isinstance(execute, bool):
        raise TypeError("execute must be boolean")
    preflight, prepared_jobs = preflight_one_click_campaign(campaign)
    if not execute:
        return preflight
    if not preflight["ready"]:
        return freeze_mapping(
            {
                **dict(preflight),
                "experiment_executed": False,
                "execution_blocked": True,
            }
        )

    campaign_root = one_click_campaign_run_root(campaign)
    campaign_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for prepared in prepared_jobs:
        if prepared.plan is None or prepared.preflight.binding_receipt is None:
            raise ArtifactError("ready one-click job lacks a frozen plan or binding receipt")
        job_output = campaign_root / prepared.job.job_id
        resume_job = campaign.resume and job_output.exists()
        try:
            with (
                _temporary_environment(prepared.environment),
                activate_guarded_driver(
                    manifest=prepared.manifest,
                    operator_entrypoint=prepared.job.driver_entrypoint,
                    operator_fingerprint=(prepared.preflight.operator_driver_fingerprint),
                    binding_receipt=prepared.preflight.binding_receipt,
                    parity_contract=prepared.preflight.parity_contract,
                ) as guarded_entrypoint,
            ):
                summary = execute_experiment_plan(
                    prepared.manifest,
                    prepared.plan,
                    driver_entrypoint=guarded_entrypoint,
                    output_root=str(job_output),
                    resume=resume_job,
                    fail_fast=campaign.fail_fast,
                )
            verified = verify_experiment_run_directory(summary.run_dir)
            if not verified["complete"] or not verified["all_succeeded"]:
                raise ArtifactError("one-click job result verification did not pass")
            results.append(
                {
                    "job_id": prepared.job.job_id,
                    "status": "succeeded",
                    "run_dir": summary.run_dir,
                    "plan_hash": summary.plan_hash,
                    "planned_cells": summary.planned_cells,
                    "attempted_cells": summary.attempted_cells,
                    "skipped_cells": summary.skipped_cells,
                    "succeeded_cells": summary.succeeded_cells,
                    "failed_cells": summary.failed_cells,
                    "verified": True,
                }
            )
        except (AgentInputLabError, OSError, TypeError, ValueError) as exc:
            results.append(
                {
                    "job_id": prepared.job.job_id,
                    "status": "aborted",
                    "failure": {
                        "category": type(exc).__name__,
                        "message": redact_secret_text(
                            str(exc),
                            secret_values=_secret_values(prepared.environment),
                        ),
                    },
                    "verified": False,
                }
            )
            if campaign.fail_fast:
                break
    succeeded = sum(row["status"] == "succeeded" for row in results)
    all_succeeded = succeeded == len(campaign.jobs)
    summary_payload = {
        "schema_version": 1,
        "protocol": "agentinputlab.one-click-campaign-result.v1",
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash,
        "source_sha256": campaign.source_sha256,
        "method_count": preflight["method_count"],
        "method_dataset_pair_count": preflight["method_dataset_pair_count"],
        "coverage_design": preflight["coverage_design"],
        "full_cartesian": preflight["full_cartesian"],
        "data_cell_count": preflight["declared_data_cell_count"],
        "job_count": len(campaign.jobs),
        "attempted_jobs": len(results),
        "succeeded_jobs": succeeded,
        "failed_jobs": len(results) - succeeded,
        "all_succeeded": all_succeeded,
        "jobs": tuple(results),
        "experiment_executed": True,
    }
    summary_payload["result_hash"] = content_hash(summary_payload)
    _write_summary(campaign_root / "campaign-summary.json", summary_payload)
    return freeze_mapping(summary_payload)


def one_click_campaign_run_root(campaign: OneClickCampaign) -> Path:
    """Return the immutable output namespace for one frozen campaign.

    A human-readable campaign ID may be reused after code, manifests, or locked
    assets change.  The content hash therefore participates in the path so a
    new campaign can never accidentally resume an incompatible prior run.
    """

    if not isinstance(campaign, OneClickCampaign):
        raise TypeError("campaign run root requires a OneClickCampaign")
    return campaign.output_root / campaign.campaign_id / campaign.campaign_hash


def _parse_job(value: Any, *, index: int) -> OneClickCampaignJob:
    data = _strict_mapping(value, _JOB_FIELDS, label=f"jobs[{index}]")
    environment = data["environment"]
    if not isinstance(environment, Mapping):
        raise ValidationError(f"jobs[{index}].environment must be an object")
    return OneClickCampaignJob(
        job_id=data["job_id"],
        method_ids=tuple(data["method_ids"]),
        benchmark_cells=tuple(data["benchmark_cells"]),
        manifest_path=_absolute_path(
            data["manifest_path"],
            label=f"jobs[{index}].manifest_path",
        ),
        manifest_sha256=data["manifest_sha256"],
        driver_entrypoint=data["driver_entrypoint"],
        environment=environment,
    )


def _validate_job_manifest(
    job: OneClickCampaignJob,
    manifest: ExperimentManifest,
) -> None:
    manifest_cells = {target.cell_id for target in manifest.benchmarks}
    if manifest_cells != set(job.benchmark_cells):
        raise ValidationError(f"{job.job_id}: declared benchmark_cells differ from the manifest")
    if not manifest_cells.issubset(_REQUIRED_DATA_CELLS):
        raise ValidationError(f"{job.job_id}: manifest touches a frozen or undeclared data cell")
    registry = AlgorithmRegistry.default()
    manifest_methods: set[str] = set()
    for arm in manifest.arms:
        if arm.release_id == "baseline.no_op":
            continue
        try:
            method, _release = registry.get_release(arm.release_id)
        except Exception as exc:
            raise ValidationError(
                f"{job.job_id}: cannot resolve release {arm.release_id!r}"
            ) from exc
        manifest_methods.add(method.method_id)
    if manifest_methods != set(job.method_ids):
        raise ValidationError(f"{job.job_id}: method_ids differ from manifest release arms")


def _job_environment(
    campaign: OneClickCampaign,
    job: OneClickCampaignJob,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(job.environment)
    prepared = str(campaign.prepared_data_root)
    current = environment.get("AILAB_PREPARED_DATA_ROOT")
    if current is not None and Path(current).expanduser().resolve() != campaign.prepared_data_root:
        raise ValidationError(
            f"{job.job_id}: AILAB_PREPARED_DATA_ROOT differs from campaign prepared_data_root"
        )
    environment["AILAB_PREPARED_DATA_ROOT"] = prepared
    return environment


@contextmanager
def _temporary_environment(environment: Mapping[str, str]) -> Iterator[None]:
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _secret_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        value
        for name, value in environment.items()
        if _SECRET_NAME.search(name) is not None and isinstance(value, str) and value
    )


def _write_summary(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(path)
    except OSError as exc:
        raise ArtifactError("cannot persist one-click campaign summary") from exc


def _verify_file(path: Path, *, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("campaign manifest must be an existing non-symlink file")
    if _sha256_file(path) != expected_sha256:
        raise ArtifactError("campaign manifest checksum differs")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot hash campaign file: {path}") from exc
    return digest.hexdigest()


def _strict_mapping(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidationError(f"{label} fields must be exactly: {', '.join(sorted(fields))}")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a stable identifier")
    return value


def _identifier_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be an array")
    result = tuple(_identifier(item, label=label) for item in value)
    if not result or len(result) != len(set(result)):
        raise ValidationError(f"{label} requires unique identifiers")
    return result


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValidationError(f"{label} must be absolute")
    return path.resolve()


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{label} must be boolean")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON value is not allowed: {value}")


__all__ = [
    "ONE_CLICK_CAMPAIGN_PROTOCOL",
    "OneClickCampaign",
    "OneClickCampaignJob",
    "build_one_click_method_manifest",
    "campaign_bundle_hash",
    "load_one_click_campaign",
    "one_click_method_targets",
    "preflight_one_click_campaign",
    "run_one_click_campaign",
]
