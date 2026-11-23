"""Opt-in, allowlisted research-campaign orchestration for Studio.

The ordinary Studio run surface remains a tiny deterministic offline tool.
This module adds a separate operator-controlled path for the shipped experiment
blueprints.  A browser can select only a profile ID from a catalog loaded at
server startup; it cannot submit paths, entrypoints, environment values,
commands, credentials, or arbitrary execution options.

Campaign preparation is deliberately the default action.  It loads the locked
blueprint, runs the existing experiment preflight, and materializes an exact
plan without executing a cell.  Execution requires a second request that
cross-binds the profile, preparation ID, plan hash, and a fixed confirmation
phrase.  Immediately before execution the worker repeats file attestation,
preflight, and planning, then uses the existing guarded driver and experiment
executor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.experiments import (
    ExperimentArtifactIdentity,
    ExperimentManifest,
    ExperimentPlan,
    PreflightReport,
    activate_guarded_driver,
    build_experiment_plan,
    execute_experiment_plan,
    load_experiment_blueprint,
    preflight_experiment,
)

CAMPAIGN_CONFIRMATION = "execute_frozen_plan"
MAX_CAMPAIGN_CATALOG_BYTES = 2 * 1024 * 1024
MAX_CAMPAIGN_PROFILES = 32
MAX_CAMPAIGN_PREPARATIONS = 100
MAX_CAMPAIGN_EXECUTIONS = 100
MAX_PLAN_PREVIEW_CELLS = 100

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOGGER = logging.getLogger(__name__)
_EXTERNAL_ARENA_ENTRYPOINT = "agentinputlab.experiments.drivers.external_arena:factory"
_TYPED_GATE_TARGETS = frozenset(
    {
        "toolsandbox.common_support",
        "appworld_ul.common_support",
        "webarena.external",
    }
)


class CampaignSecurityError(ValueError):
    """A browser attempted to cross the fixed campaign request boundary."""


class CampaignValidationError(ValueError):
    """A campaign request is well-formed but cannot be accepted."""


class CampaignQueueFullError(RuntimeError):
    """The bounded in-memory campaign queue cannot accept more records."""


@dataclass(frozen=True, slots=True)
class _LockedCampaignFile:
    path: Path
    sha256: str
    label: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValidationError(f"{self.label} path must resolve to an absolute path")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValidationError(f"{self.label} sha256 must be lowercase SHA-256")
        self.verify()

    def verify(self) -> None:
        try:
            if not self.path.is_file():
                raise ArtifactError(f"configured {self.label} is not an existing file")
            observed = _sha256_file(self.path)
        except OSError as exc:
            raise ArtifactError(f"configured {self.label} cannot be read") from exc
        if observed != self.sha256:
            raise ArtifactError(f"configured {self.label} checksum mismatch")


@dataclass(frozen=True, slots=True)
class _CampaignProfile:
    profile_id: str
    label: str
    description: str
    blueprint: _LockedCampaignFile
    arm_catalog: _LockedCampaignFile
    driver_entrypoint: str
    execution_enabled: bool

    def __post_init__(self) -> None:
        if _PROFILE_ID.fullmatch(self.profile_id) is None:
            raise ValidationError("campaign profile id is invalid")
        for value, label, maximum in (
            (self.label, "label", 160),
            (self.description, "description", 800),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                or len(value) > maximum
            ):
                raise ValidationError(
                    f"campaign profile {label} must be trimmed non-empty text "
                    f"of at most {maximum} characters"
                )
        if (
            not isinstance(self.driver_entrypoint, str)
            or _ENTRYPOINT.fullmatch(self.driver_entrypoint) is None
        ):
            raise ValidationError("campaign profile driver entrypoint is invalid")
        if not isinstance(self.execution_enabled, bool):
            raise ValidationError("campaign profile execution_enabled must be boolean")

    @property
    def identity_hash(self) -> str:
        return content_hash(
            {
                "profile_id": self.profile_id,
                "blueprint_sha256": self.blueprint.sha256,
                "arm_catalog_sha256": self.arm_catalog.sha256,
                "driver_entrypoint": self.driver_entrypoint,
                "execution_enabled": self.execution_enabled,
            }
        )

    def verify_files(self) -> None:
        self.blueprint.verify()
        self.arm_catalog.verify()

    def load_manifest(self) -> ExperimentManifest:
        # Check both sides of loading so a file replacement cannot silently
        # survive the startup lock or a preparation/execution boundary.
        self.verify_files()
        manifest = load_experiment_blueprint(
            self.blueprint.path,
            arm_catalog_path=self.arm_catalog.path,
        )
        self.verify_files()
        return manifest

    def public_payload(self, *, output_configured: bool) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "label": self.label,
            "description": self.description,
            "profile_identity_sha256": self.identity_hash,
            "blueprint_sha256": self.blueprint.sha256,
            "arm_catalog_sha256": self.arm_catalog.sha256,
            "execution_enabled": self.execution_enabled,
            "execution_configured": self.execution_enabled and output_configured,
            "default_action": "preflight_and_plan",
        }


@dataclass(frozen=True, slots=True)
class _PreparedCampaign:
    preparation_id: str
    created_at: str
    profile: _CampaignProfile
    manifest: ExperimentManifest
    report: PreflightReport
    resolved_units: Mapping[str, tuple[str, ...]]
    plan: ExperimentPlan | None
    execution_blockers: tuple[str, ...]

    @property
    def can_execute(self) -> bool:
        return self.report.ready and self.plan is not None


def load_campaign_profile_catalog(
    path: str | Path,
) -> tuple[Mapping[str, _CampaignProfile], str]:
    """Load one strict operator-owned catalog and attest all referenced files."""

    source = Path(path).expanduser().resolve()
    try:
        if not source.is_file():
            raise ArtifactError("campaign profile catalog is not an existing file")
        if source.stat().st_size > MAX_CAMPAIGN_CATALOG_BYTES:
            raise ArtifactError("campaign profile catalog exceeds the size limit")
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("campaign profile catalog is not valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "profiles"}:
        raise ValidationError("campaign profile catalog fields are invalid")
    if raw["schema_version"] != 1:
        raise ValidationError("unsupported campaign profile catalog schema")
    rows = raw["profiles"]
    if not isinstance(rows, list) or not rows or len(rows) > MAX_CAMPAIGN_PROFILES:
        raise ValidationError(
            f"campaign profile catalog requires 1-{MAX_CAMPAIGN_PROFILES} profiles"
        )
    profiles: dict[str, _CampaignProfile] = {}
    for index, value in enumerate(rows):
        profile = _parse_profile(
            value,
            base=source.parent,
            path=f"campaign_catalog.profiles[{index}]",
        )
        if profile.profile_id in profiles:
            raise ValidationError("campaign profile IDs must be unique")
        profiles[profile.profile_id] = profile
    return MappingProxyType(profiles), hashlib.sha256(raw_bytes).hexdigest()


class CampaignManager:
    """Prepare and explicitly queue allowlisted experiment campaigns."""

    def __init__(
        self,
        *,
        catalog_path: str | Path,
        output_root: str | Path | None = None,
        max_workers: int = 1,
        max_preparations: int = MAX_CAMPAIGN_PREPARATIONS,
        max_executions: int = MAX_CAMPAIGN_EXECUTIONS,
    ) -> None:
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or not 1 <= max_workers <= 4
        ):
            raise ValueError("campaign max_workers must be in [1, 4]")
        for value, label in (
            (max_preparations, "max_preparations"),
            (max_executions, "max_executions"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"campaign {label} must be a positive integer")
        profiles, catalog_sha256 = load_campaign_profile_catalog(catalog_path)
        self._profiles = profiles
        self._catalog_sha256 = catalog_sha256
        self._output_root = (
            Path(output_root).expanduser().resolve() if output_root is not None else None
        )
        self._max_preparations = max_preparations
        self._max_executions = max_executions
        self._preparations: dict[str, _PreparedCampaign] = {}
        self._preparation_payloads: dict[str, dict[str, Any]] = {}
        self._executions: dict[str, dict[str, Any]] = {}
        self._submitted_preparations: set[str] = set()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ailab-campaign",
        )

    @property
    def configured(self) -> bool:
        return True

    def catalog_payload(self) -> dict[str, Any]:
        output_configured = self._output_root is not None
        return {
            "schema_version": 1,
            "configured": True,
            "catalog_sha256": self._catalog_sha256,
            "default_action": "preflight_and_plan",
            "browser_request_schema": {
                "preflight": ["profile_id"],
                "execution": [
                    "profile_id",
                    "plan_id",
                    "plan_hash",
                    "confirmation",
                ],
            },
            "profiles": [
                profile.public_payload(output_configured=output_configured)
                for profile in self._profiles.values()
            ],
        }

    def prepare(self, payload: Any) -> dict[str, Any]:
        profile_id = _profile_only_request(payload)
        with self._lock:
            if len(self._preparations) >= self._max_preparations:
                raise CampaignQueueFullError("campaign preparation limit reached")
            profile = self._profiles.get(profile_id)
        if profile is None:
            raise CampaignValidationError("campaign profile is not allowlisted")

        manifest, report, units, plan = self._load_preflight_and_plan(profile)
        preparation = _PreparedCampaign(
            preparation_id=uuid.uuid4().hex,
            created_at=_utc_now(),
            profile=profile,
            manifest=manifest,
            report=report,
            resolved_units=units,
            plan=plan,
            execution_blockers=_campaign_execution_blockers(
                profile,
                manifest,
                report,
            ),
        )
        public = self._preparation_payload(preparation)
        with self._lock:
            if len(self._preparations) >= self._max_preparations:
                raise CampaignQueueFullError("campaign preparation limit reached")
            self._preparations[preparation.preparation_id] = preparation
            self._preparation_payloads[preparation.preparation_id] = public
        return copy.deepcopy(public)

    def get_preparation(self, preparation_id: str) -> dict[str, Any] | None:
        if not isinstance(preparation_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}",
            preparation_id,
        ):
            return None
        with self._lock:
            payload = self._preparation_payloads.get(preparation_id)
            return copy.deepcopy(payload) if payload is not None else None

    def submit_execution(self, payload: Any) -> dict[str, Any]:
        request = _execution_request(payload)
        with self._lock:
            if len(self._executions) >= self._max_executions:
                raise CampaignQueueFullError("campaign execution queue limit reached")
            preparation = self._preparations.get(request["plan_id"])
            if preparation is None:
                raise CampaignValidationError("campaign plan does not exist")
            if preparation.preparation_id in self._submitted_preparations:
                raise CampaignValidationError(
                    "campaign plan was already submitted; prepare a new plan to run again"
                )
            if request["profile_id"] != preparation.profile.profile_id:
                raise CampaignValidationError("campaign profile differs from the frozen plan")
            if preparation.plan is None or not preparation.report.ready:
                raise CampaignValidationError("campaign preflight is blocked")
            if preparation.execution_blockers:
                raise CampaignValidationError(
                    "campaign execution is blocked by an unverified driver safety contract"
                )
            if request["plan_hash"] != preparation.plan.plan_hash:
                raise CampaignValidationError("campaign plan hash does not match")
            if not preparation.profile.execution_enabled or self._output_root is None:
                raise CampaignValidationError(
                    "campaign execution is not configured for this profile"
                )
            execution_id = uuid.uuid4().hex
            record = {
                "id": execution_id,
                "profile_id": preparation.profile.profile_id,
                "profile_identity_sha256": preparation.profile.identity_hash,
                "plan_id": preparation.preparation_id,
                "plan_hash": preparation.plan.plan_hash,
                "manifest_hash": preparation.manifest.manifest_hash,
                "guarded_driver_hash": preparation.report.guarded_driver_hash,
                "status": "queued",
                "queued_at": _utc_now(),
                "started_at": None,
                "finished_at": None,
                "progress": {
                    "total_cells": len(preparation.plan.cells),
                    "completed_cells": 0,
                    "attempted_cells": 0,
                    "succeeded_cells": 0,
                    "failed_cells": 0,
                    "skipped_cells": 0,
                },
                "result_identity": None,
                "summary": None,
                "error": None,
            }
            self._executions[execution_id] = record
            self._submitted_preparations.add(preparation.preparation_id)
        try:
            self._executor.submit(self._run_execution, execution_id, preparation)
        except RuntimeError:
            with self._lock:
                self._executions.pop(execution_id, None)
                self._submitted_preparations.discard(preparation.preparation_id)
            raise CampaignQueueFullError("campaign executor is unavailable") from None
        return copy.deepcopy(record)

    def list_executions(self) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(
                self._executions.values(),
                key=lambda item: str(item["queued_at"]),
                reverse=True,
            )
            return copy.deepcopy(records)

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        if not isinstance(execution_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}",
            execution_id,
        ):
            return None
        with self._lock:
            record = self._executions.get(execution_id)
            return copy.deepcopy(record) if record is not None else None

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _load_preflight_and_plan(
        self,
        profile: _CampaignProfile,
    ) -> tuple[
        ExperimentManifest,
        PreflightReport,
        Mapping[str, tuple[str, ...]],
        ExperimentPlan | None,
    ]:
        manifest = profile.load_manifest()
        report, units = preflight_experiment(
            manifest,
            driver_entrypoint=profile.driver_entrypoint,
            resolve_units=True,
        )
        plan = build_experiment_plan(manifest, units) if report.ready else None
        return manifest, report, units, plan

    def _preparation_payload(self, preparation: _PreparedCampaign) -> dict[str, Any]:
        manifest = preparation.manifest
        report = preparation.report
        plan = preparation.plan
        output_configured = self._output_root is not None
        plan_payload: dict[str, Any] | None = None
        if plan is not None:
            preview = tuple(plan.cells[:MAX_PLAN_PREVIEW_CELLS])
            plan_payload = {
                "id": preparation.preparation_id,
                "plan_hash": plan.plan_hash,
                "cell_count": len(plan.cells),
                "preview_count": len(preview),
                "preview_truncated": len(preview) != len(plan.cells),
                "unit_inventory_hashes": dict(plan.unit_inventory_hashes),
                "cells": [
                    {
                        "cell_id": cell.cell_id,
                        "benchmark_id": cell.benchmark_id,
                        "split": cell.split,
                        "unit_id": cell.unit_id,
                        "model_slot": cell.model_slot,
                        "seed": cell.seed,
                        "arm_ids": list(cell.arm_ids),
                    }
                    for cell in preview
                ],
            }
        return {
            "schema_version": 1,
            "id": preparation.preparation_id,
            "created_at": preparation.created_at,
            "profile": preparation.profile.public_payload(output_configured=output_configured),
            "manifest": {
                "study_id": manifest.study_id,
                "stage": manifest.stage.value,
                "manifest_hash": manifest.manifest_hash,
                "estimand_profile_kind": manifest.estimand_profile_kind.value,
                "benchmarks": [
                    {
                        "cell_id": target.cell_id,
                        "split": target.split,
                        "external": target.external,
                        "source_revision": target.source_revision,
                        "arm_ids": list(
                            target.arm_ids or tuple(arm.arm_id for arm in manifest.arms)
                        ),
                    }
                    for target in manifest.benchmarks
                ],
                "arms": [
                    {
                        "arm_id": arm.arm_id,
                        "release_id": arm.release_id,
                        "family": arm.family.value,
                        "profile_kind": arm.profile_kind.value,
                        "arm_kind": arm.arm_kind.value,
                    }
                    for arm in manifest.arms
                ],
                "model_slots": list(manifest.model_slots),
                "seeds": list(manifest.seeds),
            },
            "preflight": {
                "ready": report.ready,
                "blockers": list(report.blockers),
                "warnings": list(report.warnings),
                "resolved_unit_counts": dict(report.resolved_unit_counts),
                "guarded_driver_hash": report.guarded_driver_hash,
            },
            "plan": plan_payload,
            "execution": {
                "requires_separate_confirmation": True,
                "confirmation_phrase": CAMPAIGN_CONFIRMATION,
                "profile_enabled": preparation.profile.execution_enabled,
                "output_configured": output_configured,
                "ready_for_confirmation": (
                    preparation.can_execute
                    and preparation.profile.execution_enabled
                    and output_configured
                    and not preparation.execution_blockers
                ),
                "blockers": list(preparation.execution_blockers),
            },
        }

    def _run_execution(
        self,
        execution_id: str,
        preparation: _PreparedCampaign,
    ) -> None:
        with self._lock:
            record = self._executions[execution_id]
            record["status"] = "running"
            record["started_at"] = _utc_now()
        try:
            manifest, report, _units, plan = self._load_preflight_and_plan(preparation.profile)
            execution_blockers = _campaign_execution_blockers(
                preparation.profile,
                manifest,
                report,
            )
            if (
                not report.ready
                or plan is None
                or execution_blockers
                or manifest.manifest_hash != preparation.manifest.manifest_hash
                or plan.plan_hash
                != (preparation.plan.plan_hash if preparation.plan is not None else None)
                or report.guarded_driver_hash != preparation.report.guarded_driver_hash
            ):
                raise ArtifactError("campaign bindings changed after the confirmed preflight")
            if (
                report.binding_receipt is None
                or report.guarded_driver_hash is None
                or not report.operator_driver_fingerprint
            ):
                raise ArtifactError("ready campaign preflight lacks guarded identity")
            if self._output_root is None:  # pragma: no cover - submit invariant
                raise ArtifactError("campaign output root is not configured")
            artifact_identity = ExperimentArtifactIdentity(
                study_id=manifest.study_id,
                manifest_hash=manifest.manifest_hash,
                plan_hash=plan.plan_hash,
                driver_hash=report.guarded_driver_hash,
            )
            with activate_guarded_driver(
                manifest=manifest,
                operator_entrypoint=preparation.profile.driver_entrypoint,
                operator_fingerprint=report.operator_driver_fingerprint,
                binding_receipt=report.binding_receipt,
                parity_contract=report.parity_contract,
            ) as guarded_entrypoint:
                summary = execute_experiment_plan(
                    manifest,
                    plan,
                    driver_entrypoint=guarded_entrypoint,
                    output_root=str(self._output_root / execution_id),
                    resume=False,
                    fail_fast=False,
                )
            completed = summary.succeeded_cells + summary.failed_cells + summary.skipped_cells
            with self._lock:
                record = self._executions[execution_id]
                record["status"] = (
                    "completed" if summary.failed_cells == 0 else "completed_with_failures"
                )
                record["finished_at"] = _utc_now()
                record["progress"] = {
                    "total_cells": summary.planned_cells,
                    "completed_cells": completed,
                    "attempted_cells": summary.attempted_cells,
                    "succeeded_cells": summary.succeeded_cells,
                    "failed_cells": summary.failed_cells,
                    "skipped_cells": summary.skipped_cells,
                }
                record["result_identity"] = {
                    "artifact_identity_sha256": artifact_identity.identity_hash,
                    "manifest_sha256": manifest.manifest_hash,
                    "plan_sha256": plan.plan_hash,
                    "driver_sha256": report.guarded_driver_hash,
                }
                record["summary"] = {
                    "planned_cells": summary.planned_cells,
                    "attempted_cells": summary.attempted_cells,
                    "skipped_cells": summary.skipped_cells,
                    "succeeded_cells": summary.succeeded_cells,
                    "failed_cells": summary.failed_cells,
                }
        except Exception as exc:
            _LOGGER.exception("Studio campaign execution %s failed", execution_id)
            with self._lock:
                record = self._executions[execution_id]
                record["status"] = "failed"
                record["finished_at"] = _utc_now()
                record["error"] = {
                    "category": type(exc).__name__,
                    "message": (
                        "Campaign execution aborted. Inspect the server log and "
                        "content-addressed experiment artifacts for details."
                    ),
                }


def _campaign_execution_blockers(
    profile: _CampaignProfile,
    manifest: ExperimentManifest,
    report: PreflightReport,
) -> tuple[str, ...]:
    """Require fingerprinted P0 controls before external-arena execution.

    External profiles remain available for useful preflight and exact planning.
    They cannot be confirmed until the selected driver fingerprint explicitly
    attests the safety contracts relevant to the manifest.
    """

    if profile.driver_entrypoint != _EXTERNAL_ARENA_ENTRYPOINT:
        return ()
    raw = report.operator_driver_fingerprint.get("studio_campaign_attestation")
    attestation = raw if isinstance(raw, Mapping) else {}
    blockers: list[str] = []
    if (
        attestation.get("schema_version") != 1
        or attestation.get("shared_dependency_slots") is not True
    ):
        blockers.append("external driver has not attested required shared dependency slots")
    if any(arm.arm_kind.value == "policy" for arm in manifest.arms) and (
        attestation.get("selector_group_exclusion") is not True
    ):
        blockers.append("external driver has not attested discovery-group exclusion for policies")
    if (
        any(target.cell_id in _TYPED_GATE_TARGETS for target in manifest.benchmarks)
        and attestation.get("typed_gate_receipts") is not True
    ):
        blockers.append("external driver has not attested strict typed gate/stack receipts")
    return tuple(blockers)


def disabled_campaign_catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "configured": False,
        "catalog_sha256": None,
        "default_action": "preflight_and_plan",
        "browser_request_schema": {
            "preflight": ["profile_id"],
            "execution": [
                "profile_id",
                "plan_id",
                "plan_hash",
                "confirmation",
            ],
        },
        "profiles": [],
        "reason": (
            "Start Studio with a server-side campaign profile catalog to enable "
            "research-campaign preflight."
        ),
    }


def _parse_profile(
    value: Any,
    *,
    base: Path,
    path: str,
) -> _CampaignProfile:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    expected = {
        "id",
        "label",
        "description",
        "blueprint",
        "arm_catalog",
        "driver_entrypoint",
        "execution_enabled",
    }
    if set(value) != expected:
        raise ValidationError(f"{path} fields are invalid")
    return _CampaignProfile(
        profile_id=_text(value["id"], f"{path}.id"),
        label=_text(value["label"], f"{path}.label"),
        description=_text(value["description"], f"{path}.description"),
        blueprint=_locked_file(
            value["blueprint"],
            base=base,
            path=f"{path}.blueprint",
            label="campaign blueprint",
        ),
        arm_catalog=_locked_file(
            value["arm_catalog"],
            base=base,
            path=f"{path}.arm_catalog",
            label="campaign arm catalog",
        ),
        driver_entrypoint=_text(
            value["driver_entrypoint"],
            f"{path}.driver_entrypoint",
        ),
        execution_enabled=value["execution_enabled"],
    )


def _locked_file(
    value: Any,
    *,
    base: Path,
    path: str,
    label: str,
) -> _LockedCampaignFile:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValidationError(f"{path} fields are invalid")
    raw_path = _text(value["path"], f"{path}.path")
    source = Path(raw_path).expanduser()
    if not source.is_absolute():
        source = base / source
    return _LockedCampaignFile(
        path=source.resolve(),
        sha256=_text(value["sha256"], f"{path}.sha256"),
        label=label,
    )


def _profile_only_request(payload: Any) -> str:
    if not isinstance(payload, Mapping) or set(payload) != {"profile_id"}:
        raise CampaignSecurityError("campaign preflight accepts only one allowlisted profile_id")
    profile_id = payload["profile_id"]
    if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
        raise CampaignValidationError("campaign profile_id is invalid")
    return profile_id


def _execution_request(payload: Any) -> dict[str, str]:
    expected = {"profile_id", "plan_id", "plan_hash", "confirmation"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise CampaignSecurityError(
            "campaign execution accepts only frozen profile and plan identities"
        )
    profile_id = payload["profile_id"]
    plan_id = payload["plan_id"]
    plan_hash = payload["plan_hash"]
    confirmation = payload["confirmation"]
    if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
        raise CampaignValidationError("campaign profile_id is invalid")
    if not isinstance(plan_id, str) or re.fullmatch(r"[0-9a-f]{32}", plan_id) is None:
        raise CampaignValidationError("campaign plan_id is invalid")
    if not isinstance(plan_hash, str) or _SHA256.fullmatch(plan_hash) is None:
        raise CampaignValidationError("campaign plan_hash is invalid")
    if confirmation != CAMPAIGN_CONFIRMATION:
        raise CampaignValidationError("campaign execution requires the exact confirmation phrase")
    return {
        "profile_id": profile_id,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "confirmation": confirmation,
    }


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{path} must be trimmed non-empty text")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


__all__ = [
    "CAMPAIGN_CONFIRMATION",
    "CampaignManager",
    "CampaignQueueFullError",
    "CampaignSecurityError",
    "CampaignValidationError",
    "disabled_campaign_catalog_payload",
    "load_campaign_profile_catalog",
]
