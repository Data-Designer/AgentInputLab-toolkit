"""Explicit experiment execution over an operator-supplied, auditable driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, freeze_mapping, to_jsonable
from agentinputlab.observability import ResourceReceipt

from .artifacts import ExperimentArtifactStore
from .loader import resolve_driver_entrypoint
from .plan import ExperimentCell, ExperimentPlan
from .schema import ExperimentManifest
from .secrets import redact_secret_text, require_secret_free


@dataclass(frozen=True, slots=True)
class ExperimentCellResult:
    """One fully attempted comparison cell.

    ``succeeded`` means that the driver completed the comparison protocol; an
    individual algorithm arm may still have a typed ``algorithm`` failure in
    ``payload.arms``.  ``failed`` is reserved for an explicitly returned,
    terminal algorithm/benchmark-level failure.  Infrastructure, configuration,
    and driver-protocol errors must be raised so the executor can abort without
    marking the cell complete.
    """

    cell_id: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    failure: Mapping[str, Any] = field(default_factory=dict)
    resource_receipts: tuple[ResourceReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValidationError("experiment cell result requires cell_id")
        if self.status not in {"succeeded", "failed"}:
            raise ValidationError("experiment cell status must be succeeded or failed")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValidationError("experiment metrics require names and numeric values")
            numeric = float(value)
            if not (-float("inf") < numeric < float("inf")):
                raise ValidationError("experiment metrics must be finite")
            metrics[name] = numeric
        if self.status == "succeeded" and self.failure:
            raise ValidationError("successful cell result cannot contain failure metadata")
        if self.status == "failed":
            if not self.failure:
                raise ValidationError("failed cell result requires failure metadata")
            if self.failure.get("scope") not in {"algorithm", "benchmark"}:
                raise ValidationError(
                    "failed cell result scope must be algorithm or benchmark; "
                    "infrastructure failures must be raised"
                )
        receipts = tuple(self.resource_receipts)
        if any(not isinstance(receipt, ResourceReceipt) for receipt in receipts):
            raise ValidationError("experiment resource receipts must be ResourceReceipt values")
        require_secret_free(self.payload, label="experiment result payload")
        require_secret_free(self.failure, label="experiment result failure")
        object.__setattr__(self, "metrics", freeze_mapping(metrics))
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
        object.__setattr__(self, "failure", freeze_mapping(self.failure))
        object.__setattr__(self, "resource_receipts", receipts)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@runtime_checkable
class ExperimentDriver(Protocol):
    """Operator code that binds library contracts to real local assets."""

    def fingerprint(self) -> Mapping[str, Any]: ...

    def preflight(self, manifest: ExperimentManifest) -> Sequence[str]: ...

    def execute_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentCellResult: ...


@dataclass(frozen=True, slots=True)
class ExperimentExecutionSummary:
    study_id: str
    plan_hash: str
    run_dir: str
    planned_cells: int
    attempted_cells: int
    skipped_cells: int
    succeeded_cells: int
    failed_cells: int

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def load_experiment_driver(entrypoint: str) -> ExperimentDriver:
    factory = resolve_driver_entrypoint(entrypoint)
    if not callable(factory):
        raise ArtifactError("experiment driver entrypoint must be a zero-argument factory")
    try:
        driver = factory()
    except TypeError as exc:
        raise ArtifactError("experiment driver entrypoint must be a zero-argument factory") from exc
    if not isinstance(driver, ExperimentDriver):
        raise ArtifactError("experiment driver does not implement ExperimentDriver")
    try:
        fingerprint = freeze_mapping(driver.fingerprint())
    except (TypeError, ValueError) as exc:
        raise ArtifactError("experiment driver fingerprint must be canonical JSON") from exc
    if not fingerprint:
        raise ArtifactError("experiment driver fingerprint must not be empty")
    require_secret_free(fingerprint, label="experiment driver fingerprint")
    return driver


def execute_experiment_plan(
    manifest: ExperimentManifest,
    plan: ExperimentPlan,
    *,
    driver_entrypoint: str,
    output_root: str,
    resume: bool,
    fail_fast: bool = False,
) -> ExperimentExecutionSummary:
    """Execute only after the caller has made an explicit ``--execute`` choice."""

    driver = load_experiment_driver(driver_entrypoint)
    blockers = tuple(str(item) for item in driver.preflight(manifest))
    if any(not item.strip() for item in blockers):
        raise ArtifactError("experiment driver returned an invalid preflight blocker")
    if blockers:
        raise ArtifactError("driver preflight blocked execution: " + "; ".join(blockers))

    driver_fingerprint = freeze_mapping(driver.fingerprint())
    require_secret_free(driver_fingerprint, label="experiment driver fingerprint")
    store = ExperimentArtifactStore(
        output_root,
        manifest=manifest,
        plan=plan,
        driver_fingerprint=driver_fingerprint,
    )
    store.initialize(resume=resume)
    completed = store.completed_cell_ids()
    attempted = skipped = 0
    for cell in plan.cells:
        if cell.cell_id in completed:
            skipped += 1
            continue
        attempted += 1
        store.append_event(
            {
                "schema_version": 1,
                "event": "cell_started",
                "timestamp_utc": _utc_now(),
                "cell_id": cell.cell_id,
            }
        )
        try:
            result = driver.execute_cell(manifest, cell)
            if not isinstance(result, ExperimentCellResult):
                raise ValidationError("experiment driver must return ExperimentCellResult")
            if result.cell_id != cell.cell_id:
                raise ValidationError("experiment driver returned another cell identity")
            _validate_cell_result(cell, result)
        except Exception as exc:
            receipt = getattr(exc, "resource_receipt", None)
            failure: dict[str, Any] = {
                "scope": "infrastructure_or_protocol",
                "category": type(exc).__name__,
                "message": _safe_message(exc),
            }
            if isinstance(receipt, ResourceReceipt):
                failure["resource_receipt"] = to_jsonable(receipt)
            store.append_event(
                {
                    "schema_version": 1,
                    "event": "cell_aborted",
                    "timestamp_utc": _utc_now(),
                    "cell_id": cell.cell_id,
                    "failure": failure,
                }
            )
            raise ArtifactError(
                f"experiment cell {cell.cell_id} aborted without a terminal result: "
                f"{type(exc).__name__}: {_safe_message(exc)}"
            ) from exc
        store.write_cell_result(
            cell.cell_id,
            build_experiment_result_record(
                manifest,
                plan,
                cell,
                result,
            ),
        )
        store.append_event(
            {
                "schema_version": 1,
                "event": "cell_finished",
                "timestamp_utc": _utc_now(),
                "cell_id": cell.cell_id,
                "status": result.status,
            }
        )
        if result.status == "failed" and fail_fast:
            break
    status_counts = store.result_status_counts()
    return ExperimentExecutionSummary(
        study_id=manifest.study_id,
        plan_hash=plan.plan_hash,
        run_dir=str(store.run_dir),
        planned_cells=len(plan.cells),
        attempted_cells=attempted,
        skipped_cells=skipped,
        succeeded_cells=status_counts["succeeded"],
        failed_cells=status_counts["failed"],
    )


def _validate_cell_result(
    cell: ExperimentCell,
    result: ExperimentCellResult,
) -> None:
    """Reject partial or selectively reported comparison payloads."""

    if result.cell_id != cell.cell_id:
        raise ValidationError("experiment result cell identity differs from the frozen plan")
    if result.status == "failed":
        return
    arms = result.payload.get("arms")
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise ValidationError("successful experiment cell payload requires an arms array")
    expected = set(cell.arm_ids)
    observed: set[str] = set()
    for index, raw in enumerate(arms):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"experiment arm outcome {index} must be an object")
        arm_id = raw.get("arm_id")
        status = raw.get("status")
        metrics = raw.get("metrics")
        if not isinstance(arm_id, str) or not arm_id:
            raise ValidationError("experiment arm outcome requires arm_id")
        if arm_id in observed:
            raise ValidationError("experiment cell contains duplicate arm outcomes")
        observed.add(arm_id)
        if status == "succeeded":
            if not isinstance(metrics, Mapping) or not metrics:
                raise ValidationError(f"successful experiment arm {arm_id!r} requires metrics")
            for name, value in metrics.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not (-float("inf") < float(value) < float("inf"))
                ):
                    raise ValidationError(
                        f"experiment arm {arm_id!r} metrics must be finite numerics"
                    )
        elif status == "failed":
            failure = raw.get("failure")
            if not isinstance(failure, Mapping) or failure.get("scope") != "algorithm":
                raise ValidationError(
                    f"failed experiment arm {arm_id!r} requires algorithm-scoped failure"
                )
            if metrics not in (None, {}):
                raise ValidationError(
                    f"failed experiment arm {arm_id!r} cannot carry evaluator metrics"
                )
        else:
            raise ValidationError(f"experiment arm {arm_id!r} status must be succeeded or failed")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValidationError(
            "experiment cell arm outcomes differ from the frozen plan; "
            f"missing={missing}, extra={extra}"
        )


def parse_experiment_cell_result(
    cell: ExperimentCell,
    raw: Any,
) -> ExperimentCellResult:
    """Decode and validate the exact JSON form emitted by ``ExperimentCellResult``.

    The offline artifact verifier uses this public boundary instead of trusting
    a status string from an indexed JSON file.
    """

    required = {
        "cell_id",
        "status",
        "metrics",
        "payload",
        "failure",
        "resource_receipts",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValidationError(
            "serialized experiment cell result fields must be exactly: "
            + ", ".join(sorted(required))
        )
    receipts_raw = raw["resource_receipts"]
    if not isinstance(receipts_raw, list):
        raise ValidationError("serialized experiment resource_receipts must be an array")
    receipts = tuple(
        _parse_resource_receipt(value, index=index) for index, value in enumerate(receipts_raw)
    )
    metrics = raw["metrics"]
    payload = raw["payload"]
    failure = raw["failure"]
    if not isinstance(metrics, Mapping):
        raise ValidationError("serialized experiment metrics must be an object")
    if not isinstance(payload, Mapping):
        raise ValidationError("serialized experiment payload must be an object")
    if not isinstance(failure, Mapping):
        raise ValidationError("serialized experiment failure must be an object")
    try:
        result = ExperimentCellResult(
            cell_id=raw["cell_id"],
            status=raw["status"],
            metrics=metrics,
            payload=payload,
            failure=failure,
            resource_receipts=receipts,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("serialized experiment cell result is invalid") from exc
    _validate_cell_result(cell, result)
    if canonical_json(result.to_dict()) != canonical_json(raw):
        raise ValidationError("serialized experiment cell result is not canonical")
    return result


def build_experiment_result_record(
    manifest: ExperimentManifest,
    plan: ExperimentPlan,
    cell: ExperimentCell,
    result: ExperimentCellResult,
    *,
    finished_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Build the only serialized record shape accepted by the offline verifier."""

    if (
        plan.manifest_hash != manifest.manifest_hash
        or cell.manifest_hash != manifest.manifest_hash
        or cell.study_id != manifest.study_id
        or cell.cell_id not in {planned.cell_id for planned in plan.cells}
    ):
        raise ValidationError("experiment result record identities do not share one frozen plan")
    if not isinstance(result, ExperimentCellResult) or result.cell_id != cell.cell_id:
        raise ValidationError("experiment result record requires the matching typed cell result")
    _validate_cell_result(cell, result)
    timestamp = _utc_now() if finished_at_utc is None else finished_at_utc
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ValidationError("experiment result finished_at_utc must be a UTC timestamp")
    return freeze_mapping(
        {
            "schema_version": 1,
            "study_id": manifest.study_id,
            "manifest_hash": manifest.manifest_hash,
            "plan_hash": plan.plan_hash,
            "cell": {**to_jsonable(cell), "cell_id": cell.cell_id},
            "result": result.to_dict(),
            "finished_at_utc": timestamp,
            "claims": {
                "official_parity_verified": False,
                "confirmatory_candidate_record": (
                    manifest.stage.value == "confirmatory" and result.status == "succeeded"
                ),
                "paper_claim_authorized": False,
                "resource_receipt_count": len(result.resource_receipts),
            },
        }
    )


def _parse_resource_receipt(value: Any, *, index: int) -> ResourceReceipt:
    required = {
        "limit_hash",
        "wall_seconds",
        "supervisor_identity",
        "image_identity",
        "cpu_seconds",
        "peak_rss_bytes",
        "accelerator_seconds",
        "accelerator_peak_memory_bytes",
        "oom_killed",
        "timed_out",
        "exit_reason",
        "exit_code",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValidationError(f"serialized resource receipt {index} fields are invalid")
    try:
        return ResourceReceipt(
            limit_hash=value["limit_hash"],
            wall_seconds=value["wall_seconds"],
            supervisor_identity=value["supervisor_identity"],
            image_identity=value["image_identity"],
            cpu_seconds=value["cpu_seconds"],
            peak_rss_bytes=value["peak_rss_bytes"],
            accelerator_seconds=value["accelerator_seconds"],
            accelerator_peak_memory_bytes=value["accelerator_peak_memory_bytes"],
            oom_killed=value["oom_killed"],
            timed_out=value["timed_out"],
            exit_reason=value["exit_reason"],
            exit_code=value["exit_code"],
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"serialized resource receipt {index} is invalid") from exc


def _safe_message(error: BaseException) -> str:
    return redact_secret_text(str(error))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "ExperimentCellResult",
    "ExperimentDriver",
    "ExperimentExecutionSummary",
    "build_experiment_result_record",
    "execute_experiment_plan",
    "load_experiment_driver",
    "parse_experiment_cell_result",
]
