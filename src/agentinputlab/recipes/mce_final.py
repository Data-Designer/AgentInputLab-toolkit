"""Frozen executable MCE artifact evaluation through the isolated sidecar.

No fitting, artifact import, or generated-code execution takes place on the
caller host. Test references travel only to the sidecar's trusted oracle.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from agentinputlab.contracts import AlgorithmOutputError, InterventionContext, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    resolve_worker,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

from .mce import (
    MCE_PROTOCOL,
    MCE_SEMANTIC_PATCH_MANIFEST,
    MCE_UPSTREAM_COMMIT,
    MCEContextArtifact,
    MCESandboxAttestation,
    MCESplitAudit,
    _artifact_bundle,
    _call_mce_worker,
    _model_fingerprints,
    _validate_worker_identity,
)

FINAL_RELEASE = "recipe.mce.final_artifact_eval"
_AUDIT_FIELDS = frozenset(MCESplitAudit.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class MCEFinalExample:
    example_id: str
    input: Mapping[str, Any]
    expected_output: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValidationError("MCE final example requires an ID")
        if not isinstance(self.input, Mapping) or not isinstance(self.metadata, Mapping):
            raise ValidationError("MCE final input/metadata must be mappings")
        if {"id", "ground_truth"} & set(self.input):
            raise ValidationError("MCE final input cannot override native id/ground_truth")
        try:
            object.__setattr__(self, "input", freeze_mapping(self.input))
            object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
            # Validate references too, without ever exporting them to generated code.
            content_hash(self.expected_output)
        except (TypeError, ValueError) as exc:
            raise ValidationError("MCE final example is not canonical JSON") from exc

    def as_dict(self) -> Mapping[str, Any]:
        return {**to_jsonable(self), "split": "test"}


@dataclass(frozen=True, slots=True)
class MCEFinalControls:
    max_model_calls: int = 8
    max_tool_calls: int = 8
    max_wall_seconds: int = 3600
    max_output_tokens: int = 4096
    max_output_tokens_per_call: int = 512

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValidationError("MCE final budgets must be positive integers")
        if self.max_output_tokens_per_call > self.max_output_tokens:
            raise ValidationError("MCE final per-call cap exceeds the total cap")


def validate_final_split(
    audit: Mapping[str, Any], expected_hash: str, examples: Sequence[Mapping[str, Any]],
) -> str:
    """Check the original hash-bound fit audit, then exclude heldout overlap."""
    if set(audit) != _AUDIT_FIELDS or content_hash(audit) != expected_hash:
        raise ValidationError("MCE final evaluation needs the original full split audit")
    for split in ("train", "validation"):
        groups = [audit[f"{split}_{suffix}"] for suffix in ("ids", "input_hashes", "example_hashes")]
        if any(isinstance(v, (str, bytes)) or not isinstance(v, Sequence) or not v for v in groups):
            raise ValidationError("MCE original split audit arrays are invalid")
        if any(not isinstance(v, str) or not v for values in groups for v in values):
            raise ValidationError("MCE original split audit values are invalid")
        if len({len(v) for v in groups}) != 1 or any(len(v) != len(set(v)) for v in groups):
            raise ValidationError("MCE original split audit has duplicates or unequal lengths")
    if set(audit["train_ids"]) & set(audit["validation_ids"]) or (
        set(audit["train_input_hashes"]) & set(audit["validation_input_hashes"])
    ):
        raise ValidationError("MCE original fit split overlap")
    if not examples:
        raise ValidationError("MCE final test collection is empty")
    ids, inputs, complete = [], [], []
    for row in examples:
        if set(row) != {"example_id", "input", "expected_output", "metadata", "split"} or row["split"] != "test":
            raise ValidationError("MCE final example schema or split drifted")
        parsed = MCEFinalExample(row["example_id"], row["input"], row["expected_output"], row["metadata"])
        ids.append(parsed.example_id)
        inputs.append(content_hash(parsed.input))
        complete.append(content_hash(row))
    if len(ids) != len(set(ids)) or len(inputs) != len(set(inputs)):
        raise ValidationError("MCE final examples contain duplicate IDs/inputs")
    if set(ids) & (set(audit["train_ids"]) | set(audit["validation_ids"])) or (
        set(inputs) & (set(audit["train_input_hashes"]) | set(audit["validation_input_hashes"]))
    ):
        raise ValidationError("MCE final test collection overlaps fitting/calibration")
    return content_hash({"fit_split_audit_sha256": expected_hash, "test_example_hashes": complete})


@dataclass(frozen=True, slots=True)
class MCEFinalEvaluation:
    artifact_id: str
    evaluation_id: str
    test_audit_sha256: str
    rows: tuple[Mapping[str, Any], ...]
    native_summary: Mapping[str, Any]
    all_positions_primary_metric: float
    usage: Mapping[str, Any]
    output_budget: Mapping[str, Any]
    native_evaluation_sha256: str
    evidence_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(freeze_mapping(row) for row in self.rows))
        for field_name in ("native_summary", "usage", "output_budget"):
            object.__setattr__(self, field_name, freeze_mapping(getattr(self, field_name)))

    def as_dict(self) -> Mapping[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class MCEFinalArtifactEvaluator:
    artifact: MCEContextArtifact
    split_audit: MCESplitAudit
    interface_signature: Mapping[str, Any]
    client: WorkerClient | None = None
    controls: MCEFinalControls = MCEFinalControls()
    allow_fixture_runtime: bool = False

    def evaluate(
        self, examples: Sequence[MCEFinalExample], *, context: InterventionContext,
    ) -> MCEFinalEvaluation:
        if not isinstance(self.artifact, MCEContextArtifact) or not isinstance(self.split_audit, MCESplitAudit):
            raise ValidationError("MCE final evaluation requires typed frozen artifact and split audit")
        if not isinstance(self.controls, MCEFinalControls) or type(self.allow_fixture_runtime) is not bool:
            raise ValidationError("MCE final controls/fixture policy are invalid")
        if not all(isinstance(row, MCEFinalExample) for row in examples):
            raise ValidationError("MCE final examples must be typed")
        rows = tuple(row.as_dict() for row in examples)
        audit = to_jsonable(self.split_audit)
        test_hash = validate_final_split(audit, self.artifact.split_audit_sha256, rows)
        artifact = self.artifact
        if content_hash(self.interface_signature) != artifact.interface_signature_sha256:
            raise ValidationError("MCE final interface signature differs from the fitted artifact")
        models = _model_fingerprints(context)
        if models != artifact.model_fingerprints:
            raise ValidationError("MCE final model fingerprints differ from fitting")
        worker = resolve_worker(self.client, context, label="MCE final evaluation")
        identity = worker_fingerprint(worker, label="MCE final evaluation")
        sandbox = MCESandboxAttestation.from_worker_fingerprint(identity)
        if sandbox.runtime_mode == "fixture" and not self.allow_fixture_runtime:
            raise ValidationError("MCE final fixture workers are disabled by default")
        bound = _validate_worker_identity(
            identity, sandbox=sandbox, interface_signature_sha256=artifact.interface_signature_sha256,
            initial_context_sha256=artifact.initial_context_sha256,
            environment_identity_sha256=artifact.environment_identity_sha256,
            model_fingerprints_sha256=content_hash(models),
            license_allowlist=(artifact.license_manifest_sha256,),
            provenance_allowlist=(artifact.provenance_manifest_sha256,),
        )
        for key, value in bound.items():
            original = artifact.worker_fingerprint.get(key) if key == "source_tree" else getattr(artifact, key)
            if value != original:
                raise ValidationError(f"MCE final frozen dependency changed: {key}")
        bundle = _artifact_bundle(
            files=tuple(row.as_dict() for row in artifact.files), selected_iteration=artifact.selected_iteration,
            artifact_manifest_sha256=artifact.artifact_manifest_sha256, skill_sha256=artifact.skill_sha256,
            split_audit_sha256=artifact.split_audit_sha256,
            interface_signature_sha256=artifact.interface_signature_sha256,
            initial_context_sha256=artifact.initial_context_sha256,
            environment_identity_sha256=artifact.environment_identity_sha256,
            model_fingerprints_sha256=content_hash(models), controls_sha256=content_hash(artifact.controls),
            runtime_binding_sha256=artifact.runtime_binding_sha256, bound=bound,
        )
        if content_hash(bundle) != artifact.artifact_bundle_sha256:
            raise ValidationError("MCE frozen bundle does not match its fitted identity")
        if artifact.artifact_id != content_hash({
            **bundle, "artifact_handle": artifact.artifact_handle, "release_id": artifact.release_id,
            "sandbox_attestation_id": artifact.sandbox_attestation.attestation_id,
        }):
            raise ValidationError("MCE frozen artifact ID changed")
        controls = {**asdict(self.controls), "seed": context.seed}
        payload = {
            **bound,
            "split_audit_sha256": artifact.split_audit_sha256,
            "split_audit": audit, "test_examples": rows, "test_audit_sha256": test_hash,
            "frozen_bundle": bundle, "artifact_handle": artifact.artifact_handle,
            "artifact_bundle_sha256": artifact.artifact_bundle_sha256,
            "interface_signature": self.interface_signature,
            "interface_signature_sha256": artifact.interface_signature_sha256,
            "initial_context_identity": {"schema": "agentinputlab.mce.initial-context.v1", "files": ()},
            "initial_context_sha256": artifact.initial_context_sha256,
            "environment_identity": artifact.environment_identity,
            "environment_identity_sha256": artifact.environment_identity_sha256,
            "model_fingerprints": models, "model_fingerprints_sha256": content_hash(models),
            "sandbox_attestation_id": sandbox.attestation_id,
            "controls": artifact.controls, "evaluation_controls": controls,
            "runtime_binding_sha256": artifact.runtime_binding_sha256,
            "worker_fingerprint_sha256": content_hash(identity),
            "semantic_patch_manifest": MCE_SEMANTIC_PATCH_MANIFEST,
            "host_execution_allowed": False, "artifact_egress": "manifest_and_opaque_handle_only",
        }
        state_hash = content_hash({"artifact": artifact.artifact_bundle_sha256, "test": test_hash, "controls": controls})
        request = MethodWorkerRequest(
            method_id="recipe.mce", release_id=FINAL_RELEASE, upstream_commit=MCE_UPSTREAM_COMMIT,
            protocol_version=MCE_PROTOCOL, state_hash=state_hash, payload=payload,
            dependency_fingerprints={"worker": identity}, minimum_reservation={"model_calls": 1},
        )
        response = _call_mce_worker(worker, request)
        result = response.payload
        expected_fields = {"test_audit_sha256", "artifact_bundle_sha256", "rows", "native_summary",
                           "all_positions_primary_metric", "output_budget", "native_evaluation_sha256",
                           "evidence_id", "adaptation_performed", "host_code_executed"}
        if set(result) != expected_fields or result["test_audit_sha256"] != test_hash or (
            result["artifact_bundle_sha256"] != artifact.artifact_bundle_sha256
            or result["adaptation_performed"] is not False or result["host_code_executed"] is not False
        ):
            raise AlgorithmOutputError("MCE frozen evaluation response identity/schema drifted")
        checked_rows = validate_final_summary(result["rows"], result["native_summary"], rows)
        mean = sum(row["metrics"].get(result["native_summary"]["primary_metric"], 0.0)
                   for row in checked_rows) / len(rows)
        if not math.isclose(mean, result["all_positions_primary_metric"], abs_tol=1e-12):
            raise AlgorithmOutputError("MCE all-position denominator changed")
        if response.usage.model_calls > controls["max_model_calls"] or (
            response.usage.tool_calls > controls["max_tool_calls"]
            or response.usage.output_tokens > controls["max_output_tokens"]
        ):
            raise AlgorithmOutputError("MCE final response exceeded its configured budget")
        validate_output_budget(result["output_budget"], controls, external_usage_counters(response.usage))
        digest = result["native_evaluation_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AlgorithmOutputError("MCE final raw-evidence digest is invalid")
        if result["evidence_id"] != f"mce-final-{request.request_id}":
            raise AlgorithmOutputError("MCE final evidence ID does not bind its attempt")
        return MCEFinalEvaluation(
            artifact_id=artifact.artifact_id, evaluation_id=request.request_id, test_audit_sha256=test_hash,
            rows=checked_rows, native_summary=result["native_summary"], all_positions_primary_metric=mean,
            usage=external_usage_counters(response.usage), output_budget=result["output_budget"],
            native_evaluation_sha256=result["native_evaluation_sha256"], evidence_id=result["evidence_id"],
        )


def validate_final_summary(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], expected: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Keep failed positions visible instead of silently shrinking a denominator."""
    if not isinstance(rows, (tuple, list)) or len(rows) != len(expected):
        raise AlgorithmOutputError("MCE final rows omit heldout positions")
    if any(not isinstance(row, Mapping) for row in rows) or (
        [r.get("example_id") for r in rows] != [r["example_id"] for r in expected]
    ):
        raise AlgorithmOutputError("MCE final example order or IDs drifted")
    for row in rows:
        if set(row) != {"example_id", "status", "metrics", "error"} or row["status"] not in {"scored", "failed"}:
            raise AlgorithmOutputError("MCE final row schema drifted")
        if not isinstance(row["metrics"], Mapping) or any(
            not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for key, value in row["metrics"].items()
        ):
            raise AlgorithmOutputError("MCE final metrics are invalid")
        if row["status"] == "failed" and (row["metrics"] or not isinstance(row["error"], str) or not row["error"]):
            raise AlgorithmOutputError("MCE failed row must retain its error, not invented metrics")
        if row["status"] == "scored" and row["error"] is not None:
            raise AlgorithmOutputError("MCE scored row contains an error")
    failures = sum(row["status"] == "failed" for row in rows)
    if not isinstance(summary, Mapping) or set(summary) != {
        "metrics", "primary_metric", "primary_metric_value", "total", "successful", "errors", "environment",
    }:
        raise AlgorithmOutputError("MCE native summary schema drifted")
    if any(type(summary[key]) is not int for key in ("total", "successful", "errors")):
        raise AlgorithmOutputError("MCE native summary counts must be integers")
    if summary.get("total") != len(rows) or summary.get("errors") != failures or (
        summary.get("successful") != len(rows) - failures
        or not isinstance(summary.get("primary_metric"), str) or not summary["primary_metric"]
    ):
        raise AlgorithmOutputError("MCE native summary counts are inconsistent")
    metrics = summary["metrics"]
    if not isinstance(metrics, Mapping):
        raise AlgorithmOutputError("MCE native summary metrics are invalid")
    for key, value in {**metrics, "_primary": summary["primary_metric_value"]}.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AlgorithmOutputError("MCE native summary metrics are not finite")
    primary = metrics.get(summary["primary_metric"], 0.0)
    if not math.isclose(primary, summary["primary_metric_value"], abs_tol=1e-12):
        raise AlgorithmOutputError("MCE native primary metric disagrees with its metrics")
    return tuple(rows)


def validate_output_budget(
    receipt: Mapping[str, Any], controls: Mapping[str, Any], usage: Mapping[str, Any],
) -> None:
    expected = {"schema", "max_output_tokens", "max_output_tokens_per_call", "actual_output_tokens",
                "provider_responses", "pending_reservations", "failed_unknown_output_reservations",
                "reserved_unknown_tokens_are_not_actual_usage"}
    if not isinstance(receipt, Mapping) or set(receipt) != expected or (
        receipt["schema"] != "agentinputlab.mce.final-output-budget.v1"
        or receipt["max_output_tokens"] != controls["max_output_tokens"]
        or receipt["max_output_tokens_per_call"] != controls["max_output_tokens_per_call"]
        or receipt["actual_output_tokens"] != usage["output_tokens"]
        or receipt["reserved_unknown_tokens_are_not_actual_usage"] is not True
        or receipt["pending_reservations"] != {}
        or type(receipt["provider_responses"]) is not int or receipt["provider_responses"] < 0
    ):
        raise AlgorithmOutputError("MCE final token-budget receipt is inconsistent")
    unknown = receipt["failed_unknown_output_reservations"]
    if not isinstance(unknown, Mapping) or any(
        not isinstance(key, str) or not key.isascii() or not key.isdecimal()
        or str(int(key)) != key or not 1 <= int(key) <= usage["model_calls"]
        or type(value) is not int or not 1 <= value <= controls["max_output_tokens_per_call"]
        for key, value in unknown.items()
    ):
        raise AlgorithmOutputError("MCE final unknown-token reservations are invalid")
    if receipt["provider_responses"] + len(unknown) != usage["model_calls"] or (
        sum(unknown.values()) + usage["output_tokens"] > controls["max_output_tokens"]
    ):
        raise AlgorithmOutputError("MCE final token usage/reservations exceed the attempt budget")
