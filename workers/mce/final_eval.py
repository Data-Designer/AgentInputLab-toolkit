"""Frozen-code evaluation route; generated code remains in the isolated manager."""
from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentinputlab.contracts import AlgorithmOutputError, ValidationError
from agentinputlab.recipes.mce_final import (
    FINAL_RELEASE,
    MCEFinalControls,
    validate_final_split,
    validate_final_summary,
)

from .driver import (
    PROTOCOL,
    UPSTREAM_COMMIT,
    DriverError,
    MCEWorkerDriver,
    TrustedEvaluationSupervisor,
    _artifact_bundle,
    _canonical_json,
    _content_hash,
    _mapping,
    _nonnegative_int,
    _sha256_file,
    _text,
    _validate_envelope,
    _write_native_jsonl,
)


def _save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(value))


def handle_final_evaluation(driver: MCEWorkerDriver, request: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _validate_envelope(request, release_id=FINAL_RELEASE)
    driver._verify_source()
    driver._verify_request_identity(request, payload)
    frozen = _mapping(payload, "frozen_bundle")
    digest = _text(payload, "artifact_bundle_sha256")
    handle = _text(payload, "artifact_handle")
    if _content_hash(frozen) != digest or handle != f"urn:agentinputlab:mce:sha256:{digest}":
        raise DriverError("ARTIFACT_INVALID", "MCE frozen bundle/handle identity changed")
    manifest = driver.artifact_store.resolve(handle)
    bound = {
        "source_tree": driver.source_tree,
        "source_manifest_sha256": _content_hash(driver.source_manifest),
        "license_manifest_sha256": _content_hash(driver.license_manifest),
        "provenance_manifest_sha256": _content_hash(driver.provenance_manifest),
        "semantic_patch_manifest_sha256": driver.semantic_patch_manifest_sha256,
        "storage_fingerprint_sha256": driver.storage_fingerprint_sha256,
    }
    expected = _artifact_bundle(
        files=manifest["files"], selected_iteration=_nonnegative_int(frozen, "selected_iteration"),
        artifact_manifest_sha256=_content_hash(manifest["files"]),
        skill_sha256=_text(frozen, "skill_sha256"),
        split_audit_sha256=_text(payload, "split_audit_sha256"),
        interface_signature_sha256=_content_hash(driver.interface_signature),
        initial_context_sha256=_content_hash(driver.initial_context_identity),
        environment_identity_sha256=_content_hash(driver.environment_identity),
        model_fingerprints_sha256=_content_hash(driver.model_fingerprints),
        controls_sha256=_content_hash(payload["controls"]),
        runtime_binding_sha256=_text(payload, "runtime_binding_sha256"),
        semantic_patch_manifest_sha256=driver.semantic_patch_manifest_sha256, bound=bound,
    )
    if _canonical_json(frozen) != _canonical_json(expected):
        raise DriverError("ARTIFACT_INVALID", "MCE frozen bundle dependencies changed")
    raw_controls = _mapping(payload, "evaluation_controls")
    if set(raw_controls) != set(MCEFinalControls.__dataclass_fields__) | {"seed"}:
        raise DriverError("BUDGET_INVALID", "MCE final control schema drifted")
    seed = _nonnegative_int(raw_controls, "seed")
    try:
        controls = MCEFinalControls(**{key: value for key, value in raw_controls.items() if key != "seed"})
        examples = payload["test_examples"]
        test_hash = validate_final_split(
            _mapping(payload, "split_audit"), _text(payload, "split_audit_sha256"), examples,
        )
    except (TypeError, KeyError, ValidationError) as exc:
        raise DriverError("SPLIT_IDENTITY_MISMATCH", "MCE final split/control validation failed") from exc
    if test_hash != payload.get("test_audit_sha256") or request["state_hash"] != _content_hash({
        "artifact": digest, "test": test_hash, "controls": raw_controls,
    }):
        raise DriverError("SPLIT_IDENTITY_MISMATCH", "MCE final state hash does not bind examples/budgets")
    if driver.oracle_root is None:
        raise DriverError("CONFIG_INVALID", "MCE final evaluation requires a persistent private oracle root")
    evidence = driver.oracle_root / f"mce-final-{request['request_id']}"
    try:
        evidence.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise DriverError("ATTEMPT_ALREADY_EXISTS", "MCE final attempt already has a claim; inspect it instead of retrying") from exc
    _save(evidence / "claim.json", request)
    test_path = evidence / "test.jsonl"
    _write_native_jsonl(test_path, examples)
    supervisor = TrustedEvaluationSupervisor(
        model_calls=controls.max_model_calls, tool_calls=controls.max_tool_calls,
        wall_seconds=controls.max_wall_seconds, output_tokens=controls.max_output_tokens,
        output_tokens_per_call=controls.max_output_tokens_per_call,
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="agentinputlab-mce-final-", dir=driver.workspace_root,
        ) as folder:
            workspace = Path(folder).resolve()
            frozen_root = driver.artifact_store.materialize(handle, workspace=workspace)
            run = driver.runtime.evaluate_frozen(
                workspace=workspace, artifact_root=frozen_root,
                selected_iteration=frozen["selected_iteration"], test_path=test_path,
                controls={"seed": seed, "test_limit": len(examples),
                          "max_wall_seconds": controls.max_wall_seconds,
                          "max_artifact_files": payload["controls"]["max_artifact_files"],
                          "max_artifact_bytes": payload["controls"]["max_artifact_bytes"]},
                model_fingerprints=driver.model_fingerprints, supervisor=supervisor,
            )
        if driver.artifact_store.resolve(handle) != manifest or (
            run["frozen_artifact_manifest_sha256"] != frozen["artifact_manifest_sha256"]
            or run["test_oracle_sha256"] != _sha256_file(test_path)
            or run["adaptation_performed"] is not False
            or run["generated_code_imported_by_worker"] is not False
        ):
            raise DriverError("ARTIFACT_INVALID", "MCE frozen evaluation changed its bound inputs", consumed=True)
        raw = run["evaluation"]
        _save(evidence / "native-evaluation.json", raw)
        rows_by_id = {}
        for status, collection in (("scored", raw["results"]), ("failed", raw["errors"])):
            for row in collection:
                identifier = row["sample"]["id"]
                if identifier in rows_by_id:
                    raise AlgorithmOutputError("MCE native final evaluator returned duplicate IDs")
                rows_by_id[identifier] = {
                    "example_id": identifier, "status": status,
                    "metrics": row["evaluation"]["metrics"] if status == "scored" else {},
                    "error": str(row["error"])[:4096] if status == "failed" else None,
                }
        if set(rows_by_id) != {row["example_id"] for row in examples}:
            raise AlgorithmOutputError("MCE native final evaluator omitted/substituted IDs")
        rows = validate_final_summary(
            [rows_by_id[row["example_id"]] for row in examples], raw["summary"], examples,
        )
        primary = raw["summary"]["primary_metric"]
        result = {
            "artifact_bundle_sha256": digest, "test_audit_sha256": test_hash,
            "rows": rows, "native_summary": raw["summary"],
            "all_positions_primary_metric": sum(row["metrics"].get(primary, 0.0) for row in rows) / len(rows),
            "output_budget": run["output_budget"],
            "native_evaluation_sha256": _sha256_file(evidence / "native-evaluation.json"),
            "evidence_id": evidence.name,
            "adaptation_performed": False, "host_code_executed": False,
        }
        _save(evidence / "terminal.json", {"status": "evaluated", "result": result,
                                          "usage": supervisor.usage(), "controls": asdict(controls)})
        return {"status": "ok", "request_id": request["request_id"], "upstream_commit": UPSTREAM_COMMIT,
                "protocol_version": PROTOCOL, "payload": result, "usage": supervisor.usage()}
    except Exception as exc:
        code = exc.code if isinstance(exc, DriverError) else (
            "UPSTREAM_OUTPUT_INVALID" if isinstance(exc, AlgorithmOutputError) else "UPSTREAM_EXECUTION_FAILED"
        )
        _save(evidence / "terminal.json", {
            "status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "failure_code": code, "usage": supervisor.usage(),
            "output_budget": supervisor.output_budget_receipt(), "controls": asdict(controls),
            "test_example_ids": [row["example_id"] for row in examples],
        })
        raise DriverError(
            code, "MCE frozen final attempt failed; raw evidence retained",
            consumed=bool(supervisor.usage()["model_calls"] or supervisor.usage()["tool_calls"]),
            usage=supervisor.usage(),
        ) from exc
