"""Pinned Tau capture reuse: preserve failed attempts, never generate on a miss."""
from __future__ import annotations

import hashlib
from pathlib import Path

from agentinputlab.artifacts.json_file import load_pinned_json_mapping
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.fit_configuration import _reference


def _child(root, reference):
    pin = _reference(reference, "capture bundle member")
    path = Path(pin["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValidationError("capture bundle members must use contained relative paths")
    source = root / path
    if not source.resolve().is_relative_to(root.resolve()):
        raise ValidationError("capture bundle member escapes its directory")
    return {**pin, "path": str(source)}


def _semantic(config):
    # Transport relocation does not change model identity. All scientific
    # settings, native source/scorer locks and role budgets must still match.
    value = to_jsonable(config)
    value.pop("limit", None)
    value.pop("capture_task_ids", None)
    for key in ("llm_args_agent", "llm_args_user", "llm_args_judge"):
        if key in value:
            if value[key].get("api_key") != "agentinputlab-native-role-routing":
                raise ValidationError("capture bundle must use the public role-routing configuration")
            value[key].pop("api_base", None)
    return value


def load(reference):
    """Load an exported bundle and every member under caller-trusted hashes."""
    from agentinputlab.fit_preparation import _read

    pin = _reference(reference, "capture_bundle")
    source = Path(pin["path"]).expanduser().absolute()
    bundle = load_pinned_json_mapping(source, expected_sha256=pin["sha256"])
    if (bundle.get("schema") != "agentinputlab.tau.capture_bundle.v1"
            or bundle.get("arena") not in {"tau_banking", "tau_retail"}):
        raise ValidationError("unsupported Tau capture bundle")
    config = bundle.get("provider_configuration")
    cases = bundle.get("cases")
    if not isinstance(config, dict) or not isinstance(cases, dict) or not cases:
        raise ValidationError("Tau capture bundle configuration/cases missing")
    _semantic(config)
    model = bundle.get("target_model")
    if (not isinstance(model, dict) or not isinstance(model.get("model_id"), str)
            or not isinstance(model.get("deployment_identity"), dict)
            or "openai/" + model["model_id"] != config.get("llm_agent")
            or content_hash(model["deployment_identity"]) != config.get("locked_dependency_hashes", {}).get("agent_model")):
        raise ValidationError("capture bundle target model is not bound to its native configuration")
    inventory = _child(source.parent, bundle.get("inventory"))
    raw, rows = _read(inventory["path"], 16 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != inventory["sha256"]:
        raise ValidationError("capture inventory checksum mismatch")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValidationError("capture inventory must be a JSON array of records")
    ids = [r.get("checkpoint_id") for r in rows]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValidationError("capture inventory has invalid or duplicate IDs")
    configs_pin = _child(source.parent, bundle.get("native_configurations"))
    configs = load_pinned_json_mapping(configs_pin["path"], expected_sha256=configs_pin["sha256"],
                                       max_bytes=16 * 1024 * 1024)
    if set(configs) != set(cases):
        raise ValidationError("capture case and configuration IDs differ")
    failures, captured = {}, []
    for key, case in cases.items():
        if not isinstance(case, dict) or case.get("status") not in {"captured", "failed"}:
            raise ValidationError("capture bundle contains an unfinished case")
        if configs[key] != {**config, "limit": 1, "capture_task_ids": [key]}:
            raise ValidationError("capture case configuration differs from its bundle")
        receipt_pin = _child(source.parent, case.get("receipt"))
        receipt = load_pinned_json_mapping(receipt_pin["path"], expected_sha256=receipt_pin["sha256"])
        if receipt.get("case_id") != key or receipt.get("status") != case["status"]:
            raise ValidationError("capture receipt identity/status differs")
        if case["status"] == "captured":
            captured.append(key)
        else:
            failures[key] = {**receipt, "receipt": receipt_pin["path"], "reused_capture_failure": True}
    if set(ids) != set(captured):
        raise ValidationError("capture inventory does not match captured terminal IDs")
    return bundle, inventory, configs, failures


def reuse(native, config, chosen, available, work):
    from agentinputlab.arena_configuration import _save

    bundle, inventory, _, failures = load(native.capture_bundle)
    if bundle["arena"] != native.arena or _semantic(bundle["provider_configuration"]) != _semantic(config):
        raise ValidationError("saved capture model, budget, split, source or role configuration differs")
    if set(chosen) - set(bundle["cases"]):
        raise ValidationError("requested case absent from capture bundle; no automatic capture on a miss")
    _save(work / "terminal.json", {
        "status": "captured", "case_ids": available, "inventory_path": inventory["path"],
        "inventory_sha256": inventory["sha256"], "capture_bundle": native.capture_bundle,
        "capture_failures": {key: failures[key] for key in chosen if key in failures},
        "captured_case_ids": [key for key in chosen if key not in failures],
        "capture_reused": True, "new_capture_model_calls": 0,
        "capture_cost_scope": "refer_to_original_capture_receipts_not_charged_again",
        "automatic_retry": False,
    })
    return 0
