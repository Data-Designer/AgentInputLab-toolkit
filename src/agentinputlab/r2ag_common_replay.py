"""Explicit saved R²AG anchor/feature reuse, bound to the full runtime request."""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable


def runtime(config, *, load):
    value = config.runtime
    if config.release_id != "acquire.r2ag.common_support":
        raise ValidationError("r2ag_common_support_replay requires acquire.r2ag.common_support")
    if set(value) != {"kind", "response_path", "response_sha256", "runtime_identity_sha256"}:
        raise ValidationError("R²AG replay requires response path, file SHA-256 and runtime identity SHA-256")
    if not isinstance(value["response_path"], str) or not value["response_path"].strip() or "\x00" in value["response_path"]:
        raise ValidationError("R²AG replay requires a local response path")
    for key in ("response_sha256", "runtime_identity_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValidationError(f"R²AG {key} must be a SHA-256 digest")
    return SavedRuntime(Path(value["response_path"]), value["response_sha256"],
                        value["runtime_identity_sha256"]) if load else None


class SavedRuntime:
    def __init__(self, path, digest, identity_digest):
        from agentinputlab.artifacts import load_pinned_json_mapping

        saved = load_pinned_json_mapping(path, expected_sha256=digest)
        if (set(saved) != {"schema", "request", "response", "runtime_identity"}
                or saved["schema"] != "agentinputlab.r2ag.common-response-replay.v1"
                or any(not isinstance(saved[k], Mapping) for k in ("request", "response", "runtime_identity"))
                or not isinstance(saved["runtime_identity"].get("reader"), Mapping)
                or not isinstance(saved["runtime_identity"].get("feature_encoder"), Mapping)):
            raise ValidationError("invalid R²AG saved response schema or component identities")
        if content_hash(saved["runtime_identity"]) != identity_digest:
            raise ValidationError("R²AG saved runtime identity mismatch")
        self.saved, self.digest = freeze_mapping(saved), digest

    def run(self, request):
        from agentinputlab.algorithms.acquire.r2ag.common_support import _validated_runtime_response

        if content_hash(to_jsonable(request)) != content_hash(self.saved["request"]):
            raise ValidationError("saved R²AG request differs in query/documents/seed/budget; live fallback disabled")
        response = self.saved["response"]
        _validated_runtime_response(response, len(request.documents))
        return response

    def fingerprint(self):
        return freeze_mapping({**dict(self.saved["runtime_identity"]),
            "runtime": "r2ag_common_support_saved_response.v1", "response_sha256": self.digest,
            "original_runtime_identity": self.saved["runtime_identity"],
            "new_reader_calls": 0, "replay_usage_is_original_logical_cost": True,
            "live_fallback": False})
