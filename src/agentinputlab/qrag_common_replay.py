"""Explicit, checksum-bound replay of a common-support Q-RAG policy draw.

This is not a live policy or an inference fallback. The original trajectory's
logical usage is retained, while the runtime fingerprint marks physical reuse.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable


def runtime(config, *, load):
    value = config.runtime
    if config.release_id != "acquire.qrag.common_support":
        raise ValidationError("qrag_common_support_replay is supported only by acquire.qrag.common_support")
    if set(value) != {"kind", "policy_replay_path", "policy_replay_sha256", "policy_identity_sha256"}:
        raise ValidationError("Q-RAG common replay requires path, file SHA-256 and policy identity SHA-256")
    path = value["policy_replay_path"]
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ValidationError("Q-RAG common replay requires a local path")
    for key in ("policy_replay_sha256", "policy_identity_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValidationError(f"Q-RAG {key} must be a SHA-256 digest")
    return SavedCommonPolicy(Path(path), value["policy_replay_sha256"],
                             value["policy_identity_sha256"]) if load else None


class SavedCommonPolicy:
    def __init__(self, path, digest, identity_digest):
        from agentinputlab.artifacts import load_pinned_json_mapping

        saved = load_pinned_json_mapping(path, expected_sha256=digest)
        if (set(saved) != {"schema", "request", "response", "policy_identity"}
                or saved.get("schema") != "agentinputlab.qrag.common-policy-replay.v1"
                or not isinstance(saved.get("request"), Mapping)
                or not isinstance(saved.get("response"), Mapping)
                or not isinstance(saved.get("policy_identity"), Mapping)
                or not saved["policy_identity"]):
            raise ValidationError("invalid Q-RAG common policy replay schema")
        if content_hash(saved["policy_identity"]) != identity_digest:
            raise ValidationError("Q-RAG common replay policy identity mismatch")
        self.saved = freeze_mapping(saved)
        self.digest = digest
        self.identity_digest = identity_digest

    def select(self, request):
        from agentinputlab.algorithms.acquire.qrag.common_support import (
            _policy_candidate_text,
            _validate_response,
        )

        # Full DTO binding includes candidate ranks, scores, provenance and text;
        # a coincidentally identical question or selected ID is insufficient.
        if content_hash(to_jsonable(request)) != content_hash(self.saved["request"]):
            raise ValidationError("saved Q-RAG common policy does not match question/candidates/seed/horizon; live fallback disabled")
        response = self.saved["response"]
        _validate_response(response, tuple(row.document.doc_id for row in request.candidates))
        expected_text = content_hash(tuple(_policy_candidate_text(row) for row in request.candidates))
        if response.get("candidate_text_sha256") != expected_text:
            raise ValidationError("Q-RAG common replay candidate text checksum mismatch")
        return response

    def fingerprint(self):
        return freeze_mapping({"runtime": "qrag_common_support_saved_policy.v1",
            "policy_execution": "explicit_saved_trajectory", "new_policy_calls": 0,
            "policy_replay_sha256": self.digest,
            "policy_identity_sha256": self.identity_digest,
            "original_policy_identity": self.saved["policy_identity"],
            "replay_usage_is_original_logical_cost": True, "live_fallback": False})
