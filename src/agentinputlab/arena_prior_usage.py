"""Explicit historical budget debit for cached arena continuations.

Receipts are caller-supplied provenance, not signatures or new model usage.
They cannot increase a budget, supply predictions, or replace native scoring.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agentinputlab.configuration import _known, _object, _unique
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.fit_configuration import _reference

RELEASES = frozenset({"rewrite.infocqr.cached_rw_zsl", "rewrite.infocqr.cached_rw_fsl",
                      "rewrite.infocqr.cached_edit", "acquire.adaptive_rag.selector_replay"})


def _checked_path(pin, root):
    pin = _reference(pin, "prior usage source")
    path = Path(pin["path"]).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
    except OSError:
        raise ValidationError("prior usage source is unavailable") from None
    if digest != pin["sha256"]:
        raise ValidationError("prior usage source checksum differs")
    return path


def load(pin, run):
    """Validate all declared receipt identities before any native/model work."""
    from workers.official_arenas.protocol import merge_usage

    if run.release_id not in RELEASES:
        raise ValidationError("prior_intervention_usage requires a supported cached profile")
    path = _checked_path(pin, Path.cwd())
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValidationError("prior usage receipt exceeds its size limit")
        value = _object(json.loads(raw, object_pairs_hook=_unique), "prior usage receipt")
    except (OSError, ValueError, RecursionError):
        raise ValidationError("cannot read prior usage receipt") from None
    _known(value, {"schema_version", "release_id", "model_id", "seed", "original_budget", "cases"}, "prior usage receipt")
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("release_id") != run.release_id
            or value.get("model_id") != run.backend.get("model_id")
            or type(value.get("seed")) is not int or value["seed"] != run.seed
            or value.get("original_budget") != to_jsonable(run.budget)):
        raise ValidationError("prior usage receipt run identity or budget differs")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValidationError("prior usage receipt requires nonempty cases")
    selector = run.release_id == "acquire.adaptive_rag.selector_replay"
    checksum_field = "artifact_sha256" if selector else "cache_sha256"
    cache_sha = run.artifacts.get("records" if selector else "cache", {}).get("sha256")
    if selector and (not cache_sha or run.parameters.get("artifact_sha256") != cache_sha):
        raise ValidationError("selector prior usage requires matching pinned records and artifact checksum")
    result = {}
    for entry in cases:
        entry = _object(entry, "prior usage case")
        _known(entry, {"case_id", "input_state_sha256", "output_state_sha256", checksum_field, "source", "usage"}, "prior usage case")
        case = entry.get("case_id")
        if not isinstance(case, str) or not case.strip() or case in result:
            raise ValidationError("prior usage case IDs must be unique nonempty strings")
        for key in ("input_state_sha256", "output_state_sha256", checksum_field):
            val = entry.get(key)
            if not isinstance(val, str) or len(val) != 64 or any(c not in "0123456789abcdef" for c in val):
                raise ValidationError("prior usage state/cache checksums are invalid")
        if entry[checksum_field] != cache_sha:
            raise ValidationError("prior usage cache checksum differs")
        _checked_path(entry.get("source"), path.parent)
        usage = _object(entry.get("usage"), "prior usage")
        _known(usage, set(merge_usage()), "prior usage")
        try:
            usage = merge_usage(usage)
        except RuntimeError as exc:
            raise ValidationError(str(exc)) from None
        if usage["model_calls"] < 1:
            raise ValidationError("prior usage must describe a consumed model attempt")
        for field, maximum in (("model_calls", run.budget.max_model_calls),
                               ("output_tokens", run.budget.max_output_tokens),
                               ("tool_calls", run.budget.max_tool_calls),
                               ("retrieval_calls", run.budget.max_retrieval_calls)):
            if usage[field] > maximum:
                raise ValidationError("prior usage exceeds the original budget")
        if run.budget.max_context_tokens is not None and usage["input_tokens"] > run.budget.max_context_tokens:
            raise ValidationError("prior usage exceeds the original context budget")
        result[case] = {**entry, "usage": usage}
    return result


def validate_case(entries, case_id, state):
    entry = entries.get(case_id)
    if entry is None or entry["input_state_sha256"] != state.state_hash:
        raise ValidationError("prior usage missing case or input state differs")
    return entry


def debit(entry, record):
    """Keep cache's zero model cost; debit historical work only for continuation."""
    from workers.official_arenas.protocol import merge_usage

    current = to_jsonable(record.usage)
    if any(current[k] for k in ("model_calls", "input_tokens", "output_tokens", "tool_calls", "retrieval_calls", "embedding_calls")):
        raise ValidationError("prior usage cannot also charge a newly executed intervention")
    if record.output_state_hash != entry["output_state_sha256"]:
        raise ValidationError("prior usage materialized output state differs")
    return merge_usage(entry["usage"], current)
