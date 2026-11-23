"""Explicit file binding for a source-locked BEIR compatibility sample.

This never downloads data or presents sampled-candidate scores as full-corpus
scores. The existing adapter verifies every source receipt and native scorer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agentinputlab.configuration import _known, _object, _unique
from agentinputlab.contracts.errors import ValidationError


def sources(value):
    data = _object(value, "beir_sample_sources")
    _known(data, {"root", "manifest_sha256"}, "beir_sample_sources")
    if (
        set(data) != {"root", "manifest_sha256"}
        or not isinstance(data["root"], str)
        or not data["root"].strip()
        or not isinstance(data["manifest_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", data["manifest_sha256"])
    ):
        raise ValidationError("BEIR sample requires root and a pinned manifest_sha256")
    return dict(root=str(Path(data["root"]).expanduser()), manifest_sha256=data["manifest_sha256"])


def read_sources(path):
    path = Path(path).expanduser().absolute()
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValidationError("BEIR sample configuration exceeds 1 MiB")
        data = sources(json.loads(raw, object_pairs_hook=_unique))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValidationError(
            "cannot read BEIR sample configuration: use a UTF-8 JSON object"
        ) from None
    if not Path(data["root"]).is_absolute():
        data["root"] = str(path.parent / data["root"])
    return data


def load(value, *, benchmark_id, split, accept_license_review):
    if benchmark_id not in {"beir.fever", "beir.cqadupstack"} or split != "test":
        raise ValidationError("BEIR sample binding supports only FEVER/CQADupstack test samples")
    if accept_license_review is not True:
        raise ValidationError("review BEIR sample data terms and set accept_license_review=True")
    from agentinputlab.benchmarks import BEIRSampleAdapter

    data = sources(value)
    adapter = BEIRSampleAdapter(**data, license_acknowledged=True)
    if adapter.benchmark_id != benchmark_id:
        raise ValidationError("BEIR sample benchmark identity mismatch")
    return adapter
