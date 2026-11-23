"""Opt-in scoring of original IRCoT rankings, separate from answer replay.

Operator-owned ranking receipts never become model context or new retrieval.
Checksums establish binding and integrity, not authorship of the supplied data.
"""

from collections.abc import Mapping
from pathlib import Path

from agentinputlab.algorithms.acquire.ircot_replay import IRCoTPredictionReplay, _sha
from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.benchmarks.retrieval import RetrievalPrediction
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

SCHEMA = "agentinputlab.ircot.archived_retrieval.v1"


def load(options, method, *, benchmark_id, split):
    if (not isinstance(options, Mapping) or set(options) != {"path", "sha256"}
            or not isinstance(options["path"], str) or not options["path"].strip()
            or not isinstance(method, IRCoTPredictionReplay)
            or method.benchmark != benchmark_id or method.split != split
            or not benchmark_id.startswith(("beir.", "bright.", "trec_dl."))
            or method.archive.manifest["origin_kind"] != "controlled_capture"):
        raise ValidationError("archived_retrieval requires a pinned ranking receipt and matching controlled IRCoT IR scope")
    path = Path(options["path"]).expanduser()
    if not path.is_absolute():
        raise ValidationError("archived_retrieval path must be absolute to avoid working-directory ambiguity")
    digest = _sha(options["sha256"], "ranking receipt checksum")
    data = load_pinned_json_mapping(path, expected_sha256=digest, max_bytes=8 * 1024 * 1024)
    if (set(data) != {"schema", "benchmark", "split", "prediction_manifest_sha256",
                     "generator_fingerprint_sha256", "dataset_fingerprint_sha256", "records"}
            or data["schema"] != SCHEMA or data["benchmark"] != benchmark_id or data["split"] != split
            or data["prediction_manifest_sha256"] != method.expected_manifest_sha256
            or data["generator_fingerprint_sha256"] != method.archive.generator_fingerprint_sha256):
        raise ArtifactError("archived retrieval schema, model, dataset scope or answer archive binding differs")
    _sha(data["dataset_fingerprint_sha256"], "ranking dataset fingerprint")
    records = data["records"]
    if not isinstance(records, Mapping) or not records:
        raise ArtifactError("archived retrieval requires original ranking records")
    for case_id, row in records.items():
        original = method.archive.records.get(case_id)
        if (original is None or not isinstance(row, Mapping)
                or set(row) != {"query", "input_state_hash", "origin_receipt_sha256", "answer_sha256", "ranked_ids"}
                or original["input_state_hash"] is None
                or any(row[k] != original[k] for k in ("query", "input_state_hash", "origin_receipt_sha256"))
                or row["answer_sha256"] != content_hash(original["answer"])):
            raise ArtifactError("archived ranking does not bind the exact original answer, input and inference receipt")
        ids = row["ranked_ids"]
        if (not isinstance(ids, (list, tuple)) or len(ids) > 10000
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(ids) != len(set(ids))):
            raise ArtifactError("archived ranking requires bounded, unique original document IDs")
    return freeze_mapping(dict(path=str(path), sha256=digest, data=data))


def prediction(settings, *, state, method, adapter, answer):
    # Detect a changed receipt even when an evaluation contains several cases.
    checked = load({k: settings[k] for k in ("path", "sha256")}, method,
                   benchmark_id=method.benchmark, split=method.split)
    data = checked["data"]
    if (adapter.benchmark_id != method.benchmark
            or not callable(getattr(adapter, "fingerprint", None))
            or content_hash(adapter.fingerprint()) != data["dataset_fingerprint_sha256"]):
        raise ArtifactError("archived ranking dataset/scorer fingerprint differs")
    row = data["records"].get(state.task_id)
    if (row is None or row["query"] != state.query or row["input_state_hash"] != state.state_hash
            or row["answer_sha256"] != content_hash(answer)):
        raise ArtifactError("original ranking unavailable for this exact replay; no retrieval fallback")
    return RetrievalPrediction.from_ranked_ids(row["ranked_ids"]), {
        "mode": SCHEMA,
        "receipt_sha256": settings["sha256"],
        "origin_receipt_sha256": row["origin_receipt_sha256"],
        "dataset_fingerprint_sha256": data["dataset_fingerprint_sha256"],
        "original_ranked_ids": list(row["ranked_ids"]),
        "score_mode": "descending_rank",
        "new_retrieval_calls": 0,
        "answer_used_as_query_or_evidence": False,
        "original_answer_profile_unchanged": True,
        "checksum_proves_authorship": False,
    }
