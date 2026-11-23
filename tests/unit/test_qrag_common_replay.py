"""Strict saved-policy binding; fixtures are not real-model acceptance."""
import json

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.acquire.qrag.common_support import QRAGPolicyRequest
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from tests.unit.test_qrag_file_configuration import policy_response


def setup(tmp_path):
    rows = tuple(RetrievalResult(document=Document(doc_id=str(i), text=f"text{i}",
                 corpus_revision="fixture"), rank=i+1, score=1.0) for i in range(2))
    request = QRAGPolicyRequest("question", rows, 17)
    identity = {"runtime": "synthetic-fixture-only"}
    response = policy_response([{"document_id": r.document.doc_id, "text": r.document.text} for r in rows])
    saved = {"schema": "agentinputlab.qrag.common-policy-replay.v1",
             "request": to_jsonable(request), "response": response, "policy_identity": identity}
    return request, saved, identity


def config(tmp_path, saved, identity):
    import hashlib
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(saved))
    settings = {"kind": "qrag_common_support_replay", "policy_replay_path": str(path),
                "policy_replay_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "policy_identity_sha256": content_hash(identity)}
    return RunConfig("acquire.qrag.common_support", runtime=settings)


def test_reuse_has_explicit_identity_and_original_usage(tmp_path):
    request, saved, identity = setup(tmp_path)
    runtime = config(tmp_path, saved, identity)._runtime(load=True)
    response = runtime.select(request)
    assert response["usage"]["embedding_calls"] == 6
    assert runtime.fingerprint()["new_policy_calls"] == 0
    assert runtime.fingerprint()["live_fallback"] is False


@pytest.mark.parametrize("field,value", [("question", "other"), ("seed", 18), ("max_steps", 3)])
def test_saved_request_mutation_fails_without_fallback(tmp_path, field, value):
    request, saved, identity = setup(tmp_path)
    saved["request"][field] = value
    runtime = config(tmp_path, saved, identity)._runtime(load=True)
    with pytest.raises(ValidationError, match="does not match"):
        runtime.select(request)


@pytest.mark.parametrize("field,value", [("score", 2.0), ("rank", 2)])
def test_full_candidate_dto_is_bound(tmp_path, field, value):
    request, saved, identity = setup(tmp_path)
    saved["request"]["candidates"][0][field] = value
    with pytest.raises(ValidationError, match="does not match"):
        config(tmp_path, saved, identity)._runtime(load=True).select(request)


def test_identity_and_candidate_text_cannot_be_substituted(tmp_path):
    request, saved, identity = setup(tmp_path)
    with pytest.raises(ValidationError, match="identity mismatch"):
        config(tmp_path, saved, {"runtime": "wrong"})._runtime(load=True)
    saved["response"]["candidate_text_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="text checksum"):
        config(tmp_path, saved, identity)._runtime(load=True).select(request)


def test_shape_validation_does_not_read_and_paths_resolve(tmp_path):
    settings = {"kind": "qrag_common_support_replay", "policy_replay_path": "absent.json",
                "policy_replay_sha256": "a" * 64, "policy_identity_sha256": "b" * 64}
    file = tmp_path / "run.json"
    file.write_text(json.dumps({"release_id": "acquire.qrag.common_support", "runtime": settings}))
    run = RunConfig.from_file(file)
    assert run.runtime["policy_replay_path"] == str(tmp_path / "absent.json")
    assert run.validate()["valid"]
    with pytest.raises(ValidationError, match="supported only"):
        RunConfig("baseline.no_op", runtime=settings).validate()
    with pytest.raises(ValidationError):
        RunConfig("acquire.qrag.common_support", runtime={**settings, "live_fallback": True}).validate()
