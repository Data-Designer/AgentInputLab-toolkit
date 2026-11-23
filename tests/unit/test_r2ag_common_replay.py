"""Saved real responses can be reused only for their exact request/identity."""
import hashlib
import json

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.acquire.r2ag import R2AGRuntimeRequest
from agentinputlab.contracts import AlgorithmOutputError, ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from tests.algorithms.unit.test_r2ag_common_support import FixedRetriever


def setup(tmp_path, *, mutate=None):
    docs = FixedRetriever().retrieve("query", top_k=1)
    request = R2AGRuntimeRequest(query="query", documents=docs, seed=17, max_new_tokens=96)
    saved = {"schema": "agentinputlab.r2ag.common-response-replay.v1",
        "request": to_jsonable(request), "response": {"status": "ok",
        "features": [{"rerank": 0.2, "neighbor": 0.0, "precedent": 1.0}],
        "reader": {"anchor": "Synthetic fixture only", "input_tokens": 20, "output_tokens": 5,
            "marker_positions": [4], "rformer_output_sha256": "a" * 64,
            "projection_output_sha256": "b" * 64},
        "usage": {"model_calls": 1, "embedding_calls": 1, "input_tokens": 20, "output_tokens": 5}},
        "runtime_identity": {"reader": {"fixture": True}, "feature_encoder": {"fixture": True}}}
    if mutate:
        mutate(saved)
    p = tmp_path / "saved.json"
    p.write_text(json.dumps(saved))
    return request, RunConfig("acquire.r2ag.common_support", runtime={
        "kind": "r2ag_common_support_replay", "response_path": str(p),
        "response_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "runtime_identity_sha256": content_hash(saved["runtime_identity"])})


def test_exact_saved_response_retains_cost_and_disables_live_fallback(tmp_path):
    request, config = setup(tmp_path)
    runtime = config._runtime(load=True)
    assert runtime.run(request)["usage"]["model_calls"] == 1
    assert runtime.fingerprint()["new_reader_calls"] == 0
    assert runtime.fingerprint()["live_fallback"] is False


@pytest.mark.parametrize("key,value", [("query", "different"), ("seed", 18), ("max_new_tokens", 95)])
def test_request_mutations_are_not_replayed(tmp_path, key, value):
    request, config = setup(tmp_path, mutate=lambda d: d["request"].update({key: value}))
    with pytest.raises(ValidationError, match="live fallback disabled"):
        config._runtime(load=True).run(request)


def test_candidate_provenance_is_bound(tmp_path):
    request, config = setup(tmp_path, mutate=lambda d: d["request"]["documents"][0]["document"].update(text="other"))
    with pytest.raises(ValidationError, match="request differs"):
        config._runtime(load=True).run(request)


def test_response_usage_and_feature_count_still_checked(tmp_path):
    request, config = setup(tmp_path, mutate=lambda d: d["response"]["usage"].update(output_tokens=6))
    with pytest.raises(AlgorithmOutputError, match="usage"):
        config._runtime(load=True).run(request)


def test_wrong_file_digest_and_release_fail_closed(tmp_path):
    _, config = setup(tmp_path)
    settings = dict(config.runtime)
    settings["response_sha256"] = "f" * 64
    with pytest.raises(ArtifactError, match="SHA-256|checksum"):
        RunConfig("acquire.r2ag.common_support", runtime=settings)._runtime(load=True)
    with pytest.raises(ValidationError, match="requires acquire.r2ag"):
        RunConfig("baseline.no_op", runtime=config.runtime).validate()
