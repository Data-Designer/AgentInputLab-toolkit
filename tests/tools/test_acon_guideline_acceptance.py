"""Offline operator budget tests; no real acceptance claims."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from run_acon_guideline_acceptance import (
    NativeHistorySeam,
    StrictReusingBackend,
    ask,
    system_message_reuses,
)

from agentinputlab.backends import GenerationRequest, GenerationResponse, Usage
from agentinputlab.contracts.serialization import to_jsonable


def test_fixed_training_budget_blocks_before_transport():
    calls = [{}] * 8
    backend = SimpleNamespace(generate=lambda _: pytest.fail("must not call backend"))
    with pytest.raises(ValueError, match="before transport"):
        ask(backend, calls, "synthetic prompt")


def test_native_history_calls_have_bounded_seeded_request():
    requests = []

    def generate(request):
        requests.append(request)
        return GenerationResponse(
            text="synthetic output", usage=Usage(input_tokens=7, output_tokens=2)
        )

    assert ask(SimpleNamespace(generate=generate), [], "synthetic prompt") == "synthetic output"
    assert requests[0].max_tokens == 512 and requests[0].seed == 17


def test_empty_visible_output_is_a_failure_not_a_training_label():
    backend = SimpleNamespace(
        generate=lambda _: GenerationResponse(
            text="", usage=Usage(input_tokens=7, output_tokens=512)
        )
    )
    with pytest.raises(ValueError, match="empty final"):
        ask(backend, [], "synthetic prompt")


def test_unknown_usage_does_not_report_free_training():
    backend = SimpleNamespace(generate=lambda _: GenerationResponse(text="synthetic"))
    with pytest.raises(ValueError, match="unknown token"):
        ask(backend, [], "synthetic prompt")


def test_actual_empty_reader_response_can_be_scored_as_failure_without_retry():
    calls = []

    def generate(request):
        calls.append(request)
        return GenerationResponse(
            text="", finish_reason="length", usage=Usage(input_tokens=7, output_tokens=512)
        )

    assert ask(SimpleNamespace(generate=generate), [], "synthetic", allow_empty=True) == ""
    assert len(calls) == 1


def test_native_history_seam_preserves_bound_system_message():
    requests = []

    def generate(request):
        requests.append(request)
        return GenerationResponse(text="synthetic", usage=Usage(input_tokens=9, output_tokens=2))

    seam = NativeHistorySeam(SimpleNamespace(generate=generate), [])
    seam.system_message = "source-bound synthetic system fixture"
    seam.generate("synthetic user prompt")
    assert requests[0].system == seam.system_message
    assert requests[0].prompt == "synthetic user prompt" and requests[0].max_tokens == 512


def test_unchanged_training_response_reuse_never_contacts_a_model(tmp_path):
    request = GenerationRequest(prompt="synthetic unchanged question", seed=17)
    path = tmp_path / "saved-response.json"
    path.write_text(
        json.dumps(
            {
                "request": to_jsonable(request),
                "response": to_jsonable(
                    GenerationResponse(
                        text="synthetic saved answer",
                        model="fixture",
                        usage=Usage(input_tokens=7, output_tokens=3),
                    )
                ),
            }
        )
    )
    calls = []
    backend = StrictReusingBackend(
        reuses={1: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        journal=tmp_path / "journal",
        calls=calls,
        model_id="fixture",
        base_url="http://127.0.0.1:1/v1",
        deployment_identity={"revision": "synthetic"},
        transport=lambda *a: pytest.fail("strict reuse must not contact a model"),
    )
    assert backend.generate(request).text == "synthetic saved answer"
    assert len(calls) == 1 and "strict_reuse" in calls[0]
    assert (tmp_path / "journal/0001-reuse.json").exists()
    assert not list((tmp_path / "journal").glob("*-request.json"))


def test_precompression_failure_is_not_selected_for_implementation_retest(tmp_path):
    model = {"id": "synthetic", "service_directory": "old"}
    job = tmp_path / "job.json"
    terminal = tmp_path / "terminal.json"
    job.write_text(json.dumps({"model": model, "train": []}))
    terminal.write_text(json.dumps({"status": "failed_terminal", "physical_model_attempts": 2}))
    with pytest.raises(ValueError, match="no consumed native compression"):
        system_message_reuses(
            tmp_path,
            [],
            model,
            hashlib.sha256(job.read_bytes()).hexdigest(),
            hashlib.sha256(terminal.read_bytes()).hexdigest(),
        )


def test_identical_native_update_request_is_reused_even_when_other_analysis_changed(tmp_path):
    request = GenerationRequest(prompt="synthetic identical update", seed=19)
    p = tmp_path / "old-fit-call.json"
    p.write_text(
        json.dumps(
            {
                "request": to_jsonable(request),
                "response": to_jsonable(
                    GenerationResponse(
                        text="synthetic saved candidate",
                        model="fixture",
                        finish_reason="stop",
                        usage=Usage(input_tokens=11, output_tokens=3),
                    )
                ),
            }
        )
    )
    calls = [{} for _ in range(10)]
    backend = StrictReusingBackend(
        reuses={},
        fit_reuses=[{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}],
        journal=tmp_path / "new",
        calls=calls,
        model_id="fixture",
        base_url="http://127.0.0.1:1/v1",
        deployment_identity={"revision": "synthetic"},
        transport=lambda *args: pytest.fail("identical update must not be physically regenerated"),
    )
    assert backend.generate(request).text == "synthetic saved candidate"
    assert "strict_reuse" in calls[-1] and not list((tmp_path / "new").glob("*-request.json"))


def test_identical_known_budget_failure_is_preserved_not_retried(tmp_path):
    request = GenerationRequest(prompt="synthetic known failure", seed=19)
    p = tmp_path / "old-failure.json"
    p.write_text(
        json.dumps(
            {
                "request": to_jsonable(request),
                "response": to_jsonable(
                    GenerationResponse(
                        text="",
                        model="fixture",
                        finish_reason="length",
                        usage=Usage(input_tokens=11, output_tokens=512),
                    )
                ),
            }
        )
    )
    calls = []
    backend = StrictReusingBackend(
        reuses={},
        fit_reuses=[{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}],
        journal=tmp_path / "new",
        calls=calls,
        model_id="fixture",
        base_url="http://127.0.0.1:1/v1",
        deployment_identity={"revision": "synthetic"},
        transport=lambda *args: pytest.fail("known unchanged failure must not be retried"),
    )
    response = backend.generate(request)
    assert response.text == "" and response.finish_reason == "length"
    assert calls[-1]["strict_reuse"]["known_outcome_cache"]
    assert not list((tmp_path / "new").glob("*-request.json"))
