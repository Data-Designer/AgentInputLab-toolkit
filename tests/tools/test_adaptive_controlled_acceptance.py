from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from run_adaptive_controlled_acceptance import (
    CountedRouter,
    backend_for,
    load_boundary_selection,
    select_cases,
    state_for,
)

from agentinputlab.contracts import ContextBlock, FrozenAgentState
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import content_hash
from workers.adaptive_rag.driver import RouterResult


def test_actual_backend_constructor_uses_supported_timeout_before_any_dispatch():
    calls = []
    backend = backend_for(
        {"id": "test-model", "base_url": "http://127.0.0.1:9999/v1"}, "fixed-revision", calls
    )
    assert backend.timeout_seconds == 240 and backend.model_id == "test-model"
    assert calls == []


def test_fixed_cases_are_selected_before_outcomes():
    rows = [SimpleNamespace(example_id=f"{i:04d}") for i in reversed(range(106))]
    adapter = SimpleNamespace(examples=lambda _: rows)
    assert [r.example_id for r in select_cases(adapter)] == ["0102", "0103"]


def test_frozen_boundary_selection_uses_exact_real_ids_and_does_not_fill_missing_routes():
    rows = [SimpleNamespace(example_id="first"), SimpleNamespace(example_id="second")]
    adapter = SimpleNamespace(examples=lambda _: rows)
    assert select_cases(adapter, {"selected_cases": [{"case_id": "second", "route": "C"}]}) == [
        rows[1]
    ]
    for bad in (
        [{"case_id": "absent", "route": "C"}],
        [],
        [{"case_id": "first", "route": "B"}],
        [{"case_id": "first", "route": "C"}, {"case_id": "second", "route": "C"}],
    ):
        with pytest.raises(ValueError):
            select_cases(adapter, {"selected_cases": bad})


def test_boundary_loader_checks_pinned_source_bank_and_non_oracle_protocol(tmp_path):
    state = FrozenAgentState(query="Actual visible question")
    example = SimpleNamespace(example_id="case1", visible_state=lambda: state)
    adapter = SimpleNamespace(
        examples=lambda _: [example], prepared=SimpleNamespace(file_sha256={"input": "a"})
    )
    bank = {
        "reader_quality_used_for_selection": False,
        "gold_or_training_labels_exposed_to_router": False,
        "source_file_sha256": {"input": "a"},
        "candidates": [{"case_id": "case1", "question_sha256": content_hash(state.query)}],
    }
    bank_path = tmp_path / "manifest.json"
    bank_path.write_text(json.dumps(bank))
    selection = {
        "schema": "agentinputlab.adaptive-route-boundary-selection.v1",
        "benchmark": "qasper.v0_3_dev",
        "split": "dev",
        "no_reader_model_calls": True,
        "screening_attempts": 64,
        "source_bank_manifest_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "selected_cases": [{"case_id": "case1", "route": "C"}],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    spec = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded, digest = load_boundary_selection(spec, adapter)
    assert loaded == selection and digest == spec["sha256"]
    bank_path.write_text(json.dumps({**bank, "reader_quality_used_for_selection": True}))
    with pytest.raises(ArtifactError, match="checksum"):
        load_boundary_selection(spec, adapter)
    selection["source_bank_manifest_sha256"] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(selection))
    spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="non-oracle"):
        load_boundary_selection(spec, adapter)


def test_question_only_input_keeps_case_identity_and_does_not_expose_references():
    visible = FrozenAgentState(
        query="Original question",
        instruction="Original reader instruction",
        context=(ContextBlock("External evidence"),),
        metadata={"gold": "hidden"},
    )
    example = SimpleNamespace(example_id="case-1", visible_state=lambda: visible)
    result = state_for(example)
    assert result.query == visible.query and result.task_id == "case-1"
    assert (
        not result.context
        and not result.messages
        and not result.metadata
        and not result.instruction
    )
    assert result.budget.max_model_calls == 8 and result.budget.max_output_tokens == 4096


def test_router_attempts_and_failures_are_not_inferred_from_completed_jobs():
    class Native:
        calls = 0

        def component_manifest(self):
            return {"fixture_runtime": False, "implementation_class": "native"}

        def first_decoder_step(self, question, *, seed):
            self.calls += 1
            if question == "fail":
                raise RuntimeError("one attempt")
            return RouterResult({"A": 1.0, "B": 0.0, "C": 0.0}, 8)

    native = Native()
    router = CountedRouter(native)
    assert router.first_decoder_step("ok", seed=17).input_tokens == 8
    with pytest.raises(RuntimeError):
        router.first_decoder_step("fail", seed=17)
    assert native.calls == 2 and len(router.calls) == 2
    assert "result" in router.calls[0] and router.calls[1]["error_type"] == "RuntimeError"
    assert router.component_manifest()["fixture_runtime"] is False
