from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[2] / "tools/summarize_acceptance_evidence.py"
    spec = importlib.util.spec_from_file_location("summary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_real_receipt_inventory_retains_failures_repeated_cases_and_zero_scores(tmp_path):
    job = {
        "model": {"id": "fixture-model"},
        "release": "baseline.no_op",
        "benchmark": "qasper.v0_3_dev",
        "case_id": "same-case",
        "seed": 17,
    }
    for phase, status in (
        ("qa-phase1", "failed_terminal"),
        ("qa-phase1-repair", "scored_terminal"),
    ):
        write(tmp_path / phase / "manifest.json", {"jobs": [job]})
        write(
            tmp_path / phase / "terminals/one.json",
            {"job": job, "status": status, "calls": [], "metrics": {"answer_f1": 0.0}},
        )
    report = load_module().summarize(tmp_path)
    row = report["rows"][0]
    assert row["attempt_count"] == 2
    assert row["distinct_case_count"] == 1
    assert row["terminal_statuses"] == {"failed_terminal": 1, "scored_terminal": 1}
    assert report["full_acceptance_complete"] is False
    assert report["publication_ready"] is False
    assert row["all_applicable_profiles_certified"] is False


def test_unknown_identity_is_visible_and_native_fit_is_separate(tmp_path):
    phase = tmp_path / "fit-phase2"
    write(phase / "manifest.json", {"models": []})
    write(phase / "model/fit-terminal.json", {"status": "fit_completed"})
    write(
        phase / "model/artifact.json",
        {"release_id": "recipe.ace.offline", "artifact_id": "fixture"},
    )
    write(
        phase / "terminals/unknown.json",
        {"job": {"case_id": "unknown"}, "status": "scored_terminal"},
    )
    report = load_module().summarize(tmp_path)
    assert report["mapped_terminal_attempts"] == 0
    assert len(report["native_fit_receipts"]) == 1
    assert len(report["warnings"]) == 1
    assert "manual source mapping required" in report["warnings"][0]["reason"]


def test_source_reviewed_checkpoint_and_flat_native_evaluation_are_mapped(tmp_path):
    phase = tmp_path / "native-phase3"
    write(phase / "manifest.json", {"jobs": []})
    write(
        phase / "one/terminal.json",
        {
            "job": {
                "model": {"id": "fixture"},
                "release": "clarify.udcs.repo_native_prompt_route",
                "benchmark": "tooltalk.full",
                "checkpoint_id": "protected:fixed-case",
            },
            "status": "scored_terminal",
            "calls": [],
        },
    )
    write(
        phase / "model/case/terminal.json",
        {
            "model": "fixture",
            "release": "recipe.ace.eval_only",
            "benchmark": "qasper.v0_3_dev",
            "case_id": "fixed-case",
            "status": "scored_terminal",
            "calls": [],
        },
    )
    report = load_module().summarize(tmp_path)
    assert report["mapped_terminal_attempts"] == 2
    identities = {e["identity_schema"] for row in report["rows"] for e in row["evidence"]}
    assert identities == {"native_provider_checkpoint", "flat_native_artifact_evaluation"}
    assert "tooltalk.full" in report["benchmarks_with_mapped_terminal_evidence"]


def test_domain_shards_without_parent_manifest_are_mapped_once(tmp_path):
    for cell in ("bright.robotics", "bright.stackoverflow"):
        phase = tmp_path / "bright-phase23" / cell
        job = {
            "model": {"id": "fixture"},
            "release": "baseline.no_op",
            "benchmark": cell,
            "case_id": "0",
        }
        write(phase / "manifest.json", {"jobs": [job]})
        write(phase / "terminals/one.json", {"job": job, "status": "scored_terminal"})
    report = load_module().summarize(tmp_path)
    assert report["mapped_terminal_attempts"] == 2
    assert len(report["phases"]) == 2
    assert {p["phase"] for p in report["phases"]} == {
        "bright-phase23/bright.robotics",
        "bright-phase23/bright.stackoverflow",
    }
    assert not report["warnings"]
    assert report["publication_ready"] is False


def test_native_retail_checkpoint_terminal_is_visible_without_becoming_a_pass(tmp_path):
    phase = tmp_path / "retail-phase24"
    write(phase / "manifest.json", {"jobs": []})
    write(
        phase / "one/terminal.json",
        {
            "job": {
                "model": {"id": "fixture"},
                "release": "baseline.no_op",
                "benchmark": "tau3.retail",
                "checkpoint_id": "5",
            },
            "status": "capture_dependency_terminal",
            "model_calls": 0,
        },
    )
    report = load_module().summarize(tmp_path)
    assert report["mapped_terminal_attempts"] == 1
    assert report["rows"][0]["evidence"][0]["identity_schema"] == "native_provider_checkpoint"
    assert not report["warnings"]
    assert report["publication_ready"] is False


def test_distinct_predictor_configurations_and_unknown_native_cost_stay_visible(tmp_path):
    phase = tmp_path / "schema-phase28"
    write(phase / "manifest.json", {"jobs": []})
    for mode in ("json_object", "json_schema"):
        write(phase / "terminals" / (mode + ".json"), {
            "job": {"model": {"id": "fixture"}, "release": "baseline.no_op",
                    "benchmark": "qasper.v0_3_dev", "case_id": "same-case",
                    "builder_parameters": {"structured_output_mode": mode}},
            "status": "scored_terminal", "calls": [],
        })
    report = load_module().summarize(tmp_path)
    evidence = report["rows"][0]["evidence"]
    assert len({e["job_configuration_sha256"] for e in evidence}) == 2
    assert {e["declared_release_configuration"]["builder_parameters"]["structured_output_mode"]
            for e in evidence} == {"json_object", "json_schema"}
    assert all(e["reported_model_calls"] is None for e in evidence)
    assert report["rows"][0]["attempt_count"] == 2
    assert report["rows"][0]["distinct_case_count"] == 1


def test_saved_state_export_recovery_is_not_an_extra_scientific_attempt(tmp_path):
    job = {"model": {"id": "fixture"}, "release": "baseline.no_op",
           "benchmark": "toolsandbox.common_support", "checkpoint_id": "same-case"}
    for phase, status, extra in (
        ("native-phase1", "failed_terminal", {}),
        ("native-phase1-export", "scored_terminal", {
            "evidence_role": "native_score_export_recovery", "model_calls": 0,
            "additional_attempts": 0, "original_terminal_sha256": "a" * 64,
            "saved_ending_context_sha256": "b" * 64}),
    ):
        write(tmp_path / phase / "manifest.json", {"jobs": [job]})
        write(tmp_path / phase / "case/terminal.json", {"job": job, "status": status, **extra})
    report = load_module().summarize(tmp_path)
    row = report["rows"][0]
    assert row["attempt_count"] == row["distinct_case_count"] == 1
    assert row["receipt_count"] == 2 and row["derived_score_receipts"] == 1
    assert report["mapped_terminal_attempts"] == 1
    assert report["mapped_terminal_receipts"] == 2
    assert row["terminal_statuses"] == {"failed_terminal": 1, "scored_terminal": 1}
    recovered = next(e for e in row["evidence"] if e["evidence_role"] == "native_score_export_recovery")
    assert recovered["reported_model_calls"] == recovered["reported_additional_case_attempts"] == 0
    assert recovered["saved_ending_context_sha256"] == "b" * 64


def test_posthoc_leaf_selection_is_not_an_extra_model_attempt(tmp_path):
    write(tmp_path / "rqrag-phase1/manifest.json", {"jobs": []})
    for mode in ("confidence", "ppl", "ensemble"):
        write(tmp_path / f"rqrag-phase1/terminals/{mode}.json", {
            "job": {"model": {"id": "fixture"}, "release": "recipe.rqrag.method_tree_search",
                    "benchmark": "qasper.v0_3_dev", "case_id": "same-case", "selection_mode": mode},
            "status": "scored_terminal", "calls": [],
            "evidence_role": "execution_terminal" if mode == "confidence" else "derived_leaf_selection",
            "additional_attempts": int(mode == "confidence"),
        })
    report = load_module().summarize(tmp_path)
    assert report["mapped_terminal_attempts"] == 1 and report["mapped_terminal_receipts"] == 3
    assert report["rows"][0]["derived_analysis_receipts"] == 2
