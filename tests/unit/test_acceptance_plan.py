import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "acceptance_plan", Path(__file__).resolve().parents[2] / "tools/acceptance_plan.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_inventory_has_no_implicit_dispatch_or_pass():
    plan = module.build_plan(["synthetic-backbone-label"])
    assert plan["candidate_count"] == 87 * 31
    assert len({r["id"] for r in plan["rows"]}) == plan["candidate_count"]
    assert {r["benchmark"] for r in plan["rows"]} >= {"tooltalk.full", "tau3.retail"}
    assert not plan["publication_ready"]
    assert plan["dispatchable_count"] == plan["real_passed_count"] == 0
    assert all(r["applicability"] == "needs_review" for r in plan["rows"])
    assert all(r["real_acceptance"] == "not_run" for r in plan["rows"])
    assert all(not r["ready_for_dispatch"] for r in plan["rows"])
    assert all(r["data_acquisition_review"] is None for r in plan["rows"])
    assert all(r["dataset_revision"] is None for r in plan["rows"])
    assert all(r["dataset_source_binding"] is None for r in plan["rows"])
    assert all(r["benchmark_reference_revision"] for r in plan["rows"])
    assert any(r["execution_kind"] == "training_recipe" for r in plan["rows"])
    assert any(r["entrypoint"] is None for r in plan["rows"])


def test_data_policy_defers_large_assets_without_implicit_full_download_fallback():
    policy = module.build_plan(["synthetic-backbone-label"])["data_acquisition_policy"]
    assert policy["preferred_modes"] == [
        "reuse_existing_prepared_assets", "bounded_source_sample",
    ]
    assert policy["unavailable_bounded_source_action"] == "defer"
    assert policy["unknown_size_action"] == "defer_or_use_bounded_source_interface"
    assert policy["destination_free_space_check_required"]
    assert policy["sample_selection_before_model_outcomes_required"]
    assert set(policy["required_storage_budget_components"]) == {
        "source_bytes", "extracted_bytes", "normalized_bytes", "index_bytes",
        "temporary_and_cache_bytes",
    }
    for key in (
        "full_corpus_download_allowed", "full_corpus_index_build_allowed",
        "download_full_corpus_then_sample_allowed",
        "sampled_compatibility_is_full_benchmark_evaluation",
        "shared_research_assets_cleanup_allowed",
    ):
        assert policy[key] is False


@pytest.mark.parametrize("backbones", [[], ["x", "x"], [""], [" x"]])
def test_ambiguous_model_inventory_rejected(backbones):
    with pytest.raises(ValueError):
        module.build_plan(backbones)


def test_observed_receipts_do_not_cross_backbones_profiles_or_grant_passes():
    release = module.build_plan(["one"])["rows"][0]["release"]
    group = {"release": release, "benchmark": "tooltalk.full", "backbone": "one",
             "attempt_count": 2, "distinct_case_count": 1,
             "terminal_statuses": {"failed_terminal": 1, "scored_terminal": 1},
             "evidence": [{"status": "failed_terminal"}, {"status": "scored_terminal"}]}
    baseline = {**group, "release": "baseline.no_op"}
    unknown = {**group, "release": "unknown-release"}
    observed = {"schema": "agentinputlab.acceptance-observed-evidence.v1",
                "rows": [group, baseline, unknown], "warnings": [{"reason": "fixture"}]}
    plan = module.build_plan(["one", "two"], observed=observed)
    seen = [row for row in plan["rows"] if row["observed_receipts"] is not None]
    assert len(seen) == 1
    assert seen[0]["observed_receipts"]["attempt_count"] == 2
    assert seen[0]["observed_receipts"]["evidence"] == group["evidence"]
    assert seen[0]["real_acceptance"] == "observed_terminals_review_required"
    assert plan["real_passed_count"] == plan["dispatchable_count"] == 0
    assert not plan["publication_ready"]
    assert plan["baseline_observed_groups"] == [baseline]
    assert plan["unmatched_observed_groups"] == [unknown]
    assert plan["observed_warnings"] == observed["warnings"]
    assert all(not row["ready_for_dispatch"] for row in seen)


def test_ambiguous_observed_groups_rejected():
    group = {"release": "x", "benchmark": "y", "backbone": "z"}
    with pytest.raises(ValueError, match="duplicate"):
        module.build_plan(["z"], observed={
            "schema": "agentinputlab.acceptance-observed-evidence.v1", "rows": [group, group]})
    with pytest.raises(ValueError, match="schema"):
        module.build_plan(["z"], observed={"rows": [group]})


def test_catalog_reference_is_not_a_reviewed_data_asset_revision():
    plan = module.build_plan(["synthetic-backbone"])
    refs = {c.cell_id: c.source_revision for c in module.BenchmarkRegistry.default().list()}
    for row in plan["rows"]:
        assert row["benchmark_reference_revision"] == refs[row["benchmark"]]
        assert row["dataset_revision"] is None
        assert row["dataset_source_binding"] is None
    contract = plan["revision_contract"]
    assert not contract["cross_namespace_revision_equality_required"]
    assert set(contract["dataset_source_binding_requires"]) == {
        "repository", "revision", "content_sha256",
    }
