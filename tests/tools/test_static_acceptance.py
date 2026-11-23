"""Batch selection never silently adds profiles or changes old manifests."""

import pytest
from run_static_acceptance import RELEASES, selected_releases


def test_default_static_batch_is_unchanged():
    assert selected_releases({}) == RELEASES
    assert "plan.plan_and_solve.ps_plus_prompt" not in RELEASES


def test_explicit_ps_plus_batch():
    releases = ["baseline.no_op", "plan.plan_and_solve.ps_plus_prompt"]
    assert selected_releases({"releases": releases}) == tuple(releases)
    assert releases == ["baseline.no_op", "plan.plan_and_solve.ps_plus_prompt"]


def test_explicit_planning_component_batch_does_not_change_old_defaults():
    releases = ["plan.plan_and_solve.plan_component"]
    assert selected_releases({"releases": releases}) == tuple(releases)
    assert releases[0] not in RELEASES


@pytest.mark.parametrize("releases", [None, [], "baseline.no_op", [False], ["missing"],
                                    ["baseline.no_op", "baseline.no_op"], [{}]])
def test_invalid_release_selection_is_rejected_before_data_or_model_access(releases):
    with pytest.raises(ValueError, match="unique supported"):
        selected_releases({"releases": releases})
