"""Non-neural fixtures for a native dictionary/once-only adapter."""
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes.smartsearch_grpo_contracts import NativeGRPORewardManagerContract


def test_missing_extra_info_does_not_repeat_native_batch_or_modify_tensor():
    tensor = object()  # Explicit fixture, not a real tensor or neural acceptance.
    calls = []
    original = {"reward_tensor": tensor}

    def native(data, **kwargs):
        calls.append((data, kwargs))
        return original

    adapter = NativeGRPORewardManagerContract(native)
    batch = object()
    result = adapter(batch, return_dict=True, curr_save_path="operator-bound.jsonl")
    assert result["reward_tensor"] is tensor and result["reward_extra_info"] == {}
    assert "reward_extra_info" not in original
    assert calls == [(batch, {"return_dict": True, "curr_save_path": "operator-bound.jsonl"})]
    assert adapter.native_calls == 1


def test_existing_metadata_and_non_dictionary_results_preserved():
    tensor = object()
    extras = {"fixture_metric": [0., 1.]}
    original = {"reward_tensor": tensor, "reward_extra_info": extras, "native_other_field": "kept"}
    adapter = NativeGRPORewardManagerContract(lambda *a, **kw: original if kw["return_dict"] else tensor)
    result = adapter(object(), return_dict=True)
    assert result == original and result["reward_extra_info"] is extras
    assert adapter(object()) is tensor


def test_native_cached_tensor_branch_never_recomputes_reward():
    cached = object()
    batch = SimpleNamespace(batch={"rm_scores": cached})
    adapter = NativeGRPORewardManagerContract(lambda *a, **kw: cached)
    result = adapter(batch, return_dict=True)
    assert result["reward_tensor"] is cached and result["reward_extra_info"] == {}
    assert adapter.native_calls == 1


def test_failure_after_partial_scoring_latches_entire_batch():
    costs = []

    def native(*args, **kwargs):
        costs.extend(["fixture item consumed", "fixture second item failed"])
        raise ConnectionError("fixture native helper transport failure")

    adapter = NativeGRPORewardManagerContract(native)
    with pytest.raises(ConnectionError):
        adapter(object(), return_dict=True)
    # Simulate the native PPO driver's catch/fallback branch.
    with pytest.raises(ExternalDependencyError, match="no automatic scoring retry"):
        adapter(object())
    assert len(costs) == 2 and adapter.native_calls == 1


@pytest.mark.parametrize("result", [None, {"reward_tensor": None}, {"wrong": object()},
                                    {"reward_tensor": object(), "reward_extra_info": []}])
def test_invalid_native_output_fails_closed_without_fabrication(result):
    adapter = NativeGRPORewardManagerContract(lambda *a, **kw: result)
    with pytest.raises(ExternalDependencyError):
        adapter(object(), return_dict=True)
    with pytest.raises(ExternalDependencyError, match="no automatic"):
        adapter(object(), return_dict=True)
    assert adapter.native_calls == 1


def test_bad_operator_input_rejected_before_call():
    with pytest.raises(ValidationError):
        NativeGRPORewardManagerContract(None)
    adapter = NativeGRPORewardManagerContract(lambda *a, **kw: None)
    with pytest.raises(ValidationError):
        adapter(object(), return_dict="true")
    assert adapter.native_calls == 0
