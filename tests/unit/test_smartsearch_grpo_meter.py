"""Explicit non-neural fixtures for shared accounting, never acceptance claims."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.recipes.smartsearch_grpo_meter import (
    MeteredNativeGRPOEngine,
    SmartSearchGRPOCostLedger,
    SmartSearchGRPOCostLimits,
    meter_smartsearch_grpo_helper,
    observe_native_grpo_optimizer,
)
from agentinputlab.recipes.smartsearch_teacher import SmartSearchTeacherCompletion


def ledger(tmp_path, **kwargs):
    limits = dict(policy_requests=10, policy_output_tokens=32768, helper_requests=10,
                  helper_output_tokens=20480, actor_ranks=2, optimizer_steps_per_rank=4)
    limits.update(kwargs)
    return SmartSearchGRPOCostLedger(output_root=tmp_path / "meter", limits=SmartSearchGRPOCostLimits(**limits))


def reserve(meter, **kwargs):
    values = dict(kind="policy", worker="fixture-actor-rank-0", input_token_counts=[3, 5], samples=2, max_new_tokens=8)
    values.update(kwargs)
    return meter.reserve(**values)


def test_two_ranks_share_actual_token_usage_and_synchronized_step_counts(tmp_path):
    meter = ledger(tmp_path)
    for worker in ("fixture-rank-0", "fixture-rank-1"):
        ticket = reserve(meter, worker=worker)
        meter.settle(ticket, output_token_counts=[1, 2, 3, 4])
    for rank in range(2):
        for _ in range(4):
            meter.optimizer_begin(rank)
            meter.optimizer_end(rank)
    result = meter.assert_healthy(expected_optimizer_steps=4)
    assert result["synchronized_optimizer_steps"] == 4  # Not sum=8.
    assert result["costs"]["policy"] == dict(requests=2, responses=2, logical_input_tokens=16,
        output_tokens=20, reserved_output_tokens=0, generated_samples=8)
    assert not result["physical_model_forwards_measured"]
    with pytest.raises(BudgetExceededError, match="optimizer step quota"):
        meter.optimizer_begin(0)


def test_atomic_concurrent_reservations_cannot_multiply_allowance(tmp_path):
    meter = ledger(tmp_path, policy_requests=2, policy_output_tokens=64)
    with ThreadPoolExecutor(2) as pool:
        tickets = list(pool.map(lambda worker: reserve(meter, worker=worker), ["fixture-0", "fixture-1"]))
    assert len(set(tickets)) == 2
    assert meter.snapshot()["costs"]["policy"]["reserved_output_tokens"] == 64
    with pytest.raises(BudgetExceededError, match="aggregate"):
        reserve(meter)
    assert len(meter.snapshot()["pending_generation_tickets"]) == 2


def test_failed_generation_retains_unknown_reservation_and_blocks_retry(tmp_path):
    meter = ledger(tmp_path)
    ticket = reserve(meter)
    meter.fail(ticket, error_type="FixtureTransportFailure")
    assert meter.snapshot()["costs"]["policy"]["reserved_output_tokens"] == 32
    with pytest.raises(ExternalDependencyError, match="retained failure"):
        reserve(meter)
    with pytest.raises(ExternalDependencyError):
        meter.assert_healthy()


def test_other_rank_failure_does_not_erase_inflight_actual_costs(tmp_path):
    meter = ledger(tmp_path)
    first, second = reserve(meter), reserve(meter, worker="fixture-other-rank")
    meter.optimizer_begin(0)
    meter.fail(first, error_type="FixtureFailure")
    meter.settle(second, output_token_counts=[1, 2, 3, 4])
    meter.optimizer_end(0)
    result = meter.snapshot()
    assert result["failed"] and result["costs"]["policy"]["output_tokens"] == 10
    assert result["costs"]["policy"]["reserved_output_tokens"] == 32
    assert result["optimizer_rank_steps"]["0"] == dict(attempts=1, responses=1)
    with pytest.raises(ExternalDependencyError):
        meter.assert_healthy()


@pytest.mark.parametrize("counts", [[1], [1, 2, 3, True], [1, 2, 3, 9], None])
def test_malformed_settlement_is_not_zero_cost(tmp_path, counts):
    meter = ledger(tmp_path)
    ticket = reserve(meter)
    with pytest.raises(ExternalDependencyError, match="counts differ"):
        meter.settle(ticket, output_token_counts=counts)
    assert meter.snapshot()["failed"]
    assert meter.snapshot()["costs"]["policy"]["reserved_output_tokens"] == 32
    assert meter.snapshot()["invalid_usage_reports"][0]["reservation_retained"]


def test_duplicate_settlement_and_unresolved_rank_rejected(tmp_path):
    meter = ledger(tmp_path)
    ticket = reserve(meter)
    with pytest.raises(ExternalDependencyError, match="unresolved paid"):
        meter.assert_healthy()
    meter.settle(ticket, output_token_counts=[1, 1, 1, 1])
    meter.optimizer_begin(0)
    meter.optimizer_end(0)
    with pytest.raises(ExternalDependencyError, match="rank step counts differ"):
        meter.assert_healthy()
    with pytest.raises(ExternalDependencyError, match="already settled"):
        meter.settle(ticket, output_token_counts=[1, 1, 1, 1])


@pytest.mark.parametrize("override", [dict(samples=True), dict(kind=[]), dict(max_new_tokens=8193),
    dict(input_token_counts=[True]), dict(kind="helper", samples=2), dict(worker="")])
def test_bad_reservations_never_consume_neural_work(tmp_path, override):
    meter = ledger(tmp_path)
    with pytest.raises(ValidationError):
        reserve(meter, **override)
    assert meter.snapshot()["costs"]["policy"]["requests"] == 0


def test_engine_observer_preserves_inputs_objects_and_counts_discardable_samples(tmp_path):
    meter, calls = ledger(tmp_path), []
    outputs = [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2]), SimpleNamespace(token_ids=[3])])]
    native = SimpleNamespace(generate=lambda **kw: calls.append(kw) or outputs, model_identity="fixture")
    engine = MeteredNativeGRPOEngine(native, ledger=meter, worker="fixture-actor-0")
    prompts = [{"prompt_token_ids": [1, 2, 3]}]
    params = SimpleNamespace(n=2, max_tokens=8, temperature=.9, stop=["native fixture"])
    result = engine.generate(prompts=prompts, sampling_params=params, use_tqdm=False)
    assert result is outputs and calls[0]["prompts"] is prompts and calls[0]["sampling_params"] is params
    assert params.temperature == .9 and params.stop == ["native fixture"]
    assert engine.model_identity == "fixture"
    assert meter.assert_healthy()["costs"]["policy"]["output_tokens"] == 3
    assert meter.snapshot()["costs"]["policy"]["generated_samples"] == 2


def test_native_engine_exception_is_called_once(tmp_path):
    meter, calls = ledger(tmp_path), []

    def native(**kw):
        calls.append(kw)
        raise RuntimeError("fixture engine failure")

    engine = MeteredNativeGRPOEngine(SimpleNamespace(generate=native), ledger=meter, worker="fixture-rank")
    arguments = dict(prompts=[{"prompt_token_ids": [1]}], sampling_params=SimpleNamespace(n=1, max_tokens=8))
    with pytest.raises(RuntimeError, match="fixture engine"):
        engine.generate(**arguments)
    with pytest.raises(ExternalDependencyError, match="no automatic"):
        engine.generate(**arguments)
    assert len(calls) == 1 and meter.snapshot()["costs"]["policy"]["reserved_output_tokens"] == 8


def test_helper_shared_across_prefix_and_reward_preserves_native_stops(tmp_path):
    meter, calls = ledger(tmp_path, helper_requests=2), []
    result = SmartSearchTeacherCompletion("fixture native text", 3, 4, "a" * 64)
    callback = meter_smartsearch_grpo_helper(lambda *a, **kw: calls.append((a, kw)) or result,
        ledger=meter, worker="fixture-teacher", fingerprint="a" * 64, count_input_tokens=lambda _: 3)
    for stop in ((), ("</answer>",)):
        assert callback("fixture", temperature=.6, max_new_tokens=2048, stop=stop, seed=17) is result
    assert [c[1]["stop"] for c in calls] == [(), ("</answer>",)]
    assert meter.assert_healthy()["costs"]["helper"]["output_tokens"] == 8
    with pytest.raises(BudgetExceededError):
        callback("fixture", temperature=.6, max_new_tokens=2048, stop=(), seed=17)
    assert len(calls) == 2


def test_helper_wrong_identity_retains_measured_cost_and_no_retry(tmp_path):
    meter, calls = ledger(tmp_path), []
    callback = meter_smartsearch_grpo_helper(lambda *a, **kw: calls.append(kw) or
        SmartSearchTeacherCompletion("fixture", 3, 4, "b" * 64), ledger=meter,
        worker="fixture-teacher", fingerprint="a" * 64, count_input_tokens=lambda _: 3)
    with pytest.raises(ExternalDependencyError, match="identity"):
        callback("fixture", temperature=.6, max_new_tokens=2048, stop=(), seed=17)
    with pytest.raises(ExternalDependencyError, match="no automatic"):
        callback("fixture", temperature=.6, max_new_tokens=2048, stop=(), seed=17)
    assert len(calls) == 1 and meter.snapshot()["costs"]["helper"]["output_tokens"] == 4


def test_optimizer_keeps_native_return_and_retains_partial_failure(tmp_path):
    meter, calls = ledger(tmp_path), []
    native = SimpleNamespace(step=lambda *a, **kw: calls.append((a, kw)) or "fixture native return")
    assert observe_native_grpo_optimizer(native, ledger=meter, rank=0) is native
    assert native.step("closure", native_flag=True) == "fixture native return"
    assert calls == [(("closure",), {"native_flag": True})]
    with pytest.raises(ValidationError):
        observe_native_grpo_optimizer(native, ledger=meter, rank=0)

    def broken():
        raise RuntimeError("fixture optimizer failure")

    other = observe_native_grpo_optimizer(SimpleNamespace(step=broken), ledger=meter, rank=1)
    with pytest.raises(RuntimeError):
        other.step()
    with pytest.raises(ExternalDependencyError, match="no automatic"):
        other.step()
    assert meter.snapshot()["optimizer_rank_steps"]["1"] == dict(attempts=1, responses=0)
