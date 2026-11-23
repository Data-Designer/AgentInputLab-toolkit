"""Synthetic entry/wiring tests, not real-model acceptance."""

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.contracts import ValidationError
from tests.unit.test_benchmark_configuration import Adapter, Backend
from tests.unit.test_run_subprocess_runtime_configuration import acon_artifacts


def config():
    return RunConfig(
        "curate.acon.history_runtime",
        parameters={"artifacts": acon_artifacts()},
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096),
    )


def run(cfg, backend, **options):
    return cfg.evaluate(
        "2wiki.multihop",
        split="dev",
        example_ids=["a", "b"],
        adapter=Adapter(),
        context=InterventionContext(backend=backend),
        predictor_parameters={
            "schema_version": 1,
            "qa": {"output_format": "answer_only"},
            "history": {"kind": "acon"},
        },
        **options,
    )


@pytest.mark.parametrize("failure", [False, True])
def test_rendered_boundary_case_isolation_and_original_request(monkeypatch, failure):
    import agentinputlab.integrations.live_history as module

    hooks = []

    class Hook:
        def __init__(self, method, context, **options):
            assert context.budget_ledger is None and context.monitor is None
            assert options == {"per_call_output_tokens": 512}
            self.records = []
            hooks.append(self)

        def transform(self, messages, *, state, remaining_budget):
            assert [m["role"] for m in messages] == ["system", "user"]
            assert not state["messages"] and "NEVER_SEND_GOLD" not in str(messages)
            assert remaining_budget == {"max_model_calls": 8, "max_output_tokens": 4096}
            if failure:
                self.records.append(
                    {
                        "status": "failed_terminal",
                        "monitor_snapshots": [{"usage": {"model_calls": 1, "output_tokens": 9}}],
                    }
                )
                raise RuntimeError("fixture optimizer failure")
            self.records.append(
                {"status": "native_threshold_skip", "usage": {"model_calls": 0, "output_tokens": 0}}
            )
            return {"messages": messages, "usage": {"model_calls": 0, "output_tokens": 0}}

    monkeypatch.setattr(module, "ACONLiveHistory", Hook)
    backend = Backend([("answer", 3), ("wrong", 2)])
    result = run(config(), backend)
    assert len(hooks) == 2 and hooks[0] is not hooks[1]
    assert result["failed_count"] == (2 if failure else 0)
    assert len(backend.requests) == (0 if failure else 2)
    assert all(r["intervention"] is None for r in result["records"])
    assert all(
        r["live_history_records"] == h.records
        for r, h in zip(result["records"], hooks, strict=True)
    )
    assert [r["usage"]["model_calls"] for r in result["records"]] == [1, 1]
    if failure:
        assert [r["usage"]["output_tokens"] for r in result["records"]] == [9, 9]
    else:
        original = Backend([("answer", 3), ("wrong", 2)])
        RunConfig("baseline.no_op", budget=config().budget).evaluate(
            "2wiki.multihop",
            split="dev",
            example_ids=["a", "b"],
            adapter=Adapter(),
            context=InterventionContext(backend=original),
            predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}},
        )
        assert backend.requests == original.requests
        assert result["metrics_over_scored_cases"] == {"answer_em": 0.5}


def test_history_requires_explicit_matching_profile_without_calls():
    backend = Backend([])
    with pytest.raises(ValidationError, match="history requires"):
        run(RunConfig("baseline.no_op"), backend)
    assert not backend.requests


@pytest.mark.parametrize(
    "value", [{}, True, {"kind": "acon", "invent_system": True}, {"kind": "other"}]
)
def test_invalid_history_options_reject_before_execution(value):
    backend = Backend([])
    with pytest.raises(ValidationError, match="history requires"):
        config().evaluate(
            "2wiki.multihop",
            split="dev",
            limit=1,
            adapter=Adapter(),
            context=InterventionContext(backend=backend),
            predictor_parameters={"history": value},
        )
    assert not backend.requests


@pytest.mark.parametrize("tokens,failed", [(7, False), (4097, True)])
def test_optimizer_actual_usage_is_charged_before_reader(tokens, failed):
    from agentinputlab.backends.llm import GenerationRequest
    from agentinputlab.contracts import BudgetExceededError, FrozenAgentState
    from agentinputlab.observability.budget import BudgetLedger
    from agentinputlab.static_history_configuration import QAHistoryBackend

    class Hook:
        records = ()

        def transform(self, messages, **kwargs):
            return {"messages": messages, "usage": {"model_calls": 1, "output_tokens": tokens}}

    state = FrozenAgentState(query="q", budget=config().budget)
    ledger = BudgetLedger(state.budget)
    backend = Backend([("answer", 3)])
    wrapper = QAHistoryBackend(backend, Hook(), state=state, ledger=ledger)
    request = GenerationRequest(prompt="q", system="system", max_tokens=512)
    if failed:
        with pytest.raises(BudgetExceededError):
            wrapper.generate(request)
        assert not backend.requests
    else:
        wrapper.generate(request)
        assert backend.requests == [request]
    assert ledger.usage.model_calls == 1
    assert ledger.usage.output_tokens == tokens
