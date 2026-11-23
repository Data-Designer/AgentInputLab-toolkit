"""Synthetic binding/budget tests, not new neural experiment results."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab import ExecutionBudget
from agentinputlab import arena_configuration as arena
from agentinputlab import arena_prior_usage as prior
from agentinputlab.api import _state
from agentinputlab.configuration import RunConfig
from agentinputlab.contracts import InputPatch, ValidationError
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.serialization import to_jsonable
from tests.unit.test_arena_configuration import BACKEND, selected
from workers.official_arenas.protocol import ProviderArmResult, merge_usage


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def receipt(tmp_path):
    checkout = tmp_path / "native"
    scorer = checkout / arena._PROVIDERS["tooltalk"][2]
    scorer.parent.mkdir(parents=True)
    scorer.write_text("synthetic scorer")
    run = RunConfig(
        "baseline.no_op", backend=BACKEND, budget=ExecutionBudget(max_context_tokens=262144)
    )
    native = arena.NativeArenaConfig("tooltalk", str(checkout))
    config = native.provider_config(run)
    cache = write(tmp_path / "cache.json", {"synthetic": True})
    run = replace(
        run,
        release_id="rewrite.infocqr.cached_rw_zsl",
        parameters={"split": "test"},
        artifacts={"cache": {**cache, "artifact_id": "synthetic", "dataset_revision": "synthetic"}},
    )
    capture = selected(run, config)
    state = _state(capture["visible_state"])
    value = dict(
        schema_version=1,
        release_id=run.release_id,
        model_id=run.backend["model_id"],
        seed=run.seed,
        original_budget=to_jsonable(run.budget),
        cases=[
            dict(
                case_id=state.task_id,
                input_state_sha256=state.state_hash,
                output_state_sha256=state.state_hash,
                cache_sha256=cache["sha256"],
                source=write(tmp_path / "source.json", {"synthetic_usage": 1}),
                usage=dict(model_calls=1, input_tokens=10, output_tokens=3),
            )
        ],
    )
    return run, native, config, capture, value, tmp_path / "prior.json"


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_id", "baseline.no_op"),
        ("model_id", "another"),
        ("seed", True),
        ("seed", 18),
        ("schema_version", True),
        ("original_budget", {}),
    ],
)
def test_receipt_run_binding_rejects_changes(receipt, field, value):
    run, _, _, _, v, path = receipt
    v[field] = value
    with pytest.raises(ValidationError, match="identity or budget"):
        prior.load(write(path, v), run)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_calls", True),
        ("model_calls", 9),
        ("model_calls", 0),
        ("output_tokens", 4097),
        ("output_tokens", -1),
        ("tool_calls", 9),
        ("retrieval_calls", 9),
        ("input_tokens", 262145),
        ("cost_usd", float("inf")),
        ("unknown", 3),
    ],
)
def test_invalid_or_over_budget_prior_usage_rejected(receipt, field, value):
    run, _, _, _, v, path = receipt
    v["cases"][0]["usage"][field] = value
    with pytest.raises(ValidationError):
        prior.load(write(path, v), run)


def test_source_and_receipt_checksums_are_both_required(receipt):
    run, _, _, _, v, path = receipt
    pin = write(path, v)
    path.write_text("{}")
    with pytest.raises(ValidationError, match="checksum"):
        prior.load(pin, run)
    v["cases"][0]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="checksum"):
        prior.load(write(path, v), run)


def test_relative_receipt_and_relative_source_preserved(receipt, tmp_path):
    run, native, _, _, v, path = receipt
    v["cases"][0]["source"]["path"] = "source.json"
    pin = write(path, v)
    pin["path"] = "prior.json"
    cfg = tmp_path / "arena.json"
    cfg.write_text(json.dumps({**to_jsonable(native), "prior_intervention_usage": pin}))
    native = arena.NativeArenaConfig.from_file(cfg)
    assert native.prior_intervention_usage["path"] == str(path)
    assert len(prior.load(native.prior_intervention_usage, run)) == 1


def test_no_live_profile_or_history_can_use_receipt(receipt):
    run, native, _, _, v, path = receipt
    pin = write(path, v)
    with pytest.raises(ValidationError, match="cached profile"):
        prior.load(pin, replace(run, release_id="baseline.no_op"))
    with pytest.raises(ValidationError, match="live history"):
        replace(native, prior_intervention_usage=pin, history={"kind": "acon"})


def test_duplicate_case_wrong_cache_and_missing_case_rejected(receipt):
    run, _, _, capture, v, path = receipt
    v["cases"].append(dict(v["cases"][0]))
    with pytest.raises(ValidationError, match="unique"):
        prior.load(write(path, v), run)
    v["cases"].pop()
    v["cases"][0]["cache_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="cache checksum"):
        prior.load(write(path, v), run)
    with pytest.raises(ValidationError, match="missing case"):
        prior.validate_case({}, capture["checkpoint_id"], _state(capture["visible_state"]))


def test_wrong_output_or_newly_executed_usage_cannot_be_debited(receipt):
    run, _, _, _, v, path = receipt
    entry = next(iter(prior.load(write(path, v), run).values()))
    record = SimpleNamespace(usage=merge_usage(), output_state_hash="0" * 64)
    with pytest.raises(ValidationError, match="output state"):
        prior.debit(entry, record)
    record.output_state_hash = entry["output_state_sha256"]
    record.usage["model_calls"] = 1
    with pytest.raises(ValidationError, match="newly executed"):
        prior.debit(entry, record)


def selector_receipt(receipt):
    run, native, config, capture, value, path = receipt
    pin = dict(run.artifacts["cache"])
    pin.pop("artifact_id", None)
    pin.pop("dataset_revision", None)
    run = replace(run, release_id="acquire.adaptive_rag.selector_replay",
                  parameters={"split": "test", "artifact_sha256": pin["sha256"],
                              "artifact_id": "synthetic-selector", "dependency_hash": "d" * 64},
                  artifacts={"records": pin})
    value["release_id"] = run.release_id
    entry = value["cases"][0]
    entry["artifact_sha256"] = entry.pop("cache_sha256")
    return run, native, config, capture, value, path


@pytest.mark.parametrize("fault", [None, "missing_file_pin", "parameter_pin", "case_pin", "cache_field", "source"])
def test_selector_prior_receipt_binds_pinned_records(receipt, fault):
    run, _, _, _, value, path = selector_receipt(receipt)
    if fault == "missing_file_pin":
        run = replace(run, artifacts={})
    elif fault == "parameter_pin":
        run = replace(run, parameters={**run.parameters, "artifact_sha256": "f" * 64})
    elif fault == "case_pin":
        value["cases"][0]["artifact_sha256"] = "f" * 64
    elif fault == "cache_field":
        value["cases"][0]["cache_sha256"] = value["cases"][0].pop("artifact_sha256")
    elif fault == "source":
        value["cases"][0]["source"]["sha256"] = "0" * 64
    pin = write(path, value)
    if fault:
        with pytest.raises(ValidationError):
            prior.load(pin, run)
    else:
        entries = prior.load(pin, run)
        assert next(iter(entries.values()))["usage"]["model_calls"] == 1


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("selector", [False, True])
def test_native_episode_debits_history_without_rebilling_cache(
    receipt, tmp_path, monkeypatch, fail, selector
):
    run, native, config, capture, value, path = selector_receipt(receipt) if selector else receipt
    native = replace(native, prior_intervention_usage=write(path, value))

    class Cached:
        method_id = "fixture-cache"
        release_id = run.release_id
        family = SemanticFamily.REFORMULATE

        def propose(self, state, context):
            return InputPatch.no_op(state, method_id=self.method_id, release_id=self.release_id)

    monkeypatch.setattr(RunConfig, "create_intervention", lambda self: Cached())

    class Provider:
        scorer_id = "fixture"
        scorer_revision = source_revision = "a" * 64

        def run_arm(self, config, **kwargs):
            payload = kwargs["arm_payload"]["intervention_execution"]
            assert payload["usage"]["model_calls"] == 1 and payload["usage"]["output_tokens"] == 3
            assert payload["materialized_state"]["budget"] == to_jsonable(run.budget)
            if fail:
                raise RuntimeError("retained native failure")
            return ProviderArmResult(
                metrics={"score": 0},
                trace={"synthetic": True},
                usage={"model_calls": 2, "output_tokens": 7},
                runtime_instance_id="b" * 64,
            )

    row = arena._run_case(run, Provider(), config, capture, tmp_path, native=native)
    assert row["intervention_usage"]["model_calls"] == 0 and row["model_responses"] == []
    assert row["prior_intervention"]["usage"]["model_calls"] == 1
    assert row["prior_intervention"]["historical_not_new_model_calls"]
    if fail:
        assert (
            row["status"] == "failed"
            and row["error"] == "retained native failure"
            and "metrics" not in row
        )
    else:
        assert row["usage"]["model_calls"] == 3 and row["usage"]["output_tokens"] == 10
        assert (
            row["current_run_usage"]["model_calls"] == 2
            and row["current_run_usage"]["output_tokens"] == 7
        )
        assert row["metrics"] == {"score": 0}
