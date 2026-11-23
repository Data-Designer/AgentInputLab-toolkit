"""File-bound native history wiring; fixtures are not real model acceptance."""
import json
from dataclasses import replace

import pytest

from agentinputlab import InterventionContext, RunConfig
from agentinputlab import arena_configuration as arena
from agentinputlab.contracts import ValidationError
from workers.official_arenas.protocol import ProviderArmResult
from tests.unit.test_arena_configuration import BACKEND, selected, setup  # noqa: F401
from tests.unit.test_run_subprocess_runtime_configuration import acon_artifacts


def history_run(run):
    return replace(run, release_id="curate.acon.history_runtime",
                   parameters={"artifacts": acon_artifacts()})


def test_history_file_roundtrip_and_explicit_profile_guard(setup, tmp_path):
    run, native = setup
    value = {"arena": native.arena, "checkout": native.checkout,
             "history": {"kind": "acon", "prefix_policy": "strict_system_user"}}
    path = tmp_path / "arena.json"
    path.write_text(json.dumps(value))
    loaded = arena.NativeArenaConfig.from_file(path)
    assert loaded.history == value["history"]
    with pytest.raises(ValidationError, match="history_runtime"):
        loaded.provider_config(run)
    configured = history_run(run)
    assert loaded.provider_config(configured) == native.provider_config(configured)


@pytest.mark.parametrize("history", [{}, {"kind": "arbitrary"},
    {"kind": "acon", "prefix_policy": "reorder"},
    {"kind": "acon", "factory": "arbitrary.module"},
    {"kind": "acon", "prefix_policy": []},
    {"kind": "acon", "prefix_policy": {}}, True])
def test_history_invalid_options_fail_without_execution(setup, history):
    _, native = setup
    with pytest.raises(ValidationError):
        replace(native, history=history)


@pytest.mark.parametrize("fail", [False, True])
def test_live_history_is_episode_local_without_initial_double_compression(setup, tmp_path, monkeypatch, fail):
    import agentinputlab.integrations.live_history as live_module
    run, native = setup
    run = history_run(run)
    native = replace(native, history={"kind": "acon"})
    contexts, hooks = [], []

    class Hook:
        def __init__(self, method, context, **options):
            assert context.budget_ledger is None and context.monitor is None
            assert options == {"per_call_output_tokens": 512, "prefix_policy": "strict_system_user"}
            assert method.release_id == run.release_id
            self.records = []
            contexts.append(context)
            hooks.append(self)

    class Provider:
        scorer_id = "fixture-only"
        scorer_revision = source_revision = "a" * 64

        def __init__(self, hook=None):
            self.hook = hook

        def run_arm(self, config, **options):
            payload = options["arm_payload"]["intervention_execution"]
            assert payload["usage"]["model_calls"] == 0
            assert payload["usage"]["output_tokens"] == 0
            assert "gold" not in payload["materialized_state"]
            assert options["replay_record"]["protected"]["gold"] == "NEVER_SEND_GOLD"
            self.hook.records.append({"status": "applied", "usage": {"model_calls": 1, "output_tokens": 7}})
            if fail:
                raise RuntimeError("fixture native failure, no retry")
            # Native provider owns total including the dynamic optimizer cost.
            return ProviderArmResult(metrics={"official_success": 0}, trace={"history_usage": {"model_calls": 1}},
                                     usage={"model_calls": 3, "output_tokens": 18}, runtime_instance_id="b" * 64)

    monkeypatch.setattr(live_module, "ACONLiveHistory", Hook)
    monkeypatch.setattr(arena, "_provider", lambda native, *, history_transform=None: Provider(history_transform))
    monkeypatch.setattr(RunConfig, "_configured_context", lambda self, context: InterventionContext(backend=context.backend))
    from agentinputlab.runner import InterventionRunner
    monkeypatch.setattr(InterventionRunner, "run", lambda *a: pytest.fail("no turn-zero compression before live hook"))
    config = native.provider_config(run)
    for index in range(2):
        work = tmp_path / str(index)
        work.mkdir()
        row = arena._run_case(run, Provider(), config, selected(run, config), work, native=native)
        assert row["status"] == ("failed" if fail else "scored")
        assert row["intervention"] is None
        assert row["intervention_usage"]["model_calls"] == 0
        assert row["live_history_records"] == hooks[index].records
        assert row["execution_mode"] == "native_live_history"
        if not fail:
            assert row["usage"]["model_calls"] == 3
            assert row["usage"]["output_tokens"] == 18
    assert hooks[0] is not hooks[1] and len(contexts) == 2
