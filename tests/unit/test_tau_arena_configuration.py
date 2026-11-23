"""Public Tau wiring tests; fixture outputs are not benchmark evidence."""
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from agentinputlab import ExecutionBudget, RunConfig
from agentinputlab.arena_configuration import NativeArenaConfig
from agentinputlab.contracts import ValidationError

BACKEND = {"model_id": "fixture", "base_url": "http://127.0.0.1:1/v1",
           "api_key_env": "FIXTURE_KEY", "deployment_identity": {"revision": "a" * 64}}


@pytest.fixture
def configured(tmp_path):
    root = tmp_path / "tau"
    scorer = root / "src/tau2/evaluator/evaluator.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("fixture, not native scorer")
    run = RunConfig("baseline.no_op", backend=BACKEND, seed=17,
                    budget=ExecutionBudget(max_context_tokens=262144))
    return run, root


@pytest.mark.parametrize("kind", ["tau_banking", "tau_retail"])
def test_tau_public_configuration_binds_model_and_budget(configured, kind):
    run, root = configured
    native = NativeArenaConfig(kind, str(root), environment_backend=BACKEND)
    value = native.provider_config(run)
    assert value["llm_agent"] == value["llm_user"] == "openai/fixture"
    assert value["llm_args_agent"]["max_tokens"] == 512
    assert value["llm_args_user"]["max_tokens"] == 1024
    assert value["budget_override"]["max_output_tokens"] == 4096
    assert value["llm_args_agent"]["num_retries"] == 0
    assert value["llm_args_agent"]["api_key"] == "agentinputlab-native-role-routing"
    assert "FIXTURE_KEY" not in str(value)
    if kind == "tau_retail":
        assert value["llm_args_judge"]["response_format"] == {"type": "json_object"}
        assert value["capture_seed"] == 17
    else:
        assert value["capture_max_attempts"] == 1


@pytest.mark.parametrize("change", [
    {"model_id": "other"}, {"deployment_identity": {"revision": "b" * 64}},
    {"base_url": "https://example.org/v1"},
])
def test_tau_rejects_implicit_environment_model_changes(configured, change):
    run, root = configured
    with pytest.raises(ValidationError):
        NativeArenaConfig("tau_retail", str(root), environment_backend={**BACKEND, **change}).provider_config(run)


def test_tau_retail_keeps_native_fixed_seed(configured):
    run, root = configured
    native = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND)
    with pytest.raises(ValidationError, match="seed"):
        native.provider_config(replace(run, seed=18))


@pytest.mark.parametrize("kind", ["tau_banking", "tau_retail"])
@pytest.mark.parametrize("extensions", [{}, {"chat_template_kwargs": {"enable_thinking": True}}])
def test_explicit_native_agent_extensions_do_not_change_helper_or_environment(configured, kind, extensions):
    run, root = configured
    helper = {**BACKEND, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    environment = {**BACKEND, "extra_body": {"top_p": 0.9}}
    run = replace(run, backend=helper)
    native = NativeArenaConfig(kind, str(root), environment_backend=environment,
                              parameters={"agent_extra_body": extensions})
    result = native.provider_config(run)
    assert result["llm_args_agent"].get("extra_body", {}) == extensions
    assert ("extra_body" in result["llm_args_agent"]) == bool(extensions)
    assert result["llm_args_user"]["extra_body"] == {"top_p": 0.9}
    assert run.backend["extra_body"] == helper["extra_body"]
    assert result["llm_args_agent"]["max_tokens"] == 512
    assert result["llm_args_agent"]["seed"] == 17
    assert result["budget_override"]["max_output_tokens"] == 4096
    if kind == "tau_retail":
        assert result["llm_args_judge"]["extra_body"] == {"top_p": 0.9}
    result["llm_args_agent"].get("extra_body", {}).clear()
    assert native.parameters["agent_extra_body"] == extensions


@pytest.mark.parametrize("parameters", [{}, {"agent_extra_body": None}])
def test_native_agent_extensions_default_to_existing_backend(configured, parameters):
    run, root = configured
    run = replace(run, backend={**BACKEND, "extra_body": {"top_p": 0.9}})
    result = NativeArenaConfig("tau_banking", str(root), environment_backend=BACKEND,
                               parameters=parameters).provider_config(run)
    assert result["llm_args_agent"]["extra_body"] == {"top_p": 0.9}


@pytest.mark.parametrize("extensions", [[], True, "{}", {"max_tokens": 1024},
    {"seed": 1}, {"stop": ["end"]}, {"messages": []}, {"model": "other"}])
def test_native_agent_extensions_cannot_change_protocol_fields(configured, extensions):
    from workers.official_arenas.protocol import OfficialArenaError
    run, root = configured
    native = NativeArenaConfig("tau_banking", str(root), environment_backend=BACKEND,
                               parameters={"agent_extra_body": extensions})
    with pytest.raises(OfficialArenaError, match="agent_extra_body"):
        native.provider_config(run)


@pytest.fixture
def capture_fixture(configured, monkeypatch):
    from agentinputlab import tau_arena_configuration as tau
    run, root = configured
    native = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND)
    config = native.provider_config(run)
    requests = []
    helpers = ModuleType("tau2.runner.helpers")
    def tasks(domain, **options):
        requests.append((domain, options))
        return [SimpleNamespace(id="5"), SimpleNamespace(id="9")]
    helpers.get_tasks = tasks
    monkeypatch.setitem(sys.modules, "tau2", ModuleType("tau2"))
    monkeypatch.setitem(sys.modules, "tau2.runner", ModuleType("tau2.runner"))
    monkeypatch.setitem(sys.modules, "tau2.runner.helpers", helpers)
    calls = []
    @contextmanager
    def roles(*args, **kwargs):
        yield []
    monkeypatch.setattr(tau, "role_calls", roles)
    class Provider:
        def capture(self, cfg, *, output_root):
            calls.append(cfg)
            key = cfg["capture_task_ids"][0]
            return [{"checkpoint_id": key, "protected": {"capture_seed": 17}}]
    return tau, run, native, config, Provider(), calls, requests


def test_list_tau_cases_does_not_capture_or_call_models(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, requests = capture_fixture
    tau.capture(native, run, provider, cfg, {"inventory_only": True}, tmp_path)
    receipt = json.loads((tmp_path / "terminal.json").read_text())
    assert receipt["case_ids"] == ["5", "9"]
    assert receipt["model_calls"] == 0
    assert not calls and len(requests) == 1
    assert not (tmp_path / "inventory.json").exists()


def test_capture_selected_case_only(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, _ = capture_fixture
    tau.capture(native, run, provider, cfg, {"case_ids": ["9"]}, tmp_path)
    assert len(calls) == 1
    assert calls[0]["capture_task_ids"] == ["9"] and calls[0]["limit"] == 1
    assert calls[0]["budget_override"] == cfg["budget_override"]
    assert json.loads((tmp_path / "inventory.json").read_text())[0]["checkpoint_id"] == "9"


def test_capture_positive_limit_does_not_capture_remaining_inventory(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, _ = capture_fixture
    tau.capture(native, run, provider, cfg, {"limit": 1}, tmp_path)
    assert len(calls) == 1 and calls[0]["capture_task_ids"] == ["5"]


def test_unknown_case_is_rejected_before_model_calls(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, _ = capture_fixture
    with pytest.raises(ValidationError, match="absent"):
        tau.capture(native, run, provider, cfg, {"case_ids": ["99"]}, tmp_path)
    assert not calls


def test_capture_failure_retains_attempt_and_continues_other_cases(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, _ = capture_fixture
    original = provider.capture
    def capture(config, *, output_root):
        if config["capture_task_ids"] == ["5"]:
            calls.append(config)
            raise RuntimeError("fixture user generation failed")
        return original(config, output_root=output_root)
    provider.capture = capture
    tau.capture(native, run, provider, cfg, {"case_ids": ["5", "9"]}, tmp_path)
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert len(calls) == 2
    assert terminal["captured_case_ids"] == ["9"]
    assert terminal["capture_failures"]["5"]["automatic_retry"] is False
    assert (tmp_path / "capture-0000/started.json").is_file()
    assert (tmp_path / "capture-0000/terminal.json").is_file()


def test_evaluation_counts_capture_failure_in_selected_denominator(configured, tmp_path, monkeypatch):
    from agentinputlab import arena_configuration as arena
    run, root = configured
    native = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND)
    jobs = []
    def launch(job, path):
        jobs.append(job)
        if job["mode"] == "capture":
            return {"status": "captured", "case_ids": ["5", "9"], "inventory_sha256": "a" * 64,
                    "capture_failures": {"5": {"case_id": "5", "status": "failed", "stage": "initial_user_capture"}}}
        return {"status": "scored", "metrics": {"reward": 0}}
    monkeypatch.setattr(arena, "_launch", launch)
    result = arena.evaluate_arena(run, native, output_dir=tmp_path / "run", case_ids=["5", "9"])
    assert result["selected_count"] == 2 and result["failed_count"] == 1 and result["scored_count"] == 1
    assert len(jobs) == 2 and jobs[1]["case_id"] == "9"
    assert jobs[0]["case_ids"] == ["5", "9"]


@pytest.mark.parametrize("ids", [[], ["5", "5"], ["../5"], [5], "5", ["5", "9"]])
def test_retail_native_capture_selection_validation(configured, ids):
    from workers.official_arenas.protocol import OfficialArenaError
    from workers.official_arenas.providers.tau_retail import TauRetailProvider
    run, root = configured
    config = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND).provider_config(run)
    with pytest.raises(OfficialArenaError, match="capture_task_ids"):
        TauRetailProvider()._validate_config({**config, "limit": 1, "capture_task_ids": ids})


def test_retail_accepts_single_official_id_without_changing_prior_config(configured):
    from workers.official_arenas.providers.tau_retail import TauRetailProvider
    run, root = configured
    config = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND).provider_config(run)
    parsed = TauRetailProvider()._validate_config({**config, "limit": 1, "capture_task_ids": ["9"]})
    assert parsed["capture_task_ids"] == ["9"]
    assert "capture_task_ids" not in config


def test_capture_only_public_cli_does_not_start_evaluated_agent(configured, tmp_path, monkeypatch, capsys):
    from agentinputlab import arena_configuration as arena
    from agentinputlab.cli.main import main
    from agentinputlab.contracts.serialization import to_jsonable
    run, root = configured
    native = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND)
    r, n = tmp_path / "run.json", tmp_path / "native.json"
    r.write_text(json.dumps(run.to_dict()))
    n.write_text(json.dumps(to_jsonable(native)))
    jobs = []
    def launch(job, path):
        jobs.append(job)
        assert job["mode"] == "capture"
        return {"status": "captured", "case_ids": ["5", "9"], "capture_failures": {}}
    monkeypatch.setattr(arena, "_launch", launch)
    assert main(["config", "evaluate-arena", str(r), "--arena-config", str(n),
                 "--case-id", "9", "--capture-only", "--output-dir", str(tmp_path / "result")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["captured_count"] == 1 and summary["scored_count"] == 0 and len(jobs) == 1


def test_capture_and_list_flags_are_not_ambiguous(configured, tmp_path):
    from agentinputlab import arena_configuration as arena
    run, root = configured
    native = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND)
    with pytest.raises(ValidationError, match="cannot be combined"):
        arena.evaluate_arena(run, native, output_dir=tmp_path / "result", inventory_only=True, capture_only=True)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("kind,capture,expected_seed", [
    ("tau_banking", True, 0), ("tau_banking", False, 17),
    ("tau_retail", True, 17), ("tau_retail", False, 17),
])
def test_native_capture_seed_is_separate_from_agent_seed(configured, tmp_path, monkeypatch,
                                                        kind, capture, expected_seed):
    httpx = pytest.importorskip("httpx")
    from agentinputlab import tau_arena_configuration as tau
    from agentinputlab.recipes import skillopt_tau
    run, root = configured
    native = NativeArenaConfig(kind, str(root), environment_backend=BACKEND)
    cfg = native.provider_config(run)
    monkeypatch.setenv("FIXTURE_KEY", "synthetic")
    monkeypatch.setattr(skillopt_tau, "_native_role", lambda: "user" if capture else "agent")
    requests = []
    def send(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"completion_tokens": 1}})
    with (httpx.Client(transport=httpx.MockTransport(send)) as client,
          tau.role_calls(native, run, cfg, tmp_path, initial_capture=capture) as events):
        response = client.post(BACKEND["base_url"] + "/chat/completions",
                               json={"model": "fixture", "seed": expected_seed, "max_tokens": 512})
    assert response.status_code == 200 and len(events) == 1 and len(requests) == 1
    assert requests[0]["seed"] == expected_seed
    assert run.seed == cfg["llm_args_agent"]["seed"] == 17


def test_capture_exports_pinned_reusable_configuration(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, _, _ = capture_fixture
    tau.capture(native, run, provider, cfg, {"case_ids": ["5"]}, tmp_path)
    bundle = json.loads((tmp_path / "capture-bundle.json").read_text())
    configurations = json.loads((tmp_path / bundle["native_configurations"]["path"]).read_text())
    assert configurations["5"] == {**cfg, "limit": 1, "capture_task_ids": ["5"]}
    assert bundle["cases"]["5"]["status"] == "captured"
    assert bundle["inventory"]["sha256"]


def test_native_capture_reference_resolves_from_configuration_file(configured, tmp_path):
    from agentinputlab.contracts.serialization import to_jsonable
    _, root = configured
    native = NativeArenaConfig("tau_retail", str(root), environment_backend=BACKEND,
                              capture_bundle={"path": "capture/capture-bundle.json", "sha256": "a" * 64})
    p = tmp_path / "native.json"
    p.write_text(json.dumps(to_jsonable(native)))
    assert NativeArenaConfig.from_file(p).capture_bundle["path"] == str(tmp_path / "capture/capture-bundle.json")


def saved_capture(capture_fixture, tmp_path, ids=("5",)):
    from agentinputlab.arena_configuration import _sha
    tau, run, native, cfg, provider, calls, _ = capture_fixture
    original = tmp_path / "original"
    original.mkdir()
    tau.capture(native, run, provider, cfg, {"case_ids": list(ids)}, original)
    pin = {"path": str(original / "capture-bundle.json"), "sha256": _sha(original / "capture-bundle.json")}
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    return tau, run, replace(native, capture_bundle=pin), cfg, provider, calls, original, reuse


def test_saved_capture_reuse_has_no_model_calls(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, original, reuse = saved_capture(capture_fixture, tmp_path)
    tau.capture(native, run, provider, cfg, {"case_ids": ["5"]}, reuse)
    assert len(calls) == 1
    terminal = json.loads((reuse / "terminal.json").read_text())
    assert terminal["capture_reused"] and terminal["new_capture_model_calls"] == 0
    assert terminal["inventory_path"] == str(original / "inventory.json")
    assert not (reuse / "inventory.json").exists()  # reference, not a duplicate evidence package


@pytest.mark.parametrize("field,value", [("max_steps", 99), ("task_split_name", "train"),
                                         ("llm_agent", "openai/other")])
def test_saved_capture_does_not_cross_protocols(capture_fixture, tmp_path, field, value):
    tau, run, native, cfg, provider, calls, _, reuse = saved_capture(capture_fixture, tmp_path)
    with pytest.raises(ValidationError, match="configuration differs"):
        tau.capture(native, run, provider, {**cfg, field: value}, {"case_ids": ["5"]}, reuse)
    assert len(calls) == 1


def test_saved_capture_can_relocate_transport_only(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, _, reuse = saved_capture(capture_fixture, tmp_path)
    cfg = json.loads(json.dumps(cfg))
    for role in ("agent", "user", "judge"):
        cfg["llm_args_" + role]["api_base"] = "http://127.0.0.1:999/v1"
    tau.capture(native, run, provider, cfg, {"case_ids": ["5"]}, reuse)
    assert len(calls) == 1


def test_saved_capture_missing_case_never_generates(capture_fixture, tmp_path):
    tau, run, native, cfg, provider, calls, _, reuse = saved_capture(capture_fixture, tmp_path)
    with pytest.raises(ValidationError, match="no automatic capture"):
        tau.capture(native, run, provider, cfg, {"case_ids": ["9"]}, reuse)
    assert len(calls) == 1


def test_saved_capture_failure_is_reused_not_retried(capture_fixture, tmp_path):
    provider = capture_fixture[4]
    def fail(*args, **kwargs):
        raise RuntimeError("saved user failure")
    provider.capture = fail
    tau, run, native, cfg, provider, calls, _, reuse = saved_capture(capture_fixture, tmp_path)
    tau.capture(native, run, provider, cfg, {"case_ids": ["5"]}, reuse)
    terminal = json.loads((reuse / "terminal.json").read_text())
    assert terminal["capture_failures"]["5"]["reused_capture_failure"]
    assert terminal["captured_case_ids"] == [] and terminal["new_capture_model_calls"] == 0


@pytest.mark.parametrize("member", ["inventory.json", "native-configurations.json", "capture-0000/terminal.json"])
def test_saved_capture_checks_every_member_hash(capture_fixture, tmp_path, member):
    from agentinputlab.contracts.errors import ArtifactError
    tau, run, native, cfg, provider, calls, original, reuse = saved_capture(capture_fixture, tmp_path)
    path = original / member
    path.write_text(path.read_text() + " ")
    with pytest.raises((ValidationError, ArtifactError), match="checksum"):
        tau.capture(native, run, provider, cfg, {"case_ids": ["5"]}, reuse)
    assert len(calls) == 1


def test_capture_bundle_populates_fit_inputs_without_private_configuration(capture_fixture, tmp_path):
    from agentinputlab.fit_preparation import _configuration
    from tests.unit.test_fit_preparation import config, write
    _, _, native, cfg, _, calls, original, _ = saved_capture(capture_fixture, tmp_path, ("5", "9"))
    value = config(tmp_path, "tau")
    for key in ("captures", "native_configurations", "target_model", "domain"):
        del value[key]
    value["capture_bundle"] = native.capture_bundle
    value["groups"] = {"train": ["5"], "validation": ["9"], "heldout": ["99"]}
    parsed = _configuration(write(tmp_path, value))
    assert parsed["captures"]["path"] == str(original / "inventory.json")
    assert parsed["target_model"]["model_id"] == "fixture"
    assert parsed["native_configurations"]["5"] == {**cfg, "limit": 1, "capture_task_ids": ["5"]}
    assert parsed["domain"] == "retail" and len(calls) == 2
    # Worker normalization revalidates the same pins without altering groups.
    assert _configuration(write(tmp_path, parsed)) == parsed


def test_capture_bundle_does_not_fit_missing_cases_or_override_model(capture_fixture, tmp_path):
    from agentinputlab.fit_preparation import _configuration
    from tests.unit.test_fit_preparation import config, write
    _, _, native, _, _, calls, _, _ = saved_capture(capture_fixture, tmp_path, ("5", "9"))
    value = config(tmp_path, "tau")
    for key in ("captures", "native_configurations", "target_model", "domain"):
        del value[key]
    value.update(capture_bundle=native.capture_bundle,
                 groups={"train": ["5"], "validation": ["99"], "heldout": []})
    with pytest.raises(ValidationError, match="successful captures"):
        _configuration(write(tmp_path, value))
    value["groups"]["validation"] = ["9"]
    value["target_model"] = {"model_id": "other"}
    with pytest.raises(ValidationError, match="conflicts"):
        _configuration(write(tmp_path, value))
    assert len(calls) == 2
