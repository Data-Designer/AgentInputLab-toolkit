"""Synthetic native-provider wiring tests, not real arena/model results."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab import ExecutionBudget, RunConfig
from agentinputlab import arena_configuration as arena
from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.cli.main import main
from agentinputlab.contracts import InputPatch, ValidationError
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from tests.unit.test_benchmark_configuration import Backend
from workers.official_arenas.protocol import ProviderArmResult

BACKEND = {"model_id": "fixture", "base_url": "http://127.0.0.1:1/v1",
           "api_key_env": "FIXTURE_KEY", "deployment_identity": {"revision": "fixture-revision"}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkout = tmp_path / "native"
    for _, _, scorer in arena._PROVIDERS.values():
        path = checkout / scorer
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture scorer, not native source")
    run = RunConfig("baseline.no_op", backend=BACKEND,
                    budget=ExecutionBudget(max_context_tokens=262144))
    native = arena.NativeArenaConfig("tooltalk", str(checkout))
    return run, native


@pytest.mark.parametrize("kind", ["tooltalk", "statebench", "toolsandbox"])
def test_provider_binding_uses_run_model_budget_and_separate_environment(setup, kind):
    run, native = setup
    native = replace(native, arena=kind, environment_backend={} if kind == "tooltalk" else BACKEND)
    value = native.provider_config(run)
    assert value["agent_model"] == run.backend["model_id"]
    assert value["budget_override"]["max_output_tokens"] == run.budget.max_output_tokens
    assert value["budget_override"]["max_output_tokens_per_call"] == 512
    assert len(value["locked_dependency_hashes"]["official_scorer_source"]) == 64
    if kind != "tooltalk":
        assert value["locked_dependency_hashes"]["environment_model"] == content_hash(BACKEND["deployment_identity"])
    if kind == "statebench":
        assert value["request_attempts"] == 1


@pytest.mark.parametrize("kwargs", [
    {"arena": "arbitrary.module"}, {"parameters": {"agent_model": "different"}},
    {"environment_backend": BACKEND}, {"max_output_tokens_per_call": True},
    {"arena": "statebench"},
])
def test_native_configuration_rejects_unbound_or_unknown_settings(setup, kwargs):
    _, native = setup
    with pytest.raises(ValidationError):
        replace(native, **kwargs)


@pytest.mark.parametrize("budget", [ExecutionBudget(), ExecutionBudget(max_context_tokens=100, max_cost_usd=1),
                                    ExecutionBudget(max_context_tokens=100, max_output_tokens=128)])
def test_no_implicit_budget_changes(setup, budget):
    run, native = setup
    with pytest.raises((ValidationError, RuntimeError)):
        native.provider_config(replace(run, budget=budget))


def test_visible_context_retriever_rejected_before_native_execution(setup):
    run, native = setup
    with pytest.raises(ValidationError, match="static evaluate only"):
        native.provider_config(replace(run, retriever={"kind": "visible_context_bm25"}))


@pytest.mark.parametrize("kind", ["tooltalk", "statebench", "toolsandbox"])
@pytest.mark.parametrize("policy", ["provider", "final_content"])
def test_native_entry_accepts_intervention_stop_policy_without_changing_native_request(setup, kind, policy):
    run, native = setup
    native = replace(native, arena=kind, environment_backend={} if kind == "tooltalk" else BACKEND)
    configured = replace(run, backend={**BACKEND, "stop_policy": policy})
    assert native.provider_config(configured) == native.provider_config(run)
    assert configured.to_dict()['backend']['stop_policy'] == policy


@pytest.mark.parametrize('policy', ['provider', 'final_content'])
def test_native_intervention_stop_policy_is_executed_and_charges_all_tokens(setup, tmp_path, monkeypatch, policy):
    from agentinputlab.backends.llm import OpenAICompatibleBackend

    requests = []
    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return {'model': 'fixture', 'choices': [{'message': {'content': 'answer END tail',
            'reasoning_content': 'END in reasoning'}, 'finish_reason': 'length'}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}
    original_init = OpenAICompatibleBackend.__post_init__
    def initialize(self):
        original_init(self)
        self.transport = transport
    monkeypatch.setattr(OpenAICompatibleBackend, '__post_init__', initialize)
    monkeypatch.setenv('FIXTURE_KEY', 'synthetic')
    class StoppingMethod(Method):
        def propose(self, state, context):
            response = context.backend.generate(GenerationRequest(prompt=state.query, max_tokens=8, stop=(' END',)))
            assert response.text == ('answer' if policy == 'final_content' else 'answer END tail')
            assert response.usage.output_tokens == 3
            if policy == 'final_content':
                assert response.raw['agentinputlab_final_content_stop']['provider_usage_unchanged']
            return InputPatch.no_op(state, method_id=self.method_id, release_id=self.release_id)
    monkeypatch.setattr(RunConfig, 'create_intervention', lambda self: StoppingMethod())
    run, native = setup
    run = replace(run, backend={**BACKEND, 'stop_policy': policy})
    config = native.provider_config(run)
    provider = Provider()
    row = arena._run_case(run, provider, config, selected(run, config), tmp_path)
    assert row['status'] == 'scored'
    assert len(requests) == 1 and len(provider.calls) == 1
    assert requests[0].get('stop') == (None if policy == 'final_content' else [' END'])
    assert row['usage']['output_tokens'] == 8
    assert row['model_responses'][0]['response']['text'] == (
        'answer' if policy == 'final_content' else 'answer END tail')


@pytest.mark.parametrize("kind", ["tooltalk", "statebench", "toolsandbox"])
def test_native_extra_body_binds_agent_and_environment_without_budget_changes(setup, kind):
    run, native = setup
    native = replace(native, arena=kind, environment_backend={} if kind == "tooltalk" else {
        **BACKEND, "extra_body": {"top_p": 0.8}})
    config = native.provider_config(replace(run, backend={**BACKEND,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_p": 0.9}}))
    assert config["agent_extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}, "top_p": 0.9}
    if kind != "tooltalk":
        assert config["environment_extra_body"] == {"top_p": 0.8}
    assert config["budget_override"] == native.provider_config(run)["budget_override"]


@pytest.mark.parametrize('kind',['tooltalk','statebench','toolsandbox'])
def test_native_extension_replace_mode_is_explicit_and_budget_neutral(setup,kind):
    from workers.official_arenas.protocol import native_extra_body
    run,native=setup
    environment={} if kind=='tooltalk' else {**BACKEND,'extra_body':{'thinking':{'type':'disabled'}}}
    native=replace(native,arena=kind,environment_backend=environment,agent_extra_body_mode='replace',
        environment_extra_body_mode='merge' if kind=='tooltalk' else 'replace')
    configured=replace(run,backend={**BACKEND,'extra_body':{'thinking':{'type':'disabled'}}})
    cfg=native.provider_config(configured)
    defaults={'chat_template_kwargs':{'enable_thinking':False}}
    assert native_extra_body(cfg,'agent_extra_body',defaults=defaults)=={'thinking':{'type':'disabled'}}
    if kind!='tooltalk':
        assert native_extra_body(cfg,'environment_extra_body',defaults=defaults)=={'thinking':{'type':'disabled'}}
    legacy=replace(native,agent_extra_body_mode='merge',environment_extra_body_mode='merge').provider_config(configured)
    assert 'agent_extra_body_mode' not in legacy
    assert 'environment_extra_body_mode' not in legacy
    assert cfg['budget_override']==legacy['budget_override']


@pytest.mark.parametrize('mode', [None, True, [], {}, '', 'auto'])
@pytest.mark.parametrize('field', ['agent_extra_body_mode', 'environment_extra_body_mode'])
def test_native_extension_mode_rejects_invalid_values(setup, mode, field):
    _, native = setup
    with pytest.raises(ValidationError, match='merge or replace'):
        replace(native, **{field: mode})


@pytest.mark.parametrize('kind,field', [('tooltalk', 'environment_extra_body_mode'),
    ('tau_retail', 'agent_extra_body_mode'), ('tau_banking', 'environment_extra_body_mode')])
def test_native_extension_mode_rejects_unsupported_roles(setup, kind, field):
    _, native = setup
    with pytest.raises(ValidationError, match='unsupported'):
        replace(native, arena=kind, **{field: 'replace'})


def test_native_extension_mode_round_trips_from_file(setup, tmp_path):
    _, native = setup
    path = tmp_path / 'arena.json'
    path.write_text(json.dumps({'arena': native.arena, 'checkout': native.checkout,
        'agent_extra_body_mode': 'replace'}))
    loaded = arena.NativeArenaConfig.from_file(path)
    assert loaded.agent_extra_body_mode == 'replace'
    assert loaded.environment_extra_body_mode == 'merge'


def test_state_reasoning_option_binds_only_the_agent_role(setup, tmp_path):
    run, native = setup
    with pytest.raises(ValidationError, match="only supported for STATE-Bench"):
        replace(native, agent_responses_reasoning_disabled=True)
    state = replace(native, arena="statebench", environment_backend=BACKEND)
    with pytest.raises(ValidationError, match="must be boolean"):
        replace(state, agent_responses_reasoning_disabled="false")
    default = state.provider_config(run)
    assert "agent_responses_reasoning_disabled" not in default
    path = tmp_path / "state-arena.json"
    path.write_text(json.dumps(dict(arena="statebench", checkout=native.checkout,
                                    environment_backend=BACKEND, agent_responses_reasoning_disabled=True)))
    configured = arena.NativeArenaConfig.from_file(path).provider_config(run)
    assert configured == {**default, "agent_responses_reasoning_disabled": True}
    assert arena._provider(state)._validate_config(configured) == configured


@pytest.mark.parametrize("kind", ["tooltalk", "statebench", "toolsandbox"])
def test_native_timeouts_bind_agent_and_environment_separately(setup, kind):
    run, native = setup
    native = replace(native, arena=kind, environment_backend={} if kind == "tooltalk" else {
        **BACKEND, "timeout_seconds": 73.5})
    config = native.provider_config(replace(run, backend={**BACKEND, "timeout_seconds": 31}))
    assert config["agent_timeout_seconds"] == 31
    if kind != "tooltalk":
        assert config["environment_timeout_seconds"] == 73.5
    else:
        assert "environment_timeout_seconds" not in config
    # Timeout settings cannot silently alter model/token/tool budgets.
    assert config["budget_override"] == native.provider_config(run)["budget_override"]


@pytest.mark.parametrize("value", [True, 0, -1, "30", None])
def test_native_environment_rejects_invalid_timeout(setup, value):
    _, native = setup
    with pytest.raises(ValidationError):
        replace(native, arena="statebench", environment_backend={**BACKEND, "timeout_seconds": value})


@pytest.mark.parametrize("kind", ["tooltalk", "statebench", "toolsandbox"])
@pytest.mark.parametrize("options", [{}, {"agent_timeout_seconds": 31},
    {"agent_timeout_seconds": 31, "environment_timeout_seconds": 73.5}])
def test_provider_accepts_optional_timeouts_and_rejects_invalid_values(setup, monkeypatch, kind, options):
    run, native = setup
    native = replace(native, arena=kind, environment_backend={} if kind == "tooltalk" else BACKEND)
    config = native.provider_config(run)
    monkeypatch.setenv("FIXTURE_KEY", "fixture")
    provider = arena._provider(native)
    from workers.official_arenas.protocol import OfficialArenaError
    if kind == "tooltalk" and "environment_timeout_seconds" in options:
        with pytest.raises(OfficialArenaError, match="fields"):
            provider._validate_config({**config, **options})
        return
    assert provider._validate_config({**config, **options})["agent_model"] == BACKEND["model_id"]
    with pytest.raises(OfficialArenaError, match="finite positive"):
        provider._validate_config({**config, **options, "agent_timeout_seconds": False})
    with pytest.raises(OfficialArenaError, match="fields"):
        provider._validate_config({**config, "typo_timeout": 1})


@pytest.mark.parametrize("attempts", [True, 0, 2, 1.0, None, "1"])
def test_state_retry_policy_cannot_silently_expand_attempts(setup, attempts):
    run, native = setup
    native = replace(native, arena="statebench", environment_backend=BACKEND)
    config = native.provider_config(run)
    from workers.official_arenas.protocol import OfficialArenaError
    with pytest.raises(OfficialArenaError, match="request_attempts must be 1"):
        arena._provider(native)._validate_config({**config, "request_attempts": attempts})


def test_relative_checkout_and_duplicate_config_fields(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"arena": "tooltalk", "checkout": "upstream"}))
    assert arena.NativeArenaConfig.from_file(path).checkout == str(tmp_path / "upstream")
    path.write_text('{"arena":"tooltalk","arena":"statebench","checkout":"upstream"}')
    with pytest.raises(ValidationError, match="duplicate"):
        arena.NativeArenaConfig.from_file(path)


def test_worker_python_resolves_relative_to_config_file(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"arena": "tooltalk", "checkout": "native",
                                "worker_python": "env/bin/python"}))
    native = arena.NativeArenaConfig.from_file(path)
    assert native.worker_python == str(tmp_path / "env/bin/python")


@pytest.mark.parametrize("value", [False, 12, "", "  "])
def test_invalid_worker_python_is_rejected(setup, value):
    _, native = setup
    with pytest.raises(ValidationError):
        replace(native, worker_python=value)


def test_custom_worker_python_is_used_without_host_pythonpath(setup, tmp_path, monkeypatch):
    import sys

    monkeypatch.setenv('PYTHONPATH', '/host-only/dependencies')
    calls = []
    def process(command, **options):
        calls.append((command, options))
        arena._save(tmp_path / 'terminal.json', {'status': 'captured'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(arena.subprocess, 'run', process)
    arena._launch({'work_dir': str(tmp_path), 'worker_python': sys.executable}, tmp_path / 'job.json')
    assert calls[0][0][0] == sys.executable
    assert 'PYTHONPATH' not in calls[0][1]['env']
    assert calls[0][1]['env']['PYTHONNOUSERSITE'] == '1'


def test_missing_custom_interpreter_never_falls_back(setup, tmp_path):
    with pytest.raises(ValidationError, match='worker_python'):
        arena._launch({'work_dir': str(tmp_path), 'worker_python': str(tmp_path / 'missing')},
                      tmp_path / 'job.json')


def test_custom_worker_jobs_bind_same_implementation(setup, tmp_path, monkeypatch):
    import sys

    run, native = setup
    native = replace(native, worker_python=sys.executable)
    jobs = []
    def launch(job, path):
        jobs.append(job)
        return {'status': 'captured', 'case_ids': ['one'], 'inventory_sha256': 'a' * 64}
    monkeypatch.setattr(arena, '_launch', launch)
    arena.evaluate_arena(run, native, output_dir=tmp_path / 'isolated', inventory_only=True)
    assert jobs[0]['worker_python'] == sys.executable
    assert jobs[0]['worker_code_sha256'] == arena._implementation_digest()


def test_worker_rejects_other_implementation_before_native_loading(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    work.mkdir()
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'work_dir': str(work), 'worker_code_sha256': '0' * 64}))
    monkeypatch.setattr(arena, '_provider', lambda *a: pytest.fail('must not load native provider'))
    assert arena._worker(job) == 1
    terminal = json.loads((work / 'terminal.json').read_text())
    assert terminal['stage'] == 'native_preflight'
    assert 'implementation' in terminal['error']
    receipt = json.loads((work / 'worker-runtime.json').read_text())
    assert receipt['implementation_sha256'] != '0' * 64


def test_worker_identity_includes_runtime_manifests_but_not_bytecode(tmp_path, monkeypatch):
    from workers.official_arenas import protocol

    package = tmp_path / 'agentinputlab'
    worker = tmp_path / 'workers/official_arenas'
    package.mkdir()
    worker.mkdir(parents=True)
    (package / 'arena_configuration.py').write_text('fixture')
    (worker / 'protocol.py').write_text('fixture')
    manifest = worker / 'compatibility_manifest.json'
    manifest.write_text('{"revision": 1}')
    monkeypatch.setattr(arena, '__file__', str(package / 'arena_configuration.py'))
    monkeypatch.setattr(protocol, '__file__', str(worker / 'protocol.py'))
    before = arena._implementation_digest()
    manifest.write_text('{"revision": 2}')
    after = arena._implementation_digest()
    assert before != after
    cache = worker / '__pycache__'
    cache.mkdir()
    (cache / 'protocol.cpython-312.pyc').write_bytes(b'generated')
    assert arena._implementation_digest() == after


def selected(run, config):
    return {"checkpoint_id": "fixture-case", "protected": {
        "locked_dependency_hashes": config["locked_dependency_hashes"], "gold": "NEVER_SEND_GOLD"},
        "visible_state": {"task_id": "fixture-case", "query": "Visible question", "budget": to_jsonable(run.budget),
            "tool_names": ["lookup"], "context": [{"block_id": "official_tool_catalog", "text": json.dumps([
                {"name": "lookup", "description": "Lookup", "parameters": {"type": "object", "properties": {}}}])}]}}


class Provider:
    scorer_id = "native-fixture"
    scorer_revision = source_revision = "a" * 64

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def run_arm(self, config, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["replay_record"]["protected"]["gold"] == "NEVER_SEND_GOLD"
        payload = kwargs["arm_payload"]["intervention_execution"]
        assert payload["usage"]["model_calls"] == 1
        assert payload["usage"]["output_tokens"] == 3
        assert payload["materialized_state"]["budget"]["max_output_tokens"] == 4096
        if self.fail:
            raise RuntimeError("token=secret-should-not-be-displayed")
        return ProviderArmResult(metrics={"official_success": 0.0}, trace={"native": True},
                                 usage={"model_calls": 1, "output_tokens": 5}, runtime_instance_id="b" * 64)


class Method:
    method_id = "fixture"
    release_id = "baseline.no_op"
    family = SemanticFamily.REFORMULATE

    def propose(self, state, context):
        context.backend.generate(GenerationRequest(prompt=state.query, max_tokens=8))
        return InputPatch.no_op(state, method_id=self.method_id, release_id=self.release_id)


@pytest.mark.parametrize("fail", [False, True])
def test_native_scoring_preserves_budget_failure_receipts_and_gold_boundary(setup, tmp_path, monkeypatch, fail):
    run, native = setup
    config = native.provider_config(run)
    backend = Backend([("ok", 3)])
    monkeypatch.setattr(RunConfig, "_backend", lambda self: backend)
    monkeypatch.setattr(RunConfig, "create_intervention", lambda self: Method())
    provider = Provider(fail)
    result = arena._run_case(run, provider, config, selected(run, config), tmp_path)
    assert len(backend.requests) == 1 and len(provider.calls) == 1
    assert "NEVER_SEND_GOLD" not in backend.requests[0].prompt
    assert result["intervention_usage"]["output_tokens"] == 3
    assert (tmp_path / "intervention.json").is_file()
    assert (tmp_path / "terminal.json").is_file()
    assert "secret-should-not-be-displayed" not in json.dumps(result)
    if fail:
        assert result["status"] == "failed" and "metrics" not in result
        assert result["error_type"] == "RuntimeError"
        assert "redacted" in result["error"]
    else:
        assert result["metrics"] == {"official_success": 0.0}
        assert result["usage"]["model_calls"] == 2 and result["usage"]["output_tokens"] == 8


def test_failed_intervention_keeps_consumed_usage_and_never_starts_native_episode(setup, tmp_path, monkeypatch):
    run, native = setup
    config = native.provider_config(run)
    backend = Backend([RuntimeError("API request failed token=hidden-token")])
    monkeypatch.setattr(RunConfig, "_backend", lambda self: backend)
    monkeypatch.setattr(RunConfig, "create_intervention", lambda self: Method())
    provider = Provider()
    result = arena._run_case(run, provider, config, selected(run, config), tmp_path)
    assert len(backend.requests) == 1 and not provider.calls
    assert result["status"] == "failed" and result["stage"] == "intervention"
    assert result["intervention_usage"]["model_calls"] == 1
    assert result["intervention_monitor"][0]["status"] == "failed"
    assert "hidden-token" not in json.dumps(result)
    assert "metrics" not in result


def test_capture_budget_mismatch_fails_before_method_or_model(setup, tmp_path):
    run, native = setup
    config = native.provider_config(run)
    record = selected(run, config)
    record["visible_state"]["budget"]["max_output_tokens"] = 8192
    with pytest.raises(ValidationError, match="budget differs"):
        arena._run_case(run, Provider(), config, record, tmp_path)


@pytest.mark.parametrize("options", [{}, {"case_ids": []}, {"case_ids": ["x", "x"]}, {"limit": True},
                                    {"case_ids": ["x"], "limit": 1}])
def test_bad_selection_never_creates_output(setup, tmp_path, options):
    run, native = setup
    with pytest.raises(ValidationError):
        arena.evaluate_arena(run, native, output_dir=tmp_path / "output", **options)
    assert not (tmp_path / "output").exists()


def test_batch_uses_distinct_jobs_and_keeps_failed_denominator(setup, tmp_path, monkeypatch):
    run, native = setup
    jobs = []
    def launch(job, path):
        jobs.append(job)
        if job["mode"] == "capture":
            return {"status": "captured", "case_ids": ["a", "b"], "inventory_sha256": "a" * 64}
        return {"status": "failed" if job["case_id"] == "b" else "scored", "metrics": {"score": 0}}
    monkeypatch.setattr(arena, "_launch", launch)
    result = arena.evaluate_arena(run, native, output_dir=tmp_path / "output", case_ids=["a", "b"])
    assert result["selected_count"] == 2 and result["failed_count"] == 1
    assert len(jobs) == 3 and len({j["work_dir"] for j in jobs}) == 3
    with pytest.raises(ValidationError, match="already exists"):
        arena.evaluate_arena(run, native, output_dir=tmp_path / "output", limit=1)
    assert len(jobs) == 3


def test_list_cases_and_cli_require_explicit_selection(setup, tmp_path, monkeypatch, capsys):
    run, native = setup
    run_path, native_path = tmp_path / "run.json", tmp_path / "arena.json"
    arena._save(run_path, run.to_dict())
    arena._save(native_path, native)
    monkeypatch.setattr(arena, "_launch", lambda *a: {"status": "captured", "case_ids": ["a"], "inventory_sha256": "a" * 64})
    assert main(["config", "evaluate-arena", str(run_path), "--arena-config", str(native_path),
                 "--list-cases", "--output-dir", str(tmp_path / "out")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["selected_count"] == 0 and summary["available_case_ids"] == ["a"]


def test_subprocess_command_is_not_shell_or_retry(setup, tmp_path, monkeypatch):
    jobs = []
    def process(command, **options):
        jobs.append((command, options))
        arena._save(tmp_path / "terminal.json", {"status": "failed"})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(arena.subprocess, "run", process)
    row = arena._launch({"work_dir": str(tmp_path)}, tmp_path / "job.json")
    assert row["status"] == "failed" and len(jobs) == 1
    assert jobs[0][0][1:3] == ["-m", "agentinputlab.arena_configuration"]
    assert "shell" not in jobs[0][1] and "timeout" not in jobs[0][1]


def test_tooltalk_scoring_dependency_is_prepared_once_before_agent_use(setup, monkeypatch):
    _, native = setup
    calls = []
    def construct():
        calls.append("load original scorer")
        return object()
    module = SimpleNamespace(_vectorize_text=None, _TextVectorizer=construct)
    monkeypatch.setattr(arena.importlib, "import_module", lambda name: module)
    arena._prepare_scoring_runtime(native)
    instance = module._vectorize_text
    arena._prepare_scoring_runtime(native)
    assert module._vectorize_text is instance and len(calls) == 1


def test_missing_tooltalk_scorer_is_not_silently_substituted(setup, monkeypatch):
    _, native = setup
    def missing():
        raise OSError("pinned scorer weights missing from configured cache")
    monkeypatch.setattr(arena.importlib, "import_module", lambda name: SimpleNamespace(
        _vectorize_text=None, _TextVectorizer=missing))
    with pytest.raises(OSError, match="weights missing"):
        arena._prepare_scoring_runtime(native)
