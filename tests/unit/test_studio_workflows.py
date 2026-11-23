"""Web/CLI parity and persistence. Fixture replies are not real benchmark evidence."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.studio import workflows
from agentinputlab.studio.server import create_server
from agentinputlab.studio.workflows import WorkflowBusyError, WorkflowManager, generate_workflow
from agentinputlab.workflows import WorkflowConfig, execution, planning, workflow_template
from tests.unit.test_workflows import Adapter, benchmark
from tests.unit.test_workflows import provider as provider


def wait(manager, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = manager.get(job_id)
        if not result['active']:
            return result
        time.sleep(.01)
    pytest.fail('workflow did not terminate')


@pytest.fixture
def web(tmp_path):
    server = create_server(port=0, workflow_root=tmp_path / 'web')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None, headers=None, raw=None):
        data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        h = {'Content-Type': 'application/json', 'Origin': base,
             'X-Studio-Session': server.workflow_session}
        h.update(headers or {})
        req = urllib.request.Request(base + path, data=data, headers=h)
        try:
            response = opener.open(req, timeout=15)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read()
            return response.status, (json.loads(body) if 'application/json' in response.headers.get('Content-Type', '') else body)
    try:
        yield server.workflow_manager, request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_builder_discovery_and_editable_native_config(web):
    _, request = web
    status, catalog = request('/api/workflows/catalog')
    assert status == 200 and catalog['execution_enabled']
    assert len(catalog['benchmarks']) == 31
    assert len(catalog['presets']) == 6
    assert any(r['route'] == 'native_python_api' for r in catalog['recipes'])
    for path in ('/workflows', '/workflows.js', '/workflows.css'):
        assert request(path)[0] == 200
    status, value = request('/api/workflows/template', {
        'release_id': 'plan.plan_and_solve.ps_prompt', 'benchmark_id': 'bfcl.v3.simple_python',
        'backend': {'model_id': 'user-model', 'base_url': 'http://localhost:8000/v1',
                    'deployment_identity': {'revision': 'explicit-user-revision'}},
        'budget': {'max_output_tokens': 321, 'max_model_calls': 2}, 'seed': 19, 'max_workers': 2,
    })
    assert status == 200
    config = WorkflowConfig.from_dict(value['workflow'])
    assert config.run.budget.max_output_tokens == 321 and config.run.seed == 19
    assert config.run.backend['model_id'] == 'user-model'
    assert config.target['benchmark_id'] == 'bfcl.v3.simple_python'
    assert config.max_workers == 2
    assert RunConfig.from_dict(value['run']).to_dict() == config.run.to_dict()


def test_web_prompt_launch_preview_restart_and_resume_without_reexecution(web, monkeypatch):
    manager, request = web
    workflow = generate_workflow({'query': 'Find a flight'})['workflow']
    status, inspection = request('/api/workflows/inspect', {'workflow': workflow})
    assert status == 200 and inspection['can_attempt'] and inspection['model_calls'] == 0
    status, job = request('/api/workflows/launch', {'workflow': workflow})
    assert status == 202
    finished = wait(manager, job['id'])
    assert finished['status'] == 'completed'
    assert finished['preview']['records'][0]['scored'] is False
    assert finished['preview']['records'][0]['output']['state']['instruction']
    journal = manager.root / job['id'] / 'launch/run/events.jsonl'
    before = journal.read_bytes()
    monkeypatch.setattr(RunConfig, 'run', lambda *_a, **_kw: pytest.fail('repeated generation'))
    # A new manager discovers disk records; neither a page refresh nor restart loses them.
    reopened = WorkflowManager(manager.root)
    try:
        assert reopened.list_runs()[0]['id'] == job['id']
        resumed = reopened.submit(resume_id=job['id'])
        assert wait(reopened, resumed['id'])['status'] == 'completed'
    finally:
        reopened.shutdown()
    assert journal.read_bytes() == before


@pytest.mark.parametrize('headers', [
    {'Origin': 'https://evil.example'}, {'Origin': 'null'},
    {'Sec-Fetch-Site': 'cross-site'}, {'X-Studio-Session': ''},
])
def test_foreign_pages_cannot_submit_local_workflows(web, headers):
    _, request = web
    assert request('/api/workflows/template', {}, headers=headers)[0] == 403


@pytest.mark.parametrize('raw', [b'{"query":"a","query":"b"}', b'{"seed":NaN}', b'[]'])
def test_invalid_json_rejected_before_preparation(web, raw):
    _, request = web
    assert request('/api/workflows/template', raw=raw)[0] in {400, 422}


def test_secrets_unknown_fields_bad_paths_and_implicit_launch_rejected(web):
    manager, request = web
    status, _ = request('/api/workflows/template', {'backend': {'api_key': 'opaque-secret'}})
    assert status == 422
    assert request('/api/workflows/template', {'shell': 'echo injected'})[0] == 422
    assert request('/api/workflows/resume', {'id': '../../elsewhere'})[0] == 422
    assert request('/api/workflows/launch', {'workflow': {}, 'output_dir': '/tmp/elsewhere'})[0] == 422
    assert not manager.list_runs()


def test_disabled_execution_still_offers_templates_and_remote_mode_rejected(tmp_path):
    server = create_server(port=0)
    try:
        assert server.workflow_manager is None
    finally:
        server.server_close()
    with pytest.raises(ValueError, match='loopback'):
        create_server(host='0.0.0.0', port=0, allow_remote=True, workflow_root=tmp_path)


def test_busy_run_and_interrupted_guard_do_not_duplicate_attempts(tmp_path, monkeypatch):
    manager = WorkflowManager(tmp_path)
    entered, finish = threading.Event(), threading.Event()
    def hold(*_a, **_kw):
        entered.set()
        assert finish.wait(5)
        return {'status': 'completed'}
    monkeypatch.setattr(workflows, 'launch_workflow', hold)
    try:
        job = manager.submit(workflow_template().to_dict())
        assert entered.wait(5)
        with pytest.raises(WorkflowBusyError):
            manager.submit(resume_id=job['id'])
        finish.set()
        wait(manager, job['id'])
        (tmp_path / job['id'] / '.studio-running').write_text('old-process')
        with pytest.raises(WorkflowBusyError, match='reconcile|Reconcile'):
            manager.submit(resume_id=job['id'])
    finally:
        finish.set()
        manager.shutdown()


def test_changed_saved_workflow_does_not_resume(tmp_path):
    manager = WorkflowManager(tmp_path)
    try:
        job = manager.submit(workflow_template().to_dict())
        wait(manager, job['id'])
        path = tmp_path / job['id'] / 'workflow.json'
        value = json.loads(path.read_text())
        value['run']['seed'] += 1
        path.write_text(json.dumps(value))
        with pytest.raises(ValidationError, match='identity changed'):
            manager.submit(resume_id=job['id'])
    finally:
        manager.shutdown()


def test_blocked_native_python_and_failed_attempt_are_not_success(tmp_path, monkeypatch):
    manager = WorkflowManager(tmp_path)
    try:
        config = workflow_template(release_id='acquire.r2ag.controlled_same_candidates').to_dict()
        inspection = manager.inspect(config)
        assert not inspection['can_attempt']
        assert inspection['model_calls'] == 0
        monkeypatch.setattr(RunConfig, 'run', lambda *_a, **_kw: (_ for _ in ()).throw(ValueError('fixture failure')))
        job = manager.submit(workflow_template().to_dict())
        failed = wait(manager, job['id'])
        assert failed['status'] == 'completed_with_failures'
        assert failed['preview']['records'][0]['status'] == 'failed'
        before = (tmp_path / job['id'] / 'launch/run/events.jsonl').read_bytes()
        monkeypatch.setattr(RunConfig, 'run', lambda *_a, **_kw: pytest.fail('retry'))
        manager.submit(resume_id=job['id'])
        assert wait(manager, job['id'])['status'] == 'completed_with_failures'
        assert (tmp_path / job['id'] / 'launch/run/events.jsonl').read_bytes() == before
    finally:
        manager.shutdown()


def test_default_presets_generate_and_retained_inputs_are_not_invented():
    for preset in workflows.PRESETS:
        config = generate_workflow({'benchmark_id': preset['benchmark_id']})['workflow']
        assert config['run']['release_id'] == 'plan.plan_and_solve.ps_prompt'
        assert not config['run']['backend']
    with pytest.raises(ArtifactError):
        generate_workflow({'parameters': {'password': 'do-not-persist'}})


def test_import_run_keeps_budget_seed_and_method_parameters():
    run = RunConfig('plan.plan_and_solve.ps_prompt', seed=29).to_dict()
    run['budget']['max_output_tokens'] = 700
    generated = generate_workflow({'run': run, 'benchmark_id': 'bfcl.v3.simple_python'})
    assert generated['run']['seed'] == 29
    assert generated['run']['budget']['max_output_tokens'] == 700
    assert generated['run']['parameters'] == run['parameters']
    with pytest.raises(ValidationError, match='choose run'):
        generate_workflow({'run': run, 'release_id': 'baseline.no_op'})


def test_web_native_scoring_keeps_zero_and_resume_reuses_responses(web, monkeypatch, provider):  # noqa: F811
    manager, request = web
    backend, calls, answers = provider
    answers.extend(['answer', 'wrong'])
    adapter = Adapter()
    monkeypatch.setattr(planning, 'load_adapter', lambda *_a, **_kw: adapter)
    monkeypatch.setattr(execution, 'load_adapter', lambda *_a, **_kw: adapter)
    status, job = request('/api/workflows/launch', {'workflow': benchmark(backend).to_dict()})
    assert status == 202, job
    finished = wait(manager, job['id'])
    assert finished['status'] == 'completed'
    records = finished['preview']['records']
    assert [r['evaluation']['metrics']['answer_em'] for r in records] == [1.0, 0.0]
    assert sum(verb == 'POST' for verb, _ in calls) == 2
    assert request('/api/workflows/resume', {'id': job['id']})[0] == 202
    assert wait(manager, job['id'])['status'] == 'completed'
    assert sum(verb == 'POST' for verb, _ in calls) == 2


def test_nested_run_symlink_is_not_followed(tmp_path):
    manager = WorkflowManager(tmp_path / 'root')
    try:
        job = manager.submit(workflow_template().to_dict())
        wait(manager, job['id'])
        run = manager.root / job['id'] / 'launch/run'
        run.rename(run.with_name('saved-run'))
        run.symlink_to(run.with_name('saved-run'), target_is_directory=True)
        with pytest.raises(ValidationError, match='symlink'):
            manager.get(job['id'])
    finally:
        manager.shutdown()


def test_web_inspection_serializes_nested_artifact_model_identity(web, tmp_path, provider, monkeypatch):  # noqa: F811
    from tests.unit.test_workflow_artifacts import catalog_workflow
    backend, calls, _ = provider
    backend = {**backend, "deployment_identity": {
        "revision": "fixture", "model_json_hashes": {"config.json": "a" * 64}}}
    workflow, _, _ = catalog_workflow(tmp_path, backend)
    monkeypatch.setattr(planning, "load_adapter", lambda *_a, **_k: Adapter())
    _, request = web
    status, report = request('/api/workflows/inspect', {'workflow': workflow.to_dict()})
    assert status == 200 and report['can_attempt'] and calls == []
    details = next(c['details'] for c in report['checks'] if c['check'] == 'artifact_discovery')
    assert details['identity']['deployment_identity'] == backend['deployment_identity']


def test_status_snapshot_cannot_mix_running_record_with_finished_activity(tmp_path, monkeypatch):
    entered = threading.Event()
    finish = threading.Event()
    execution_returning = threading.Event()

    def launch(*_args, **_kwargs):
        entered.set()
        assert finish.wait(5)
        execution_returning.set()
        return {'status': 'completed', 'all_attempts_terminal': True}

    monkeypatch.setattr(workflows, 'launch_workflow', launch)
    manager = WorkflowManager(tmp_path / 'root')
    try:
        job = manager.submit(workflow_template().to_dict())
        assert entered.wait(5)
        original_read = workflows.read_json
        intercepted = False

        def read_during_completion(path):
            nonlocal intercepted
            value = original_read(path)
            if path.name == 'status.json' and not intercepted:
                intercepted = True
                assert value['status'] == 'running'
                finish.set()
                assert execution_returning.wait(5)
                # Give the worker time to attempt its final state transition.
                time.sleep(.05)
            return value

        monkeypatch.setattr(workflows, 'read_json', read_during_completion)
        snapshot = manager.get(job['id'])
        assert snapshot['status'] == 'running' and snapshot['active']
        final = wait(manager, job['id'])
        assert final['status'] == 'completed' and not final['active']
    finally:
        finish.set()
        manager.shutdown()
