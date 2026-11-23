"""Non-neural fixtures for actual GRPO orchestration seams; no model acceptance."""
import ast
import contextlib
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes.smartsearch_grpo_data import SmartSearchGRPOControls
from agentinputlab.recipes.smartsearch_grpo_meter import (
    SmartSearchGRPOCostLedger,
    SmartSearchGRPOCostLimits,
)
from agentinputlab.recipes.smartsearch_grpo_training import (
    SmartSearchNativeGRPOTraining,
    _visible_actor_devices,
)
from agentinputlab.recipes.smartsearch_grpo_transport import (
    SmartSearchGRPOCoordinator,
    SmartSearchGRPOCoordinatorClient,
    make_smartsearch_grpo_coordinator_server,
)
from agentinputlab.recipes.smartsearch_grpo_worker import (
    _log_safe,
    _nonfinite,
    _task_runner_ast,
    _verify_runtime,
)
from agentinputlab.recipes.smartsearch_teacher import SmartSearchTeacherCompletion


@contextlib.contextmanager
def service(tmp_path, generate=None, **overrides):
    settings = dict(policy_requests=2, policy_output_tokens=32, helper_requests=2, helper_output_tokens=4096,
                    actor_ranks=2, optimizer_steps_per_rank=2)
    settings.update(overrides)
    ledger = SmartSearchGRPOCostLedger(output_root=tmp_path / "ledger", limits=SmartSearchGRPOCostLimits(**settings))
    generate = generate or (lambda *a, **kw: SmartSearchTeacherCompletion("explicit fixture", 3, 4, "a" * 64))
    coordinator = SmartSearchGRPOCoordinator(ledger=ledger, generate=generate, count_input_tokens=lambda _: 3,
        model_fingerprint_sha256="a" * 64)
    server = make_smartsearch_grpo_coordinator_server(coordinator, capability="f" * 32)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield coordinator, f"http://127.0.0.1:{server.server_port}/{'f' * 32}/rpc"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_two_worker_clients_share_generation_helper_and_optimizer_allowances(tmp_path):
    with service(tmp_path) as (coordinator, endpoint):
        clients = [SmartSearchGRPOCoordinatorClient(endpoint) for _ in range(2)]
        for rank, client in enumerate(clients):
            ticket = client.reserve(kind="policy", worker=f"fixture-rank-{rank}",
                input_token_counts=[2], samples=1, max_new_tokens=8)
            client.settle(ticket, output_token_counts=[3])
            reply = client.generate("fixture", temperature=.6, max_new_tokens=2048, stop=(), seed=17)
            assert reply.text == "explicit fixture" and reply.output_tokens == 4
            for _ in range(2):
                client.optimizer_begin(rank)
                client.optimizer_end(rank)
            for role in ("actor_rollout", "ref"):
                client.worker_completed(rank=rank, role=role, usage=dict(
                    forward_attempts=2, forward_responses=2, input_token_positions=4, nonfinite_native_metrics=0))
        result = coordinator.assert_healthy(expected_optimizer_steps=2, required_roles=("actor_rollout", "ref"))
        assert result["costs"]["helper"]["requests"] == 2 and result["costs"]["policy"]["output_tokens"] == 6
        assert result["synchronized_optimizer_steps"] == 2 and len(result["workers"]) == 4
        assert clients[0].snapshot()["costs"] == result["costs"]


def test_unauthenticated_request_does_not_touch_the_run(tmp_path):
    with service(tmp_path) as (coordinator, endpoint):
        request = urllib.request.Request(endpoint.replace("f" * 32, "z" * 32),
            data=b'{}', headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request)
        assert caught.value.code == 401 and coordinator.ledger.snapshot()["failed"] is False
        assert coordinator.ledger.snapshot()["costs"]["helper"]["requests"] == 0


def test_no_arbitrary_operation_or_code_execution_fallback(tmp_path):
    with service(tmp_path) as (coordinator, endpoint):
        client = SmartSearchGRPOCoordinatorClient(endpoint)
        with pytest.raises(ExternalDependencyError):
            client._rpc("execute_python", {"code": "raise AssertionError('not executed')"})
        assert coordinator.ledger.snapshot()["failed"]
        assert coordinator.ledger.snapshot()["costs"]["helper"]["requests"] == 0


def test_disconnected_or_failed_provider_is_not_repeated(tmp_path):
    consumed = []
    def failed(*args, **kwargs):
        consumed.append(kwargs)
        raise ConnectionError("explicit non-neural provider failure fixture")
    with service(tmp_path, generate=failed) as (coordinator, endpoint):
        client = SmartSearchGRPOCoordinatorClient(endpoint)
        for _ in range(2):
            with pytest.raises(ExternalDependencyError):
                client.generate("fixture", temperature=.6, max_new_tokens=2048, stop=(), seed=17)
        assert len(consumed) == 1
        costs = coordinator.ledger.snapshot()["costs"]["helper"]
        assert costs["requests"] == 1 and costs["responses"] == 0 and costs["reserved_output_tokens"] == 2048


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:12345/" + "f" * 32 + "/rpc",
    "http://example.com:12345/" + "f" * 32 + "/rpc",
    "http://127.0.0.1:12345/short/rpc",
    "http://user:pass@127.0.0.1:12345/" + "f" * 32 + "/rpc",
    "http://127.0.0.1:12345/" + "f" * 32 + "/rpc?redirect=1",
])
def test_coordinator_transport_stays_loopback(url):
    with pytest.raises(ValidationError):
        SmartSearchGRPOCoordinatorClient(url)


def test_missing_rank_or_forward_usage_cannot_accept_a_fit(tmp_path):
    with service(tmp_path) as (coordinator, endpoint):
        client = SmartSearchGRPOCoordinatorClient(endpoint)
        for rank in range(2):
            client.optimizer_begin(rank)
            client.optimizer_end(rank)
        with pytest.raises(ExternalDependencyError, match="receipts"):
            coordinator.assert_healthy(expected_optimizer_steps=1, required_roles=("actor_rollout", "ref"))


def test_duplicate_keys_and_nonfinite_json_fail_before_helper(tmp_path):
    with service(tmp_path) as (coordinator, endpoint):
        request = urllib.request.Request(endpoint, data=b'{"operation":"helper","operation":"helper","arguments":{}}',
            headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request)
        assert coordinator.ledger.snapshot()["failed"]
        assert coordinator.ledger.snapshot()["costs"]["helper"]["requests"] == 0


def test_mesh_and_cost_admission_are_explicit(monkeypatch):
    controls = SmartSearchGRPOControls(train_batch_size=4, ppo_mini_batch_size=2,
                                     gpu_processes=2, bounded_training_variant=True)
    limits = SmartSearchGRPOCostLimits(10, 81920, 10, 20480, 2, 8)
    recipe = SmartSearchNativeGRPOTraining(controls=controls, cost_limits=limits)
    assert recipe.controls is controls
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    assert _visible_actor_devices(controls) == "0,2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    with pytest.raises(ValidationError):
        _visible_actor_devices(controls)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0, 2")
    with pytest.raises(ValidationError):
        _visible_actor_devices(controls)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValidationError):
        _visible_actor_devices(controls)
    with pytest.raises(ValidationError):
        SmartSearchNativeGRPOTraining(controls=controls,
            cost_limits=SmartSearchGRPOCostLimits(10, 81920, 10, 20480, 1, 8))
    with pytest.raises(ValidationError):
        SmartSearchNativeGRPOTraining(controls=controls, cost_limits=limits, max_helper_rows=0)


def test_native_validation_journal_parent_exists_before_paid_work(tmp_path):
    from agentinputlab.recipes.smartsearch_grpo_training import _prepare_native_output_directories
    root = tmp_path / "fresh-fit"
    root.mkdir()
    native = _prepare_native_output_directories(root)
    assert native == root / "native" and (native / "rollouts").is_dir()
    # Same filesystem operation that the pinned reward manager performs,
    # without constructing any worker/model or generating a response.
    journal = native / "rollouts/val_0.jsonl"
    with journal.open("a") as stream:
        stream.write('{"fixture_only": true}\n')
    assert journal.read_text() == '{"fixture_only": true}\n'
    with pytest.raises(FileExistsError):
        _prepare_native_output_directories(root)
    assert journal.read_text() == '{"fixture_only": true}\n'


def test_native_output_preflight_rejects_existing_links_and_non_owned_roots(tmp_path):
    from agentinputlab.recipes.smartsearch_grpo_training import _prepare_native_output_directories
    root, other = tmp_path / "fresh-fit", tmp_path / "outside"
    root.mkdir()
    other.mkdir()
    (root / "native").symlink_to(other, target_is_directory=True)
    with pytest.raises(FileExistsError):
        _prepare_native_output_directories(root)
    assert list(other.iterdir()) == []
    link = tmp_path / "fit-link"
    link.symlink_to(other, target_is_directory=True)
    for invalid in (link, tmp_path / "absent"):
        with pytest.raises(ValidationError, match="own regular fit root"):
            _prepare_native_output_directories(invalid)
    assert list(other.iterdir()) == []


FIXTURE_RUNNER = """
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        if config.strategy == 'fsdp':
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        else:
            raise NotImplementedError
        actors = {0: ray.remote(ActorRolloutRefWorker), 1: ray.remote(CriticWorker)}
        extra = ray.remote(ActorRolloutRefWorker)
        reward = ray.remote(RewardModelWorker)
        resource_pool_manager = ResourcePoolManager()
        trainer = RayPPOTrainer(reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        trainer.fit()
"""


def test_only_execution_seams_are_adapted_in_explicit_ast_fixture():
    result = _task_runner_ast(FIXTURE_RUNNER)
    text = ast.unparse(result)
    assert not result.decorator_list and text.count("max_task_retries=0") == 4
    assert "ActorRolloutRefWorker = _ail_actor_class" in text
    assert "reward_fn = _ail_reward_contract(reward_fn)" in text
    assert isinstance(result.body[0].body[-1], ast.Return)
    with pytest.raises(ExternalDependencyError, match="constructor inventory"):
        _task_runner_ast(FIXTURE_RUNNER.replace("extra = ray.remote(ActorRolloutRefWorker)", "extra = None"))
    with pytest.raises(ExternalDependencyError, match="terminal call"):
        _task_runner_ast(FIXTURE_RUNNER.replace("trainer.fit()", "trainer.other()"))
    assert _nonfinite({"loss": [0., float("nan")], "gradient": float("inf")}) == 2


def test_child_module_binding_and_private_capability_log_redaction():
    import hashlib

    from agentinputlab.recipes import smartsearch_grpo_training as module
    names = module._RUNTIME_FILES
    descriptor = {"toolkit_modules": {name: hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in names}}
    _verify_runtime(descriptor)
    descriptor["toolkit_modules"]["smartsearch_grpo_worker.py"] = "0" * 64
    with pytest.raises(ExternalDependencyError, match="bytes differ"):
        _verify_runtime(descriptor)
    config = {"data": {"search_url": "fixture private URL", "batch": 4}, "rollout": {"sandbox_url": "fixture secret"}}
    assert _log_safe(config) == {"data": {"search_url": "<private loopback capability>", "batch": 4},
                                "rollout": {"sandbox_url": "<private loopback capability>"}}
    assert config["data"]["search_url"] == "fixture private URL"


def test_actual_pinned_native_runner_ast_when_source_is_mounted():
    source = os.environ.get("AGENTINPUTLAB_SMARTSEARCH_TRAINING_CHECKOUT")
    if not source:
        pytest.skip("mount pinned SmartSearch source for native driver AST fixture")
    from agentinputlab.recipes.smartsearch_grpo_data import verify_smartsearch_grpo_source
    root, _ = verify_smartsearch_grpo_source(source)
    path = Path(root) / "src/verl/trainer/main_ppo.py"
    original = path.read_bytes()
    owner = _task_runner_ast(original)
    compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec")
    assert path.read_bytes() == original
    text = ast.unparse(owner)
    assert text.count("max_task_retries=0") == 4 and "return trainer.fit()" in text
