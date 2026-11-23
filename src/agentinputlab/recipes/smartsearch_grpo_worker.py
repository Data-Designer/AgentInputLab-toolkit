"""Fresh native Ray worker bindings; native losses and rollout math stay upstream."""
from __future__ import annotations

import ast
import inspect
import math
import numbers
import random
import re
from pathlib import Path
from types import FunctionType

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .qrag import _sha
from .smartsearch_grpo_contracts import NativeGRPORewardManagerContract
from .smartsearch_grpo_data import verify_smartsearch_grpo_source
from .smartsearch_grpo_meter import MeteredNativeGRPOEngine, observe_native_grpo_optimizer
from .smartsearch_grpo_prefix import SmartSearchNativeGRPOPrefixBuilder
from .smartsearch_grpo_rollout import install_native_rollout_batch_contracts
from .smartsearch_grpo_tools import SmartSearchNativeGRPOReward, _new_output, _save
from .smartsearch_grpo_transport import SmartSearchGRPOCoordinatorClient
from .smartsearch_teacher import SmartSearchTeacherControls

_NATIVE_COLOCATION_SHA256 = "b3a4237fc9b37d1011a259210196d259efbb1c68a50d25fe2616f24ee98075d1"


def _verify_runtime(descriptor):
    from .smartsearch_grpo_training import _RUNTIME_FILES
    if set(descriptor.get("toolkit_modules", {})) != set(_RUNTIME_FILES) or any(
        _sha(Path(__file__).with_name(name)) != descriptor["toolkit_modules"][name] for name in _RUNTIME_FILES
    ):
        raise ExternalDependencyError("SmartSearch GRPO child toolkit bytes differ from its launcher")


def _teacher_controls(descriptor):
    return SmartSearchTeacherControls(max_model_calls=descriptor["cost_limits"]["helper_requests"],
        max_total_output_tokens=descriptor["cost_limits"]["helper_output_tokens"],
        max_rollout_rows=descriptor["max_helper_rows"], seed=descriptor["seed"])


def _nonfinite(value):
    if isinstance(value, dict):
        return sum(_nonfinite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_nonfinite(v) for v in value)
    return int(isinstance(value, numbers.Real) and not math.isfinite(value))


def native_grpo_actor_class(native_class, descriptor):
    """Subclass the pinned worker, adding transport/accounting/export only."""
    _verify_runtime(descriptor)
    source, _ = verify_smartsearch_grpo_source(descriptor["checkout"])
    if Path(inspect.getfile(native_class)).resolve() != source / "src/verl/workers/fsdp_workers.py":
        raise ExternalDependencyError("SmartSearch GRPO actor class is not the bound native worker")
    import torch
    from verl.single_controller.base.decorator import Dispatch, register

    class ObservedSmartSearchActor(native_class):
        _ail_native_worker_class = native_class

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import numpy as np
            random.seed(descriptor["seed"])
            np.random.seed(descriptor["seed"])
            torch.manual_seed(descriptor["seed"])
            torch.cuda.manual_seed_all(descriptor["seed"])
            self._ail_descriptor = descriptor
            self._ail_root = _new_output(Path(descriptor["output_root"]) / "workers" / f"{self.role}-rank-{self.rank}")
            self._ail_client = SmartSearchGRPOCoordinatorClient(descriptor["coordinator_endpoint"])
            self._ail_prefix = None
            self._ail_usage = dict(forward_attempts=0, forward_responses=0,
                                   input_token_positions=0, nonfinite_native_metrics=0)
            _save(self._ail_root, "claim.json", dict(rank=self.rank, role=self.role,
                descriptor_sha256=content_hash(descriptor), source_worker_sha256=_sha(source / "src/verl/workers/fsdp_workers.py"),
                forward_scope="FSDP outer forward boundary after native init_model",
                initialization_and_internal_checkpoint_recomputation_forwards_unmeasured=True,
                no_automatic_retry=True))

        def _build_rollout(self, *args, **kwargs):
            rollout, sharding = super()._build_rollout(*args, **kwargs)
            self._ail_prefix = SmartSearchNativeGRPOPrefixBuilder(self._ail_client.generate,
                checkout=source, references=descriptor["train_references"],
                model_fingerprint_sha256=descriptor["helper_fingerprint_sha256"],
                controls=_teacher_controls(descriptor), output_root=self._ail_root / "prefix")
            binding = install_native_rollout_batch_contracts(checkout=source, rollout_class=type(rollout),
                prefix_builder=self._ail_prefix,
                prefix_builder_binding=dict(descriptor_sha256=content_hash(descriptor), shared_coordinator=True))
            _save(self._ail_root, "rollout-adapter.json", binding)
            rollout.inference_engine = MeteredNativeGRPOEngine(rollout.inference_engine,
                ledger=self._ail_client, worker=f"{self.role}-rank-{self.rank}")
            return rollout, sharding

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            modules = []
            if self._is_actor:
                modules.append(self.actor_module_fsdp)
                observe_native_grpo_optimizer(self.actor_optimizer, ledger=self._ail_client, rank=self.rank)
                original_update = self.actor.update_policy

                def update_policy(*args, **kwargs):
                    result = original_update(*args, **kwargs)
                    bad = _nonfinite(result)
                    self._ail_usage["nonfinite_native_metrics"] += bad
                    if bad:
                        self._ail_client.fail(f"nonfinite-actor-rank-{self.rank}", error_type="NonfiniteNativeMetrics")
                        raise ExternalDependencyError("SmartSearch native actor returned nonfinite metrics")
                    return result

                self.actor.update_policy = update_policy
            if self._is_ref:
                modules.append(self.ref_module_fsdp)
            for module in modules:
                def before(_module, args, kwargs):
                    self._ail_usage["forward_attempts"] += 1
                    ids = kwargs.get("input_ids")
                    if ids is None and args:
                        ids = args[0]
                    if hasattr(ids, "numel"):
                        self._ail_usage["input_token_positions"] += int(ids.numel())
                def after(*_args):
                    self._ail_usage["forward_responses"] += 1
                module.register_forward_pre_hook(before, with_kwargs=True)
                module.register_forward_hook(after)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def finalize_smartsearch_costs(self):
            if self._ail_prefix is not None:
                self._ail_prefix.assert_healthy()
            report = dict(rank=self.rank, role=self.role, usage=dict(self._ail_usage))
            _save(self._ail_root, "terminal.json", report)
            self._ail_client.worker_completed(**report)
            return report

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def export_smartsearch_safe_policy(self):
            # The pinned native 'hf_model' checkpoint branch writes config and
            # tokenizer ONLY. Export the ACTUAL trained state while workers
            # are alive, rather than calling that metadata a trained HF model.
            if not self._is_actor:
                raise ValidationError("Only the native trained actor can export the learned policy")
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            destination = Path(descriptor["output_root"]) / "exported-policy"
            if self.rank == 0:
                destination.mkdir(exist_ok=False)
            torch.distributed.barrier()
            with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT,
                                     FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state = self.actor_module_fsdp.state_dict()
            if self.rank == 0:
                original = self.actor_module_fsdp._fsdp_wrapped_module
                original.save_pretrained(destination, state_dict=state, safe_serialization=True, max_shard_size="4GB")
                self.tokenizer.save_pretrained(destination)
                files = {p.name: _sha(p) for p in destination.iterdir() if p.is_file()}
                if not any(n.endswith(".safetensors") for n in files) or any(
                    p.is_symlink() for p in destination.iterdir()
                ):
                    raise ExternalDependencyError("SmartSearch native trained state was not safely exported")
                _save(Path(descriptor["output_root"]), "safe-export.json",
                    dict(files=files, rank=0, same_live_trained_actor=True, new_optimizer_steps=0))
            torch.distributed.barrier()
            return {"rank": self.rank, "safe_export_completed": True}

    return ObservedSmartSearchActor


def _colocation_worker_base(actor_class):
    """Keep the native Worker root despite our accounting subclass layer."""
    if not isinstance(actor_class, type):
        raise ExternalDependencyError("SmartSearch colocated actor must be a class")
    native = getattr(actor_class, "_ail_native_worker_class", None)
    if native is None:
        return actor_class.__base__
    if not isinstance(native, type) or not issubclass(actor_class, native) or (
        len(native.__bases__) != 1 or native.__base__ is object
    ):
        raise ExternalDependencyError("SmartSearch colocated actor native-base binding differs")
    return native.__base__


def _colocation_factory_ast(source_text):
    """Change the two base lookups, not delegated role constructors or methods."""
    tree = ast.parse(source_text)
    owners = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "create_colocated_worker_cls"]
    if len(owners) != 1:
        raise ExternalDependencyError("SmartSearch native colocation factory differs")
    expected = ast.dump(ast.parse("cls.cls.__ray_actor_class__.__base__", mode="eval").body)

    class BindBase(ast.NodeTransformer):
        changed = 0

        def visit_Attribute(self, node):
            if ast.dump(node) == expected:
                self.changed += 1
                return ast.copy_location(ast.Call(func=ast.Name(id="_ail_colocation_worker_base", ctx=ast.Load()),
                    args=[node.value], keywords=[]), node)
            return self.generic_visit(node)

    binding = BindBase()
    owner = binding.visit(owners[0])
    if binding.changed != 2:
        raise ExternalDependencyError("SmartSearch native colocation base inventory differs")
    calls = [node for node in ast.walk(owner) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "ray" and node.func.attr == "remote"]
    if len(calls) != 1 or len(calls[0].args) != 1 or calls[0].keywords:
        raise ExternalDependencyError("SmartSearch native colocated actor constructor differs")
    calls[0].func = ast.parse("ray.remote(max_restarts=0, max_task_retries=0)", mode="eval").body
    return ast.fix_missing_locations(owner)


def _clone_init_workers_with_factory(method, factory):
    if not isinstance(method, FunctionType) or not callable(factory):
        raise ExternalDependencyError("SmartSearch native worker initialization surface differs")
    namespace = dict(method.__globals__, create_colocated_worker_cls=factory)
    clone = FunctionType(method.__code__, namespace, "_ail_native_init_workers", method.__defaults__, method.__closure__)
    clone.__kwdefaults__ = None if method.__kwdefaults__ is None else dict(method.__kwdefaults__)
    clone.__annotations__ = dict(method.__annotations__)
    return clone


def _native_colocated_init_workers(native_trainer, source):
    method = native_trainer.init_workers
    if not isinstance(method, FunctionType) or Path(inspect.getfile(method)).resolve() != (
        source / "src/verl/trainer/ppo/ray_trainer.py"
    ):
        raise ExternalDependencyError("SmartSearch worker initialization came from another trainer")
    factory = method.__globals__.get("create_colocated_worker_cls")
    path = source / "src/verl/single_controller/ray/base.py"
    if not isinstance(factory, FunctionType) or Path(inspect.getfile(factory)).resolve() != path or (
        _sha(path) != _NATIVE_COLOCATION_SHA256
    ):
        raise ExternalDependencyError("SmartSearch native colocation source changed")
    namespace = dict(factory.__globals__, _ail_colocation_worker_base=_colocation_worker_base)
    owner = _colocation_factory_ast(path.read_text())
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec"), namespace)
    return _clone_init_workers_with_factory(method, namespace["create_colocated_worker_cls"])


def native_grpo_reward_router(descriptor):
    """One native train scorer and one native dev scorer, no reference mixing."""
    _verify_runtime(descriptor)
    client = SmartSearchGRPOCoordinatorClient(descriptor["coordinator_endpoint"])
    controls = _teacher_controls(descriptor)
    scorers = {}
    for role in ("train", "validation"):
        scorers[role] = SmartSearchNativeGRPOReward(client.generate, checkout=descriptor["checkout"],
            references=descriptor[role + "_references"], model_fingerprint_sha256=descriptor["helper_fingerprint_sha256"],
            controls=controls, output_root=Path(descriptor["output_root"]) / ("reward-" + role), reference_split=role)

    def score(*, data_source, tokenizer, solution_str, ground_truth):
        if data_source != "re_call":
            raise ValidationError("SmartSearch native reward received a different data source")
        match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        question = match.group(1).strip() if match else None
        roles = [role for role, scorer in scorers.items() if question in scorer.references]
        if len(roles) != 1:
            raise ValidationError("SmartSearch native reward question is absent or crosses split roles")
        return scorers[roles[0]](tokenizer, solution_str, ground_truth)

    def healthy():
        for scorer in scorers.values():
            usage = scorer.usage
            if scorer.failed or usage["score_attempts"] != usage["score_responses"] or (
                usage["model_attempts"] != usage["model_responses"] or usage["unknown_output_token_reservations"]
            ):
                raise ExternalDependencyError("SmartSearch native reward has an unresolved scoring/helper request")
        return {role: dict(scorer.usage) for role, scorer in scorers.items()}

    score.assert_healthy = healthy
    return score


def _task_runner_ast(source_text):
    """Adapt only worker wiring, once-only reward envelope and return value."""
    tree = ast.parse(source_text)
    owners = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaskRunner"]
    if len(owners) != 1:
        raise ExternalDependencyError("SmartSearch native TaskRunner class differs")
    owner = owners[0]
    owner.decorator_list = []
    run = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    changed = 0
    for branch in ast.walk(run):
        if isinstance(branch, ast.If):
            for sequence in (branch.body, branch.orelse):
                for index, node in list(enumerate(sequence)):
                    if isinstance(node, ast.ImportFrom) and node.module == "verl.workers.fsdp_workers" and (
                        [n.name for n in node.names] == ["ActorRolloutRefWorker", "CriticWorker"]
                    ):
                        sequence[index + 1:index + 1] = ast.parse(
                            "ActorRolloutRefWorker = _ail_actor_class(ActorRolloutRefWorker, _ail_descriptor)").body
                        changed += 1
    if changed != 1:
        raise ExternalDependencyError("SmartSearch native FSDP worker binding differs")
    assignments = [(i, n) for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "resource_pool_manager" for t in n.targets)]
    if len(assignments) != 1:
        raise ExternalDependencyError("SmartSearch native reward-to-trainer boundary differs")
    run.body[assignments[0][0]:assignments[0][0]] = ast.parse(
        "reward_fn = _ail_reward_contract(reward_fn)\nval_reward_fn = _ail_reward_contract(val_reward_fn)").body
    last = run.body[-1]
    if not isinstance(last, ast.Expr) or ast.unparse(last.value) != "trainer.fit()":
        raise ExternalDependencyError("SmartSearch native trainer terminal call differs")
    run.body[-1] = ast.Return(value=last.value)
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "ray" and n.func.attr == "remote"]
    if len(calls) != 4 or any(len(n.args) != 1 or n.keywords for n in calls):
        raise ExternalDependencyError("SmartSearch native Ray worker constructor inventory differs")
    for call in calls:
        call.func = ast.parse("ray.remote(max_restarts=0, max_task_retries=0)", mode="eval").body
    for call in ast.walk(run):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "pprint":
            call.args = [ast.Call(func=ast.Name(id="_ail_log_safe", ctx=ast.Load()), args=[arg], keywords=[])
                         for arg in call.args]
    return ast.fix_missing_locations(owner)


def native_grpo_task_runner(native_driver, descriptor):
    """Build the pinned native TaskRunner with explicit local runtime seams."""
    _verify_runtime(descriptor)
    source, _ = verify_smartsearch_grpo_source(descriptor["checkout"])
    path = source / "src/verl/trainer/main_ppo.py"
    if Path(native_driver.__file__).resolve() != path:
        raise ExternalDependencyError("SmartSearch GRPO driver was imported from a different installation")
    NativeTrainer = native_driver.RayPPOTrainer
    init_workers = _native_colocated_init_workers(NativeTrainer, source)

    class ObservedTrainer(NativeTrainer):
        def init_workers(self):
            return init_workers(self)

        def fit(self):
            super().fit()
            router = self.reward_fn.native_manager.compute_score
            reward_usage = router.assert_healthy()
            actor = self.actor_rollout_wg.finalize_smartsearch_costs()
            reference = self.ref_policy_wg.finalize_smartsearch_costs() if self.use_reference_policy else []
            exported = self.actor_rollout_wg.export_smartsearch_safe_policy()
            return dict(global_steps=self.global_steps, actor=actor, reference=reference,
                        reward_usage=reward_usage, safe_export=exported)

    namespace = dict(native_driver.__dict__, RayPPOTrainer=ObservedTrainer,
        get_custom_reward_fn=lambda _config: native_grpo_reward_router(descriptor),
        _ail_descriptor=descriptor, _ail_actor_class=native_grpo_actor_class,
        _ail_reward_contract=NativeGRPORewardManagerContract, _ail_log_safe=_log_safe)
    owner = _task_runner_ast(path.read_text())
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TaskRunner"]


def _log_safe(value):
    if isinstance(value, dict):
        return {key: "<private loopback capability>" if key in {"sandbox_url", "search_url", "coordinator_endpoint"}
                else _log_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_log_safe(item) for item in value]
    return value
