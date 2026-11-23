"""No-model fixtures for the actual native colocated-worker inheritance seam."""
import ast
import os
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ExternalDependencyError
from agentinputlab.recipes.smartsearch_grpo_worker import (
    _clone_init_workers_with_factory,
    _colocation_factory_ast,
    _colocation_worker_base,
)

FACTORY = """
def create_colocated_worker_cls(class_dict):
    cls_dict = {}
    init_args_dict = {}
    worker_cls = None
    for key, cls in class_dict.items():
        if worker_cls is None:
            worker_cls = cls.cls.__ray_actor_class__.__base__
        else:
            assert worker_cls == cls.cls.__ray_actor_class__.__base__
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}
    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                self.worker_dict[key] = user_defined_cls.__ray_actor_class__(
                    *init_args_dict[key]['args'], **init_args_dict[key]['kwargs'])
    return ray.remote(WorkerDict)
"""


class FixtureWorker:
    def __init__(self):
        self.root_initialized = True


class FixtureNativeActor(FixtureWorker):
    def __init__(self, config, role):
        super().__init__()
        self.config, self.role = config, role


class FixtureObservedActor(FixtureNativeActor):
    _ail_native_worker_class = FixtureNativeActor


def factory(adapted):
    class Ray:
        @staticmethod
        def remote(cls=None, **options):
            if cls is None:
                return lambda item: SimpleNamespace(__ray_actor_class__=item, options=options)
            return SimpleNamespace(__ray_actor_class__=cls, options=options)

    namespace = dict(ray=Ray(), _ail_colocation_worker_base=_colocation_worker_base)
    owner = _colocation_factory_ast(FACTORY) if adapted else ast.parse(FACTORY).body[0]
    exec(compile(ast.Module(body=[owner], type_ignores=[]), "explicit-colocation-fixture", "exec"), namespace)
    return namespace["create_colocated_worker_cls"]


def specifications(actor):
    return {role: SimpleNamespace(cls=SimpleNamespace(__ray_actor_class__=actor),
        args=(f"configuration-{role}",), kwargs=dict(role=role)) for role in ("actor_rollout", "ref")}


def test_regression_reproduces_missing_config_role_before_any_model():
    container = factory(False)(specifications(FixtureObservedActor))
    with pytest.raises(TypeError, match="config.*role"):
        container.__ray_actor_class__()


@pytest.mark.parametrize("actor", [FixtureObservedActor, FixtureNativeActor])
def test_accounting_subclass_uses_worker_root_and_preserves_role_constructor_arguments(actor):
    container = factory(True)(specifications(actor))
    assert container.__ray_actor_class__.__base__ is FixtureWorker
    assert container.options == dict(max_restarts=0, max_task_retries=0)
    instance = container.__ray_actor_class__()
    assert instance.root_initialized
    for role, worker in instance.worker_dict.items():
        assert type(worker) is actor and worker.config == f"configuration-{role}" and worker.role == role


def test_factory_guard_rejects_changed_native_base_inventory():
    with pytest.raises(ExternalDependencyError, match="base inventory"):
        _colocation_factory_ast(FACTORY.replace("assert worker_cls == cls.cls.__ray_actor_class__.__base__", "assert True"))
    with pytest.raises(ExternalDependencyError, match="constructor"):
        _colocation_factory_ast(FACTORY.replace("ray.remote(WorkerDict)", "ray.remote(WorkerDict, num_gpus=1)"))
    with pytest.raises(ExternalDependencyError, match="factory"):
        _colocation_factory_ast(FACTORY.replace("create_colocated_worker_cls", "other_factory"))


def test_invalid_native_markers_cannot_select_an_unrelated_worker_base():
    for marker in (str, object, "not-a-class"):
        invalid = type("InvalidFixtureActor", (FixtureNativeActor,), {"_ail_native_worker_class": marker})
        with pytest.raises(ExternalDependencyError, match="native-base"):
            _colocation_worker_base(invalid)
    with pytest.raises(ExternalDependencyError, match="must be a class"):
        _colocation_worker_base(object())


def test_native_init_code_defaults_and_closure_are_preserved_without_global_mutation():
    namespace = {"create_colocated_worker_cls": lambda value: ("original", value)}
    exec("def init_workers(self, value=3, *, offset=1):\n return create_colocated_worker_cls((self, value+offset))", namespace)
    original = namespace["init_workers"]
    bound = _clone_init_workers_with_factory(original, lambda value: ("bound", value))
    assert bound.__code__ is original.__code__ and bound.__defaults__ == original.__defaults__
    assert bound.__kwdefaults__ == original.__kwdefaults__
    assert bound("worker") == ("bound", ("worker", 4))
    assert original("worker") == ("original", ("worker", 4))
    with pytest.raises(ExternalDependencyError, match="initialization surface"):
        _clone_init_workers_with_factory(None, lambda: None)


def test_actual_pinned_native_factory_ast_is_read_only_when_source_mounted():
    source = os.environ.get("AGENTINPUTLAB_SMARTSEARCH_TRAINING_CHECKOUT")
    if not source:
        pytest.skip("mount pinned SmartSearch source for native factory fixture")
    from agentinputlab.recipes.qrag import _sha
    from agentinputlab.recipes.smartsearch_grpo_data import verify_smartsearch_grpo_source
    from agentinputlab.recipes.smartsearch_grpo_worker import _NATIVE_COLOCATION_SHA256
    root, _ = verify_smartsearch_grpo_source(source)
    path = root / "src/verl/single_controller/ray/base.py"
    raw = path.read_bytes()
    assert _sha(path) == _NATIVE_COLOCATION_SHA256
    owner = _colocation_factory_ast(raw)
    compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec")
    text = ast.unparse(owner)
    assert text.count("_ail_colocation_worker_base(") == 2
    assert "super().__init__()" in text and "max_task_retries=0" in text
    assert path.read_bytes() == raw
