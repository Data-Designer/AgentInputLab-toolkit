"""Bounded object-graph inspection for executable runtime dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any

from agentinputlab.backends.llm import LLMBackend
from agentinputlab.backends.retrieval import Retriever
from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher
from agentinputlab.connectors.tools import ToolConnector
from agentinputlab.contracts.errors import ValidationError

RUNTIME_DEPENDENCY_TYPES = (
    LLMBackend,
    DenseEmbedder,
    Retriever,
    VectorSearcher,
    ToolConnector,
)

_EXTERNAL_EXECUTOR_METHODS = ("call", "invoke", "reformulate", "run")


@dataclass(frozen=True, slots=True)
class ObjectGraphNode:
    path: str
    value: object


def inspect_object_graph(
    root: object,
    *,
    root_path: str,
    terminal_types: tuple[type[Any], ...] = (),
    traverse_runtime_delegates: bool = False,
    max_depth: int = 8,
    max_nodes: int = 4096,
) -> tuple[ObjectGraphNode, ...]:
    """Return a cycle-safe snapshot of declared object state and callable captures.

    Mapping keys, bound-method owners, function defaults/closures/referenced
    globals, partial arguments, dataclass fields, ``__dict__``, and ``__slots__``
    are included.  The walk never executes user callables.  Hitting a bound is
    an error rather than a silent partial audit.
    """

    nodes: list[ObjectGraphNode] = []
    visited: set[int] = set()

    def walk_function_captures(
        function: FunctionType,
        path: str,
        *,
        depth: int,
        include_globals: bool,
    ) -> None:
        walk(function.__defaults__, f"{path}.__defaults__", depth=depth + 1)
        walk(function.__kwdefaults__, f"{path}.__kwdefaults__", depth=depth + 1)
        for index, cell in enumerate(function.__closure__ or ()):
            try:
                captured = cell.cell_contents
            except ValueError:  # empty closure cell
                continue
            walk(captured, f"{path}.__closure__[{index}]", depth=depth + 1)
        if include_globals:
            for name in function.__code__.co_names:
                if name in function.__globals__:
                    walk(
                        function.__globals__[name],
                        f"{path}.__globals__[{name}]",
                        depth=depth + 1,
                    )

    def walk(value: object, path: str, *, depth: int) -> None:
        if value is None or isinstance(value, (str, bytes, int, float, bool, type, ModuleType)):
            return
        if depth > max_depth:
            raise ValidationError(
                f"runtime-dependency inspection exceeded depth {max_depth} at {path}"
            )

        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        nodes.append(ObjectGraphNode(path, value))
        if len(nodes) > max_nodes:
            raise ValidationError(f"runtime-dependency inspection exceeded {max_nodes} objects")
        if terminal_types and isinstance(value, terminal_types):
            return
        # An executable dependency is normally an auditable boundary: callers
        # bind its public fingerprint and execution instance, while its private
        # model graph (for example a SentenceTransformer/PyTorch module tree)
        # is not part of hidden-dependency discovery.  Balance checks may opt
        # into following only direct runtime delegates and ledgers so a proxy
        # cannot conceal a shared executable object without opening the full
        # model graph.
        if is_runtime_dependency(value):
            if traverse_runtime_delegates:
                for name, item in _instance_state_items(value):
                    if is_runtime_dependency(item) or (
                        terminal_types and isinstance(item, terminal_types)
                    ):
                        walk(item, f"{path}.{name}", depth=depth + 1)
            return

        if isinstance(value, Mapping):
            for index, (key, item) in enumerate(value.items()):
                walk(key, f"{path}.key[{index}]", depth=depth + 1)
                walk(item, f"{path}.value[{index}]", depth=depth + 1)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth=depth + 1)
            return
        if isinstance(value, partial):
            walk(value.func, f"{path}.func", depth=depth + 1)
            walk(value.args, f"{path}.args", depth=depth + 1)
            walk(value.keywords, f"{path}.keywords", depth=depth + 1)
            return
        if isinstance(value, MethodType):
            walk(value.__self__, f"{path}.__self__", depth=depth + 1)
            walk(value.__func__, f"{path}.__func__", depth=depth + 1)
            return
        if isinstance(value, FunctionType):
            walk_function_captures(value, path, depth=depth, include_globals=True)
            return
        if isinstance(value, BuiltinFunctionType):
            owner = getattr(value, "__self__", None)
            if owner is not None:
                walk(owner, f"{path}.__self__", depth=depth + 1)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for declared in fields(value):
                walk(
                    getattr(value, declared.name),
                    f"{path}.{declared.name}",
                    depth=depth + 1,
                )
        else:
            attributes = getattr(value, "__dict__", {})
            if isinstance(attributes, Mapping):
                for name, item in attributes.items():
                    walk(item, f"{path}.{name}", depth=depth + 1)
        for owner in type(value).__mro__:
            declared_slots = owner.__dict__.get("__slots__", ())
            if isinstance(declared_slots, str):
                declared_slots = (declared_slots,)
            for name in declared_slots:
                if name in {"__dict__", "__weakref__"} or not hasattr(value, name):
                    continue
                walk(getattr(value, name), f"{path}.{name}", depth=depth + 1)
        ignored_class_names = {
            "__annotations__",
            "__dict__",
            "__doc__",
            "__module__",
            "__slots__",
            "__weakref__",
        }
        for owner in type(value).__mro__:
            for name, declared in owner.__dict__.items():
                if name in ignored_class_names or (name.startswith("__") and name.endswith("__")):
                    continue
                function: FunctionType | None = None
                if isinstance(declared, (classmethod, staticmethod)):
                    unwrapped = declared.__func__
                    if isinstance(unwrapped, FunctionType):
                        function = unwrapped
                elif isinstance(declared, FunctionType):
                    function = declared
                if function is not None:
                    walk_function_captures(
                        function,
                        f"{path}.__class__.{name}",
                        depth=depth + 1,
                        include_globals=False,
                    )
                    continue
                if isinstance(declared, property):
                    for accessor_name, accessor in (
                        ("fget", declared.fget),
                        ("fset", declared.fset),
                        ("fdel", declared.fdel),
                    ):
                        if isinstance(accessor, FunctionType):
                            walk_function_captures(
                                accessor,
                                f"{path}.__class__.{name}.{accessor_name}",
                                depth=depth + 1,
                                include_globals=False,
                            )
                    continue
                if hasattr(declared, "__get__"):
                    continue
                walk(declared, f"{path}.__class__.{name}", depth=depth + 1)

    walk(root, root_path, depth=0)
    return tuple(nodes)


def _instance_state_items(value: object) -> tuple[tuple[str, object], ...]:
    """Return direct instance state once, without executing descriptors."""

    items: dict[str, object] = {}
    if is_dataclass(value) and not isinstance(value, type):
        for declared in fields(value):
            items[declared.name] = getattr(value, declared.name)
    attributes = getattr(value, "__dict__", {})
    if isinstance(attributes, Mapping):
        items.update((str(name), item) for name, item in attributes.items())
    for owner in type(value).__mro__:
        declared_slots = owner.__dict__.get("__slots__", ())
        if isinstance(declared_slots, str):
            declared_slots = (declared_slots,)
        for name in declared_slots:
            if name in {"__dict__", "__weakref__"} or name in items:
                continue
            if hasattr(value, name):
                items[name] = getattr(value, name)
    return tuple(items.items())


def runtime_dependency_nodes(
    root: object,
    *,
    root_path: str,
    terminal_types: tuple[type[Any], ...] = (),
) -> tuple[ObjectGraphNode, ...]:
    return tuple(
        node
        for node in inspect_object_graph(
            root,
            root_path=root_path,
            terminal_types=terminal_types,
        )
        if is_runtime_dependency(node.value)
    )


def is_runtime_dependency(value: object) -> bool:
    """Return whether ``value`` can perform an in-scope external execution.

    The core backend protocols are nominally identifiable.  Pinned upstream
    bridges intentionally use small structural seams instead (``call``,
    ``run``, or ``reformulate`` plus a stable fingerprint), so they must also
    be treated as executable dependencies regardless of their field name.
    """

    if isinstance(value, RUNTIME_DEPENDENCY_TYPES):
        return True
    fingerprint = getattr(value, "fingerprint", None)
    return callable(fingerprint) and any(
        callable(getattr(value, method, None)) for method in _EXTERNAL_EXECUTOR_METHODS
    )


__all__ = [
    "RUNTIME_DEPENDENCY_TYPES",
    "ObjectGraphNode",
    "inspect_object_graph",
    "is_runtime_dependency",
    "runtime_dependency_nodes",
]
