"""Batch-assembly repairs for the exact native SmartSearch GRPO rollout.

No native file is modified. The original group/selection/truncation logic and
single-trajectory generator remain intact; a fresh worker supplies its explicit
metered prefix builder instead of the upstream hardcoded network helper.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from collections.abc import Mapping
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable

from .qrag import _sha
from .smartsearch_grpo_data import verify_smartsearch_grpo_source

_PATH = "src/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
_CLASS = "vLLMRolloutWithTool"


def _name(node, expected):
    return isinstance(node, ast.Name) and node.id == expected


def _batch_contract_ast(method):
    """Move one pinned assembly block; refuse a structurally different source."""
    method = copy.deepcopy(method)
    loops = [n for n in method.body if isinstance(n, ast.While) and _name(n.test, "active_indices")]
    if len(loops) != 1:
        raise ExternalDependencyError("SmartSearch native active-rollout loop differs")
    loop = loops[0]
    outer = [n for n in loop.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
             and [_n.id for _n in n.target.elts if isinstance(_n, ast.Name)] == ["i", "idx"]]
    if len(outer) != 1:
        raise ExternalDependencyError("SmartSearch native per-input prefix loop differs")
    outer = outer[0]
    starts = [i for i, n in enumerate(outer.body) if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and _name(n.targets[0], "prefix_input_ids")
              and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
              and _name(n.value.func.value, "torch") and n.value.func.attr == "stack"]
    if len(starts) != 1:
        raise ExternalDependencyError("SmartSearch native prefix stack block differs")
    position = starts[0]
    tail = outer.body[position:]
    if len(tail) != 11 or not isinstance(tail[-1], ast.For):
        raise ExternalDependencyError("SmartSearch native assembly statement inventory differs")
    for statement, name in zip(tail[:5], ("prefix_input_ids", "prefix_loss_mask", "input_ids",
                                            "attention_mask", "position_ids"), strict=True):
        expected = ast.parse(f"{name} = torch.stack({name}_list, dim=0)").body[0]
        if ast.dump(statement) != ast.dump(expected):
            raise ExternalDependencyError("SmartSearch native prefix tensor assembly differs")
    if ast.dump(outer.iter) != ast.dump(ast.parse("enumerate(active_indices)", mode="eval").body):
        raise ExternalDependencyError("SmartSearch native prefix input selection differs")
    if ast.dump(tail[-1].iter) != ast.dump(ast.parse("enumerate(index)", mode="eval").body):
        raise ExternalDependencyError("SmartSearch native generated-prefix selection differs")
    selected = [n for n in tail if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute) and _name(n.targets[0].value, "new_prompts")
                and n.targets[0].attr == "non_tensor_batch"]
    if len(selected) != 1 or ast.dump(selected[0].value) != ast.dump(ast.parse("prompts.non_tensor_batch", mode="eval").body):
        raise ExternalDependencyError("SmartSearch native prefix metadata assignment differs")
    selected[0].value = ast.parse("prompts.select_idxs(index).non_tensor_batch", mode="eval").body
    outer.body = outer.body[:position]
    insertion = loop.body.index(outer) + 1
    loop.body.insert(insertion, ast.If(test=ast.Name(id="prefix_input_ids_list", ctx=ast.Load()), body=tail, orelse=[]))
    return ast.fix_missing_locations(method)


def _compile_batch_contract(method, namespace, prefix_builder, filename):
    repaired = _batch_contract_ast(method)
    local = dict(namespace)
    local["generate_new_prompts"] = prefix_builder
    exec(compile(ast.Module(body=[repaired], type_ignores=[]), filename, "exec"), local)
    return local[method.name], repaired


def install_native_rollout_batch_contracts(*, checkout, rollout_class, prefix_builder, prefix_builder_binding):
    """Install source-bound assembly fixes in one fresh native worker.

    Caller must provide a metered, fail-closed native prefix-builder seam and
    its provenance. There is intentionally no default hardcoded HTTP helper.
    The executor must meter ALL actual generation calls, including outputs
    subsequently discarded by unchanged native group truncation.
    """
    if not isinstance(rollout_class, type) or rollout_class.__name__ != _CLASS or not callable(prefix_builder):
        raise ValidationError("SmartSearch rollout repair requires the native class and explicit prefix provider")
    if not isinstance(prefix_builder_binding, Mapping) or not prefix_builder_binding:
        raise ValidationError("SmartSearch rollout prefix provider needs explicit provenance")
    binding = to_jsonable(prefix_builder_binding)
    if getattr(rollout_class, "_agentinputlab_batch_contract_installed", False):
        raise ExternalDependencyError("SmartSearch rollout contracts are already installed; use a fresh native worker")
    source, source_binding = verify_smartsearch_grpo_source(checkout)
    path = source / _PATH
    original_method = inspect.unwrap(rollout_class.generate_sequences)
    if not hasattr(original_method, "__code__") or Path(original_method.__code__.co_filename).resolve() != path:
        raise ExternalDependencyError("SmartSearch rollout method was not loaded from the pinned native source")
    tree = ast.parse(path.read_text())
    owners = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == _CLASS]
    methods = [n for owner in owners for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "generate_sequences"]
    if len(owners) != 1 or len(methods) != 1:
        raise ExternalDependencyError("SmartSearch native rollout class surface differs")
    method = methods[0]
    implementation, repaired = _compile_batch_contract(method, original_method.__globals__, prefix_builder, str(path))
    record = dict(native_source=source_binding, native_rollout_sha256=_sha(path),
        original_method_ast_sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),
        adapter_method_ast_sha256=hashlib.sha256(ast.dump(repaired).encode()).hexdigest(),
        adapter_toolkit_sha256=_sha(Path(__file__)), prefix_builder_binding=binding,
        native_source_files_modified=False, native_single_trajectory_generator_replaced=False,
        native_group_size_and_selection_unchanged=True, prefix_batch_generated_once=True,
        empty_prefix_batch_skipped=True, prefix_metadata_selected_by_original_indices=True,
        all_generated_and_discarded_outputs_require_metering=True,
        full_grpo_training_certified=False)
    rollout_class.generate_sequences = implementation
    rollout_class._agentinputlab_batch_contract_installed = True
    return record
