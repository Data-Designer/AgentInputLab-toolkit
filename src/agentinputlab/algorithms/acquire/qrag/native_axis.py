"""Opt-in shape compatibility for the hash-pinned mounted Q-RAG policy.

No upstream source or weights are copied into the distribution. The adapter
preserves the batch/candidate axes after the existing dot-product reduction.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.contracts import ExternalDependencyError

QRAG_SINGLE_CANDIDATE_AXIS_COMPAT = "qrag.native_policy.candidate_axis_compat.v1"
QRAG_Q_MODULE_SHA256 = "8f5e50f1033aa04a326bba2ba14eff2ff8631d494f01913179652644e7832148"


def _axis_stable_forward(source: str, namespace: Mapping[str, Any], filename: str):
    """Compile the mounted method with exactly one verified AST replacement."""
    tree = ast.parse(source)
    expected = ast.parse("(s_embed * a_embeds).sum(-1).squeeze(-1)", mode="eval").body
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "forward":
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "logits"
                and ast.dump(statement.value) == ast.dump(expected)
            ):
                matches.append((node, statement))
    if len(matches) != 1:
        raise ExternalDependencyError("Q-RAG native policy axis expression does not match")
    original, assignment = matches[0]
    node = copy.deepcopy(original)
    target = next(n for n in node.body if isinstance(n, ast.Assign)
                  and ast.dump(n) == ast.dump(assignment))
    target.value = target.value.func.value  # Remove only the final squeeze(-1).
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(module, filename, "exec"), scope)
    return scope["forward"]


def install_single_candidate_axis_compat(policy: Any, source_root: Path) -> None:
    """Repair tensor rank, without modifying mounted files or model parameters."""
    path = source_root / "rl/q_module.py"
    if path.is_symlink() or not path.is_file():
        raise ExternalDependencyError("Q-RAG native policy source must be a regular mounted file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != QRAG_Q_MODULE_SHA256:
        raise ExternalDependencyError("Q-RAG native policy source checksum mismatch")
    previous = getattr(policy, "_agentinputlab_axis_compat", None)
    if previous is not None:
        if previous != QRAG_SINGLE_CANDIDATE_AXIS_COMPAT:
            raise ExternalDependencyError("Q-RAG native policy already has a different adapter")
        return
    original = inspect.unwrap(policy.forward)
    if Path(original.__code__.co_filename).resolve() != path.resolve():
        raise ExternalDependencyError("Q-RAG policy method is not from the pinned mounted source")
    fixed = _axis_stable_forward(raw.decode("utf-8"), original.__globals__, str(path))
    policy.forward = types.MethodType(fixed, policy)
    policy._agentinputlab_axis_compat = QRAG_SINGLE_CANDIDATE_AXIS_COMPAT
