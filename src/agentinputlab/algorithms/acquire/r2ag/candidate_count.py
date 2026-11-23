"""Explicit short-pool adaptation of the pinned frozen R²-Former.

Use existing positional weights for the real input prefix. No synthetic
document, padded feature row, resized weight, or new training is introduced.
The singleton's absent-neighbor feature is explicitly defined as zero.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import types
from collections.abc import Sequence
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import freeze_mapping

from .bridge import _cosine, _validate_vectors, compute_r2ag_retrieval_features

R2AG_SHORT_CANDIDATE_COMPAT = "r2ag.real_candidate_prefix.absent_neighbor_zero.v1"
R2AG_MODELING_SOURCE_SHA256 = "6b63388bfc97ad093aee9464c73098b1a198e7dd4e8b439404dc1be4309b4be9"


def short_pool_features(query: Sequence[float], documents: Sequence[Sequence[float]]):
    """Preserve all original multi-document formulas, including first precedent=1."""
    if len(documents) != 1:
        return compute_r2ag_retrieval_features(query, documents)
    vectors = _validate_vectors((query, documents[0]), expected_count=2)
    return (freeze_mapping({"rerank": _cosine(vectors[0], vectors[1]),
                            "neighbor": 0.0, "precedent": 1.0}),)


def _short_pool_forward(source: str, namespace: dict, filename: str):
    tree = ast.parse(source)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RFormer"), None)
    if cls is None:
        raise ExternalDependencyError("R²AG modeling source lacks RFormer")
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    if method is None:
        raise ExternalDependencyError("R²AG modeling source lacks RFormer.forward")
    method = copy.deepcopy(method)
    expected = ast.parse(
        "torch.arange(self.num_k, device=x.device).expand(x.shape[0], self.num_k)",
        mode="eval",
    ).body
    matches = [n for n in method.body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == "position_ids" and ast.dump(n.value) == ast.dump(expected)]
    if len(matches) != 1:
        raise ExternalDependencyError("R²AG positional-axis expression does not match pinned source")
    matches[0].value = ast.parse(
        "torch.arange(x.shape[1], device=x.device).expand(x.shape[0], x.shape[1])", mode="eval",
    ).body
    validation = ast.parse(
        "if not 1 <= x.shape[1] <= self.num_k:\n"
        "    raise ValueError('R2AG real candidate count exceeds frozen positional capacity')"
    ).body
    # Keep the original 2D->3D batch normalization, then check the real axis.
    if not method.body or not isinstance(method.body[0], ast.If):
        raise ExternalDependencyError("R²AG input-axis normalization does not match")
    method.body[1:1] = validation
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(module, filename, "exec"), scope)
    return scope["forward"]


def install_short_pool_compat(rformer, source_path: Path) -> None:
    """Install only on the exact mounted source; leave source and weights untouched."""
    if source_path.is_symlink() or not source_path.is_file():
        raise ExternalDependencyError("R²AG modeling source must be a regular mounted file")
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != R2AG_MODELING_SOURCE_SHA256:
        raise ExternalDependencyError("R²AG short-pool source checksum mismatch")
    previous = getattr(rformer, "_agentinputlab_candidate_compat", None)
    if previous is not None:
        if previous != R2AG_SHORT_CANDIDATE_COMPAT:
            raise ValidationError("R²AG already has a different candidate-count adapter")
        return
    original = inspect.unwrap(rformer.forward)
    if Path(original.__code__.co_filename).resolve() != source_path.resolve():
        raise ExternalDependencyError("R²AG forward is not from the pinned mounted source")
    adapted = _short_pool_forward(raw.decode("utf-8"), original.__globals__, str(source_path))
    rformer.forward = types.MethodType(adapted, rformer)
    rformer._agentinputlab_candidate_compat = R2AG_SHORT_CANDIDATE_COMPAT
