"""Prepare existing controlled HyDE bindings without generating or embedding."""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

from agentinputlab.algorithms.acquire.hyde import HyDEConfig
from agentinputlab.algorithms.acquire.hyde_controlled import HyDEControlledBindings
from agentinputlab.configuration import _known, _object
from agentinputlab.contracts import InterventionContext, ValidationError
from agentinputlab.contracts.serialization import to_jsonable


def prepare(config, output_dir):
    if config.release_id != "acquire.hyde.controlled":
        raise ValidationError("prepare-hyde is for acquire.hyde.controlled")
    if not config.backend or not config.vector_runtime:
        raise ValidationError("controlled HyDE needs explicit backend and vector_runtime")
    _known(config.parameters, {"config", "profile_kind"}, "preparation parameters")
    settings = _object(config.parameters.get("config", {}), "HyDE settings")
    _known(settings, {f.name for f in fields(HyDEConfig)}, "HyDE settings")
    settings = HyDEConfig(**settings)
    profile_kind = config.parameters.get("profile_kind", "controlled_adapter")
    if profile_kind not in {"controlled_adapter", "factorial_adapter"}:
        raise ValidationError("controlled HyDE profile_kind must be controlled_adapter or factorial_adapter")
    output = Path(output_dir).absolute()
    if output.exists():
        raise ValidationError("HyDE preparation requires a new output directory")
    backend, vector = config._backend(), config._vector_runtime()
    context = InterventionContext(backend=backend, dense_embedder=vector, vector_searcher=vector,
                                  seed=config.seed)
    binding = HyDEControlledBindings.from_context(context, budget=config.budget, config=settings)
    ready = replace(config, parameters={"bindings": to_jsonable(binding), "profile_kind": profile_kind})
    ready.validate()
    output.mkdir(parents=True)
    with (output / "run.json").open("x") as stream:
        json.dump(ready.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"release_id": config.release_id, "binding_sha256": binding.binding_hash,
             "run_path": str(output / "run.json"),
             "run_sha256": hashlib.sha256((output / "run.json").read_bytes()).hexdigest(),
             "model_calls": 0, "embedding_calls": 0, "retrieval_calls": 0, "fit_performed": False}
    with (output / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof
