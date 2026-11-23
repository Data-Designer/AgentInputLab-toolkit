"""Prepare the existing controlled QueryGym profile from public file bindings."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from agentinputlab.algorithms.reformulate.query2doc import (
    Query2DocLiveBindings,
    Query2DocLiveConfig,
    QueryGymMode,
)
from agentinputlab.configuration import _known, _object
from agentinputlab.contracts import InterventionContext
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable


def prepare(config, output_dir):
    """Read real identities, never generate, retrieve, fit or guess dependencies."""
    if config.release_id != "rewrite.query2doc.controlled_live":
        raise ValidationError("prepare-query2doc is for rewrite.query2doc.controlled_live")
    if config.runtime.get("kind") != "querygym_query2doc" or not config.backend or not config.retriever:
        raise ValidationError("controlled Query2doc needs explicit backend, retriever and querygym_query2doc runtime")
    _known(config.parameters, {"config", "profile_kind"}, "preparation parameters")
    settings = _object(config.parameters.get("config", {}), "Query2doc settings")
    _known(settings, {"mode", "demonstration_ids", "temperature", "max_tokens"}, "Query2doc settings")
    settings = Query2DocLiveConfig(**settings)
    if settings.mode is QueryGymMode.FEW_SHOT:
        raise ValidationError("pinned live QueryGym has no file-bound few-shot training-data adapter; use zs or cot")
    profile_kind = config.parameters.get("profile_kind", "controlled_adapter")
    if profile_kind not in {"controlled_adapter", "factorial_adapter"}:
        raise ValidationError("controlled Query2doc profile_kind must be controlled_adapter or factorial_adapter")
    output = Path(output_dir).absolute()
    if output.exists():
        raise ValidationError("Query2doc preparation requires a new output directory")
    backend, retriever = config._backend(), config._retriever()
    runtime = config._runtime(load=True, backend=backend)
    context = InterventionContext(backend=backend, retriever=retriever, external_runtime=runtime,
                                  seed=config.seed)
    binding = Query2DocLiveBindings.from_context(context, budget=config.budget, config=settings)
    ready = replace(config, parameters={"bindings": to_jsonable(binding), "profile_kind": profile_kind})
    ready.validate()
    output.mkdir(parents=True)
    with (output / "run.json").open("x") as stream:
        json.dump(ready.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"release_id": config.release_id, "binding_sha256": binding.binding_hash,
             "run_path": str(output / "run.json"),
             "run_sha256": hashlib.sha256((output / "run.json").read_bytes()).hexdigest(),
             "model_calls": 0, "retrieval_calls": 0, "fit_performed": False}
    with (output / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof
