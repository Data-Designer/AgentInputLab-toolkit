"""File bindings for the existing source-pinned LongLLMLingua runtime."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields, replace
from pathlib import Path, PurePosixPath

from agentinputlab.algorithms.curate.llmlingua import LongLLMLinguaConfig
from agentinputlab.algorithms.curate.llmlingua.controlled import LongLLMLinguaControlledBindings
from agentinputlab.configuration import _known, _object
from agentinputlab.contracts import InterventionContext, ValidationError
from agentinputlab.contracts.serialization import to_jsonable


def runtime(config, *, load):
    if config.release_id != "curate.longllmlingua.controlled":
        raise ValidationError("longllmlingua_native requires curate.longllmlingua.controlled")
    value = dict(config.runtime)
    _known(value, {"kind", "snapshot", "model_id", "revision", "file_sha256",
                   "device_map", "torch_dtype"}, "LongLLMLingua runtime")
    if value.pop("kind", None) != "longllmlingua_native":
        raise ValidationError("runtime.kind must be longllmlingua_native")
    if any(not isinstance(value.get(key), str) or not value[key].strip()
           for key in ("snapshot", "model_id", "revision")):
        raise ValidationError("LongLLMLingua requires snapshot, model_id and revision strings")
    if re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is None:
        raise ValidationError("LongLLMLingua revision must be a pinned 40-hex commit")
    if (value.get("device_map", "cpu") not in {"cpu", "cuda:0", "mps"}
            or value.get("torch_dtype", "float16") not in {"float16", "bfloat16", "float32"}):
        raise ValidationError("LongLLMLingua requires an explicit supported device and floating dtype")
    pins = _object(value.get("file_sha256", {}), "LongLLMLingua file_sha256")
    for name, digest in pins.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValidationError("LongLLMLingua requires safe file names and SHA-256 values")
    if (not {"config.json", "tokenizer_config.json", "special_tokens_map.json"}.issubset(pins)
            or not any(name.endswith(".safetensors") for name in pins)
            or not {"tokenizer.json", "tokenizer.model"}.intersection(pins)):
        raise ValidationError("LongLLMLingua requires the model/tokenizer file closure")
    if not load:
        return None
    from agentinputlab.integrations.external_workers.longllmlingua import LongLLMLinguaNativeRuntime

    return LongLLMLinguaNativeRuntime(**value)


def prepare(config, output_dir):
    if not config.backend or not config.runtime:
        raise ValidationError("controlled LongLLMLingua needs explicit backend and runtime")
    _known(config.parameters, {"config", "profile_kind"}, "preparation parameters")
    settings = _object(config.parameters.get("config", {}), "LongLLMLingua settings")
    _known(settings, {f.name for f in fields(LongLLMLinguaConfig)}, "LongLLMLingua settings")
    settings = LongLLMLinguaConfig(**settings)
    profile_kind = config.parameters.get("profile_kind", "controlled_adapter")
    if profile_kind not in {"controlled_adapter", "factorial_adapter"}:
        raise ValidationError("controlled LongLLMLingua requires controlled_adapter or factorial_adapter")
    output = Path(output_dir).absolute()
    if output.exists():
        raise ValidationError("LongLLMLingua preparation requires a new output directory")
    context = InterventionContext(backend=config._backend(), external_runtime=runtime(config, load=True),
                                  seed=config.seed)
    binding = LongLLMLinguaControlledBindings.from_context(context, budget=config.budget, config=settings)
    ready = replace(config, parameters={"bindings": to_jsonable(binding), "profile_kind": profile_kind})
    ready.validate()
    output.mkdir(parents=True)
    with (output / "run.json").open("x") as stream:
        json.dump(ready.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"release_id": config.release_id, "binding_sha256": binding.binding_hash,
             "run_path": str(output / "run.json"),
             "run_sha256": hashlib.sha256((output / "run.json").read_bytes()).hexdigest(),
             "model_calls": 0, "compression_calls": 0, "fit_performed": False}
    with (output / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof
