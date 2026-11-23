"""Pinned local LLMLingua runtime; executable objects stay in the context."""

from __future__ import annotations

import importlib.metadata
import json
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from agentinputlab.algorithms.curate.llmlingua import (
    LongLLMLinguaOfficialAdapter,
)
from agentinputlab.algorithms.curate.llmlingua.controlled import LongLLMLinguaCompressionRequest
from agentinputlab.artifacts import sha256_file
from agentinputlab.contracts import (
    ExternalDependencyError,
    InterventionContext,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import freeze_mapping

_SOURCE_FILES = {
    "prompt_compressor.py": "472fde8f6f55f273fe94eee40c25c73508f1ed3a7f5d61788636c20fc4be5068",
    "utils.py": "1ed3eb0734354ea0136a0d98a9366233a872d22ff350fab107e53de45c954e33",
}
_SHA = re.compile(r"[0-9a-f]{64}")


@lru_cache(maxsize=4)
def _verify_files(snapshot: str, pins: tuple[tuple[str, str], ...],
                  identities: tuple[tuple[str, int, int], ...]) -> None:
    del identities
    for name, expected in pins:
        if sha256_file(Path(snapshot) / name) != expected:
            raise ExternalDependencyError(f"LongLLMLingua snapshot checksum differs: {name}")


class LongLLMLinguaNativeRuntime:
    """Use a caller-mounted safetensors snapshot, never an implicit alternative.

    The supplied checksums are operator trust. Live calls verify every file
    before loading. Injected factories are explicit caller code; their identity
    is recorded and they do not certify model execution or verify model weights.
    """

    def __init__(self, snapshot: str | Path, *, model_id: str, revision: str,
                 file_sha256: Mapping[str, str], device_map: str = "cpu",
                 torch_dtype: str = "float16", compressor_factory: Any = None):
        self.snapshot = Path(snapshot).resolve()
        if not self.snapshot.is_dir() or not model_id or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValidationError("LongLLMLingua needs a local snapshot and pinned model identity")
        if device_map not in {"cpu", "cuda:0", "mps"} or torch_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValidationError("LongLLMLingua requires an explicit device and floating dtype")
        pins = dict(file_sha256)
        for name, value in pins.items():
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                    or not isinstance(value, str) or _SHA.fullmatch(value) is None):
                raise ValidationError("LongLLMLingua snapshot pins must be safe paths and SHA-256")
        if (not {"config.json", "tokenizer_config.json", "special_tokens_map.json"}.issubset(pins)
                or not any(n.endswith(".safetensors") for n in pins)
                or not any(n in pins for n in ("tokenizer.json", "tokenizer.model"))):
            raise ValidationError("LongLLMLingua snapshot closure lacks model/tokenizer files")
        self._pins = tuple(sorted(pins.items()))
        self._model_id, self._revision = model_id, revision
        self._device, self._dtype = device_map, torch_dtype
        self._factory = compressor_factory
        self._lock = threading.RLock()

    @staticmethod
    def _source() -> Mapping[str, Any]:
        try:
            distribution = importlib.metadata.distribution("llmlingua")
            if distribution.version != "0.2.2":
                raise ExternalDependencyError("LongLLMLingua requires llmlingua==0.2.2")
            for name, expected in _SOURCE_FILES.items():
                if sha256_file(distribution.locate_file("llmlingua/" + name)) != expected:
                    raise ExternalDependencyError("LongLLMLingua upstream source differs from its lock")
        except (importlib.metadata.PackageNotFoundError, OSError) as exc:
            raise ExternalDependencyError("install agentinputlab[compression] and mount the pinned assets") from exc
        return {"package": "llmlingua", "version": "0.2.2", "file_sha256": _SOURCE_FILES}

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping({
            "runtime_id": "agentinputlab.longllmlingua.native.v1", "source": self._source(),
            "model": {"id": self._model_id, "locator": str(self.snapshot),
                "revision": self._revision, "file_sha256": dict(self._pins)},
            "tokenizer": {"revision": self._revision, "file_sha256": {
                name: sha for name, sha in self._pins if "tokenizer" in name or "special_tokens" in name}},
            "load": {"device_map": self._device, "torch_dtype": self._dtype,
                "local_files_only": True, "trust_remote_code": False},
            "injected_factory": self._factory is not None,
            "file_verification": "caller_runtime" if self._factory is not None else "full_sha256_before_live_load",
        })

    def _require_loader_closure(self) -> None:
        """Reject undeclared sidecars/shards that native auto-loading could use."""
        pins = dict(self._pins)
        for name in ("generation_config.json", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer.model", "added_tokens.json", "chat_template.jinja"):
            if (self.snapshot/name).exists() and name not in pins:
                raise ExternalDependencyError(f"LongLLMLingua loader file is not pinned: {name}")
        if (self.snapshot/'adapter_config.json').exists():
            raise ExternalDependencyError("LongLLMLingua does not implicitly load PEFT adapters")
        try:
            config = json.loads((self.snapshot/'config.json').read_text())
            if not any(str(name).endswith('ForCausalLM') for name in config.get('architectures', [])):
                raise ExternalDependencyError("LongLLMLingua perplexity scoring requires a causal LM")
            index = self.snapshot/'model.safetensors.index.json'
            if index.exists():
                if (self.snapshot/'model.safetensors').exists():
                    raise ExternalDependencyError("LongLLMLingua snapshot has ambiguous weight layouts")
                shards = set(json.loads(index.read_text())['weight_map'].values())
                if not shards or any(not isinstance(n, str) or n not in pins for n in shards):
                    raise ExternalDependencyError("LongLLMLingua weight index references unpinned shards")
            elif 'model.safetensors' not in pins:
                raise ExternalDependencyError("LongLLMLingua needs pinned single or indexed safetensors")
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            raise ExternalDependencyError("LongLLMLingua loader config/index is malformed") from exc

    def run(self, request: LongLLMLinguaCompressionRequest) -> Mapping[str, Any]:
        if not isinstance(request, LongLLMLinguaCompressionRequest):
            raise ValidationError("LongLLMLingua runtime requires a typed compression request")
        state, config, seed = request.state, request.config, request.seed
        with self._lock:
            self._source()
            started = time.monotonic()
            cpu_started = time.process_time()
            if self._factory is None:
                try:
                    self._require_loader_closure()
                    identities = tuple((name, (self.snapshot/name).stat().st_size,
                                        (self.snapshot/name).stat().st_mtime_ns) for name, _ in self._pins)
                    _verify_files(str(self.snapshot), self._pins, identities)
                except OSError as exc:
                    raise ExternalDependencyError("LongLLMLingua snapshot closure is incomplete") from exc
            adapter = LongLLMLinguaOfficialAdapter(model_name=str(self.snapshot), model_revision=self._revision,
                device_map=self._device, model_config={"local_files_only": True, "torch_dtype": self._dtype},
                config=config, compressor_factory=self._factory)
            native = adapter.propose(state, InterventionContext(seed=seed))
            native = replace(native, trace=replace(native.trace, events=(*native.trace.events,
                TraceEvent(TracePhase.FINALIZE, "native_helper_execution", {
                    "wall_seconds": time.monotonic()-started,
                    "injected_factory": self._factory is not None,
                    "compression_invocations": 1,
                    "internal_forward_count": "not_metered_native_perplexity_passes",
                    "helper_sampled_output_tokens": 0,
                    "helper_compute_free": False}))))
            return {"patch": native, "usage": {"model_calls": 0, "retrieval_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "cpu_seconds": time.process_time()-cpu_started},
                "usage_scope": "zero sampled generation; helper forward passes and wall time are separate"}


__all__ = ["LongLLMLinguaNativeRuntime"]
