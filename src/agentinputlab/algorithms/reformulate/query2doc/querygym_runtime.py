"""Pinned QueryGym runtime backed by an AgentInputLab LLM backend.

This module is deliberately a thin execution adapter.  QueryGym continues to
own prompt rendering, Query2Doc's fallback behavior, and its five-query-plus-
pseudo-document concatenation.  AgentInputLab only verifies the pinned source
closure, adapts the model call to :class:`LLMBackend`, and returns auditable
usage information to the existing :class:`QueryGymLiveBridge`.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, GenerationResponse, LLMBackend
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    ExternalDependencyError,
    Message,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .models import QueryGymMode, QueryGymRawResult, QueryGymRequest
from .querygym_bridge import QUERYGYM_REVISION

_SOURCE_SHA256 = freeze_mapping(
    {
        "querygym/methods/query2doc.py": (
            "5af116ce29492cf1571c178c6f8b82e68a9d20cbc23e38be78b1819bf88f655e"
        ),
        "querygym/core/base.py": (
            "e0d35055c5086ad498293f3e0373b9b5d48587cb28f963f0e223e5fb42900210"
        ),
        "querygym/core/prompts.py": (
            "ca76d5878d158b9ba529bed92fd86db26a6e935f617b9edc33e17d1041d1433a"
        ),
        "querygym/prompt_bank.yaml": (
            "3e056850e52245860331a668426be566fc7a6cc0888f1764685a3416ef593e39"
        ),
    }
)
_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class _UpstreamSymbols:
    method_config: type[Any]
    prompt_bank: type[Any]
    query_item: type[Any]
    query2doc: type[Any]


class _BackendChatClient:
    """Expose QueryGym's one-method ``chat`` seam without changing prompts."""

    def __init__(self, backend: LLMBackend, *, seed: int) -> None:
        self.backend = backend
        self.seed = seed
        self.last_messages: tuple[Mapping[str, str], ...] = ()
        self.last_response: GenerationResponse | None = None

    def chat(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        normalized: list[Message] = []
        raw_messages: list[Mapping[str, str]] = []
        for item in messages:
            if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
                raise AlgorithmOutputError("QueryGym emitted an invalid chat message")
            role = item["role"]
            content = item["content"]
            if not isinstance(role, str) or not isinstance(content, str):
                raise AlgorithmOutputError("QueryGym chat messages require string fields")
            normalized.append(Message(role=role, content=content))
            raw_messages.append(freeze_mapping({"role": role, "content": content}))
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 256)
        if kwargs:
            raise ValidationError(
                "QueryGym passed unsupported LLM options: " + ", ".join(sorted(kwargs))
            )
        response = self.backend.generate(
            GenerationRequest(
                messages=tuple(normalized),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.seed,
                metadata={
                    "method_id": "query2doc",
                    "release_id": "rewrite.query2doc.querygym_live",
                    "upstream_revision": QUERYGYM_REVISION,
                },
            )
        )
        self.last_messages = tuple(raw_messages)
        self.last_response = response
        return response.text


class PinnedQueryGymRuntime:
    """Execute QueryGym Query2Doc from an exact, source-verified checkout."""

    worker_id = "querygym-query2doc-pinned"

    def __init__(self, checkout: str | Path, backend: LLMBackend) -> None:
        if not isinstance(backend, LLMBackend):
            raise TypeError("PinnedQueryGymRuntime requires an LLMBackend")
        self.checkout = Path(checkout).expanduser().resolve()
        self.backend = backend
        self.checkout_identity = _checkout_identity(self.checkout)
        self._symbols = _load_upstream_symbols(self.checkout)

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "implementation": "agentinputlab.querygym.pinned_runtime.v1",
                "revision": QUERYGYM_REVISION,
                "source_closure_sha256": self.checkout_identity["source_closure_sha256"],
                "source_files": dict(_SOURCE_SHA256),
                "backend": dict(self.backend.fingerprint()),
                "upstream_logic_owner": "ls3-lab/QueryGym",
            }
        )

    def reformulate(self, request: QueryGymRequest) -> QueryGymRawResult:
        if not isinstance(request, QueryGymRequest):
            raise TypeError("PinnedQueryGymRuntime requires QueryGymRequest")
        expected_backend = dict(self.backend.fingerprint())
        if request.backend_fingerprint and content_hash(
            request.backend_fingerprint
        ) != content_hash(expected_backend):
            raise ValidationError("QueryGym request backend fingerprint differs from runtime")
        if request.mode is QueryGymMode.FEW_SHOT:
            raise ExternalDependencyError(
                "pinned QueryGym few-shot execution requires an explicit training-data binding; "
                "use zero-shot or CoT for the production smoke"
            )
        upstream_mode = "cot" if request.mode is QueryGymMode.COT else "zs"
        config = self._symbols.method_config(
            name="query2doc",
            params={"mode": upstream_mode},
            llm={
                "model": self.backend.model_id,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
            seed=request.seed,
            retries=0,
        )
        client = _BackendChatClient(self.backend, seed=request.seed)
        prompts = self._symbols.prompt_bank(self.checkout / "querygym" / "prompt_bank.yaml")
        method = self._symbols.query2doc(config, client, prompts)
        result = method.reformulate(self._symbols.query_item(request.query_id, request.query))
        metadata = getattr(result, "metadata", None)
        reformulated = getattr(result, "reformulated", None)
        if not isinstance(metadata, Mapping) or not isinstance(reformulated, str):
            raise AlgorithmOutputError("QueryGym returned an invalid ReformulationResult")
        fallback = bool(metadata.get("fallback", False))
        response = client.last_response
        if response is None:
            if not fallback:
                raise AlgorithmOutputError("QueryGym returned without an observable model call")
            usage = {
                "model_calls": 1,
                "retrieval_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            messages: tuple[Mapping[str, str], ...] = ()
        else:
            usage = {
                "model_calls": 1,
                "retrieval_calls": 0,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            if response.usage.cost_usd is not None:
                usage["cost_usd"] = response.usage.cost_usd
            messages = client.last_messages
        generated = metadata.get("pseudo_doc", "")
        if not isinstance(generated, str):
            raise AlgorithmOutputError("QueryGym pseudo_doc metadata must be text")
        return QueryGymRawResult(
            reformulated=reformulated,
            generated_text=generated,
            prompt_hash=content_hash(messages),
            fallback=fallback,
            usage=usage,
            metadata={
                **dict(metadata),
                "upstream_revision": QUERYGYM_REVISION,
                "source_closure_sha256": self.checkout_identity["source_closure_sha256"],
            },
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkout_identity(root: Path) -> Mapping[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("QueryGym checkout must be a non-symlink directory")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactError("cannot verify QueryGym checkout revision") from exc
    if revision != QUERYGYM_REVISION:
        raise ArtifactError("QueryGym checkout does not match the pinned revision")
    observed: dict[str, str] = {}
    for relative, expected in _SOURCE_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"QueryGym source file is unavailable: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ArtifactError(f"QueryGym source checksum mismatch: {relative}")
        observed[relative] = actual
    return freeze_mapping(
        {
            "revision": revision,
            "source_closure_sha256": content_hash(observed),
        }
    )


def _load_upstream_symbols(root: Path) -> _UpstreamSymbols:
    """Load only Query2Doc's audited source closure.

    Importing ``querygym.core.base`` normally executes QueryGym's top-level
    ``__init__`` and therefore imports every optional method, search adapter,
    and data loader.  That is both unnecessary and a source of hidden runtime
    dependencies.  The isolated namespace below executes the four pinned
    modules (plus the small registry module) with their original relative
    imports and does not modify their algorithm logic.
    """

    with _IMPORT_LOCK:
        namespace = (
            "_agentinputlab_querygym_" + str(_checkout_identity(root)["source_closure_sha256"])[:16]
        )
        existing = sys.modules.get(f"{namespace}.methods.query2doc")
        if isinstance(existing, ModuleType):
            return _symbols_from_modules(
                sys.modules[f"{namespace}.core.base"],
                sys.modules[f"{namespace}.core.prompts"],
                existing,
            )
        try:
            _install_namespace_package(namespace, root / "querygym")
            _install_namespace_package(f"{namespace}.core", root / "querygym" / "core")
            _install_namespace_package(f"{namespace}.methods", root / "querygym" / "methods")
            base = _load_source_module(
                f"{namespace}.core.base", root / "querygym" / "core" / "base.py"
            )
            prompts = _load_source_module(
                f"{namespace}.core.prompts", root / "querygym" / "core" / "prompts.py"
            )
            _load_source_module(
                f"{namespace}.core.registry", root / "querygym" / "core" / "registry.py"
            )
            method = _load_source_module(
                f"{namespace}.methods.query2doc",
                root / "querygym" / "methods" / "query2doc.py",
            )
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise ExternalDependencyError(
                "the pinned QueryGym checkout is missing a runtime dependency; "
                "install agentinputlab[querygym]"
            ) from exc
        return _symbols_from_modules(base, prompts, method)


def _install_namespace_package(name: str, path: Path) -> None:
    package = sys.modules.get(name)
    if package is not None:
        return
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = package


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"cannot load pinned QueryGym module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Native fallback diagnostics must not corrupt the CLI's JSON stdout.
    # Bind only this isolated module: global stdout redirection would also
    # capture unrelated output from concurrent library calls.
    module.__dict__["print"] = _diagnostic_print
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _diagnostic_print(*values: Any, **kwargs: Any) -> None:
    if kwargs.get("file") is None:
        kwargs["file"] = sys.stderr
    builtins.print(*values, **kwargs)


def _symbols_from_modules(
    base: ModuleType,
    prompts: ModuleType,
    method: ModuleType,
) -> _UpstreamSymbols:
    return _UpstreamSymbols(
        method_config=base.MethodConfig,
        prompt_bank=prompts.PromptBank,
        query_item=base.QueryItem,
        query2doc=method.Query2Doc,
    )
