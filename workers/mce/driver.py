#!/usr/bin/env python3
"""Attested v2 sidecar for Meta Context Engineering.

The sidecar is the trusted boundary around generated code.  It binds a clean
source checkout, the environment and three model deployments, enforces model
and tool budgets before provider calls, keeps validation data outside the
generated workspace, and persists the selected artifact in a content-addressed
store.  Only an opaque handle and a regular-file manifest cross back to the
AgentInputLab host.

The fixture runtime is deliberately explicit.  It proves the protocol and an
L2 method path; it is not evidence of paper-result parity.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import math
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Protocol, TextIO

UPSTREAM_COMMIT = "c4b7a7c2ce3ffc4bf4a74c52d2dd8a9a8fb14c30"
PROTOCOL = "agentinputlab.mce.v2"
RELEASE_ID = "recipe.mce.sandboxed_offline"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
HANDLE_PREFIX = "urn:agentinputlab:mce:sha256:"
UPSTREAM_MAIN_SHA256 = "769b9ea180ea468530c38893ad15178422d4ba0bbb23ff183d96bde86fdd8ffe"
UPSTREAM_EVAL_SHA256 = "20ed30d4271ac01ff56318ea89b1b55cc19f5ff8ffbc45361f6a86541ea6fe57"
SEED_MODE = "python_once_and_manager_logical_call_index_v1"
PROVIDER_SUBCALL_SEED_MODE = "sha256_logical_seed_and_stable_subcall_index_63bit_v1"
SEMANTIC_PATCH_MANIFEST = {
    "schema": "agentinputlab.mce.semantic-patch.v2",
    "patch_id": "agentinputlab.mce.semantic-patch.v2",
    "upstream_commit": UPSTREAM_COMMIT,
    "upstream_main_sha256": UPSTREAM_MAIN_SHA256,
    "seams": {
        "mce.main.run_meta_agent": "isolated_manager.run_agent.meta",
        "mce.main.run_base_agent": "isolated_manager.run_agent.base",
        "mce.main.load_interfaces": "isolated_manager.validate_interfaces_static",
        "mce.main.batch_evaluate": "trusted_oracle_supervisor.batch_evaluate",
    },
    "seed_derivation": {
        "python_random": "base_seed_once_per_run",
        "manager_logical_call": "base_seed_plus_logical_call_index",
        "provider_subcall": "sha256_canonical_logical_seed_and_stable_subcall_index_63bit",
    },
    "generated_code_imported_by_worker": False,
}
SEMANTIC_PATCH_MANIFEST_SHA256 = "18f5e2452d60db75d837ccccce61b13b0d3c5d7d3c9cb260250e215e8f9076ca"
EMPTY_INITIAL_CONTEXT = {
    "schema": "agentinputlab.mce.initial-context.v1",
    "files": (),
}
_HEX = frozenset("0123456789abcdef")
_CONTROL_KEYS = frozenset(
    {
        "iterations",
        "train_batch_size",
        "max_model_calls",
        "max_tool_calls",
        "max_wall_seconds",
        "max_artifact_files",
        "max_artifact_bytes",
        "seed",
    }
)
_MODEL_FINGERPRINT_FIELDS = frozenset({"provider", "model", "revision", "configuration_sha256"})
_EXAMPLE_KEYS = frozenset({"example_id", "input", "expected_output", "split", "metadata"})
_FORBIDDEN_EXAMPLE_KEYS = frozenset(
    {"test_answer", "test_gold", "test_label", "test_reward", "test_target"}
)
_ALLOWED_ARTIFACT_PREFIXES = (
    ".claude/skills/learning-context/",
    "context/",
    "interfaces/",
)
_ALLOWED_ARTIFACT_FILES = frozenset({"retrieve_context.py"})
_MAX_JSONL_BYTES = 8 * 1024 * 1024


class DriverError(RuntimeError):
    """Stable sidecar error, including trusted consumed usage when available."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: Mapping[str, int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.usage = dict(usage or {})


@dataclass(frozen=True, slots=True)
class MCEEvolutionIteration:
    """One generated iteration before trusted validation."""

    index: int
    artifact_root: Path
    validation_score: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise DriverError("RUNTIME_INVALID", "MCE iteration index is invalid", consumed=True)
        artifact_root = Path(self.artifact_root)
        if artifact_root.is_symlink():
            raise DriverError(
                "PATH_REJECTED",
                "MCE iteration artifact root may not be a symlink",
                consumed=True,
            )
        object.__setattr__(self, "artifact_root", artifact_root.resolve())
        object.__setattr__(
            self,
            "validation_score",
            _finite(self.validation_score, "MCE validation score"),
        )


@dataclass(frozen=True, slots=True)
class MCEEvolutionRun:
    """Generated candidates returned by the isolated upstream loop."""

    iterations: tuple[MCEEvolutionIteration, ...]
    official_run_iteration_called: bool = True
    meta_agent_enabled: bool = True
    base_agent_enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "iterations", tuple(self.iterations))
        if not self.iterations:
            raise DriverError("RUNTIME_INVALID", "MCE returned no iterations", consumed=True)
        for name in (
            "official_run_iteration_called",
            "meta_agent_enabled",
            "base_agent_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise DriverError(
                    "RUNTIME_INVALID",
                    f"MCE runtime {name} must be boolean",
                    consumed=True,
                )


class TrustedProviderSupervisor:
    """Provider-side meter that reserves each call before the side effect."""

    def __init__(self, *, model_calls: int, tool_calls: int, wall_seconds: int) -> None:
        self._model_limit = _positive_int_value(model_calls, "model_calls")
        self._tool_limit = _positive_int_value(tool_calls, "tool_calls")
        self._wall_limit = _positive_int_value(wall_seconds, "wall_seconds")
        self._started = time.monotonic()
        self._model_calls = 0
        self._tool_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._unpriced_model_calls = 0
        self._cost_usd = 0.0
        self._accelerator_seconds = 0.0
        self._seeded_model_calls = 0
        self._next_subcall_index: dict[int, int] = {}

    def before_model_call(self, *, input_tokens: int = 0) -> None:
        """Reserve one model call before invoking the provider."""

        self._check_wall()
        tokens = _nonnegative_int_value(input_tokens, "input_tokens")
        if self._model_calls >= self._model_limit:
            raise DriverError(
                "BUDGET_EXCEEDED",
                "MCE model-call budget exhausted before provider invocation",
                consumed=self._model_calls > 0 or self._tool_calls > 0,
                usage=self.usage(),
            )
        self._model_calls += 1
        self._unpriced_model_calls += 1
        self._input_tokens += tokens

    def before_seeded_model_call(
        self,
        *,
        logical_seed: int,
        stable_subcall_index: int,
        provider_seed: int,
        input_tokens: int = 0,
    ) -> None:
        """Verify the manager's deterministic subcall seed, then reserve the call."""

        expected = derive_provider_subcall_seed(logical_seed, stable_subcall_index)
        expected_index = self._next_subcall_index.get(logical_seed, 0)
        if stable_subcall_index != expected_index:
            raise DriverError(
                "SEED_INVALID",
                "MCE provider subcall index is not stable and contiguous",
                consumed=self._model_calls > 0 or self._tool_calls > 0,
                usage=self.usage(),
            )
        if provider_seed != expected:
            raise DriverError(
                "SEED_INVALID",
                "MCE provider subcall seed does not match the attested derivation",
                consumed=self._model_calls > 0 or self._tool_calls > 0,
                usage=self.usage(),
            )
        self.before_model_call(input_tokens=input_tokens)
        self._next_subcall_index[logical_seed] = stable_subcall_index + 1
        self._seeded_model_calls += 1

    def verify_all_model_calls_seeded(self) -> None:
        """Fail closed when an attested manager bypassed seeded reservation."""

        if self._seeded_model_calls != self._model_calls:
            raise DriverError(
                "SEED_INVALID",
                "MCE manager reserved an unseeded provider model call",
                consumed=self._model_calls > 0 or self._tool_calls > 0,
                usage=self.usage(),
            )

    def after_model_call(
        self,
        *,
        output_tokens: int = 0,
        cost_usd: float | None = None,
        accelerator_seconds: float = 0.0,
    ) -> None:
        self._output_tokens += _nonnegative_int_value(output_tokens, "output_tokens")
        self._accelerator_seconds += _nonnegative_float_value(
            accelerator_seconds,
            "accelerator_seconds",
        )
        if cost_usd is not None:
            self._cost_usd += _nonnegative_float_value(cost_usd, "cost_usd")
            if self._unpriced_model_calls:
                self._unpriced_model_calls -= 1

    def before_tool_call(self) -> None:
        """Reserve one tool call before invoking the tool provider."""

        self._check_wall()
        if self._tool_calls >= self._tool_limit:
            raise DriverError(
                "BUDGET_EXCEEDED",
                "MCE tool-call budget exhausted before provider invocation",
                consumed=self._model_calls > 0 or self._tool_calls > 0,
                usage=self.usage(),
            )
        self._tool_calls += 1

    def usage(self) -> Mapping[str, int | float]:
        return {
            "model_calls": self._model_calls,
            "retrieval_calls": 0,
            "tool_calls": self._tool_calls,
            "embedding_calls": 0,
            "unpriced_model_calls": self._unpriced_model_calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cpu_seconds": 0.0,
            "accelerator_seconds": self._accelerator_seconds,
            "cost_usd": self._cost_usd,
            "compute_cost_usd": 0.0,
        }

    def check_wall(self) -> None:
        """Reject a completed operation that crossed the configured deadline."""

        self._check_wall()

    def _check_wall(self) -> None:
        if time.monotonic() - self._started > self._wall_limit:
            raise DriverError(
                "BUDGET_EXCEEDED",
                "MCE wall budget exhausted before provider invocation",
                consumed=self._model_calls > 0 or self._tool_calls > 0,
                usage=self.usage(),
            )


def _serialized_evaluation_meter(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._evaluation_lock:
            return method(self, *args, **kwargs)
    return guarded


class TrustedEvaluationSupervisor(TrustedProviderSupervisor):
    """Reserve actual provider-request token caps before frozen-artifact use.

    A returned reservation ID must accompany each provider response. Failed
    requests with unknown token usage keep their reservation quarantined;
    they are never treated as zero-cost calls or retried here.
    """

    def __init__(
        self,
        *,
        model_calls: int = 8,
        tool_calls: int = 8,
        wall_seconds: int = 3600,
        output_tokens: int = 4096,
        output_tokens_per_call: int = 512,
    ) -> None:
        super().__init__(model_calls=model_calls, tool_calls=tool_calls, wall_seconds=wall_seconds)
        self._output_limit = _positive_int_value(output_tokens, "output_tokens")
        self._output_per_call = _positive_int_value(output_tokens_per_call, "output_tokens_per_call")
        if self._output_per_call > self._output_limit:
            raise DriverError("BUDGET_INVALID", "MCE per-call output cap exceeds the total cap")
        self._reservations: dict[int, int] = {}
        self._failed_unknown: set[int] = set()
        self._responses = 0
        self._evaluation_lock = threading.RLock()

    def before_model_call(self, *, input_tokens: int = 0) -> None:
        del input_tokens
        raise DriverError("SEED_INVALID", "MCE final evaluation requires seeded token reservations")

    @_serialized_evaluation_meter
    def before_seeded_model_call(
        self,
        *,
        logical_seed: int,
        stable_subcall_index: int,
        provider_seed: int,
        requested_output_tokens: int,
        input_tokens: int = 0,
    ) -> int:
        requested = _positive_int_value(requested_output_tokens, "requested_output_tokens")
        if (
            requested > self._output_per_call
            or self._output_tokens + sum(self._reservations.values()) + requested > self._output_limit
        ):
            raise DriverError(
                "BUDGET_EXCEEDED", "MCE output budget exhausted before provider invocation",
                consumed=bool(self._model_calls), usage=self.usage(),
            )
        expected = derive_provider_subcall_seed(logical_seed, stable_subcall_index)
        if (
            stable_subcall_index != self._next_subcall_index.get(logical_seed, 0)
            or provider_seed != expected
        ):
            raise DriverError("SEED_INVALID", "MCE final-evaluation provider seed drifted")
        TrustedProviderSupervisor.before_model_call(self, input_tokens=input_tokens)
        self._next_subcall_index[logical_seed] = stable_subcall_index + 1
        self._seeded_model_calls += 1
        reservation_id = self._model_calls
        self._reservations[reservation_id] = requested
        return reservation_id

    @_serialized_evaluation_meter
    def after_model_call(
        self,
        *,
        reservation_id: int,
        output_tokens: int = 0,
        cost_usd: float | None = None,
        accelerator_seconds: float = 0.0,
    ) -> None:
        reservation_id = _positive_int_value(reservation_id, "reservation_id")
        if reservation_id not in self._reservations or reservation_id in self._failed_unknown:
            raise DriverError("USAGE_INVALID", "MCE response has no open token reservation")
        tokens = _nonnegative_int_value(output_tokens, "output_tokens")
        cap = self._reservations[reservation_id]
        # Validate all fields before changing the reservation/counters.
        _nonnegative_float_value(accelerator_seconds, "accelerator_seconds")
        if cost_usd is not None:
            _nonnegative_float_value(cost_usd, "cost_usd")
        super().after_model_call(
            output_tokens=tokens, cost_usd=cost_usd, accelerator_seconds=accelerator_seconds,
        )
        del self._reservations[reservation_id]
        self._responses += 1
        if tokens > cap:
            # Preserve measured overrun cost before rejecting the provider contract.
            raise DriverError(
                "BUDGET_EXCEEDED", "MCE provider exceeded its reserved output-token cap",
                consumed=True, usage=self.usage(),
            )

    @_serialized_evaluation_meter
    def failed_model_call(self, *, reservation_id: int) -> None:
        reservation_id = _positive_int_value(reservation_id, "reservation_id")
        if reservation_id not in self._reservations or reservation_id in self._failed_unknown:
            raise DriverError("USAGE_INVALID", "MCE failure has no open token reservation")
        self._failed_unknown.add(reservation_id)

    @_serialized_evaluation_meter
    def before_tool_call(self) -> None:
        super().before_tool_call()

    @_serialized_evaluation_meter
    def usage(self) -> Mapping[str, int | float]:
        return super().usage()

    @_serialized_evaluation_meter
    def output_budget_receipt(self) -> Mapping[str, Any]:
        return {
            "schema": "agentinputlab.mce.final-output-budget.v1",
            "max_output_tokens": self._output_limit,
            "max_output_tokens_per_call": self._output_per_call,
            "actual_output_tokens": self._output_tokens,
            "provider_responses": self._responses,
            "pending_reservations": {
                str(key): value for key, value in sorted(self._reservations.items())
                if key not in self._failed_unknown
            },
            "failed_unknown_output_reservations": {
                str(key): self._reservations[key] for key in sorted(self._failed_unknown)
            },
            "reserved_unknown_tokens_are_not_actual_usage": True,
        }


class MCERuntime(Protocol):
    """Runtime seam executed inside the attested generated-code boundary."""

    def evolve(
        self,
        *,
        workspace: Path,
        train_path: Path,
        validation_path: Path,
        controls: Mapping[str, int],
        model_fingerprints: Mapping[str, Any],
        supervisor: TrustedProviderSupervisor,
    ) -> MCEEvolutionRun: ...


def final_evaluation_manager_contract() -> Mapping[str, Any]:
    """Operator attestation for the frozen evaluator's provider/oracle boundary."""
    return {
        "schema": "agentinputlab.mce.final-evaluation-manager.v1",
        "source_commit": UPSTREAM_COMMIT,
        "upstream_eval_source_sha256": UPSTREAM_EVAL_SHA256,
        "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
        "frozen_interfaces_only": True,
        "adaptation_disabled": True,
        "test_oracle_mode": "supervisor_only_hidden",
        "output_token_guard": "actual_provider_request_pre_call_reservation_v1",
        "response_reservation_ids": True,
        "failed_unknown_output_budget_quarantined": True,
    }


def _verify_generated_workspace_boundary(sandbox_manager: Any) -> Mapping[str, Any]:
    """Validate the exact operator-attested isolated-manager contract."""

    fingerprint = getattr(sandbox_manager, "fingerprint", None)
    if not callable(fingerprint):
        raise DriverError(
            "ATTESTATION_INVALID",
            "MCE sandbox manager must expose a trusted fingerprint()",
        )
    try:
        value = fingerprint()
    except Exception as exc:
        raise DriverError(
            "ATTESTATION_INVALID",
            "MCE sandbox manager fingerprint could not be read",
        ) from exc
    expected = {
        "schema": "agentinputlab.mce.isolated-manager.v1",
        "generated_code_execution": "isolated_worker_only",
        "oracle_mount_mode": "supervisor_only_hidden",
        "oracle_process_boundary": "trusted_supervisor_out_of_process",
        "generated_workspace_oracle_visible": False,
        "provider_guard_mode": "provider_transport_pre_call",
        "seed_mode": SEED_MODE,
        "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
        "validation_scoring_mode": "supervisor_only",
        "run_agent_mode": "remote_only",
        "interface_validation_mode": "remote_static_no_import",
        "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
        "source_commit": UPSTREAM_COMMIT,
        "upstream_main_sha256": UPSTREAM_MAIN_SHA256,
        "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
    }
    identity_keys = {
        "generated_worker_identity_sha256",
        "oracle_supervisor_identity_sha256",
        "provider_proxy_identity_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != set(expected) | identity_keys
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
    ):
        raise DriverError(
            "ATTESTATION_INVALID",
            "MCE sandbox manager fingerprint does not match the required boundary",
        )
    for key in identity_keys:
        _require_sha256(value.get(key), f"sandbox manager {key}")
    for method in (
        "bind_run",
        "run_agent",
        "validate_interfaces_static",
        "batch_evaluate",
        "usage_receipt",
    ):
        if not callable(getattr(sandbox_manager, method, None)):
            raise DriverError(
                "ATTESTATION_INVALID",
                f"MCE sandbox manager lacks required {method}()",
            )
    return json.loads(_canonical_json(value))


@contextmanager
def _pinned_upstream_modules(
    source_root: Path,
) -> Iterator[tuple[ModuleType, ModuleType]]:
    """Import the pinned upstream modules without accepting ``sys.modules`` injection."""

    root = Path(source_root).resolve()
    main_path = _safe_source_file(root, "mce/main.py")
    if _sha256_file(main_path) != UPSTREAM_MAIN_SHA256:
        raise DriverError("SOURCE_MISMATCH", "MCE upstream main.py hash drifted")
    displaced = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "mce" or name.startswith("mce.") or name == "env" or name.startswith("env.")
    }
    for name in displaced:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    original_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    try:
        upstream_main = importlib.import_module("mce.main")
        upstream_utils = importlib.import_module("mce.utils")
        for module, relative in (
            (upstream_main, "mce/main.py"),
            (upstream_utils, "mce/utils.py"),
        ):
            origin = getattr(module, "__file__", None)
            if not isinstance(origin, str):
                raise DriverError(
                    "UPSTREAM_IMPORT_FAILED",
                    f"pinned MCE module lacks an origin: {module.__name__}",
                )
            try:
                Path(origin).resolve(strict=True).relative_to(root)
            except (FileNotFoundError, ValueError) as exc:
                raise DriverError(
                    "UPSTREAM_IMPORT_FAILED",
                    f"MCE module resolved outside pinned source: {module.__name__}",
                ) from exc
            if Path(origin).resolve() != (root / relative).resolve():
                raise DriverError(
                    "UPSTREAM_IMPORT_FAILED",
                    f"MCE module origin drifted: {module.__name__}",
                )
        yield upstream_main, upstream_utils
    except DriverError:
        raise
    except Exception as exc:
        raise DriverError(
            "UPSTREAM_IMPORT_FAILED",
            "pinned MCE runtime could not be imported",
        ) from exc
    finally:
        for name in tuple(sys.modules):
            if name == "mce" or name.startswith("mce.") or name == "env" or name.startswith("env."):
                sys.modules.pop(name, None)
        sys.modules.update(displaced)
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        importlib.invalidate_caches()


class MCEOfficialRuntime:
    """Run pinned orchestration with four narrowly scoped isolation patches."""

    def __init__(
        self,
        *,
        source_root: Path,
        environment_name: str,
        sandbox_manager: Any,
    ) -> None:
        raw_source_root = Path(source_root)
        if raw_source_root.is_symlink():
            raise DriverError("PATH_REJECTED", "MCE source root may not be a symlink")
        self.source_root = raw_source_root.resolve()
        if not isinstance(environment_name, str) or not environment_name.strip():
            raise DriverError("CONFIG_INVALID", "MCE environment name is required")
        self.environment_name = environment_name
        self.sandbox_manager = sandbox_manager
        self.sandbox_manager_fingerprint = _verify_generated_workspace_boundary(sandbox_manager)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "schema": "agentinputlab.mce.official-runtime.v1",
            "source_commit": UPSTREAM_COMMIT,
            "upstream_main_sha256": UPSTREAM_MAIN_SHA256,
            "semantic_patch_manifest": SEMANTIC_PATCH_MANIFEST,
            "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
            "sandbox_manager": self.sandbox_manager_fingerprint,
            "sandbox_manager_fingerprint_sha256": _content_hash(self.sandbox_manager_fingerprint),
        }

    def _bind_manager_run(
        self,
        *,
        model_fingerprints: Mapping[str, Any],
        seed: int,
        max_wall_seconds: int,
        supervisor: TrustedProviderSupervisor,
    ) -> None:
        binding = self.sandbox_manager.bind_run(
            supervisor=supervisor,
            model_fingerprints=json.loads(_canonical_json(model_fingerprints)),
            seed=seed,
            seed_mode=SEED_MODE,
            provider_subcall_seed_mode=PROVIDER_SUBCALL_SEED_MODE,
            max_wall_seconds=max_wall_seconds,
            semantic_patch_manifest=json.loads(_canonical_json(SEMANTIC_PATCH_MANIFEST)),
        )
        expected = {
            "schema": "agentinputlab.mce.manager-binding.v1",
            "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
            "model_fingerprints_sha256": _content_hash(model_fingerprints),
            "seed": seed,
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "max_wall_seconds": max_wall_seconds,
            "provider_guard_installed": True,
            "provider_pre_call_reservation": True,
        }
        if not isinstance(binding, Mapping) or dict(binding) != expected:
            raise DriverError(
                "ATTESTATION_INVALID", "MCE isolated manager did not bind the run budget and identities",
            )

    def _verify_manager_usage(self, supervisor: TrustedProviderSupervisor) -> None:
        supervisor.check_wall()
        supervisor.verify_all_model_calls_seeded()
        receipt = self.sandbox_manager.usage_receipt()
        expected = {
            "schema": "agentinputlab.mce.usage-receipt.v1",
            "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
            "usage": supervisor.usage(),
        }
        if not isinstance(receipt, Mapping) or dict(receipt) != expected:
            raise DriverError(
                "USAGE_INVALID", "MCE isolated manager usage receipt does not match the trusted meter",
                consumed=True, usage=supervisor.usage(),
            )

    def evolve(
        self,
        *,
        workspace: Path,
        train_path: Path,
        validation_path: Path,
        controls: Mapping[str, int],
        model_fingerprints: Mapping[str, Any],
        supervisor: TrustedProviderSupervisor,
    ) -> MCEEvolutionRun:
        workspace = Path(workspace).resolve()
        train_path = Path(train_path).resolve()
        validation_path = Path(validation_path).resolve()
        target = _mapping(model_fingerprints, "target")
        target_model = _text(target, "model")
        seed = controls["seed"]
        logger = logging.getLogger(f"agentinputlab.mce.{id(self)}")
        logger.handlers[:] = [logging.NullHandler()]
        logger.propagate = False
        run_dir = workspace / "logs" / "agentinputlab"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._bind_manager_run(
            supervisor=supervisor,
            model_fingerprints=model_fingerprints,
            seed=seed,
            max_wall_seconds=controls["max_wall_seconds"],
        )

        with _pinned_upstream_modules(self.source_root) as (
            upstream_main,
            upstream_utils,
        ):
            run_iteration = getattr(upstream_main, "run_iteration", None)
            setup_reference = getattr(
                upstream_utils,
                "setup_meta_agent_reference",
                None,
            )
            if not callable(run_iteration) or not callable(setup_reference):
                raise DriverError(
                    "UPSTREAM_API_DRIFT",
                    "pinned MCE run_iteration/setup_meta_agent_reference is unavailable",
                )
            code = getattr(run_iteration, "__code__", None)
            if (
                code is None
                or Path(code.co_filename).resolve()
                != (self.source_root / "mce" / "main.py").resolve()
            ):
                raise DriverError(
                    "UPSTREAM_API_DRIFT",
                    "MCE run_iteration did not originate from pinned main.py",
                )
            patches = self._semantic_patches(
                workspace=workspace,
                model_fingerprints=model_fingerprints,
                seed=seed,
            )
            for name, replacement in patches.items():
                original = getattr(upstream_main, name, None)
                if not callable(original):
                    raise DriverError(
                        "UPSTREAM_API_DRIFT",
                        f"MCE semantic-patch seam is unavailable: {name}",
                    )
                setattr(upstream_main, name, replacement)
            setup_reference(workspace, str(train_path), controls["train_limit"], logger)

            async def execute() -> tuple[MCEEvolutionIteration, ...]:
                iterations: list[MCEEvolutionIteration] = []
                for index in range(controls["iterations"] + 1):
                    result = await run_iteration(
                        workspace_base=workspace,
                        iteration=index,
                        env_name=self.environment_name,
                        val_data_path=str(validation_path),
                        train_data_path=str(train_path),
                        train_limit=controls["train_limit"],
                        val_limit=controls["validation_limit"],
                        model=target_model,
                        logger=logger,
                        run_dir=run_dir,
                        e2b_sandbox_manager=self.sandbox_manager,
                        train_batch_size=controls["train_batch_size"],
                        skill_path=None,
                        no_meta_agent=False,
                    )
                    if not isinstance(result, Mapping) or "error" in result:
                        raise DriverError(
                            "UPSTREAM_EXECUTION_FAILED",
                            f"MCE upstream iteration {index} failed",
                            consumed=True,
                            usage=supervisor.usage(),
                        )
                    score = _finite(
                        result.get("val_primary_metric"),
                        "MCE upstream validation score",
                    )
                    if index == 0:
                        root = workspace / "iter0"
                    else:
                        sub_iterations = result.get("sub_iterations")
                        if (
                            isinstance(sub_iterations, (str, bytes))
                            or not isinstance(sub_iterations, Sequence)
                            or not sub_iterations
                            or not isinstance(sub_iterations[-1], Mapping)
                        ):
                            raise DriverError(
                                "UPSTREAM_OUTPUT_INVALID",
                                "MCE final sub-iteration folder is missing",
                                consumed=True,
                                usage=supervisor.usage(),
                            )
                        root = Path(_text(sub_iterations[-1], "folder"))
                    iterations.append(
                        MCEEvolutionIteration(
                            index=index,
                            artifact_root=root,
                            validation_score=score,
                        )
                    )
                return tuple(iterations)

            random_state = random.getstate()
            random.seed(seed)
            try:
                iterations = asyncio.run(
                    asyncio.wait_for(
                        execute(),
                        timeout=controls["max_wall_seconds"],
                    )
                )
            except asyncio.TimeoutError as exc:
                raise DriverError(
                    "BUDGET_EXCEEDED",
                    "MCE wall budget expired during upstream orchestration",
                    consumed=bool(supervisor.usage()["model_calls"]),
                    usage=supervisor.usage(),
                ) from exc
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "UPSTREAM_EXECUTION_FAILED",
                    "MCE upstream evolution failed",
                    consumed=True,
                    usage=supervisor.usage(),
                ) from exc
            finally:
                random.setstate(random_state)
        self._verify_manager_usage(supervisor)
        return MCEEvolutionRun(iterations=iterations)

    def evaluate_frozen(
        self,
        *,
        workspace: Path,
        artifact_root: Path,
        selected_iteration: int,
        test_path: Path,
        controls: Mapping[str, int],
        model_fingerprints: Mapping[str, Any],
        supervisor: TrustedEvaluationSupervisor,
    ) -> Mapping[str, Any]:
        """Evaluate frozen interfaces through the same isolated native-task seam.

        The caller must first verify the store handle and heldout split audit.
        This runtime never evolves a skill or imports generated interfaces. The
        manager performs native task evaluation in its attested boundary; it
        must install token reservations at the actual provider transport.
        """
        if not isinstance(supervisor, TrustedEvaluationSupervisor):
            raise DriverError("BUDGET_INVALID", "MCE frozen evaluation requires an output-token guard")
        if _sha256_file(_safe_source_file(self.source_root, "mce/eval.py")) != UPSTREAM_EVAL_SHA256:
            raise DriverError("SOURCE_MISMATCH", "MCE native evaluation source hash drifted")
        workspace = Path(workspace).resolve(strict=True)
        root = _contained_directory(Path(artifact_root), workspace)
        raw_test_path = Path(test_path)
        if raw_test_path.is_symlink() or not raw_test_path.is_file():
            raise DriverError("PATH_REJECTED", "MCE test oracle must be a regular file")
        test_path = raw_test_path.resolve(strict=True)
        if _paths_overlap(workspace, test_path):
            raise DriverError("ISOLATION_INVALID", "MCE test oracle overlaps generated workspace")
        _nonnegative_int_value(selected_iteration, "selected_iteration")
        seed = _nonnegative_int(controls, "seed")
        test_limit = _positive_int(controls, "test_limit")
        wall = _positive_int(controls, "max_wall_seconds")
        max_files = _positive_int(controls, "max_artifact_files")
        max_bytes = _positive_int(controls, "max_artifact_bytes")
        manifest_before = _frozen_artifact_manifest(root, max_files=max_files, max_bytes=max_bytes)
        if selected_iteration == 0 and manifest_before:
            raise DriverError("ARTIFACT_INVALID", "MCE frozen baseline must remain empty")
        if selected_iteration > 0 and _skill_sha256(root) == EMPTY_SKILL_SHA256:
            raise DriverError("ARTIFACT_INVALID", "MCE frozen evolved artifact lacks SKILL.md")
        oracle_hash = _sha256_file(test_path)
        contract = getattr(self.sandbox_manager, "final_evaluation_fingerprint", None)
        if not callable(contract) or contract() != final_evaluation_manager_contract():
            raise DriverError("ATTESTATION_INVALID", "MCE manager lacks frozen-evaluation token/oracle guards")
        self._bind_manager_run(
            model_fingerprints=model_fingerprints, seed=seed, max_wall_seconds=wall,
            supervisor=supervisor,
        )
        run_dir = workspace / "final-evaluation-logs"
        run_dir.mkdir(exist_ok=False)
        random_state = random.getstate()
        random.seed(seed)
        try:
            with _pinned_upstream_modules(self.source_root) as (upstream_main, _):
                environment = upstream_main.EnvironmentRegistry.get(self.environment_name)
                signatures = environment.get_interface_signatures()
                samples = environment.load_samples(
                    path=str(test_path), limit=test_limit, random_sample=False,
                )
                if len(samples) != test_limit:
                    raise DriverError("SPLIT_IDENTITY_MISMATCH", "MCE native loader changed the heldout count")
                seams = self._semantic_patches(
                    workspace=workspace, model_fingerprints=model_fingerprints, seed=seed,
                )
                interfaces = seams["load_interfaces"](root, signatures) if selected_iteration else {}
                result = asyncio.run(asyncio.wait_for(
                    seams["batch_evaluate"](
                        interfaces=interfaces, samples=samples, env_name=self.environment_name,
                        iter_dir=root, log_dir=run_dir,
                    ), timeout=wall,
                ))
            self._verify_manager_usage(supervisor)
            receipt = supervisor.output_budget_receipt()
            if receipt["pending_reservations"]:
                raise DriverError("USAGE_INVALID", "MCE final evaluation left provider calls in flight")
            return {
                "evaluation": result,
                "output_budget": receipt,
                "frozen_artifact_manifest_sha256": _content_hash(manifest_before),
                "test_oracle_sha256": oracle_hash,
                "upstream_eval_source_sha256": UPSTREAM_EVAL_SHA256,
                "adaptation_performed": False,
                "generated_code_imported_by_worker": False,
            }
        except asyncio.TimeoutError as exc:
            raise DriverError(
                "BUDGET_EXCEEDED", "MCE frozen evaluation exceeded its wall deadline",
                consumed=bool(supervisor.usage()["model_calls"]), usage=supervisor.usage(),
            ) from exc
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_EXECUTION_FAILED", "MCE frozen native-task evaluation failed",
                consumed=bool(supervisor.usage()["model_calls"]), usage=supervisor.usage(),
            ) from exc
        finally:
            random.setstate(random_state)
            if (
                _frozen_artifact_manifest(root, max_files=max_files, max_bytes=max_bytes) != manifest_before
                or _sha256_file(test_path) != oracle_hash
            ):
                raise DriverError(
                    "ARTIFACT_INVALID", "MCE frozen artifact or test oracle changed during evaluation",
                    consumed=bool(supervisor.usage()["model_calls"]), usage=supervisor.usage(),
                )

    def _semantic_patches(
        self,
        *,
        workspace: Path,
        model_fingerprints: Mapping[str, Any],
        seed: int,
    ) -> Mapping[str, Callable[..., Any]]:
        manager = self.sandbox_manager
        next_logical_call_index = 0

        def reserve_logical_seed() -> tuple[int, int]:
            nonlocal next_logical_call_index
            logical_call_index = next_logical_call_index
            next_logical_call_index += 1
            return logical_call_index, seed + logical_call_index

        async def run_meta_agent(**kwargs: Any) -> Mapping[str, Any]:
            logical_call_index, logical_seed = reserve_logical_seed()
            return await _run_isolated_agent(
                manager,
                role="meta",
                kwargs=kwargs,
                workspace=workspace,
                model_fingerprint=_mapping(model_fingerprints, "meta"),
                logical_call_index=logical_call_index,
                logical_seed=logical_seed,
            )

        async def run_base_agent(**kwargs: Any) -> Mapping[str, Any]:
            logical_call_index, logical_seed = reserve_logical_seed()
            return await _run_isolated_agent(
                manager,
                role="base",
                kwargs=kwargs,
                workspace=workspace,
                model_fingerprint=_mapping(model_fingerprints, "base"),
                logical_call_index=logical_call_index,
                logical_seed=logical_seed,
            )

        def load_interfaces(
            iter_dir: Path,
            signatures: Sequence[Any],
        ) -> Mapping[str, Any]:
            root = _contained_directory(Path(iter_dir), workspace)
            signature_payload = _interface_signatures_payload(signatures)
            result = manager.validate_interfaces_static(
                iter_dir=root,
                interface_signatures=signature_payload,
                semantic_patch_manifest_sha256=SEMANTIC_PATCH_MANIFEST_SHA256,
            )
            return _validated_remote_interface_handle(
                result,
                iter_dir=root,
                signatures=signature_payload,
            )

        async def batch_evaluate(
            interfaces: Mapping[str, Any],
            samples: Sequence[Any],
            env_name: str,
            llm: Any = None,
            iter_dir: Path | None = None,
            log_dir: Path | None = None,
        ) -> Mapping[str, Any]:
            del llm
            root = _contained_directory(Path(iter_dir), workspace) if iter_dir is not None else None
            if interfaces:
                interface_handle: Mapping[str, Any] | None = _validated_remote_interface_handle(
                    interfaces
                )
            else:
                interface_handle = None
            sample_rows = _sample_payload(samples)
            logical_call_index, logical_seed = reserve_logical_seed()
            result = await manager.batch_evaluate(
                interface_handle=interface_handle,
                samples=sample_rows,
                environment_name=env_name,
                iter_dir=root,
                log_dir=log_dir,
                model_fingerprint=json.loads(
                    _canonical_json(_mapping(model_fingerprints, "target"))
                ),
                seed=logical_seed,
                logical_call_index=logical_call_index,
                seed_mode=SEED_MODE,
                provider_subcall_seed_mode=PROVIDER_SUBCALL_SEED_MODE,
                semantic_patch_manifest_sha256=SEMANTIC_PATCH_MANIFEST_SHA256,
            )
            return _validated_batch_evaluation(result, len(sample_rows))

        return {
            "run_meta_agent": run_meta_agent,
            "run_base_agent": run_base_agent,
            "load_interfaces": load_interfaces,
            "batch_evaluate": batch_evaluate,
        }


async def _run_isolated_agent(
    manager: Any,
    *,
    role: str,
    kwargs: Mapping[str, Any],
    workspace: Path,
    model_fingerprint: Mapping[str, Any],
    logical_call_index: int,
    logical_seed: int,
) -> Mapping[str, Any]:
    if role not in {"meta", "base"}:
        raise DriverError("RUNTIME_INVALID", "MCE isolated-agent role is invalid")
    iter_dir = _contained_directory(Path(kwargs.get("iter_dir")), workspace)
    workspace_base = Path(kwargs.get("workspace_base", workspace)).resolve()
    if workspace_base != workspace:
        raise DriverError("PATH_REJECTED", "MCE agent workspace identity drifted")
    run_dir_value = kwargs.get("run_dir")
    run_dir = (
        _contained_directory(Path(run_dir_value), workspace) if run_dir_value is not None else None
    )
    task_instruction = kwargs.get("task_instruction")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise DriverError("RUNTIME_INVALID", "MCE agent task instruction is invalid")
    iteration = kwargs.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise DriverError("RUNTIME_INVALID", "MCE evolved agent iteration is invalid")
    signatures = _interface_signatures_payload(kwargs.get("interface_signatures", ()))
    result = await manager.run_agent(
        role=role,
        iter_dir=iter_dir,
        workspace_base=workspace,
        run_dir=run_dir,
        task_instruction=task_instruction,
        interface_signatures=signatures,
        iteration=iteration,
        initial_prompt=kwargs.get("initial_prompt"),
        model_fingerprint=json.loads(_canonical_json(model_fingerprint)),
        seed=logical_seed,
        logical_call_index=logical_call_index,
        seed_mode=SEED_MODE,
        provider_subcall_seed_mode=PROVIDER_SUBCALL_SEED_MODE,
        semantic_patch_manifest_sha256=SEMANTIC_PATCH_MANIFEST_SHA256,
    )
    if not isinstance(result, Mapping) or not isinstance(result.get("success"), bool):
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            f"MCE isolated {role} agent returned an invalid receipt",
            consumed=True,
        )
    expected_keys = {"success"} if result["success"] else {"success", "error"}
    if set(result) != expected_keys:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            f"MCE isolated {role} agent receipt schema drifted",
            consumed=True,
        )
    if not result["success"] and (
        not isinstance(result.get("error"), str) or not result["error"].strip()
    ):
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            f"MCE isolated {role} agent error is invalid",
            consumed=True,
        )
    return dict(result)


def _interface_signatures_payload(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("RUNTIME_INVALID", "MCE interface signatures are invalid")
    rows: list[Mapping[str, Any]] = []
    for signature in value:
        if isinstance(signature, Mapping):
            raw = dict(signature)
        else:
            raw = {
                "name": getattr(signature, "name", None),
                "inputs": getattr(signature, "inputs", None),
                "output": getattr(signature, "output", None),
                "description": getattr(signature, "description", None),
            }
        name = raw.get("name")
        description = raw.get("description")
        inputs = raw.get("inputs")
        output = raw.get("output")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(description, str)
            or isinstance(inputs, (str, bytes))
            or not isinstance(inputs, Sequence)
            or isinstance(output, (str, bytes))
            or not isinstance(output, Sequence)
        ):
            raise DriverError("RUNTIME_INVALID", "MCE interface signature drifted")
        rows.append(
            json.loads(
                _canonical_json(
                    {
                        "name": name,
                        "inputs": tuple(tuple(item) for item in inputs),
                        "output": tuple(output),
                        "description": description,
                    }
                )
            )
        )
    return tuple(rows)


def _interface_source_manifest(iter_dir: Path) -> tuple[Mapping[str, Any], ...]:
    interface_root = iter_dir / "interfaces"
    if interface_root.is_symlink():
        raise DriverError("PATH_REJECTED", "MCE interface directory may not be a symlink")
    if not interface_root.is_dir() or not (interface_root / "__init__.py").is_file():
        raise FileNotFoundError(f"MCE interfaces are unavailable in {iter_dir}")
    rows: list[Mapping[str, Any]] = []
    for path in sorted(interface_root.rglob("*.py")):
        if path.is_symlink():
            raise DriverError("PATH_REJECTED", "MCE interface source may not be a symlink")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(interface_root.resolve())
        except ValueError as exc:
            raise DriverError("PATH_REJECTED", "MCE interface source escapes its root") from exc
        rows.append(
            {
                "path": resolved.relative_to(iter_dir).as_posix(),
                "sha256": _sha256_file(resolved),
                "byte_size": resolved.stat().st_size,
            }
        )
    if not rows:
        raise FileNotFoundError(f"MCE interfaces are unavailable in {iter_dir}")
    return tuple(rows)


def _validated_remote_interface_handle(
    value: Any,
    *,
    iter_dir: Path | None = None,
    signatures: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    expected_keys = {
        "schema",
        "semantic_patch_manifest_sha256",
        "interface_signatures_sha256",
        "artifact_manifest_sha256",
        "interface_names",
        "validation_receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "MCE remote interface handle schema drifted",
            consumed=True,
        )
    if value.get("schema") != "agentinputlab.mce.remote-interface-handle.v1":
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface handle schema is invalid")
    if value.get("semantic_patch_manifest_sha256") != SEMANTIC_PATCH_MANIFEST_SHA256:
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface patch identity drifted")
    names = value.get("interface_names")
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface names are invalid")
    names_tuple = tuple(names)
    if any(not isinstance(name, str) or not name for name in names_tuple):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface names are invalid")
    if signatures is not None:
        expected_names = tuple(signature["name"] for signature in signatures)
        if names_tuple != expected_names:
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface names drifted")
        expected_signatures_sha256 = _content_hash(signatures)
    else:
        expected_signatures_sha256 = _require_sha256(
            value.get("interface_signatures_sha256"),
            "interface signatures",
        )
    if value.get("interface_signatures_sha256") != expected_signatures_sha256:
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface signature hash drifted")
    artifact_sha256 = _require_sha256(
        value.get("artifact_manifest_sha256"),
        "interface artifact manifest",
    )
    if iter_dir is not None:
        actual_artifact_sha256 = _content_hash(_interface_source_manifest(iter_dir))
        if artifact_sha256 != actual_artifact_sha256:
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface artifact hash drifted")
    receipt_body = {
        "schema": value["schema"],
        "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
        "interface_signatures_sha256": expected_signatures_sha256,
        "artifact_manifest_sha256": artifact_sha256,
        "interface_names": names_tuple,
    }
    if value.get("validation_receipt_sha256") != _content_hash(receipt_body):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE interface receipt hash drifted")
    return json.loads(_canonical_json(value))


def _sample_payload(samples: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise DriverError("RUNTIME_INVALID", "MCE evaluation samples are invalid")
    rows: list[Mapping[str, Any]] = []
    for sample in samples:
        to_dict = getattr(sample, "to_dict", None)
        raw = to_dict() if callable(to_dict) else sample
        if not isinstance(raw, Mapping):
            raise DriverError("RUNTIME_INVALID", "MCE evaluation sample is invalid")
        rows.append(json.loads(_canonical_json(raw)))
    return tuple(rows)


def _validated_batch_evaluation(value: Any, expected_total: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"summary", "errors", "results"}:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "MCE trusted batch-evaluation schema drifted",
            consumed=True,
        )
    summary = value.get("summary")
    expected_summary_keys = {
        "metrics",
        "primary_metric",
        "primary_metric_value",
        "total",
        "successful",
        "errors",
        "environment",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_summary_keys:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "MCE trusted evaluation summary schema drifted",
            consumed=True,
        )
    total = summary.get("total")
    successful = summary.get("successful")
    errors_count = summary.get("errors")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total != expected_total
        or isinstance(successful, bool)
        or not isinstance(successful, int)
        or isinstance(errors_count, bool)
        or not isinstance(errors_count, int)
        or successful < 0
        or errors_count < 0
        or successful + errors_count != total
    ):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE evaluation counts drifted")
    if not isinstance(summary.get("primary_metric"), str) or not summary["primary_metric"]:
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE primary metric is invalid")
    _finite(summary.get("primary_metric_value"), "MCE primary metric")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE evaluation metrics are invalid")
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE metric name is invalid")
        _finite(metric_value, f"MCE metric {metric_name}")
    results = value.get("results")
    errors = value.get("errors")
    if (
        isinstance(results, (str, bytes))
        or not isinstance(results, Sequence)
        or len(results) != successful
        or isinstance(errors, (str, bytes))
        or not isinstance(errors, Sequence)
        or len(errors) != errors_count
    ):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "MCE evaluation rows drifted")
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "MCE evaluation result is not canonical JSON",
        ) from exc


class PersistentArtifactStore:
    """Small persistent content-addressed store owned by the worker."""

    def __init__(self, root: Path) -> None:
        raw_root = Path(root)
        if raw_root.is_symlink():
            raise DriverError("STORE_INVALID", "MCE artifact store may not be a symlink")
        self.root = raw_root.resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.objects.is_symlink():
            raise DriverError("STORE_INVALID", "MCE artifact store may not be a symlink")

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "mode": "content_addressed_v1",
            "persistent": True,
            "worker_resolve_supported": True,
            "root_identity_sha256": _content_hash(str(self.root)),
        }

    def put(
        self,
        bundle_sha256: str,
        *,
        artifact_root: Path,
        files: Sequence[Mapping[str, Any]],
    ) -> str:
        digest = _require_sha256(bundle_sha256, "artifact bundle")
        destination = self.objects / digest
        manifest = {
            "schema": "agentinputlab.mce.stored-artifact.v1",
            "artifact_bundle_sha256": digest,
            "files": tuple(files),
        }
        if destination.exists():
            stored = self._verified_object(destination, digest)
            if stored != json.loads(_canonical_json(manifest)):
                raise DriverError("STORE_COLLISION", "MCE artifact-store collision")
            return f"{HANDLE_PREFIX}{digest}"
        staging = Path(tempfile.mkdtemp(prefix=".mce-stage-", dir=self.root))
        try:
            payload_root = staging / "payload"
            payload_root.mkdir()
            source_root = Path(artifact_root).resolve()
            for row in files:
                relative = _text(row, "path")
                source = _safe_artifact_file(source_root, relative)
                expected_sha256 = row.get("sha256")
                expected_size = row.get("byte_size")
                if (
                    _sha256_file(source) != expected_sha256
                    or source.stat().st_size != expected_size
                ):
                    raise DriverError("ARTIFACT_INVALID", "MCE artifact changed before storage")
                target = payload_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target, follow_symlinks=False)
                if (
                    target.is_symlink()
                    or _sha256_file(target) != expected_sha256
                    or target.stat().st_size != expected_size
                    or _sha256_file(source) != expected_sha256
                    or source.stat().st_size != expected_size
                ):
                    raise DriverError("ARTIFACT_INVALID", "MCE artifact changed during storage")
            (staging / "manifest.json").write_text(
                _canonical_json(manifest),
                encoding="utf-8",
            )
            os.replace(staging, destination)
            self._verified_object(destination, digest)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return f"{HANDLE_PREFIX}{digest}"

    def resolve(self, handle: str) -> Mapping[str, Any]:
        if not isinstance(handle, str) or not handle.startswith(HANDLE_PREFIX):
            raise DriverError("HANDLE_INVALID", "MCE artifact handle is invalid")
        digest = _require_sha256(handle[len(HANDLE_PREFIX) :], "artifact handle")
        return self._verified_object(self.objects / digest, digest)

    def materialize(
        self,
        handle: str,
        *,
        workspace: Path,
        max_files: int = 128,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> Path:
        """Copy a frozen selection into a fresh external-worker workspace.

        This is byte transport for isolated final evaluation, not execution or
        an import. The caller still has to mount the returned directory inside
        its attested generated-code boundary. No oracle data is copied here.
        """
        max_files = _positive_int_value(max_files, "max_files")
        max_bytes = _positive_int_value(max_bytes, "max_bytes")
        raw_workspace = Path(workspace)
        if raw_workspace.is_symlink() or not raw_workspace.is_dir():
            raise DriverError("PATH_REJECTED", "MCE evaluation workspace must be regular")
        workspace = raw_workspace.resolve()
        if _paths_overlap(workspace, self.root):
            raise DriverError("PATH_REJECTED", "MCE evaluation workspace overlaps the artifact store")
        manifest = self.resolve(handle)
        files = manifest["files"]
        if len(files) > max_files or sum(row["byte_size"] for row in files) > max_bytes:
            raise DriverError("ARTIFACT_LIMIT_EXCEEDED", "MCE frozen selection exceeds evaluation limits")
        digest = manifest["artifact_bundle_sha256"]
        source_root = self.objects / digest / "payload"
        destination = Path(tempfile.mkdtemp(prefix="mce-frozen-selection-", dir=workspace))
        try:
            for row in files:
                relative = row["path"]
                source = _safe_artifact_file(source_root, relative)
                if _sha256_file(source) != row["sha256"] or source.stat().st_size != row["byte_size"]:
                    raise DriverError("ARTIFACT_INVALID", "MCE frozen selection changed before copy")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                target.chmod(0o700 if row["executable"] else 0o600)
                if (
                    target.is_symlink()
                    or _sha256_file(target) != row["sha256"]
                    or target.stat().st_size != row["byte_size"]
                    or _sha256_file(source) != row["sha256"]
                ):
                    raise DriverError("ARTIFACT_INVALID", "MCE frozen selection changed during copy")
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
            if actual != {row["path"] for row in files} or self.resolve(handle) != manifest:
                raise DriverError("ARTIFACT_INVALID", "MCE frozen selection closure changed")
            return destination
        except Exception:
            # Only the freshly created, worker-owned temporary directory is removed.
            shutil.rmtree(destination)
            raise

    def _verified_object(self, destination: Path, digest: str) -> Mapping[str, Any]:
        if destination.is_symlink() or not destination.is_dir():
            raise DriverError("HANDLE_INVALID", "MCE stored artifact object is invalid")
        manifest_path = destination / "manifest.json"
        if manifest_path.is_symlink():
            raise DriverError("HANDLE_INVALID", "MCE stored manifest may not be a symlink")
        manifest = _load_json(manifest_path, code="HANDLE_INVALID")
        if (
            set(manifest) != {"schema", "artifact_bundle_sha256", "files"}
            or manifest.get("schema") != "agentinputlab.mce.stored-artifact.v1"
            or manifest.get("artifact_bundle_sha256") != digest
        ):
            raise DriverError("HANDLE_INVALID", "MCE stored artifact identity drifted")
        files = manifest.get("files")
        if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
            raise DriverError("HANDLE_INVALID", "MCE stored artifact files are invalid")
        payload_root = destination / "payload"
        if payload_root.is_symlink() or not payload_root.is_dir():
            raise DriverError("HANDLE_INVALID", "MCE stored payload is invalid")
        expected_paths: set[str] = set()
        for row in files:
            expected_row_keys = {
                "path",
                "sha256",
                "byte_size",
                "media_type",
                "file_type",
                "generated",
                "executable",
            }
            if not isinstance(row, Mapping) or set(row) != expected_row_keys:
                raise DriverError("HANDLE_INVALID", "MCE stored file manifest drifted")
            relative = _text(row, "path")
            if relative in expected_paths:
                raise DriverError("HANDLE_INVALID", "MCE stored file paths are duplicated")
            expected_paths.add(relative)
            path = _safe_artifact_file(payload_root, relative)
            if _sha256_file(path) != row.get("sha256") or path.stat().st_size != row.get(
                "byte_size"
            ):
                raise DriverError("HANDLE_INVALID", "MCE stored payload hash drifted")
        actual_paths: set[str] = set()
        for path in payload_root.rglob("*"):
            if path.is_symlink():
                raise DriverError("HANDLE_INVALID", "MCE stored payload contains a symlink")
            if path.is_file():
                actual_paths.add(path.relative_to(payload_root).as_posix())
        if actual_paths != expected_paths:
            raise DriverError("HANDLE_INVALID", "MCE stored payload file set drifted")
        return manifest


class MCEWorkerDriver:
    """Verify, execute, validate, persist, and attest one MCE run."""

    def __init__(
        self,
        *,
        source_root: Path,
        runtime: MCERuntime,
        artifact_store: PersistentArtifactStore,
        sandbox_attestation: Mapping[str, Any],
        interface_signature: Mapping[str, Any],
        initial_context_identity: Mapping[str, Any],
        environment_identity: Mapping[str, Any],
        model_fingerprints: Mapping[str, Any],
        manifest_path: Path = DEFAULT_MANIFEST,
        workspace_root: Path | None = None,
        oracle_root: Path | None = None,
        fixture_mode: bool = False,
    ) -> None:
        raw_source_root = Path(source_root)
        raw_manifest_path = Path(manifest_path)
        raw_workspace_root = Path(workspace_root) if workspace_root else None
        raw_oracle_root = Path(oracle_root) if oracle_root else None
        for path, label in (
            (raw_source_root, "source root"),
            (raw_manifest_path, "compatibility manifest"),
            (raw_workspace_root, "workspace root"),
            (raw_oracle_root, "oracle root"),
        ):
            if path is not None and path.is_symlink():
                raise DriverError("CONFIG_INVALID", f"MCE {label} may not be a symlink")
        self.source_root = raw_source_root.resolve()
        self.runtime = runtime
        self.artifact_store = artifact_store
        self.manifest_path = raw_manifest_path.resolve()
        self.manifest = _load_json(self.manifest_path)
        self.workspace_root = raw_workspace_root.resolve() if raw_workspace_root else None
        self.oracle_root = raw_oracle_root.resolve() if raw_oracle_root else None
        for root, label in (
            (self.workspace_root, "workspace root"),
            (self.oracle_root, "oracle root"),
        ):
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
                if root.is_symlink():
                    raise DriverError("CONFIG_INVALID", f"MCE {label} may not be a symlink")
        if (
            self.workspace_root is not None
            and self.oracle_root is not None
            and _paths_overlap(self.workspace_root, self.oracle_root)
        ):
            raise DriverError(
                "CONFIG_INVALID",
                "MCE workspace and oracle roots must be disjoint",
            )
        self.fixture_mode = _strict_bool_value(fixture_mode, "fixture_mode")
        self.interface_signature = _canonical_mapping(interface_signature, "interface signature")
        self.initial_context_identity = _canonical_mapping(
            initial_context_identity,
            "initial context",
        )
        if self.initial_context_identity != json.loads(_canonical_json(EMPTY_INITIAL_CONTEXT)):
            raise DriverError(
                "CONFIG_INVALID",
                "MCE requires the canonical empty initial context",
            )
        self.environment_identity = _canonical_mapping(
            environment_identity,
            "environment identity",
        )
        self.model_fingerprints = _canonical_mapping(
            model_fingerprints,
            "model fingerprints",
        )
        _validate_model_fingerprints(self.model_fingerprints)
        self.source_manifest = _canonical_mapping(
            _mapping(self.manifest, "source"),
            "source manifest",
        )
        self.license_manifest = _canonical_mapping(
            _mapping(self.manifest, "license"),
            "license manifest",
        )
        self.provenance_manifest = _canonical_mapping(
            _mapping(self.manifest, "provenance"),
            "provenance manifest",
        )
        self.semantic_patch_manifest = _canonical_mapping(
            _mapping(self.manifest, "semantic_patch"),
            "semantic patch manifest",
        )
        self.semantic_patch_manifest_sha256 = _content_hash(self.semantic_patch_manifest)
        self.source_tree = _require_sha1(self.source_manifest.get("tree"), "source tree")
        if not self.fixture_mode and not isinstance(self.runtime, MCEOfficialRuntime):
            raise DriverError(
                "CONFIG_INVALID",
                "production MCE requires MCEOfficialRuntime",
            )
        self.store_identity = self.artifact_store.fingerprint()
        self.storage_fingerprint_sha256 = _content_hash(self.store_identity)
        self.sandbox_attestation = self._normalize_sandbox_attestation(sandbox_attestation)
        self._validate_sandbox_attestation()
        self._verify_manifest_contract()

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": "agentinputlab.mce.v2",
            "protocol_version": PROTOCOL,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_tree": self.source_tree,
            "source_manifest": self.source_manifest,
            "source_manifest_sha256": _content_hash(self.source_manifest),
            "license_manifest": self.license_manifest,
            "license_manifest_sha256": _content_hash(self.license_manifest),
            "provenance_manifest": self.provenance_manifest,
            "provenance_manifest_sha256": _content_hash(self.provenance_manifest),
            "semantic_patch_manifest": self.semantic_patch_manifest,
            "semantic_patch_manifest_sha256": self.semantic_patch_manifest_sha256,
            "artifact_store": self.store_identity,
            "storage_fingerprint_sha256": self.storage_fingerprint_sha256,
            "interface_signature_sha256": _content_hash(self.interface_signature),
            "initial_context_sha256": _content_hash(self.initial_context_identity),
            "environment_identity_sha256": _content_hash(self.environment_identity),
            "model_fingerprints_sha256": _content_hash(self.model_fingerprints),
            "image_digest": self.sandbox_attestation["image_digest"],
            "fixture_mode": self.fixture_mode,
            "sandbox_attestation": self.sandbox_attestation,
        }

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self._handle(request)
        except DriverError as exc:
            response: dict[str, Any] = {
                "status": "error",
                "failure_code": exc.code,
                "message": exc.public_message,
                "consumed": exc.consumed,
            }
            if exc.usage:
                response["usage"] = exc.usage
            return response

    def _handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(request, Mapping) and request.get("release_id") == "recipe.mce.final_artifact_eval":
            from .final_eval import handle_final_evaluation
            return handle_final_evaluation(self, request)
        payload = _validate_envelope(request)
        self._verify_source()
        self._verify_request_identity(request, payload)
        controls = _validate_controls(_mapping(payload, "controls"))
        train = _examples(payload.get("train_examples"), expected_split="train")
        validation = _examples(
            payload.get("validation_examples"),
            expected_split="validation",
        )
        split_audit_sha256 = _validate_split_audit(train, validation)
        if request.get("state_hash") != split_audit_sha256:
            raise DriverError("SPLIT_IDENTITY_MISMATCH", "MCE state hash omitted full examples")
        if payload.get("split_audit_sha256") != split_audit_sha256:
            raise DriverError(
                "SPLIT_IDENTITY_MISMATCH",
                "MCE split audit omitted full examples",
            )
        pre_call_budget = _mapping(payload, "pre_call_budget")
        expected_budget = {
            "model_calls": controls["max_model_calls"],
            "tool_calls": controls["max_tool_calls"],
            "wall_seconds": controls["max_wall_seconds"],
        }
        if pre_call_budget != expected_budget:
            raise DriverError("BUDGET_INVALID", "MCE pre-call budget does not match controls")
        supervisor = TrustedProviderSupervisor(**expected_budget)

        workspace_parent = self.workspace_root
        oracle_parent = self.oracle_root
        with (
            tempfile.TemporaryDirectory(
                prefix="agentinputlab-mce-workspace-",
                dir=str(workspace_parent) if workspace_parent else None,
            ) as raw_workspace,
            tempfile.TemporaryDirectory(
                prefix="agentinputlab-mce-oracle-",
                dir=str(oracle_parent) if oracle_parent else None,
            ) as raw_oracle,
        ):
            workspace = Path(raw_workspace).resolve()
            oracle = Path(raw_oracle).resolve()
            if _paths_overlap(workspace, oracle):
                raise DriverError(
                    "ISOLATION_INVALID",
                    "MCE validation oracle overlaps the generated workspace",
                )
            train_path = workspace / "train.jsonl"
            validation_path = oracle / "validation.jsonl"
            _write_native_jsonl(train_path, train)
            _write_native_jsonl(validation_path, validation)
            runtime_controls = {
                **controls,
                "train_limit": len(train),
                "validation_limit": len(validation),
            }
            try:
                result = self.runtime.evolve(
                    workspace=workspace,
                    train_path=train_path,
                    validation_path=validation_path,
                    controls=runtime_controls,
                    model_fingerprints=self.model_fingerprints,
                    supervisor=supervisor,
                )
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "UPSTREAM_EXECUTION_FAILED",
                    "MCE isolated runtime failed",
                    consumed=True,
                    usage=supervisor.usage(),
                ) from exc
            payload_out = self._materialize_response(
                result=result,
                workspace=workspace,
                request_payload=payload,
                controls=controls,
                split_audit_sha256=split_audit_sha256,
                usage=supervisor.usage(),
            )

        return {
            "status": "ok",
            "request_id": request["request_id"],
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL,
            "payload": payload_out,
            "usage": supervisor.usage(),
        }

    def _verify_request_identity(
        self,
        request: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        identity = self.fingerprint()
        expected = {
            "interface_signature_sha256": _content_hash(self.interface_signature),
            "initial_context_sha256": _content_hash(self.initial_context_identity),
            "environment_identity_sha256": _content_hash(self.environment_identity),
            "sandbox_attestation_id": self.sandbox_attestation["attestation_id"],
            "model_fingerprints_sha256": _content_hash(self.model_fingerprints),
            "source_tree": self.source_tree,
            "source_manifest_sha256": _content_hash(self.source_manifest),
            "license_manifest_sha256": _content_hash(self.license_manifest),
            "provenance_manifest_sha256": _content_hash(self.provenance_manifest),
            "semantic_patch_manifest_sha256": self.semantic_patch_manifest_sha256,
            "storage_fingerprint_sha256": self.storage_fingerprint_sha256,
            "worker_fingerprint_sha256": _content_hash(identity),
            "host_execution_allowed": False,
            "artifact_egress": "manifest_and_opaque_handle_only",
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise DriverError("IDENTITY_MISMATCH", f"MCE request mismatch: {key}")
        body_expectations = {
            "interface_signature": self.interface_signature,
            "initial_context_identity": self.initial_context_identity,
            "environment_identity": self.environment_identity,
            "model_fingerprints": self.model_fingerprints,
            "semantic_patch_manifest": self.semantic_patch_manifest,
        }
        for key, value in body_expectations.items():
            if _canonical_json(payload.get(key)) != _canonical_json(value):
                raise DriverError("IDENTITY_MISMATCH", f"MCE request body mismatch: {key}")
        dependencies = request.get("dependency_fingerprints")
        if not isinstance(dependencies, Mapping) or _canonical_json(
            dependencies.get("worker")
        ) != _canonical_json(identity):
            raise DriverError("IDENTITY_MISMATCH", "MCE worker dependency identity drifted")
        controls = _mapping(payload, "controls")
        expected_runtime_binding = _runtime_binding_sha256(
            controls=controls,
            split_audit_sha256=_text(payload, "split_audit_sha256"),
            interface_signature_sha256=expected["interface_signature_sha256"],
            initial_context_sha256=expected["initial_context_sha256"],
            environment_identity_sha256=expected["environment_identity_sha256"],
            model_fingerprints_sha256=expected["model_fingerprints_sha256"],
            source_tree=self.source_tree,
            source_manifest_sha256=expected["source_manifest_sha256"],
            semantic_patch_manifest_sha256=self.semantic_patch_manifest_sha256,
        )
        if payload.get("runtime_binding_sha256") != expected_runtime_binding:
            raise DriverError("IDENTITY_MISMATCH", "MCE runtime binding drifted")

    def _materialize_response(
        self,
        *,
        result: MCEEvolutionRun,
        workspace: Path,
        request_payload: Mapping[str, Any],
        controls: Mapping[str, int],
        split_audit_sha256: str,
        usage: Mapping[str, int | float],
    ) -> Mapping[str, Any]:
        if not isinstance(result, MCEEvolutionRun):
            raise DriverError(
                "RUNTIME_INVALID",
                "MCE runtime result is invalid",
                consumed=True,
                usage=usage,
            )
        expected_indices = tuple(range(controls["iterations"] + 1))
        if tuple(item.index for item in result.iterations) != expected_indices:
            raise DriverError(
                "RUNTIME_INVALID",
                "MCE runtime must return baseline zero through the configured iteration",
                consumed=True,
                usage=usage,
            )
        if not (
            result.official_run_iteration_called
            and result.meta_agent_enabled
            and result.base_agent_enabled
        ):
            raise DriverError(
                "RUNTIME_INVALID",
                "MCE runtime omitted an official bi-level stage",
                consumed=True,
                usage=usage,
            )
        if int(usage["model_calls"]) < 1:
            raise DriverError(
                "USAGE_INVALID",
                "MCE runtime did not record a provider-side model call",
                consumed=True,
                usage=usage,
            )

        iteration_rows: list[Mapping[str, Any]] = []
        manifests: dict[int, tuple[Mapping[str, Any], ...]] = {}
        roots: dict[int, Path] = {}
        for item in result.iterations:
            root = _contained_directory(item.artifact_root, workspace)
            files = _artifact_manifest(
                root,
                max_files=controls["max_artifact_files"],
                max_bytes=controls["max_artifact_bytes"],
            )
            skill_sha256 = _skill_sha256(root)
            if item.index == 0:
                if files or skill_sha256 != EMPTY_SKILL_SHA256:
                    raise DriverError(
                        "ARTIFACT_INVALID",
                        "MCE baseline must be the canonical empty artifact",
                        consumed=True,
                        usage=usage,
                    )
            elif skill_sha256 == EMPTY_SKILL_SHA256:
                raise DriverError(
                    "ARTIFACT_INVALID",
                    "every evolved MCE iteration must contain SKILL.md",
                    consumed=True,
                    usage=usage,
                )
            roots[item.index] = root
            manifests[item.index] = files
            iteration_rows.append(
                {
                    "index": item.index,
                    "skill_sha256": skill_sha256,
                    "artifact_manifest_sha256": _content_hash(files),
                    "validation_score": item.validation_score,
                }
            )
        selected_iteration = max(
            range(len(iteration_rows)),
            key=lambda index: iteration_rows[index]["validation_score"],
        )
        selected_files = manifests[selected_iteration]
        selected_row = iteration_rows[selected_iteration]
        runtime_binding_sha256 = _text(request_payload, "runtime_binding_sha256")
        bound = {
            "source_tree": self.source_tree,
            "source_manifest_sha256": _content_hash(self.source_manifest),
            "license_manifest_sha256": _content_hash(self.license_manifest),
            "provenance_manifest_sha256": _content_hash(self.provenance_manifest),
            "semantic_patch_manifest_sha256": self.semantic_patch_manifest_sha256,
            "storage_fingerprint_sha256": self.storage_fingerprint_sha256,
        }
        bundle = _artifact_bundle(
            files=selected_files,
            selected_iteration=selected_iteration,
            artifact_manifest_sha256=selected_row["artifact_manifest_sha256"],
            skill_sha256=selected_row["skill_sha256"],
            split_audit_sha256=split_audit_sha256,
            interface_signature_sha256=_content_hash(self.interface_signature),
            initial_context_sha256=_content_hash(self.initial_context_identity),
            environment_identity_sha256=_content_hash(self.environment_identity),
            model_fingerprints_sha256=_content_hash(self.model_fingerprints),
            controls_sha256=_content_hash(_mapping(request_payload, "controls")),
            runtime_binding_sha256=runtime_binding_sha256,
            semantic_patch_manifest_sha256=self.semantic_patch_manifest_sha256,
            bound=bound,
        )
        bundle_sha256 = _content_hash(bundle)
        artifact_handle = self.artifact_store.put(
            bundle_sha256,
            artifact_root=roots[selected_iteration],
            files=selected_files,
        )
        runtime_attestation = {
            "source_commit": UPSTREAM_COMMIT,
            "source_tree": self.source_tree,
            "source_manifest_sha256": bound["source_manifest_sha256"],
            "runtime_binding_sha256": runtime_binding_sha256,
            "semantic_patch_manifest_sha256": self.semantic_patch_manifest_sha256,
            "semantic_patch_applied": True,
            "upstream_run_iteration_source_sha256": UPSTREAM_MAIN_SHA256,
            "interface_validation_mode": "remote_static_no_import",
            "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
            "official_run_iteration_called": True,
            "baseline_iteration_executed": True,
            "meta_agent_enabled": True,
            "base_agent_enabled": True,
            "validation_selection": "max_primary_metric_first_tie",
            "validation_oracle_mode": "evaluator_only",
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "generated_agents_validation_access": False,
            "generated_workspace_oracle_visible": False,
            "usage_accounting_mode": "trusted_supervisor_ipc",
            "provider_guard_mode": "provider_transport_pre_call",
            "pre_call_budget_enforced": True,
            "wall_timeout_enforced": True,
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "generated_code_execution": "isolated_worker_only",
            "artifact_egress": "manifest_and_opaque_handle_only",
            "artifact_store_persistent": True,
            "host_code_executed": False,
            "test_split_used": False,
            "runtime_mode": "fixture" if self.fixture_mode else "production",
        }
        return {
            "sandbox_attestation_id": self.sandbox_attestation["attestation_id"],
            "host_code_executed": False,
            "artifact_egress": "manifest_and_opaque_handle_only",
            "split_audit_sha256": split_audit_sha256,
            "interface_signature_sha256": _content_hash(self.interface_signature),
            "initial_context_sha256": _content_hash(self.initial_context_identity),
            "environment_identity_sha256": _content_hash(self.environment_identity),
            "model_fingerprints_sha256": _content_hash(self.model_fingerprints),
            "runtime_binding_sha256": runtime_binding_sha256,
            "semantic_patch_manifest_sha256": self.semantic_patch_manifest_sha256,
            **bound,
            "files": selected_files,
            "artifact_manifest_sha256": selected_row["artifact_manifest_sha256"],
            "artifact_bundle_sha256": bundle_sha256,
            "artifact_handle": artifact_handle,
            "skill_sha256": selected_row["skill_sha256"],
            "iterations": tuple(iteration_rows),
            "selected_iteration": selected_iteration,
            "provenance_status": "generated_unreviewed",
            "runtime_attestation": runtime_attestation,
        }

    def _verify_manifest_contract(self) -> None:
        if self.manifest.get("schema") != "agentinputlab.mce.compatibility.v2":
            raise DriverError("MANIFEST_INVALID", "MCE compatibility schema must be v2")
        source = self.source_manifest
        if source.get("commit") != UPSTREAM_COMMIT:
            raise DriverError("MANIFEST_INVALID", "MCE source commit drifted")
        if source.get("fixture") is not self.fixture_mode:
            raise DriverError("MANIFEST_INVALID", "MCE fixture/source mode mismatch")
        if self.license_manifest.get("license_id") != "MIT":
            raise DriverError("MANIFEST_INVALID", "MCE source license identity drifted")
        if self.provenance_manifest.get("paper_result_parity") is not False:
            raise DriverError("MANIFEST_INVALID", "MCE provenance must not claim L3 parity")
        if (
            self.semantic_patch_manifest != json.loads(_canonical_json(SEMANTIC_PATCH_MANIFEST))
            or self.semantic_patch_manifest_sha256 != SEMANTIC_PATCH_MANIFEST_SHA256
        ):
            raise DriverError("MANIFEST_INVALID", "MCE semantic patch manifest drifted")
        closure = _mapping(source, "import_closure")
        required = _mapping(source, "required_files")
        if not closure or not set(closure).issubset(required):
            raise DriverError("MANIFEST_INVALID", "MCE import closure is incomplete")

    def _verify_source(self) -> None:
        source = self.source_manifest
        for group in ("required_files", "import_closure"):
            for relative, expected in _mapping(source, group).items():
                path = _safe_source_file(self.source_root, relative)
                if _sha256_file(path) != expected:
                    raise DriverError(
                        "SOURCE_MISMATCH",
                        f"MCE source hash mismatch: {relative}",
                    )
        license_path = _safe_source_file(
            self.source_root,
            self.license_manifest.get("path"),
        )
        if _sha256_file(license_path) != self.license_manifest.get("sha256"):
            raise DriverError("SOURCE_MISMATCH", "MCE license hash mismatch")
        if self.fixture_mode:
            return
        head = _git(self.source_root, "rev-parse", "HEAD")
        tree = _git(self.source_root, "rev-parse", "HEAD^{tree}")
        dirty = _git(self.source_root, "status", "--porcelain", "--untracked-files=all")
        if head != UPSTREAM_COMMIT:
            raise DriverError("SOURCE_MISMATCH", "MCE source commit mismatch")
        if tree != self.source_tree:
            raise DriverError("SOURCE_MISMATCH", "MCE source tree mismatch")
        if dirty:
            raise DriverError("SOURCE_DIRTY", "MCE production checkout must be clean")

    def _normalize_sandbox_attestation(
        self,
        value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = _canonical_mapping(value, "sandbox attestation")
        expected = {
            "runtime_mode": "fixture" if self.fixture_mode else "production",
            "source_mode": "fixture" if self.fixture_mode else "git_checkout",
            "artifact_store_fingerprint_sha256": self.storage_fingerprint_sha256,
        }
        result = {**result, **expected}
        return json.loads(_canonical_json(result))

    def _validate_sandbox_attestation(self) -> None:
        expected = {
            "validation_oracle_mode": "evaluator_only",
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "generated_agents_validation_access": False,
            "generated_workspace_oracle_visible": False,
            "usage_accounting_mode": "trusted_supervisor_ipc",
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "pre_call_budget_enforced": True,
            "artifact_store_persistent": True,
            "semantic_patch_manifest_sha256": self.semantic_patch_manifest_sha256,
            "semantic_patch_applied": True,
            "upstream_run_iteration_source_sha256": UPSTREAM_MAIN_SHA256,
            "interface_validation_mode": "remote_static_no_import",
            "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
        }
        for key, value in expected.items():
            if self.sandbox_attestation.get(key) != value:
                raise DriverError(
                    "ATTESTATION_INVALID",
                    f"MCE sandbox attestation mismatch: {key}",
                )
        required = self.sandbox_attestation.get("required_worker_contract")
        expected_contract = {
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "generated_workspace_oracle_visible": False,
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "validation_scoring_mode": "supervisor_only",
        }
        if not isinstance(required, Mapping) or dict(required) != expected_contract:
            raise DriverError(
                "ATTESTATION_INVALID",
                "MCE sandbox required_worker_contract mismatch",
            )


def _validate_envelope(
    request: Mapping[str, Any], *, release_id: str = RELEASE_ID,
) -> Mapping[str, Any]:
    if not isinstance(request, Mapping):
        raise DriverError("REQUEST_INVALID", "MCE request must be a mapping")
    expected = {
        "method_id": "recipe.mce",
        "release_id": release_id,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_version": PROTOCOL,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise DriverError("IDENTITY_MISMATCH", f"MCE envelope mismatch: {key}")
    claimed = request.get("request_id")
    if not isinstance(claimed, str) or not claimed:
        raise DriverError("REQUEST_INVALID", "MCE request_id is required")
    identity = {key: value for key, value in request.items() if key != "request_id"}
    if _content_hash(identity) != claimed:
        raise DriverError("REQUEST_ID_MISMATCH", "MCE request_id does not bind the envelope")
    return _mapping(request, "payload")


def _validate_controls(value: Mapping[str, Any]) -> Mapping[str, int]:
    if set(value) != _CONTROL_KEYS:
        raise DriverError("REQUEST_INVALID", "MCE controls do not match the frozen profile")
    result = {key: _positive_int(value, key) for key in _CONTROL_KEYS if key != "seed"}
    result["seed"] = _nonnegative_int(value, "seed")
    return result


def _examples(value: Any, *, expected_split: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise DriverError("REQUEST_INVALID", f"MCE {expected_split} examples must be non-empty")
    rows: list[Mapping[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _EXAMPLE_KEYS:
            raise DriverError("REQUEST_INVALID", "MCE example schema drifted")
        if raw.get("split") != expected_split:
            raise DriverError("SPLIT_LEAKAGE", "MCE example split label mismatch")
        identifier = raw.get("example_id")
        metadata = raw.get("metadata")
        if not isinstance(identifier, str) or not identifier:
            raise DriverError("REQUEST_INVALID", "MCE example_id is required")
        if not isinstance(metadata, Mapping):
            raise DriverError("REQUEST_INVALID", "MCE example metadata must be a mapping")
        lowered = {str(key).lower() for key in metadata}
        if lowered & _FORBIDDEN_EXAMPLE_KEYS:
            raise DriverError("SPLIT_LEAKAGE", "MCE example contains test-only metadata")
        _canonical_json(raw)
        rows.append(json.loads(_canonical_json(raw)))
    return tuple(rows)


def _validate_split_audit(
    train: tuple[Mapping[str, Any], ...],
    validation: tuple[Mapping[str, Any], ...],
) -> str:
    groups = {"train": train, "validation": validation}
    result: dict[str, tuple[str, ...]] = {}
    for split, rows in groups.items():
        ids = tuple(str(row["example_id"]) for row in rows)
        input_hashes = tuple(_content_hash(row["input"]) for row in rows)
        example_hashes = tuple(_content_hash(row) for row in rows)
        if len(ids) != len(set(ids)):
            raise DriverError("SPLIT_LEAKAGE", f"MCE {split} contains duplicate IDs")
        if len(input_hashes) != len(set(input_hashes)):
            raise DriverError("SPLIT_LEAKAGE", f"MCE {split} contains duplicate inputs")
        if len(example_hashes) != len(set(example_hashes)):
            raise DriverError("SPLIT_LEAKAGE", f"MCE {split} contains duplicate examples")
        result[f"{split}_ids"] = ids
        result[f"{split}_input_hashes"] = input_hashes
        result[f"{split}_example_hashes"] = example_hashes
    if set(result["train_ids"]) & set(result["validation_ids"]):
        raise DriverError("SPLIT_LEAKAGE", "MCE train/validation IDs overlap")
    if set(result["train_input_hashes"]) & set(result["validation_input_hashes"]):
        raise DriverError("SPLIT_LEAKAGE", "MCE train/validation inputs overlap")
    return _content_hash(result)


def _write_native_jsonl(path: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for example in examples:
            input_value = example["input"]
            if not isinstance(input_value, Mapping):
                raise DriverError(
                    "REQUEST_INVALID",
                    "MCE official runtime requires mapping-shaped native inputs",
                )
            record = dict(input_value)
            if {"id", "ground_truth"} & set(record):
                raise DriverError(
                    "REQUEST_INVALID",
                    "MCE native input may not override id or ground_truth",
                )
            record["id"] = example["example_id"]
            record["ground_truth"] = example["expected_output"]
            record["_agentinputlab_metadata"] = dict(example["metadata"])
            handle.write(_canonical_json(record) + "\n")


def _frozen_artifact_manifest(
    root: Path, *, max_files: int, max_bytes: int,
) -> tuple[Mapping[str, Any], ...]:
    files = _artifact_manifest(root, max_files=max_files, max_bytes=max_bytes)
    if {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} != {
        row["path"] for row in files
    }:
        raise DriverError("ARTIFACT_INVALID", "MCE frozen selection contains an unbound file")
    return files


def _artifact_manifest(
    root: Path,
    *,
    max_files: int,
    max_bytes: int,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DriverError("ARTIFACT_INVALID", "MCE artifact may not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not _allowed_artifact_path(relative):
            continue
        size = path.stat().st_size
        total_bytes += size
        if len(rows) + 1 > max_files or total_bytes > max_bytes:
            raise DriverError("ARTIFACT_LIMIT_EXCEEDED", "MCE artifact limits exceeded")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "byte_size": size,
                "media_type": mimetypes.guess_type(relative)[0] or "application/octet-stream",
                "file_type": "regular",
                "generated": True,
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
    return tuple(rows)


def _skill_sha256(root: Path) -> str:
    skill = root / ".claude" / "skills" / "learning-context" / "SKILL.md"
    if skill.is_symlink():
        raise DriverError("ARTIFACT_INVALID", "MCE skill may not be a symlink")
    return _sha256_file(skill) if skill.is_file() else EMPTY_SKILL_SHA256


def _artifact_bundle(
    *,
    files: Sequence[Mapping[str, Any]],
    selected_iteration: int,
    artifact_manifest_sha256: str,
    skill_sha256: str,
    split_audit_sha256: str,
    interface_signature_sha256: str,
    initial_context_sha256: str,
    environment_identity_sha256: str,
    model_fingerprints_sha256: str,
    controls_sha256: str,
    runtime_binding_sha256: str,
    semantic_patch_manifest_sha256: str,
    bound: Mapping[str, str],
) -> Mapping[str, Any]:
    return {
        "schema": "agentinputlab.mce.bundle.v2",
        "files": tuple(files),
        "selected_iteration": selected_iteration,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "skill_sha256": skill_sha256,
        "source_commit": UPSTREAM_COMMIT,
        "source_tree": bound["source_tree"],
        "source_manifest_sha256": bound["source_manifest_sha256"],
        "license_manifest_sha256": bound["license_manifest_sha256"],
        "provenance_manifest_sha256": bound["provenance_manifest_sha256"],
        "semantic_patch_manifest_sha256": semantic_patch_manifest_sha256,
        "storage_fingerprint_sha256": bound["storage_fingerprint_sha256"],
        "split_audit_sha256": split_audit_sha256,
        "interface_signature_sha256": interface_signature_sha256,
        "initial_context_sha256": initial_context_sha256,
        "environment_identity_sha256": environment_identity_sha256,
        "model_fingerprints_sha256": model_fingerprints_sha256,
        "controls_sha256": controls_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
    }


def _runtime_binding_sha256(
    *,
    controls: Mapping[str, Any],
    split_audit_sha256: str,
    interface_signature_sha256: str,
    initial_context_sha256: str,
    environment_identity_sha256: str,
    model_fingerprints_sha256: str,
    source_tree: str,
    source_manifest_sha256: str,
    semantic_patch_manifest_sha256: str,
) -> str:
    return _content_hash(
        {
            "schema": "agentinputlab.mce.runtime-binding.v2",
            "source_commit": UPSTREAM_COMMIT,
            "source_tree": source_tree,
            "source_manifest_sha256": source_manifest_sha256,
            "semantic_patch_manifest_sha256": semantic_patch_manifest_sha256,
            "split_audit_sha256": split_audit_sha256,
            "interface_signature_sha256": interface_signature_sha256,
            "initial_context_sha256": initial_context_sha256,
            "environment_identity_sha256": environment_identity_sha256,
            "model_fingerprints_sha256": model_fingerprints_sha256,
            "controls_sha256": _content_hash(controls),
        }
    )


def _validate_model_fingerprints(value: Mapping[str, Any]) -> None:
    if set(value) != {"target", "meta", "base"}:
        raise DriverError("CONFIG_INVALID", "MCE requires target/meta/base model identities")
    for role in ("target", "meta", "base"):
        model = _mapping(value, role)
        if set(model) != _MODEL_FINGERPRINT_FIELDS:
            raise DriverError(
                "CONFIG_INVALID",
                f"MCE {role} model identity fields drifted",
            )
        for key in ("provider", "model", "revision", "configuration_sha256"):
            _text(model, key)
        _require_sha256(model["configuration_sha256"], f"{role} configuration")


def _contained_directory(path: Path, workspace: Path) -> Path:
    if path.is_symlink():
        raise DriverError("PATH_REJECTED", "MCE artifact root may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(workspace)
    except (FileNotFoundError, ValueError) as exc:
        raise DriverError("PATH_REJECTED", "MCE artifact root escapes the workspace") from exc
    if not resolved.is_dir():
        raise DriverError("PATH_REJECTED", "MCE artifact root must be a directory")
    return resolved


def _safe_source_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DriverError("PATH_REJECTED", "MCE source path is invalid")
    path = root / relative
    if path.is_symlink():
        raise DriverError("PATH_REJECTED", "MCE source symlinks are forbidden")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise DriverError("PATH_REJECTED", "MCE source path escapes or is missing") from exc
    if not resolved.is_file():
        raise DriverError("PATH_REJECTED", "MCE source path must be a file")
    return resolved


def _safe_artifact_file(root: Path, relative: str) -> Path:
    if not _allowed_artifact_path(relative):
        raise DriverError("ARTIFACT_INVALID", "MCE stored artifact path is invalid")
    path = root / relative
    if path.is_symlink():
        raise DriverError("ARTIFACT_INVALID", "MCE stored artifact may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise DriverError("ARTIFACT_INVALID", "MCE stored artifact escapes its root") from exc
    return resolved


def _allowed_artifact_path(relative: str) -> bool:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        return False
    return relative in _ALLOWED_ARTIFACT_FILES or relative.startswith(_ALLOWED_ARTIFACT_PREFIXES)


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError("SOURCE_MISMATCH", "MCE Git verification failed") from exc
    if completed.returncode:
        raise DriverError("SOURCE_MISMATCH", "MCE Git verification failed")
    return completed.stdout.strip()


def _load_json(path: Path, *, code: str = "MANIFEST_INVALID") -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError(code, "MCE JSON manifest is unreadable") from exc
    if not isinstance(value, Mapping):
        raise DriverError(code, "MCE JSON manifest must be an object")
    return value


def _canonical_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DriverError("CONFIG_INVALID", f"MCE {label} must be a mapping")
    try:
        result = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise DriverError("CONFIG_INVALID", f"MCE {label} is not canonical") from exc
    if not isinstance(result, Mapping) or not result:
        raise DriverError("CONFIG_INVALID", f"MCE {label} must not be empty")
    return result


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise DriverError("REQUEST_INVALID", f"MCE {key} must be a mapping")
    return result


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise DriverError("REQUEST_INVALID", f"MCE {key} must be non-empty text")
    return result


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    return _positive_int_value(value.get(key), key)


def _positive_int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverError("REQUEST_INVALID", f"MCE {label} must be a positive integer")
    return value


def _nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    return _nonnegative_int_value(value.get(key), key)


def _nonnegative_int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("REQUEST_INVALID", f"MCE {label} must be non-negative")
    return value


def derive_provider_subcall_seed(logical_seed: int, stable_subcall_index: int) -> int:
    """Derive the manager's provider seed from a logical-call namespace."""

    base = _nonnegative_int_value(logical_seed, "logical_seed")
    index = _nonnegative_int_value(stable_subcall_index, "stable_subcall_index")
    payload = {
        "mode": PROVIDER_SUBCALL_SEED_MODE,
        "logical_seed": base,
        "stable_subcall_index": index,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _nonnegative_float_value(value: Any, label: str) -> float:
    result = _finite(value, f"MCE {label}")
    if result < 0:
        raise DriverError("USAGE_INVALID", f"MCE {label} must be non-negative")
    return result


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise DriverError("RUNTIME_INVALID", f"{label} must be finite", consumed=True)
    return float(value)


def _strict_bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise DriverError("CONFIG_INVALID", f"MCE {label} must be boolean")
    return value


def _require_sha1(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _HEX for character in value)
    ):
        raise DriverError("MANIFEST_INVALID", f"MCE {label} must be a Git tree SHA-1")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise DriverError("CONFIG_INVALID", f"MCE {label} must be a SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


EMPTY_SKILL_SHA256 = _content_hash("")


def serve_jsonl(
    driver: MCEWorkerDriver,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int:
    """Serve an operator-constructed, attested driver over strict JSONL."""

    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        try:
            if len(raw_line.encode("utf-8")) > _MAX_JSONL_BYTES:
                raise DriverError("REQUEST_TOO_LARGE", "MCE JSONL request is too large")
            request = json.loads(raw_line)
            if not isinstance(request, Mapping):
                raise DriverError("REQUEST_INVALID", "MCE JSONL request must be an object")
            response = driver.handle(request)
        except json.JSONDecodeError:
            response = {
                "status": "error",
                "failure_code": "REQUEST_INVALID",
                "message": "MCE JSONL request is not valid JSON",
                "consumed": False,
            }
        except DriverError as exc:
            response = {
                "status": "error",
                "failure_code": exc.code,
                "message": exc.public_message,
                "consumed": exc.consumed,
            }
            if exc.usage:
                response["usage"] = exc.usage
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()
    return 0


def main() -> int:
    print(
        "Instantiate MCEWorkerDriver inside an attested sidecar; "
        "fixture mode is disabled by default.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
