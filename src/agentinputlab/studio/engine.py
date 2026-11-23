"""Safe, deterministic experiment execution used by the packaged local Studio.

The Studio deliberately exposes only profiles that can be constructed without
arbitrary imports, shell commands, uploaded code, or network access.  The full
registry remains visible so that unavailable profiles fail with an explicit
compatibility reason instead of silently falling back to another algorithm.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from types import MappingProxyType
from typing import Any

from agentinputlab.algorithms.clarify import UDCSPromptNativePolicy
from agentinputlab.algorithms.decompose.plan_and_budget import (
    CachedPlanBudgetRecord,
    PlanAndBudgetCachedSoftBudget,
)
from agentinputlab.algorithms.decompose.plan_and_solve import PlanAndSolvePSPrompt
from agentinputlab.algorithms.reformulate.query2doc import (
    OfficialQuery2DocRow,
    Query2DocCache,
    Query2DocFixtureCached,
)
from agentinputlab.backends.llm import ReplayBackend
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import BenchmarkRegistry, BenchmarkSmokeAssetRegistry
from agentinputlab.contracts import (
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    SemanticFamily,
)
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.metrics import compression_ratio, intent_token_recall, token_jaccard
from agentinputlab.metrics.paired import summarize_pairs
from agentinputlab.observability import ResourceUsage
from agentinputlab.registry import AlgorithmRegistry
from agentinputlab.runner import InterventionRunner
from agentinputlab.selectors import list_selector_cards

from .model_backends import (
    StudioModelBackendGate,
    StudioModelBackendUnavailable,
    StudioModelRuntime,
)
from .prepared import (
    MAX_PREPARED_SAMPLE_LIMIT,
    discover_prepared_benchmarks,
    discover_prepared_production_cells,
    load_verified_studio_dataset,
    prepared_catalog_payload,
)

_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|bearer|password|secret|access[_-]?token|private[_-]?key)",
    re.IGNORECASE,
)

MAX_RUNS = 100
MAX_TEXT_LENGTH = 4_096
_LOGGER = logging.getLogger(__name__)


class StudioValidationError(ValueError):
    """Raised when an execution request does not pass Studio validation."""


class StudioSecurityError(ValueError):
    """Raised when a request attempts to submit credentials or executable input."""


class StudioQueueFullError(RuntimeError):
    """Raised when the bounded local queue cannot accept more work."""


@dataclass(frozen=True, slots=True)
class StudioExample:
    example_id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    pseudo_doc: str = ""
    instruction: str = ""
    visible_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.relevant_document_ids:
            raise ValueError("Studio examples require at least one relevant document ID")
        object.__setattr__(self, "visible_metadata", MappingProxyType(dict(self.visible_metadata)))

    @property
    def target_doc_id(self) -> str:
        """Compatibility label for the first evaluator-only relevant ID."""

        return self.relevant_document_ids[0]


TINY_EXAMPLES = (
    StudioExample(
        example_id="capital-france",
        query="capital france",
        relevant_document_ids=("france",),
        pseudo_doc="Paris is France's capital city and the seat of its national government.",
    ),
    StudioExample(
        example_id="python-creator",
        query="creator python programming language",
        relevant_document_ids=("python",),
        pseudo_doc="Guido van Rossum created the Python programming language.",
    ),
    StudioExample(
        example_id="water-freezing",
        query="water freezing point celsius",
        relevant_document_ids=("water",),
        pseudo_doc=(
            "At standard atmospheric pressure, pure water freezes at zero degrees Celsius."
        ),
    ),
    StudioExample(
        example_id="pride-author",
        query="author pride prejudice novel",
        relevant_document_ids=("austen",),
        pseudo_doc="Jane Austen wrote the novel Pride and Prejudice.",
    ),
)

TINY_DOCUMENTS = (
    Document(
        doc_id="france",
        title="France",
        text="Paris is the capital and most populous city of France.",
        corpus_revision="studio-tiny-v1",
    ),
    Document(
        doc_id="germany",
        title="Germany",
        text="Berlin is the capital city of Germany.",
        corpus_revision="studio-tiny-v1",
    ),
    Document(
        doc_id="python",
        title="Python",
        text="Python was created by Guido van Rossum and first released in 1991.",
        corpus_revision="studio-tiny-v1",
    ),
    Document(
        doc_id="java",
        title="Java",
        text="James Gosling designed Java at Sun Microsystems.",
        corpus_revision="studio-tiny-v1",
    ),
    Document(
        doc_id="water",
        title="Freezing point",
        text="Water freezes at 0 degrees Celsius under standard pressure.",
        corpus_revision="studio-tiny-v1",
    ),
    Document(
        doc_id="austen",
        title="Pride and Prejudice",
        text="Pride and Prejudice is an 1813 novel written by Jane Austen.",
        corpus_revision="studio-tiny-v1",
    ),
)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    algorithm_id: str
    release_id: str
    family: str
    backend_ids: tuple[str, ...]
    note: str


RUNTIME_PROFILES = {
    "baseline.no_op": RuntimeProfile(
        "baseline.no_op",
        "baseline.no_op",
        "baseline",
        ("offline-replay",),
        "Typed NO_OP paired baseline from the core patch contract.",
    ),
    "fixture.query2doc.cached": RuntimeProfile(
        "rewrite.query2doc",
        "fixture.query2doc.cached",
        SemanticFamily.REFORMULATE.value,
        ("offline-replay",),
        "Local unverified Query2doc fixture; never presented as the official artifact.",
    ),
    "plan.plan_and_solve.ps_prompt": RuntimeProfile(
        "plan.plan_and_solve",
        "plan.plan_and_solve.ps_prompt",
        SemanticFamily.DECOMPOSE.value,
        ("offline-replay",),
        "Clean-room paper prompt adapter; downstream answer generation is not simulated.",
    ),
    "plan.plan_and_budget.native_cached_soft_budget": RuntimeProfile(
        "plan.plan_and_budget",
        "plan.plan_and_budget.native_cached_soft_budget",
        SemanticFamily.DECOMPOSE.value,
        ("offline-replay",),
        "Pinned cached-plan adapter with deterministic soft word allocation.",
    ),
    "clarify.udcs.repo_native_prompt_route": RuntimeProfile(
        "clarify.udcs",
        "clarify.udcs.repo_native_prompt_route",
        SemanticFamily.CLARIFY.value,
        ("offline-replay", "openai-compatible", "transformers-local"),
        (
            "Prompt-native routing evaluated against replay by default, or an "
            "operator-attested ProductionBackendBuilder deployment."
        ),
    ),
}

_OFFLINE_BACKEND = {
    "id": "offline-replay",
    "name": "Offline Replay",
    "ready": True,
    "network": False,
    "description": "Recorded deterministic responses; zero billable cost.",
    "operator_attested": False,
}

DEFAULT_PARAMETERS: dict[str, int | float] = {
    "seed": 7,
    "sample_limit": 4,
    "top_k": 3,
    "temperature": 0.0,
    "soft_word_budget": 180,
    "clarification_threshold": 0.5,
}
DEFAULT_FAIRNESS = {
    "paired_baseline": True,
    "fixed_seed": True,
    "shared_retriever": True,
    "freeze_dependencies": True,
    "fail_on_fingerprint_drift": True,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _version_payload(release: Any) -> dict[str, Any]:
    payload = release.to_dict()
    profile = RUNTIME_PROFILES.get(release.id)
    payload.update(
        {
            "studio_ready": profile is not None,
            "studio_reason": (
                profile.note
                if profile is not None
                else "; ".join(release.blocking_reasons)
                or "This release is catalogued but has no safe deterministic Studio constructor."
            ),
        }
    )
    return payload


def build_catalog(
    prepared_data_root: str | Path | None = None,
    *,
    model_backend_gate: StudioModelBackendGate | None = None,
) -> dict[str, Any]:
    """Build the UI catalog from registries and verified local prepared caches."""

    registry = AlgorithmRegistry.default()
    backend_gate = model_backend_gate or StudioModelBackendGate.from_operator_config(None, None)
    backends = [
        dict(_OFFLINE_BACKEND),
        dict(backend_gate.catalog_entry("openai-compatible")),
        dict(backend_gate.catalog_entry("transformers-local")),
    ]
    prepared_benchmarks = discover_prepared_benchmarks(prepared_data_root)
    prepared_production_cells = discover_prepared_production_cells(prepared_data_root)
    algorithms: list[dict[str, Any]] = [
        {
            "id": "baseline.no_op",
            "name": "Paired baseline",
            "family": "baseline",
            "tier": "builtin",
            "status": "supported",
            "paper_title": "AgentInputLab typed NO_OP",
            "versions": [
                {
                    "id": "baseline.no_op",
                    "status": "supported",
                    "delivery_tier": "builtin",
                    "profile_modes": ["controlled_adapter"],
                    "studio_ready": True,
                    "studio_reason": RUNTIME_PROFILES["baseline.no_op"].note,
                }
            ],
        }
    ]
    for method in registry:
        versions = [_version_payload(release) for release in method.releases]
        if method.method_id == "rewrite.query2doc":
            versions.insert(
                0,
                {
                    "id": "fixture.query2doc.cached",
                    "status": "experimental",
                    "delivery_tier": "studio_fixture",
                    "profile_modes": ["fixture_unverified"],
                    "studio_ready": True,
                    "studio_reason": RUNTIME_PROFILES["fixture.query2doc.cached"].note,
                },
            )
        algorithms.append(
            {
                "id": method.method_id,
                "name": method.display_name,
                "family": method.semantic_family.value,
                "tier": method.release_tier.value,
                "status": method.status.value,
                "paper_title": method.paper_title,
                "publication": method.publication.to_dict(),
                "versions": versions,
            }
        )

    datasets: list[dict[str, Any]] = [
        {
            "id": "studio.tiny_general",
            "suite": "Studio fixture",
            "task": "Offline query intervention + BM25 top-1 retrieval",
            "size": f"{len(TINY_EXAMPLES)} queries; {len(TINY_DOCUMENTS)} documents",
            "families": [item.value for item in SemanticFamily],
            "metrics": [
                "top1_accuracy",
                "intent_token_recall",
                "token_jaccard",
                "compression_ratio",
            ],
            "metric_layers": {
                "adapter": [
                    "top1_accuracy",
                    "intent_token_recall",
                    "token_jaccard",
                    "compression_ratio",
                ],
                "paired": [],
                "study": [],
            },
            "family_support": "cell_declared",
            "data_ready": True,
            "execution_ready": True,
            "ready": True,
            "reason": "Bundled synthetic fixture; no download or external license gate.",
            "studio_release_ids": sorted(RUNTIME_PROFILES),
            "sample": {
                "id": TINY_EXAMPLES[0].example_id,
                "query": TINY_EXAMPLES[0].query,
                "target_document": TINY_EXAMPLES[0].target_doc_id,
            },
        }
    ]
    for cell in BenchmarkRegistry.default().list():
        execution_prepared = prepared_benchmarks.get(cell.cell_id)
        production_prepared = prepared_production_cells.get(cell.cell_id)
        execution_ready = execution_prepared is not None
        data_ready = production_prepared is not None or execution_ready
        if execution_ready:
            reason = (
                "Checksum-verified prepared cache; bounded baseline retrieval smoke is enabled."
            )
            data_reason = "The complete prepared source set passed checksum verification."
        elif data_ready:
            reason = (
                "Prepared data is verified, but Studio has no configured downstream "
                "predictor or retrieval index for this cell."
            )
            data_reason = "The complete prepared source set passed checksum verification."
        else:
            reason = (
                cell.eligibility_gate
                or "Registered benchmark; no complete checksum-verified prepared asset set "
                "was found."
            )
            data_reason = "A complete checksum-verified prepared asset set is unavailable."
        dataset_payload: dict[str, Any] = {
            "id": cell.cell_id,
            "suite": cell.suite,
            "task": cell.task,
            "size": cell.size,
            "families": [family.value for family in cell.families],
            "metrics": list(cell.metrics),
            "metric_layers": {
                "adapter": list(cell.adapter_metrics),
                "paired": list(cell.paired_metrics),
                "study": list(cell.study_metrics),
            },
            "family_support": cell.family_support,
            "data_ready": data_ready,
            "execution_ready": execution_ready,
            "ready": execution_ready,
            "reason": reason,
            "data_reason": data_reason,
            "source_revision": cell.source_revision,
            "studio_release_ids": ["baseline.no_op"] if execution_ready else [],
        }
        if production_prepared is not None:
            dataset_payload["prepared"] = production_prepared.audit_payload()
        elif execution_prepared is not None:
            prepared_asset = BenchmarkSmokeAssetRegistry.default().get("beir.arguana.official")
            dataset_payload["prepared"] = prepared_catalog_payload(
                prepared_asset,
                execution_prepared,
            )
        datasets.append(dataset_payload)

    selectors = [
        {
            "id": card.selector_id,
            "name": card.display_name,
            "kind": card.kind.value,
            "requires_fit": card.requires_fit,
            "optional_dependency": card.optional_dependency or None,
            "artifact_lifecycle": ["fitted", "calibrated", "frozen"],
            "studio_execution_ready": False,
            "reason": (
                "Catalog and exact artifact inspection are available; executing a "
                "selector policy requires an operator-frozen artifact and locked "
                "release builders."
            ),
        }
        for card in list_selector_cards()
    ]
    return {
        "schema_version": 1,
        "algorithms": algorithms,
        "datasets": datasets,
        "selectors": selectors,
        "backends": backends,
        "parameters": {
            "defaults": dict(DEFAULT_PARAMETERS),
            "bounds": {
                "seed": [0, 2_147_483_647],
                "sample_limit": [1, max(len(TINY_EXAMPLES), MAX_PREPARED_SAMPLE_LIMIT)],
                "top_k": [1, 10],
                "temperature": [0.0, 2.0],
                "soft_word_budget": [1, 4_096],
                "clarification_threshold": [0.0, 1.0],
            },
        },
        "fairness_defaults": dict(DEFAULT_FAIRNESS),
        "default_config": {
            "algorithm_id": "rewrite.query2doc",
            "release_id": "fixture.query2doc.cached",
            "backend_id": "offline-replay",
            "dataset_id": "studio.tiny_general",
            "parameters": dict(DEFAULT_PARAMETERS),
            "fairness": dict(DEFAULT_FAIRNESS),
        },
        "execution_policy": {
            "local_only": True,
            "network_enabled": backend_gate.network_enabled,
            "operator_model_backend_id": backend_gate.ready_backend_id,
            "accepts_credentials": False,
            "accepts_commands": False,
            "queue_limit": MAX_RUNS,
            "prepared_data_root_configured": prepared_data_root is not None,
            "verified_prepared_cells": sorted(prepared_benchmarks),
            "verified_production_data_cells": sorted(prepared_production_cells),
        },
    }


def _contains_secret(value: Any, path: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            rendered = str(key)
            child_path = f"{path}.{rendered}" if path else rendered
            if _SECRET_KEY.search(rendered):
                return child_path
            found = _contains_secret(item, child_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _contains_secret(item, f"{path}[{index}]")
            if found:
                return found
    return None


def _issue(path: str, code: str, message: str) -> dict[str, str]:
    return {"path": path, "code": code, "message": message}


def _number(
    value: Any,
    *,
    path: str,
    minimum: float,
    maximum: float,
    integer: bool,
    errors: list[dict[str, str]],
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(_issue(path, "type", "must be a number"))
        return int(minimum) if integer else minimum
    converted: int | float = int(value) if integer else float(value)
    if integer and converted != value:
        errors.append(_issue(path, "integer", "must be an integer"))
    if not math.isfinite(float(converted)) or not minimum <= converted <= maximum:
        errors.append(_issue(path, "range", f"must be between {minimum:g} and {maximum:g}"))
    return converted


def validate_config(
    raw: Any,
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, normalize, and explain one Studio run configuration."""

    if not isinstance(raw, dict):
        return {
            "valid": False,
            "errors": [_issue("$", "type", "configuration must be a JSON object")],
            "warnings": [],
            "normalized": None,
        }
    secret_path = _contains_secret(raw)
    if secret_path:
        raise StudioSecurityError(
            f"credentials are not accepted by Studio; forbidden field: {secret_path}"
        )

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    allowed_top = {
        "algorithm_id",
        "release_id",
        "backend_id",
        "dataset_id",
        "parameters",
        "fairness",
    }
    for key in sorted(set(raw) - allowed_top):
        errors.append(_issue(key, "unknown_field", "field is not accepted"))

    def text_field(name: str, default: str) -> str:
        value = raw.get(name, default)
        if not isinstance(value, str) or not value.strip():
            errors.append(_issue(name, "type", "must be a non-empty string"))
            return default
        if len(value) > 128:
            errors.append(_issue(name, "length", "must be at most 128 characters"))
        return value

    algorithm_id = text_field("algorithm_id", "rewrite.query2doc")
    release_id = text_field("release_id", "fixture.query2doc.cached")
    backend_id = text_field("backend_id", "offline-replay")
    dataset_id = text_field("dataset_id", "studio.tiny_general")

    params_raw = raw.get("parameters", {})
    if not isinstance(params_raw, dict):
        errors.append(_issue("parameters", "type", "must be an object"))
        params_raw = {}
    for key in sorted(set(params_raw) - set(DEFAULT_PARAMETERS)):
        errors.append(_issue(f"parameters.{key}", "unknown_field", "parameter is not accepted"))
    parameters = {
        "seed": _number(
            params_raw.get("seed", DEFAULT_PARAMETERS["seed"]),
            path="parameters.seed",
            minimum=0,
            maximum=2_147_483_647,
            integer=True,
            errors=errors,
        ),
        "sample_limit": _number(
            params_raw.get("sample_limit", DEFAULT_PARAMETERS["sample_limit"]),
            path="parameters.sample_limit",
            minimum=1,
            maximum=len(TINY_EXAMPLES),
            integer=True,
            errors=errors,
        ),
        "top_k": _number(
            params_raw.get("top_k", DEFAULT_PARAMETERS["top_k"]),
            path="parameters.top_k",
            minimum=1,
            maximum=10,
            integer=True,
            errors=errors,
        ),
        "temperature": _number(
            params_raw.get("temperature", DEFAULT_PARAMETERS["temperature"]),
            path="parameters.temperature",
            minimum=0,
            maximum=2,
            integer=False,
            errors=errors,
        ),
        "soft_word_budget": _number(
            params_raw.get("soft_word_budget", DEFAULT_PARAMETERS["soft_word_budget"]),
            path="parameters.soft_word_budget",
            minimum=1,
            maximum=4_096,
            integer=True,
            errors=errors,
        ),
        "clarification_threshold": _number(
            params_raw.get(
                "clarification_threshold", DEFAULT_PARAMETERS["clarification_threshold"]
            ),
            path="parameters.clarification_threshold",
            minimum=0,
            maximum=1,
            integer=False,
            errors=errors,
        ),
    }

    fairness_raw = raw.get("fairness", {})
    if not isinstance(fairness_raw, dict):
        errors.append(_issue("fairness", "type", "must be an object"))
        fairness_raw = {}
    for key in sorted(set(fairness_raw) - set(DEFAULT_FAIRNESS)):
        errors.append(_issue(f"fairness.{key}", "unknown_field", "setting is not accepted"))
    fairness: dict[str, bool] = {}
    for key, default in DEFAULT_FAIRNESS.items():
        value = fairness_raw.get(key, default)
        if not isinstance(value, bool):
            errors.append(_issue(f"fairness.{key}", "type", "must be true or false"))
            value = default
        fairness[key] = value
        if not value:
            warnings.append(
                _issue(
                    f"fairness.{key}",
                    "exploratory_only",
                    "disabled control makes this run exploratory, not confirmatory",
                )
            )

    active_catalog = catalog or build_catalog()
    profile = RUNTIME_PROFILES.get(release_id)
    if profile is None:
        errors.append(
            _issue(
                "release_id",
                "not_executable",
                "release is visible in the registry but lacks a safe local Studio constructor",
            )
        )
    else:
        if profile.algorithm_id != algorithm_id:
            errors.append(
                _issue(
                    "release_id",
                    "algorithm_release_mismatch",
                    f"release belongs to {profile.algorithm_id}",
                )
            )
        if backend_id not in profile.backend_ids:
            errors.append(
                _issue(
                    "backend_id",
                    "backend_incompatible",
                    f"release supports: {', '.join(profile.backend_ids)}",
                )
            )

    backend = next(
        (item for item in active_catalog["backends"] if item["id"] == backend_id),
        None,
    )
    if backend is None:
        errors.append(_issue("backend_id", "unknown_backend", "backend is not registered"))
    elif not backend["ready"]:
        errors.append(
            _issue(
                "backend_id",
                "backend_not_ready",
                str(backend["description"]),
            )
        )
    elif backend_id == "transformers-local" and parameters["temperature"] != 0:
        errors.append(
            _issue(
                "parameters.temperature",
                "backend_incompatible",
                "local Transformers Studio runs require deterministic temperature=0",
            )
        )

    dataset = next(
        (item for item in active_catalog["datasets"] if item["id"] == dataset_id),
        None,
    )
    if dataset is None:
        errors.append(_issue("dataset_id", "unknown_dataset", "dataset is not registered"))
    elif not dataset["ready"]:
        errors.append(_issue("dataset_id", "dataset_not_ready", str(dataset["reason"])))
    elif profile is not None and profile.family not in {"baseline", *dataset["families"]}:
        errors.append(
            _issue(
                "dataset_id",
                "family_mismatch",
                f"dataset does not declare support for the {profile.family} family",
            )
        )
    elif release_id not in dataset.get("studio_release_ids", []):
        supported = dataset.get("studio_release_ids", [])
        errors.append(
            _issue(
                "dataset_id",
                "dataset_release_incompatible",
                (
                    f"this prepared Studio adapter currently supports: {', '.join(supported)}"
                    if supported
                    else "this dataset has no executable Studio release"
                ),
            )
        )

    if not fairness["paired_baseline"]:
        warnings.append(
            _issue(
                "fairness.paired_baseline",
                "result_scope",
                "Studio still records a diagnostic baseline, but it is not a confirmatory pair",
            )
        )

    normalized = {
        "algorithm_id": algorithm_id,
        "release_id": release_id,
        "backend_id": backend_id,
        "dataset_id": dataset_id,
        "parameters": parameters,
        "fairness": fairness,
    }
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized,
        "compatibility": {
            "profile_ready": profile is not None,
            "backend_ready": bool(backend and backend["ready"]),
            "dataset_data_ready": bool(dataset and dataset.get("data_ready")),
            "dataset_execution_ready": bool(dataset and dataset.get("execution_ready")),
            "dataset_ready": bool(dataset and dataset["ready"]),
            "confirmatory": not warnings,
        },
    }


class _NoOpIntervention:
    method_id = "baseline"
    release_id = "baseline.no_op"
    family = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        return InputPatch.no_op(
            state,
            method_id=self.method_id,
            release_id=self.release_id,
            rationale="paired baseline selected in Studio",
        )


def _udcs_replay_text() -> str:
    return (
        "<think>The fixture query is sufficiently specified.</think>"
        "<u_request>0.08</u_request>"
        "<u_request_explanation>No material ambiguity is present.</u_request_explanation>"
        "<action>answer_directly</action>"
        "<confidence>0.96</confidence>"
        "<explanation>Proceed without asking a clarification question.</explanation>"
    )


def _intervention_for(
    release_id: str,
    example: StudioExample,
    parameters: dict[str, int | float],
) -> Any:
    if release_id == "baseline.no_op":
        return _NoOpIntervention()
    if release_id == "fixture.query2doc.cached":
        cache = Query2DocCache(
            (
                OfficialQuery2DocRow(
                    query_id=example.example_id,
                    query=example.query,
                    pseudo_doc=example.pseudo_doc,
                    split="studio",
                ),
            )
        )
        return Query2DocFixtureCached(cache, split="studio")
    if release_id == "plan.plan_and_solve.ps_prompt":
        return PlanAndSolvePSPrompt()
    if release_id == "plan.plan_and_budget.native_cached_soft_budget":
        record = CachedPlanBudgetRecord(
            record_id=example.example_id,
            query=example.query,
            steps=("Identify the required fact.", "Verify and state the final answer."),
            credits=(45.0, 55.0),
            level=1,
        )
        return PlanAndBudgetCachedSoftBudget(
            {example.example_id: record},
            total_word_budget=int(parameters["soft_word_budget"]),
        )
    if release_id == "clarify.udcs.repo_native_prompt_route":
        return UDCSPromptNativePolicy(
            clarification_threshold=float(parameters["clarification_threshold"]),
            temperature=float(parameters["temperature"]),
        )
    raise StudioValidationError(f"unsupported Studio release: {release_id}")


def _runtime_backend_for(release_id: str) -> ReplayBackend | None:
    if release_id == "clarify.udcs.repo_native_prompt_route":
        return ReplayBackend((_udcs_replay_text(),), model_id="studio-replay-v1")
    return None


def _token_count(text: str) -> int:
    return len(_TOKEN.findall(text))


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _operator_safe_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Remove provider error text while retaining typed observability evidence."""

    public = copy.deepcopy(dict(snapshot))
    failure = public.get("failure")
    if isinstance(failure, dict) and failure:
        failure["message"] = "operator-owned model backend call failed"
    events = public.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if "model_id" in payload:
                payload["model_id"] = "<operator-owned-model>"
            if event.get("kind") == "failure_recorded":
                payload["message"] = "operator-owned model backend call failed"
    return public


def execute_config(
    config: dict[str, Any],
    *,
    prepared_data_root: str | Path | None = None,
    model_backend_gate: StudioModelBackendGate | None = None,
) -> dict[str, Any]:
    """Execute a normalized configuration through real library contracts."""

    sample_limit = int(config["parameters"]["sample_limit"])
    dataset_id = str(config["dataset_id"])
    if dataset_id == "studio.tiny_general":
        examples = TINY_EXAMPLES[:sample_limit]
        documents = TINY_DOCUMENTS
        dataset_revision = "studio-tiny-v1"
        dataset_audit: dict[str, Any] = {
            "cell_id": dataset_id,
            "kind": "bundled_synthetic_fixture",
            "gold_separated": True,
            "selection_rule": "bundled_order_first_n",
            "selected_examples": len(examples),
        }
        result_scope = "synthetic_smoke_not_benchmark_claim"
        primary_metric = "top1_accuracy"
    elif dataset_id == "beir.arguana":
        if config["release_id"] != "baseline.no_op":
            raise StudioValidationError(
                "the verified ArguAna Studio adapter currently supports baseline.no_op only"
            )
        if prepared_data_root is None:
            raise StudioValidationError(
                "beir.arguana requires a server-configured verified prepared-data root"
            )
        prepared_dataset = load_verified_studio_dataset(
            cell_id=dataset_id,
            prepared_root=prepared_data_root,
            sample_limit=sample_limit,
        )
        examples = tuple(
            StudioExample(
                example_id=example.example_id,
                query=example.query,
                relevant_document_ids=example.relevant_document_ids,
                instruction=example.instruction,
                visible_metadata=example.visible_metadata,
            )
            for example in prepared_dataset.examples
        )
        documents = prepared_dataset.documents
        dataset_revision = f"{prepared_dataset.asset_id}:{prepared_dataset.source_sha256}"
        dataset_audit = prepared_dataset.audit_payload()
        dataset_audit["kind"] = "checksum_verified_prepared_smoke"
        result_scope = "verified_prepared_bounded_smoke_not_full_benchmark_claim"
        primary_metric = "top1_relevance_diagnostic"
    else:
        raise StudioValidationError(f"dataset has no Studio execution adapter: {dataset_id}")

    retriever = InMemoryBM25Retriever(documents)
    retriever_fingerprint_before = dict(retriever.fingerprint())
    runner = InterventionRunner()
    seed = int(config["parameters"]["seed"])
    top_k = int(config["parameters"]["top_k"])
    release_id = str(config["release_id"])
    backend_id = str(config["backend_id"])
    operator_runtime: StudioModelRuntime | None = None
    if backend_id != "offline-replay":
        if model_backend_gate is None:
            raise StudioModelBackendUnavailable(
                "the selected operator-owned model backend is unavailable"
            )
        operator_runtime = model_backend_gate.build(backend_id, seed=seed)

    started = time.perf_counter()
    baseline_scores: list[float] = []
    treatment_scores: list[float] = []
    intent_recalls: list[float] = []
    jaccards: list[float] = []
    compression: list[float] = []
    latencies: list[float] = []
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    core_usage = ResourceUsage()
    core_durations: list[float] = []
    core_snapshots: list[dict[str, Any]] = []

    for index, example in enumerate(examples):
        sample_started = time.perf_counter()
        state = FrozenAgentState(
            task_id=example.example_id,
            query=example.query,
            instruction=example.instruction,
            metadata={"dataset_id": dataset_id, **dict(example.visible_metadata)},
        )
        baseline_results = retriever.retrieve(state.query, top_k=top_k)
        baseline_top = baseline_results[0].document.doc_id if baseline_results else None
        baseline_score = float(baseline_top in example.relevant_document_ids)
        monitor_ids_before = {snapshot.run_id for snapshot in runner.monitor.list_snapshots()}
        try:
            intervention = _intervention_for(release_id, example, config["parameters"])
            runtime_backend = (
                operator_runtime.backend
                if operator_runtime is not None
                else _runtime_backend_for(release_id)
            )
            runtime_config = (
                dict(operator_runtime.run_config) if operator_runtime is not None else {}
            )
            runtime_config["studio"] = {
                "enabled": True,
                "backend_id": backend_id,
                "fairness": config["fairness"],
            }
            context = InterventionContext(
                backend=runtime_backend,
                seed=seed + index,
                run_config=runtime_config,
                intent_contract=(
                    operator_runtime.intent_contract if operator_runtime is not None else None
                ),
                pricing_catalog=(
                    operator_runtime.pricing_catalog if operator_runtime is not None else None
                ),
                strict_intent=(
                    operator_runtime.strict_intent if operator_runtime is not None else True
                ),
            )
            record = runner.run(intervention, state, context)
            monitor_snapshot = runner.monitor.snapshot(record.monitor_run_id).as_dict()
            if operator_runtime is not None:
                monitor_snapshot = _operator_safe_snapshot(monitor_snapshot)
            core_snapshots.append(monitor_snapshot)
            core_usage = core_usage + record.usage
            if record.duration_ms is not None:
                core_durations.append(record.duration_ms)
            transformed = record.patch.apply(state)
            treatment_results = retriever.retrieve(transformed.query, top_k=top_k)
            treatment_top = treatment_results[0].document.doc_id if treatment_results else None
            treatment_score = float(treatment_top in example.relevant_document_ids)
            latency_ms = (time.perf_counter() - sample_started) * 1_000
            latencies.append(latency_ms)
            baseline_scores.append(baseline_score)
            treatment_scores.append(treatment_score)
            intent_recalls.append(intent_token_recall(state.query, transformed.query))
            jaccards.append(token_jaccard(state.query, transformed.query))
            compression.append(compression_ratio(state.query, transformed.query))
            total_input_tokens += _token_count(f"{state.query} {state.instruction}")
            total_output_tokens += _token_count(f"{transformed.query} {transformed.instruction}")
            samples.append(
                {
                    "example_id": example.example_id,
                    "status": record.status.value,
                    "original_query": state.query,
                    "transformed_query": transformed.query,
                    "transformed_instruction": transformed.instruction,
                    "target_document_id": example.target_doc_id,
                    "relevant_document_ids": list(example.relevant_document_ids),
                    "baseline_top_document_id": baseline_top,
                    "treatment_top_document_id": treatment_top,
                    "baseline_score": baseline_score,
                    "treatment_score": treatment_score,
                    "effect": treatment_score - baseline_score,
                    "latency_ms": round(latency_ms, 3),
                    "patch_hash": record.patch.patch_hash,
                    "input_state_hash": record.input_state_hash,
                    "output_state_hash": record.output_state_hash,
                    "trace": record.patch.trace.as_dict(),
                    "observability": {
                        "monitor_run_id": record.monitor_run_id,
                        "duration_ms": record.duration_ms,
                        "usage": record.usage.as_dict(),
                        "intent_check": (
                            to_jsonable(record.intent_check)
                            if record.intent_check is not None
                            else None
                        ),
                        "snapshot": monitor_snapshot,
                    },
                }
            )
        except Exception as exc:  # Boundary records failures rather than hiding them.
            latency_ms = (time.perf_counter() - sample_started) * 1_000
            latencies.append(latency_ms)
            new_snapshots = [
                (
                    _operator_safe_snapshot(snapshot.as_dict())
                    if operator_runtime is not None
                    else snapshot.as_dict()
                )
                for snapshot in runner.monitor.list_snapshots()
                if snapshot.run_id not in monitor_ids_before
            ]
            if new_snapshots:
                core_snapshots.extend(new_snapshots)
                failure_usage = next(
                    (
                        snapshot.usage
                        for snapshot in runner.monitor.list_snapshots()
                        if snapshot.run_id == new_snapshots[-1]["run_id"]
                    ),
                    ResourceUsage(),
                )
                core_usage = core_usage + failure_usage
                duration = new_snapshots[-1].get("duration_ms")
                if isinstance(duration, (int, float)):
                    core_durations.append(float(duration))
            public_failure_message = (
                "operator-owned model backend call failed"
                if operator_runtime is not None
                else str(exc)[:MAX_TEXT_LENGTH]
            )
            failures.append(
                {
                    "example_id": example.example_id,
                    "category": type(exc).__name__,
                    "message": public_failure_message,
                    "latency_ms": round(latency_ms, 3),
                    "observability_snapshot": new_snapshots[-1] if new_snapshots else None,
                }
            )

    elapsed_ms = (time.perf_counter() - started) * 1_000
    retriever_fingerprint_after = dict(retriever.fingerprint())
    fingerprint_drift = retriever_fingerprint_before != retriever_fingerprint_after
    if config["fairness"]["fail_on_fingerprint_drift"] and fingerprint_drift:
        raise RuntimeError("retriever fingerprint drifted during a paired run")

    quality: dict[str, Any]
    if baseline_scores:
        paired = summarize_pairs(tuple(baseline_scores), tuple(treatment_scores))
        quality = {
            "primary_metric": primary_metric,
            "baseline": fmean(baseline_scores),
            "treatment": fmean(treatment_scores),
            "mean_effect": paired.mean_effect,
            "win_rate": paired.win_rate,
            "tie_rate": paired.tie_rate,
            "harm_rate": paired.harm_rate,
            "intent_token_recall": fmean(intent_recalls),
            "token_jaccard": fmean(jaccards),
            "compression_ratio": fmean(compression),
        }
    else:
        quality = {
            "primary_metric": primary_metric,
            "baseline": None,
            "treatment": None,
            "mean_effect": None,
            "win_rate": None,
            "tie_rate": None,
            "harm_rate": None,
            "intent_token_recall": None,
            "token_jaccard": None,
            "compression_ratio": None,
        }

    if operator_runtime is None:
        backend_audit: Mapping[str, Any] = {
            "backend_id": "offline-replay",
            "backend": "replay",
            "model_id": "studio-replay-v1",
            "network": False,
        }
        execution_mode = "offline_replay"
        cost_basis = "offline replay; no billable provider calls"
    else:
        backend_audit = to_jsonable(operator_runtime.public_fingerprint)
        execution_mode = f"operator_{backend_id.replace('-', '_')}"
        cost_basis = (
            "provider/runtime usage reported by the operator-owned backend; "
            "unpriced calls remain explicit"
        )

    return {
        "execution_mode": execution_mode,
        "result_scope": result_scope,
        "summary": {
            "examples_requested": len(examples),
            "examples_succeeded": len(samples),
            "examples_failed": len(failures),
            "quality": quality,
            "usage": {
                "model_calls": core_usage.model_calls,
                "optimizer_retrieval_calls": core_usage.retrieval_calls,
                "tool_calls": core_usage.tool_calls,
                "embedding_calls": core_usage.embedding_calls,
                "unpriced_model_calls": core_usage.unpriced_model_calls,
                "evaluation_retrieval_calls": len(examples) + len(samples),
                "core_input_tokens": core_usage.input_tokens,
                "core_output_tokens": core_usage.output_tokens,
                "input_tokens_estimate": total_input_tokens,
                "output_tokens_estimate": total_output_tokens,
                "token_estimator": "unicode_word_v1_not_provider_billing_tokens",
                "provider_cost_usd": core_usage.cost_usd,
                "compute_cost_usd": core_usage.compute_cost_usd,
                "cost_usd": core_usage.total_cost_usd,
                "cpu_seconds": core_usage.cpu_seconds,
                "accelerator_seconds": core_usage.accelerator_seconds,
                "cost_basis": cost_basis,
            },
            "timing": {
                "wall_ms": round(elapsed_ms, 3),
                "mean_sample_ms": round(fmean(latencies), 3) if latencies else 0.0,
                "median_sample_ms": round(median(latencies), 3) if latencies else 0.0,
                "p95_sample_ms": round(_percentile_95(latencies), 3) if latencies else 0.0,
                "core_intervention_ms": round(sum(core_durations), 3),
                "core_mean_intervention_ms": (
                    round(fmean(core_durations), 3) if core_durations else 0.0
                ),
            },
            "failure_count": len(failures),
        },
        "metric_audit": {
            "primary": {
                "name": primary_metric,
                "owner": "AgentInputLab Studio",
                "role": "bounded retrieval diagnostic",
                "definition": "fraction of examples whose rank-1 document has positive relevance",
            },
            "intent_diagnostics": [
                "intent_token_recall",
                "token_jaccard",
                "compression_ratio",
            ],
            "declared_benchmark_metrics": (
                ["ndcg_at_10", "recall"] if dataset_id == "beir.arguana" else []
            ),
            "official_benchmark_metrics_computed": False,
            "scope_note": (
                "This bounded Studio smoke validates data, retrieval, pairing, and telemetry. "
                "It is not the full official benchmark evaluation."
            ),
        },
        "fairness_audit": {
            "settings": copy.deepcopy(config["fairness"]),
            "seed": seed,
            "retriever_before": retriever_fingerprint_before,
            "retriever_after": retriever_fingerprint_after,
            "fingerprint_drift": fingerprint_drift,
            "backend": backend_audit,
            "dataset_revision": dataset_revision,
            "dataset": dataset_audit,
        },
        "observability_audit": {
            "source": "agentinputlab.observability.RunMonitor + runner.RunRecord",
            "aggregate_usage": core_usage.as_dict(),
            "monitor_run_count": len(core_snapshots),
            "snapshots": core_snapshots,
        },
        "samples": samples,
        "failures": failures,
    }


class RunManager:
    """Bounded in-memory run queue with immutable JSON-facing snapshots."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_runs: int = MAX_RUNS,
        prepared_data_root: str | Path | None = None,
        model_backend_gate: StudioModelBackendGate | None = None,
    ) -> None:
        if max_workers < 1 or max_workers > 4:
            raise ValueError("Studio max_workers must be between 1 and 4")
        if max_runs < 1:
            raise ValueError("Studio max_runs must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agentinputlab-studio",
        )
        self._max_runs = max_runs
        self._prepared_data_root = (
            Path(prepared_data_root).expanduser().resolve()
            if prepared_data_root is not None
            else None
        )
        self._model_backend_gate = (
            model_backend_gate or StudioModelBackendGate.from_operator_config(None, None)
        )
        self._catalog = build_catalog(
            self._prepared_data_root,
            model_backend_gate=self._model_backend_gate,
        )
        self._lock = threading.RLock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def submit(self, config: Any) -> dict[str, Any]:
        validation = validate_config(config, catalog=self._catalog)
        if not validation["valid"]:
            raise StudioValidationError("configuration did not pass validation")
        normalized = copy.deepcopy(validation["normalized"])
        with self._lock:
            active = sum(run["status"] in {"queued", "running"} for run in self._runs.values())
            if active >= self._max_runs:
                raise StudioQueueFullError("local Studio queue is full")
            while len(self._runs) >= self._max_runs:
                oldest_terminal = next(
                    (
                        stored_id
                        for stored_id in reversed(self._order)
                        if self._runs[stored_id]["status"] not in {"queued", "running"}
                    ),
                    None,
                )
                if oldest_terminal is None:
                    raise StudioQueueFullError("local Studio queue is full")
                self._order.remove(oldest_terminal)
                del self._runs[oldest_terminal]
            run_id = uuid.uuid4().hex
            record = {
                "id": run_id,
                "status": "queued",
                "created_at": _utc_now(),
                "started_at": None,
                "finished_at": None,
                "config": normalized,
                "validation": validation,
                "result": None,
                "error": None,
                "events": [
                    {"at": _utc_now(), "state": "queued", "message": "Run accepted locally"}
                ],
            }
            self._runs[run_id] = record
            self._order.insert(0, run_id)
            self._executor.submit(self._execute, run_id)
            return self._snapshot(record, compact=True)

    def _execute(self, run_id: str) -> None:
        with self._lock:
            record = self._runs[run_id]
            record["status"] = "running"
            record["started_at"] = _utc_now()
            record["events"].append(
                {"at": _utc_now(), "state": "running", "message": "Worker started"}
            )
            config = copy.deepcopy(record["config"])
        try:
            result = execute_config(
                config,
                prepared_data_root=self._prepared_data_root,
                model_backend_gate=self._model_backend_gate,
            )
        except Exception as exc:  # Queue boundary records terminal failure.
            if isinstance(exc, StudioModelBackendUnavailable):
                _LOGGER.exception("Studio operator model backend execution failed")
                public_category = "StudioModelBackendUnavailable"
                public_message = (
                    "operator-owned model backend is unavailable or failed re-attestation"
                )
            elif isinstance(exc, (ArtifactError, OSError)):
                _LOGGER.exception("Studio prepared benchmark execution failed")
                public_category = "PreparedBenchmarkUnavailable"
                public_message = (
                    "prepared benchmark is unavailable or failed integrity verification"
                )
            else:
                public_category = type(exc).__name__
                public_message = str(exc)[:MAX_TEXT_LENGTH]
            with self._lock:
                record = self._runs[run_id]
                record["status"] = "failed"
                record["finished_at"] = _utc_now()
                record["error"] = {
                    "category": public_category,
                    "message": public_message,
                }
                record["events"].append(
                    {"at": _utc_now(), "state": "failed", "message": public_category}
                )
            return
        with self._lock:
            record = self._runs[run_id]
            record["status"] = (
                "completed_with_failures" if result["summary"]["examples_failed"] else "completed"
            )
            record["finished_at"] = _utc_now()
            record["result"] = result
            record["events"].append(
                {"at": _utc_now(), "state": record["status"], "message": "Run finalized"}
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._snapshot(self._runs[run_id], compact=True) for run_id in self._order]

    @property
    def catalog(self) -> dict[str, Any]:
        return copy.deepcopy(self._catalog)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            return self._snapshot(record, compact=False) if record is not None else None

    def wait(self, run_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record["status"] not in {"queued", "running"}:
                return record
            time.sleep(0.01)
        raise TimeoutError(f"run {run_id} did not finish within {timeout} seconds")

    @staticmethod
    def _snapshot(record: dict[str, Any], *, compact: bool) -> dict[str, Any]:
        value = copy.deepcopy(record)
        if compact and value.get("result"):
            value["result"] = {"summary": value["result"]["summary"]}
        return value

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


__all__ = [
    "MAX_RUNS",
    "RUNTIME_PROFILES",
    "RunManager",
    "StudioQueueFullError",
    "StudioSecurityError",
    "StudioValidationError",
    "build_catalog",
    "execute_config",
    "validate_config",
]
