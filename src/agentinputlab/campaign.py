"""Run-readiness orchestration for the frozen existing-data campaign.

This module deliberately distinguishes a broad benchmark inventory from the
cells that already have accepted source bytes, a production adapter, or a
stateful external bridge.  Campaign preparation is restricted to the latter
set so a convenience command cannot silently expand the research scope.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.benchmarks import (
    BenchmarkRegistry,
    BenchmarkSmokeAssetRegistry,
    PreparedQABenchmarkAdapter,
    PreparedRetrievalBenchmarkAdapter,
    QAPrediction,
    RetrievalPrediction,
    load_prepared_benchmark,
    load_production_benchmark_adapter,
    load_tau3_banking_bundle,
    production_benchmark_specs,
)
from agentinputlab.contracts.errors import AgentInputLabError, ArtifactError, ValidationError
from agentinputlab.experiments import (
    load_adapter_readiness_matrix,
    verify_experiment_run_directory,
)
from agentinputlab.registry import AlgorithmRegistry

CAMPAIGN_SCOPE_PROTOCOL = "agentinputlab.existing-data-campaign-scope.v1"

_STATIC_PRODUCTION_CELLS = (
    "beir.arguana",
    "bright.earth_science",
    "bright.economics",
    "musique.answerable",
    "2wiki.multihop",
    "hotpotqa.distractor",
    "qasper.v0_3_dev",
)
_ACCEPTED_SOURCE_ONLY_CELLS = ("clariq.main",)
_STATEFUL_BRIDGE_CELLS = (
    "tau3.banking_knowledge",
    "appworld_ul.common_support",
    "state_bench.common_support",
    "toolsandbox.common_support",
)
_AUXILIARY_CELLS = ("bfcl.v3.simple_python", "ale.external_transfer")
_FROZEN_INVENTORY_CELLS = (
    "tooltalk.full",
    "tau3.retail",
    "beir.cqadupstack",
    "beir.quora",
    "beir.dbpedia_entity",
    "beir.fever",
    "beir.fiqa_2018",
    "bright.robotics",
    "bright.stackoverflow",
    "trec_dl.2019",
    "trec_dl.2020",
    "longbench.qasper",
    "longbench.multifieldqa_en",
    "longbench.govreport",
    "longbench.qmsum",
    "longbench.passage_retrieval_en",
    "longbench.repobench_p",
)
_DISCOVERY_TARGETS = MappingProxyType(
    {
        "reformulate": ("beir.arguana", "test"),
        "decompose": ("musique.answerable", "dev"),
        "acquire": ("bright.earth_science", "test"),
        "clarify": ("tau3.banking_knowledge", "discovery_train"),
        "curate": ("qasper.v0_3_dev", "dev"),
        "composite": ("tau3.banking_knowledge", "discovery_train"),
    }
)
_METHOD_PROCESS_DRIVER = "agentinputlab.experiments.drivers.method_process:factory"
_ATOMIC_ARENA_DRIVER = "agentinputlab.experiments.drivers.external_arena:factory"
_OFFICIAL_PROVIDER_IMPLEMENTATIONS = MappingProxyType(
    {
        "tau3.banking_knowledge": {
            "provider_entrypoint": "workers.official_arenas.providers.tau3:factory",
            "resolution_kind": "public_runtime_bootstrap",
            "next_step": (
                "capture checkpoints with scripts/experiments/capture_official_arena.py, "
                "then freeze the common-support config with "
                "scripts/experiments/prepare_official_common_support_config.py"
            ),
        },
        "state_bench.common_support": {
            "provider_entrypoint": "workers.official_arenas.providers.statebench:factory",
            "resolution_kind": "public_runtime_bootstrap",
            "next_step": (
                "capture checkpoints with scripts/experiments/capture_official_arena.py, "
                "then freeze the common-support config with "
                "scripts/experiments/prepare_official_common_support_config.py"
            ),
        },
        "toolsandbox.common_support": {
            "provider_entrypoint": "workers.official_arenas.providers.toolsandbox:factory",
            "resolution_kind": "public_runtime_bootstrap",
            "next_step": (
                "capture checkpoints with scripts/experiments/capture_official_arena.py, "
                "then freeze the common-support config with "
                "scripts/experiments/prepare_official_common_support_config.py"
            ),
        },
        "appworld_ul.common_support": {
            "provider_entrypoint": "workers.official_arenas.providers.appworld_ul:factory",
            "resolution_kind": "official_plugin_binding",
            "next_step": (
                "bind the official appworld_ul.* runtime plugin and its checksum in "
                "prepare_official_common_support_config.py; base AppWorld is not a "
                "scientifically equivalent substitute"
            ),
        },
    }
)


@dataclass(frozen=True, slots=True)
class ExistingDataCampaignScope:
    """Exact, non-expanding data scope for run-readiness work."""

    scope_id: str = "existing-data-main"
    production_static: tuple[str, ...] = _STATIC_PRODUCTION_CELLS
    accepted_source_only: tuple[str, ...] = _ACCEPTED_SOURCE_ONLY_CELLS
    stateful_bridges: tuple[str, ...] = _STATEFUL_BRIDGE_CELLS
    auxiliary: tuple[str, ...] = _AUXILIARY_CELLS
    frozen_inventory: tuple[str, ...] = _FROZEN_INVENTORY_CELLS

    def __post_init__(self) -> None:
        groups = (
            self.production_static,
            self.accepted_source_only,
            self.stateful_bridges,
            self.auxiliary,
            self.frozen_inventory,
        )
        flattened = tuple(cell for group in groups for cell in group)
        if len(flattened) != len(set(flattened)):
            raise ValidationError("existing-data campaign scope contains duplicate cells")
        registered = {cell.cell_id for cell in BenchmarkRegistry.default().list()}
        if set(flattened) != registered:
            missing = sorted(registered - set(flattened))
            extra = sorted(set(flattened) - registered)
            raise ValidationError(
                "existing-data campaign scope differs from the benchmark registry; "
                f"missing={missing}, extra={extra}"
            )
        production = {
            spec.cell_id
            for spec in production_benchmark_specs()
            if spec.adapter_kind in {"qa", "retrieval"}
        }
        if (
            set(self.production_static) != set(_STATIC_PRODUCTION_CELLS)
            or not set(self.production_static) <= production
        ):
            raise ValidationError(
                "existing-data campaign static scope differs from production adapters"
            )
        if self.frozen_inventory != _FROZEN_INVENTORY_CELLS:
            raise ValidationError("existing-data campaign must preserve its inventory-only cells")

    @property
    def active_cells(self) -> tuple[str, ...]:
        """Cells that preparation may touch."""

        return (
            *self.production_static,
            *self.accepted_source_only,
            *self.stateful_bridges,
            *self.auxiliary,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": 1,
                "protocol": CAMPAIGN_SCOPE_PROTOCOL,
                "scope_id": self.scope_id,
                "data_policy": "existing_implementation_only",
                "production_static": self.production_static,
                "accepted_source_only": self.accepted_source_only,
                "stateful_bridges": self.stateful_bridges,
                "auxiliary": self.auxiliary,
                "frozen_inventory": self.frozen_inventory,
            }
        )


def existing_data_campaign_scope() -> ExistingDataCampaignScope:
    """Return the validated campaign scope."""

    return ExistingDataCampaignScope()


def existing_campaign_assets() -> tuple[str, ...]:
    """Return only accepted-source asset IDs inside the frozen active scope."""

    scope = existing_data_campaign_scope()
    return tuple(
        asset.asset_id
        for asset in BenchmarkSmokeAssetRegistry.default().list(include_license_review=True)
        if asset.cell_id in scope.active_cells
    )


def campaign_doctor(
    prepared_root: str | Path,
    *,
    readiness_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Inspect data, method-surface, and stateful-runtime readiness.

    The doctor never downloads data, imports an upstream benchmark runtime, or
    calls a model.  A missing optional readiness catalog remains an explicit
    blocker instead of being interpreted as a ready method matrix.
    """

    scope = existing_data_campaign_scope()
    prepared = Path(prepared_root).expanduser().resolve()
    asset_rows: list[dict[str, Any]] = []
    for asset_id in existing_campaign_assets():
        asset = BenchmarkSmokeAssetRegistry.default().get(asset_id)
        try:
            loaded = load_prepared_benchmark(asset, prepared)
        except (AgentInputLabError, OSError) as exc:
            asset_rows.append(
                {
                    "asset_id": asset_id,
                    "cell_id": asset.cell_id,
                    "prepared": False,
                    "reason": str(exc),
                    "resolution_kind": "registered_asset_acquisition",
                    "next_command": (
                        "agentinputlab campaign prepare --store data/raw "
                        f"--prepared {prepared} --allow-network"
                    ),
                }
            )
        else:
            asset_rows.append(
                {
                    "asset_id": asset_id,
                    "cell_id": asset.cell_id,
                    "prepared": True,
                    "counts": dict(loaded.counts),
                    "prepared_identity_sha256": _prepared_identity(loaded),
                    "resolution_kind": "none",
                    "next_command": None,
                }
            )

    method_summary: dict[str, Any]
    if readiness_path is None:
        method_summary = {
            "method_count": 27,
            "l2_method_paths": 27,
            "discovery_implemented": 0,
            "discovery_blocked": 27,
            "catalog_checked": False,
            "blocker": "provide the checksum-reviewed adapter readiness catalog",
        }
    else:
        methods = AlgorithmRegistry.default().list_methods()
        # Profile kinds are catalog-owned.  Omitting expected_methods retains
        # the loader's strict exact-count/schema checks without duplicating the
        # selected release catalog here.
        readiness = load_adapter_readiness_matrix(readiness_path)
        registered_ids = {method.method_id for method in methods}
        if set(readiness.entries) != registered_ids:
            raise ValidationError(
                "adapter readiness catalog differs from the exact algorithm registry"
            )
        implemented = sum(
            entry.discovery.status == "implemented" for entry in readiness.entries.values()
        )
        method_summary = {
            "method_count": len(methods),
            "l2_method_paths": len(methods),
            "discovery_implemented": implemented,
            "discovery_blocked": len(methods) - implemented,
            "catalog_checked": True,
            "catalog_sha256": readiness.file_sha256,
            "catalog_hash": readiness.catalog_hash,
        }

    static_asset_ids = {
        asset.asset_id
        for asset in BenchmarkSmokeAssetRegistry.default().list(include_license_review=True)
        if asset.cell_id in scope.production_static
    }
    static_assets_ready = all(
        row["prepared"] for row in asset_rows if row["asset_id"] in static_asset_ids
    ) and static_asset_ids == {
        row["asset_id"] for row in asset_rows if row["asset_id"] in static_asset_ids
    }
    all_source_assets_ready = len(asset_rows) == len(existing_campaign_assets()) and all(
        row["prepared"] for row in asset_rows
    )
    stateful = []
    for cell_id in scope.stateful_bridges:
        implementation = _OFFICIAL_PROVIDER_IMPLEMENTATIONS[cell_id]
        stateful.append(
            {
                "cell_id": cell_id,
                "contract_available": True,
                "provider_implemented": True,
                "provider_entrypoint": implementation["provider_entrypoint"],
                # A provider implementation and one operator's materialized
                # runtime are different facts.  The latter becomes ready only
                # after exact checkouts, captures, and model bindings exist.
                "official_atomic_runtime_ready": False,
                "resolution_kind": implementation["resolution_kind"],
                "next_step": implementation["next_step"],
                "missing_runtime_inputs": (
                    "pinned checkout",
                    "captured checkpoint inventory",
                    "protected replay inventory",
                    "model/runtime binding",
                ),
            }
        )
    full_ready = (
        static_assets_ready
        and method_summary["discovery_implemented"] == method_summary["method_count"]
        and all(row["official_atomic_runtime_ready"] for row in stateful)
    )
    return MappingProxyType(
        {
            "schema_version": 1,
            "protocol": "agentinputlab.campaign-doctor.v1",
            "scope": dict(scope.to_dict()),
            "prepared_root": str(prepared),
            "assets": tuple(asset_rows),
            "methods": method_summary,
            "stateful": tuple(stateful),
            "static_data_ready": static_assets_ready,
            "all_accepted_sources_ready": all_source_assets_ready,
            "full_campaign_ready": full_ready,
            "experiments_executed": False,
        }
    )


def campaign_data_smoke(prepared_root: str | Path) -> Mapping[str, Any]:
    """Verify all accepted sources and execute every available static scorer.

    Seven D2 cells expose complete static production adapters and therefore run
    one real evaluator call.  BFCL, ClariQ, and tau3 are also reloaded from
    their checksum-locked prepared bytes.  This command does not launch an
    external model or arena, but it reports the implemented resolution path
    instead of treating an absent operator binding as an absent implementation.
    """

    root = Path(prepared_root).expanduser().resolve()
    static_rows: list[dict[str, Any]] = []
    for spec in production_benchmark_specs():
        if spec.cell_id not in _STATIC_PRODUCTION_CELLS:
            continue
        adapter = load_production_benchmark_adapter(spec.cell_id, root)
        example = adapter.examples(spec.split)[0]
        visible = example.visible_state()
        if isinstance(adapter, PreparedQABenchmarkAdapter):
            metrics = adapter.score(example, QAPrediction(answer=""))
        elif isinstance(adapter, PreparedRetrievalBenchmarkAdapter):
            metrics = adapter.score(example, RetrievalPrediction({}))
        else:  # pragma: no cover - production factory invariant
            raise ArtifactError("unexpected static production adapter type")
        if not metrics or any(not isinstance(value, float) for value in metrics.values()):
            raise ArtifactError(f"{spec.cell_id}: scorer smoke returned invalid metrics")
        static_rows.append(
            {
                "cell_id": spec.cell_id,
                "split": spec.split,
                "example_count": len(adapter.examples(spec.split)),
                "smoke_example_id": visible.task_id,
                "visible_state_hash": visible.state_hash,
                "metric_names": tuple(sorted(metrics)),
                "scorer_executed": True,
            }
        )

    source_rows: list[dict[str, Any]] = []
    registry = BenchmarkSmokeAssetRegistry.default()
    for asset_id in existing_campaign_assets():
        asset = registry.get(asset_id)
        prepared = load_prepared_benchmark(asset, root)
        source_rows.append(
            {
                "asset_id": asset.asset_id,
                "cell_id": asset.cell_id,
                "counts": dict(prepared.counts),
                "prepared_identity_sha256": _prepared_identity(prepared),
                "source_integrity_verified": True,
            }
        )

    tau3 = load_tau3_banking_bundle(root)
    tau3.verify_integrity()
    boundary_rows = (
        {
            "cell_id": "bfcl.v3.simple_python",
            "source_integrity_verified": True,
            "adapter_level": "loader_plus_typed_external_scorer_bridge",
            "scorer_executed": False,
            "resolution_kind": "operator_runtime_binding",
            "next_step": (
                "bind a BFCL model handler to the implemented typed scorer bridge; "
                "the source loader and AST response normalizer are already present"
            ),
        },
        {
            "cell_id": "clariq.main",
            "source_integrity_verified": True,
            "adapter_level": "accepted_source_only",
            "scorer_executed": False,
            "resolution_kind": "official_bundle_acquisition",
            "next_step": (
                "the accepted dev TSV is not a complete official RQ1/RQ2 evaluation bundle"
            ),
        },
        {
            "cell_id": "tau3.banking_knowledge",
            "source_integrity_verified": True,
            "adapter_level": "static_bundle_plus_typed_checkpoint_bridge",
            "task_count": len(tau3.task_ids),
            "document_count": tau3.document_count,
            "scorer_executed": False,
            "provider_implemented": True,
            "provider_entrypoint": _OFFICIAL_PROVIDER_IMPLEMENTATIONS["tau3.banking_knowledge"][
                "provider_entrypoint"
            ],
            "resolution_kind": "public_runtime_bootstrap",
            "next_step": _OFFICIAL_PROVIDER_IMPLEMENTATIONS["tau3.banking_knowledge"]["next_step"],
        },
    )
    return MappingProxyType(
        {
            "schema_version": 1,
            "protocol": "agentinputlab.existing-data-smoke.v1",
            "static_cells": tuple(static_rows),
            "static_cell_count": len(static_rows),
            "source_assets": tuple(source_rows),
            "source_asset_count": len(source_rows),
            "bridge_boundaries": boundary_rows,
            "accepted_source_cells": tuple(sorted({row["cell_id"] for row in source_rows})),
            "frozen_inventory_cells_touched": 0,
            "real_prepared_data": True,
            "model_called": False,
            "experiment_executed": False,
        }
    )


def campaign_method_matrix(
    method_catalog_path: str | Path,
    readiness_path: str | Path,
) -> Mapping[str, Any]:
    """Build the exact 27-method discovery launch matrix.

    This is an execution inventory, not an adapter waiver.  Every row includes
    a copyable preflight command, while ``adapter_ready`` remains tied to the
    reviewed readiness catalog.
    """

    catalog_path = Path(method_catalog_path).expanduser()
    if catalog_path.is_symlink():
        raise ArtifactError("campaign method catalog must not be a symlink")
    catalog_path = catalog_path.resolve()
    try:
        payload = catalog_path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("campaign method catalog is not valid UTF-8 JSON") from exc
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "method_count", "methods"}
        or raw.get("schema_version") != 1
        or raw.get("method_count") != 27
        or not isinstance(raw.get("methods"), list)
        or len(raw["methods"]) != 27
    ):
        raise ValidationError("campaign method catalog schema is invalid")

    registry = AlgorithmRegistry.default()
    readiness = load_adapter_readiness_matrix(readiness_path)
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, item in enumerate(raw["methods"]):
        if not isinstance(item, Mapping) or set(item) != {
            "method_id",
            "family",
            "release_id",
            "profile_kind",
            "upstream_commit",
        }:
            raise ValidationError(f"campaign method catalog row {index} is invalid")
        method_id = item["method_id"]
        family = item["family"]
        release_id = item["release_id"]
        profile_kind = item["profile_kind"]
        upstream_commit = item["upstream_commit"]
        if (
            not all(
                isinstance(value, str) and value.strip()
                for value in (
                    method_id,
                    family,
                    release_id,
                    profile_kind,
                    upstream_commit,
                )
            )
            or method_id in observed
        ):
            raise ValidationError(f"campaign method catalog row {index} identity is invalid")
        observed.add(method_id)
        method, release = registry.get_release(release_id)
        if (
            method.method_id != method_id
            or method.semantic_family.value != family
            or method.upstream.commit != upstream_commit
            or profile_kind not in {mode.value for mode in release.profile_modes}
        ):
            raise ValidationError(
                f"campaign method catalog row {method_id!r} differs from the registry"
            )
        try:
            gate = readiness.entries[method_id]
            target_cell, target_split = _DISCOVERY_TARGETS[family]
        except KeyError as exc:
            raise ValidationError(
                f"campaign method {method_id!r} lacks a readiness or target row"
            ) from exc
        external = family in {"clarify", "composite"}
        driver = _ATOMIC_ARENA_DRIVER if external else _METHOD_PROCESS_DRIVER
        rows.append(
            {
                "method_id": method_id,
                "display_name": method.display_name,
                "family": family,
                "release_id": release_id,
                "profile_kind": profile_kind,
                "upstream_commit": upstream_commit,
                "release_entrypoint_available": release.entrypoint_available,
                "release_availability": release.availability,
                "l2_method_path": True,
                "target": {
                    "cell_id": target_cell,
                    "split": target_split,
                    "external": external,
                },
                "driver": driver,
                "integration_kind": gate.discovery.integration_kind,
                "adapter_ready": gate.discovery.status == "implemented",
                "adapter_blockers": gate.discovery.blockers,
                "implementation_ready": gate.discovery.status == "implemented",
                "runtime_binding_required": True,
                "runtime_resolution": {
                    "checkout": (
                        "agentinputlab campaign bootstrap --root operator/methods "
                        f"--method {method_id} --no-arenas --allow-network"
                    ),
                    "preflight": (
                        f"python scripts/experiments/run_method_discovery.py --method {method_id}"
                    ),
                    "execute": (
                        "python scripts/experiments/run_method_discovery.py "
                        f"--method {method_id} --execute"
                    ),
                },
                "preflight_command": (
                    f"python scripts/experiments/run_method_discovery.py --method {method_id}"
                ),
                "execute_command": (
                    "python scripts/experiments/run_method_discovery.py "
                    f"--method {method_id} --execute"
                ),
            }
        )
    registered = {method.method_id for method in registry.list_methods()}
    if observed != registered or observed != set(readiness.entries):
        raise ValidationError("campaign method catalog does not cover the exact 27-method roster")
    implemented = sum(row["adapter_ready"] for row in rows)
    return MappingProxyType(
        {
            "schema_version": 1,
            "protocol": "agentinputlab.campaign-method-matrix.v1",
            "method_catalog": str(catalog_path),
            "method_catalog_sha256": hashlib.sha256(payload).hexdigest(),
            "readiness_catalog": str(readiness.path),
            "readiness_catalog_sha256": readiness.file_sha256,
            "method_count": len(rows),
            "l2_method_paths": len(rows),
            "integration_implemented": implemented,
            "adapter_ready": implemented,
            "adapter_blocked": len(rows) - implemented,
            "experiments_executed": False,
            "methods": tuple(rows),
        }
    )


def campaign_verify_run(
    run_dir: str | Path,
    *,
    allow_incomplete: bool = False,
) -> Mapping[str, Any]:
    """Verify one append-only run directory under the shared strict verifier."""

    result = verify_experiment_run_directory(run_dir)
    if not allow_incomplete and not (result["complete"] and result["all_succeeded"]):
        raise ArtifactError("campaign run is incomplete or contains failed cells")
    return result


def _prepared_identity(prepared: Any) -> str:
    payload = {
        "asset_id": prepared.asset_id,
        "cell_id": prepared.cell_id,
        "counts": dict(prepared.counts),
        "file_sha256": dict(prepared.file_sha256),
        "processor": prepared.processor,
        "processor_revision": prepared.processor_revision,
        "source_revision": prepared.source_revision,
        "source_sha256": prepared.source_sha256,
        "split": prepared.split,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "CAMPAIGN_SCOPE_PROTOCOL",
    "ExistingDataCampaignScope",
    "campaign_data_smoke",
    "campaign_doctor",
    "campaign_method_matrix",
    "campaign_verify_run",
    "existing_campaign_assets",
    "existing_data_campaign_scope",
]
