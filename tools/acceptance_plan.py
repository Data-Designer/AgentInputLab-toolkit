"""Create a full, non-executable acceptance review inventory; never claim passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.benchmarks.production import production_benchmark_specs
from agentinputlab.registry import AlgorithmRegistry


def build_plan(backbones: list[str], *, observed: dict | None = None) -> dict:
    if not backbones or len(backbones) != len(set(backbones)) or any(
        not isinstance(b, str) or not b.strip() or b != b.strip() for b in backbones
    ):
        raise ValueError("backbones must be unique non-empty model identities")
    methods = AlgorithmRegistry.default()
    cells = BenchmarkRegistry.default().list()
    factory_cells = {s.cell_id for s in production_benchmark_specs()}
    observed_groups = {}
    if observed is not None:
        if observed.get("schema") != "agentinputlab.acceptance-observed-evidence.v1":
            raise ValueError("unknown observed receipt inventory schema")
        for group in observed.get("rows", []):
            key = (group.get("release"), group.get("benchmark"), group.get("backbone"))
            if any(not isinstance(value, str) or not value for value in key):
                raise ValueError("observed inventory has an ambiguous identity")
            if key in observed_groups:
                raise ValueError("duplicate observed group; merge receipts without dropping attempts first")
            observed_groups[key] = group
    rows = []
    for method, release in methods.list_releases():
        for cell in cells:
            for backbone in backbones:
                identity = [release.id, cell.cell_id, backbone]
                row_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                evidence = observed_groups.get(tuple(identity))
                rows.append({
                    "id": row_id,
                    "method": method.method_id,
                    "release": release.id,
                    "execution_kind": release.execution_kind.value,
                    "benchmark": cell.cell_id,
                    # The catalog reference can identify benchmark/scorer code
                    # (e.g. a GitHub commit), while the actual data asset has a
                    # separate repository revision and content lock. Never
                    # relabel one namespace as the other or require equal SHAs.
                    "benchmark_reference_revision": cell.source_revision,
                    "dataset_revision": None,
                    "dataset_source_binding": None,
                    "backbone": backbone,
                    "family_declared": method.semantic_family in cell.families,
                    "eligibility_gate": cell.eligibility_gate,
                    "entrypoint": release.entrypoint,
                    "catalog_status": release.status.value,
                    "catalog_availability": release.availability,
                    "delivery_tier": release.delivery_tier,
                    "declared_blockers": list(release.blocking_reasons),
                    "static_factory_listed": cell.cell_id in factory_cells,
                    "applicability": "needs_review",
                    "applicability_evidence": None,
                    "case_ids": [],
                    "budget": None,
                    "model_fingerprint": None,
                    "scorer_binding": None,
                    "installed_artifact_hash": None,
                    "data_acquisition_review": None,
                    "ready_for_dispatch": False,
                    "real_acceptance": (
                        "observed_terminals_review_required" if evidence else "not_run"
                    ),
                    # Keep exact release/backbone identities and all original
                    # attempts, including failures and dependency-only receipts.
                    # A source-mounted full worker does not certify a common-
                    # support component (or vice versa) through family aliases.
                    "observed_receipts": evidence,
                })
    candidate_keys = {(r["release"], r["benchmark"], r["backbone"]) for r in rows}
    return {
        "schema_version": 1,
        "purpose": "acceptance_review_not_dispatch",
        "revision_contract": {
            "benchmark_reference_revision": "catalog benchmark/scorer reference; repository-specific",
            "dataset_revision": "unbound until an actual source asset is reviewed",
            "dataset_source_binding_requires": ["repository", "revision", "content_sha256"],
            "cross_namespace_revision_equality_required": False,
        },
        "data_acquisition_policy": {
            "preferred_modes": ["reuse_existing_prepared_assets", "bounded_source_sample"],
            "full_corpus_download_allowed": False,
            "full_corpus_index_build_allowed": False,
            "download_full_corpus_then_sample_allowed": False,
            "unavailable_bounded_source_action": "defer",
            "unknown_size_action": "defer_or_use_bounded_source_interface",
            "required_storage_budget_components": [
                "source_bytes", "extracted_bytes", "normalized_bytes",
                "index_bytes", "temporary_and_cache_bytes",
            ],
            "destination_free_space_check_required": True,
            "sample_selection_before_model_outcomes_required": True,
            "sampled_compatibility_is_full_benchmark_evaluation": False,
            "shared_research_assets_cleanup_allowed": False,
        },
        "method_count": len(methods.list_methods()),
        "release_count": len(methods.list_releases()),
        "benchmark_count": len(cells),
        "backbones": backbones,
        "candidate_count": len(rows),
        "candidate_kinds": dict(Counter(r["execution_kind"] for r in rows)),
        "dispatchable_count": 0,
        "real_passed_count": 0,
        "candidates_with_observed_receipts": sum(r["observed_receipts"] is not None for r in rows),
        "baseline_observed_groups": [group for key, group in observed_groups.items()
                                     if key[0] == "baseline.no_op"],
        "unmatched_observed_groups": [group for key, group in observed_groups.items()
                                      if key not in candidate_keys and key[0] != "baseline.no_op"],
        "observed_warnings": observed.get("warnings", []) if observed else [],
        "publication_ready": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed", type=Path, help="saved receipt inventory; never reruns model calls")
    args = parser.parse_args()
    output = args.output.resolve()
    for source_root in Path(__file__).resolve().parents:
        if (source_root / ".git").exists():
            if output == source_root or source_root in output.parents:
                parser.error("save acceptance inventories outside the public source tree")
            break
    observed = json.loads(args.observed.read_text()) if args.observed else None
    result = build_plan(args.backbone, observed=observed)
    if args.observed:
        result["observed_inventory"] = {
            "path": str(args.observed.resolve()),
            "sha256": hashlib.sha256(args.observed.read_bytes()).hexdigest(),
        }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {
        "rows", "baseline_observed_groups", "unmatched_observed_groups", "observed_warnings"
    }}))


if __name__ == "__main__":
    main()
