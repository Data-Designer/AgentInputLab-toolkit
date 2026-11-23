"""Read actual isolated acceptance receipts without granting compatibility passes.

The report preserves multiple attempts for the same case and lists evidence
gaps explicitly. A scored case is an endpoint, not certification of an entire
dataset, release configuration, training recipe or backbone family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.registry import AlgorithmRegistry

DERIVED_EVIDENCE_ROLES = frozenset({"native_score_export_recovery", "derived_leaf_selection"})


def summarize(root):
    catalog = AlgorithmRegistry.default()
    method_ids = {method.method_id for method in catalog.list_methods()}
    release_methods = {release.id: method.method_id for method, release in catalog.list_releases()}
    # NO_OP is an API baseline, not one of the 27 catalog algorithm cards.
    release_methods["baseline.no_op"] = "baseline.no_op"
    benchmarks = {entry.cell_id for entry in BenchmarkRegistry.default().list()}
    groups = defaultdict(list)
    phases, warnings, fits, helpers = [], [], [], []
    phase_roots = []
    for parent in sorted(root.iterdir()):
        if not parent.is_dir() or "phase" not in parent.name:
            continue
        if (parent / "manifest.json").is_file():
            phase_roots.append(parent)
        else:
            # Domain-sharded batches have one manifest per child, not a fake
            # merged run at the parent. Read each exactly once.
            phase_roots.extend(
                child
                for child in sorted(parent.iterdir())
                if child.is_dir() and (child / "manifest.json").is_file()
            )
    for phase in phase_roots:
        phase_name = phase.relative_to(root).as_posix()
        manifest = phase / "manifest.json"
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text())
        paths = set(phase.glob("terminals/*.json"))
        paths.update(phase.glob("*/terminal.json"))
        paths.update(phase.glob("*/*/terminal.json"))
        phase_statuses = Counter()
        for path in sorted(paths):
            row = json.loads(path.read_text())
            job = row.get("job")
            identity_schema = "nested_job"
            if not isinstance(job, dict) and all(
                key in row for key in ("model", "release", "benchmark", "case_id")
            ):
                # The native ACE frozen-evaluation runner writes a flat DTO.
                job, identity_schema = row, "flat_native_artifact_evaluation"
            if not isinstance(job, dict):
                helpers.append(
                    {
                        "path": str(path),
                        "status": row.get("status"),
                        "automatic_certification": False,
                    }
                )
                continue
            model = job.get("model")
            model = model.get("id") if isinstance(model, dict) else model
            release, benchmark, case = job.get("release"), job.get("benchmark"), job.get("case_id")
            if case is None and benchmark in {
                "tooltalk.full",
                "tau3.banking_knowledge",
                "tau3.retail",
                "state_bench.common_support",
                "toolsandbox.common_support",
            }:
                # The source-reviewed native runners use protected checkpoint IDs.
                case = job.get("checkpoint_id")
                identity_schema = "native_provider_checkpoint"
            if not all(isinstance(v, str) and v for v in (model, release, benchmark, case)):
                warnings.append(
                    {
                        "path": str(path),
                        "reason": "unrecognized explicit job identity; manual source mapping required",
                    }
                )
                continue
            evidence = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "case_id": case,
                "identity_schema": identity_schema,
                "status": row.get("status"),
                "phase": phase_name,
                "observed_backend_call_records": len(row.get("calls", [])),
                "error_type": row.get("error_type"),
                "error": row.get("error"),
                "seed": job.get("seed"),
                "budget": job.get("budget"),
                "input_state_hash": row.get("input_state_hash"),
                "package_location": row.get("package_location"),
                "job_configuration_sha256": hashlib.sha256(
                    json.dumps(job, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest(),
                "declared_release_configuration": {
                    key: value for key, value in job.items() if key not in {
                        "id", "release", "benchmark", "model", "case_id", "checkpoint_id",
                        "split", "seed", "budget",
                    }
                },
                "reported_model_calls": row.get("model_calls"),
                "usage_scope": "reported_terminal_only_not_all_native_environment_or_fit_cost",
                "evidence_role": row.get("evidence_role", "execution_terminal"),
                "reported_additional_case_attempts": row.get("additional_attempts"),
                "original_terminal_path": row.get("original_terminal_path"),
                "original_terminal_sha256": row.get("original_terminal_sha256"),
                "saved_ending_context_sha256": row.get("saved_ending_context_sha256"),
                "native_scorer_sha256": row.get("native_scorer_sha256"),
            }
            groups[(release, benchmark, model)].append(evidence)
            phase_statuses[str(row.get("status"))] += 1
        for path in sorted(phase.glob("*/fit-terminal.json")):
            fit = json.loads(path.read_text())
            artifact_path = path.parent / "artifact.json"
            artifact = json.loads(artifact_path.read_text()) if artifact_path.exists() else {}
            fits.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": fit.get("status"),
                    "model_directory": path.parent.name,
                    "artifact_release": artifact.get("release_id"),
                    "artifact_id": artifact.get("artifact_id"),
                    "scope_requires_separate_recipe_review": True,
                }
            )
        phases.append(
            {
                "phase": phase_name,
                "manifest": str(manifest),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "manifest_job_count": len(data.get("jobs", [])),
                "mapped_terminal_statuses": dict(phase_statuses),
                "complete_receipt_present": (phase / "complete.json").exists(),
                "attention_receipt_present": (phase / "attention.json").exists(),
            }
        )
    rows = []
    for (release, benchmark, model), evidence in sorted(groups.items()):
        if release not in release_methods or benchmark not in benchmarks:
            warnings.append(
                {
                    "release": release,
                    "benchmark": benchmark,
                    "reason": "receipt identity not in current catalog; review alias/source explicitly",
                }
            )
        rows.append(
            {
                "release": release,
                "method": release_methods.get(release),
                "benchmark": benchmark,
                "backbone": model,
                "attempt_count": sum(e["evidence_role"] not in DERIVED_EVIDENCE_ROLES for e in evidence),
                "receipt_count": len(evidence),
                "derived_score_receipts": sum(e["evidence_role"] == "native_score_export_recovery" for e in evidence),
                "derived_analysis_receipts": sum(e["evidence_role"] == "derived_leaf_selection" for e in evidence),
                "distinct_case_count": len({e["case_id"] for e in evidence}),
                "terminal_statuses": dict(Counter(str(e["status"]) for e in evidence)),
                "backend_call_record_count": sum(
                    e["observed_backend_call_records"] for e in evidence
                ),
                "all_applicable_profiles_certified": False,
                "applicability_review": "required_separate_source_bound_review",
                "evidence": evidence,
            }
        )
    seen_benchmarks = {r["benchmark"] for r in rows if r["benchmark"] in benchmarks}
    seen_methods = {r["method"] for r in rows if r["method"] in method_ids}
    return {
        "schema": "agentinputlab.acceptance-observed-evidence.v1",
        "purpose": "observed_receipt_inventory_not_publication_approval",
        "root": str(root),
        "current_catalog_method_count": len(catalog.list_methods()),
        "current_catalog_benchmark_count": len(benchmarks),
        "observed_release_benchmark_backbone_groups": len(rows),
        "mapped_terminal_attempts": sum(r["attempt_count"] for r in rows),
        "mapped_terminal_receipts": sum(r["receipt_count"] for r in rows),
        "attempt_unit": "execution_terminal_including_pre_call_guards_not_physical_model_requests",
        "methods_with_mapped_terminal_evidence": sorted(seen_methods),
        "methods_without_mapped_terminal_evidence": sorted(method_ids - seen_methods),
        "benchmarks_with_mapped_terminal_evidence": sorted(seen_benchmarks),
        "benchmarks_without_mapped_terminal_evidence": sorted(benchmarks - seen_benchmarks),
        "native_fit_receipts": fits,
        "helper_receipts": helpers,
        "phases": phases,
        "warnings": warnings,
        "rows": rows,
        "quality_zero_is_not_implementation_failure": True,
        "failed_attempts_not_dropped": True,
        "old_and_repaired_attempts_not_double_counted_as_unique_cases": True,
        "full_acceptance_complete": False,
        "publication_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            if output == parent or parent in output.parents:
                parser.error("save raw acceptance evidence outside the public Git checkout")
            break
    report = summarize(args.root.resolve())
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"rows", "native_fit_receipts", "helper_receipts", "phases", "warnings"}
            }
        )
    )


if __name__ == "__main__":
    main()
