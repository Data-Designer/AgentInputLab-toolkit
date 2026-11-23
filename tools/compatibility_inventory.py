"""Enumerate every registry combination without pretending to run models."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.registry import AlgorithmRegistry


def inventory(backbones: list[str]) -> dict:
    registry = AlgorithmRegistry.default()
    profiles = []
    for method, release in registry.list_releases():
        item = {"method": method.method_id, "release": release.id,
                "execution_kind": release.execution_kind.value,
                "status": release.status.value, "entrypoint": release.entrypoint,
                "declared_blockers": list(release.blocking_reasons),
                "real_acceptance": "not_run"}
        if release.entrypoint:
            try:
                factory = registry.resolve_entrypoint(release.id)
                item["import_check"] = "passed"
                item["constructor"] = str(inspect.signature(factory))
                item["required_parameters"] = [p.name for p in inspect.signature(factory).parameters.values()
                    if p.default is inspect.Parameter.empty and p.kind not in
                    (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            except Exception as exc:
                item["import_check"] = "failed"
                item["error_type"] = type(exc).__name__
        else:
            item["import_check"] = "no_entrypoint"
        profiles.append(item)
    combinations = []
    for method in registry.list_methods():
        for cell in BenchmarkRegistry.default().list():
            declared = method.semantic_family in cell.families
            for backbone in backbones:
                combinations.append({"method": method.method_id, "benchmark": cell.cell_id,
                    "backbone": backbone, "family_declared": declared,
                    "checkpoint_gate": cell.eligibility_gate,
                    "family_support": cell.family_support,
                    "runtime_compatibility": "unverified" if declared else "family_not_declared",
                    "real_acceptance": "not_run"})
    return {"schema_version": 1, "method_count": len(registry.list_methods()),
            "benchmark_count": len(BenchmarkRegistry.default()), "backbones": backbones,
            "profiles": profiles, "combinations": combinations,
            "note": "A family declaration is necessary, not sufficient. No inference, scorer, assets or official endpoints were exercised."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", action="append", help="User-configured model identity; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.backbone or ["user-configured-backbone"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"methods": result["method_count"], "benchmarks": result["benchmark_count"],
                      "profiles": len(result["profiles"]), "combinations": len(result["combinations"]),
                      "real_acceptance": "not_run"}))


if __name__ == "__main__":
    main()
