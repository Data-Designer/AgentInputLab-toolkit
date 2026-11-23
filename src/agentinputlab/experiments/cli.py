"""Command-line entry point for validating, planning, and explicitly running studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError

from .driver_guard import activate_guarded_driver
from .executor import execute_experiment_plan
from .loader import load_experiment_manifest, preflight_experiment
from .plan import build_experiment_plan


def _dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _preflight(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    manifest = load_experiment_manifest(args.manifest)
    report, units = preflight_experiment(
        manifest,
        driver_entrypoint=args.driver,
        resolve_units=not args.no_resolve_units,
    )
    plan = None
    if report.ready and units:
        plan = build_experiment_plan(manifest, units)
        if args.shard_count is not None:
            plan = plan.shard(index=args.shard_index, count=args.shard_count)
    return manifest, report, plan


def _cmd_validate(args: argparse.Namespace) -> int:
    manifest, report, plan = _preflight(args)
    _dump(
        {
            "manifest": {
                "study_id": manifest.study_id,
                "stage": manifest.stage.value,
                "manifest_hash": manifest.manifest_hash,
            },
            "preflight": report.to_dict(),
            "plan": (
                {
                    "plan_hash": plan.plan_hash,
                    "cell_count": len(plan.cells),
                }
                if plan is not None
                else None
            ),
        }
    )
    return 0 if report.ready else 2


def _cmd_plan(args: argparse.Namespace) -> int:
    _manifest, report, plan = _preflight(args)
    if not report.ready or plan is None:
        _dump({"preflight": report.to_dict(), "plan": None})
        return 2
    _dump(plan.to_dict())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    manifest, report, plan = _preflight(args)
    if not report.ready or plan is None:
        _dump({"preflight": report.to_dict(), "executed": False})
        return 2
    if not args.execute:
        _dump(
            {
                "preflight": report.to_dict(),
                "plan_hash": plan.plan_hash,
                "cell_count": len(plan.cells),
                "executed": False,
                "next_step": "repeat with --execute after reviewing this frozen plan",
            }
        )
        return 0
    effective_driver = args.driver or manifest.driver_entrypoint
    if effective_driver is None:  # pragma: no cover - preflight invariant
        raise ArtifactError("experiment driver is not configured")
    if report.binding_receipt is None:  # pragma: no cover - preflight invariant
        raise ArtifactError("ready preflight did not produce a frozen binding receipt")
    with activate_guarded_driver(
        manifest=manifest,
        operator_entrypoint=effective_driver,
        operator_fingerprint=report.operator_driver_fingerprint,
        binding_receipt=report.binding_receipt,
        parity_contract=report.parity_contract,
    ) as guarded_entrypoint:
        summary = execute_experiment_plan(
            manifest,
            plan,
            driver_entrypoint=guarded_entrypoint,
            output_root=str(Path(args.output_root).expanduser()),
            resume=args.resume,
            fail_fast=args.fail_fast,
        )
    _dump({"preflight": report.to_dict(), "executed": True, "summary": summary.to_dict()})
    return 0 if summary.failed_cells == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentinputlab.experiments.cli",
        description=(
            "Validate and materialize experiments by default; real execution requires --execute."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text, handler in (
        ("validate", "Validate manifest, prerequisites, and exact units.", _cmd_validate),
        ("plan", "Print a stable, optionally sharded experiment plan.", _cmd_plan),
        ("run", "Preflight by default; execute only with --execute.", _cmd_run),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("manifest", help="Path to a strict experiment JSON manifest.")
        child.add_argument(
            "--driver",
            help="Operator driver override using module:zero_argument_factory.",
        )
        child.add_argument(
            "--no-resolve-units",
            action="store_true",
            help="Validate schema and static prerequisites without reading unit inventories.",
        )
        child.add_argument("--shard-count", type=int)
        child.add_argument("--shard-index", type=int, default=0)
        child.set_defaults(handler=handler)
        if command == "run":
            child.add_argument("--execute", action="store_true")
            child.add_argument("--output-root", default="results")
            child.add_argument("--resume", action="store_true")
            child.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.shard_count is None and args.shard_index != 0:
        parser.error("--shard-index requires --shard-count")
    try:
        return int(args.handler(args))
    except (ArtifactError, ValidationError, OSError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
