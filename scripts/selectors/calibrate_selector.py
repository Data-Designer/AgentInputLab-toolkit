"""Calibrate a fitted selector from an explicit discovery freeze."""

from __future__ import annotations

import argparse

from _common import (
    add_discovery_identity,
    add_selector_identity,
    discovery_identity,
    dump,
    selector_identity,
)

from agentinputlab.selectors import (
    SelectorArtifactStage,
    SelectorArtifactStore,
    SelectorDiscoveryStore,
    calibrate_selector_bundle,
    load_selector_calibration_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate selector policy gates without running experiments."
    )
    add_discovery_identity(parser)
    parser.add_argument("--selector-store", required=True)
    add_selector_identity(parser, prefix="fitted")
    parser.add_argument("--config", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    discovery = SelectorDiscoveryStore(args.discovery_store).load(discovery_identity(args))
    store = SelectorArtifactStore(args.selector_store)
    fitted_identity = selector_identity(args, prefix="fitted")
    _selector, fitted = store.load(
        fitted_identity,
        require_stage=SelectorArtifactStage.FITTED,
    )
    calibrated, summary = calibrate_selector_bundle(
        fitted,
        discovery,
        revision=args.revision,
        calibration_config=load_selector_calibration_config(args.config),
    )
    identity = store.save(calibrated, previous_identity=fitted_identity)
    dump(
        {
            "schema_version": 1,
            "artifact": "selector_bundle",
            "stage": SelectorArtifactStage.CALIBRATED.value,
            "identity": {
                "selector_id": identity.selector_id,
                "revision": identity.revision,
                "sha256": identity.sha256,
            },
            "path": str(store.path_for(identity)),
            "candidates_considered": summary.candidates_considered,
            "evaluation": dict(summary.receipt["evaluation"]),
            "experiments_executed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
