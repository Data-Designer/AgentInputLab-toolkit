"""Freeze an exact calibrated selector policy."""

from __future__ import annotations

import argparse

from _common import add_selector_identity, dump, selector_identity

from agentinputlab.selectors import (
    SelectorArtifactStage,
    SelectorArtifactStore,
    freeze_selector_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a calibrated selector bundle without running experiments."
    )
    parser.add_argument("--selector-store", required=True)
    add_selector_identity(parser, prefix="calibrated")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    store = SelectorArtifactStore(args.selector_store)
    calibrated_identity = selector_identity(args, prefix="calibrated")
    _selector, calibrated = store.load(
        calibrated_identity,
        require_stage=SelectorArtifactStage.CALIBRATED,
    )
    frozen = freeze_selector_bundle(calibrated, revision=args.revision)
    identity = store.save(frozen, previous_identity=calibrated_identity)
    dump(
        {
            "schema_version": 1,
            "artifact": "selector_bundle",
            "stage": SelectorArtifactStage.FROZEN.value,
            "identity": {
                "selector_id": identity.selector_id,
                "revision": identity.revision,
                "sha256": identity.sha256,
            },
            "path": str(store.path_for(identity)),
            "model_hash": frozen.model_hash,
            "calibration_receipt_preserved": True,
            "experiments_executed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
