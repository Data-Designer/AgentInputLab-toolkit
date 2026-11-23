"""Export verified experiment runs as one content-addressed discovery freeze."""

from __future__ import annotations

import argparse

from _common import dump

from agentinputlab.selectors import (
    SelectorDiscoveryStore,
    build_selector_discovery_freeze,
    load_discovery_export_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert completed discovery run artifacts into a selector discovery "
            "freeze; this command never executes experiments."
        )
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--store", required=True)
    args = parser.parse_args()

    revision, extractor, utility, representatives = load_discovery_export_config(args.config)
    freeze = build_selector_discovery_freeze(
        args.run_dir,
        revision=revision,
        extractor=extractor,
        utility_definition=utility,
        representative_by_family=representatives,
    )
    store = SelectorDiscoveryStore(args.store)
    identity = store.save(freeze)
    dump(
        {
            "schema_version": 1,
            "artifact": "selector_discovery_freeze",
            "identity": {
                "revision": identity.revision,
                "sha256": identity.sha256,
            },
            "path": str(store.path_for(identity)),
            "source_runs": len(freeze.source_runs),
            "feature_observations": len(freeze.observations),
            "evaluated_arm_outcomes": len(freeze.outcomes),
            "training_examples": len(freeze.training_table.examples),
            "training_table_sha256": freeze.training_table.table_hash,
            "experiments_executed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
