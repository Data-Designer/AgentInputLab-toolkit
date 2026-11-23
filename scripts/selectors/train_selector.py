"""Fit one selector from an explicit discovery freeze."""

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
    create_fitted_selector_bundle,
    load_selector_fit_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a selector artifact without running discovery experiments."
    )
    add_discovery_identity(parser)
    parser.add_argument("--selector-store", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-selector-store")
    add_selector_identity(parser, prefix="base", required=False)
    args = parser.parse_args()

    discovery = SelectorDiscoveryStore(args.discovery_store).load(discovery_identity(args))
    selector_kind, fit_config = load_selector_fit_config(args.config)
    base_selector_bundle = None
    base_values = (
        args.base_selector_store,
        args.base_selector_id,
        args.base_revision,
        args.base_sha256,
    )
    if selector_kind == "cost_risk_guardrail":
        if any(item is None for item in base_values):
            parser.error(
                "cost_risk_guardrail requires --base-selector-store, "
                "--base-selector-id, --base-revision, and --base-sha256"
            )
        _base_selector, base_selector_bundle = SelectorArtifactStore(args.base_selector_store).load(
            selector_identity(args, prefix="base")
        )
    elif any(item is not None for item in base_values):
        parser.error("base selector arguments are valid only for cost_risk_guardrail")

    bundle = create_fitted_selector_bundle(
        discovery,
        selector_kind=selector_kind,
        revision=args.revision,
        fit_config=fit_config,
        base_selector_bundle=base_selector_bundle,
    )
    store = SelectorArtifactStore(args.selector_store)
    identity = store.save(bundle)
    dump(
        {
            "schema_version": 1,
            "artifact": "selector_bundle",
            "stage": SelectorArtifactStage.FITTED.value,
            "selector_kind": selector_kind,
            "identity": {
                "selector_id": identity.selector_id,
                "revision": identity.revision,
                "sha256": identity.sha256,
            },
            "path": str(store.path_for(identity)),
            "discovery_sha256": discovery.discovery_hash,
            "experiments_executed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
