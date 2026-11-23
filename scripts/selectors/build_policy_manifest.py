"""Build the frozen selector/global-comparator policy manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from _common import (
    add_discovery_identity,
    add_selector_identity,
    discovery_identity,
    dump,
    selector_identity,
)

from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.selectors import (
    SelectorArtifactStage,
    SelectorArtifactStore,
    SelectorDiscoveryStore,
    build_selector_policy_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the locked best-eligible-global comparator and policy "
            "provenance from exact frozen selector/discovery artifacts."
        )
    )
    add_discovery_identity(parser)
    parser.add_argument("--selector-store", required=True)
    add_selector_identity(parser, prefix="frozen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    discovery_store = SelectorDiscoveryStore(args.discovery_store)
    discovery = discovery_store.load(discovery_identity(args))
    selector_store = SelectorArtifactStore(args.selector_store)
    _selector, frozen = selector_store.load(
        selector_identity(args, prefix="frozen"),
        require_stage=SelectorArtifactStage.FROZEN,
    )
    manifest = build_selector_policy_manifest(frozen, discovery)
    content = (canonical_json(manifest) + "\n").encode("utf-8")
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise RuntimeError("policy manifest destination already contains different bytes")
    else:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".partial-policy-",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    digest = hashlib.sha256(content).hexdigest()
    selected = manifest["locked_best_eligible_global"]
    dump(
        {
            "schema_version": 1,
            "artifact": "agentinputlab.selector_policy_manifest",
            "stage": "frozen",
            "path": str(destination),
            "sha256": digest,
            "selected_family": selected["selected_family"],
            "selected_release_id": selected["selected_release_id"],
            "experiments_executed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
