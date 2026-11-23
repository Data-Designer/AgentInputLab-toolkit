from __future__ import annotations

import argparse
import json

from agentinputlab.selectors import (
    SelectorArtifactIdentity,
    SelectorDiscoveryIdentity,
)


def add_discovery_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--discovery-store", required=True)
    parser.add_argument("--discovery-revision", required=True)
    parser.add_argument("--discovery-sha256", required=True)


def add_selector_identity(
    parser: argparse.ArgumentParser,
    *,
    prefix: str = "",
    required: bool = True,
) -> None:
    option = f"{prefix}-" if prefix else ""
    destination = f"{prefix.replace('-', '_')}_" if prefix else ""
    parser.add_argument(
        f"--{option}selector-id",
        required=required,
        dest=f"{destination}selector_id",
    )
    parser.add_argument(
        f"--{option}revision",
        required=required,
        dest=f"{destination}revision",
    )
    parser.add_argument(
        f"--{option}sha256",
        required=required,
        dest=f"{destination}sha256",
    )


def discovery_identity(args: argparse.Namespace) -> SelectorDiscoveryIdentity:
    return SelectorDiscoveryIdentity(
        revision=args.discovery_revision,
        sha256=args.discovery_sha256,
    )


def selector_identity(
    args: argparse.Namespace,
    *,
    prefix: str = "",
) -> SelectorArtifactIdentity:
    destination = f"{prefix.replace('-', '_')}_" if prefix else ""
    return SelectorArtifactIdentity(
        selector_id=getattr(args, f"{destination}selector_id"),
        revision=getattr(args, f"{destination}revision"),
        sha256=getattr(args, f"{destination}sha256"),
    )


def dump(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
