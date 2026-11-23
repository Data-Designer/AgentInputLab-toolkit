"""Idempotent bootstrap for public, checksum-pinned runtime source checkouts.

This module turns a resolvable "missing checkout" condition into an explicit
operation.  It never installs packages, downloads model weights, or starts an
experiment.  Network access is opt-in and every checkout is detached at the
registry-pinned commit.  Existing dirty or unrelated directories are preserved
and reported instead of being overwritten.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.registry import AlgorithmRegistry

RUNTIME_BOOTSTRAP_PROTOCOL = "agentinputlab.runtime-bootstrap.v1"

_ARENAS = MappingProxyType(
    {
        "tooltalk.full": (
            "https://github.com/microsoft/ToolTalk",
            "e05f4ce6132c80ed33392b81535b077d56ab28fd",
        ),
        "tau3.banking_knowledge": (
            "https://github.com/sierra-research/tau2-bench",
            "cf71a8070269883e38a365ffa85f78f46844c1f4",
        ),
        "state_bench.common_support": (
            "https://github.com/microsoft/STATE-Bench",
            "4efcbf2d4fe60df04878859b692d9391f3d5b33a",
        ),
        "toolsandbox.common_support": (
            "https://github.com/apple/ToolSandbox",
            "165848b9a78cead7ca7fe7c89c688b58e6501219",
        ),
        # This checkout supplies base AppWorld only.  The official UL plugin is
        # intentionally a separate operator binding because the public base
        # repository is not a scientifically equivalent user-loop runtime.
        "appworld_ul.common_support": (
            "https://github.com/StonyBrookNLP/appworld",
            "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
        ),
        "ale.external_transfer": (
            "https://github.com/rdi-berkeley/agents-last-exam",
            "1e615e456de7cef57706680613cb80ee13c7fc76",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeSource:
    """One public source tree required by an algorithm or official arena."""

    source_id: str
    kind: str
    repo_url: str
    commit: str
    checkout: Path

    def __post_init__(self) -> None:
        if self.kind not in {"method", "arena"}:
            raise ValidationError("runtime source kind must be method or arena")
        if not self.source_id or not self.repo_url.startswith("https://github.com/"):
            raise ValidationError("runtime source identity or repository is invalid")
        if len(self.commit) != 40 or any(ch not in "0123456789abcdef" for ch in self.commit):
            raise ValidationError("runtime source commit must be a lowercase Git SHA-1")
        if not self.checkout.is_absolute():
            raise ValidationError("runtime source checkout must be absolute")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "repo_url": self.repo_url,
            "commit": self.commit,
            "checkout": str(self.checkout),
        }


def runtime_source_plan(
    root: str | Path,
    *,
    method_ids: Sequence[str] | None = None,
    include_arenas: bool = True,
    arena_ids: Sequence[str] | None = None,
) -> tuple[RuntimeSource, ...]:
    """Return the exact public checkout plan without touching disk or network."""

    target = Path(root).expanduser().resolve()
    registry = AlgorithmRegistry.default()
    methods = {method.method_id: method for method in registry.list_methods()}
    selected = tuple(sorted(methods) if method_ids is None else method_ids)
    if len(selected) != len(set(selected)):
        raise ValidationError("runtime bootstrap method_ids must be unique")
    unknown = sorted(set(selected) - set(methods))
    if unknown:
        raise ValidationError(f"unknown runtime bootstrap methods: {unknown}")
    if arena_ids is not None and not include_arenas:
        raise ValidationError("arena_ids cannot be combined with include_arenas=False")
    arenas = tuple(sorted(_ARENAS) if arena_ids is None else arena_ids) if include_arenas else ()
    if len(arenas) != len(set(arenas)):
        raise ValidationError("runtime bootstrap arena_ids must be unique")
    unknown_arenas = sorted(set(arenas) - set(_ARENAS))
    if unknown_arenas:
        raise ValidationError(f"unknown runtime bootstrap arenas: {unknown_arenas}")
    if not selected and not arenas:
        raise ValidationError("runtime bootstrap requires at least one method or arena")
    rows = [
        RuntimeSource(
            source_id=method_id,
            kind="method",
            repo_url=methods[method_id].upstream.repo_url,
            commit=methods[method_id].upstream.commit,
            # Matches prepare_one_click_campaign.py's method_bindings_root
            # convention, so bootstrapped sources need no manual relocation.
            checkout=target / method_id / "checkout",
        )
        for method_id in selected
    ]
    if include_arenas:
        rows.extend(
            RuntimeSource(
                source_id=cell_id,
                kind="arena",
                repo_url=_ARENAS[cell_id][0],
                commit=_ARENAS[cell_id][1],
                checkout=target / "_arenas" / cell_id / "checkout",
            )
            for cell_id in arenas
        )
    return tuple(rows)


def bootstrap_runtime_sources(
    root: str | Path,
    *,
    method_ids: Sequence[str] | None = None,
    include_arenas: bool = True,
    arena_ids: Sequence[str] | None = None,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Verify or materialize every selected public source at its pinned commit."""

    plan = runtime_source_plan(
        root,
        method_ids=method_ids,
        include_arenas=include_arenas,
        arena_ids=arena_ids,
    )
    rows = tuple(_materialize_source(source, allow_network=allow_network) for source in plan)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUNTIME_BOOTSTRAP_PROTOCOL,
        "root": str(Path(root).expanduser().resolve()),
        "allow_network": allow_network,
        "source_count": len(rows),
        "ready_count": sum(row["ready"] for row in rows),
        "actionable_count": sum(not row["ready"] for row in rows),
        "sources": rows,
    }
    payload["plan_sha256"] = _content_hash(payload)
    return payload


def _materialize_source(source: RuntimeSource, *, allow_network: bool) -> dict[str, Any]:
    checkout = source.checkout
    if checkout.exists():
        if not checkout.is_dir() or not (checkout / ".git").exists():
            return _unready(
                source,
                "checkout path exists but is not a Git worktree",
                "move the unrelated path or choose another --root",
            )
        observed = _git(checkout, "rev-parse", "HEAD", check=False)
        if observed.returncode != 0:
            return _unready(
                source,
                "existing checkout HEAD cannot be inspected",
                "repair the checkout or choose another --root",
            )
        head = observed.stdout.strip()
        dirty = bool(_git(checkout, "status", "--porcelain", check=False).stdout.strip())
        if dirty:
            return _unready(
                source,
                "existing checkout contains local changes",
                "preserve or commit the local changes before bootstrapping",
                observed_commit=head,
            )
        if head == source.commit:
            return _ready(source, acquired_from="existing", observed_commit=head)
        if not allow_network:
            return _unready(
                source,
                f"checkout is at {head}, not the pinned commit",
                "rerun with --allow-network to fetch the pinned commit",
                observed_commit=head,
            )
        _git(checkout, "fetch", "--depth", "1", "origin", source.commit)
        _git(checkout, "checkout", "--detach", "FETCH_HEAD")
        return _verify_materialized(source, acquired_from="network_update")

    if not allow_network:
        return _unready(
            source,
            "pinned public checkout is absent",
            "rerun with --allow-network to fetch the exact registry commit",
        )
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--quiet", str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        _git(checkout, "remote", "add", "origin", source.repo_url)
        _git(checkout, "fetch", "--depth", "1", "origin", source.commit)
        _git(checkout, "checkout", "--detach", "FETCH_HEAD")
    except Exception:
        # Keep the partial checkout for diagnosis and a resumable next call.
        raise
    return _verify_materialized(source, acquired_from="network")


def _verify_materialized(source: RuntimeSource, *, acquired_from: str) -> dict[str, Any]:
    head = _git(source.checkout, "rev-parse", "HEAD").stdout.strip()
    if head != source.commit:
        raise ArtifactError(f"{source.source_id}: fetched checkout differs from pinned commit")
    if _git(source.checkout, "status", "--porcelain").stdout.strip():
        raise ArtifactError(f"{source.source_id}: fetched checkout is unexpectedly dirty")
    return _ready(source, acquired_from=acquired_from, observed_commit=head)


def _git(checkout: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = getattr(exc, "stderr", None) or str(exc)
        raise ArtifactError(f"Git operation failed: {message.strip()}") from exc


def _ready(
    source: RuntimeSource,
    *,
    acquired_from: str,
    observed_commit: str,
) -> dict[str, Any]:
    return {
        **source.to_dict(),
        "ready": True,
        "acquired_from": acquired_from,
        "observed_commit": observed_commit,
        "resolution_kind": "none",
        "next_step": None,
    }


def _unready(
    source: RuntimeSource,
    reason: str,
    next_step: str,
    *,
    observed_commit: str | None = None,
) -> dict[str, Any]:
    return {
        **source.to_dict(),
        "ready": False,
        "reason": reason,
        "observed_commit": observed_commit,
        "resolution_kind": "public_runtime_bootstrap",
        "next_step": next_step,
    }


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RUNTIME_BOOTSTRAP_PROTOCOL",
    "RuntimeSource",
    "bootstrap_runtime_sources",
    "runtime_source_plan",
]
