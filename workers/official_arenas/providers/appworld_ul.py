"""Fail-closed provider for an official AppWorld-UL runtime plugin.

The public, pinned AppWorld checkout is not itself the AppWorld-UL user-loop
runtime.  This provider therefore refuses to substitute base AppWorld.  It
loads a checksum-bound ``appworld_ul.*`` plugin supplied with the official UL
assets and normalizes that plugin to AgentInputLab's atomic arena protocol.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..protocol import OfficialArenaError, ProviderArmResult

_ENTRYPOINT = re.compile(
    r"^appworld_ul(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AppWorldULProvider:
    """Delegate to a reviewed official UL plugin; never emulate it with AppWorld."""

    benchmark_id = "appworld_ul.common_support"
    source_revision = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
    scorer_id = "appworld_ul.official_task_goal_completion"
    scorer_revision = "59cc58737a6e84d400dce073800a4e94670bc97ad147e24e36a7d13e9df6e71b"

    _CONFIG_FIELDS = frozenset(
        {
            "runtime_entrypoint",
            "runtime_revision_sha256",
            "runtime_config",
            "locked_dependency_hashes",
        }
    )

    def doctor(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            parsed = self._validate_config(config)
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if head != self.source_revision:
                raise OfficialArenaError("AppWorld checkout revision differs")
            runtime = self._runtime(parsed)
            blockers = runtime.doctor(parsed["runtime_config"])
            if (
                isinstance(blockers, (str, bytes))
                or not isinstance(blockers, Sequence)
                or any(not isinstance(item, str) for item in blockers)
            ):
                raise OfficialArenaError("AppWorld-UL plugin doctor returned an invalid result")
            if blockers:
                return tuple(blockers)
        except Exception as exc:
            return (f"{type(exc).__name__}: {exc}",)
        return ()

    def capture(
        self,
        config: Mapping[str, Any],
        *,
        output_root: Path,
    ) -> tuple[Mapping[str, Any], ...]:
        parsed = self._validate_config(config)
        runtime = self._runtime(parsed)
        rows = runtime.capture(
            parsed["runtime_config"],
            output_root=output_root,
        )
        if (
            isinstance(rows, (str, bytes))
            or not isinstance(rows, Sequence)
            or not rows
            or any(not isinstance(item, Mapping) for item in rows)
        ):
            raise OfficialArenaError("AppWorld-UL plugin capture returned no valid checkpoints")
        return tuple(rows)

    def run_arm(
        self,
        config: Mapping[str, Any],
        *,
        replay_record: Mapping[str, Any],
        arm_payload: Mapping[str, Any],
        seed: int,
        role: str,
        work_dir: Path,
    ) -> ProviderArmResult:
        parsed = self._validate_config(config)
        runtime = self._runtime(parsed)
        result = runtime.run_arm(
            parsed["runtime_config"],
            replay_record=replay_record,
            arm_payload=arm_payload,
            seed=seed,
            role=role,
            work_dir=work_dir,
        )
        if isinstance(result, ProviderArmResult):
            if "task_success" not in result.metrics:
                raise OfficialArenaError(
                    "AppWorld-UL plugin result has no official task_success metric"
                )
            return result
        if not isinstance(result, Mapping) or set(result) != {
            "metrics",
            "trace",
            "usage",
            "runtime_instance_id",
        }:
            raise OfficialArenaError("AppWorld-UL plugin run_arm returned an invalid result")
        try:
            normalized = ProviderArmResult(
                metrics=result["metrics"],
                trace=result["trace"],
                usage=result["usage"],
                runtime_instance_id=result["runtime_instance_id"],
            )
        except (TypeError, ValueError) as exc:
            raise OfficialArenaError(
                "AppWorld-UL plugin result violates the provider contract"
            ) from exc
        if "task_success" not in normalized.metrics:
            raise OfficialArenaError(
                "AppWorld-UL plugin result has no official task_success metric"
            )
        return normalized

    def _runtime(self, config: Mapping[str, Any]) -> Any:
        module_name, attribute = config["runtime_entrypoint"].split(":", 1)
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
            runtime = factory()
        except Exception as exc:
            raise OfficialArenaError(
                "official AppWorld-UL runtime plugin is unavailable; "
                "base AppWorld is intentionally not used as a substitute"
            ) from exc
        for name, expected in (
            ("benchmark_id", self.benchmark_id),
            ("source_revision", self.source_revision),
            ("scorer_id", self.scorer_id),
            (
                "runtime_revision_sha256",
                config["runtime_revision_sha256"],
            ),
        ):
            if getattr(runtime, name, None) != expected:
                raise OfficialArenaError(
                    f"AppWorld-UL plugin {name} differs from the locked identity"
                )
        for name in ("doctor", "capture", "run_arm"):
            if not callable(getattr(runtime, name, None)):
                raise OfficialArenaError(f"AppWorld-UL plugin does not implement {name}")
        return runtime

    def _validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(config, Mapping) or set(config) != self._CONFIG_FIELDS:
            raise OfficialArenaError("AppWorld-UL provider config fields are invalid")
        if (
            not isinstance(config["runtime_entrypoint"], str)
            or _ENTRYPOINT.fullmatch(config["runtime_entrypoint"]) is None
        ):
            raise OfficialArenaError(
                "AppWorld-UL runtime entrypoint must be an appworld_ul.* plugin"
            )
        if (
            not isinstance(config["runtime_revision_sha256"], str)
            or _SHA256.fullmatch(config["runtime_revision_sha256"]) is None
        ):
            raise OfficialArenaError("AppWorld-UL runtime revision must be a SHA-256 digest")
        if not isinstance(config["runtime_config"], Mapping):
            raise OfficialArenaError("AppWorld-UL runtime_config must be an object")
        dependencies = config["locked_dependency_hashes"]
        if not isinstance(dependencies, Mapping) or not dependencies:
            raise OfficialArenaError("AppWorld-UL dependency locks are missing")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in dependencies.values()
        ):
            raise OfficialArenaError("AppWorld-UL dependency locks are invalid")
        return config


def factory() -> AppWorldULProvider:
    return AppWorldULProvider()
