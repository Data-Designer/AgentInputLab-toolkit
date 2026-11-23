"""Pinned upstream bridge for SkillOpt offline text-space optimization.

AgentInputLab does not reimplement SkillOpt's six-stage optimizer.  This module
verifies the audited source files, launches the upstream ``scripts/train.py``
entry point without a shell, freezes the fairness-relevant overrides, and
returns a content-addressed ``best_skill.md`` artifact.  Provider credentials
remain environment-only and are never written to the artifact manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, InterventionContext, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from .skillopt_environment import ENVIRONMENT_RECEIPT, SkillOptEnvironmentBinding

SKILLOPT_UPSTREAM_COMMIT = "9639719632daecacd1baaa47fe781f3c0253600a"
SKILLOPT_RELEASE_ID = "recipe.skillopt.offline"
SKILLOPT_PROTOCOL = "agentinputlab.skillopt.upstream-cli.v1"
SKILLOPT_SOURCE_HASHES = MappingProxyType(
    {
        "LICENSE": "9515a04280cf50ce154eb3ab8fa375c7cc5e6b45ea5cb23b7ff3830737020c15",
        "scripts/train.py": ("e4423a5da59db78c9dd58c0c8c7736863131c5f5d87c4d0a23a61a714fd62703"),
        "skillopt/engine/trainer.py": (
            "ff71a1480ee7c7c21a2574bb429eeb7dc0c88da0ea018feb22da4fa00b663a7f"
        ),
        "skillopt/evaluation/gate.py": (
            "3c2405f13288e97f53afa0bdfc0752ba6917c51edd1904a810deb4b5a1a81bef"
        ),
        "skillopt/config.py": ("cf28e722ac31b6a60ffc168202e0cc07d4f78581506b8df6acd44fc38e03b629"),
        "configs/_base_/default.yaml": (
            "385c1d79af97796f28051589e60551b36668e526b5a9788af53a5196997f42c1"
        ),
    }
)
_REQUIRED_OUTPUTS = (
    "best_skill.md",
    "config.json",
    "history.json",
    "runtime_state.json",
)


@dataclass(frozen=True, slots=True)
class SkillOptControls:
    """Fairness-relevant overrides passed to the upstream trainer unchanged."""

    num_epochs: int = 4
    batch_size: int = 40
    learning_rate: int = 4
    seed: int = 0
    max_steps: int | None = None
    timeout_seconds: float = 7200.0

    def __post_init__(self) -> None:
        integer_values = (self.num_epochs, self.batch_size, self.learning_rate, self.seed)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValidationError(
                "SkillOpt epochs, batch size, learning rate, and seed must be integers"
            )
        if min(self.num_epochs, self.batch_size, self.learning_rate) < 1:
            raise ValidationError("SkillOpt epochs, batch size, and learning rate must be positive")
        if self.seed < 0:
            raise ValidationError("SkillOpt seed must be non-negative")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValidationError("SkillOpt max_steps must be null or a positive integer")
        try:
            valid_timeout = (
                not isinstance(self.timeout_seconds, bool)
                and isinstance(self.timeout_seconds, (int, float))
                and math.isfinite(self.timeout_seconds)
                and self.timeout_seconds > 0
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValidationError("SkillOpt timeout_seconds must be finite and positive")

    @property
    def cfg_options(self) -> tuple[str, ...]:
        options = (
            f"train.num_epochs={self.num_epochs}",
            f"train.batch_size={self.batch_size}",
            f"train.seed={self.seed}",
            f"optimizer.learning_rate={self.learning_rate}",
            "evaluation.use_gate=true",
        )
        if self.max_steps is not None:
            return (*options, f"env.max_steps={self.max_steps}")
        return options


@dataclass(frozen=True, slots=True)
class SkillOptArtifact:
    """Frozen deployment artifact produced by one audited upstream run."""

    artifact_id: str
    release_id: str
    best_skill: str
    best_skill_sha256: str
    upstream_commit: str
    protocol: str
    config_sha256: str
    source_hashes: Mapping[str, str]
    output_hashes: Mapping[str, str]
    controls: Mapping[str, Any]
    model_fingerprints: Mapping[str, Mapping[str, Any]]
    environment_keys: tuple[str, ...]
    command_sha256: str

    def __post_init__(self) -> None:
        if self.release_id != SKILLOPT_RELEASE_ID:
            raise ValidationError("SkillOpt artifact release identity mismatch")
        if self.upstream_commit != SKILLOPT_UPSTREAM_COMMIT:
            raise ValidationError("SkillOpt artifact upstream revision mismatch")
        if self.protocol != SKILLOPT_PROTOCOL:
            raise ValidationError("SkillOpt artifact protocol mismatch")
        if not isinstance(self.best_skill, str) or not self.best_skill.strip():
            raise ValidationError("SkillOpt best_skill.md must be non-empty")
        if _sha256_bytes(self.best_skill.encode("utf-8")) != self.best_skill_sha256:
            raise ValidationError("SkillOpt best_skill text differs from its digest")
        if any(
            not isinstance(getattr(self, name), Mapping)
            for name in ("source_hashes", "output_hashes", "controls", "model_fingerprints")
        ) or any(not isinstance(value, Mapping) for value in self.model_fingerprints.values()):
            raise ValidationError(
                "SkillOpt artifact hashes, controls, and fingerprints must be mappings"
            )
        if not isinstance(self.environment_keys, (list, tuple)):
            raise ValidationError("SkillOpt environment_keys must be an array of names")
        if self.output_hashes.get("best_skill.md") != self.best_skill_sha256:
            raise ValidationError("SkillOpt best_skill digest differs from its output manifest")
        object.__setattr__(self, "source_hashes", freeze_mapping(self.source_hashes))
        object.__setattr__(self, "output_hashes", freeze_mapping(self.output_hashes))
        object.__setattr__(self, "controls", freeze_mapping(self.controls))
        object.__setattr__(
            self,
            "model_fingerprints",
            MappingProxyType(
                {
                    str(role): freeze_mapping(fingerprint)
                    for role, fingerprint in self.model_fingerprints.items()
                }
            ),
        )
        keys = tuple(self.environment_keys)
        if len(keys) != len(set(keys)) or any(
            not isinstance(key, str) or not key.strip() or key != key.strip() for key in keys
        ):
            raise ValidationError("SkillOpt environment key names must be unique and non-empty")
        object.__setattr__(self, "environment_keys", keys)
        if self.artifact_id != content_hash(self.identity_payload()):
            raise ValidationError("SkillOpt artifact content identity mismatch")

    def identity_payload(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"artifact_id", "best_skill"}
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SkillOptArtifact:
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValidationError("SkillOpt artifact fields are invalid")
        malformed = False
        try:
            return cls(**dict(value))
        except ValidationError:
            raise
        except (AttributeError, TypeError, ValueError, RecursionError):
            malformed = True
        if malformed:
            raise ValidationError("SkillOpt artifact values are malformed")

    @classmethod
    def from_json(
        cls, path: str | Path, *, expected_sha256: str, max_bytes: int = 4 * 1024 * 1024
    ) -> SkillOptArtifact:
        return cls.from_mapping(
            load_pinned_json_mapping(path, expected_sha256=expected_sha256, max_bytes=max_bytes)
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class SkillOptOfflineRecipe:
    """Execute the pinned upstream trainer and validate its deployable artifact."""

    controls: SkillOptControls = field(default_factory=SkillOptControls)
    python_executable: str = sys.executable
    method_id: str = "recipe.skillopt"
    release_id: str = SKILLOPT_RELEASE_ID
    environment_binding: SkillOptEnvironmentBinding | None = None

    def build_command(
        self,
        *,
        checkout: str | Path,
        config_path: str | Path,
        output_root: str | Path,
    ) -> tuple[str, ...]:
        source = Path(checkout).expanduser().resolve()
        config = Path(config_path).expanduser().resolve()
        output = Path(output_root).expanduser().resolve()
        if self.environment_binding is None:
            prefix = (self.python_executable, str(source / "scripts" / "train.py"))
        else:
            if not isinstance(self.environment_binding, SkillOptEnvironmentBinding):
                raise ValidationError("SkillOpt environment_binding must be explicit and checksum-bound")
            prefix = (self.python_executable, "-m", "agentinputlab.recipes.skillopt_environment_cli",
                      "--checkout", str(source), *self.environment_binding.command_options(),
                      "--receipt", str(output / ENVIRONMENT_RECEIPT), "--")
        return (
            *prefix,
            "--config",
            str(config),
            "--out_root",
            str(output),
            "--cfg-options",
            *self.controls.cfg_options,
        )

    def fit(
        self,
        *,
        checkout: str | Path,
        config_path: str | Path,
        output_root: str | Path,
        context: InterventionContext,
        environment: Mapping[str, str] | None = None,
    ) -> SkillOptArtifact:
        if self.controls.seed != context.seed:
            raise ValidationError("SkillOpt controls.seed must equal InterventionContext.seed")
        if self.environment_binding is not None and not isinstance(self.environment_binding, SkillOptEnvironmentBinding):
            raise ValidationError("SkillOpt environment_binding must be explicit and checksum-bound")
        source = validate_skillopt_checkout(checkout)
        binding_identity = self.environment_binding.verify() if self.environment_binding is not None else None
        config = Path(config_path).expanduser().resolve()
        if not config.is_file() or config.is_symlink():
            raise ExternalDependencyError("SkillOpt config must be a regular, non-symlink file")
        output = Path(output_root).expanduser().resolve()
        if output == source or source in output.parents:
            raise ValidationError("SkillOpt output_root must be outside the pinned checkout")
        if output.exists() and any(output.iterdir()):
            raise ValidationError("SkillOpt output_root must be absent or empty")
        output.mkdir(parents=True, exist_ok=True)

        model_fingerprints = _model_fingerprints(context)
        command = self.build_command(
            checkout=source,
            config_path=config,
            output_root=output,
        )
        extra_environment = _validated_environment(environment or {})
        process_environment = dict(os.environ)
        process_environment.update(extra_environment)
        completed = subprocess.run(
            command,
            cwd=source,
            env=process_environment,
            capture_output=True,
            text=True,
            timeout=float(self.controls.timeout_seconds),
            check=False,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "").strip()[-4000:]
            raise ExternalDependencyError(
                f"SkillOpt upstream trainer exited with code {completed.returncode}: {diagnostic}"
            )
        artifact = load_skillopt_artifact(
            output,
            config_sha256=_sha256_file(config),
            controls=asdict(self.controls),
            model_fingerprints=model_fingerprints,
            environment_keys=tuple(sorted(extra_environment)),
            command_sha256=content_hash(command),
            environment_binding=binding_identity,
        )
        manifest = output / "agentinputlab_skillopt_manifest.json"
        manifest.write_text(
            json.dumps(artifact.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact


def load_skillopt_artifact(
    output_root: str | Path,
    *,
    config_sha256: str,
    controls: Mapping[str, Any],
    model_fingerprints: Mapping[str, Mapping[str, Any]],
    environment_keys: tuple[str, ...] = (),
    command_sha256: str,
    environment_binding: Mapping[str, Any] | None = None,
) -> SkillOptArtifact:
    """Validate a completed upstream output directory without rerunning training."""

    output = Path(output_root).expanduser().resolve()
    output_hashes: dict[str, str] = {}
    for relative in _REQUIRED_OUTPUTS:
        path = output / relative
        if not path.is_file() or path.is_symlink():
            raise ExternalDependencyError(f"SkillOpt output is missing regular file {relative}")
        output_hashes[relative] = _sha256_file(path)
    source_hashes = dict(SKILLOPT_SOURCE_HASHES)
    if environment_binding is None and (output / ENVIRONMENT_RECEIPT).exists():
        raise ExternalDependencyError("SkillOpt custom environment output requires its explicit environment binding")
    if environment_binding is not None:
        path = output / ENVIRONMENT_RECEIPT
        if not path.is_file() or path.is_symlink():
            raise ExternalDependencyError("SkillOpt completed environment binding receipt is missing")
        receipt = load_pinned_json_mapping(path, expected_sha256=_sha256_file(path), max_bytes=1024 * 1024)
        if (receipt.get("schema") != "agentinputlab.skillopt.environment_binding.v1"
                or dict(receipt.get("binding", {})) != dict(environment_binding)
                or not receipt.get("environment_fingerprint")
                or receipt.get("native_training_algorithm_unchanged") is not True
                or receipt.get("profile_kind") != "caller_environment_adapter"):
            raise ExternalDependencyError("SkillOpt completed environment binding differs from the configured factory")
        output_hashes[ENVIRONMENT_RECEIPT] = _sha256_file(path)
        source_hashes.update(operator_environment_factory=environment_binding["factory_sha256"],
                             operator_environment_configuration=environment_binding["configuration_sha256"])
    best_skill = (output / "best_skill.md").read_text(encoding="utf-8")
    artifact_base = {
        "release_id": SKILLOPT_RELEASE_ID,
        "best_skill_sha256": output_hashes["best_skill.md"],
        "upstream_commit": SKILLOPT_UPSTREAM_COMMIT,
        "protocol": SKILLOPT_PROTOCOL,
        "config_sha256": config_sha256,
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
        "controls": dict(controls),
        "model_fingerprints": {
            role: dict(fingerprint) for role, fingerprint in model_fingerprints.items()
        },
        "environment_keys": tuple(environment_keys),
        "command_sha256": command_sha256,
    }
    return SkillOptArtifact(
        artifact_id=content_hash(artifact_base),
        best_skill=best_skill,
        **artifact_base,
    )


def validate_skillopt_checkout(checkout: str | Path) -> Path:
    """Verify the audited SkillOpt commit and critical optimization files."""

    checkout = Path(checkout).expanduser().resolve()
    if not checkout.is_dir() or checkout.is_symlink():
        raise ExternalDependencyError("SkillOpt checkout must be a regular directory")
    for relative, expected_hash in SKILLOPT_SOURCE_HASHES.items():
        path = checkout / relative
        if not path.is_file() or path.is_symlink():
            raise ExternalDependencyError(f"SkillOpt checkout is missing {relative}")
        if _sha256_file(path) != expected_hash:
            raise ExternalDependencyError(f"SkillOpt pinned source hash changed: {relative}")
    try:
        head = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(checkout), "status", "--porcelain"),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ExternalDependencyError("SkillOpt Git checkout verification failed") from exc
    if head != SKILLOPT_UPSTREAM_COMMIT:
        raise ExternalDependencyError("SkillOpt checkout is not at the pinned commit")
    if status:
        raise ExternalDependencyError("SkillOpt pinned checkout must be clean")
    return checkout


def _model_fingerprints(
    context: InterventionContext,
) -> Mapping[str, Mapping[str, Any]]:
    configured = context.run_config.get("skillopt_model_fingerprints")
    if configured is not None:
        if not isinstance(configured, Mapping):
            raise ValidationError("skillopt_model_fingerprints must be a mapping")
        result: dict[str, Mapping[str, Any]] = {}
        for role in ("target", "optimizer"):
            value = configured.get(role)
            if not isinstance(value, Mapping) or not value:
                raise ValidationError(f"SkillOpt is missing the {role} model fingerprint")
            result[role] = dict(value)
        return MappingProxyType(result)
    if context.backend is None:
        raise ExternalDependencyError(
            "SkillOpt requires explicit target/optimizer fingerprints or a backend"
        )
    try:
        shared = dict(context.backend.fingerprint())
    except AttributeError as exc:
        raise ExternalDependencyError("SkillOpt backend does not expose a fingerprint") from exc
    return MappingProxyType({"target": dict(shared), "optimizer": dict(shared)})


def _validated_environment(value: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValidationError("SkillOpt environment keys must be trimmed non-empty strings")
        if not isinstance(item, str):
            raise ValidationError("SkillOpt environment values must be strings")
        result[key] = item
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "SKILLOPT_PROTOCOL",
    "SKILLOPT_RELEASE_ID",
    "SKILLOPT_SOURCE_HASHES",
    "SKILLOPT_UPSTREAM_COMMIT",
    "SkillOptArtifact",
    "SkillOptControls",
    "SkillOptEnvironmentBinding",
    "SkillOptOfflineRecipe",
    "load_skillopt_artifact",
    "validate_skillopt_checkout",
]
