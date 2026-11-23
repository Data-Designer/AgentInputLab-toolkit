"""Strict one-command DAG for the complete production experiment package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.experiments.secrets import (
    environment_secret_values,
    redact_secret_text,
)

PRODUCTION_CAMPAIGN_PROTOCOL = "agentinputlab.production-campaign.v1"

_REQUIRED_STAGES = (
    "data",
    "method_parity",
    "discovery",
    "representative_freeze",
    "family_confirmation",
    "common_support",
    "selector",
    "robustness",
    "stateful_replication",
    "paper_export",
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "campaign_id",
        "workspace",
        "resume",
        "steps",
        "campaign_hash",
    }
)
_STEP_FIELDS = frozenset(
    {
        "step_id",
        "stage",
        "requires",
        "command",
        "cwd",
        "environment",
        "pass_environment",
        "inputs",
        "outputs",
        "timeout_seconds",
    }
)
_INPUT_FIELDS = frozenset({"path", "sha256"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENVIRONMENT = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA_TEMPLATE = re.compile(r"^\{sha256:(.+)\}$")
_REPOSITORY_COMMIT_ENV = "AILAB_REPOSITORY_COMMIT"


@dataclass(frozen=True, slots=True)
class ProductionInput:
    path: str
    sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValidationError("production campaign input path is invalid")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValidationError("production campaign input SHA-256 is invalid")


@dataclass(frozen=True, slots=True)
class ProductionStep:
    step_id: str
    stage: str
    requires: tuple[str, ...]
    command: tuple[str, ...]
    cwd: str
    environment: Mapping[str, str]
    pass_environment: tuple[str, ...]
    inputs: tuple[ProductionInput, ...]
    outputs: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or _IDENTIFIER.fullmatch(self.step_id) is None:
            raise ValidationError("production campaign step_id is invalid")
        if self.stage not in _REQUIRED_STAGES:
            raise ValidationError(f"unsupported production campaign stage: {self.stage}")
        for label, values in (
            ("requires", self.requires),
            ("command", self.command),
            ("pass_environment", self.pass_environment),
            ("outputs", self.outputs),
        ):
            if (
                isinstance(values, (str, bytes))
                or not isinstance(values, tuple)
                or any(not isinstance(item, str) or not item for item in values)
            ):
                raise ValidationError(f"production campaign step {self.step_id} {label} is invalid")
        if not self.command or self.command[0] not in {"@python", "@agentinputlab"}:
            raise ValidationError(
                f"production campaign step {self.step_id} command must start "
                "with @python or @agentinputlab"
            )
        if self.command[0] == "@python" and len(self.command) < 2:
            raise ValidationError(f"production campaign step {self.step_id} has no Python script")
        if not isinstance(self.cwd, str) or not self.cwd:
            raise ValidationError(f"production campaign step {self.step_id} cwd is invalid")
        if not isinstance(self.environment, Mapping) or any(
            not isinstance(name, str)
            or _ENVIRONMENT.fullmatch(name) is None
            or not isinstance(value, str)
            or not value
            for name, value in self.environment.items()
        ):
            raise ValidationError(f"production campaign step {self.step_id} environment is invalid")
        if len(set(self.requires)) != len(self.requires):
            raise ValidationError(f"production campaign step {self.step_id} repeats dependencies")
        if len(set(self.pass_environment)) != len(self.pass_environment) or any(
            _ENVIRONMENT.fullmatch(name) is None for name in self.pass_environment
        ):
            raise ValidationError(
                f"production campaign step {self.step_id} environment allowlist is invalid"
            )
        if len(set(self.outputs)) != len(self.outputs):
            raise ValidationError(f"production campaign step {self.step_id} repeats outputs")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValidationError(f"production campaign step {self.step_id} timeout is invalid")


@dataclass(frozen=True, slots=True)
class ProductionCampaign:
    campaign_id: str
    workspace: Path
    resume: bool
    steps: tuple[ProductionStep, ...]
    campaign_hash: str
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or _IDENTIFIER.fullmatch(self.campaign_id) is None:
            raise ValidationError("production campaign_id is invalid")
        if not isinstance(self.workspace, Path) or not self.workspace.is_absolute():
            raise ValidationError("production campaign workspace must be absolute")
        if not isinstance(self.resume, bool):
            raise ValidationError("production campaign resume must be boolean")
        if not self.steps:
            raise ValidationError("production campaign requires steps")
        identifiers = tuple(step.step_id for step in self.steps)
        if len(set(identifiers)) != len(identifiers):
            raise ValidationError("production campaign step IDs must be unique")
        seen: set[str] = set()
        for step in self.steps:
            unknown = set(step.requires).difference(seen)
            if unknown:
                raise ValidationError(
                    f"step {step.step_id} depends on unknown or later steps: "
                    + ", ".join(sorted(unknown))
                )
            seen.add(step.step_id)
        observed_stages = tuple(dict.fromkeys(step.stage for step in self.steps))
        if observed_stages != _REQUIRED_STAGES:
            raise ValidationError(
                "production campaign must cover the exact ordered stages: "
                + ", ".join(_REQUIRED_STAGES)
            )
        outputs: dict[str, str] = {}
        for step in self.steps:
            for path in step.outputs:
                if path in outputs:
                    raise ValidationError(f"production output {path!r} has multiple producers")
                outputs[path] = step.step_id
        if _SHA256.fullmatch(self.campaign_hash) is None:
            raise ValidationError("production campaign_hash is invalid")
        if _SHA256.fullmatch(self.source_sha256) is None:
            raise ValidationError("production campaign source_sha256 is invalid")


def production_campaign_hash(value: Mapping[str, Any]) -> str:
    """Hash a campaign before the self-hash field is populated."""

    return content_hash({key: item for key, item in value.items() if key != "campaign_hash"})


def load_production_campaign(path: Path | str) -> ProductionCampaign:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ArtifactError("production campaign file must not be a symlink")
    source = source.resolve()
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("cannot load production campaign JSON") from exc
    if not isinstance(value, Mapping) or set(value) != _ROOT_FIELDS:
        raise ValidationError("production campaign fields are invalid")
    if value["schema_version"] != 1 or value["protocol"] != PRODUCTION_CAMPAIGN_PROTOCOL:
        raise ValidationError("unsupported production campaign protocol")
    if value["campaign_hash"] != production_campaign_hash(value):
        raise ArtifactError("production campaign self-hash differs")
    raw_steps = value["steps"]
    if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
        raise ValidationError("production campaign steps must be an array")
    steps: list[ProductionStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping) or set(raw_step) != _STEP_FIELDS:
            raise ValidationError(f"production step {index} fields are invalid")
        inputs: list[ProductionInput] = []
        raw_inputs = raw_step["inputs"]
        if isinstance(raw_inputs, (str, bytes)) or not isinstance(raw_inputs, Sequence):
            raise ValidationError(f"production step {index} inputs are invalid")
        for raw_input in raw_inputs:
            if not isinstance(raw_input, Mapping) or set(raw_input) != _INPUT_FIELDS:
                raise ValidationError(f"production step {index} input fields are invalid")
            inputs.append(
                ProductionInput(
                    path=raw_input["path"],
                    sha256=raw_input["sha256"],
                )
            )
        steps.append(
            ProductionStep(
                step_id=raw_step["step_id"],
                stage=raw_step["stage"],
                requires=tuple(raw_step["requires"]),
                command=tuple(raw_step["command"]),
                cwd=raw_step["cwd"],
                environment=freeze_mapping(raw_step["environment"]),
                pass_environment=tuple(raw_step["pass_environment"]),
                inputs=tuple(inputs),
                outputs=tuple(raw_step["outputs"]),
                timeout_seconds=raw_step["timeout_seconds"],
            )
        )
    return ProductionCampaign(
        campaign_id=value["campaign_id"],
        workspace=Path(value["workspace"]).expanduser().resolve(),
        resume=value["resume"],
        steps=tuple(steps),
        campaign_hash=value["campaign_hash"],
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def run_production_campaign(
    campaign: ProductionCampaign,
    *,
    execute: bool,
) -> Mapping[str, Any]:
    """Preflight the full DAG, then optionally execute it in topological order."""

    repository = Path(__file__).resolve().parents[2]
    producers = {output: step.step_id for step in campaign.steps for output in step.outputs}
    blockers: dict[str, tuple[str, ...]] = {}
    for step in campaign.steps:
        messages = _step_preflight(
            step,
            campaign=campaign,
            repository=repository,
            producers=producers,
            execution_time=False,
        )
        if messages:
            blockers[step.step_id] = messages
    ready = not blockers
    if not execute:
        return freeze_mapping(
            {
                "schema_version": 1,
                "protocol": "agentinputlab.production-campaign-result.v1",
                "campaign_id": campaign.campaign_id,
                "campaign_hash": campaign.campaign_hash,
                "ready": ready,
                "executed": False,
                "required_stages": _REQUIRED_STAGES,
                "step_count": len(campaign.steps),
                "blockers": blockers,
            }
        )
    if not ready:
        return freeze_mapping(
            {
                "schema_version": 1,
                "protocol": "agentinputlab.production-campaign-result.v1",
                "campaign_id": campaign.campaign_id,
                "campaign_hash": campaign.campaign_hash,
                "ready": False,
                "executed": False,
                "execution_blocked": True,
                "required_stages": _REQUIRED_STAGES,
                "step_count": len(campaign.steps),
                "blockers": blockers,
            }
        )

    campaign.workspace.mkdir(parents=True, exist_ok=True)
    receipt_root = campaign.workspace / ".agentinputlab" / "receipts"
    log_root = campaign.workspace / ".agentinputlab" / "logs"
    receipt_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    results: list[Mapping[str, Any]] = []
    for step in campaign.steps:
        receipt_path = receipt_root / f"{step.step_id}.json"
        if campaign.resume and _completed_receipt(
            receipt_path,
            step=step,
            campaign=campaign,
            repository=repository,
        ):
            results.append(
                {
                    "step_id": step.step_id,
                    "stage": step.stage,
                    "status": "resumed",
                    "receipt": str(receipt_path),
                }
            )
            continue
        runtime_blockers = _step_preflight(
            step,
            campaign=campaign,
            repository=repository,
            producers=producers,
            execution_time=True,
        )
        if runtime_blockers:
            results.append(
                {
                    "step_id": step.step_id,
                    "stage": step.stage,
                    "status": "blocked",
                    "blockers": runtime_blockers,
                }
            )
            break
        command = _command(
            step.command,
            campaign=campaign,
            repository=repository,
        )
        cwd = _resolved_path(
            step.cwd,
            campaign=campaign,
            repository=repository,
        )
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join((str(repository / "src"), str(repository))),
            "PYTHONUNBUFFERED": "1",
        }
        for name, value in step.environment.items():
            environment[name] = _expand_argument(
                value,
                campaign=campaign,
                repository=repository,
            )
        for name in step.pass_environment:
            environment[name] = os.environ[name]
        secrets = environment_secret_values(
            environment,
            step.pass_environment,
        )
        attempt_id = f"{time.time_ns()}-{os.getpid()}"
        started = time.time()
        return_code, raw_stdout, raw_stderr, timed_out = _run_command(
            command,
            cwd=cwd,
            env=environment,
            timeout_seconds=step.timeout_seconds,
        )
        duration = time.time() - started
        stdout = redact_secret_text(raw_stdout, secret_values=secrets)
        stderr = redact_secret_text(raw_stderr, secret_values=secrets)
        stdout_path = log_root / f"{step.step_id}.{attempt_id}.stdout.log"
        stderr_path = log_root / f"{step.step_id}.{attempt_id}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        output_hashes = _output_hashes(
            step,
            campaign=campaign,
            repository=repository,
        )
        input_hashes = _input_hashes(
            step,
            campaign=campaign,
            repository=repository,
        )
        status = (
            "completed"
            if return_code == 0 and not timed_out and len(output_hashes) == len(step.outputs)
            else ("timed_out" if timed_out else "failed")
        )
        receipt = {
            "schema_version": 1,
            "protocol": "agentinputlab.production-step-receipt.v1",
            "campaign_id": campaign.campaign_id,
            "campaign_hash": campaign.campaign_hash,
            "step_id": step.step_id,
            "stage": step.stage,
            "status": status,
            "command_hash": content_hash(command),
            "return_code": return_code,
            "timed_out": timed_out,
            "timeout_seconds": step.timeout_seconds,
            "duration_seconds": duration,
            "inputs": dict(input_hashes),
            "outputs": dict(output_hashes),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        _archive_receipt(receipt_path, receipt_root=receipt_root)
        _write_atomic(receipt_path, receipt)
        results.append(
            {
                "step_id": step.step_id,
                "stage": step.stage,
                "status": status,
                "return_code": return_code,
                "receipt": str(receipt_path),
            }
        )
        if status != "completed":
            break
    all_succeeded = len(results) == len(campaign.steps) and all(
        item["status"] in {"completed", "resumed"} for item in results
    )
    return freeze_mapping(
        {
            "schema_version": 1,
            "protocol": "agentinputlab.production-campaign-result.v1",
            "campaign_id": campaign.campaign_id,
            "campaign_hash": campaign.campaign_hash,
            "ready": True,
            "executed": True,
            "all_succeeded": all_succeeded,
            "required_stages": _REQUIRED_STAGES,
            "steps": results,
        }
    )


def _step_preflight(
    step: ProductionStep,
    *,
    campaign: ProductionCampaign,
    repository: Path,
    producers: Mapping[str, str],
    execution_time: bool,
) -> tuple[str, ...]:
    blockers: list[str] = []
    try:
        _command(
            step.command,
            campaign=campaign,
            repository=repository,
            allow_missing_hash=not execution_time,
        )
        cwd = _resolved_path(
            step.cwd,
            campaign=campaign,
            repository=repository,
        )
        if not cwd.is_dir():
            blockers.append(f"cwd is absent: {cwd}")
    except (ArtifactError, ValidationError) as exc:
        blockers.append(str(exc))
    missing_environment = [name for name in step.pass_environment if not os.environ.get(name)]
    if missing_environment:
        blockers.append("missing environment: " + ", ".join(sorted(missing_environment)))
    for name, value in step.environment.items():
        try:
            _expand_argument(
                value,
                campaign=campaign,
                repository=repository,
                allow_missing_hash=not execution_time,
            )
        except (ArtifactError, ValidationError) as exc:
            blockers.append(f"environment {name}: {exc}")
    expected_commit = step.environment.get(_REPOSITORY_COMMIT_ENV)
    if expected_commit is not None:
        try:
            observed_commit = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            blockers.append("cannot resolve the executing repository commit")
        else:
            if observed_commit != expected_commit:
                blockers.append(
                    "repository commit differs: "
                    f"expected {expected_commit}, observed {observed_commit}"
                )
    for item in step.inputs:
        path = _resolved_path(
            item.path,
            campaign=campaign,
            repository=repository,
        )
        if not path.is_file():
            producer = producers.get(item.path)
            if not execution_time and producer in step.requires:
                continue
            blockers.append(f"input is absent: {path}")
            continue
        if item.sha256 is not None and _sha256(path) != item.sha256:
            blockers.append(f"input checksum differs: {path}")
    return tuple(blockers)


def _command(
    raw: Sequence[str],
    *,
    campaign: ProductionCampaign,
    repository: Path,
    allow_missing_hash: bool = False,
) -> tuple[str, ...]:
    expanded = tuple(
        _expand_argument(
            item,
            campaign=campaign,
            repository=repository,
            allow_missing_hash=allow_missing_hash,
        )
        for item in raw
    )
    if expanded[0] == "@agentinputlab":
        return (sys.executable, "-m", "agentinputlab.cli.main", *expanded[1:])
    script = Path(expanded[1])
    if not script.is_absolute():
        script = (repository / script).resolve()
    allowed_roots = (
        repository / "scripts" / "experiments",
        repository / "doc" / "paper_kdd2027" / "scripts",
    )
    if not script.is_file() or not any(script.is_relative_to(root) for root in allowed_roots):
        raise ArtifactError(
            "production Python commands are restricted to reviewed experiment and paper scripts"
        )
    return (sys.executable, str(script), *expanded[2:])


def _expand_argument(
    value: str,
    *,
    campaign: ProductionCampaign,
    repository: Path,
    allow_missing_hash: bool = False,
) -> str:
    match = _SHA_TEMPLATE.fullmatch(value)
    if match is not None:
        path = _resolved_path(
            match.group(1),
            campaign=campaign,
            repository=repository,
        )
        if not path.is_file():
            if allow_missing_hash:
                return "0" * 64
            raise ArtifactError(f"cannot hash absent generated artifact: {path}")
        return _sha256(path)
    return value.replace("{repo}", str(repository)).replace(
        "{workspace}",
        str(campaign.workspace),
    )


def _resolved_path(
    value: str,
    *,
    campaign: ProductionCampaign,
    repository: Path,
) -> Path:
    expanded = value.replace("{repo}", str(repository)).replace(
        "{workspace}",
        str(campaign.workspace),
    )
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_hashes(
    step: ProductionStep,
    *,
    campaign: ProductionCampaign,
    repository: Path,
) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for raw in step.outputs:
        path = _resolved_path(raw, campaign=campaign, repository=repository)
        if path.is_file():
            result[raw] = _sha256(path)
        elif path.is_dir():
            result[raw] = content_hash(
                {
                    str(item.relative_to(path)): _sha256(item)
                    for item in sorted(path.rglob("*"))
                    if item.is_file()
                }
            )
    return MappingProxyType(result)


def _input_hashes(
    step: ProductionStep,
    *,
    campaign: ProductionCampaign,
    repository: Path,
) -> Mapping[str, str]:
    """Return the exact input identities consumed by one production step.

    Inputs are recorded independently of the optional checksum pinned in the
    campaign document.  Generated upstream artifacts normally cannot have a
    checksum before execution, but their observed bytes must still be bound to
    the downstream receipt so that resume cannot reuse stale derived outputs.
    """

    result: dict[str, str] = {}
    for item in step.inputs:
        path = _resolved_path(item.path, campaign=campaign, repository=repository)
        if path.is_file():
            result[item.path] = _sha256(path)
    return MappingProxyType(result)


def _completed_receipt(
    path: Path,
    *,
    step: ProductionStep,
    campaign: ProductionCampaign,
    repository: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(value, Mapping)
        or value.get("campaign_hash") != campaign.campaign_hash
        or value.get("step_id") != step.step_id
        or value.get("status") != "completed"
    ):
        return False
    observed = _output_hashes(
        step,
        campaign=campaign,
        repository=repository,
    )
    observed_inputs = _input_hashes(
        step,
        campaign=campaign,
        repository=repository,
    )
    return (
        dict(observed) == value.get("outputs")
        and len(observed) == len(step.outputs)
        and dict(observed_inputs) == value.get("inputs")
        and len(observed_inputs) == len(step.inputs)
    )


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _archive_receipt(path: Path, *, receipt_root: Path) -> None:
    if not path.is_file():
        return
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    history = receipt_root / "history" / path.stem
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"{time.time_ns()}-{digest}.json"
    if destination.exists():
        raise ArtifactError("production receipt archive collision")
    path.replace(destination)


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, str, str, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return -signal.SIGTERM, stdout, stderr, True


__all__ = [
    "PRODUCTION_CAMPAIGN_PROTOCOL",
    "ProductionCampaign",
    "ProductionInput",
    "ProductionStep",
    "load_production_campaign",
    "production_campaign_hash",
    "run_production_campaign",
]
