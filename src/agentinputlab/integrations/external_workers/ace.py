"""Schema and validators for the pinned ACE offline worker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

ACE_UPSTREAM_COMMIT = "bcb7cea0504afad6f55fec4845dd4864c9f9eee7"
_OPERATIONS = {"add", "edit", "delete", "merge"}


@dataclass(frozen=True, slots=True)
class PlaybookBullet:
    bullet_id: str
    content: str
    helpful: int = 0
    harmful: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bullet_id.strip() or not self.content.strip():
            raise ValidationError("ACE playbook bullets require ID and content")
        if self.helpful < 0 or self.harmful < 0:
            raise ValidationError("ACE helpful/harmful counters must be non-negative")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PlaybookBullet:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("ACE bullet metadata must be a mapping")
        try:
            return cls(
                bullet_id=str(value["bullet_id"]),
                content=str(value["content"]),
                helpful=int(value.get("helpful", 0)),
                harmful=int(value.get("harmful", 0)),
                metadata=dict(metadata),
            )
        except KeyError as exc:
            raise ValidationError("ACE bullet lacks a required field") from exc


@dataclass(frozen=True, slots=True)
class ACEPlaybook:
    bullets: tuple[PlaybookBullet, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bullets", tuple(self.bullets))
        ids = [bullet.bullet_id for bullet in self.bullets]
        if len(ids) != len(set(ids)):
            raise ValidationError("ACE playbook bullet IDs must be unique")

    @property
    def playbook_hash(self) -> str:
        return content_hash(self)

    @classmethod
    def from_sequence(cls, value: Sequence[Mapping[str, Any]]) -> ACEPlaybook:
        return cls(tuple(PlaybookBullet.from_mapping(item) for item in value))


@dataclass(frozen=True, slots=True)
class ACEDiffOperation:
    operation: str
    bullet_id: str
    content: str = ""
    source_ids: tuple[str, ...] = ()
    helpful: int | None = None
    harmful: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in _OPERATIONS:
            raise ValidationError(f"unsupported ACE curator operation: {self.operation}")
        if not self.bullet_id.strip():
            raise ValidationError("ACE curator operation requires bullet_id")
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValidationError("ACE merge source_ids must be unique")
        for counter in (self.helpful, self.harmful):
            if counter is not None and counter < 0:
                raise ValidationError("ACE helpful/harmful counters must be non-negative")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ACEDiffOperation:
        sources = value.get("source_ids", ())
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            raise ValidationError("ACE source_ids must be a sequence")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("ACE diff metadata must be a mapping")
        try:
            return cls(
                operation=str(value["operation"]),
                bullet_id=str(value["bullet_id"]),
                content=str(value.get("content", "")),
                source_ids=tuple(str(item) for item in sources),
                helpful=_optional_counter(value, "helpful"),
                harmful=_optional_counter(value, "harmful"),
                metadata=dict(metadata),
            )
        except KeyError as exc:
            raise ValidationError("ACE diff lacks a required field") from exc
        except (TypeError, ValueError) as exc:
            raise ValidationError("ACE diff counters are malformed") from exc


def apply_playbook_diff(
    playbook: ACEPlaybook,
    operations: Sequence[ACEDiffOperation],
) -> ACEPlaybook:
    """Replay curator operations exactly and fail on ambiguous diffs."""

    bullets = {bullet.bullet_id: bullet for bullet in playbook.bullets}
    order = [bullet.bullet_id for bullet in playbook.bullets]
    for operation in operations:
        if operation.operation == "add":
            if operation.bullet_id in bullets:
                raise ValidationError("ACE add targets an existing bullet")
            bullet = _bullet_from_operation(operation)
            bullets[bullet.bullet_id] = bullet
            order.append(bullet.bullet_id)
        elif operation.operation == "edit":
            if operation.bullet_id not in bullets:
                raise ValidationError("ACE edit targets a missing bullet")
            old = bullets[operation.bullet_id]
            if not operation.content.strip():
                raise ValidationError("ACE edit requires non-empty content")
            bullets[operation.bullet_id] = PlaybookBullet(
                bullet_id=operation.bullet_id,
                content=operation.content,
                helpful=(old.helpful if operation.helpful is None else operation.helpful),
                harmful=(old.harmful if operation.harmful is None else operation.harmful),
                metadata={**dict(old.metadata), **dict(operation.metadata)},
            )
        elif operation.operation == "delete":
            if operation.bullet_id not in bullets:
                raise ValidationError("ACE delete targets a missing bullet")
            del bullets[operation.bullet_id]
            order.remove(operation.bullet_id)
        elif operation.operation == "merge":
            if len(operation.source_ids) < 2:
                raise ValidationError("ACE merge requires at least two source bullets")
            if not set(operation.source_ids) <= set(bullets):
                raise ValidationError("ACE merge references a missing source bullet")
            if operation.bullet_id in bullets and operation.bullet_id not in operation.source_ids:
                raise ValidationError("ACE merge target conflicts with an unrelated bullet")
            if not operation.content.strip():
                raise ValidationError("ACE merge requires non-empty content")
            insertion = min(order.index(source) for source in operation.source_ids)
            helpful = (
                sum(bullets[source].helpful for source in operation.source_ids)
                if operation.helpful is None
                else operation.helpful
            )
            harmful = (
                sum(bullets[source].harmful for source in operation.source_ids)
                if operation.harmful is None
                else operation.harmful
            )
            for source in operation.source_ids:
                del bullets[source]
                order.remove(source)
            bullet = PlaybookBullet(
                bullet_id=operation.bullet_id,
                content=operation.content,
                helpful=helpful,
                harmful=harmful,
                metadata=dict(operation.metadata),
            )
            bullets[bullet.bullet_id] = bullet
            order.insert(insertion, bullet.bullet_id)
    return ACEPlaybook(tuple(bullets[bullet_id] for bullet_id in order))


def _bullet_from_operation(operation: ACEDiffOperation) -> PlaybookBullet:
    return PlaybookBullet(
        bullet_id=operation.bullet_id,
        content=operation.content,
        helpful=0 if operation.helpful is None else operation.helpful,
        harmful=0 if operation.harmful is None else operation.harmful,
        metadata=dict(operation.metadata),
    )


@dataclass(frozen=True, slots=True)
class ACEWorkerIteration:
    index: int
    operations: tuple[ACEDiffOperation, ...]
    candidate_playbook_hash: str
    validation_score: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ACEWorkerIteration:
        raw_operations = value.get("operations", ())
        if not isinstance(raw_operations, Sequence) or isinstance(raw_operations, (str, bytes)):
            raise ValidationError("ACE iteration operations must be a sequence")
        try:
            return cls(
                index=int(value["index"]),
                operations=tuple(
                    ACEDiffOperation.from_mapping(_mapping(item, "ACE operation"))
                    for item in raw_operations
                ),
                candidate_playbook_hash=str(value["candidate_playbook_hash"]),
                validation_score=_finite_float(
                    value["validation_score"],
                    "ACE validation_score",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("ACE iteration fields are malformed") from exc


@dataclass(frozen=True, slots=True)
class ACEWorkerRequest:
    request_id: str
    train_examples: tuple[Mapping[str, Any], ...]
    validation_examples: tuple[Mapping[str, Any], ...]
    initial_playbook: ACEPlaybook
    controls: Mapping[str, Any]
    model_fingerprints: Mapping[str, Mapping[str, Any]]
    split_audit_hash: str
    upstream_commit: str = ACE_UPSTREAM_COMMIT
    protocol_version: str = "agentinputlab.ace.offline.v1"

    def __post_init__(self) -> None:
        if not self.request_id or not self.split_audit_hash:
            raise ValidationError("ACE worker request requires request and split-audit IDs")
        if not self.train_examples or not self.validation_examples:
            raise ValidationError("ACE offline worker requires train and validation examples")
        if set(self.model_fingerprints) != {"target", "reflector", "curator"}:
            raise ValidationError("ACE model_fingerprints must bind target, reflector, and curator")
        if self.upstream_commit != ACE_UPSTREAM_COMMIT:
            raise ValidationError("ACE request does not target the pinned upstream commit")
        object.__setattr__(self, "controls", _frozen_mapping(self.controls))
        object.__setattr__(
            self,
            "model_fingerprints",
            MappingProxyType(
                {
                    str(role): _frozen_mapping(fingerprint)
                    for role, fingerprint in self.model_fingerprints.items()
                }
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class ACEWorkerResponse:
    request_id: str
    final_playbook: ACEPlaybook
    iterations: tuple[ACEWorkerIteration, ...]
    selected_iteration: int
    usage: Mapping[str, int | float]
    split_audit_hash: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    upstream_commit: str
    protocol_version: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        request: ACEWorkerRequest,
    ) -> ACEWorkerResponse:
        if str(value.get("status", "")) != "ok":
            code = str(value.get("failure_code", "WORKER_FAILURE"))
            raise ExternalDependencyError(f"ACE worker failed with {code}")
        if str(value.get("request_id", "")) != request.request_id:
            raise ValidationError("ACE response request_id mismatch")
        if str(value.get("upstream_commit", "")) != request.upstream_commit:
            raise ValidationError("ACE worker upstream revision mismatch")
        if str(value.get("protocol_version", "")) != request.protocol_version:
            raise ValidationError("ACE worker protocol version mismatch")
        if str(value.get("split_audit_hash", "")) != request.split_audit_hash:
            raise ValidationError("ACE worker split-audit hash mismatch")
        train_ids = _response_ids(value, "train_ids")
        validation_ids = _response_ids(value, "validation_ids")
        expected_train_ids = _request_example_ids(request.train_examples, "train")
        expected_validation_ids = _request_example_ids(
            request.validation_examples,
            "validation",
        )
        if train_ids != expected_train_ids:
            raise ValidationError("ACE worker train IDs differ from the audited split")
        if validation_ids != expected_validation_ids:
            raise ValidationError("ACE worker validation IDs differ from the audited split")
        raw_iterations = value.get("iterations", ())
        if not isinstance(raw_iterations, Sequence) or isinstance(raw_iterations, (str, bytes)):
            raise ValidationError("ACE response iterations must be a sequence")
        iterations = tuple(
            ACEWorkerIteration.from_mapping(_mapping(item, "ACE iteration"))
            for item in raw_iterations
        )
        if not iterations:
            raise ValidationError("ACE worker returned no optimization iterations")
        if len(iterations) > int(request.controls.get("max_rounds", len(iterations))):
            raise ValidationError("ACE worker exceeded max_rounds")
        snapshots: dict[int, ACEPlaybook] = {}
        current = request.initial_playbook
        for expected, iteration in enumerate(iterations):
            if iteration.index != expected:
                raise ValidationError("ACE iteration indices must be contiguous and zero-based")
            current = apply_playbook_diff(current, iteration.operations)
            if current.playbook_hash != iteration.candidate_playbook_hash:
                raise ValidationError("ACE candidate playbook hash does not match its diff")
            snapshots[iteration.index] = current
        try:
            selected = int(value["selected_iteration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("ACE selected_iteration is malformed") from exc
        if selected not in snapshots:
            raise ValidationError("ACE selected_iteration does not exist")
        best_score = max(iteration.validation_score for iteration in iterations)
        selected_score = iterations[selected].validation_score
        if selected_score != best_score:
            raise ValidationError("ACE selected iteration is not validation-best")
        raw_playbook = value.get("final_playbook")
        if not isinstance(raw_playbook, Sequence) or isinstance(raw_playbook, (str, bytes)):
            raise ValidationError("ACE final_playbook must be a sequence")
        final_playbook = ACEPlaybook.from_sequence(
            tuple(_mapping(item, "ACE final bullet") for item in raw_playbook)
        )
        if final_playbook.playbook_hash != snapshots[selected].playbook_hash:
            raise ValidationError("ACE final playbook differs from the selected snapshot")
        usage = value.get("usage", {})
        if not isinstance(usage, Mapping):
            raise ValidationError("ACE usage must be a mapping")
        numeric_usage: dict[str, int | float] = {}
        for key, amount in usage.items():
            if (
                not isinstance(amount, (int, float))
                or isinstance(amount, bool)
                or not isfinite(float(amount))
                or amount < 0
            ):
                raise ValidationError("ACE usage values must be non-negative numbers")
            numeric_usage[str(key)] = amount
        _validate_usage_limits(numeric_usage, request.controls)
        return cls(
            request_id=request.request_id,
            final_playbook=final_playbook,
            iterations=iterations,
            selected_iteration=selected,
            usage=numeric_usage,
            split_audit_hash=request.split_audit_hash,
            train_ids=train_ids,
            validation_ids=validation_ids,
            upstream_commit=request.upstream_commit,
            protocol_version=request.protocol_version,
        )


def _validate_usage_limits(
    usage: Mapping[str, int | float],
    controls: Mapping[str, Any],
) -> None:
    pairs = (
        ("model_calls", "max_model_calls"),
        ("tokens", "max_tokens"),
        ("cost_usd", "max_cost_usd"),
    )
    for usage_key, control_key in pairs:
        if (
            usage_key in usage
            and control_key in controls
            and float(usage[usage_key]) > float(controls[control_key])
        ):
            raise ValidationError(f"ACE worker exceeded {control_key}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping")
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _optional_counter(value: Mapping[str, Any], key: str) -> int | None:
    raw = value.get(key)
    return None if raw is None else int(raw)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be numeric") from exc
    if not isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def _response_ids(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = value.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValidationError(f"ACE response {key} must be a sequence")
    ids = tuple(str(item) for item in raw)
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValidationError(f"ACE response {key} contains empty or duplicate IDs")
    return ids


def _request_example_ids(
    examples: Sequence[Mapping[str, Any]],
    split: str,
) -> tuple[str, ...]:
    try:
        ids = tuple(str(item["example_id"]) for item in examples)
    except KeyError as exc:
        raise ValidationError(f"ACE {split} request examples lack example_id") from exc
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValidationError(f"ACE {split} request IDs are empty or duplicated")
    return ids
