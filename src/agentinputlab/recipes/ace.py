"""ACE offline recipe delegating optimization to the pinned upstream worker."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import (
    ExternalDependencyError,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import (
    content_hash,
    deep_freeze_jsonable,
    freeze_mapping,
    to_jsonable,
)
from agentinputlab.integrations.external_workers.ace import (
    ACE_UPSTREAM_COMMIT,
    ACEPlaybook,
    ACEWorkerRequest,
    ACEWorkerResponse,
    PlaybookBullet,
)
from agentinputlab.integrations.external_workers.base import WorkerClient


@dataclass(frozen=True, slots=True)
class ACEExample:
    example_id: str
    input: Any
    expected_output: Any = None
    split: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValidationError("ACE examples require a stable example_id")
        if self.split not in {"train", "validation", "test"}:
            raise ValidationError("ACE example split must be train/validation/test")
        try:
            frozen_input = deep_freeze_jsonable(self.input)
            frozen_expected_output = deep_freeze_jsonable(self.expected_output)
            frozen_metadata = freeze_mapping(self.metadata)
            content_hash(frozen_input)
            content_hash(frozen_expected_output)
        except (TypeError, ValueError) as exc:
            raise ValidationError("ACE example values must be canonically serializable") from exc
        object.__setattr__(self, "input", frozen_input)
        object.__setattr__(self, "expected_output", frozen_expected_output)
        object.__setattr__(self, "metadata", frozen_metadata)

    @property
    def input_hash(self) -> str:
        return content_hash(self.input)

    @classmethod
    def from_value(cls, value: Any, *, split: str) -> ACEExample:
        if isinstance(value, cls):
            if value.split != split:
                raise ValidationError(
                    f"ACE example {value.example_id!r} is labeled {value.split}, expected {split}"
                )
            return value
        if not isinstance(value, Mapping):
            raise ValidationError("ACE examples must be ACEExample or mappings")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("ACE example metadata must be a mapping")
        try:
            return cls(
                example_id=str(value["example_id"]),
                input=value["input"],
                expected_output=value.get("expected_output"),
                split=split,
                metadata=dict(metadata),
            )
        except KeyError as exc:
            raise ValidationError("ACE example lacks example_id or input") from exc

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class SplitAudit:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_input_hashes: tuple[str, ...]
    validation_input_hashes: tuple[str, ...]
    test_input_hashes: tuple[str, ...]

    @property
    def audit_hash(self) -> str:
        return content_hash(self)


def validate_split_isolation(
    train: Sequence[ACEExample],
    validation: Sequence[ACEExample],
    test: Sequence[ACEExample] = (),
) -> SplitAudit:
    """Fail on ID or exact-input overlap between any ACE data split."""

    groups = {
        "train": tuple(train),
        "validation": tuple(validation),
        "test": tuple(test),
    }
    for split, examples in groups.items():
        if any(example.split != split for example in examples):
            raise ValidationError(f"ACE {split} collection contains a mislabeled example")
        ids = [example.example_id for example in examples]
        hashes = [example.input_hash for example in examples]
        if len(ids) != len(set(ids)):
            raise ValidationError(f"ACE {split} split contains duplicate IDs")
        if len(hashes) != len(set(hashes)):
            raise ValidationError(f"ACE {split} split contains duplicate inputs")
    names = tuple(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_ids = {item.example_id for item in groups[left]}
            right_ids = {item.example_id for item in groups[right]}
            if left_ids & right_ids:
                raise ValidationError(f"ACE {left}/{right} example IDs overlap")
            left_hashes = {item.input_hash for item in groups[left]}
            right_hashes = {item.input_hash for item in groups[right]}
            if left_hashes & right_hashes:
                raise ValidationError(f"ACE {left}/{right} inputs overlap")
    return SplitAudit(
        train_ids=tuple(item.example_id for item in groups["train"]),
        validation_ids=tuple(item.example_id for item in groups["validation"]),
        test_ids=tuple(item.example_id for item in groups["test"]),
        train_input_hashes=tuple(item.input_hash for item in groups["train"]),
        validation_input_hashes=tuple(item.input_hash for item in groups["validation"]),
        test_input_hashes=tuple(item.input_hash for item in groups["test"]),
    )


@dataclass(frozen=True, slots=True)
class ACEControls:
    max_rounds: int = 4
    max_model_calls: int = 100
    max_tokens: int = 200_000
    max_cost_usd: float = 100.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("max_rounds", "max_model_calls", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValidationError(f"ACE {name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValidationError("ACE seed must be a non-negative integer")
        if (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or (isinstance(self.max_cost_usd, float) and not math.isfinite(self.max_cost_usd))
            or self.max_cost_usd < 0
        ):
            raise ValidationError("ACE max_cost_usd must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ACEContextArtifact:
    artifact_id: str
    release_id: str
    playbook: ACEPlaybook
    selected_iteration: int
    split_audit_hash: str
    upstream_commit: str
    worker_fingerprint: Mapping[str, Any]
    model_fingerprints: Mapping[str, Mapping[str, Any]]
    controls: Mapping[str, Any]
    usage: Mapping[str, int | float]
    iteration_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.release_id != "recipe.ace.offline":
            raise ValidationError("ACE artifact release identity mismatch")
        if self.upstream_commit != ACE_UPSTREAM_COMMIT:
            raise ValidationError("ACE artifact upstream revision mismatch")
        if not isinstance(self.playbook, ACEPlaybook):
            raise ValidationError("ACE artifact playbook must be an ACEPlaybook")
        if type(self.selected_iteration) is not int or self.selected_iteration < 0:
            raise ValidationError("ACE selected_iteration must be a non-negative integer")
        if any(
            not isinstance(getattr(self, name), Mapping)
            for name in (
                "worker_fingerprint",
                "model_fingerprints",
                "controls",
                "usage",
            )
        ) or any(not isinstance(value, Mapping) for value in self.model_fingerprints.values()):
            raise ValidationError("ACE artifact fingerprints, controls, and usage must be mappings")
        if not isinstance(self.iteration_hashes, (list, tuple)) or any(
            not isinstance(value, str) for value in self.iteration_hashes
        ):
            raise ValidationError("ACE iteration_hashes must be an array of strings")
        object.__setattr__(
            self,
            "playbook",
            ACEPlaybook(
                tuple(
                    PlaybookBullet(
                        b.bullet_id,
                        b.content,
                        b.helpful,
                        b.harmful,
                        metadata=freeze_mapping(b.metadata),
                    )
                    for b in self.playbook.bullets
                )
            ),
        )
        for name in ("worker_fingerprint", "model_fingerprints", "controls", "usage"):
            object.__setattr__(self, name, freeze_mapping(getattr(self, name)))
        object.__setattr__(self, "iteration_hashes", tuple(self.iteration_hashes))
        if self.artifact_id != content_hash(self.identity_payload()):
            raise ValidationError("ACE artifact content identity mismatch")

    def identity_payload(self) -> dict[str, Any]:
        """Original fit identity; a pinned file checksum separately protects usage."""
        return {
            "release_id": self.release_id,
            "playbook_hash": self.playbook.playbook_hash,
            "selected_iteration": self.selected_iteration,
            "split_audit_hash": self.split_audit_hash,
            "upstream_commit": self.upstream_commit,
            "worker_fingerprint": self.worker_fingerprint,
            "model_fingerprints": self.model_fingerprints,
            "controls": self.controls,
            "iteration_hashes": self.iteration_hashes,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ACEContextArtifact:
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValidationError("ACE artifact fields are invalid")
        book = value["playbook"]
        if (
            not isinstance(book, Mapping)
            or set(book) != {"bullets"}
            or not isinstance(book["bullets"], (list, tuple))
        ):
            raise ValidationError("ACE artifact playbook schema is invalid")
        bullets = []
        for item in book["bullets"]:
            if (
                not isinstance(item, Mapping)
                or set(item) != {f.name for f in fields(PlaybookBullet)}
                or any(
                    not isinstance(item.get(k), str) or not item[k].strip()
                    for k in ("bullet_id", "content")
                )
                or any(type(item.get(k)) is not int or item[k] < 0 for k in ("helpful", "harmful"))
                or not isinstance(item.get("metadata"), Mapping)
            ):
                raise ValidationError("ACE artifact bullet schema is invalid")
            bullets.append(PlaybookBullet(**dict(item)))
        malformed = False
        try:
            return cls(**{**dict(value), "playbook": ACEPlaybook(tuple(bullets))})
        except ValidationError:
            raise
        except (AttributeError, TypeError, ValueError, RecursionError):
            malformed = True
        if malformed:
            raise ValidationError("ACE artifact values are malformed")

    @classmethod
    def from_json(
        cls, path: str | Path, *, expected_sha256: str, max_bytes: int = 4 * 1024 * 1024
    ) -> ACEContextArtifact:
        return cls.from_mapping(
            load_pinned_json_mapping(path, expected_sha256=expected_sha256, max_bytes=max_bytes)
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class ACEOfflineRecipe:
    """Run train/validation-only ACE and return an eval-only playbook artifact."""

    client: WorkerClient
    initial_playbook: ACEPlaybook = field(default_factory=ACEPlaybook)
    controls: ACEControls = field(default_factory=ACEControls)
    method_id: str = "ace"
    release_id: str = "recipe.ace.offline"

    def __post_init__(self) -> None:
        if not isinstance(self.client, WorkerClient):
            raise ExternalDependencyError("ACE offline recipe requires a WorkerClient")

    def fit(
        self,
        train_examples: tuple[Any, ...],
        *,
        validation_examples: tuple[Any, ...] = (),
        context: InterventionContext,
    ) -> ACEContextArtifact:
        train = tuple(ACEExample.from_value(item, split="train") for item in train_examples)
        validation = tuple(
            ACEExample.from_value(item, split="validation") for item in validation_examples
        )
        if not train or not validation:
            raise ValidationError("ACE offline recipe requires non-empty train and validation")
        if self.controls.seed != context.seed:
            raise ValidationError(
                "ACE controls.seed must equal InterventionContext.seed; "
                "the recipe does not silently override either seed"
            )
        audit = validate_split_isolation(train, validation)
        fingerprints = _model_fingerprints(context)
        controls = asdict(self.controls)
        serialized_train = tuple(item.as_dict() for item in train)
        serialized_validation = tuple(item.as_dict() for item in validation)
        request_base = {
            "train_ids": audit.train_ids,
            "validation_ids": audit.validation_ids,
            "train_examples": serialized_train,
            "validation_examples": serialized_validation,
            "initial_playbook_hash": self.initial_playbook.playbook_hash,
            "controls": controls,
            "model_fingerprints": fingerprints,
            "split_audit_hash": audit.audit_hash,
            "upstream_commit": ACE_UPSTREAM_COMMIT,
        }
        request = ACEWorkerRequest(
            request_id=content_hash(request_base),
            train_examples=serialized_train,
            validation_examples=serialized_validation,
            initial_playbook=self.initial_playbook,
            controls=controls,
            model_fingerprints=fingerprints,
            split_audit_hash=audit.audit_hash,
        )
        raw = self.client.call(request.as_dict())
        response = ACEWorkerResponse.from_mapping(raw, request=request)
        iteration_hashes = tuple(
            content_hash(
                {
                    "index": iteration.index,
                    "operations": iteration.operations,
                    "candidate_playbook_hash": iteration.candidate_playbook_hash,
                    "validation_score": iteration.validation_score,
                }
            )
            for iteration in response.iterations
        )
        artifact_base = {
            "release_id": self.release_id,
            "playbook_hash": response.final_playbook.playbook_hash,
            "selected_iteration": response.selected_iteration,
            "split_audit_hash": audit.audit_hash,
            "upstream_commit": response.upstream_commit,
            "worker_fingerprint": self.client.fingerprint(),
            "model_fingerprints": fingerprints,
            "controls": controls,
            "iteration_hashes": iteration_hashes,
        }
        return ACEContextArtifact(
            artifact_id=content_hash(artifact_base),
            release_id=self.release_id,
            playbook=response.final_playbook,
            selected_iteration=response.selected_iteration,
            split_audit_hash=audit.audit_hash,
            upstream_commit=response.upstream_commit,
            worker_fingerprint=dict(self.client.fingerprint()),
            model_fingerprints=fingerprints,
            controls=controls,
            usage=response.usage,
            iteration_hashes=iteration_hashes,
        )


def _model_fingerprints(
    context: InterventionContext,
) -> Mapping[str, Mapping[str, Any]]:
    configured = context.run_config.get("ace_model_fingerprints")
    if configured is not None:
        if not isinstance(configured, Mapping):
            raise ValidationError("ace_model_fingerprints must be a mapping")
        fingerprints: dict[str, Mapping[str, Any]] = {}
        for role in ("target", "reflector", "curator"):
            value = configured.get(role)
            if not isinstance(value, Mapping):
                raise ValidationError(f"ACE is missing the {role} model fingerprint")
            fingerprints[role] = dict(value)
        return fingerprints
    if context.backend is None:
        raise ExternalDependencyError(
            "ACE requires explicit target/reflector/curator fingerprints or a backend"
        )
    try:
        shared = dict(context.backend.fingerprint())
    except AttributeError as exc:
        raise ExternalDependencyError("ACE backend does not expose a fingerprint") from exc
    return {role: dict(shared) for role in ("target", "reflector", "curator")}


def playbook_from_strings(values: Sequence[str]) -> ACEPlaybook:
    """Convenience constructor with stable, position-derived bullet IDs."""

    return ACEPlaybook(
        tuple(
            PlaybookBullet(bullet_id=f"bullet-{index:04d}", content=value)
            for index, value in enumerate(values)
        )
    )
