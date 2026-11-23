"""Leakage-aware selector-to-intervention orchestration.

This module deliberately keeps feature extraction, release selection, and
intervention construction as separate auditable boundaries.  A selector score
is not executable until the chosen release is validated against the canonical
registry and constructed by an explicitly locked factory.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentinputlab.contracts.enums import SemanticFamily, StrEnum, SupportStatus
from agentinputlab.contracts.errors import RegistryError, ValidationError
from agentinputlab.contracts.lifecycle import Intervention, InterventionContext
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.registry import AlgorithmRegistry
from agentinputlab.runner import InterventionRunner, RunRecord

from .base import SelectionDecision, SelectionFeatures, Selector
from .no_op import NoOpIntervention

if TYPE_CHECKING:
    from .artifacts import SelectorArtifactBundle

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_RUNNABLE_STATUSES = frozenset(
    {
        SupportStatus.EXPERIMENTAL,
        SupportStatus.SUPPORTED,
    }
)
_SELECTOR_ACTIONS = frozenset(
    {
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)


class FeatureAccessDomain(StrEnum):
    """Domains a feature extractor may attest that it consulted."""

    VISIBLE_PRE_ACTION = "visible_pre_action"
    EVALUATOR_GOLD = "evaluator_gold"
    POST_ACTION = "post_action"
    OTHER_ARM_OUTCOME = "other_arm_outcome"


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One feature vector bound to a visible pre-action state and extractor."""

    features: SelectionFeatures
    state_hash: str
    extractor_id: str
    extractor_revision: str
    access_domains: tuple[FeatureAccessDomain, ...] = (FeatureAccessDomain.VISIBLE_PRE_ACTION,)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.features, SelectionFeatures):
            raise ValidationError("feature observation requires SelectionFeatures")
        for label, value in (
            ("state_hash", self.state_hash),
            ("extractor_id", self.extractor_id),
            ("extractor_revision", self.extractor_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"feature observation {label} must be non-empty")
        domains = tuple(self.access_domains)
        if not domains or any(not isinstance(item, FeatureAccessDomain) for item in domains):
            raise ValidationError("feature access domains must be typed and non-empty")
        if set(domains) != {FeatureAccessDomain.VISIBLE_PRE_ACTION}:
            raise ValidationError(
                "selector features may access visible_pre_action state only; "
                "gold, post-action data, and other-arm outcomes are forbidden"
            )
        object.__setattr__(self, "access_domains", domains)
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))

    @property
    def feature_hash(self) -> str:
        return content_hash(self)


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extract features from the physically gold-free state supplied by the runner."""

    def extract(self, state: FrozenAgentState) -> FeatureObservation: ...


@dataclass(frozen=True, slots=True)
class SelectorArtifactIdentity:
    selector_id: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.selector_id, str) or not self.selector_id.strip():
            raise ValidationError("selector_id must be non-empty")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValidationError("selector revision must be non-empty")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValidationError("selector identity requires a lowercase SHA-256")

    @classmethod
    def from_selector(
        cls,
        selector: Selector,
        *,
        selector_id: str,
        revision: str,
    ) -> SelectorArtifactIdentity:
        declared = getattr(selector, "model_hash", None)
        if isinstance(declared, str) and _SHA256.fullmatch(declared):
            digest = declared
        else:
            try:
                digest = content_hash(selector)
            except TypeError as exc:
                raise ValidationError(
                    "selector is not canonically hashable; provide a frozen selector artifact"
                ) from exc
        return cls(selector_id=selector_id, revision=revision, sha256=digest)


@runtime_checkable
class InterventionFactory(Protocol):
    def create(self, release_id: str) -> Intervention: ...


@dataclass(frozen=True, slots=True)
class RegistryInterventionFactory:
    """Construct only explicitly registered and locally approved releases."""

    builders: Mapping[str, Callable[[], Intervention]]
    registry: AlgorithmRegistry = field(default_factory=AlgorithmRegistry.default)
    allowed_statuses: frozenset[SupportStatus] = _DEFAULT_RUNNABLE_STATUSES

    def __post_init__(self) -> None:
        builders = dict(self.builders)
        if not builders:
            raise ValidationError("intervention factory requires at least one builder")
        statuses = frozenset(self.allowed_statuses)
        if not statuses or any(not isinstance(item, SupportStatus) for item in statuses):
            raise ValidationError("allowed_statuses must contain SupportStatus values")
        for release_id, builder in builders.items():
            if not isinstance(release_id, str) or not release_id.strip() or not callable(builder):
                raise ValidationError("factory builders require release IDs and callables")
            self.registry.get_release(release_id)
        object.__setattr__(self, "builders", MappingProxyType(builders))
        object.__setattr__(self, "allowed_statuses", statuses)

    def create(self, release_id: str) -> Intervention:
        method, release = self.registry.get_release(release_id)
        if release.status not in self.allowed_statuses:
            raise RegistryError(
                f"selector release {release_id} has status {release.status.value}, "
                "which is not enabled by this factory"
            )
        try:
            intervention = self.builders[release_id]()
        except KeyError as exc:
            raise RegistryError(f"selector release {release_id} has no locked builder") from exc
        if not isinstance(intervention, Intervention):
            raise ValidationError("release builder did not return an Intervention")
        if intervention.release_id != release_id:
            raise ValidationError("release builder returned a different release identity")
        if intervention.method_id != method.method_id:
            raise ValidationError("release builder returned a different method identity")
        if intervention.family is not method.semantic_family:
            raise ValidationError("release builder returned a different semantic family")
        return intervention


@dataclass(frozen=True, slots=True)
class SelectorRunRecord:
    input_state_hash: str
    feature_hash: str
    selector_identity: SelectorArtifactIdentity
    decision: SelectionDecision
    executed_release_id: str
    run: RunRecord
    feature_observation: FeatureObservation | None = None

    def __post_init__(self) -> None:
        if self.input_state_hash != self.run.input_state_hash:
            raise ValidationError("selector and intervention input state hashes differ")
        if self.executed_release_id != self.run.release_id:
            raise ValidationError("selector and intervention release IDs differ")
        if self.feature_observation is not None:
            if not isinstance(self.feature_observation, FeatureObservation):
                raise ValidationError("selector run feature_observation has an invalid type")
            if self.feature_observation.feature_hash != self.feature_hash:
                raise ValidationError("selector run feature hash differs from its observation")
            if self.feature_observation.state_hash != self.input_state_hash:
                raise ValidationError("selector run observation is bound to another input state")


class SelectorRuntime:
    """Extract, select, resolve, and execute one intervention or explicit NO_OP."""

    def __init__(
        self,
        *,
        selector: Selector,
        selector_identity: SelectorArtifactIdentity,
        feature_extractor: FeatureExtractor,
        factory: InterventionFactory,
        representative_by_family: Mapping[SemanticFamily, str] | None = None,
        registry: AlgorithmRegistry | None = None,
        runner: InterventionRunner | None = None,
        selector_bundle: SelectorArtifactBundle | None = None,
    ) -> None:
        if not isinstance(selector, Selector):
            raise ValidationError("selector does not satisfy the Selector protocol")
        if not isinstance(feature_extractor, FeatureExtractor):
            raise ValidationError("feature extractor does not satisfy its protocol")
        if not isinstance(factory, InterventionFactory):
            raise ValidationError("intervention factory does not satisfy its protocol")
        if not isinstance(selector_identity, SelectorArtifactIdentity):
            raise ValidationError("selector_identity must be SelectorArtifactIdentity")
        from .artifacts import (
            SelectorArtifactBundle,
            SelectorArtifactStage,
            selector_model_hash,
        )

        expected_identity = SelectorArtifactIdentity.from_selector(
            selector,
            selector_id=selector_identity.selector_id,
            revision=selector_identity.revision,
        )
        extractor_id = getattr(feature_extractor, "extractor_id", None)
        extractor_revision = getattr(feature_extractor, "extractor_revision", None)
        extractor_hash = getattr(feature_extractor, "extractor_hash", None)
        feature_schema_id = getattr(feature_extractor, "feature_schema_id", None)
        feature_schema_hash = getattr(feature_extractor, "feature_schema_hash", None)
        if selector_bundle is None:
            if selector_identity.sha256 != expected_identity.sha256:
                raise ValidationError("selector identity does not match the supplied model")
        else:
            if not isinstance(selector_bundle, SelectorArtifactBundle):
                raise ValidationError("selector_bundle must be SelectorArtifactBundle")
            if selector_bundle.stage is not SelectorArtifactStage.FROZEN:
                raise ValidationError("SelectorRuntime requires a frozen selector bundle")
            if (
                selector_identity.selector_id != selector_bundle.selector_card_id
                or selector_identity.revision != selector_bundle.revision
                or selector_identity.sha256 != selector_bundle.bundle_hash
            ):
                raise ValidationError("selector identity does not match its frozen bundle")
            if selector_model_hash(selector) != selector_bundle.model_hash:
                raise ValidationError("selector model does not match its frozen bundle")
            if (
                extractor_id != selector_bundle.extractor_id
                or extractor_revision != selector_bundle.extractor_revision
                or extractor_hash != selector_bundle.extractor_hash
            ):
                raise ValidationError(
                    "feature extractor identity does not match the frozen selector bundle"
                )
            if (
                feature_schema_id != selector_bundle.feature_schema_id
                or feature_schema_hash != selector_bundle.feature_schema_hash
            ):
                raise ValidationError(
                    "feature schema identity does not match the frozen selector bundle"
                )
        representatives = dict(representative_by_family or {})
        if any(
            not isinstance(family, SemanticFamily)
            or family in {SemanticFamily.NO_OP, SemanticFamily.COMPOSITE}
            or not isinstance(release_id, str)
            or not release_id.strip()
            for family, release_id in representatives.items()
        ):
            raise ValidationError("representative mapping requires intervention families and IDs")
        model_representatives = getattr(selector, "release_by_family", None)
        if isinstance(model_representatives, Mapping):
            declared = dict(model_representatives)
            overlap = set(representatives) & set(declared)
            if any(representatives[family] != declared[family] for family in overlap):
                raise ValidationError(
                    "runtime representatives differ from selector release mapping"
                )
            representatives = {**declared, **representatives}
        if selector_bundle is not None:
            bundle_representatives = dict(selector_bundle.representative_by_family)
            if representatives and representatives != bundle_representatives:
                raise ValidationError(
                    "runtime representatives differ from the frozen selector bundle"
                )
            representatives = bundle_representatives
        self.selector = selector
        self.selector_identity = selector_identity
        self.selector_bundle = selector_bundle
        self.feature_extractor = feature_extractor
        self.extractor_id = extractor_id
        self.extractor_revision = extractor_revision
        self.extractor_hash = extractor_hash
        self.feature_schema_id = feature_schema_id
        self.feature_schema_hash = feature_schema_hash
        self.factory = factory
        self.representative_by_family = MappingProxyType(representatives)
        self.registry = registry or AlgorithmRegistry.default()
        self.runner = runner or InterventionRunner()
        for family, release_id in representatives.items():
            method, _ = self.registry.get_release(release_id)
            if method.semantic_family is not family:
                raise ValidationError(
                    f"representative {release_id} does not belong to {family.value}"
                )

    def run(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectorRunRecord:
        observation = self.feature_extractor.extract(state)
        if not isinstance(observation, FeatureObservation):
            raise ValidationError("feature extractor did not return FeatureObservation")
        if observation.state_hash != state.state_hash:
            raise ValidationError("feature observation is bound to another input state")
        if (self.extractor_id is not None and observation.extractor_id != self.extractor_id) or (
            self.extractor_revision is not None
            and observation.extractor_revision != self.extractor_revision
        ):
            raise ValidationError(
                "feature observation identity differs from the configured extractor"
            )
        if self.selector_bundle is not None:
            expected_provenance = {
                "extractor_hash": self.selector_bundle.extractor_hash,
                "schema_id": self.selector_bundle.feature_schema_id,
                "schema_hash": self.selector_bundle.feature_schema_hash,
            }
            if any(
                observation.provenance.get(key) != value
                for key, value in expected_provenance.items()
            ):
                raise ValidationError(
                    "feature observation provenance differs from the frozen selector bundle"
                )
        decision = self.selector.select(
            state,
            observation.features,
            eligible_families=eligible_families,
        )
        if not isinstance(decision, SelectionDecision):
            raise ValidationError("selector did not return SelectionDecision")
        if decision.family not in {*_SELECTOR_ACTIONS, SemanticFamily.NO_OP}:
            raise ValidationError(f"selector chose unsupported family {decision.family.value}")
        if eligible_families is not None:
            eligible = tuple(eligible_families)
            if any(not isinstance(family, SemanticFamily) for family in eligible):
                raise ValidationError("eligible_families must contain SemanticFamily values")
            if decision.family is not SemanticFamily.NO_OP and decision.family not in eligible:
                raise ValidationError(f"selector chose ineligible family {decision.family.value}")
        if decision.family is SemanticFamily.NO_OP:
            if decision.release_id not in {None, "baseline.no_op"}:
                raise ValidationError("NO_OP selector decisions may reference baseline.no_op only")
            intervention: Intervention = NoOpIntervention()
        else:
            locked_release = self.representative_by_family.get(decision.family)
            if (
                decision.release_id is not None
                and locked_release is not None
                and decision.release_id != locked_release
            ):
                raise RegistryError(
                    f"selector chose release {decision.release_id} instead of frozen "
                    f"representative {locked_release}"
                )
            release_id = decision.release_id or locked_release
            if release_id is None:
                raise RegistryError(
                    f"selector chose {decision.family.value} without a release or locked representative"
                )
            method, _ = self.registry.get_release(release_id)
            if method.semantic_family is not decision.family:
                raise RegistryError(
                    f"selector release {release_id} belongs to {method.semantic_family.value}, "
                    f"not {decision.family.value}"
                )
            intervention = self.factory.create(release_id)
        run = self.runner.run(intervention, state, context)
        return SelectorRunRecord(
            input_state_hash=state.state_hash,
            feature_hash=observation.feature_hash,
            selector_identity=self.selector_identity,
            decision=decision,
            executed_release_id=intervention.release_id,
            run=run,
            feature_observation=observation,
        )


__all__ = [
    "FeatureAccessDomain",
    "FeatureExtractor",
    "FeatureObservation",
    "InterventionFactory",
    "RegistryInterventionFactory",
    "SelectorArtifactIdentity",
    "SelectorRunRecord",
    "SelectorRuntime",
]
