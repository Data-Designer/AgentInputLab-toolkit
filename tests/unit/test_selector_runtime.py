from __future__ import annotations

import unittest
from dataclasses import dataclass

from agentinputlab.contracts import (
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
)
from agentinputlab.contracts.errors import RegistryError, ValidationError
from agentinputlab.contracts.lifecycle import InterventionContext as Context
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.registry import AlgorithmRegistry
from agentinputlab.selectors import (
    FeatureAccessDomain,
    FeatureObservation,
    RegistryInterventionFactory,
    RuleSelector,
    SelectionDecision,
    SelectionFeatures,
    SelectorArtifactBundle,
    SelectorArtifactIdentity,
    SelectorArtifactStage,
    SelectorRuntime,
)


@dataclass(frozen=True)
class AcquireFixture:
    method_id: str = "acquire.hyde"
    release_id: str = "acquire.hyde.paper_reimplementation"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def propose(self, state: FrozenAgentState, context: Context) -> InputPatch:
        del context
        return InputPatch(
            source_state_hash=state.state_hash,
            method_id=self.method_id,
            release_id=self.release_id,
            operations=(PatchOperation(PatchOp.APPEND_QUERY, " evidence"),),
            trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
            implementation_identity="selector-runtime-fixture",
            profile_kind="controlled_adapter",
        )


@dataclass(frozen=True)
class FixedExtractor:
    features: SelectionFeatures
    domains: tuple[FeatureAccessDomain, ...] = (FeatureAccessDomain.VISIBLE_PRE_ACTION,)

    def extract(self, state: FrozenAgentState) -> FeatureObservation:
        return FeatureObservation(
            features=self.features,
            state_hash=state.state_hash,
            extractor_id="fixture.visible",
            extractor_revision="v1",
            access_domains=self.domains,
            provenance={"schema": "five-visible-features-v1"},
        )


@dataclass(frozen=True)
class BoundExtractor:
    features: SelectionFeatures
    extractor_id: str = "visible_state_heuristic"
    extractor_revision: str = "v1"
    extractor_hash: str = content_hash({"extractor": "visible-v1"})
    feature_schema_id: str = "family_defect_features.v1"
    feature_schema_hash: str = content_hash({"schema": "five-features-v1"})

    def extract(self, state: FrozenAgentState) -> FeatureObservation:
        return FeatureObservation(
            features=self.features,
            state_hash=state.state_hash,
            extractor_id=self.extractor_id,
            extractor_revision=self.extractor_revision,
            provenance={
                "extractor_hash": self.extractor_hash,
                "schema_id": self.feature_schema_id,
                "schema_hash": self.feature_schema_hash,
            },
        )


@dataclass(frozen=True)
class IgnoringEligibilitySelector:
    release_id: str | None = None

    @property
    def model_hash(self) -> str:
        return content_hash({"selector": "ignoring", "release_id": self.release_id})

    def select(self, state, features, *, eligible_families=None):
        del state, features, eligible_families
        return SelectionDecision(
            SemanticFamily.ACQUIRE,
            self.release_id,
            1.0,
            "fixture deliberately ignores runtime eligibility",
        )


@dataclass(frozen=True)
class InvalidNoOpSelector:
    @property
    def model_hash(self) -> str:
        return content_hash({"selector": "invalid-no-op"})

    def select(self, state, features, *, eligible_families=None):
        del state, features, eligible_families
        return SelectionDecision(
            SemanticFamily.NO_OP,
            "acquire.hyde.paper_reimplementation",
            0.0,
            "fixture deliberately emits an invalid NO_OP release",
        )


def _runtime(extractor: FixedExtractor) -> SelectorRuntime:
    selector = RuleSelector(threshold=0.5, margin=0.1)
    identity = SelectorArtifactIdentity.from_selector(
        selector,
        selector_id="rule.visible.v1",
        revision="v1",
    )
    factory = RegistryInterventionFactory(
        {"acquire.hyde.paper_reimplementation": AcquireFixture},
        registry=AlgorithmRegistry.default(),
    )
    return SelectorRuntime(
        selector=selector,
        selector_identity=identity,
        feature_extractor=extractor,
        factory=factory,
        representative_by_family={SemanticFamily.ACQUIRE: "acquire.hyde.paper_reimplementation"},
    )


class SelectorRuntimeTests(unittest.TestCase):
    def test_rule_route_resolves_locked_representative_and_runs(self) -> None:
        state = FrozenAgentState(query="find evidence")
        record = _runtime(FixedExtractor(SelectionFeatures(retrieval_need_score=0.9))).run(
            state, InterventionContext()
        )

        self.assertIs(record.decision.family, SemanticFamily.ACQUIRE)
        self.assertEqual(record.executed_release_id, "acquire.hyde.paper_reimplementation")
        self.assertNotEqual(record.run.output_state_hash, state.state_hash)
        self.assertEqual(len(record.feature_hash), 64)
        self.assertEqual(len(record.selector_identity.sha256), 64)

    def test_no_op_does_not_require_a_representative(self) -> None:
        state = FrozenAgentState(query="hello")
        record = _runtime(FixedExtractor(SelectionFeatures())).run(
            state,
            InterventionContext(),
        )

        self.assertIs(record.decision.family, SemanticFamily.NO_OP)
        self.assertEqual(record.executed_release_id, "baseline.no_op")
        self.assertEqual(record.run.input_state_hash, record.run.output_state_hash)

    def test_feature_access_fail_closes_on_gold_or_post_action(self) -> None:
        state = FrozenAgentState(query="q")
        runtime = _runtime(
            FixedExtractor(
                SelectionFeatures(retrieval_need_score=0.9),
                (
                    FeatureAccessDomain.VISIBLE_PRE_ACTION,
                    FeatureAccessDomain.EVALUATOR_GOLD,
                ),
            )
        )
        with self.assertRaisesRegex(ValidationError, "visible_pre_action"):
            runtime.run(state, InterventionContext())

    def test_feature_state_identity_is_enforced(self) -> None:
        class WrongStateExtractor:
            def extract(self, state: FrozenAgentState) -> FeatureObservation:
                del state
                return FeatureObservation(
                    SelectionFeatures(),
                    "wrong-state",
                    "fixture",
                    "v1",
                )

        selector = RuleSelector()
        runtime = SelectorRuntime(
            selector=selector,
            selector_identity=SelectorArtifactIdentity.from_selector(
                selector,
                selector_id="rule",
                revision="v1",
            ),
            feature_extractor=WrongStateExtractor(),
            factory=RegistryInterventionFactory(
                {"acquire.hyde.paper_reimplementation": AcquireFixture}
            ),
        )
        with self.assertRaisesRegex(ValidationError, "another input state"):
            runtime.run(FrozenAgentState(query="q"), InterventionContext())

    def test_factory_rejects_non_runnable_release_by_default(self) -> None:
        factory = RegistryInterventionFactory(
            {"acquire.adaptive_rag.selector_replay": AcquireFixture}
        )
        with self.assertRaisesRegex(RegistryError, "status planned"):
            factory.create("acquire.adaptive_rag.selector_replay")

    def test_representative_family_must_match_registry(self) -> None:
        selector = RuleSelector()
        with self.assertRaisesRegex(ValidationError, "does not belong"):
            SelectorRuntime(
                selector=selector,
                selector_identity=SelectorArtifactIdentity.from_selector(
                    selector,
                    selector_id="rule",
                    revision="v1",
                ),
                feature_extractor=FixedExtractor(SelectionFeatures()),
                factory=RegistryInterventionFactory(
                    {"acquire.hyde.paper_reimplementation": AcquireFixture}
                ),
                representative_by_family={
                    SemanticFamily.CLARIFY: "acquire.hyde.paper_reimplementation"
                },
            )

    def test_model_identity_is_verified_at_construction(self) -> None:
        selector = RuleSelector()
        with self.assertRaisesRegex(ValidationError, "does not match"):
            SelectorRuntime(
                selector=selector,
                selector_identity=SelectorArtifactIdentity(
                    "rule.visible.v1",
                    "v1",
                    "0" * 64,
                ),
                feature_extractor=FixedExtractor(SelectionFeatures()),
                factory=RegistryInterventionFactory(
                    {"acquire.hyde.paper_reimplementation": AcquireFixture}
                ),
            )

    def test_runtime_enforces_eligibility_even_if_selector_ignores_it(self) -> None:
        selector = IgnoringEligibilitySelector()
        runtime = SelectorRuntime(
            selector=selector,
            selector_identity=SelectorArtifactIdentity.from_selector(
                selector,
                selector_id="fixture.ignoring",
                revision="v1",
            ),
            feature_extractor=FixedExtractor(SelectionFeatures()),
            factory=RegistryInterventionFactory(
                {"acquire.hyde.paper_reimplementation": AcquireFixture}
            ),
            representative_by_family={
                SemanticFamily.ACQUIRE: "acquire.hyde.paper_reimplementation"
            },
        )
        with self.assertRaisesRegex(ValidationError, "ineligible family"):
            runtime.run(
                FrozenAgentState(query="q"),
                InterventionContext(),
                eligible_families=(SemanticFamily.DECOMPOSE,),
            )

    def test_runtime_rejects_release_other_than_frozen_representative(self) -> None:
        selector = IgnoringEligibilitySelector("acquire.other")
        runtime = SelectorRuntime(
            selector=selector,
            selector_identity=SelectorArtifactIdentity.from_selector(
                selector,
                selector_id="fixture.release",
                revision="v1",
            ),
            feature_extractor=FixedExtractor(SelectionFeatures()),
            factory=RegistryInterventionFactory(
                {"acquire.hyde.paper_reimplementation": AcquireFixture}
            ),
            representative_by_family={
                SemanticFamily.ACQUIRE: "acquire.hyde.paper_reimplementation"
            },
        )
        with self.assertRaisesRegex(RegistryError, "frozen representative"):
            runtime.run(FrozenAgentState(query="q"), InterventionContext())

    def test_runtime_rejects_non_baseline_release_for_no_op(self) -> None:
        selector = InvalidNoOpSelector()
        runtime = SelectorRuntime(
            selector=selector,
            selector_identity=SelectorArtifactIdentity.from_selector(
                selector,
                selector_id="fixture.invalid_no_op",
                revision="v1",
            ),
            feature_extractor=FixedExtractor(SelectionFeatures()),
            factory=RegistryInterventionFactory(
                {"acquire.hyde.paper_reimplementation": AcquireFixture}
            ),
        )
        with self.assertRaisesRegex(ValidationError, "baseline.no_op"):
            runtime.run(FrozenAgentState(query="q"), InterventionContext())

    def test_frozen_bundle_binds_model_extractor_and_representatives(self) -> None:
        selector = RuleSelector(threshold=0.5, margin=0.1)
        extractor = BoundExtractor(SelectionFeatures(retrieval_need_score=0.9))
        bundle = SelectorArtifactBundle.create(
            selector,
            stage=SelectorArtifactStage.FROZEN,
            revision="frozen-v1",
            extractor_id=extractor.extractor_id,
            extractor_revision=extractor.extractor_revision,
            extractor_hash=extractor.extractor_hash,
            feature_schema_id=extractor.feature_schema_id,
            feature_schema_hash=extractor.feature_schema_hash,
            representative_by_family={
                SemanticFamily.ACQUIRE: "acquire.hyde.paper_reimplementation"
            },
            training_table_sha256=content_hash({"table": "fixture"}),
            split_provenance_hash=content_hash({"split": "fixture"}),
            comparison_profile_hash=content_hash({"profile": "fixture"}),
            calibration_receipt={"threshold": 0.5, "margin": 0.1},
        )
        identity = SelectorArtifactIdentity(
            bundle.selector_card_id,
            bundle.revision,
            bundle.bundle_hash,
        )
        runtime = SelectorRuntime(
            selector=selector,
            selector_identity=identity,
            selector_bundle=bundle,
            feature_extractor=extractor,
            factory=RegistryInterventionFactory(
                {"acquire.hyde.paper_reimplementation": AcquireFixture}
            ),
        )
        record = runtime.run(FrozenAgentState(query="find evidence"), InterventionContext())
        self.assertEqual(
            record.executed_release_id,
            "acquire.hyde.paper_reimplementation",
        )
        self.assertIsNotNone(record.feature_observation)
        with self.assertRaisesRegex(ValidationError, "extractor identity"):
            SelectorRuntime(
                selector=selector,
                selector_identity=identity,
                selector_bundle=bundle,
                feature_extractor=BoundExtractor(
                    SelectionFeatures(),
                    extractor_hash="0" * 64,
                ),
                factory=RegistryInterventionFactory(
                    {"acquire.hyde.paper_reimplementation": AcquireFixture}
                ),
            )
        with self.assertRaisesRegex(ValidationError, "feature schema identity"):
            SelectorRuntime(
                selector=selector,
                selector_identity=identity,
                selector_bundle=bundle,
                feature_extractor=BoundExtractor(
                    SelectionFeatures(),
                    feature_schema_hash="0" * 64,
                ),
                factory=RegistryInterventionFactory(
                    {"acquire.hyde.paper_reimplementation": AcquireFixture}
                ),
            )

        class WrongProvenanceExtractor:
            extractor_id = extractor.extractor_id
            extractor_revision = extractor.extractor_revision
            extractor_hash = extractor.extractor_hash
            feature_schema_id = extractor.feature_schema_id
            feature_schema_hash = extractor.feature_schema_hash

            def extract(self, state: FrozenAgentState) -> FeatureObservation:
                return FeatureObservation(
                    features=SelectionFeatures(),
                    state_hash=state.state_hash,
                    extractor_id=self.extractor_id,
                    extractor_revision=self.extractor_revision,
                    provenance={
                        "extractor_hash": self.extractor_hash,
                        "schema_id": self.feature_schema_id,
                        "schema_hash": "0" * 64,
                    },
                )

        wrong_provenance_runtime = SelectorRuntime(
            selector=selector,
            selector_identity=identity,
            selector_bundle=bundle,
            feature_extractor=WrongProvenanceExtractor(),
            factory=RegistryInterventionFactory(
                {"acquire.hyde.paper_reimplementation": AcquireFixture}
            ),
        )
        with self.assertRaisesRegex(ValidationError, "observation provenance"):
            wrong_provenance_runtime.run(
                FrozenAgentState(query="find evidence"),
                InterventionContext(),
            )


if __name__ == "__main__":
    unittest.main()
