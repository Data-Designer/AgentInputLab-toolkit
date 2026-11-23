"""Offline schema, roster, and snapshot tests for the algorithm registry."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from agentinputlab.contracts.enums import ReleaseTier, SemanticFamily, SupportStatus
from agentinputlab.contracts.errors import RegistryError
from agentinputlab.registry import (
    AlgorithmRegistry,
    RegistryDocument,
    load_packaged_registry,
    load_registry_file,
    registry_documents_equal,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "manifests" / "algorithms" / "registry.yaml"


class RegistryTests(unittest.TestCase):
    def test_frozen_roster_and_delivery_counts(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        self.assertEqual(len(registry), 27)
        self.assertEqual(
            len(registry.list_methods(tier=ReleaseTier.CORE)),
            20,
        )
        self.assertEqual(
            len(registry.list_methods(tier=ReleaseTier.OPTIONAL)),
            7,
        )
        expected = {
            SemanticFamily.REFORMULATE: 4,
            SemanticFamily.DECOMPOSE: 5,
            SemanticFamily.ACQUIRE: 5,
            SemanticFamily.CLARIFY: 2,
            SemanticFamily.CURATE: 4,
        }
        for family, count in expected.items():
            self.assertEqual(
                len(
                    registry.list_methods(
                        family=family,
                        tier=ReleaseTier.CORE,
                    )
                ),
                count,
            )
        self.assertEqual(
            len(registry.list_releases(delivery_tier="must_ship")),
            12,
        )

    def test_source_and_packaged_snapshot_are_equal(self) -> None:
        source = load_registry_file(SOURCE)
        packaged = load_packaged_registry()
        self.assertTrue(registry_documents_equal(source, packaged))

    def test_noassertion_sources_cannot_be_vendored(self) -> None:
        document = load_registry_file(SOURCE)
        for algorithm in document.algorithms:
            if algorithm.upstream.license_id == "NOASSERTION":
                self.assertFalse(algorithm.upstream.vendoring_allowed)

    def test_truncated_commit_is_rejected(self) -> None:
        raw = json.loads(SOURCE.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(raw)
        invalid["algorithms"][0]["upstream"]["commit"] = "6d2d192"
        with self.assertRaisesRegex(RegistryError, "40-character Git SHA"):
            RegistryDocument.from_mapping(invalid)

    def test_experimental_release_without_entrypoint_is_rejected(self) -> None:
        raw = json.loads(SOURCE.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(raw)
        target = next(
            release
            for algorithm in invalid["algorithms"]
            for release in algorithm["releases"]
            if release["status"] == "experimental"
        )
        target["entrypoint"] = None
        with self.assertRaisesRegex(
            RegistryError,
            "experimental releases require an entrypoint",
        ):
            RegistryDocument.from_mapping(invalid)

    def test_any_release_without_entrypoint_requires_a_blocker(self) -> None:
        raw = json.loads(SOURCE.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(raw)
        target = next(
            release
            for algorithm in invalid["algorithms"]
            for release in algorithm["releases"]
            if release["entrypoint"] is None
        )
        target["blocking_reasons"] = []
        with self.assertRaisesRegex(
            RegistryError,
            "releases without an entrypoint require blocking_reasons",
        ):
            RegistryDocument.from_mapping(invalid)

    def test_release_availability_is_explicit_and_locked_releases_are_explained(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        releases = [release for _method, release in registry.list_releases()]
        self.assertEqual(sum(item.entrypoint_available for item in releases), 86)
        self.assertEqual(sum(not item.entrypoint_available for item in releases), 1)
        self.assertTrue(
            all(item.blocking_reasons for item in releases if not item.entrypoint_available)
        )
        self.assertTrue(
            all(item.availability == "locked" for item in releases if not item.entrypoint_available)
        )
        self.assertTrue(all(item.availability == "runnable" for item in releases if item.runnable))
        self.assertTrue(all(item.blocker for item in releases if not item.runnable))

    def test_acon_training_entrypoint_is_not_a_runtime_acceptance_claim(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        _method, release = registry.get_release("training.acon.guideline_optimization")
        self.assertTrue(callable(registry.resolve_entrypoint(release.id)))
        self.assertEqual(release.status, SupportStatus.RETRAIN_REQUIRED)
        self.assertEqual(release.availability, "gated")
        self.assertFalse(release.runnable)
        self.assertTrue(release.blocking_reasons)

    def test_smartsearch_native_recipe_entrypoint_retains_training_acceptance_gate(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        _method, release = registry.get_release("training.smartsearch.sft_dpo_grpo")
        self.assertTrue(callable(registry.resolve_entrypoint(release.id)))
        self.assertEqual(release.status, SupportStatus.RETRAIN_REQUIRED)
        self.assertEqual(release.availability, "gated")
        self.assertFalse(release.runnable)
        self.assertTrue(release.blocking_reasons)

    def test_release_lookup_preserves_parent_method(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        method, release = registry.get_release("acquire.ircot.controlled")
        self.assertEqual(method.method_id, "acquire.ircot")
        self.assertTrue(release.must_ship)

    def test_every_must_ship_profile_has_a_resolvable_entrypoint(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        releases = registry.list_releases(delivery_tier="must_ship")
        self.assertEqual(len(releases), 12)
        for _method, release in releases:
            with self.subTest(release_id=release.id):
                resolved = registry.resolve_entrypoint(release.id)
                self.assertTrue(callable(resolved))

    def test_every_method_has_an_integration_entrypoint_and_all_resolve(self) -> None:
        """Code coverage is independent of runtime/parity maturity status."""

        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        for method in registry.list_methods():
            entrypoint_releases = tuple(
                release for release in method.releases if release.entrypoint is not None
            )
            with self.subTest(method_id=method.method_id):
                self.assertTrue(entrypoint_releases)
            for release in entrypoint_releases:
                with self.subTest(release_id=release.id):
                    self.assertTrue(callable(registry.resolve_entrypoint(release.id)))

    def test_former_contract_only_methods_expose_honest_method_level_profiles(self) -> None:
        registry = AlgorithmRegistry(load_registry_file(SOURCE))
        expected = {
            "acquire.adaptive_rag.controlled_reimplementation": SupportStatus.EXTERNAL_ONLY,
            "acquire.r2ag.controlled_same_candidates": SupportStatus.RETRAIN_REQUIRED,
            "recipe.rqrag.method_tree_search": SupportStatus.EXPERIMENTAL,
            "recipe.rqrag.common_support_queries": SupportStatus.EXPERIMENTAL,
            "recipe.mce.sandboxed_offline": SupportStatus.EXTERNAL_ONLY,
            "recipe.mce.text_artifact_eval": SupportStatus.EXPERIMENTAL,
        }
        for release_id, status in expected.items():
            with self.subTest(release_id=release_id):
                _method, release = registry.get_release(release_id)
                self.assertEqual(release.status, status)
                self.assertTrue(callable(registry.resolve_entrypoint(release_id)))


if __name__ == "__main__":
    unittest.main()
