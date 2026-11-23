from __future__ import annotations

import unittest

from agentinputlab import FrozenAgentState
from agentinputlab.contracts import (
    DependencyBinding,
    DependencyRole,
    InterventionContext,
    ProfileKind,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.runner import (
    BalanceAuditor,
    ComparisonArm,
    ComparisonProfile,
    context_dependency_fingerprints,
    run_cache_key,
)


def binding(slot: str, revision: str) -> DependencyBinding:
    return DependencyBinding(
        slot=slot,
        role=DependencyRole.NUISANCE,
        implementation_id=f"{slot}-impl",
        revision=revision,
    )


class BalanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ComparisonProfile(
            profile_id="retrieval-controlled",
            kind=ProfileKind.CONTROLLED_ADAPTER,
            required_slots=("corpus", "index", "retriever", "top_k"),
        )

    def test_equal_resolved_dependencies_pass(self) -> None:
        bindings = tuple(binding(slot, "v1") for slot in self.profile.required_slots)
        report = BalanceAuditor().audit(
            self.profile,
            (
                ComparisonArm("a", "r1", bindings),
                ComparisonArm("b", "r2", bindings),
            ),
        )
        self.assertTrue(report.eligible)

    def test_retriever_difference_fails(self) -> None:
        left = tuple(binding(slot, "v1") for slot in self.profile.required_slots)
        right = tuple(
            binding(slot, "v2" if slot == "retriever" else "v1")
            for slot in self.profile.required_slots
        )
        report = BalanceAuditor().audit(
            self.profile,
            (ComparisonArm("a", "r1", left), ComparisonArm("b", "r2", right)),
        )
        self.assertFalse(report.eligible)
        self.assertIn("retriever differs across arms", report.differences)

    def test_missing_varied_slot_fails_closed(self) -> None:
        profile = ComparisonProfile(
            profile_id="controlled-treatment",
            kind=ProfileKind.CONTROLLED_ADAPTER,
            required_slots=("model",),
            varied_slots=("query_optimizer",),
        )
        report = BalanceAuditor().audit(
            profile,
            (
                ComparisonArm("noop", "baseline.no_op", (binding("model", "v1"),)),
                ComparisonArm(
                    "method",
                    "method.v1",
                    (
                        binding("model", "v1"),
                        binding("query_optimizer", "method.v1"),
                    ),
                ),
            ),
        )

        self.assertFalse(report.eligible)
        self.assertIn("query_optimizer missing in noop", report.missing)

    def test_backend_identity_changes_cache_key(self) -> None:
        state = FrozenAgentState(query="q")
        common = dict(
            state=state,
            release_id="r",
            profile_id="p",
            retriever_fingerprint={"id": "ret"},
            seed=0,
        )
        left = run_cache_key(backend_fingerprint={"model": "a"}, **common)
        right = run_cache_key(backend_fingerprint={"model": "b"}, **common)
        self.assertNotEqual(left, right)

    def test_dependency_complete_cache_key_tracks_dense_runtime_slots(self) -> None:
        class Runtime:
            def __init__(self, revision: str) -> None:
                self.revision = revision

            def fingerprint(self):
                return {"revision": self.revision}

        def dependencies(embedder_revision: str, searcher_revision: str) -> dict[str, str]:
            return context_dependency_fingerprints(
                InterventionContext(
                    dense_embedder=Runtime(embedder_revision),
                    vector_searcher=Runtime(searcher_revision),
                    seed=7,
                )
            )

        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=7,
        )
        baseline = run_cache_key(dependency_hashes=dependencies("embed-v1", "search-v1"), **common)
        changed_embedder = run_cache_key(
            dependency_hashes=dependencies("embed-v2", "search-v1"), **common
        )
        changed_searcher = run_cache_key(
            dependency_hashes=dependencies("embed-v1", "search-v2"), **common
        )

        self.assertNotEqual(baseline, changed_embedder)
        self.assertNotEqual(baseline, changed_searcher)

    def test_dependency_hash_order_does_not_change_cache_key(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=0,
        )
        first = run_cache_key(
            dependency_hashes={"dense_embedder": "a" * 64, "vector_searcher": "b" * 64},
            **common,
        )
        second = run_cache_key(
            dependency_hashes={"vector_searcher": "b" * 64, "dense_embedder": "a" * 64},
            **common,
        )

        self.assertEqual(first, second)

    def test_explicit_and_legacy_dependency_slots_cannot_overlap(self) -> None:
        with self.assertRaisesRegex(ValidationError, "both dependency_hashes and legacy"):
            run_cache_key(
                state=FrozenAgentState(query="q"),
                release_id="r",
                profile_id="p",
                seed=0,
                dependency_hashes={"backend": "a" * 64},
                backend_fingerprint={"model": "a"},
            )

    def test_legacy_and_explicit_dependency_hashes_are_equivalent(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=3,
        )
        backend = {"model": "model-a", "revision": "v1"}
        retriever = {"index": "index-a", "revision": "v2"}

        legacy = run_cache_key(
            backend_fingerprint=backend,
            retriever_fingerprint=retriever,
            **common,
        )
        explicit = run_cache_key(
            dependency_hashes={
                "backend": content_hash(backend),
                "retriever": content_hash(retriever),
            },
            **common,
        )

        self.assertEqual(legacy, explicit)

    def test_legacy_dependency_fingerprints_must_be_non_empty(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=0,
        )
        for slot in ("backend_fingerprint", "retriever_fingerprint"):
            with (
                self.subTest(slot=slot),
                self.assertRaisesRegex(ValidationError, "non-empty mapping"),
            ):
                run_cache_key(**common, **{slot: {}})

    def test_legacy_fingerprint_rejects_non_string_keys_recursively(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=0,
        )
        valid = run_cache_key(backend_fingerprint={"nested": {"1": "value"}}, **common)
        self.assertIsInstance(valid, str)

        with self.assertRaisesRegex(ValidationError, "canonically serializable"):
            run_cache_key(backend_fingerprint={"nested": {1: "value"}}, **common)

    def test_cache_dependency_hashes_fail_closed_on_invalid_values(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=0,
        )
        with self.subTest("hash"), self.assertRaisesRegex(ValidationError, "SHA-256"):
            run_cache_key(dependency_hashes={"backend": "not-a-hash"}, **common)
        with self.subTest("slot"), self.assertRaisesRegex(ValidationError, "slots"):
            run_cache_key(dependency_hashes={"": "a" * 64}, **common)
        with self.subTest("seed"), self.assertRaisesRegex(ValidationError, "does not match"):
            run_cache_key(dependency_hashes={"seed": "f" * 64}, **common)

    def test_matching_context_seed_hash_is_validated_and_normalized(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=11,
        )
        seed_hash = context_dependency_fingerprints(InterventionContext(seed=11))["seed"]

        self.assertEqual(
            run_cache_key(dependency_hashes={"seed": seed_hash}, **common),
            run_cache_key(**common),
        )

    def test_cache_seed_must_be_a_non_negative_integer(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
        )
        for invalid_seed in (-1, True, 1.0):
            with (
                self.subTest(seed=invalid_seed),
                self.assertRaisesRegex(ValidationError, "non-negative integer"),
            ):
                run_cache_key(seed=invalid_seed, **common)

    def test_cache_extra_is_canonical_and_fail_closed(self) -> None:
        common = dict(
            state=FrozenAgentState(query="q"),
            release_id="r",
            profile_id="p",
            seed=0,
        )
        self.assertEqual(run_cache_key(extra=None, **common), run_cache_key(extra={}, **common))

        for invalid in ([], "", 0):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValidationError, "mapping or None"),
            ):
                run_cache_key(extra=invalid, **common)

        with (
            self.subTest("numeric key"),
            self.assertRaisesRegex(ValidationError, "canonically serializable"),
        ):
            run_cache_key(extra={1: "value"}, **common)
        with (
            self.subTest("nested numeric key"),
            self.assertRaisesRegex(ValidationError, "canonically serializable"),
        ):
            run_cache_key(extra={"nested": {1: "value"}}, **common)
        with (
            self.subTest("non-finite float"),
            self.assertRaisesRegex(ValidationError, "canonically serializable"),
        ):
            run_cache_key(extra={"value": float("nan")}, **common)

    def test_context_runtime_fingerprint_canonicalization_fails_closed(self) -> None:
        class Runtime:
            def __init__(self, identity) -> None:
                self.identity = identity

            def fingerprint(self):
                return self.identity

        valid = context_dependency_fingerprints(
            InterventionContext(backend=Runtime({"nested": {"1": "value"}}))
        )
        self.assertIn("backend", valid)

        invalid_identities = (
            {1: "value"},
            {"nested": {1: "value"}},
            {"unsupported": object()},
            {"non_finite": float("nan")},
        )
        for identity in invalid_identities:
            with (
                self.subTest(identity=identity),
                self.assertRaisesRegex(ValidationError, "canonically serializable"),
            ):
                context_dependency_fingerprints(InterventionContext(backend=Runtime(identity)))

    def test_varied_runtime_slot_must_exist_in_every_arm(self) -> None:
        profile = ComparisonProfile(
            profile_id="retriever-factorial",
            kind=ProfileKind.FACTORIAL_ADAPTER,
            required_slots=("backend",),
            varied_slots=("retriever",),
        )
        bindings = (
            binding("backend", "v1"),
            binding("retriever", "v1"),
        )

        report = BalanceAuditor().audit(
            profile,
            (
                ComparisonArm("a", "r1", bindings),
                ComparisonArm("b", "r2", bindings),
            ),
            runtime_fingerprints={
                "a": {"backend": "a" * 64, "retriever": "b" * 64},
                "b": {"backend": "a" * 64},
            },
        )

        self.assertFalse(report.eligible)
        self.assertIn("retriever runtime missing in b", report.missing)

    def test_declared_fingerprint_cannot_spoof_a_runtime_component(self) -> None:
        with self.assertRaisesRegex(ValidationError, "actual context field"):
            context_dependency_fingerprints(
                InterventionContext(
                    run_config={
                        "dependency_fingerprints": {"retriever": {"id": "not-an-actual-retriever"}}
                    }
                )
            )

    def test_runtime_fingerprint_values_must_be_sha256_strings(self) -> None:
        bindings = (binding("backend", "v1"),)
        with self.assertRaisesRegex(ValidationError, "SHA-256"):
            BalanceAuditor().audit(
                ComparisonProfile(
                    profile_id="bad-runtime-hash",
                    kind=ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("backend",),
                ),
                (
                    ComparisonArm("a", "r1", bindings),
                    ComparisonArm("b", "r2", bindings),
                ),
                runtime_fingerprints={
                    "a": {"backend": "not-a-hash"},
                    "b": {"backend": "not-a-hash"},
                },
            )

    def test_external_runtime_is_fingerprinted_from_the_actual_context_slot(self) -> None:
        class Runtime:
            def __init__(self, revision: str) -> None:
                self.revision = revision

            def fingerprint(self):
                return {"worker_id": "fixture", "revision": self.revision}

            def run(self, request):
                raise AssertionError(request)

        first = context_dependency_fingerprints(InterventionContext(external_runtime=Runtime("v1")))
        second = context_dependency_fingerprints(
            InterventionContext(external_runtime=Runtime("v2"))
        )
        self.assertIn("external_runtime", first)
        self.assertNotEqual(first["external_runtime"], second["external_runtime"])

    def test_declared_fingerprint_cannot_spoof_external_runtime(self) -> None:
        with self.assertRaisesRegex(ValidationError, "actual context field"):
            context_dependency_fingerprints(
                InterventionContext(
                    run_config={
                        "dependency_fingerprints": {"external_runtime": {"worker_id": "spoofed"}}
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
