from __future__ import annotations

import unittest

from agentinputlab.backends.llm import CallableBackend
from agentinputlab.contracts import InterventionContext, ValidationError
from agentinputlab.integrations.external_workers.ace import (
    ACE_UPSTREAM_COMMIT,
    ACEDiffOperation,
    ACEPlaybook,
    PlaybookBullet,
    apply_playbook_diff,
)
from agentinputlab.recipes.ace import (
    ACEControls,
    ACEExample,
    ACEOfflineRecipe,
    validate_split_isolation,
)


class FakeACEWorker:
    worker_id = "ace-fixture"

    def __init__(self, mutate=None):
        self.payload = None
        self.mutate = mutate

    def call(self, payload):
        self.payload = payload
        first = ACEPlaybook((PlaybookBullet("b1", "Prefer explicit evidence."),))
        second = ACEPlaybook((PlaybookBullet("b1", "Verify evidence before answering."),))
        response = {
            "status": "ok",
            "request_id": payload["request_id"],
            "protocol_version": "agentinputlab.ace.offline.v1",
            "upstream_commit": ACE_UPSTREAM_COMMIT,
            "split_audit_hash": payload["split_audit_hash"],
            "train_ids": [item["example_id"] for item in payload["train_examples"]],
            "validation_ids": [item["example_id"] for item in payload["validation_examples"]],
            "iterations": [
                {
                    "index": 0,
                    "operations": [
                        {
                            "operation": "add",
                            "bullet_id": "b1",
                            "content": "Prefer explicit evidence.",
                        }
                    ],
                    "candidate_playbook_hash": first.playbook_hash,
                    "validation_score": 0.4,
                },
                {
                    "index": 1,
                    "operations": [
                        {
                            "operation": "edit",
                            "bullet_id": "b1",
                            "content": "Verify evidence before answering.",
                        }
                    ],
                    "candidate_playbook_hash": second.playbook_hash,
                    "validation_score": 0.8,
                },
            ],
            "selected_iteration": 1,
            "final_playbook": [
                {
                    "bullet_id": "b1",
                    "content": "Verify evidence before answering.",
                }
            ],
            "usage": {"model_calls": 12, "tokens": 2000, "cost_usd": 1.25},
        }
        if self.mutate:
            self.mutate(response)
        return response

    def fingerprint(self):
        return {"worker_id": self.worker_id, "image_digest": "sha256:fixture"}


def example(example_id, value, split):
    return ACEExample(example_id=example_id, input=value, split=split)


class ACESplitTests(unittest.TestCase):
    def test_examples_detach_nested_values_and_keep_hashes_stable(self):
        input_value = {"q": ["original"]}
        expected_output = {"answers": ["one"]}
        metadata = {"source": {"labels": ["fixture"]}}
        item = ACEExample(
            example_id="train-1",
            input=input_value,
            expected_output=expected_output,
            split="train",
            metadata=metadata,
        )
        input_hash = item.input_hash
        serialized = item.as_dict()

        input_value["q"].append("late")
        expected_output["answers"].append("late")
        metadata["source"]["labels"].append("late")
        self.assertEqual(item.input_hash, input_hash)
        self.assertEqual(item.as_dict(), serialized)
        self.assertEqual(item.input["q"], ("original",))
        self.assertEqual(item.expected_output["answers"], ("one",))
        self.assertEqual(item.metadata["source"]["labels"], ("fixture",))

    def test_split_audit_is_stable_and_disjoint(self):
        audit = validate_split_isolation(
            (example("train-1", {"q": "a"}, "train"),),
            (example("val-1", {"q": "b"}, "validation"),),
            (example("test-1", {"q": "c"}, "test"),),
        )
        self.assertEqual(audit.train_ids, ("train-1",))
        self.assertEqual(len(audit.audit_hash), 64)

    def test_equal_input_across_splits_is_leakage(self):
        with self.assertRaisesRegex(ValidationError, "inputs overlap"):
            validate_split_isolation(
                (example("train-1", {"q": "same"}, "train"),),
                (example("val-1", {"q": "same"}, "validation"),),
            )


class ACEDiffTests(unittest.TestCase):
    def test_add_edit_merge_delete_replay(self):
        initial = ACEPlaybook(
            (
                PlaybookBullet("a", "A"),
                PlaybookBullet("b", "B"),
            )
        )
        merged = apply_playbook_diff(
            initial,
            (
                ACEDiffOperation("add", "c", "C"),
                ACEDiffOperation("edit", "c", "C2"),
                ACEDiffOperation(
                    "merge",
                    "ab",
                    "AB",
                    source_ids=("a", "b"),
                ),
                ACEDiffOperation("delete", "c"),
            ),
        )
        self.assertEqual(
            [(item.bullet_id, item.content) for item in merged.bullets], [("ab", "AB")]
        )

    def test_invalid_edit_fails_closed(self):
        with self.assertRaisesRegex(ValidationError, "missing bullet"):
            apply_playbook_diff(
                ACEPlaybook(),
                (ACEDiffOperation("edit", "missing", "text"),),
            )

    def test_edit_preserves_counters_when_worker_omits_them(self):
        revised = apply_playbook_diff(
            ACEPlaybook((PlaybookBullet("a", "A", helpful=3, harmful=1),)),
            (ACEDiffOperation("edit", "a", "A revised"),),
        )
        self.assertEqual(revised.bullets[0].helpful, 3)
        self.assertEqual(revised.bullets[0].harmful, 1)


class ACERecipeTests(unittest.TestCase):
    def test_offline_recipe_returns_frozen_selected_playbook(self):
        worker = FakeACEWorker()
        recipe = ACEOfflineRecipe(client=worker, controls=ACEControls(seed=11))
        artifact = recipe.fit(
            ({"example_id": "t1", "input": {"q": "train"}},),
            validation_examples=({"example_id": "v1", "input": {"q": "validation"}},),
            context=InterventionContext(
                backend=CallableBackend(
                    lambda _: "unused",
                    model_id="shared",
                    revision="fixture-v1",
                    callable_id="tests.ace.shared.v1",
                ),
                seed=11,
            ),
        )
        self.assertEqual(artifact.selected_iteration, 1)
        self.assertEqual(
            artifact.playbook.bullets[0].content,
            "Verify evidence before answering.",
        )
        self.assertEqual(artifact.upstream_commit, ACE_UPSTREAM_COMMIT)
        self.assertEqual(worker.payload["controls"]["seed"], 11)
        self.assertNotIn("test_examples", worker.payload)
        self.assertEqual(
            set(worker.payload["model_fingerprints"]),
            {"target", "reflector", "curator"},
        )

    def test_recipe_rejects_conflicting_control_and_context_seeds(self):
        with self.assertRaisesRegex(ValidationError, "does not silently override"):
            ACEOfflineRecipe(
                client=FakeACEWorker(),
                controls=ACEControls(seed=7),
            ).fit(
                ({"example_id": "t1", "input": "train"},),
                validation_examples=({"example_id": "v1", "input": "valid"},),
                context=InterventionContext(
                    backend=CallableBackend(
                        lambda _: "unused",
                        revision="fixture-v1",
                        callable_id="tests.ace.seed-conflict.v1",
                    ),
                    seed=8,
                ),
            )

    def test_worker_cannot_select_non_best_iteration(self):
        worker = FakeACEWorker(
            lambda response: response.update(
                {
                    "selected_iteration": 0,
                    "final_playbook": [
                        {
                            "bullet_id": "b1",
                            "content": "Prefer explicit evidence.",
                        }
                    ],
                }
            )
        )
        with self.assertRaisesRegex(ValidationError, "not validation-best"):
            ACEOfflineRecipe(client=worker).fit(
                ({"example_id": "t1", "input": "train"},),
                validation_examples=({"example_id": "v1", "input": "valid"},),
                context=InterventionContext(
                    backend=CallableBackend(
                        lambda _: "unused",
                        revision="fixture-v1",
                        callable_id="tests.ace.split-overlap.v1",
                    )
                ),
            )

    def test_worker_must_echo_audited_split(self):
        worker = FakeACEWorker(lambda response: response.update({"validation_ids": ["t1"]}))
        with self.assertRaisesRegex(ValidationError, "validation IDs"):
            ACEOfflineRecipe(client=worker).fit(
                ({"example_id": "t1", "input": "train"},),
                validation_examples=({"example_id": "v1", "input": "valid"},),
                context=InterventionContext(
                    backend=CallableBackend(
                        lambda _: "unused",
                        revision="fixture-v1",
                        callable_id="tests.ace.split-audit.v1",
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
