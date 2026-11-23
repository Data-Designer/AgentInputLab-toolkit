from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentinputlab.benchmarks import (
    TAU3_OFFICIAL_SOURCE_REVISION,
    BenchmarkBatchRunner,
    CallerAssertedTau3EvaluationBridge,
    PreparedBenchmark,
    PreparedTau3BankingBundle,
    Tau3BankingCheckpoint,
    Tau3BankingCheckpointAdapter,
    Tau3EvaluationResult,
)
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.contracts.state import ContextBlock, FrozenAgentState, Message


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    content = ("\n".join(canonical_json(row) for row in rows) + "\n").encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _prepared(root: Path) -> PreparedBenchmark:
    task_spec = {
        "annotations": None,
        "description": {"purpose": "fixture"},
        "evaluation_criteria": {
            "actions": [{"name": "private_action", "arguments": {"secret": "gold"}}],
            "communicate_info": [],
            "reward_basis": ["DB"],
        },
        "id": "task_001",
        "initial_state": {"initialization_data": {"database_secret": "hidden-db-row"}},
        "required_documents": ["doc_001"],
        "user_scenario": {"instructions": "hidden-user-simulator-script", "persona": None},
        "user_tools": ["private_user_tool"],
    }
    hashes = {
        "inputs.jsonl": _write_jsonl(
            root / "inputs.jsonl",
            [
                {
                    "checkpoint_required": True,
                    "id": "task_001",
                    "visible_metadata": {
                        "benchmark_cell": "tau3.banking_knowledge",
                        "source_split": "base",
                        "source_task_id": "task_001",
                    },
                }
            ],
        ),
        "gold.jsonl": _write_jsonl(
            root / "gold.jsonl",
            [{"id": "task_001", "task_spec": task_spec}],
        ),
        "corpus.jsonl": _write_jsonl(
            root / "corpus.jsonl",
            [
                {"id": "doc_001", "text": "Visible document one.", "title": "One"},
                {"id": "doc_002", "text": "Visible document two.", "title": "Two"},
            ],
        ),
        "audit.jsonl": _write_jsonl(
            root / "audit.jsonl",
            [
                {
                    "database": {"database_secret": "evaluator-only-database"},
                    "id": "tau3-banking-knowledge-source-audit",
                    "source_revision": TAU3_OFFICIAL_SOURCE_REVISION,
                }
            ],
        ),
    }
    return PreparedBenchmark(
        asset_id="fixture.tau3.banking",
        cell_id="tau3.banking_knowledge",
        processor="tau3_banking_tar_v1",
        processor_revision="1",
        source_revision=TAU3_OFFICIAL_SOURCE_REVISION,
        source_sha256="a" * 64,
        split="base",
        root=root,
        counts={"audit": 1, "corpus": 2, "documents": 2, "gold": 1, "inputs": 1, "tasks": 1},
        file_sha256=hashes,
    )


def _checkpoint(bundle: PreparedTau3BankingBundle, *, full: bool = False) -> Tau3BankingCheckpoint:
    messages = (
        (
            Message(role="user", content="Earlier visible request."),
            Message(role="assistant", content="Earlier visible reply."),
            Message(role="user", content="Please compare the account options."),
        )
        if full
        else (Message(role="user", content="Please compare the account options."),)
    )
    return Tau3BankingCheckpoint(
        checkpoint_id="checkpoint-001",
        task_id="task_001",
        query="Please compare the account options.",
        instruction="Help the user through the captured official tool environment.",
        visible_messages=messages,
        visible_context=(
            (
                ContextBlock(text="Visible retrieved policy A.", block_id="a"),
                ContextBlock(text="Visible retrieved policy B.", block_id="b"),
            )
            if full
            else ()
        ),
        tool_names=("KB_search", "transfer_to_human_agents") if full else ("agent_status",),
        runtime_snapshot_id="1" * 64,
        retrieval_revision="2" * 64,
        corpus_revision=bundle.corpus_revision,
        replay_identity="3" * 64,
        retrieval_config_id="alltools@fixture",
        structural_capabilities=(
            {
                "clarification_roundtrip_available": True,
                "state_restore_available": True,
            }
            if full
            else {}
        ),
    )


def _bridge(callback: Any) -> CallerAssertedTau3EvaluationBridge:
    return CallerAssertedTau3EvaluationBridge(
        bridge_id="fixture.external-evaluator",
        bridge_revision="4" * 64,
        cost_basis="fixture_total_cost_units",
        callback=callback,
        supports_state_restore=True,
        supports_user_simulator_replay=True,
    )


class Tau3BankingBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = PreparedTau3BankingBundle(_prepared(self.root))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bundle_exposes_inventory_and_keeps_evaluator_state_private(self) -> None:
        self.assertEqual(self.bundle.task_ids, ("task_001",))
        self.assertEqual(self.bundle.document_count, 2)
        self.assertEqual(
            self.bundle.corpus_revision,
            hashlib.sha256((self.root / "corpus.jsonl").read_bytes()).hexdigest(),
        )
        with self.assertRaises(TypeError):
            self.bundle._source_audit["database"] = {}  # type: ignore[index]

    def test_checkpoint_uses_only_structural_family_eligibility(self) -> None:
        checkpoint = _checkpoint(self.bundle)
        support = checkpoint.family_eligibility

        self.assertEqual(
            support.eligible_families,
            (SemanticFamily.REFORMULATE, SemanticFamily.DECOMPOSE),
        )
        self.assertFalse(support.supports_full_block)
        self.assertFalse(support.eligibility[SemanticFamily.ACQUIRE.value])
        self.assertFalse(support.eligibility[SemanticFamily.CLARIFY.value])
        self.assertFalse(support.eligibility[SemanticFamily.CURATE.value])
        self.assertEqual(support.provenance_status, "caller_asserted")
        self.assertEqual(support.support_status, "structural_candidate")

    def test_defect_labels_are_rejected_as_family_eligibility_inputs(self) -> None:
        checkpoint = _checkpoint(self.bundle)

        for defect_label in (
            "clarification_opportunity",
            "hidden_defect_label",
            "multi_step_request",
        ):
            with (
                self.subTest(defect_label=defect_label),
                self.assertRaisesRegex(
                    ValidationError,
                    "defect labels are not eligibility inputs",
                ),
            ):
                replace(checkpoint, structural_capabilities={defect_label: True})

    def test_full_block_requires_visible_evidence_for_every_family(self) -> None:
        checkpoint = _checkpoint(self.bundle, full=True)

        self.assertTrue(checkpoint.family_eligibility.supports_full_block)
        self.assertEqual(
            checkpoint.family_eligibility.eligible_families,
            (
                SemanticFamily.REFORMULATE,
                SemanticFamily.DECOMPOSE,
                SemanticFamily.ACQUIRE,
                SemanticFamily.CLARIFY,
                SemanticFamily.CURATE,
            ),
        )

    def test_adapter_passes_only_captured_visible_state_to_prediction(self) -> None:
        evaluator_calls: list[tuple[Any, ...]] = []

        def evaluate(gold: Any, checkpoint: Any, prediction: Any) -> Tau3EvaluationResult:
            evaluator_calls.append((gold, checkpoint, prediction))
            self.assertEqual(
                gold["source_audit"]["database"],
                {"database_secret": "evaluator-only-database"},
            )
            self.assertEqual(gold["task_spec"]["required_documents"], ("doc_001",))
            return Tau3EvaluationResult(official_reward=1.0, cost=2.5)

        adapter = Tau3BankingCheckpointAdapter(
            self.bundle,
            checkpoints=(_checkpoint(self.bundle, full=True),),
            evaluation_bridge=_bridge(evaluate),
        )
        seen: list[FrozenAgentState] = []

        def predict(state: FrozenAgentState) -> dict[str, str]:
            seen.append(state)
            serialized = canonical_json(state)
            for hidden in (
                "checkpoint-001",
                "hidden-user-simulator-script",
                "hidden-db-row",
                "evaluator-only-database",
                "required_documents",
                "evaluation_criteria",
            ):
                self.assertNotIn(hidden, serialized)
            for opaque_id in (
                "runtime_snapshot_id",
                "replay_identity",
                "retrieval_revision",
                "retrieval_config_id",
                "source_task_id",
            ):
                self.assertNotIn(opaque_id, state.metadata)
            return {"answer": "captured prediction"}

        result = BenchmarkBatchRunner().run(adapter, "base", predict)

        self.assertEqual(result.aggregate_metrics["official_reward"], 1.0)
        self.assertEqual(result.aggregate_metrics["cost"], 2.5)
        self.assertEqual(seen[0].tool_names, ("KB_search", "transfer_to_human_agents"))
        self.assertEqual(len(evaluator_calls), 1)
        self.assertTrue(adapter.supports_full_block("checkpoint-001"))
        self.assertEqual(
            adapter.evaluation_provenance,
            {
                "bridge_id": "fixture.external-evaluator",
                "bridge_revision": "4" * 64,
                "benchmark_id": "tau3.banking_knowledge",
                "cost_basis": "fixture_total_cost_units",
                "evaluation_scope": "external_bridge",
                "official_runtime_verified": False,
                "provenance_status": "caller_asserted",
                "source_revision": TAU3_OFFICIAL_SOURCE_REVISION,
                "supports_state_restore": True,
                "supports_user_simulator_replay": True,
            },
        )
        self.assertEqual(
            seen[0].metadata["family_eligibility_provenance"],
            "caller_asserted",
        )
        self.assertEqual(
            seen[0].metadata["family_support_status"],
            "structural_candidate",
        )

    def test_missing_checkpoint_and_bad_bindings_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one captured"):
            Tau3BankingCheckpointAdapter(
                self.bundle,
                checkpoints=(),
                evaluation_bridge=_bridge(
                    lambda *_: Tau3EvaluationResult(official_reward=1.0, cost=0.0)
                ),
            )

        checkpoint = _checkpoint(self.bundle)
        with self.assertRaisesRegex(ValidationError, "corpus revision"):
            Tau3BankingCheckpointAdapter(
                self.bundle,
                checkpoints=(replace(checkpoint, corpus_revision="f" * 64),),
                evaluation_bridge=_bridge(
                    lambda *_: Tau3EvaluationResult(official_reward=1.0, cost=0.0)
                ),
            )
        with self.assertRaisesRegex(ValidationError, "latest visible user message"):
            replace(checkpoint, query="Different request.")
        with self.assertRaisesRegex(ValidationError, "lowercase SHA-256"):
            replace(checkpoint, replay_identity="not-a-hash")

        duplicate = replace(checkpoint, checkpoint_id="checkpoint-duplicate")
        with self.assertRaisesRegex(ValidationError, "protected experimental units"):
            Tau3BankingCheckpointAdapter(
                self.bundle,
                checkpoints=(checkpoint, duplicate),
                evaluation_bridge=_bridge(
                    lambda *_: Tau3EvaluationResult(official_reward=1.0, cost=0.0)
                ),
            )

        wrong_source_root = self.root / "wrong-source"
        wrong_source_root.mkdir()
        with self.assertRaisesRegex(ArtifactError, "source revision"):
            PreparedTau3BankingBundle(
                replace(
                    _prepared(wrong_source_root),
                    source_revision="0" * 40,
                )
            )

    def test_evaluation_bridge_rejects_bare_callable_and_untyped_metrics(self) -> None:
        checkpoint = _checkpoint(self.bundle)
        with self.assertRaisesRegex(TypeError, "CallerAssertedTau3EvaluationBridge"):
            Tau3BankingCheckpointAdapter(
                self.bundle,
                checkpoints=(checkpoint,),
                evaluation_bridge=lambda *_: Tau3EvaluationResult(  # type: ignore[arg-type]
                    official_reward=1.0,
                    cost=0.0,
                ),
            )

        adapter = Tau3BankingCheckpointAdapter(
            self.bundle,
            checkpoints=(checkpoint,),
            evaluation_bridge=_bridge(
                lambda *_: {"official_reward": 1.0, "cost": 0.0}  # type: ignore[arg-type]
            ),
        )
        with self.assertRaisesRegex(ValidationError, "Tau3EvaluationResult"):
            adapter.score(adapter.examples("base")[0], "prediction")

    def test_bundle_hashes_are_verified_at_construction_and_runtime_use(self) -> None:
        changed_root = self.root / "changed-before-construction"
        changed_root.mkdir()
        prepared = _prepared(changed_root)
        with (changed_root / "corpus.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ArtifactError, "file hash mismatch: corpus.jsonl"):
            PreparedTau3BankingBundle(prepared)

        checkpoint = _checkpoint(self.bundle)
        adapter = Tau3BankingCheckpointAdapter(
            self.bundle,
            checkpoints=(checkpoint,),
            evaluation_bridge=_bridge(
                lambda *_: Tau3EvaluationResult(official_reward=1.0, cost=0.0)
            ),
        )
        example = adapter.examples("base")[0]
        with (self.root / "gold.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ArtifactError, "file hash mismatch: gold.jsonl"):
            adapter.score(example, "prediction")

    def test_bundle_rejects_static_input_that_attempts_to_supply_a_query(self) -> None:
        bad_root = self.root / "drift"
        bad_root.mkdir()
        original = _prepared(bad_root)
        rows = [json.loads((bad_root / "inputs.jsonl").read_text().strip())]
        rows[0]["query"] = "hidden synthesized query"
        input_hash = _write_jsonl(bad_root / "inputs.jsonl", rows)
        drifted = replace(
            original,
            file_sha256={**dict(original.file_sha256), "inputs.jsonl": input_hash},
        )

        with self.assertRaisesRegex(ArtifactError, "static input schema"):
            PreparedTau3BankingBundle(drifted)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
