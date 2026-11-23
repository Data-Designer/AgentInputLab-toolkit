from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentinputlab.algorithms.acquire import (
    AdaptiveRAGCommonSupport,
    AdaptiveRAGCommonSupportConfig,
    AdaptiveRAGRoute,
    IRCoTCommonSupportConfig,
    T5AdaptiveRAGRouter,
)
from agentinputlab.algorithms.acquire.adaptive_rag.runtime import (
    ADAPTIVE_RAG_UPSTREAM_COMMIT,
)
from agentinputlab.backends.llm import CallableBackend
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    BudgetExceededError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
    ValidationError,
)

DEMONSTRATIONS = (
    "Q: Answer the following question by reasoning step-by-step.\n"
    "What city is the capital of France?\n"
    "A: France has Paris as its capital. So the answer is: Paris."
)


class FixedRouter:
    def __init__(self, label: str) -> None:
        logits = {"A": -2.0, "B": -2.0, "C": -2.0}
        logits[label] = 2.0
        self.result = AdaptiveRAGRoute(label=label, logits=logits, input_tokens=7)
        self.questions: list[str] = []

    def route(self, question: str) -> AdaptiveRAGRoute:
        self.questions.append(question)
        return self.result

    def fingerprint(self):
        return {
            "router_kind": "test_first_step",
            "manifest_sha256": "1" * 64,
            "architecture": "test-t5",
            "architecture_substitution": "test-only",
        }


class RecordingRetriever:
    retriever_id = "recording-bm25"

    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int = 5):
        self.queries.append((query, top_k))
        index = len(self.queries)
        return (
            RetrievalResult(
                document=Document(
                    doc_id=f"d{index}",
                    title=f"Document {index}",
                    text=f"Real corpus evidence {index}",
                    corpus_revision="corpus-v1",
                ),
                score=1.0,
                rank=1,
            ),
        )

    def fingerprint(self):
        return {"retriever_id": self.retriever_id, "revision": "v1"}


def config() -> AdaptiveRAGCommonSupportConfig:
    return AdaptiveRAGCommonSupportConfig(
        ircot=IRCoTCommonSupportConfig(
            demonstrations=DEMONSTRATIONS,
            prompt_sha256=hashlib.sha256(DEMONSTRATIONS.encode()).hexdigest(),
            max_steps=2,
        )
    )


class AdaptiveRAGCommonSupportTests(unittest.TestCase):
    def test_route_a_is_explicit_no_op_and_accounts_for_router(self):
        router = FixedRouter("A")
        retriever = RecordingRetriever()
        state = FrozenAgentState(query="Who wrote Hamlet?", task_id="q-a")

        patch = AdaptiveRAGCommonSupport(router=router, config=config()).propose(
            state,
            InterventionContext(retriever=retriever, seed=17),
        )

        self.assertEqual(router.questions, [state.query])
        self.assertEqual(retriever.queries, [])
        self.assertEqual(patch.operations[0].op, PatchOp.NO_OP)
        self.assertEqual(patch.trace.counters["model_calls"], 1)
        self.assertEqual(patch.trace.counters["router_calls"], 1)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 0)
        self.assertEqual(patch.metadata["selected_strategy"], "no_retrieval")
        self.assertFalse(patch.metadata["controlled_upstream_router_parity"])

    def test_route_b_adds_only_shared_retriever_documents(self):
        retriever = RecordingRetriever()
        state = FrozenAgentState(query="Who wrote Hamlet?", task_id="q-b")

        patch = AdaptiveRAGCommonSupport(router=FixedRouter("B"), config=config()).propose(
            state, InterventionContext(retriever=retriever)
        )

        self.assertEqual(retriever.queries, [(state.query, 6)])
        self.assertEqual(patch.operations[0].op, PatchOp.ADD_CONTEXT)
        updated = patch.apply(state)
        self.assertEqual([block.block_id for block in updated.context], ["d1"])
        self.assertEqual(updated.context[0].metadata["evidence_kind"], "real_corpus_document")
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        self.assertEqual(patch.trace.counters["generator_calls"], 0)

    def test_route_c_executes_ircot_but_keeps_reasoning_private(self):
        backend = CallableBackend(
            lambda _request: {
                "text": "So the answer is: William Shakespeare.",
                "usage": {"input_tokens": 20, "output_tokens": 7},
            },
            callable_id="tests.adaptive-rag.common-support",
            revision="v1",
        )
        retriever = RecordingRetriever()
        state = FrozenAgentState(query="Who wrote Hamlet?", task_id="q-c")

        patch = AdaptiveRAGCommonSupport(router=FixedRouter("C"), config=config()).propose(
            state,
            InterventionContext(backend=backend, retriever=retriever, seed=23),
        )

        self.assertEqual([request.seed for request in backend.calls], [23])
        self.assertEqual(patch.operations[0].op, PatchOp.ADD_CONTEXT)
        self.assertEqual(patch.trace.counters["model_calls"], 2)
        self.assertEqual(patch.trace.counters["router_calls"], 1)
        self.assertEqual(patch.trace.counters["generator_calls"], 1)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        updated = patch.apply(state)
        self.assertNotIn("William Shakespeare", " ".join(x.text for x in updated.context))
        self.assertFalse(patch.metadata["generated_answer_visible_to_downstream"])

    def test_route_contract_rejects_label_that_disagrees_with_logits(self):
        with self.assertRaisesRegex(ValidationError, "A-first argmax"):
            AdaptiveRAGRoute(
                label="A",
                logits={"A": 0.0, "B": 1.0, "C": -1.0},
                input_tokens=3,
            )

    def test_ircot_route_requires_router_plus_generator_budget(self):
        retriever = RecordingRetriever()
        with self.assertRaisesRegex(BudgetExceededError, "generator-call"):
            AdaptiveRAGCommonSupport(router=FixedRouter("C"), config=config()).propose(
                FrozenAgentState(
                    query="question",
                    budget=ExecutionBudget(max_model_calls=1),
                ),
                InterventionContext(retriever=retriever),
            )
        self.assertEqual(retriever.queries, [])

    def test_t5_router_manifest_closure_is_verified_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "config.json": b"{}",
                "tokenizer_config.json": b"{}",
                "model.safetensors": b"weights",
            }
            digests = {}
            for name, content in files.items():
                (root / name).write_bytes(content)
                digests[name] = hashlib.sha256(content).hexdigest()
            manifest = {
                "schema_version": "agentinputlab.adaptive-rag-router-artifact.v1",
                "upstream_commit": ADAPTIVE_RAG_UPSTREAM_COMMIT,
                "route_rule": "first_decoder_step_logit_argmax_A_first",
                "strategy_map": {"A": "no_retrieval", "B": "single_retrieval", "C": "ircot"},
                "upstream_router_checkpoint_parity_claimed": False,
                "paper_result_parity_claimed": False,
                "artifact_files": digests,
            }
            manifest_path = root / "artifact-manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            router = T5AdaptiveRAGRouter(root, manifest_sha, device="cpu")
            self.assertEqual(router.fingerprint()["manifest_sha256"], manifest_sha)
            linked_root = root / "linked-root"
            linked_root.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValidationError, "real directory"):
                T5AdaptiveRAGRouter(linked_root, manifest_sha, device="cpu")
            linked_root.unlink()
            (root / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValidationError, "unlisted or missing"):
                T5AdaptiveRAGRouter(root, manifest_sha, device="cpu")
            (root / "adapter_config.json").unlink()
            (root / "unlisted-link").symlink_to(root / "config.json")
            with self.assertRaisesRegex(ValidationError, "unlisted symlink"):
                T5AdaptiveRAGRouter(root, manifest_sha, device="cpu")
            (root / "unlisted-link").unlink()
            (root / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValidationError, "closure mismatch"):
                T5AdaptiveRAGRouter(root, manifest_sha, device="cpu")


if __name__ == "__main__":
    unittest.main()
