from __future__ import annotations

import unittest
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentinputlab.algorithms.acquire.qrag import (
    QRAGCommonSupport,
    QRAGCommonSupportConfig,
)
from agentinputlab.algorithms.acquire.qrag.common_support import (
    LocalQRAGFrozenPolicyRuntime,
    QRAGPolicyRequest,
    _unused_vllm_import_compatibility,
)
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.runner import InterventionRunner


class FixedRetriever:
    retriever_id = "qrag-shared-bm25"

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        assert query
        return tuple(
            RetrievalResult(
                document=Document(
                    doc_id=f"d{index}",
                    title=f"Document {index}",
                    text=f"Real shared-corpus evidence {index}",
                    corpus_revision="corpus-v1",
                ),
                score=1.0 / index,
                rank=index,
            )
            for index in range(1, top_k + 1)
        )

    def fingerprint(self) -> Mapping[str, object]:
        return {"retriever_id": self.retriever_id, "revision": "v1"}


class FixedRuntime:
    def __init__(self, *, corrupt: bool = False) -> None:
        self.corrupt = corrupt
        self.requests = []

    def select(self, request):
        self.requests.append(request)
        ids = tuple(row.document.doc_id for row in request.candidates)
        first_q = (0.1, 0.2, 0.9, 0.3, 0.4, 0.5)
        second_available = (0, 1, 3, 4, 5)
        second_q = (0.8, 0.2, 0.3, 0.4, 0.5)
        steps = (
            {
                "index": 0,
                "available_indices": tuple(range(6)),
                "q_values": first_q,
                "action_index": 2,
                "selected_q_value": 0.9,
                "selected_indices": (2,),
                "selected_document_ids": (ids[2],),
            },
            {
                "index": 1,
                "available_indices": second_available,
                "q_values": second_q,
                "action_index": 0,
                "selected_q_value": 0.8,
                "selected_indices": (2, 0),
                "selected_document_ids": (ids[2], ids[0]),
            },
        )
        selected = (2, 1) if self.corrupt else (2, 0)
        return {
            "status": "ok",
            "trajectory": steps,
            "trajectory_sha256": content_hash(steps),
            "selected_indices": selected,
            "selected_document_ids": tuple(ids[index] for index in selected),
            "candidate_order_sha256": content_hash(ids),
            "candidate_text_sha256": "a" * 64,
            "usage": {
                "model_calls": 0,
                "retrieval_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_calls": 6,
                "accelerator_seconds": 0.25,
            },
            "policy_steps": 2,
            "attestation": {
                "official_candidate_builder_called": True,
                "official_pqn_select_action_called": True,
                "official_qa_env_step_called": True,
                "safe_weights_only_load": True,
                "strict_state_dict_load": True,
                "checkpoint_eval_mode": True,
                "dummy_feedback_used": True,
                "gold_answer_used": False,
                "supporting_fact_labels_used": False,
                "reward_used": False,
                "training_called": False,
                "stop_action_used": False,
                "seed_applied": True,
            },
        }

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "runtime": "qrag-released-policy-fixture",
            "checkpoint_file_sha256": "b" * 64,
            "paper_result_parity_claimed": False,
        }


class QRAGCommonSupportTests(unittest.TestCase):
    def test_policy_cpu_execution_does_not_report_accelerator_time(self) -> None:
        # Exercise the actual select loop with shape-only policy/environment
        # fixtures. No pretrained weights, training or benchmark inference.
        for device, accelerator_seconds in (("cpu", 0.0), ("cpu:0", 0.0), ("cuda:0", 2.5), ("mps", 2.5)):
            with self.subTest(device=device):
                runtime = object.__new__(LocalQRAGFrozenPolicyRuntime)
                runtime.device = device
                state = SimpleNamespace(available_ids={0, 1}, item_ids=[])
                q_values = Mock()
                q_values.detach.return_value.to.return_value.reshape.return_value = q_values
                q_values.numel.return_value = 2
                q_values.__getitem__ = Mock(return_value=SimpleNamespace(item=lambda: 0.5))
                agent = SimpleNamespace(action_tokenizer=None, critic=SimpleNamespace(action_embed=None),
                                        action_embed_target=None,
                                        select_action=lambda *a, state=state, q_values=q_values, **k:
                                            (min(state.available_ids), q_values, None))
                def step(action, state=state):
                    state.available_ids.remove(action)
                    state.item_ids.append(action)
                    return state, None, None, not state.available_ids
                env = SimpleNamespace(reset=lambda sample, state=state: state,
                    get_extra_embeds=lambda *a: ({"rope": None}, {"rope": None}),
                    update_embeds=lambda embeds, embedder: embeds, step=step)
                torch = SimpleNamespace(manual_seed=Mock(), float64="float64",
                                        cuda=SimpleNamespace(manual_seed_all=Mock(), synchronize=Mock()))
                with (
                    patch.object(LocalQRAGFrozenPolicyRuntime, "_load", return_value=(torch, agent, env)),
                    patch.dict("sys.modules", {"numpy": SimpleNamespace(random=SimpleNamespace(seed=Mock()))}),
                    patch("agentinputlab.algorithms.acquire.qrag.common_support.time.perf_counter",
                          side_effect=[10.0, 12.5]),
                ):
                    result = runtime.select(QRAGPolicyRequest(question="fixture", seed=17, max_steps=2,
                        candidates=FixedRetriever().retrieve("fixture", top_k=2)))
                self.assertEqual(result["usage"]["accelerator_seconds"], accelerator_seconds)
                self.assertEqual(result["selected_indices"], (0, 1))
                self.assertEqual(result["usage"]["embedding_calls"], 6)
                self.assertEqual(torch.cuda.synchronize.call_count, 2 if device.startswith("cuda") else 0)

    def test_missing_vllm_gets_fail_closed_import_only_compatibility(self) -> None:
        with (
            patch("importlib.util.find_spec", return_value=None),
            _unused_vllm_import_compatibility(),
        ):
            from vllm import LLM, SamplingParams

            self.assertIs(LLM, SamplingParams)
            with self.assertRaisesRegex(RuntimeError, "must not instantiate"):
                LLM()

    def test_released_policy_selects_real_shared_corpus_documents(self) -> None:
        runtime = FixedRuntime()
        state = FrozenAgentState(
            query="Who wrote Hamlet?",
            task_id="qrag-common",
            budget=ExecutionBudget(max_retrieval_calls=1),
        )
        patch = QRAGCommonSupport(
            config=QRAGCommonSupportConfig(candidate_top_k=6),
        ).propose(
            state,
            InterventionContext(
                retriever=FixedRetriever(),
                external_runtime=runtime,
                seed=17,
            ),
        )

        self.assertIs(patch.operations[0].op, PatchOp.ADD_CONTEXT)
        updated = patch.apply(state)
        self.assertEqual([row.block_id for row in updated.context], ["d3", "d1"])
        self.assertEqual(
            [row.metadata["evidence_kind"] for row in updated.context],
            ["real_corpus_document", "real_corpus_document"],
        )
        self.assertEqual(runtime.requests[0].seed, 17)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        self.assertEqual(patch.trace.counters["embedding_calls"], 6)
        self.assertEqual(patch.trace.counters["policy_steps"], 2)
        self.assertFalse(patch.metadata["gold_answer_used"])
        self.assertFalse(patch.metadata["paper_result_parity_claimed"])

    def test_runner_observes_policy_embedding_usage(self) -> None:
        runtime = FixedRuntime()
        record = InterventionRunner().run(
            QRAGCommonSupport(config=QRAGCommonSupportConfig(candidate_top_k=6)),
            FrozenAgentState(
                query="Who wrote Hamlet?",
                task_id="qrag-observed",
                budget=ExecutionBudget(max_retrieval_calls=1),
            ),
            InterventionContext(
                retriever=FixedRetriever(),
                external_runtime=runtime,
                seed=17,
            ),
        )

        self.assertEqual(record.usage.retrieval_calls, 1)
        self.assertEqual(record.usage.embedding_calls, 6)
        self.assertEqual(record.patch.trace.counters["policy_steps"], 2)

    def test_inconsistent_final_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(AlgorithmOutputError, "trajectory"):
            QRAGCommonSupport().propose(
                FrozenAgentState(query="question"),
                InterventionContext(
                    retriever=FixedRetriever(),
                    external_runtime=FixedRuntime(corrupt=True),
                    seed=17,
                ),
            )


if __name__ == "__main__":
    unittest.main()
