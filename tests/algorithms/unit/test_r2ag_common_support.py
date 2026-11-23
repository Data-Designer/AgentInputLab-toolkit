from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from agentinputlab.algorithms.acquire.r2ag import (
    LocalR2AGCommonSupportRuntime,
    R2AGCommonSupport,
    R2AGCommonSupportConfig,
    R2AGReaderResult,
)
from agentinputlab.algorithms.acquire.r2ag.bridge import compute_r2ag_retrieval_features
from agentinputlab.algorithms.acquire.r2ag.common_support import (
    _qwen38_fp8_min_new_tokens,
    _repair_qwen38_fp8_model_config,
    r2ag_reader_backbone_identity,
)
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    BudgetExceededError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
)


class FixedRetriever:
    retriever_id = "fixed-r2ag-bm25"

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        assert query
        return tuple(
            RetrievalResult(
                document=Document(
                    doc_id=f"d{index}",
                    title=f"Document {index}",
                    text=f"Real corpus evidence {index}",
                    corpus_revision="corpus-v1",
                ),
                score=1.0 / index,
                rank=index,
            )
            for index in range(1, top_k + 1)
        )

    def fingerprint(self) -> Mapping[str, object]:
        return {"retriever_id": self.retriever_id, "revision": "v1"}


class FixedEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def encode(
        self,
        *,
        query: str,
        documents: Sequence[RetrievalResult],
    ) -> tuple[Sequence[float], tuple[Sequence[float], ...]]:
        self.calls.append((query, tuple(row.document.doc_id for row in documents)))
        return (1.0, 0.0), tuple(
            (1.0 - index / 20.0, index / 20.0) for index in range(len(documents))
        )

    def fingerprint(self) -> Mapping[str, object]:
        return {"model_load_set_sha256": "e" * 64, "prefixes": "role-correct"}


class FixedReader:
    def __init__(self) -> None:
        self.features: tuple[Mapping[str, float], ...] = ()

    def run(
        self,
        *,
        query: str,
        documents: Sequence[RetrievalResult],
        features: Sequence[Mapping[str, float]],
        max_new_tokens: int,
        seed: int,
    ) -> R2AGReaderResult:
        assert query and len(documents) == 6 and max_new_tokens == 32 and seed == 17
        self.features = tuple(features)
        return R2AGReaderResult(
            anchor="A concise generated hypothesis.",
            input_tokens=41,
            output_tokens=7,
            marker_positions=(11, 21, 31, 41, 51, 61),
            rformer_output_sha256="1" * 64,
            projection_output_sha256="2" * 64,
        )

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "manifest_sha256": "a" * 64,
            "upstream_checkpoint_parity_claimed": False,
            "paper_result_parity_claimed": False,
        }


class R2AGCommonSupportTests(unittest.TestCase):
    def test_implementation_label_does_not_misidentify_other_reader_backbones(self) -> None:
        class Reader(FixedReader):
            def __init__(self, backbone: str) -> None:
                super().__init__()
                self.backbone = backbone

            def fingerprint(self) -> Mapping[str, object]:
                return {**super().fingerprint(), "reader_backbone": self.backbone}

        state = FrozenAgentState(query="Who wrote Hamlet?", task_id="reader-identity")
        operations = []
        for backbone in ("Qwen3-4B", "gemma-12b-unified", "gpt-oss-20b", "Qwen3.8-27B-FP8"):
            with self.subTest(backbone=backbone):
                patch = R2AGCommonSupport(
                    config=R2AGCommonSupportConfig(retrieval_top_k=6, max_new_tokens=32),
                ).propose(
                    state,
                    InterventionContext(
                        retriever=FixedRetriever(),
                        external_runtime=LocalR2AGCommonSupportRuntime(
                            reader=Reader(backbone), feature_encoder=FixedEncoder(),
                        ),
                        seed=17,
                    ),
                )
                self.assertTrue(patch.implementation_identity.startswith(
                    "controlled_marker_conditioned_reader:yefd/RRAG@"
                ))
                self.assertNotIn("qwen3_4b_retraining", patch.implementation_identity)
                self.assertEqual(patch.metadata["reader_backbone"], backbone)
                configuration = patch.trace.events[0].payload
                self.assertEqual(configuration["reader"]["reader_backbone"], backbone)
                operations.append(patch.operations)
        # Correcting the label cannot change retrieved evidence or the generated anchor.
        self.assertTrue(all(value == operations[0] for value in operations))

    def test_missing_reader_backbone_is_not_invented(self) -> None:
        patch = R2AGCommonSupport(
            config=R2AGCommonSupportConfig(retrieval_top_k=6, max_new_tokens=32),
        ).propose(
            FrozenAgentState(query="Who wrote Hamlet?", task_id="unknown-reader"),
            InterventionContext(
                retriever=FixedRetriever(),
                external_runtime=LocalR2AGCommonSupportRuntime(
                    reader=FixedReader(), feature_encoder=FixedEncoder(),
                ),
                seed=17,
            ),
        )
        self.assertIsNone(patch.metadata["reader_backbone"])

    def test_reader_backbone_identity_tracks_selected_local_snapshot(self) -> None:
        assert r2ag_reader_backbone_identity(Path("/models/Qwen3-4B")) == "Qwen3-4B"
        assert r2ag_reader_backbone_identity(Path("/models/gpt-oss-20b")) == "gpt-oss-20b"

    def test_qwen38_fp8_reader_removes_only_stale_mlp_gate_exclusions(self) -> None:
        config = SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "modules_to_not_convert": [
                    "model.language_model.layers.0.mlp.gate",
                    "model.language_model.layers.0.input_layernorm",
                    "model.language_model.layers.1.mlp.gate",
                ],
            }
        )

        repaired = _repair_qwen38_fp8_model_config(
            config,
            backbone="Qwen3.8-27B-FP8",
        )

        assert repaired == 2
        assert config.quantization_config["modules_to_not_convert"] == [
            "model.language_model.layers.0.input_layernorm"
        ]

    def test_fp8_reader_config_repair_is_scoped_to_qwen38(self) -> None:
        config = SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "modules_to_not_convert": ["model.language_model.layers.0.mlp.gate"],
            }
        )

        repaired = _repair_qwen38_fp8_model_config(config, backbone="Qwen3-4B")

        assert repaired == 0
        assert config.quantization_config["modules_to_not_convert"] == [
            "model.language_model.layers.0.mlp.gate"
        ]

    def test_qwen38_fp8_reader_requires_one_non_eos_anchor_token(self) -> None:
        assert (
            _qwen38_fp8_min_new_tokens(
                backbone="Qwen3.8-27B-FP8",
                max_new_tokens=96,
            )
            == 1
        )
        assert (
            _qwen38_fp8_min_new_tokens(
                backbone="Qwen3-4B",
                max_new_tokens=96,
            )
            is None
        )

    def test_common_support_adds_real_documents_and_one_non_authoritative_anchor(self) -> None:
        encoder = FixedEncoder()
        reader = FixedReader()
        state = FrozenAgentState(query="Who wrote Hamlet?", task_id="r2ag-common")

        patch = R2AGCommonSupport(
            config=R2AGCommonSupportConfig(retrieval_top_k=6, max_new_tokens=32),
        ).propose(
            state,
            InterventionContext(
                retriever=FixedRetriever(),
                external_runtime=LocalR2AGCommonSupportRuntime(
                    reader=reader,
                    feature_encoder=encoder,
                ),
                seed=17,
            ),
        )

        assert patch.operations[0].op is PatchOp.ADD_CONTEXT
        updated = patch.apply(state)
        assert len(updated.context) == 7
        assert [row.metadata["evidence_kind"] for row in updated.context[:6]] == [
            "real_corpus_document"
        ] * 6
        anchor = updated.context[-1]
        assert anchor.metadata["evidence_kind"] == "model_generated_non_authoritative_anchor"
        assert anchor.metadata["generated_anchor_is_evidence"] is False
        assert encoder.calls == [(state.query, ("d1", "d2", "d3", "d4", "d5", "d6"))]
        assert len(reader.features) == 6
        assert patch.trace.counters["model_calls"] == 1
        assert patch.trace.counters["retrieval_calls"] == 1
        assert patch.trace.counters["embedding_calls"] == 1
        assert patch.metadata["upstream_checkpoint_parity_claimed"] is False
        assert patch.metadata["paper_result_parity_claimed"] is False

    def test_context_budget_fails_after_reader_usage_is_known(self) -> None:
        with self.assertRaisesRegex(BudgetExceededError, "max_context_tokens"):
            R2AGCommonSupport(
                config=R2AGCommonSupportConfig(retrieval_top_k=6, max_new_tokens=32),
            ).propose(
                FrozenAgentState(
                    query="question",
                    budget=ExecutionBudget(max_context_tokens=40),
                ),
                InterventionContext(
                    retriever=FixedRetriever(),
                    external_runtime=LocalR2AGCommonSupportRuntime(
                        reader=FixedReader(),
                        feature_encoder=FixedEncoder(),
                    ),
                    seed=17,
                ),
            )

    def test_public_feature_wrapper_rejects_zero_and_ragged_vectors(self) -> None:
        with self.assertRaisesRegex(Exception, "invalid"):
            compute_r2ag_retrieval_features((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaisesRegex(Exception, "invalid"):
            compute_r2ag_retrieval_features((1.0, 0.0), ((1.0,), (0.0, 1.0)))
