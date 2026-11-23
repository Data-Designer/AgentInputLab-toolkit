from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentinputlab.backends.embedding import CallableEmbeddingBackend
from agentinputlab.backends.retrieval import (
    DENSE_RETRIEVER_PROFILE_IDS,
    RETRIEVER_PROFILES,
    Document,
    EmbeddingBackendEncoder,
    InMemoryBM25Retriever,
    SentenceTransformerDenseRetriever,
    build_retriever,
    retriever_profile,
)
from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher
from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash


def _basis(dimensions: int, index: int = 0, scale: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * dimensions
    values[index] = scale
    return tuple(values)


class RecordingEncoder:
    def __init__(self, dimensions: int, vectors=None):
        self.dimensions = dimensions
        self.vectors = vectors or (lambda _text: _basis(dimensions))
        self.calls: list[dict[str, object]] = []

    def fingerprint(self):
        return {
            "implementation": "tests.recording_encoder",
            "revision": "fixture-v1",
            "dimensions": self.dimensions,
        }

    def encode(
        self,
        sentences,
        *,
        batch_size,
        normalize_embeddings,
        show_progress_bar,
        convert_to_numpy,
    ):
        batch = tuple(sentences)
        self.calls.append(
            {
                "sentences": batch,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return tuple(self.vectors(text) for text in batch)


class CardinalityChangingEncoder(RecordingEncoder):
    def encode(self, sentences, **kwargs):
        super().encode(sentences, **kwargs)
        return ()


class FingerprintMutatingEncoder(RecordingEncoder):
    def encode(self, sentences, **kwargs):
        result = super().encode(sentences, **kwargs)
        self.dimensions += 1
        return result


class DenseRetrieverConformanceTests(unittest.TestCase):
    def test_builtin_catalog_is_exactly_one_sparse_and_seven_pinned_dense_profiles(self) -> None:
        self.assertEqual(len(RETRIEVER_PROFILES), 8)
        self.assertEqual(
            set(DENSE_RETRIEVER_PROFILE_IDS),
            {
                "dense.bge.base_en_v1_5",
                "dense.bge.m3",
                "dense.e5.base_v2",
                "dense.e5.large_v2",
                "dense.e5.multilingual_large",
                "dense.qwen3.embedding_0_6b",
                "dense.contriever.msmarco",
            },
        )
        self.assertEqual(RETRIEVER_PROFILES["sparse.bm25.memory.v1"].metric, "bm25")
        for profile_id in DENSE_RETRIEVER_PROFILE_IDS:
            with self.subTest(profile_id=profile_id):
                profile = RETRIEVER_PROFILES[profile_id]
                self.assertEqual(len(profile.model_revision), 40)
                self.assertEqual(profile.metric, "cosine")
                self.assertFalse(profile.weights_bundled)
                self.assertFalse(profile.weight_files_verified)

    def test_profile_prompt_contracts_are_explicit(self) -> None:
        bge = retriever_profile("dense.bge.base_en_v1_5")
        e5 = retriever_profile("dense.e5.base_v2")
        qwen = retriever_profile("dense.qwen3.embedding_0_6b")
        contriever = retriever_profile("dense.contriever.msmarco")
        self.assertEqual(
            bge.query_prefix,
            "Represent this sentence for searching relevant passages: ",
        )
        self.assertEqual((e5.query_prefix, e5.document_prefix), ("query: ", "passage: "))
        self.assertTrue(qwen.query_prefix.endswith("\nQuery:"))
        self.assertEqual(qwen.tokenizer_padding_side, "left")
        self.assertEqual((contriever.query_prefix, contriever.document_prefix), ("", ""))
        self.assertEqual(contriever.license_spdx, "CC-BY-NC-4.0")

    def test_unknown_and_sparse_profiles_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown retriever profile"):
            retriever_profile("dense.missing")
        with self.assertRaisesRegex(ValidationError, "requires a dense profile"):
            SentenceTransformerDenseRetriever(
                [],
                profile="sparse.bm25.memory.v1",
                encoder=RecordingEncoder(1),
            )

    def test_index_batches_inputs_applies_e5_prefixes_and_ranks_cosine(self) -> None:
        profile = retriever_profile("dense.e5.base_v2")
        doc_inputs = {
            "passage: Beta\nshared": _basis(768, 0, 2.0),
            "passage: shared": _basis(768, 0),
            "passage: unrelated": _basis(768, 1),
            "query: target": _basis(768, 0, 3.0),
        }
        encoder = RecordingEncoder(768, doc_inputs.__getitem__)
        retriever = SentenceTransformerDenseRetriever(
            [
                Document("b", "shared", title="Beta", corpus_revision="fixture-v1"),
                Document("a", "shared", corpus_revision="fixture-v1"),
                Document("c", "unrelated", corpus_revision="fixture-v1"),
            ],
            profile=profile,
            batch_size=2,
            encoder=encoder,
        )

        summary = retriever.build_index()
        first_call_count = len(encoder.calls)
        self.assertEqual((summary.document_count, summary.dimensions), (3, 768))
        self.assertTrue(retriever.indexed)
        self.assertEqual(retriever.build_index(), summary)
        self.assertEqual(len(encoder.calls), first_call_count)

        results = retriever.retrieve("target", top_k=3)
        self.assertEqual(tuple(item.document.doc_id for item in results), ("a", "b", "c"))
        self.assertEqual(tuple(item.rank for item in results), (1, 2, 3))
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, 1.0)
        self.assertAlmostEqual(results[2].score, 0.0)
        self.assertEqual(
            [call["sentences"] for call in encoder.calls],
            [
                ("passage: Beta\nshared", "passage: shared"),
                ("passage: unrelated",),
                ("query: target",),
            ],
        )
        for call in encoder.calls:
            self.assertFalse(call["normalize_embeddings"])
            self.assertFalse(call["show_progress_bar"])
            self.assertTrue(call["convert_to_numpy"])

    def test_each_dense_profile_applies_its_document_and_query_contract(self) -> None:
        for profile_id in DENSE_RETRIEVER_PROFILE_IDS:
            with self.subTest(profile_id=profile_id):
                profile = retriever_profile(profile_id)
                encoder = RecordingEncoder(profile.expected_dimensions or 0)
                retriever = SentenceTransformerDenseRetriever(
                    [Document("d", "body", title="title")],
                    profile=profile,
                    encoder=encoder,
                )
                retriever.retrieve("question", top_k=1)
                self.assertEqual(
                    encoder.calls[0]["sentences"],
                    (f"{profile.document_prefix}title\nbody",),
                )
                self.assertEqual(
                    encoder.calls[1]["sentences"],
                    (f"{profile.query_prefix}question",),
                )

    def test_hyde_embedder_and_vector_searcher_reuse_the_fixed_index(self) -> None:
        vectors = {
            "passage: alpha": _basis(768, 0),
            "passage: beta": _basis(768, 1),
            "query: question": _basis(768, 0, 3.0),
            "query: hypothetical": _basis(768, 1, 4.0),
        }
        encoder = RecordingEncoder(768, vectors.__getitem__)
        retriever = SentenceTransformerDenseRetriever(
            [
                Document("a", "alpha", corpus_revision="fixture-v1"),
                Document("b", "beta", corpus_revision="fixture-v1"),
            ],
            profile="dense.e5.base_v2",
            encoder=encoder,
        )

        self.assertIsInstance(retriever, DenseEmbedder)
        self.assertIsInstance(retriever, VectorSearcher)
        encoded = retriever.embed(("question", "hypothetical"))
        results = retriever.search(
            tuple(left + right for left, right in zip(*encoded, strict=True)),
            top_k=2,
        )

        self.assertEqual(retriever.embedder_id, f"{retriever.retriever_id}:embedder")
        self.assertEqual(retriever.searcher_id, f"{retriever.retriever_id}:index")
        self.assertEqual(tuple(item.document.doc_id for item in results), ("a", "b"))
        self.assertEqual(
            [call["sentences"] for call in encoder.calls],
            [("query: question", "query: hypothetical"), ("passage: alpha", "passage: beta")],
        )

    def test_construction_and_fingerprint_do_not_encode_or_mutate(self) -> None:
        documents = [Document("d", "body")]
        encoder = RecordingEncoder(768)
        retriever = SentenceTransformerDenseRetriever(
            documents,
            profile="dense.contriever.msmarco",
            encoder=encoder,
        )
        first = retriever.fingerprint()
        documents.append(Document("late", "not part of corpus"))
        second = retriever.fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(first["document_count"], 1)
        self.assertEqual(first["runtime_kind"], "injected_encoder_unverified")
        self.assertEqual(
            first["encoder_identity_binding"],
            "derived_fingerprint_monitored",
        )
        self.assertFalse(first["weights_bundled"])
        self.assertFalse(first["weight_files_verified"])
        self.assertEqual(encoder.calls, [])

    def test_configuration_is_read_only_and_cannot_stale_the_cached_index(self) -> None:
        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            batch_size=4,
            encoder=RecordingEncoder(768),
        )
        retriever.build_index()
        replacements = {
            "documents": (Document("b", "two"),),
            "profile": retriever_profile("dense.e5.base_v2"),
            "batch_size": 8,
            "device": "cpu",
            "local_files_only": False,
            "retriever_id": "changed",
        }
        for name, value in replacements.items():
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(retriever, name, value)
        self.assertEqual(retriever.retrieve("q", top_k=1)[0].document.doc_id, "a")

    def test_derived_encoder_identity_is_checked_before_and_after_encoding(self) -> None:
        encoder = RecordingEncoder(768)
        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            encoder=encoder,
        )
        encoder.dimensions = 767
        with self.assertRaisesRegex(ValidationError, "fingerprint changed"):
            retriever.fingerprint()

        mutating = FingerprintMutatingEncoder(768)
        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            encoder=mutating,
        )
        with self.assertRaisesRegex(ValidationError, "fingerprint changed"):
            retriever.build_index()

    def test_explicit_encoder_identity_remains_marked_unverified(self) -> None:
        encoder = RecordingEncoder(768)
        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            encoder=encoder,
            encoder_identity={"implementation": "external", "revision": "v1"},
        )
        encoder.dimensions = 767
        self.assertEqual(
            retriever.fingerprint()["encoder_identity_binding"],
            "explicit_identity_unverified",
        )

    def test_fingerprint_changes_for_fairness_relevant_settings(self) -> None:
        encoder_identity = {
            "implementation": "tests.encoder",
            "revision": "fixture-v1",
        }
        first = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            batch_size=4,
            encoder=RecordingEncoder(768),
            encoder_identity=encoder_identity,
        )
        corpus_change = SentenceTransformerDenseRetriever(
            [Document("a", "two")],
            profile="dense.bge.base_en_v1_5",
            batch_size=4,
            encoder=RecordingEncoder(768),
            encoder_identity=encoder_identity,
        )
        batch_change = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            batch_size=8,
            encoder=RecordingEncoder(768),
            encoder_identity=encoder_identity,
        )
        profile_change = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.e5.base_v2",
            batch_size=4,
            encoder=RecordingEncoder(768),
            encoder_identity=encoder_identity,
        )
        fingerprints = {
            str(candidate.fingerprint())
            for candidate in (first, corpus_change, batch_change, profile_change)
        }
        self.assertEqual(len(fingerprints), 4)

    def test_embedding_backend_adapter_supports_transport_neutral_dense_retrieval(self) -> None:
        requests = []

        def client(request):
            requests.append(request)
            return tuple(
                _basis(768, 1 if text.endswith("different") else 0) for text in request.texts
            )

        backend = CallableEmbeddingBackend(
            client=client,
            model_id="BAAI/bge-base-en-v1.5",
            revision="served-revision-v1",
            dimensions=768,
            identity={"deployment": "fixture-api", "protocol": "openai-compatible"},
            network=True,
        )
        adapter = EmbeddingBackendEncoder(
            backend=backend,
            dimensions=768,
            request_metadata={"tenant": "fixture"},
        )
        adapter_fingerprint = adapter.fingerprint()
        self.assertEqual(adapter_fingerprint["backend_identity"], backend.fingerprint())

        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "relevant"), Document("b", "different")],
            profile="dense.bge.base_en_v1_5",
            batch_size=2,
            encoder=adapter,
        )
        self.assertEqual(
            retriever.fingerprint()["encoder_identity_hash"],
            content_hash(adapter_fingerprint),
        )
        results = retriever.retrieve("relevant", top_k=2)
        self.assertEqual(tuple(item.document.doc_id for item in results), ("a", "b"))
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].dimensions, 768)
        self.assertEqual(dict(requests[0].metadata), {"tenant": "fixture"})
        self.assertTrue(requests[1].texts[0].startswith("Represent this sentence"))

        backend.revision = "mutated-revision"
        with self.assertRaisesRegex(ValidationError, "fingerprint changed"):
            adapter.encode(
                ("q",),
                batch_size=1,
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

    def test_factory_constructs_sparse_or_dense_profiles_and_rejects_cross_family_options(
        self,
    ) -> None:
        documents = [Document("a", "relevant"), Document("b", "other")]
        sparse = build_retriever("sparse.bm25.memory.v1", documents, k1=1.5)
        self.assertIsInstance(sparse, InMemoryBM25Retriever)
        self.assertEqual(sparse.fingerprint()["retriever_id"], "bm25:sparse.bm25.memory.v1")
        self.assertEqual(sparse.retrieve("relevant", top_k=1)[0].document.doc_id, "a")
        with self.assertRaisesRegex(ValidationError, "does not accept option.*batch_size"):
            build_retriever("sparse.bm25.memory.v1", documents, batch_size=2)
        with self.assertRaisesRegex(ValidationError, "does not accept option.*k1"):
            build_retriever("dense.bge.base_en_v1_5", documents, k1=1.5)

        backend = CallableEmbeddingBackend(
            client=lambda request: tuple(_basis(768) for _ in request.texts),
            model_id="fixture",
            revision="fixture-v1",
            dimensions=768,
            identity={"deployment": "local-test"},
        )
        dense = build_retriever(
            "dense.bge.base_en_v1_5",
            documents,
            embedding_backend=backend,
            embedding_request_metadata={"route": "offline"},
        )
        self.assertIsInstance(dense, SentenceTransformerDenseRetriever)
        self.assertEqual(len(dense.retrieve("q", top_k=1)), 1)
        with self.assertRaisesRegex(ValidationError, "not both"):
            build_retriever(
                "dense.bge.base_en_v1_5",
                documents,
                embedding_backend=backend,
                encoder=RecordingEncoder(768),
            )
        with self.assertRaisesRegex(ValidationError, "requires embedding_backend"):
            build_retriever(
                "dense.bge.base_en_v1_5",
                documents,
                embedding_request_metadata={"route": "offline"},
            )

    def test_empty_corpus_and_nonpositive_top_k_do_not_load_encoder(self) -> None:
        encoder = RecordingEncoder(768)
        empty = SentenceTransformerDenseRetriever(
            [],
            profile="dense.bge.base_en_v1_5",
            encoder=encoder,
        )
        self.assertEqual(empty.retrieve("q"), ())
        self.assertEqual(empty.build_index().document_count, 0)
        populated = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
            encoder=encoder,
        )
        self.assertEqual(populated.retrieve("q", top_k=0), ())
        self.assertEqual(encoder.calls, [])

    def test_invalid_configuration_and_encoder_identity_fail_closed(self) -> None:
        profile = "dense.bge.base_en_v1_5"
        invalid = (
            {"batch_size": 0},
            {"batch_size": True},
            {"device": ""},
            {"local_files_only": "yes"},
            {"model_path": "relative/model"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                SentenceTransformerDenseRetriever([], profile=profile, **values)
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            SentenceTransformerDenseRetriever(
                [Document("same", "one"), Document("same", "two")],
                profile=profile,
            )
        encoder = RecordingEncoder(768)
        encoder.fingerprint = None
        with self.assertRaisesRegex(ValidationError, "non-empty identity"):
            SentenceTransformerDenseRetriever([], profile=profile, encoder=encoder)

    def test_wrong_cardinality_dimension_and_numeric_vectors_fail_closed(self) -> None:
        cases = (
            (CardinalityChangingEncoder(768), "cardinality"),
            (RecordingEncoder(767), "dimension"),
            (RecordingEncoder(768, lambda _text: (0.0,) * 768), "zero-norm"),
            (
                RecordingEncoder(
                    768,
                    lambda _text: (math.nan,) + (0.0,) * 767,
                ),
                "non-finite",
            ),
        )
        for encoder, message in cases:
            with self.subTest(message=message):
                retriever = SentenceTransformerDenseRetriever(
                    [Document("a", "one")],
                    profile="dense.bge.base_en_v1_5",
                    encoder=encoder,
                )
                with self.assertRaisesRegex(ValidationError, message):
                    retriever.build_index()

    def test_lazy_loader_uses_pinned_safe_arguments(self) -> None:
        encoder = RecordingEncoder(1024)
        constructor_calls = []

        def constructor(model_id, **kwargs):
            constructor_calls.append((model_id, kwargs))
            return encoder

        fake_module = SimpleNamespace(SentenceTransformer=constructor)
        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.qwen3.embedding_0_6b",
            device="cpu",
        )
        with patch(
            "agentinputlab.backends.retrieval.dense.importlib.import_module",
            return_value=fake_module,
        ) as importer:
            retriever.retrieve("q", top_k=1)
        importer.assert_called_once_with("sentence_transformers")
        model_id, kwargs = constructor_calls[0]
        profile = retriever_profile("dense.qwen3.embedding_0_6b")
        self.assertEqual(model_id, profile.model_id)
        self.assertEqual(kwargs["revision"], profile.model_revision)
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["tokenizer_kwargs"], {"padding_side": "left"})

    def test_lazy_loader_can_use_an_explicit_absolute_local_snapshot(self) -> None:
        encoder = RecordingEncoder(768)
        calls = []

        def constructor(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return encoder

        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.e5.base_v2",
            model_path=Path("/models/e5-base-v2"),
        )
        with patch(
            "agentinputlab.backends.retrieval.dense.importlib.import_module",
            return_value=SimpleNamespace(SentenceTransformer=constructor),
        ):
            retriever.retrieve("q", top_k=1)

        self.assertEqual(calls[0][0], "/models/e5-base-v2")
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(retriever.fingerprint()["model_source"], "local_snapshot")

    def test_missing_optional_dependency_is_typed_and_never_silently_downloaded(self) -> None:
        retriever = SentenceTransformerDenseRetriever(
            [Document("a", "one")],
            profile="dense.bge.base_en_v1_5",
        )
        with (
            patch(
                "agentinputlab.backends.retrieval.dense.importlib.import_module",
                side_effect=ModuleNotFoundError("sentence_transformers"),
            ),
            self.assertRaisesRegex(ExternalDependencyError, "dense-retrieval"),
        ):
            retriever.build_index()

    def test_documentation_lists_every_built_in_profile(self) -> None:
        text = (Path(__file__).parents[2] / "docs" / "retriever_profiles.md").read_text(
            encoding="utf-8"
        )
        for profile_id in RETRIEVER_PROFILES:
            with self.subTest(profile_id=profile_id):
                self.assertIn(f"`{profile_id}`", text)


if __name__ == "__main__":
    unittest.main()
