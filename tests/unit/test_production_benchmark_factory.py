from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from agentinputlab.benchmarks import (
    BenchmarkSmokeAssetRegistry,
    ProductionBenchmarkSpec,
    load_production_benchmark_adapter,
    load_tau3_banking_bundle,
    production_benchmark_specs,
)
from agentinputlab.contracts.errors import ArtifactError, ValidationError

EXPECTED_SPECS = (
    ("beir.arguana", "test", "retrieval", "beir.arguana.official", None),
    (
        "bright.earth_science",
        "test",
        "retrieval",
        "bright.earth_science.examples",
        "bright.earth_science.corpus",
    ),
    (
        "bright.economics",
        "test",
        "retrieval",
        "bright.economics.examples",
        "bright.economics.corpus",
    ),
    (
        "bright.robotics",
        "test",
        "retrieval",
        "bright.robotics.examples",
        "bright.robotics.corpus",
    ),
    (
        "bright.stackoverflow",
        "test",
        "retrieval",
        "bright.stackoverflow.examples",
        "bright.stackoverflow.corpus",
    ),
    ("musique.answerable", "dev", "qa", "musique.answerable.dev", None),
    ("2wiki.multihop", "dev", "qa", "2wiki.multihop.dev", None),
    (
        "hotpotqa.distractor",
        "validation",
        "qa",
        "hotpotqa.distractor.validation",
        None,
    ),
    ("qasper.v0_3_dev", "dev", "qa", "qasper.dev.official", None),
    (
        "tau3.banking_knowledge",
        "base",
        "checkpoint",
        "tau3.banking_knowledge.v1_0_1",
        None,
    ),
)


def _asset(asset_id: str, cell_id: str) -> SimpleNamespace:
    return SimpleNamespace(asset_id=asset_id, cell_id=cell_id)


class ProductionBenchmarkFactoryTests(unittest.TestCase):
    def test_inventory_is_the_locked_public_surface(self) -> None:
        specs = production_benchmark_specs()

        self.assertIsInstance(specs, tuple)
        self.assertEqual(
            tuple(
                (
                    spec.cell_id,
                    spec.split,
                    spec.adapter_kind,
                    spec.query_asset_id,
                    spec.corpus_asset_id,
                )
                for spec in specs
            ),
            EXPECTED_SPECS,
        )

    def test_inventory_asset_ids_resolve_to_the_same_packaged_cell(self) -> None:
        registry = BenchmarkSmokeAssetRegistry.default()

        for spec in production_benchmark_specs():
            with self.subTest(cell_id=spec.cell_id, asset_id=spec.query_asset_id):
                self.assertEqual(registry.get(spec.query_asset_id).cell_id, spec.cell_id)
            if spec.corpus_asset_id is not None:
                with self.subTest(cell_id=spec.cell_id, asset_id=spec.corpus_asset_id):
                    self.assertEqual(registry.get(spec.corpus_asset_id).cell_id, spec.cell_id)

    def test_spec_validation_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            ProductionBenchmarkSpec("", "test", "qa", "asset")
        with self.assertRaisesRegex(ValueError, "checkpoint, qa, or retrieval"):
            ProductionBenchmarkSpec("cell", "test", "other", "asset")
        with self.assertRaisesRegex(ValueError, "QA specs cannot"):
            ProductionBenchmarkSpec("cell", "test", "qa", "asset", "corpus")
        with self.assertRaisesRegex(ValueError, "checkpoint specs cannot"):
            ProductionBenchmarkSpec("cell", "base", "checkpoint", "asset", "corpus")

    @patch("agentinputlab.benchmarks.production.PreparedQABenchmarkAdapter")
    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_qa_cells_load_the_locked_asset_and_split(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
        qa_adapter: MagicMock,
    ) -> None:
        prepared_root = Path("prepared-cache")
        registry = MagicMock()
        default_registry.return_value = registry
        load_prepared.side_effect = lambda asset, _root: f"prepared:{asset.asset_id}"
        qa_adapter.side_effect = lambda *_args, **kwargs: kwargs

        for cell_id, split, kind, asset_id, corpus_id in EXPECTED_SPECS:
            if kind != "qa":
                continue
            with self.subTest(cell_id=cell_id):
                registry.reset_mock()
                load_prepared.reset_mock()
                qa_adapter.reset_mock()
                registry.get.return_value = _asset(asset_id, cell_id)

                result = load_production_benchmark_adapter(cell_id, prepared_root)

                registry.get.assert_called_once_with(asset_id)
                load_prepared.assert_called_once_with(registry.get.return_value, prepared_root)
                qa_adapter.assert_called_once_with(
                    f"prepared:{asset_id}",
                    benchmark_id=cell_id,
                    split=split,
                )
                self.assertEqual(result, {"benchmark_id": cell_id, "split": split})
                self.assertIsNone(corpus_id)

    @patch("agentinputlab.benchmarks.production.PreparedRetrievalBenchmarkAdapter")
    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_arguana_uses_its_combined_query_gold_corpus_asset(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
        retrieval_adapter: MagicMock,
    ) -> None:
        prepared_root = Path("prepared-cache")
        asset = _asset("beir.arguana.official", "beir.arguana")
        registry = MagicMock()
        registry.get.return_value = asset
        default_registry.return_value = registry
        prepared = object()
        load_prepared.return_value = prepared
        expected_adapter = object()
        retrieval_adapter.return_value = expected_adapter

        result = load_production_benchmark_adapter("beir.arguana", prepared_root)

        registry.get.assert_called_once_with("beir.arguana.official")
        load_prepared.assert_called_once_with(asset, prepared_root)
        retrieval_adapter.assert_called_once_with(
            prepared,
            benchmark_id="beir.arguana",
            split="test",
            corpus_prepared=None,
        )
        self.assertIs(result, expected_adapter)

    @patch("agentinputlab.benchmarks.production.PreparedRetrievalBenchmarkAdapter")
    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_bright_cells_load_separate_locked_example_and_corpus_assets(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
        retrieval_adapter: MagicMock,
    ) -> None:
        prepared_root = Path("prepared-cache")
        registry = MagicMock()
        default_registry.return_value = registry
        assets = {
            asset_id: _asset(asset_id, cell_id)
            for cell_id, _split, kind, asset_id, corpus_id in EXPECTED_SPECS
            if kind == "retrieval"
            for asset_id in (asset_id, corpus_id)
            if asset_id is not None
        }
        registry.get.side_effect = assets.__getitem__
        load_prepared.side_effect = lambda asset, _root: f"prepared:{asset.asset_id}"

        for cell_id, split, kind, query_id, corpus_id in EXPECTED_SPECS:
            if kind != "retrieval" or corpus_id is None:
                continue
            with self.subTest(cell_id=cell_id):
                registry.get.reset_mock()
                load_prepared.reset_mock()
                retrieval_adapter.reset_mock()

                load_production_benchmark_adapter(cell_id, prepared_root)

                self.assertEqual(registry.get.call_args_list, [call(query_id), call(corpus_id)])
                self.assertEqual(
                    load_prepared.call_args_list,
                    [
                        call(assets[query_id], prepared_root),
                        call(assets[corpus_id], prepared_root),
                    ],
                )
                retrieval_adapter.assert_called_once_with(
                    f"prepared:{query_id}",
                    benchmark_id=cell_id,
                    split=split,
                    corpus_prepared=f"prepared:{corpus_id}",
                )

    @patch("agentinputlab.benchmarks.production.Tau3BankingCheckpointAdapter")
    @patch("agentinputlab.benchmarks.production.PreparedTau3BankingBundle")
    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_tau3_factory_requires_and_forwards_explicit_checkpoints(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
        bundle_type: MagicMock,
        adapter_type: MagicMock,
    ) -> None:
        asset = _asset("tau3.banking_knowledge.v1_0_1", "tau3.banking_knowledge")
        registry = MagicMock()
        registry.get.return_value = asset
        default_registry.return_value = registry
        prepared = object()
        load_prepared.return_value = prepared
        bundle = object()
        bundle_type.return_value = bundle
        checkpoint = object()
        evaluator = MagicMock()
        expected = object()
        adapter_type.return_value = expected

        result = load_production_benchmark_adapter(
            "tau3.banking_knowledge",
            "prepared-cache",
            checkpoints=(checkpoint,),  # type: ignore[arg-type]
            evaluation_bridge=evaluator,
        )

        registry.get.assert_called_once_with("tau3.banking_knowledge.v1_0_1")
        load_prepared.assert_called_once_with(asset, "prepared-cache")
        bundle_type.assert_called_once_with(prepared)
        adapter_type.assert_called_once_with(
            bundle,
            checkpoints=(checkpoint,),
            evaluation_bridge=evaluator,
        )
        self.assertIs(result, expected)

    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_tau3_static_bundle_loader_does_not_invent_a_checkpoint(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
    ) -> None:
        asset = _asset("tau3.banking_knowledge.v1_0_1", "tau3.banking_knowledge")
        registry = MagicMock()
        registry.get.return_value = asset
        default_registry.return_value = registry
        prepared = object()
        load_prepared.return_value = prepared
        expected_bundle = object()

        with patch(
            "agentinputlab.benchmarks.production.PreparedTau3BankingBundle",
            return_value=expected_bundle,
        ) as bundle_type:
            result = load_tau3_banking_bundle("prepared-cache")

        registry.get.assert_called_once_with("tau3.banking_knowledge.v1_0_1")
        load_prepared.assert_called_once_with(asset, "prepared-cache")
        bundle_type.assert_called_once_with(prepared)
        self.assertIs(result, expected_bundle)

    def test_tau3_factory_fails_closed_without_runtime_bridge(self) -> None:
        with patch(
            "agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default"
        ) as default_registry:
            default_registry.return_value.get.return_value = _asset(
                "tau3.banking_knowledge.v1_0_1",
                "tau3.banking_knowledge",
            )
            with (
                patch(
                    "agentinputlab.benchmarks.production.load_prepared_benchmark",
                    return_value=object(),
                ),
                self.assertRaisesRegex(
                    ValidationError,
                    "requires externally captured checkpoints",
                ),
            ):
                load_production_benchmark_adapter("tau3.banking_knowledge", "prepared-cache")

    def test_unknown_or_empty_cell_fails_before_registry_access(self) -> None:
        with patch(
            "agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default"
        ) as default_registry:
            for cell_id in ("", "unknown.cell"):
                with self.subTest(cell_id=cell_id), self.assertRaises(ValidationError):
                    load_production_benchmark_adapter(cell_id, "prepared-cache")
            default_registry.assert_not_called()

    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_registry_query_identity_drift_fails_closed(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
    ) -> None:
        registry = MagicMock()
        default_registry.return_value = registry

        for asset in (
            _asset("wrong.asset", "musique.answerable"),
            _asset("musique.answerable.dev", "wrong.cell"),
        ):
            with self.subTest(asset=asset):
                registry.get.return_value = asset
                with self.assertRaisesRegex(ArtifactError, "query asset identity"):
                    load_production_benchmark_adapter("musique.answerable", "prepared-cache")
        load_prepared.assert_not_called()

    @patch("agentinputlab.benchmarks.production.load_prepared_benchmark")
    @patch("agentinputlab.benchmarks.production.BenchmarkSmokeAssetRegistry.default")
    def test_registry_corpus_identity_drift_fails_closed(
        self,
        default_registry: MagicMock,
        load_prepared: MagicMock,
    ) -> None:
        registry = MagicMock()
        registry.get.side_effect = (
            _asset("bright.earth_science.examples", "bright.earth_science"),
            _asset("wrong.corpus", "bright.earth_science"),
        )
        default_registry.return_value = registry
        load_prepared.return_value = object()

        with self.assertRaisesRegex(ArtifactError, "corpus asset identity"):
            load_production_benchmark_adapter("bright.earth_science", "prepared-cache")
        load_prepared.assert_called_once()


if __name__ == "__main__":
    unittest.main()
