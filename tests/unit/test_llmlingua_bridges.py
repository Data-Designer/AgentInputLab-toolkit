from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentinputlab.algorithms.curate.llmlingua import (
    LLMLingua2OfficialAdapter,
    LongLLMLinguaConfig,
    LongLLMLinguaOfficialAdapter,
    build_span_map,
    parse_context_budget,
)
from agentinputlab.algorithms.curate.llmlingua.adapters import (
    _bind_native_tokenizer_family,
    _cached_official_compressor,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)


class FakeCompressor:
    def __init__(self, *, llmlingua2):
        self.llmlingua2 = llmlingua2
        self.calls = []

    def compress_prompt(self, contexts, **kwargs):
        self.calls.append((tuple(contexts), dict(kwargs)))
        if self.llmlingua2:
            return {
                "compressed_prompt": "Alpha fact.",
                "origin_tokens": 20,
                "compressed_tokens": 3,
                "ratio": "6.7x",
                "rate": "15.0%",
                "saving": "native",
                "fn_labeled_original_prompt": "fixture",
            }
        compressed = "Alpha fact."
        instruction = kwargs["instruction"]
        question = kwargs["question"]
        return {
            "compressed_prompt": f"{instruction}\n\n{compressed}\n\n{question}",
            "origin_tokens": 30,
            "compressed_tokens": 8,
            "ratio": "3.8x",
            "rate": "26.7%",
            "saving": "native",
        }


class FakeFactory:
    def __init__(self):
        self.kwargs = None
        self.compressor = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.compressor = FakeCompressor(llmlingua2=kwargs["use_llmlingua2"])
        return self.compressor


class BudgetTests(unittest.TestCase):
    def test_allowed_context_budget_forms_are_canonical(self):
        self.assertEqual(parse_context_budget(5), "+5")
        self.assertEqual(parse_context_budget("+005"), "+5")
        self.assertEqual(parse_context_budget("-7"), "-7")
        self.assertEqual(parse_context_budget("*1.50"), "*1.5")

    def test_code_and_unbounded_values_are_rejected(self):
        for value in (
            "__import__('os').system('id')",
            "+1 or 1",
            "**2",
            "/2",
            "*0",
            "*101",
            "+1000001",
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                parse_context_budget(value)


class SpanMapTests(unittest.TestCase):
    def test_reordered_exact_spans_map_to_source_blocks(self):
        blocks = (
            ContextBlock(block_id="a", text="Alpha fact. Extra."),
            ContextBlock(block_id="b", text="Beta fact. More."),
        )
        spans = build_span_map("Beta fact.\nAlpha fact.", blocks)
        self.assertEqual({item.source_block_id for item in spans}, {"a", "b"})
        self.assertTrue(all(item.exact for item in spans))


class NativeTokenizerFamilyTests(unittest.TestCase):
    """API fixtures only; no neural acceptance or model quality is claimed."""

    def test_actual_family_routes_generic_path_without_replacing_model(self):
        for kind, family in (
            ("bert", "bert-base-multilingual-cased"),
            ("xlm-roberta", "xlm-roberta-large"),
        ):
            with self.subTest(kind=kind):
                model = SimpleNamespace(config=SimpleNamespace(model_type=kind))
                native = SimpleNamespace(model=model, model_name="/offline/snapshot")
                with patch("importlib.metadata.version", return_value="0.2.2"):
                    _bind_native_tokenizer_family(native, "/offline/snapshot")
                    _bind_native_tokenizer_family(native, "/offline/snapshot")
                self.assertIs(native.model, model)
                self.assertEqual(native.model_name, family)

    def test_unreviewed_version_is_rejected_before_mutation(self):
        native = SimpleNamespace(model_name="/offline/snapshot")
        with (
            patch("importlib.metadata.version", return_value="unreviewed"),
            self.assertRaisesRegex(ExternalDependencyError, "0.2.2"),
        ):
            _bind_native_tokenizer_family(native, "/offline/snapshot")
        self.assertEqual(native.model_name, "/offline/snapshot")

    def test_unsupported_config_and_wrong_source_are_rejected(self):
        for kind, path in (("llama", "/offline/snapshot"), ("bert", "/other/weights")):
            native = SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(model_type=kind)), model_name=path
            )
            with (
                patch("importlib.metadata.version", return_value="0.2.2"),
                self.assertRaisesRegex(ValidationError, "source binding"),
            ):
                _bind_native_tokenizer_family(native, "/offline/snapshot")
            self.assertEqual(native.model_name, path)

    def test_builtin_loader_keeps_actual_load_path_and_cache_identity(self):
        calls = []
        model = SimpleNamespace(config=SimpleNamespace(model_type="bert"))

        def factory(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(model=model, model_name=kwargs["model_name"])

        _cached_official_compressor.cache_clear()
        try:
            with (
                patch.dict(sys.modules, {
                    "torch": SimpleNamespace(float32="fixture-dtype"),
                    "llmlingua": SimpleNamespace(PromptCompressor=factory),
                }),
                patch("importlib.metadata.version", return_value="0.2.2"),
            ):
                args = ("/offline/snapshot", "cpu", '{"revision":"fixed"}', "float32", True)
                first = _cached_official_compressor(*args)
                second = _cached_official_compressor(*args)
            self.assertIs(first, second)
            self.assertIs(first.model, model)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["model_name"], "/offline/snapshot")
            self.assertEqual(calls[0]["model_config"]["revision"], "fixed")
            self.assertEqual(first.model_name, "bert-base-multilingual-cased")
        finally:
            _cached_official_compressor.cache_clear()

    def test_longllmlingua_causal_path_is_not_changed(self):
        native = SimpleNamespace(model_name="/causal/weights")
        _cached_official_compressor.cache_clear()
        try:
            with (
                patch.dict(sys.modules, {
                    "torch": SimpleNamespace(float32="fixture-dtype"),
                    "llmlingua": SimpleNamespace(PromptCompressor=lambda **kwargs: native),
                }),
                patch("importlib.metadata.version", side_effect=AssertionError("must not route")),
            ):
                actual = _cached_official_compressor("/causal/weights", "cpu", "{}", "float32", False)
            self.assertIs(actual, native)
            self.assertEqual(native.model_name, "/causal/weights")
        finally:
            _cached_official_compressor.cache_clear()


class RuntimeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.state = FrozenAgentState(
            query="Which fact matters?",
            instruction="Answer from evidence.",
            context=(
                ContextBlock(block_id="a", text="Alpha fact. Extra."),
                ContextBlock(block_id="b", text="Beta fact. More."),
            ),
        )

    def test_longllmlingua_calls_official_runtime_and_strips_wrapper(self):
        factory = FakeFactory()
        adapter = LongLLMLinguaOfficialAdapter(
            compressor_factory=factory,
            config=LongLLMLinguaConfig(context_budget="*1.50"),
        )
        patch_value = adapter.propose(self.state, InterventionContext())
        materialized = patch_value.apply(self.state)

        self.assertEqual(materialized.context[0].text, "Alpha fact.")
        call = factory.compressor.calls[0][1]
        self.assertEqual(call["rank_method"], "longllmlingua")
        self.assertEqual(call["context_budget"], "*1.5")
        self.assertEqual(call["reorder_context"], "sort")
        self.assertEqual(call["question"], self.state.query)
        self.assertTrue(patch_value.metadata["query_consumed"])
        self.assertEqual(patch_value.profile_kind, "official_native")
        self.assertEqual(factory.kwargs["model_config"]["trust_remote_code"], False)
        self.assertEqual(
            factory.kwargs["model_config"]["revision"],
            adapter.model_revision,
        )
        self.assertTrue(materialized.context[0].metadata["span_map"])

    def test_llmlingua2_native_dispatch_does_not_receive_query_or_instruction(self):
        factory = FakeFactory()
        adapter = LLMLingua2OfficialAdapter(compressor_factory=factory)
        patch_value = adapter.propose(self.state, InterventionContext())
        call = factory.compressor.calls[0][1]

        self.assertNotIn("question", call)
        self.assertNotIn("instruction", call)
        self.assertFalse(patch_value.metadata["query_consumed"])
        self.assertEqual(patch_value.apply(self.state).context[0].text, "Alpha fact.")

    def test_controlled_profile_is_explicit_and_preserves_runtime_logic(self):
        factory = FakeFactory()
        adapter = LLMLingua2OfficialAdapter(
            compressor_factory=factory,
            profile_kind="controlled_adapter",
        )

        patch_value = adapter.propose(self.state, InterventionContext())

        self.assertEqual(patch_value.profile_kind, "controlled_adapter")
        self.assertEqual(len(factory.compressor.calls), 1)

    def test_unknown_profile_kind_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "profile_kind"):
            LLMLingua2OfficialAdapter(profile_kind="moving-profile")

    def test_runtime_config_cannot_override_pinned_loading_policy(self):
        for model_config in (
            {"revision": "moving-main"},
            {"trust_remote_code": True},
            {"torch_dtype": "int8"},
        ):
            with self.subTest(model_config=model_config), self.assertRaises(ValidationError):
                LongLLMLinguaOfficialAdapter(
                    compressor_factory=FakeFactory(),
                    model_config=model_config,
                ).propose(self.state, InterventionContext())

    def test_missing_optional_package_is_typed(self):
        adapter = LongLLMLinguaOfficialAdapter()
        with (
            patch.dict(sys.modules, {"llmlingua": None}),
            self.assertRaises(ExternalDependencyError),
        ):
            adapter.propose(self.state, InterventionContext())

    def test_native_weight_cache_is_not_stored_in_intervention_configuration(self):
        from agentinputlab.runner import InterventionRunner

        compressor = FakeCompressor(llmlingua2=True)
        compressor.large_tensor_graph = tuple(object() for _ in range(5000))
        adapter = LLMLingua2OfficialAdapter()
        with (
            patch.dict(sys.modules, {'llmlingua': SimpleNamespace(PromptCompressor=object)}),
            patch.object(type(adapter), '_runtime_model_config', new=lambda self, config: {'torch_dtype':'torch.float32'}),
            patch('agentinputlab.algorithms.curate.llmlingua.adapters._cached_official_compressor', return_value=compressor),
        ):
            for _ in range(2):
                result = InterventionRunner().run(adapter, self.state, InterventionContext())
                self.assertEqual(result.patch.apply(self.state).context[0].text, 'Alpha fact.')
        self.assertIsNone(adapter._compressor)
        self.assertEqual(len(compressor.calls), 2)

    def test_modern_transformers_cache_is_bridged_to_upstream_legacy_shape(self):
        class ReturnedCache:
            def to_legacy_cache(self):
                return (("legacy-key", "legacy-value"),)

        class FakeDynamicCache:
            observed = None

            @classmethod
            def from_legacy_cache(cls, value):
                cls.observed = value
                return cls()

        class Model:
            def __init__(self):
                self.received = None

            def forward(self, **kwargs):
                self.received = kwargs["past_key_values"]
                return SimpleNamespace(past_key_values=ReturnedCache())

        compressor = SimpleNamespace(model=Model())
        adapter = LongLLMLinguaOfficialAdapter(compressor_factory=FakeFactory())
        cache_module = SimpleNamespace(DynamicCache=FakeDynamicCache)
        with patch(
            "agentinputlab.algorithms.curate.llmlingua.adapters.importlib.import_module",
            return_value=cache_module,
        ):
            adapter._install_legacy_cache_bridge(compressor)

        response = compressor.model.forward(past_key_values=[[("key",), ("value",)]])
        self.assertIsInstance(compressor.model.received, FakeDynamicCache)
        self.assertEqual(
            FakeDynamicCache.observed,
            ((("key",), ("value",)),),
        )
        self.assertEqual(
            response.past_key_values,
            (("legacy-key", "legacy-value"),),
        )

    def test_transformers_v5_cache_is_bridged_without_removed_helpers(self):
        class Layer:
            keys = "returned-key"
            values = "returned-value"

        class ReturnedCache:
            layers = (Layer(),)

        class FakeDynamicCache:
            observed = None

            def __init__(self, *, ddp_cache_data):
                type(self).observed = ddp_cache_data

        class Model:
            def __init__(self):
                self.received = None

            def forward(self, **kwargs):
                self.received = kwargs["past_key_values"]
                return SimpleNamespace(past_key_values=ReturnedCache())

        compressor = SimpleNamespace(model=Model())
        adapter = LongLLMLinguaOfficialAdapter(compressor_factory=FakeFactory())
        cache_module = SimpleNamespace(DynamicCache=FakeDynamicCache)
        with patch(
            "agentinputlab.algorithms.curate.llmlingua.adapters.importlib.import_module",
            return_value=cache_module,
        ):
            adapter._install_legacy_cache_bridge(compressor)

        response = compressor.model.forward(past_key_values=[["key", "value"]])
        self.assertIsInstance(compressor.model.received, FakeDynamicCache)
        self.assertEqual(FakeDynamicCache.observed, (("key", "value"),))
        self.assertEqual(
            response.past_key_values,
            (("returned-key", "returned-value"),),
        )

    def test_empty_context_fails_instead_of_silent_noop(self):
        adapter = LLMLingua2OfficialAdapter(compressor_factory=FakeFactory())
        with self.assertRaises(ValidationError):
            adapter.propose(
                FrozenAgentState(query="q"),
                InterventionContext(),
            )

    def test_malformed_runtime_result_is_typed_as_arm_local_output_failure(self):
        class MalformedCompressor:
            def compress_prompt(self, contexts, **kwargs):
                return {"origin_tokens": 12}

        def factory(**kwargs):
            return MalformedCompressor()

        adapter = LLMLingua2OfficialAdapter(compressor_factory=factory)
        with self.assertRaisesRegex(AlgorithmOutputError, "compressed_prompt"):
            adapter.propose(self.state, InterventionContext())

    def test_longllmlingua_wrapper_mismatch_is_arm_local_output_failure(self):
        class BoundaryDroppingCompressor:
            def compress_prompt(self, contexts, **kwargs):
                return {"compressed_prompt": "Alpha fact."}

        def factory(**kwargs):
            return BoundaryDroppingCompressor()

        adapter = LongLLMLinguaOfficialAdapter(compressor_factory=factory)
        with self.assertRaisesRegex(AlgorithmOutputError, "instruction boundary"):
            adapter.propose(self.state, InterventionContext())


if __name__ == "__main__":
    unittest.main()
