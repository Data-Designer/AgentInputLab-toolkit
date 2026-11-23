"""Data-only, synthetic comparison/SDK fixtures; no pretrained model results."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from agentinputlab import ContextBlock, ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.curate import (
    LongLLMLinguaConfig,
    LongLLMLinguaControlled,
    LongLLMLinguaControlledBindings,
    LongLLMLinguaOfficialAdapter,
)
from agentinputlab.algorithms.curate.llmlingua.controlled import LongLLMLinguaCompressionRequest
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.contracts import AlgorithmOutputError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.integrations.external_workers.longllmlingua import LongLLMLinguaNativeRuntime
from agentinputlab.runner import InterventionRunner

RID = "curate.longllmlingua.controlled"


class Reader:
    def fingerprint(self):
        return {"model": "fixture-reader", "revision": "fixture"}

    def generate(self, request):
        raise AssertionError("curator must not invoke the reader")


class Factory:
    def __call__(self, **kwargs):
        assert kwargs["use_llmlingua2"] is False
        return self

    def compress_prompt(self, contexts, **kwargs):
        assert contexts == ["Alpha evidence. Extra."]
        return {"compressed_prompt": f'{kwargs["instruction"]}\n\nAlpha evidence.\n\n{kwargs["question"]}',
                "origin_tokens": 8, "compressed_tokens": 4, "rate": "50%", "ratio": "2x"}


class Runtime:
    def __init__(self):
        self.revision = "a" * 40
        self.called = 0
        self.mutate = False

    def fingerprint(self):
        return {"model": {"locator": "fixture-model", "revision": self.revision},
                "tokenizer": {"revision": "fixture"}, "source": {"version": "fixture"}}

    def run(self, request):
        state, config, seed = request.state, request.config, request.seed
        self.called += 1
        result = LongLLMLinguaOfficialAdapter(model_name="fixture-model", model_revision=self.revision,
            config=config, compressor_factory=Factory()).propose(state, InterventionContext(seed=seed))
        if self.mutate:
            self.revision = "b" * 40
        return {"patch": result, "usage": {"model_calls":0,"retrieval_calls":0,"input_tokens":0,"output_tokens":0}}


def fixture():
    state = FrozenAgentState(query="Question?", instruction="Answer.",
        context=(ContextBlock(text="Alpha evidence. Extra."),))
    context = InterventionContext(backend=Reader(), external_runtime=Runtime(), seed=17)
    bindings = LongLLMLinguaControlledBindings.from_context(context, budget=state.budget)
    return state, context, bindings


def test_frozen_constructor_and_native_dispatch_via_instrumented_context():
    state, context, bindings = fixture()
    adapter = create_intervention(RID, bindings=to_jsonable(bindings))
    record = InterventionRunner().run(adapter, state, context)
    assert record.patch.apply(state).context[0].text == "Alpha evidence."
    assert record.patch.profile_kind == "controlled_adapter" and record.patch.release_id == RID
    assert record.usage.model_calls == 0
    assert context.external_runtime.called == 1
    assert record.patch.metadata["reader_generation_performed"] is False
    assert record.patch.metadata["controlled_binding_sha256"] == bindings.binding_hash


def test_config_budget_canonicalization_and_detached_mapping():
    _, context, bindings = fixture()
    raw = to_jsonable(bindings)
    raw["config"]["context_budget"] = "+00100"
    parsed = LongLLMLinguaControlledBindings.from_mapping(raw)
    assert parsed.config.context_budget == "+100"
    before = parsed.binding_hash
    raw["reader"]["revision"] = "changed"
    assert parsed.binding_hash == before
    assert parsed.reader == bindings.reader
    assert context.external_runtime.called == 0


@pytest.mark.parametrize("change", ["reader", "compressor", "seed", "budget"])
def test_changed_comparison_dependency_is_rejected_before_compression(change):
    state, context, bindings = fixture()
    if change == "reader":
        context.backend.fingerprint = lambda: {"model": "other"}
    elif change == "compressor":
        context.external_runtime.revision = "b" * 40
    elif change == "seed":
        context = replace(context, seed=18)
    else:
        state = replace(state, budget=ExecutionBudget(max_model_calls=7))
    with pytest.raises(ValidationError):
        LongLLMLinguaControlled(bindings).propose(state, context)
    assert context.external_runtime.called == 0


def test_runtime_identity_mutation_during_execution_is_not_accepted():
    state, context, bindings = fixture()
    context.external_runtime.mutate = True
    with pytest.raises(ValidationError, match="compressor"):
        LongLLMLinguaControlled(bindings).propose(state, context)
    assert context.external_runtime.called == 1


def test_wrong_native_source_or_settings_is_not_relabeled():
    state, context, bindings = fixture()
    original = context.external_runtime.run

    def wrong(request):
        result = original(request)
        return {**result, "patch": replace(result["patch"], source_state_hash="other")}

    context.external_runtime.run = wrong
    with pytest.raises(AlgorithmOutputError, match="identity"):
        LongLLMLinguaControlled(bindings).propose(state, context)


def test_missing_runtime_has_typed_dependency_failure():
    state, _, bindings = fixture()
    with pytest.raises(ExternalDependencyError):
        LongLLMLinguaControlled(bindings).propose(state, InterventionContext(seed=17))


def test_operator_sdk_service_uses_registered_controlled_profile():
    state, context, bindings = fixture()
    service = ToolkitService(factories={RID: lambda: create_intervention(RID, bindings=to_jsonable(bindings))},
        context_factory=lambda seed: replace(context, seed=seed))
    result = service.execute({"release_id": RID, "state": to_jsonable(state), "seed": 17})
    assert result["scored"] is False and result["state"]["context"][0]["text"] == "Alpha evidence."


def test_native_runtime_injected_fixture_is_explicit_and_does_not_claim_weight_verification():
    with TemporaryDirectory() as folder:
        pins = {n: "a"*64 for n in ("config.json", "tokenizer_config.json", "special_tokens_map.json",
                                    "tokenizer.json", "model.safetensors")}
        runtime = LongLLMLinguaNativeRuntime(folder, model_id="fixture-model", revision="a"*40,
            file_sha256=pins, compressor_factory=Factory())
        with patch.object(runtime, "_source", return_value={"package": "synthetic-fixture"}):
            assert runtime.fingerprint()["injected_factory"] is True
            assert runtime.fingerprint()["file_verification"] == "caller_runtime"
            state, _, _ = fixture()
            native = runtime.run(LongLLMLinguaCompressionRequest(state, LongLLMLinguaConfig(), 17))["patch"]
            assert native.metadata["model_name"] == str(Path(folder).resolve())
            assert native.apply(state).context[0].text == "Alpha evidence."
            assert native.trace.events[-1].payload["helper_compute_free"] is False


def test_native_runtime_rejects_missing_live_assets_before_native_execution():
    with TemporaryDirectory() as folder:
        pins = {n: "a"*64 for n in ("config.json", "tokenizer_config.json", "special_tokens_map.json",
                                    "tokenizer.json", "model.safetensors")}
        runtime = LongLLMLinguaNativeRuntime(folder, model_id="fixture-model", revision="a"*40,
            file_sha256=pins)
        state, _, _ = fixture()
        with (patch.object(runtime, "_source", return_value={"package": "synthetic-fixture"}),
              pytest.raises(ExternalDependencyError, match="incomplete")):
            runtime.run(LongLLMLinguaCompressionRequest(state, LongLLMLinguaConfig(), 17))


def test_recipe_contains_only_identity_data_not_runtime_objects():
    _, _, bindings = fixture()
    assert content_hash(to_jsonable(bindings)) == bindings.binding_hash
    with pytest.raises(ValidationError):
        LongLLMLinguaControlledBindings.from_mapping({**to_jsonable(bindings), "runtime": Runtime()})


@pytest.mark.parametrize("mode", ["undeclared_sidecar", "unlisted_shard", "noncausal"])
def test_live_loader_rejects_incomplete_or_mismatched_auto_load_closure(mode):
    with TemporaryDirectory() as folder:
        root = Path(folder)
        (root/'config.json').write_text(json.dumps({"architectures": [
            "BertForTokenClassification" if mode == "noncausal" else "LlamaForCausalLM"]}))
        pins = {n: "a"*64 for n in ("config.json", "tokenizer_config.json", "special_tokens_map.json",
                                    "tokenizer.json", "model.safetensors")}
        if mode == "undeclared_sidecar":
            (root/'added_tokens.json').write_text('{}')
        if mode == "unlisted_shard":
            (root/'model.safetensors.index.json').write_text(json.dumps({"weight_map": {"weight":"missing.safetensors"}}))
            pins['model.safetensors.index.json'] = 'a'*64
        runtime = LongLLMLinguaNativeRuntime(folder, model_id="fixture-model", revision="a"*40, file_sha256=pins)
        with pytest.raises(ExternalDependencyError):
            runtime._require_loader_closure()
