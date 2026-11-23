"""Synthetic host/SDK fixtures; these are not real-model acceptance results."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentinputlab.algorithms.curate.llmlingua import (
    LLMLingua2BertMultilingualOfficialAdapter,
)
from agentinputlab.algorithms.curate.llmlingua import multilingual as module
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.contracts import (
    ContextBlock,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)

RID = "curate.llmlingua2.bert_multilingual_official"


class Compressor:
    def compress_prompt(self, contexts, **kwargs):
        assert contexts == ["Visible multilingual evidence."]
        assert "question" not in kwargs and "instruction" not in kwargs
        return {"compressed_prompt": "multilingual evidence", "origin_tokens": 4,
                "compressed_tokens": 2}


def test_registry_resolves_dedicated_pinned_bert_profile():
    adapter = create_intervention(RID)
    assert isinstance(adapter, LLMLingua2BertMultilingualOfficialAdapter)
    assert adapter.model_name == module.BERT_MULTILINGUAL_MODEL
    assert adapter.model_revision == module.BERT_MULTILINGUAL_REVISION
    assert adapter.compressor_factory is None


def test_factory_preserves_native_query_agnostic_dispatch_and_identity():
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return Compressor()

    adapter = create_intervention(RID, compressor_factory=factory)
    state = FrozenAgentState(query="not a compressor input",
        context=(ContextBlock(text="Visible multilingual evidence."),))
    result = adapter.propose(state, InterventionContext())
    assert result.release_id == RID
    assert result.metadata["query_consumed"] is False
    assert result.apply(state).context[0].text == "multilingual evidence"
    assert calls[0]["use_llmlingua2"] is True
    assert calls[0]["model_name"] == module.BERT_MULTILINGUAL_MODEL
    assert calls[0]["model_config"]["revision"] == module.BERT_MULTILINGUAL_REVISION


@pytest.mark.parametrize("kwargs", [
    {"model_name": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"},
    {"model_revision": "main"}, {"model_config": {"revision": "main"}},
    {"model_config": {"trust_remote_code": True}},
    {"model_config": {"subfolder": "other"}},
    {"model_config": {"use_safetensors": False}},
    {"model_config": {"torch_dtype": "int8"}},
    {"model_config": {"local_files_only": "false"}},
    {"profile_kind": "controlled_adapter"}, {"release_id": "other"},
])
def test_relabeling_and_asset_policy_overrides_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        LLMLingua2BertMultilingualOfficialAdapter(**kwargs)


def test_local_snapshot_all_files_verified_before_any_native_loading():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        data = b"synthetic-not-weights"
        (root / "model.safetensors").write_bytes(data)
        pin = (("model.safetensors", hashlib.sha256(data).hexdigest()),)
        module._verify_snapshot.cache_clear()
        with (
            patch.object(module, "BERT_MULTILINGUAL_FILES", pin),
            patch.object(module._OfficialAdapterBase, "_load", return_value="native") as loader,
        ):
            adapter = LLMLingua2BertMultilingualOfficialAdapter(model_name=folder)
            assert adapter._load(use_llmlingua2=True) == "native"
            assert loader.call_count == 1
            (root / "model.safetensors").write_bytes(b"changed")
            with pytest.raises(ExternalDependencyError, match="checksum"):
                adapter._load(use_llmlingua2=True)
            assert loader.call_count == 1
        module._verify_snapshot.cache_clear()


def test_remote_loader_download_is_bounded_to_pinned_files():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        data = b"synthetic-config"
        (root / "config.json").write_bytes(data)
        pin = (("config.json", hashlib.sha256(data).hexdigest()),)
        calls = []

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            return folder

        module._verify_snapshot.cache_clear()
        with (
            patch.object(module, "BERT_MULTILINGUAL_FILES", pin),
            patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}),
            patch.object(module._OfficialAdapterBase, "_load", return_value="native"),
        ):
            adapter = LLMLingua2BertMultilingualOfficialAdapter(model_config={"local_files_only": True})
            assert adapter._load(use_llmlingua2=True) == "native"
        assert calls == [((module.BERT_MULTILINGUAL_MODEL,), {
            "revision": module.BERT_MULTILINGUAL_REVISION,
            "allow_patterns": ["config.json"], "local_files_only": True})]
        module._verify_snapshot.cache_clear()


def test_operator_sdk_service_uses_same_registered_profile():
    service = ToolkitService(factories={RID: lambda: create_intervention(
        RID, compressor_factory=lambda **_: Compressor())})
    result = service.execute({"release_id": RID, "seed": 17,
        "state": {"query": "q", "context": [{"text": "Visible multilingual evidence."}]}})
    assert result["scored"] is False
    assert result["record"]["patch"]["release_id"] == RID
    assert result["state"]["context"][0]["text"] == "multilingual evidence"
