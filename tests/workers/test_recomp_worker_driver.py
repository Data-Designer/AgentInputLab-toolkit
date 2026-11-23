from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import SkipTest
from unittest.mock import MagicMock, patch

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.algorithms.curate.recomp import (
    RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
    RECOMPAbstractiveControls,
    RECOMPAbstractiveExternal,
    RECOMPExtractiveControls,
    RECOMPExtractiveExternal,
)
from agentinputlab.contracts import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)
from workers.recomp import driver


def test_legacy_cpu_build_is_explicit_and_does_not_widen_versions():
    manifest = {
        "legacy_dependencies": {
            "required_versions": {
                "torch": "1.13.0",
                "transformers": "4.24.0",
                "tokenizers": "0.13.2",
            }
        }
    }
    versions = {"torch": "1.13.0+cpu", "transformers": "4.24.0", "tokenizers": "0.13.2"}
    with patch.object(driver.importlib.metadata, "version", side_effect=versions.__getitem__):
        driver._verify_dependencies(manifest, mode="abstractive", device="cpu")
        with pytest.raises(driver.DriverError, match="requires torch"):
            driver._verify_dependencies(manifest, mode="abstractive", device="cuda:0")
        versions["torch"] = "1.13.1+cpu"
        with pytest.raises(driver.DriverError, match="requires torch"):
            driver._verify_dependencies(manifest, mode="abstractive", device="cpu")
        versions["torch"] = "1.13.0+unverified"
        with pytest.raises(driver.DriverError, match="requires torch"):
            driver._verify_dependencies(manifest, mode="abstractive", device="cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:2"])
def test_legacy_official_cuda117_build_keeps_public_version_pin(device):
    manifest = {"legacy_dependencies": {"required_versions": {
        "torch": "1.13.0", "transformers": "4.24.0", "tokenizers": "0.13.2",
    }}}
    versions = {"torch": "1.13.0+cu117", "transformers": "4.24.0", "tokenizers": "0.13.2"}
    with patch.object(driver.importlib.metadata, "version", side_effect=versions.__getitem__):
        driver._verify_dependencies(manifest, mode="abstractive", device=device)


@pytest.mark.parametrize("version", [
    "1.13.1+cu117", "1.13.0+cu116", "1.13.0+cu118", "1.13.0+unverified", "2.0.0+cu117",
])
def test_legacy_cuda_gate_rejects_unvalidated_builds_and_public_versions(version):
    manifest = {"legacy_dependencies": {"required_versions": {
        "torch": "1.13.0", "transformers": "4.24.0", "tokenizers": "0.13.2",
    }}}
    versions = {"torch": version, "transformers": "4.24.0", "tokenizers": "0.13.2"}
    with (
        patch.object(driver.importlib.metadata, "version", side_effect=versions.__getitem__),
        pytest.raises(driver.DriverError, match="requires torch"),
    ):
        driver._verify_dependencies(manifest, mode="abstractive", device="cuda:0")


@pytest.mark.parametrize("device", ["", "cuda:invalid", "cuda:-1", "mps"])
def test_legacy_cuda_build_requires_a_supported_explicit_device(device):
    manifest = {"legacy_dependencies": {"required_versions": {
        "torch": "1.13.0", "transformers": "4.24.0", "tokenizers": "0.13.2",
    }}}
    versions = {"torch": "1.13.0+cu117", "transformers": "4.24.0", "tokenizers": "0.13.2"}
    with (
        patch.object(driver.importlib.metadata, "version", side_effect=versions.__getitem__),
        pytest.raises(driver.DriverError, match="requires torch"),
    ):
        driver._verify_dependencies(manifest, mode="abstractive", device=device)


REAL_CHECKOUT = Path(os.environ.get("AGENTINPUTLAB_RECOMP_CHECKOUT", "/private/tmp/agentinputlab-upstreams/recomp"))


def test_legacy_default_loader_refuses_pickle_before_import_or_load(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"untrusted fixture")
    with (
        patch.object(driver.importlib, "import_module") as load,
        pytest.raises(driver.DriverError, match="safetensors") as error,
    ):
        driver._default_model_loader("abstractive", tmp_path, "cpu")
    assert error.value.code == "CHECKPOINT_FORMAT_UNSAFE"
    load.assert_not_called()


@pytest.mark.parametrize("mode", ["extractive", "abstractive"])
def test_legacy_default_loader_forces_safe_local_weights(tmp_path, mode):
    (tmp_path / "model.safetensors").write_bytes(b"fixture, not real weights")
    tokenizer, model_class, model = MagicMock(), MagicMock(), MagicMock()
    model_class.from_pretrained.return_value = model
    model.to.return_value = model
    module = SimpleNamespace(
        AutoTokenizer=tokenizer,
        AutoModel=model_class,
        AutoModelForSeq2SeqLM=model_class,
        utils=SimpleNamespace(is_safetensors_available=lambda: True),
    )
    with patch.object(driver.importlib, "import_module", return_value=module):
        actual, _ = driver._default_model_loader(mode, tmp_path, "cpu")
    assert actual is model
    model.requires_grad_.assert_called_once_with(False)
    model.eval.assert_called_once_with()
    model_class.from_pretrained.assert_called_once_with(
        str(tmp_path), local_files_only=True, trust_remote_code=False
    )


def test_legacy_loader_refuses_mixed_pickle_mount_or_pickle_shard(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    unsafe = tmp_path / "model.bin"
    unsafe.write_bytes(b"pickle fixture")
    with pytest.raises(driver.DriverError, match="exclude pickle"):
        driver._default_model_loader("abstractive", tmp_path, "cpu")
    unsafe.unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model.bin"}})
    )
    with pytest.raises(driver.DriverError, match="unsafe weights"):
        driver._default_model_loader("abstractive", tmp_path, "cpu")


def _is_git_checkout(root: Path) -> bool:
    if not root.is_dir():
        return False
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "--verify", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _fake_checkout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkout"
    sources = {
        "README.md": b"fixture only\n",
        "requirements.txt": b"",
        "run_extractive_compressor.py": (
            b"def get_contriever_scores(model, tokenizer, data_row, device, top_k):\n"
            b"    assert top_k == -1\n"
            b"    inputs = tokenizer([data_row['query']] + [d['text'] for d in "
            b"data_row['retrieved_docs']], padding=True, truncation=True, return_tensors='pt')\n"
            b"    model(**inputs)\n"
            b"    return model.scores\n"
        ),
        "train_hf_summarization_model.py": (
            b"PROMPT = 'Question: {}\\n Document: {}\\n Summary: '\n"
        ),
    }
    license_raw = b"MIT fixture attribution\n"
    _write(root / ".git" / "HEAD", (driver.UPSTREAM_COMMIT + "\n").encode())
    _write(root / "LICENSE", license_raw)
    for relative, raw in sources.items():
        _write(root / relative, raw)
    manifest = {
        "schema_version": driver.MANIFEST_SCHEMA,
        "evidence_level": "fake_checkout_plumbing",
        "upstream": {
            "repository": "fixture://recomp",
            "commit": driver.UPSTREAM_COMMIT,
            "license_spdx": "MIT",
            "license_sha256": _sha(license_raw),
            "require_clean_checkout": False,
        },
        "source_files": {name: _sha(raw) for name, raw in sources.items()},
        "runtime_contract": {
            "extractive_symbol": "run_extractive_compressor.get_contriever_scores",
            "extractive_top_k": -1,
            "candidate_unit_kind": driver.CANDIDATE_UNIT_KIND,
            "abstractive_input_template": driver.ABSTRACTIVE_INPUT_TEMPLATE,
            "abstractive_decode": "batch_decode_skip_special_tokens_strip",
        },
        "legacy_dependencies": {
            "required_versions": {},
            "extractive_unpinned_imports": [],
        },
        "artifact_policy": {
            "checkpoint_license": "NOASSERTION",
            "bundle_checkpoint": False,
            "bundle_training_data": False,
            "local_files_only": True,
            "trust_remote_code": False,
        },
    }
    manifest_path = tmp_path / "compatibility.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root, manifest_path


def _fake_checkpoint(tmp_path: Path, mode: str) -> tuple[Path, Path, str]:
    root = tmp_path / f"checkpoint-{mode}"
    files = {
        "config.json": b"{}\n",
        "model.safetensors": b"synthetic-not-a-model",
        "tokenizer.json": b"{}\n",
    }
    for relative, raw in files.items():
        _write(root / relative, raw)
    manifest = {
        "schema_version": driver.CHECKPOINT_SCHEMA,
        "mode": mode,
        "checkpoint_revision": f"{mode}-revision",
        "tokenizer_revision": "tokenizer-revision",
        "license": "NOASSERTION",
        "files": {name: _sha(raw) for name, raw in files.items()},
    }
    manifest_path = tmp_path / f"checkpoint-{mode}.json"
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(raw)
    return root, manifest_path, _sha(raw)


class _Mask:
    def __init__(self, count: int) -> None:
        self.count = count

    def sum(self) -> int:
        return self.count


class _Encoding(dict[str, Any]):
    def to(self, device: str) -> _Encoding:
        assert device == "cpu"
        return self


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, raw_generation: str = " compressed evidence ") -> None:
        self.raw_generation = raw_generation
        self.prompts: list[Any] = []

    def __call__(self, value: Any, **kwargs: Any) -> _Encoding:
        self.prompts.append((value, kwargs))
        return _Encoding(input_ids=[[1, 2, 3]], attention_mask=_Mask(3))

    def batch_decode(self, outputs: Any, **kwargs: Any) -> list[str]:
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": True,
        }
        return [self.raw_generation]


class _Model:
    def __init__(self, *, scores: list[float] | None = None) -> None:
        self.scores = scores or []
        self.generate_kwargs: dict[str, Any] = {}
        self.forward_calls = 0

    def __call__(self, **kwargs: Any) -> object:
        assert "input_ids" in kwargs
        self.forward_calls += 1
        return object()

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_kwargs = kwargs
        return [[5, 6, 0]]


class _Retriever:
    def fingerprint(self) -> dict[str, str]:
        return {"retriever_id": "frozen-fixture", "index_sha256": "e" * 64}


class _InProcessWorker:
    worker_id = "recomp-fake-checkout-e2e"

    def __init__(self, runtime: driver.RECOMPDriver) -> None:
        self.runtime = runtime
        self.last_request: Any = None
        self.requests: list[Any] = []

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(payload)

    def call_controlled(self, request: Any) -> dict[str, Any]:
        self.last_request = request
        self.requests.append(request)
        return dict(self.runtime.call(request.as_dict()))

    def fingerprint(self) -> dict[str, Any]:
        return dict(self.runtime.fingerprint())


def _state(*, budget: ExecutionBudget | None = None) -> FrozenAgentState:
    return FrozenAgentState(
        query="Which sentence?",
        task_id="recomp-e2e",
        context=(
            ContextBlock("first candidate", block_id="s-1", source="fixed"),
            ContextBlock("second candidate", block_id="s-2", source="fixed"),
            ContextBlock("third candidate", block_id="s-3", source="fixed"),
        ),
        budget=budget or ExecutionBudget(),
    )


def _runtime(
    tmp_path: Path,
    mode: str,
    *,
    model: _Model,
    tokenizer: _Tokenizer,
) -> tuple[_InProcessWorker, str]:
    checkout, compat = _fake_checkout(tmp_path)
    checkpoint, checkpoint_manifest, asset_hash = _fake_checkpoint(tmp_path, mode)
    runtime = driver.RECOMPDriver(
        checkout_path=checkout,
        checkpoint_path=checkpoint,
        checkpoint_manifest_path=checkpoint_manifest,
        manifest_path=compat,
        allow_fixture_manifest=True,
        model_loader=lambda actual_mode, actual_root, device: (model, tokenizer),
        checkpoint_license_acknowledged=True,
    )
    return _InProcessWorker(runtime), asset_hash


def test_extractive_fake_checkout_e2e_binds_inputs_usage_and_paper_top_n(
    tmp_path: Path,
) -> None:
    model = _Model(scores=[0.2, 0.9, 0.8])
    tokenizer = _Tokenizer()
    worker, asset_hash = _runtime(tmp_path, "extractive", model=model, tokenizer=tokenizer)
    adapter = RECOMPExtractiveExternal(
        client=worker,
        checkpoint_revision="extractive-revision",
        tokenizer_revision="tokenizer-revision",
        sentence_splitter_revision="upstream-presegmented-v1",
        asset_manifest_hash=asset_hash,
        asset_license_acknowledged=True,
        controls=RECOMPExtractiveControls.paper_native("hotpotqa"),
    )

    patch = adapter.propose(_state(), InterventionContext(retriever=_Retriever()))

    assert [block.text for block in patch.operations[0].value] == [
        "second candidate",
        "third candidate",
    ]
    assert patch.profile_kind == "paper_native_adapter"
    assert patch.metadata["execution_evidence"] == "fake_checkout_plumbing"
    assert patch.trace.counters["model_calls"] == 1
    assert patch.trace.counters["retrieval_calls"] == 0
    assert patch.trace.counters["input_tokens"] == 3
    assert patch.trace.counters["worker_calls"] == 2
    assert [request.payload["phase"] for request in worker.requests] == ["preflight", "execute"]
    assert model.forward_calls == 1
    assert tokenizer.prompts
    assert not any((tmp_path / "checkout").rglob("__pycache__"))

    bad = worker.last_request.as_dict()
    bad["payload"]["checkpoint_path"] = "/tmp/untrusted"
    with pytest.raises(driver.DriverError) as caught:
        driver.parse_request(bad)
    assert caught.value.code == "INVALID_REQUEST"

    bad = worker.last_request.as_dict()
    bad["payload"]["input_artifact_hash"] = "f" * 64
    bad["request_id"] = driver._content_hash(
        {key: value for key, value in bad.items() if key != "request_id"}
    )
    with pytest.raises(driver.DriverError) as caught:
        driver.parse_request(bad)
    assert caught.value.code == "SOURCE_MAPPING_MISMATCH"


@pytest.mark.parametrize(("dataset", "expected"), [("nq", 1), ("tqa", 1), ("hotpotqa", 2)])
def test_paper_native_top_n_is_frozen(dataset: str, expected: int) -> None:
    controls = RECOMPExtractiveControls.paper_native(dataset)
    assert controls.max_sentences == expected
    assert controls.output_order == "score_descending"
    assert controls.minimum_score is None


@pytest.mark.parametrize("raw_generation", [" summary ", "   "])
def test_abstractive_fake_checkout_uses_official_prompt_and_preserves_empty(
    tmp_path: Path, raw_generation: str
) -> None:
    model = _Model()
    tokenizer = _Tokenizer(raw_generation)
    worker, asset_hash = _runtime(tmp_path, "abstractive", model=model, tokenizer=tokenizer)
    adapter = RECOMPAbstractiveExternal(
        client=worker,
        checkpoint_revision="abstractive-revision",
        tokenizer_revision="tokenizer-revision",
        input_template_hash=RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
        asset_manifest_hash=asset_hash,
        asset_license_acknowledged=True,
        controls=RECOMPAbstractiveControls(),
    )

    patch = adapter.propose(_state(), InterventionContext(retriever=_Retriever()))

    expected_prompt = (
        "Question: Which sentence?\n Document: first candidatesecond candidatethird candidate"
        "\n Summary: "
    )
    assert tokenizer.prompts[0][0] == [expected_prompt]
    assert model.generate_kwargs["max_length"] == 512
    assert patch.apply(_state()).context == (
        () if not raw_generation.strip() else patch.operations[0].value
    )
    assert patch.metadata["execution_evidence"] == "fake_checkout_plumbing"


def test_score_length_mismatch_fails_with_typed_consumed_error(tmp_path: Path) -> None:
    model = _Model(scores=[0.2])
    worker, asset_hash = _runtime(tmp_path, "extractive", model=model, tokenizer=_Tokenizer())
    adapter = RECOMPExtractiveExternal(
        client=worker,
        checkpoint_revision="extractive-revision",
        tokenizer_revision="tokenizer-revision",
        sentence_splitter_revision="fixed-v1",
        asset_manifest_hash=asset_hash,
        asset_license_acknowledged=True,
    )
    with pytest.raises(driver.DriverError) as caught:
        adapter.propose(_state(), InterventionContext(retriever=_Retriever()))
    assert caught.value.code == "SCORE_LENGTH_MISMATCH"
    assert caught.value.consumed is True
    assert caught.value.usage == {
        "model_calls": 1,
        "retrieval_calls": 0,
        "unpriced_model_calls": 1,
        "input_tokens": 3,
        "output_tokens": 0,
    }


def test_checkpoint_manifest_rejects_symlink_and_wrong_license(tmp_path: Path) -> None:
    root, manifest_path, _ = _fake_checkpoint(tmp_path, "extractive")
    manifest = json.loads(manifest_path.read_text())
    manifest["license"] = "MIT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(driver.DriverError, match="NOASSERTION"):
        driver.verify_checkpoint(root, manifest_path)

    root, manifest_path, _ = _fake_checkpoint(tmp_path / "symlink", "extractive")
    weight = root / "model.safetensors"
    weight.unlink()
    weight.symlink_to(root / "config.json")
    with pytest.raises(driver.DriverError, match="symlink"):
        driver.verify_checkpoint(root, manifest_path)


@pytest.mark.skipif(
    not _is_git_checkout(REAL_CHECKOUT),
    reason="opt-in real checkout is absent",
)
def test_real_checkout_cli_smoke_is_honestly_typed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(driver.__file__).resolve()),
            "--checkout",
            str(REAL_CHECKOUT),
            "--smoke-import",
            "extractive",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] in {"ok", "error"}
    if payload["status"] == "ok":
        assert payload["execution_evidence"] == "pinned_upstream_source"
    else:
        assert payload["failure_code"] in {
            "LEGACY_DEPENDENCY_MISSING",
            "LEGACY_DEPENDENCY_VERSION_MISMATCH",
        }
