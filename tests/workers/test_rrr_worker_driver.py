from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.algorithms.reformulate.rrr import RRRPromptControls, RRRPromptExternal
from agentinputlab.contracts import FrozenAgentState, InterventionContext
from workers.rrr import driver

REAL_TARBALL = Path(os.environ.get("AGENTINPUTLAB_RRR_TARBALL", "/private/tmp/rrr-6bab6243.tar.gz"))
REAL_CHECKOUT = os.environ.get("AGENTINPUTLAB_RRR_CHECKOUT")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fake_source(tmp_path: Path) -> tuple[Path, Path]:
    sources = {
        "README.md": b"fixture source without a license\n",
        "generate/inference.py": b"def add_prompt(): pass\ndef run_main(): pass\ndef run_searchre(): pass\n",
        "generate/main-0514.py": b"def rewrite(): pass\ndef rewrite2(): pass\n",
        "generate/bingenv.py": b"class BingEnv: pass\n",
        "generate/bing.py": b"api_key = 'fixture-sensitive-value-never-echo'\n",
        "rl/RL4LMs/rl4lms/envs/text_generation/warm_start.py": (
            b"class ActorOnlyWarmStartMixin: pass\n"
        ),
    }
    root = f"RAG-query-rewriting-{driver.UPSTREAM_COMMIT}"
    tar_path = tmp_path / "source.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for relative, raw in sources.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    contract = {
        "stage_order": ["rewrite", "retrieve", "snippet", "reader"],
        "query_separator": ";",
        "generation_terminator": "**",
        "search_wrapper": "search[{query}]",
        "snippet_joiner": " ",
        "supported_rewriter_profiles": [
            "prompt",
            "warmup_checkpoint",
            "ppo_checkpoint",
        ],
        "supported_search_modes": ["bing_cache", "controlled_static"],
        "imports_upstream_runtime": False,
        "trains_ppo": False,
    }
    manifest = {
        "schema_version": driver.MANIFEST_SCHEMA,
        "evidence_level": "fake_checkout_plumbing",
        "upstream": {
            "repository": "fixture://rrr",
            "commit": driver.UPSTREAM_COMMIT,
            "archive_root": root,
            "tarball_sha256": _sha(tar_path.read_bytes()),
            "license_spdx": "NOASSERTION",
            "license_file_present": False,
        },
        "source_files": {name: _sha(raw) for name, raw in sources.items()},
        "required_symbols": {
            "generate/inference.py": ["add_prompt", "run_main", "run_searchre"],
            "generate/main-0514.py": ["rewrite", "rewrite2"],
            "generate/bingenv.py": ["BingEnv"],
            "rl/RL4LMs/rl4lms/envs/text_generation/warm_start.py": ["ActorOnlyWarmStartMixin"],
        },
        "credential_risk_paths": ["generate/bing.py"],
        "runtime_contract": contract,
        "artifact_policy": {
            "checkpoint_license": "NOASSERTION",
            "bundle_source": False,
            "bundle_checkpoints": False,
            "bundle_search_cache": False,
            "local_files_only": True,
            "trust_remote_code": False,
        },
    }
    manifest_path = tmp_path / "compatibility.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return tar_path, manifest_path


class _Generator:
    def __init__(self, text: str, *, role: str, profile: str, revision: str) -> None:
        self.text = text
        self.identity = {"role": role, "profile": profile, "revision": revision}
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> driver.Generation:
        self.calls.append({"prompt": prompt, "seed": seed, "max_tokens": max_tokens})
        return driver.Generation(self.text, input_tokens=7, output_tokens=3)

    def fingerprint(self) -> dict[str, str]:
        return dict(self.identity)


class _BackendIdentity:
    def __init__(self, rewriter: _Generator, reader: _Generator) -> None:
        self.rewriter = rewriter
        self.reader = reader

    def fingerprint(self) -> dict[str, Any]:
        return {
            "rewriter": self.rewriter.fingerprint(),
            "reader": self.reader.fingerprint(),
        }


class _RuntimeWorker:
    worker_id = "rrr-fake-e2e"

    def __init__(self, runtime: driver.RRRRuntime) -> None:
        self.runtime = runtime
        self.last_request: Any = None

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(payload)

    def call_controlled(self, request: Any) -> dict[str, Any]:
        self.last_request = request
        return dict(self.runtime.call(request.as_dict()))

    def fingerprint(self) -> dict[str, Any]:
        return dict(self.runtime.fingerprint())


def _runtime(tmp_path: Path, *, mode: str = "bing_cache") -> tuple[Any, ...]:
    tarball, manifest = _fake_source(tmp_path)
    source = driver.verify_source_tarball(
        tarball, manifest_path=manifest, allow_fixture_manifest=True
    )
    rewrite_prompt = "Rewrite for search, end with '**': {query}"
    reader_prompt = "Passage: {background}\nQuestion: {query}\nAnswer:"
    prompt_path = tmp_path / "rewrite-prompt.txt"
    reader_prompt_path = tmp_path / "reader-prompt.txt"
    prompt_path.write_text(rewrite_prompt, encoding="utf-8")
    reader_prompt_path.write_text(reader_prompt, encoding="utf-8")
    search = {
        "schema_version": driver.SEARCH_SCHEMA,
        "mode": mode,
        "corpus_revision": "corpus-v1",
        "index_revision": "index-v1",
        "window_start": "2023-01-01",
        "window_end": "2023-12-31",
        "entries": {
            "better search query": [
                {
                    "document_id": "d1",
                    "title": "Dune",
                    "text": "Frank Herbert wrote Dune.",
                    "score": 0.9,
                    "metadata": {"cache_row": 1},
                }
            ]
        },
    }
    search_path = tmp_path / "search.json"
    search_path.write_text(json.dumps(search, sort_keys=True), encoding="utf-8")
    searcher = driver.SearchCache(search_path)
    rewriter = _Generator(
        "better search query**", role="rewriter", profile="prompt", revision="prompt-v1"
    )
    reader = _Generator(
        "Frank Herbert**", role="reader", profile="frozen_reader", revision="reader-v1"
    )
    runtime = driver.RRRRuntime(
        source=source,
        prompt_path=prompt_path,
        reader_prompt_path=reader_prompt_path,
        rewriter=rewriter,
        searcher=searcher,
        reader=reader,
        acknowledge_noassertion_source=True,
        acknowledge_noassertion_checkpoints=True,
        acknowledge_credential_quarantine=True,
    )
    return runtime, rewriter, reader, searcher, rewrite_prompt, reader_prompt


def test_fake_tarball_full_chain_binds_trajectory_fairness_and_usage(tmp_path: Path) -> None:
    runtime, rewriter, reader, searcher, rewrite_prompt, reader_prompt = _runtime(tmp_path)
    worker = _RuntimeWorker(runtime)
    adapter = RRRPromptExternal(
        client=worker,
        source_license_acknowledged=True,
        checkpoint_license_acknowledged=True,
        prompt_hash=_sha(rewrite_prompt.encode()),
        reader_prompt_hash=_sha(reader_prompt.encode()),
        search_artifact_hash=searcher.digest,
        corpus_revision="corpus-v1",
        index_revision="index-v1",
        rewriter_checkpoint_revision="prompt-v1",
        reader_checkpoint_revision="reader-v1",
        controls=RRRPromptControls(
            top_k=1,
            search_window_start="2023-01-01",
            search_window_end="2023-12-31",
        ),
    )
    state = FrozenAgentState(query="Dune author?", task_id="rrr-e2e")

    patch = adapter.propose(
        state,
        InterventionContext(backend=_BackendIdentity(rewriter, reader), seed=4),
    )

    updated = patch.apply(state)
    assert updated.query == "better search query"
    assert [block.block_id for block in updated.context] == ["d1"]
    assert patch.profile_kind == "controlled_adapter"
    assert patch.metadata["search_profile"] == "bing_cache"
    assert patch.trace.counters["model_calls"] == 2
    assert patch.trace.counters["retrieval_calls"] == 1
    assert patch.metadata["execution_evidence"] == "fake_checkout_plumbing"
    assert reader.calls[0]["prompt"] == (
        "Passage: Frank Herbert wrote Dune.\nQuestion: Dune author?\nAnswer:"
    )

    bad = worker.last_request.as_dict()
    bad["payload"]["source_path"] = "/tmp/untrusted"
    bad["request_id"] = driver._content_hash(
        {key: value for key, value in bad.items() if key != "request_id"}
    )
    with pytest.raises(driver.DriverError) as caught:
        driver._validate_request(bad)
    assert caught.value.code == "INVALID_REQUEST"


def test_controlled_retriever_is_separately_labelled(tmp_path: Path) -> None:
    runtime, rewriter, reader, searcher, rewrite_prompt, reader_prompt = _runtime(
        tmp_path, mode="controlled_static"
    )
    worker = _RuntimeWorker(runtime)

    class Retriever:
        def fingerprint(self) -> dict[str, Any]:
            return dict(searcher.fingerprint())

    patch = RRRPromptExternal(
        client=worker,
        source_license_acknowledged=True,
        checkpoint_license_acknowledged=True,
        prompt_hash=_sha(rewrite_prompt.encode()),
        reader_prompt_hash=_sha(reader_prompt.encode()),
        search_artifact_hash=searcher.digest,
        corpus_revision="corpus-v1",
        index_revision="index-v1",
        rewriter_checkpoint_revision="prompt-v1",
        reader_checkpoint_revision="reader-v1",
        controls=RRRPromptControls(
            top_k=1,
            search_mode="controlled_static",
            search_window_start="2023-01-01",
            search_window_end="2023-12-31",
        ),
    ).propose(
        FrozenAgentState(query="Dune author?"),
        InterventionContext(backend=_BackendIdentity(rewriter, reader), retriever=Retriever()),
    )
    assert patch.profile_kind == "controlled_adapter"


def _checkout_fixture(tmp_path, monkeypatch):
    tarball, manifest = _fake_source(tmp_path)
    root = tmp_path / "checkout"
    root.mkdir()
    entries = []
    with tarfile.open(tarball, "r:gz") as archive:
        for member in archive:
            name = member.name.split("/", 1)[1]
            raw = archive.extractfile(member).read()
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            entries.append(f"100644 blob {blob}\t{name}")
    # This deliberately absent data blob must not be read by executable-source
    # verification; real checkout metadata is tested separately when mounted.
    entries.append("100644 blob " + "0"*40 + "\tdatasets/large-unneeded.bin")

    def metadata(path, *arguments):
        assert path == root
        if arguments == ("rev-parse", "HEAD"):
            return driver.UPSTREAM_COMMIT + "\n"
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root) + "\n"
        assert arguments == ("ls-tree", "-r", "--full-tree", "-z", "HEAD")
        return "\0".join(entries) + "\0"

    monkeypatch.setattr(driver, "_git_read", metadata)
    return root, manifest, entries


def test_pinned_checkout_contract_checks_source_without_reading_bundled_data(tmp_path, monkeypatch):
    root, manifest, _ = _checkout_fixture(tmp_path, monkeypatch)
    source = driver.verify_source_checkout(root, manifest_path=manifest, allow_fixture_manifest=True)
    assert source.evidence_level == "fake_checkout_plumbing" and source.credential_risk_count == 1
    assert not (root / "datasets/large-unneeded.bin").exists()
    with pytest.raises(driver.DriverError, match="manifest hash"):
        driver.verify_source_checkout(root, manifest_path=manifest)


def test_pinned_checkout_changed_source_link_and_inventory_are_rejected(tmp_path, monkeypatch):
    root, manifest, entries = _checkout_fixture(tmp_path, monkeypatch)
    path = root / "generate/inference.py"
    raw = path.read_bytes()
    path.write_bytes(raw + b"\n# modified")
    with pytest.raises(driver.DriverError, match="Git blob"):
        driver.verify_source_checkout(root, manifest_path=manifest, allow_fixture_manifest=True)
    path.write_bytes(raw)
    entries.append(entries[0])
    with pytest.raises(driver.DriverError, match="link or duplicate"):
        driver.verify_source_checkout(root, manifest_path=manifest, allow_fixture_manifest=True)
    entries.pop()
    path.unlink()
    path.symlink_to(root / "README.md")
    with pytest.raises(driver.DriverError, match="missing, linked"):
        driver.verify_source_checkout(root, manifest_path=manifest, allow_fixture_manifest=True)
    link = tmp_path / "root-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(driver.DriverError, match="regular directory"):
        driver.verify_source_checkout(link, manifest_path=manifest, allow_fixture_manifest=True)


def test_failed_rewrite_parser_reports_the_already_consumed_generation(tmp_path):
    runtime, rewriter, _, _, _, _ = _runtime(tmp_path)
    rewriter.text = "first query; second query**"
    # Start from a valid source-bound request and alter only the fixture model
    # response; no downstream reader or search is executed after parser failure.
    worker = _RuntimeWorker(runtime)
    method = RRRPromptExternal(client=worker, source_license_acknowledged=True,
        checkpoint_license_acknowledged=True, prompt_hash=_sha(runtime.prompt.encode()),
        reader_prompt_hash=_sha(runtime.reader_prompt.encode()), search_artifact_hash=runtime.searcher.digest,
        corpus_revision="corpus-v1", index_revision="index-v1", rewriter_checkpoint_revision="prompt-v1",
        reader_checkpoint_revision="reader-v1", controls=RRRPromptControls(top_k=1,
            search_window_start="2023-01-01", search_window_end="2023-12-31"))
    with pytest.raises(driver.DriverError) as caught:
        method.propose(FrozenAgentState(query="Dune author?"),
            InterventionContext(backend=_BackendIdentity(runtime.rewriter, runtime.reader)))
    assert caught.value.consumed is True
    assert caught.value.usage == dict(model_calls=1, retrieval_calls=0, input_tokens=7, output_tokens=3)
    assert len(runtime.rewriter.calls) == 1 and runtime.reader.calls == []


def test_character_budget_fails_before_search_and_reader(tmp_path: Path) -> None:
    runtime, rewriter, reader, searcher, rewrite_prompt, reader_prompt = _runtime(tmp_path)
    adapter = RRRPromptExternal(
        client=_RuntimeWorker(runtime),
        source_license_acknowledged=True,
        checkpoint_license_acknowledged=True,
        prompt_hash=_sha(rewrite_prompt.encode()),
        reader_prompt_hash=_sha(reader_prompt.encode()),
        search_artifact_hash=searcher.digest,
        corpus_revision="corpus-v1",
        index_revision="index-v1",
        rewriter_checkpoint_revision="prompt-v1",
        reader_checkpoint_revision="reader-v1",
        controls=RRRPromptControls(
            top_k=1,
            search_window_start="2023-01-01",
            search_window_end="2023-12-31",
            max_input_chars=5,
        ),
    )
    with pytest.raises(driver.DriverError) as caught:
        adapter.propose(
            FrozenAgentState(query="Dune author?"),
            InterventionContext(backend=_BackendIdentity(rewriter, reader)),
        )
    assert caught.value.code == "BUDGET_EXCEEDED"
    assert rewriter.calls == []
    assert reader.calls == []


def test_double_license_and_credential_quarantine_gates(tmp_path: Path) -> None:
    tarball, manifest = _fake_source(tmp_path)
    source = driver.verify_source_tarball(
        tarball, manifest_path=manifest, allow_fixture_manifest=True
    )
    prompt = tmp_path / "p.txt"
    reader_prompt = tmp_path / "r.txt"
    prompt.write_text("{query}")
    reader_prompt.write_text("{background}{query}")
    generator = _Generator("x**", role="rewriter", profile="prompt", revision="v1")

    with pytest.raises(driver.DriverError, match="quarantined") as caught:
        driver.RRRRuntime(
            source,
            prompt,
            reader_prompt,
            generator,
            object(),
            generator,
            True,
            True,
            False,
        )
    assert "fixture-sensitive-value-never-echo" not in str(caught.value)


def test_checkpoint_manifest_allows_warmup_and_ppo_inference_but_no_training(
    tmp_path: Path,
) -> None:
    for profile in ("warmup_checkpoint", "ppo_checkpoint"):
        root = tmp_path / profile
        root.mkdir()
        files = {"config.json": b"{}", "model.safetensors": b"fixture"}
        for name, raw in files.items():
            (root / name).write_bytes(raw)
        manifest = {
            "schema_version": driver.CHECKPOINT_SCHEMA,
            "role": "rewriter",
            "profile": profile,
            "revision": f"{profile}-v1",
            "model_type": "seq2seq",
            "license": "NOASSERTION",
            "files": {name: _sha(raw) for name, raw in files.items()},
        }
        path = tmp_path / f"{profile}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        verified = driver.verify_checkpoint(root, path)
        assert verified["profile"] == profile
    assert driver._parser().parse_known_args(["--source-tarball", "x", "--train"])[1] == ["--train"]


@pytest.mark.parametrize(
    ("model_type", "tokens", "decoder_tokens", "expected_count", "expected_decoded"),
    [
        ("seq2seq", [0, 11, 12, 1], None, 3, [0, 11, 12, 1]),
        ("seq2seq", [0], None, 0, [0]),
        ("seq2seq", [0, 9, 11, 1], 2, 2, [0, 9, 11, 1]),
        ("causal", [7, 8, 9, 10, 11, 1], None, 2, [11, 1]),
    ],
)
def test_local_generation_counts_only_generated_tokens_without_changing_decode(
    monkeypatch: pytest.MonkeyPatch, model_type: str, tokens: list[int],
    decoder_tokens: int | None, expected_count: int, expected_decoded: list[int],
) -> None:
    class Row(list):
        @property
        def shape(self):
            return (len(self),)

        def __getitem__(self, index):
            value = super().__getitem__(index)
            return Row(value) if isinstance(index, slice) else value

    class Tokenizer:
        decoded = None

        def __call__(self, prompt, **kwargs):
            assert prompt == "native prompt" and kwargs == {"return_tensors": "pt", "truncation": True}
            encoded = {"input_ids": SimpleNamespace(shape=(1, 4))}
            if decoder_tokens is not None:
                encoded["decoder_input_ids"] = SimpleNamespace(shape=(1, decoder_tokens))
            return encoded

        def decode(self, row, *, skip_special_tokens):
            assert skip_special_tokens
            self.decoded = list(row)
            return "native decoded text"

    class Model:
        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 3 and kwargs["do_sample"] is False
            return [Row(tokens)]

    tokenizer = Tokenizer()
    factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model())
    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoModelForSeq2SeqLM=factory, AutoModelForCausalLM=factory,
    )
    monkeypatch.setattr(driver.importlib, "import_module", lambda name: module)
    runtime = driver.LocalHFGenerator({"root": "/synthetic-checkpoint", "model_type": model_type})
    result = runtime.generate("native prompt", seed=17, max_tokens=3)
    assert (result.text, result.input_tokens, result.output_tokens) == (
        "native decoded text", 4, expected_count)
    assert tokenizer.decoded == expected_decoded


@pytest.mark.skipif(not REAL_CHECKOUT and not REAL_TARBALL.is_file(),
                    reason="mount a pinned RRR checkout or tarball for source-only acceptance")
def test_real_source_symbol_smoke_redacts_credentials() -> None:
    # Both are supported production source modes. An explicitly supplied bad
    # checkout must fail, not silently skip or fall back to another source.
    source = ["--source-checkout", REAL_CHECKOUT] if REAL_CHECKOUT else ["--source-tarball", str(REAL_TARBALL)]
    result = subprocess.run(
        [
            sys.executable,
            str(Path(driver.__file__).resolve()),
            *source,
            "--smoke-source",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["credential_values_redacted"] is True
    assert payload["credential_risk_count"] == 7
    assert payload["execution_evidence"] == (
        "pinned_checkout_compatible_chain" if REAL_CHECKOUT else "pinned_tarball_compatible_chain")
    assert payload["upstream_imported"] is False
    assert "api_key" not in result.stdout.lower()
