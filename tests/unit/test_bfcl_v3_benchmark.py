from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

import agentinputlab.benchmarks.bfcl_v3 as bfcl_v3
from agentinputlab.benchmarks import (
    BFCL_V3_OFFICIAL_SOURCE_REVISION,
    BFCL_V3_SELECTED_SOURCE_SHA256,
    BFCL_V3_SOURCE_ARCHIVE_SHA256,
    BenchmarkSmokeAsset,
    BFCLV3EvaluationResult,
    CallerAssertedBFCLV3ScorerBridge,
    PreparedBenchmark,
    PreparedBFCLV3SimpleAdapter,
    prepare_smoke_asset,
)
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json

_SOURCE_ROOT = "berkeley-function-call-leaderboard"
_PROMPT_PATH = f"{_SOURCE_ROOT}/bfcl_eval/data/BFCL_v3_simple.json"
_GOLD_PATH = f"{_SOURCE_ROOT}/bfcl_eval/data/possible_answer/BFCL_v3_simple.json"
_SELECTED_PLACEHOLDERS = (
    f"{_SOURCE_ROOT}/bfcl_eval/constants/category_mapping.py",
    f"{_SOURCE_ROOT}/bfcl_eval/eval_checker/ast_eval/ast_checker.py",
    f"{_SOURCE_ROOT}/bfcl_eval/eval_checker/eval_runner.py",
    f"{_SOURCE_ROOT}/bfcl_eval/model_handler/base_handler.py",
)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode()


def _rows(count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index in range(count):
        example_id = f"simple_{index}"
        function_name = f"fixture_{index}"
        prompts.append(
            {
                "id": example_id,
                "question": [[{"role": "user", "content": f"Call {function_name} with {index}."}]],
                "function": [
                    {
                        "name": function_name,
                        "description": "Return the provided integer.",
                        "parameters": {
                            "type": "dict",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                        },
                    }
                ],
            }
        )
        answers.append(
            {
                "id": example_id,
                "ground_truth": [{function_name: {"value": [index]}}],
            }
        )
    return prompts, answers


def _bfcl_tar(count: int, *, unsafe: bool = False) -> bytes:
    prompts, answers = _rows(count)
    members = {
        "LICENSE": (
            b"Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n"
        ),
        f"{_SOURCE_ROOT}/pyproject.toml": (
            b'[project]\nname = "bfcl_eval"\nlicense = { "text" = "Apache 2.0" }\n'
        ),
        _PROMPT_PATH: _jsonl(prompts),
        _GOLD_PATH: _jsonl(answers),
        **{name: b"# selected scorer source fixture\n" for name in _SELECTED_PLACEHOLDERS},
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for relative, payload in sorted(members.items()):
            info = tarfile.TarInfo(f"gorilla-1.3/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if unsafe:
            payload = b"escape"
            info = tarfile.TarInfo("gorilla-1.3/../escape")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _asset(content: bytes, *, count: int) -> BenchmarkSmokeAsset:
    return BenchmarkSmokeAsset(
        asset_id="fixture.bfcl.v3.simple",
        cell_id="bfcl.v3.simple_python",
        source_url="https://example.com/gorilla-v1.3.tar.gz",
        source_revision=BFCL_V3_OFFICIAL_SOURCE_REVISION,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="Apache-2.0",
        redistribution="not_redistributed",
        status="verified_smoke",
        processor="bfcl_v3_simple_tar_v1",
        expected_counts={
            "audit": 1,
            "functions": count,
            "gold": count,
            "ground_truth_candidates": count,
            "inputs": count,
            "source_files": 8,
        },
        prepared_sha256={},
        allowed_hosts=("example.com",),
        provenance={"fixture": "unit-test"},
    )


def _official_shape_prepared(root: Path) -> PreparedBenchmark:
    prompts, answers = _rows(400)
    inputs = []
    gold = []
    for prompt, answer in zip(prompts, answers, strict=True):
        function = prompt["function"][0]
        inputs.append(
            {
                "id": prompt["id"],
                "query": prompt["question"][0][0]["content"],
                "instruction": (
                    "Return the single Python function call requested by the user, "
                    "using the provided function schema."
                ),
                "visible_metadata": {
                    "benchmark_cell": "bfcl.v3.simple_python",
                    "bfcl_category": "simple",
                    "function": function,
                    "source_split": "test",
                },
                "visible_tool_names": [function["name"]],
            }
        )
        gold.append(answer)
    audit = [
        {
            "id": "bfcl-v3-simple-source-audit",
            "category": "simple",
            "evaluation_boundary": (
                "The official possible answers remain evaluator-only. AgentInputLab "
                "does not execute BFCL or claim an official score during preprocessing."
            ),
            "license_id": "Apache-2.0",
            "official_cli": "bfcl evaluate --test-category simple",
            "selected_source_sha256": dict(BFCL_V3_SELECTED_SOURCE_SHA256),
            "source_revision": BFCL_V3_OFFICIAL_SOURCE_REVISION,
            "source_tag": "v1.3",
        }
    ]
    payloads = {
        "inputs.jsonl": _jsonl(inputs),
        "gold.jsonl": _jsonl(gold),
        "audit.jsonl": _jsonl(audit),
    }
    root.mkdir()
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return PreparedBenchmark(
        asset_id="bfcl.v3.simple_python.v1_3",
        cell_id="bfcl.v3.simple_python",
        processor="bfcl_v3_simple_tar_v1",
        processor_revision="1",
        source_revision=BFCL_V3_OFFICIAL_SOURCE_REVISION,
        source_sha256=BFCL_V3_SOURCE_ARCHIVE_SHA256,
        split="test",
        root=root,
        counts={
            "audit": 1,
            "functions": 400,
            "gold": 400,
            "ground_truth_candidates": 400,
            "inputs": 400,
            "source_files": 8,
        },
        file_sha256={
            name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
        },
    )


def _fixture_adapter(
    prepared: PreparedBenchmark,
    bridge: CallerAssertedBFCLV3ScorerBridge,
) -> PreparedBFCLV3SimpleAdapter:
    return bfcl_v3._construct_bfcl_v3_simple_adapter(
        prepared,
        evaluation_bridge=bridge,
        _construction_token=bfcl_v3._LOADER_CONSTRUCTION_TOKEN,
    )


def test_bfcl_preprocessing_is_deterministic_and_gold_isolated() -> None:
    content = _bfcl_tar(2)
    asset = _asset(content, count=2)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.tar.gz"
        source.write_bytes(content)
        first = prepare_smoke_asset(asset, source, root / "prepared-a")
        second = prepare_smoke_asset(asset, source, root / "prepared-b")

        assert first.split == "test"
        assert dict(first.file_sha256) == dict(second.file_sha256)
        visible = first.inputs_path.read_text(encoding="utf-8")
        evaluator = first.gold_path.read_text(encoding="utf-8")
        assert "ground_truth" not in visible
        assert "ground_truth" in evaluator
        assert "selected_source_sha256" in first.audit_path.read_text(encoding="utf-8")


def test_bfcl_preprocessing_rejects_unsafe_archive_paths() -> None:
    content = _bfcl_tar(1, unsafe=True)
    asset = _asset(content, count=1)
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.tar.gz"
        source.write_bytes(content)
        with pytest.raises(ArtifactError, match="unsafe path"):
            prepare_smoke_asset(asset, source, Path(directory) / "prepared")


def test_bfcl_adapter_hides_gold_and_requires_typed_caller_asserted_bridge() -> None:
    received: list[tuple[dict[str, Any], Any]] = []

    def callback(payload: dict[str, Any], prediction: Any) -> BFCLV3EvaluationResult:
        received.append((payload, prediction))
        return BFCLV3EvaluationResult(ast_accuracy=1)

    bridge = CallerAssertedBFCLV3ScorerBridge(
        bridge_id="fixture-bfcl-v1.3",
        bridge_revision="1" * 64,
        model_handler_id="fixture.model_handler:decode_ast",
        model_handler_revision="2" * 64,
        callback=callback,
    )
    with tempfile.TemporaryDirectory() as directory:
        prepared = _official_shape_prepared(Path(directory) / "prepared")
        with pytest.raises(ValidationError, match="load_bfcl_v3_simple_adapter"):
            PreparedBFCLV3SimpleAdapter(prepared, evaluation_bridge=bridge)
        with pytest.raises(ValidationError, match="invalid BFCL v3 loader construction token"):
            bfcl_v3._construct_bfcl_v3_simple_adapter(
                prepared,
                evaluation_bridge=bridge,
                _construction_token=object(),
            )
        adapter = _fixture_adapter(prepared, bridge)
        example = adapter.example("test", "simple_0")
        visible = canonical_json(example.visible_state())

        assert "ground_truth" not in visible
        assert adapter.score(example, "fixture_0(value=0)") == {"ast_accuracy": 1.0}
        assert received[0][0]["ground_truth"][0]["fixture_0"]["value"] == (0,)
        assert received[0][0]["test_category"] == "simple"
        assert adapter.evaluation_provenance["provenance_status"] == "caller_asserted"
        assert adapter.evaluation_provenance["official_runtime_verified"] is False

        prepared.inputs_path.write_text("tampered\n", encoding="utf-8")
        with pytest.raises(ArtifactError, match="hash mismatch"):
            adapter.examples("test")


def test_bfcl_result_rejects_fractional_or_bare_bridge_scores() -> None:
    with pytest.raises(ValidationError, match="exactly 0 or 1"):
        BFCLV3EvaluationResult(ast_accuracy=0.5)

    bridge = CallerAssertedBFCLV3ScorerBridge(
        bridge_id="fixture-bfcl-v1.3",
        bridge_revision="1" * 64,
        model_handler_id="fixture.model_handler:decode_ast",
        model_handler_revision="2" * 64,
        callback=lambda _payload, _prediction: {"ast_accuracy": 1.0},  # type: ignore[arg-type]
    )
    with pytest.raises(ValidationError, match="bare metrics"):
        bridge.evaluate({}, "fixture")
