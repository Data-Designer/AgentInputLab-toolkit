"""Gold-isolated BFCL v3 simple-Python adapter and external scorer bridge.

The official v1.3 source archive supplies the prompt, possible answers, model
decoders, and AST checker. AgentInputLab separates prompts and evaluator gold.
Callers can bind an explicit non-attested callback or a source-verified Python
function-calling bridge. Case-level source verification does not assert the
full official CLI/aggregate leaderboard workflow.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.artifacts import sha256_file
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, freeze_mapping

from .assets import BenchmarkSmokeAssetRegistry
from .base import BenchmarkExample
from .preprocess import PreparedBenchmark, load_prepared_benchmark

BFCL_V3_OFFICIAL_SOURCE_REVISION = "ea13468e4423454d0c213704fb87cf7cb3990433"
BFCL_V3_SOURCE_ARCHIVE_SHA256 = "57854157e82aed5723a1458d2a4ac426e3256959ded4a20ddc4d627c22edb7fe"
BFCL_V3_SCORER_ID = (
    "bfcl_eval.eval_checker.eval_runner:ast_file_runner+"
    "bfcl_eval.eval_checker.ast_eval.ast_checker:ast_checker"
)
BFCL_V3_SCORER_REVISION = "31f1420ae78709920d2fc49ceb93ea3dc451379cf5f320ac0b02b9845d0cbb36"
BFCL_V3_AST_CHECKER_REVISION = "58a8a1d8a42ec4f7e0f340a8bf11ab4035c54d38d933ae2621f22bbc7c8d1c9b"

_BENCHMARK_ID = "bfcl.v3.simple_python"
_ASSET_ID = "bfcl.v3.simple_python.v1_3"
_PROCESSOR = "bfcl_v3_simple_tar_v1"
_SPLIT = "test"
_CALLER_ASSERTED = "caller_asserted"
_EXTERNAL_BRIDGE = "external_bridge"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOADER_CONSTRUCTION_TOKEN = object()
_EXPECTED_IDS = frozenset(f"simple_{index}" for index in range(400))
_EXPECTED_COUNTS = {
    "audit": 1,
    "functions": 400,
    "gold": 400,
    "ground_truth_candidates": 400,
    "inputs": 400,
    "source_files": 8,
}
_INPUT_FIELDS = {
    "id",
    "instruction",
    "query",
    "visible_metadata",
    "visible_tool_names",
}
_VISIBLE_METADATA_FIELDS = {
    "benchmark_cell",
    "bfcl_category",
    "function",
    "source_split",
}
_FUNCTION_FIELDS = {"description", "name", "parameters"}
_GOLD_FIELDS = {"ground_truth", "id"}
BFCL_V3_SELECTED_SOURCE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        (
            "berkeley-function-call-leaderboard/bfcl_eval/constants/category_mapping.py"
        ): "c08dfe14283990bc4f67333183d8c14a2506da6d99ac00c6ef5f018dd44757fc",
        (
            "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_simple.json"
        ): "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
        (
            "berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_simple.json"
        ): "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
        (
            "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py"
        ): BFCL_V3_AST_CHECKER_REVISION,
        (
            "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/eval_runner.py"
        ): BFCL_V3_SCORER_REVISION,
        (
            "berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py"
        ): "4a2f4e677d1f6930f396dbc92c50138f56268f970de7e4f815fb801607cafad1",
        (
            "berkeley-function-call-leaderboard/pyproject.toml"
        ): "20d86859dc988e442c13a6c4f3567094bd8f2d4a269d59015337bdf6aeba7dd3",
    }
)


@dataclass(frozen=True, slots=True)
class BFCLV3EvaluationResult:
    """One caller-asserted official-AST validity bit."""

    ast_accuracy: float
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if isinstance(self.ast_accuracy, bool) or not isinstance(self.ast_accuracy, (int, float)):
            raise ValidationError("BFCL v3 ast_accuracy must be numeric")
        numeric = float(self.ast_accuracy)
        if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
            raise ValidationError("BFCL v3 case-level ast_accuracy must be exactly 0 or 1")
        object.__setattr__(self, "ast_accuracy", numeric)

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType({"ast_accuracy": self.ast_accuracy})


@runtime_checkable
class BFCLV3ScorerBridge(Protocol):
    """Revision-bound external BFCL decoder/scorer contract."""

    bfcl_bridge_contract: str
    benchmark_id: str
    source_revision: str
    scorer_id: str
    scorer_revision: str
    ast_checker_revision: str
    model_handler_id: str
    model_handler_revision: str
    bridge_id: str
    bridge_revision: str
    test_category: str
    language: str
    provenance_status: str
    evaluation_scope: str
    official_runtime_verified: bool

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        prediction: Any,
    ) -> BFCLV3EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CallerAssertedBFCLV3ScorerBridge:
    """Explicit wrapper for a non-attested external BFCL v1.3 callback."""

    bridge_id: str
    bridge_revision: str
    model_handler_id: str
    model_handler_revision: str
    callback: Callable[
        [Mapping[str, Any], Any],
        BFCLV3EvaluationResult,
    ] = field(repr=False)
    bfcl_bridge_contract: str = field(init=False, default="bfcl_v3_scorer_bridge_v1")
    benchmark_id: str = field(init=False, default=_BENCHMARK_ID)
    source_revision: str = field(init=False, default=BFCL_V3_OFFICIAL_SOURCE_REVISION)
    scorer_id: str = field(init=False, default=BFCL_V3_SCORER_ID)
    scorer_revision: str = field(init=False, default=BFCL_V3_SCORER_REVISION)
    ast_checker_revision: str = field(
        init=False,
        default=BFCL_V3_AST_CHECKER_REVISION,
    )
    test_category: str = field(init=False, default="simple")
    language: str = field(init=False, default="Python")
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        for label in ("bridge_id", "model_handler_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"BFCL v3 scorer bridge {label} must be non-empty")
        for label in ("bridge_revision", "model_handler_revision"):
            value = getattr(self, label)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValidationError(f"BFCL v3 scorer bridge {label} must be a lowercase SHA-256")
        if not callable(self.callback):
            raise TypeError("BFCL v3 scorer bridge callback must be callable")

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        prediction: Any,
    ) -> BFCLV3EvaluationResult:
        result = self.callback(evaluator_payload, prediction)
        if not isinstance(result, BFCLV3EvaluationResult):
            raise ValidationError(
                "BFCL v3 scorer callback must return BFCLV3EvaluationResult; "
                "bare metrics are not accepted"
            )
        if (
            result.provenance_status != self.provenance_status
            or result.evaluation_scope != self.evaluation_scope
            or result.official_runtime_verified is not False
        ):
            raise ValidationError(
                "BFCL v3 evaluation result provenance differs from its caller-asserted bridge"
            )
        return result


class PreparedBFCLV3SimpleAdapter:
    """Loader-only production adapter for the locked BFCL v3 simple-Python cell."""

    benchmark_id = _BENCHMARK_ID
    adapter_metric_names = ("ast_accuracy",)
    split = _SPLIT
    pairing_mode = "stateless_score_twice"

    def __init__(
        self,
        prepared: PreparedBenchmark,
        *,
        evaluation_bridge: BFCLV3ScorerBridge,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _LOADER_CONSTRUCTION_TOKEN:
            raise ValidationError(
                "BFCL v3 production adapters must be constructed with load_bfcl_v3_simple_adapter()"
            )
        if not isinstance(prepared, PreparedBenchmark):
            raise TypeError("prepared must be a PreparedBenchmark")
        if not isinstance(evaluation_bridge, BFCLV3ScorerBridge):
            raise TypeError(
                "evaluation_bridge must implement BFCLV3ScorerBridge; wrap external "
                "callbacks with CallerAssertedBFCLV3ScorerBridge"
            )
        _validate_prepared_identity(prepared)
        _validate_scorer_bridge(evaluation_bridge)
        _verify_prepared_file_hashes(prepared)

        input_rows = _records_by_id(prepared.inputs_path, label="BFCL v3 prepared inputs")
        gold_rows = _records_by_id(prepared.gold_path, label="BFCL v3 evaluator gold")
        audit_rows = _read_jsonl(prepared.audit_path, label="BFCL v3 source audit")
        if set(input_rows) != _EXPECTED_IDS or set(gold_rows) != _EXPECTED_IDS:
            raise ArtifactError("BFCL v3 simple IDs differ from the audited 400-case cell")
        if len(audit_rows) != 1:
            raise ArtifactError("BFCL v3 source audit must contain exactly one record")
        _validate_source_audit(audit_rows[0])

        examples: list[BenchmarkExample] = []
        for index in range(400):
            example_id = f"simple_{index}"
            visible = input_rows[example_id]
            gold = gold_rows[example_id]
            if set(visible) != _INPUT_FIELDS:
                raise ArtifactError("BFCL v3 prepared input schema differs from the lock")
            if set(gold) != _GOLD_FIELDS or gold.get("id") != example_id:
                raise ArtifactError("BFCL v3 evaluator gold schema differs from the lock")
            query = _required_text(visible, "query", label="BFCL v3 prepared input")
            instruction = _required_text(
                visible,
                "instruction",
                label="BFCL v3 prepared input",
            )
            metadata = _validated_visible_metadata(visible["visible_metadata"])
            function = metadata["function"]
            if not isinstance(function, Mapping):  # pragma: no cover - validation invariant
                raise ArtifactError("BFCL v3 function metadata is invalid")
            tool_names = visible["visible_tool_names"]
            if not isinstance(tool_names, list) or tool_names != [function["name"]]:
                raise ArtifactError(
                    "BFCL v3 visible tool names must contain the prompt function exactly"
                )
            ground_truth = gold["ground_truth"]
            if (
                not isinstance(ground_truth, list)
                or len(ground_truth) != 1
                or not isinstance(ground_truth[0], Mapping)
            ):
                raise ArtifactError("BFCL v3 simple evaluator gold must contain one answer object")
            examples.append(
                BenchmarkExample(
                    example_id=example_id,
                    query=query,
                    instruction=instruction,
                    visible_metadata=metadata,
                    visible_tool_names=(function["name"],),
                    evaluator_gold=_frozen_mapping(
                        {"ground_truth": ground_truth},
                        label="BFCL v3 evaluator gold",
                    ),
                )
            )

        self.prepared = prepared
        self._evaluation_bridge = evaluation_bridge
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._expected_file_sha256 = MappingProxyType(dict(prepared.file_sha256))

    def verify_integrity(self) -> None:
        """Re-hash all prepared files before exposure or scoring."""

        if dict(self.prepared.file_sha256) != dict(self._expected_file_sha256):
            raise ArtifactError("prepared BFCL v3 declared file hashes changed after loading")
        _verify_prepared_file_hashes(self.prepared)

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(
                f"BFCL v3 simple adapter exposes split {self.split!r}, not {split!r}"
            )
        self.verify_integrity()
        return self._examples

    def example(self, split: str, example_id: str) -> BenchmarkExample:
        if split != self.split:
            raise ValidationError(
                f"BFCL v3 simple adapter exposes split {self.split!r}, not {split!r}"
            )
        self.verify_integrity()
        try:
            return self._examples_by_id[example_id]
        except KeyError as exc:
            raise ValidationError(f"unknown BFCL v3 simple example {example_id!r}") from exc

    def function_tools(self, example: BenchmarkExample) -> tuple[Mapping[str, Any], ...]:
        """Expose the bound handler's native function schema, including name conversion."""
        if self._examples_by_id.get(getattr(example, "example_id", None)) is not example:
            raise ValidationError("benchmark example does not belong to this BFCL v3 adapter")
        self.verify_integrity()
        convert = getattr(self._evaluation_bridge, "tools", None)
        if not callable(convert):
            raise ValidationError("this BFCL scorer bridge does not expose native function tools")
        tools = tuple(convert(example.visible_metadata["function"]))
        if not tools or any(not isinstance(tool, Mapping) for tool in tools):
            raise ValidationError("BFCL native schema converter returned invalid tools")
        return tools

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this BFCL v3 adapter")
        self.verify_integrity()
        function = example.visible_metadata.get("function")
        if not isinstance(function, Mapping):  # pragma: no cover - construction invariant
            raise ArtifactError("BFCL v3 example function metadata is invalid")
        ground_truth = example.evaluator_gold.get("ground_truth")
        if not isinstance(ground_truth, tuple):  # frozen by construction
            raise ArtifactError("BFCL v3 example evaluator gold is invalid")
        evaluator_payload = _frozen_mapping(
            {
                "benchmark_id": self.benchmark_id,
                "function": function,
                "ground_truth": ground_truth,
                "language": self._evaluation_bridge.language,
                "source_revision": BFCL_V3_OFFICIAL_SOURCE_REVISION,
                "test_category": self._evaluation_bridge.test_category,
            },
            label="BFCL v3 evaluator payload",
        )
        result = self._evaluation_bridge.evaluate(evaluator_payload, prediction)
        if not isinstance(result, BFCLV3EvaluationResult):
            raise ValidationError("BFCL v3 scorer bridge returned an invalid result type")
        return result.metrics

    @property
    def evaluation_provenance(self) -> Mapping[str, Any]:
        """Return explicit scorer/decoder identities and evaluation scope."""

        return _frozen_mapping(
            {
                "ast_checker_revision": self._evaluation_bridge.ast_checker_revision,
                "benchmark_id": self._evaluation_bridge.benchmark_id,
                "bridge_id": self._evaluation_bridge.bridge_id,
                "bridge_revision": self._evaluation_bridge.bridge_revision,
                "evaluation_scope": self._evaluation_bridge.evaluation_scope,
                "language": self._evaluation_bridge.language,
                "model_handler_id": self._evaluation_bridge.model_handler_id,
                "model_handler_revision": self._evaluation_bridge.model_handler_revision,
                "official_runtime_verified": self._evaluation_bridge.official_runtime_verified,
                "provenance_status": self._evaluation_bridge.provenance_status,
                "scorer_id": self._evaluation_bridge.scorer_id,
                "scorer_revision": self._evaluation_bridge.scorer_revision,
                "source_revision": self._evaluation_bridge.source_revision,
                "test_category": self._evaluation_bridge.test_category,
            },
            label="BFCL v3 evaluation provenance",
        )


def load_bfcl_v3_simple_adapter(
    prepared_root: str | Path,
    *,
    evaluation_bridge: BFCLV3ScorerBridge,
) -> PreparedBFCLV3SimpleAdapter:
    """Load the exact checksum-pinned BFCL cell and its typed scorer bridge."""

    asset = BenchmarkSmokeAssetRegistry.default().get(_ASSET_ID)
    if asset.cell_id != _BENCHMARK_ID or asset.sha256 != BFCL_V3_SOURCE_ARCHIVE_SHA256:
        raise ArtifactError("registered BFCL v3 source identity differs from the production lock")
    prepared = load_prepared_benchmark(asset, prepared_root)
    return _construct_bfcl_v3_simple_adapter(
        prepared,
        evaluation_bridge=evaluation_bridge,
        _construction_token=_LOADER_CONSTRUCTION_TOKEN,
    )


def _construct_bfcl_v3_simple_adapter(
    prepared: PreparedBenchmark,
    *,
    evaluation_bridge: BFCLV3ScorerBridge,
    _construction_token: object,
) -> PreparedBFCLV3SimpleAdapter:
    """Internal seam shared only by the verified loader and isolated unit fixtures."""

    if _construction_token is not _LOADER_CONSTRUCTION_TOKEN:
        raise ValidationError("invalid BFCL v3 loader construction token")
    return PreparedBFCLV3SimpleAdapter(
        prepared,
        evaluation_bridge=evaluation_bridge,
        _construction_token=_construction_token,
    )


def _validate_prepared_identity(prepared: PreparedBenchmark) -> None:
    expected = {
        "asset_id": _ASSET_ID,
        "cell_id": _BENCHMARK_ID,
        "processor": _PROCESSOR,
        "processor_revision": "1",
        "source_revision": BFCL_V3_OFFICIAL_SOURCE_REVISION,
        "source_sha256": BFCL_V3_SOURCE_ARCHIVE_SHA256,
        "split": _SPLIT,
    }
    for name, value in expected.items():
        if getattr(prepared, name) != value:
            raise ArtifactError(f"prepared BFCL v3 identity mismatch for {name}")
    if dict(prepared.counts) != _EXPECTED_COUNTS:
        raise ArtifactError("prepared BFCL v3 counts differ from the audited cell")
    if set(prepared.file_sha256) != {"audit.jsonl", "gold.jsonl", "inputs.jsonl"}:
        raise ArtifactError("prepared BFCL v3 file set is incomplete")
    if prepared.audit_path is None:
        raise ArtifactError("prepared BFCL v3 source audit is missing")


def _validate_scorer_bridge(bridge: BFCLV3ScorerBridge) -> None:
    from .bfcl_native import BFCL_NATIVE_SCOPE, BFCL_NATIVE_SCORER_ID, SourceVerifiedBFCLV3ASTBridge
    if isinstance(bridge,SourceVerifiedBFCLV3ASTBridge):
        if (getattr(bridge,'_verified_source_sha256',None)!=BFCL_V3_SOURCE_ARCHIVE_SHA256
                or bridge.provenance_status!='source_verified' or bridge.evaluation_scope!=BFCL_NATIVE_SCOPE
                or bridge.scorer_id!=BFCL_NATIVE_SCORER_ID or bridge.official_runtime_verified is not False):
            raise ValidationError('BFCL native source/scoring scope differs')
        return
    expected = {
        "ast_checker_revision": BFCL_V3_AST_CHECKER_REVISION,
        "benchmark_id": _BENCHMARK_ID,
        "bfcl_bridge_contract": "bfcl_v3_scorer_bridge_v1",
        "evaluation_scope": _EXTERNAL_BRIDGE,
        "language": "Python",
        "provenance_status": _CALLER_ASSERTED,
        "scorer_id": BFCL_V3_SCORER_ID,
        "scorer_revision": BFCL_V3_SCORER_REVISION,
        "source_revision": BFCL_V3_OFFICIAL_SOURCE_REVISION,
        "test_category": "simple",
    }
    for name, value in expected.items():
        if getattr(bridge, name) != value:
            raise ValidationError(f"BFCL v3 scorer bridge identity mismatch for {name}")
    if bridge.official_runtime_verified is not False:
        raise ValidationError("current BFCL v3 bridge cannot claim official runtime verification")
    for label in ("bridge_id", "model_handler_id"):
        value = getattr(bridge, label)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValidationError(f"BFCL v3 scorer bridge {label} must be non-empty")
    for label in ("bridge_revision", "model_handler_revision"):
        value = getattr(bridge, label)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValidationError(f"BFCL v3 scorer bridge {label} must be a lowercase SHA-256")


def _validate_source_audit(audit: Mapping[str, Any]) -> None:
    expected_fields = {
        "category",
        "evaluation_boundary",
        "id",
        "license_id",
        "official_cli",
        "selected_source_sha256",
        "source_revision",
        "source_tag",
    }
    if set(audit) != expected_fields:
        raise ArtifactError("BFCL v3 source audit schema differs from the production lock")
    expected = {
        "category": "simple",
        "id": "bfcl-v3-simple-source-audit",
        "license_id": "Apache-2.0",
        "official_cli": "bfcl evaluate --test-category simple",
        "source_revision": BFCL_V3_OFFICIAL_SOURCE_REVISION,
        "source_tag": "v1.3",
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ArtifactError(f"BFCL v3 source audit identity mismatch for {key}")
    if dict(audit.get("selected_source_sha256", {})) != dict(BFCL_V3_SELECTED_SOURCE_SHA256):
        raise ArtifactError("BFCL v3 selected source hashes differ from the audited lock")
    boundary = audit.get("evaluation_boundary")
    if (
        not isinstance(boundary, str)
        or "evaluator-only" not in boundary
        or "does not execute BFCL" not in boundary
    ):
        raise ArtifactError("BFCL v3 source audit does not preserve the evaluation boundary")


def _validated_visible_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _VISIBLE_METADATA_FIELDS:
        raise ArtifactError("BFCL v3 visible metadata schema differs from the lock")
    if (
        value.get("benchmark_cell") != _BENCHMARK_ID
        or value.get("bfcl_category") != "simple"
        or value.get("source_split") != _SPLIT
    ):
        raise ArtifactError("BFCL v3 visible metadata identity differs from the lock")
    function = value.get("function")
    if not isinstance(function, Mapping) or set(function) != _FUNCTION_FIELDS:
        raise ArtifactError("BFCL v3 function schema differs from the lock")
    _required_text(function, "name", label="BFCL v3 function")
    _required_text(function, "description", label="BFCL v3 function")
    if not isinstance(function.get("parameters"), Mapping):
        raise ArtifactError("BFCL v3 function parameters must be an object")
    return _frozen_mapping(value, label="BFCL v3 visible metadata")


def _verify_prepared_file_hashes(prepared: PreparedBenchmark) -> None:
    for name, expected in sorted(prepared.file_sha256.items()):
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ArtifactError(f"prepared BFCL v3 hash is invalid for {name}")
        try:
            actual = sha256_file(prepared.root / name)
        except OSError as exc:
            raise ArtifactError(f"prepared BFCL v3 file cannot be verified: {name}") from exc
        if actual != expected:
            raise ArtifactError(f"prepared BFCL v3 file hash mismatch: {name}")


def _records_by_id(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path, label=label):
        row_id = _required_text(row, "id", label=label)
        if row_id in indexed:
            raise ArtifactError(f"{label} contains duplicate IDs")
        indexed[row_id] = row
    return indexed


def _read_jsonl(path: Path | None, *, label: str) -> list[dict[str, Any]]:
    if path is None:
        raise ArtifactError(f"{label} is missing")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactError(f"{label} cannot be read as UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ArtifactError(f"{label} contains a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{label} contains invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"{label} rows must be objects")
        rows.append(value)
    if not rows:
        raise ArtifactError(f"{label} must not be empty")
    return rows


def _required_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} field {key!r} must be a non-empty string")
    return value


def _frozen_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        detached = json.loads(canonical_json(value))
        return freeze_mapping(detached)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} must be canonically serializable") from exc


__all__ = [
    "BFCL_V3_AST_CHECKER_REVISION",
    "BFCL_V3_OFFICIAL_SOURCE_REVISION",
    "BFCL_V3_SCORER_ID",
    "BFCL_V3_SCORER_REVISION",
    "BFCL_V3_SELECTED_SOURCE_SHA256",
    "BFCL_V3_SOURCE_ARCHIVE_SHA256",
    "BFCLV3EvaluationResult",
    "BFCLV3ScorerBridge",
    "CallerAssertedBFCLV3ScorerBridge",
    "PreparedBFCLV3SimpleAdapter",
    "load_bfcl_v3_simple_adapter",
]
