"""Bounded pattern induction using a caller-mounted, pinned ReFormeR checkout.

The native extractor's prompt, parser and iterative replacement rule execute
unchanged. Only its Ollama transport and two broken import paths are bridged.
Use a dedicated process: upstream imports and logging are process-global.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import threading
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentinputlab.contracts import ValidationError

from .driver import (
    INDUCTION_PATH,
    PROMPT_MANAGER_PATH,
    PROMPT_PATH,
    UPSTREAM_COMMIT,
    BackendConfig,
    CompatibilityPolicy,
    CompletionBackend,
    Controls,
    DriverError,
    OpenAICompatibleBackend,
    UsageTracker,
    _canonical_hash,
    _canonical_json,
    _load_module,
    _load_patterns,
    _loads_strict_json,
    _sha256_file,
    load_compatibility_policy,
    verify_checkout,
)

RELEASE_ID = "training.reformer.pattern_induction"
PROMPT_BUILDER_PATH = "reformer/patterns/query_reformulation_prompts.py"
ADAPTER_REVISION = "native-induction-import-transport-bridge.v1"
_IMPORT_LOCK = threading.RLock()


def _save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(value) + "\n")


@dataclass(frozen=True, slots=True)
class ReFormeRQueryPair:
    query_id: str
    original_query: str
    reformulated_query: str | None = None
    split: str = "train"

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or self.split not in {"train", "validation", "test"}:
            raise ValidationError("ReFormeR pair split must be train, validation or test")
        for name in ("query_id", "original_query"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 48_000:
                raise ValidationError(f"ReFormeR {name} must be non-empty bounded text")
            if "\x00" in value:
                raise ValidationError(f"ReFormeR {name} must not contain NUL")
        # Held-out queries are used for split isolation, never for induction.
        # Their callers need not invent unused target rewrites to declare them.
        if self.split != "train" and self.reformulated_query is None:
            return
        value = self.reformulated_query
        if not isinstance(value, str) or not value.strip() or len(value) > 48_000:
            raise ValidationError("ReFormeR reformulated_query must be non-empty bounded text")
        if "\x00" in value:
            raise ValidationError("ReFormeR reformulated_query must not contain NUL")


@dataclass(frozen=True, slots=True)
class ReFormeRInductionControls:
    batch_size: int = 2
    max_patterns: int = 25
    max_model_calls: int = 8
    max_tokens: int = 4096
    max_cost_usd: float = 10.0
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_patterns", "max_model_calls", "max_tokens", "seed"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValidationError(f"ReFormeR induction {name} is invalid")
        if (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValidationError("ReFormeR induction max_cost_usd is invalid")


def validate_induction_splits(
    training_pairs: Sequence[ReFormeRQueryPair],
    held_out_pairs: Sequence[ReFormeRQueryPair],
) -> dict[str, Any]:
    """Audit caller-declared splits; does not infer unseen dataset membership."""
    if not training_pairs or not held_out_pairs:
        raise ValidationError("ReFormeR fit requires train pairs and declared held-out pairs")
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for pairs, train in ((training_pairs, True), (held_out_pairs, False)):
        for pair in pairs:
            if not isinstance(pair, ReFormeRQueryPair):
                raise ValidationError("ReFormeR fit requires ReFormeRQueryPair records")
            if (pair.split == "train") is not train:
                raise ValidationError("ReFormeR training/held-out split assignment is invalid")
            query = " ".join(pair.original_query.casefold().split())
            if pair.query_id in seen_ids or query in seen_queries:
                raise ValidationError("ReFormeR duplicate ID/query or train/held-out overlap")
            seen_ids.add(pair.query_id)
            seen_queries.add(query)
    return {
        "training_ids": [pair.query_id for pair in training_pairs],
        "held_out_ids": [pair.query_id for pair in held_out_pairs],
        "training_sha256": _canonical_hash([asdict(pair) for pair in training_pairs]),
        "held_out_sha256": _canonical_hash([asdict(pair) for pair in held_out_pairs]),
        "scope": "caller_declared_records_only",
        "held_out_used_in_model_prompts": False,
    }


def _blocked_ollama(*args, **kwargs):
    raise DriverError("UNDECLARED_MODEL_TRANSPORT", "native Ollama transport is not enabled")


@contextmanager
def _native_induction(root: Path):
    """Bind the pinned source's missing relative/absolute imports, then restore."""
    with _IMPORT_LOCK:
        prefix = "_ail_reformer_induction_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        package = prefix + ".reformer"
        patterns = package + ".patterns"
        names = (
            prefix,
            package,
            patterns,
            package + ".prompt_manager",
            patterns + ".prompt_manager",
            patterns + ".query_reformulation_prompts",
            patterns + ".iterative_pattern_extraction",
            "query_reformulation_prompts",
            "ollama",
        )
        previous = {name: sys.modules.get(name) for name in names}
        bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for name, path in (
                (prefix, root),
                (package, root / "reformer"),
                (patterns, root / "reformer/patterns"),
            ):
                module = types.ModuleType(name)
                module.__package__ = name
                module.__path__ = [str(path)]
                sys.modules[name] = module
            stub = types.ModuleType("ollama")
            stub.Client = _blocked_ollama
            sys.modules["ollama"] = stub
            manager = _load_module(package + ".prompt_manager", root / PROMPT_MANAGER_PATH)
            sys.modules[patterns + ".prompt_manager"] = manager
            prompts = _load_module(
                patterns + ".query_reformulation_prompts", root / PROMPT_BUILDER_PATH
            )
            sys.modules["query_reformulation_prompts"] = prompts
            native = _load_module(patterns + ".iterative_pattern_extraction", root / INDUCTION_PATH)
            yield native.IterativePatternExtractor, prompts.QueryPair
        finally:
            sys.dont_write_bytecode = bytecode
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


@dataclass(frozen=True, slots=True)
class ReFormeRPatternArtifact:
    root: Path
    artifact_id: str
    pattern_sha256: str
    patterns: tuple[Mapping[str, Any], ...]

    @property
    def pattern_path(self) -> Path:
        return self.root / "patterns.json"

    @classmethod
    def load(cls, root: str | Path) -> ReFormeRPatternArtifact:
        supplied = Path(root).expanduser()
        if supplied.is_symlink():
            raise ValidationError("ReFormeR artifact root must not be a symlink")
        folder = supplied.resolve(strict=True)
        for name in ("artifact.json", "fit-terminal.json", "patterns.json"):
            if (folder / name).is_symlink() or not (folder / name).is_file():
                raise ValidationError("ReFormeR fitted artifact file is unavailable")
        terminal = _loads_strict_json((folder / "fit-terminal.json").read_text(encoding="utf-8"))
        meta = _loads_strict_json((folder / "artifact.json").read_text(encoding="utf-8"))
        if (
            not isinstance(terminal, dict)
            or terminal.get("status") != "candidate_generated"
            or not isinstance(meta, dict)
            or meta.get("release_id") != RELEASE_ID
            or meta.get("schema_version") != "agentinputlab.reformer.pattern_artifact.v1"
            or meta.get("upstream_commit") != UPSTREAM_COMMIT
            or meta.get("artifact_id")
            != _canonical_hash({key: value for key, value in meta.items() if key != "artifact_id"})
            or terminal.get("artifact_id") != meta.get("artifact_id")
            or _sha256_file(folder / "patterns.json") != meta.get("pattern_sha256")
        ):
            raise ValidationError("ReFormeR fitted artifact identity/status mismatch")
        patterns = _load_patterns(folder / "patterns.json")
        if not meta.get("usage", {}).get("model_calls") or not meta.get("split_audit"):
            raise ValidationError("ReFormeR fitted artifact lacks fit provenance")
        return cls(folder, meta["artifact_id"], meta["pattern_sha256"], patterns)


class ReFormeRPatternInduction:
    """Fit a pattern library; candidate generation is not held-out acceptance."""

    def __init__(self, controls: ReFormeRInductionControls | None = None):
        self.controls = controls or ReFormeRInductionControls()

    def fit(
        self,
        *,
        checkout: str | Path,
        training_pairs: Sequence[ReFormeRQueryPair],
        held_out_pairs: Sequence[ReFormeRQueryPair],
        output_root: str | Path,
        backend_config: BackendConfig,
        backend: CompletionBackend,
        source_license_acknowledged: bool = False,
        policy: CompatibilityPolicy | None = None,
    ) -> ReFormeRPatternArtifact:
        if source_license_acknowledged is not True:
            raise ValidationError("ReFormeR requires acknowledgement of caller-mounted source")
        training_pairs, held_out_pairs = tuple(training_pairs), tuple(held_out_pairs)
        audit = validate_induction_splits(training_pairs, held_out_pairs)
        c = self.controls
        batches = (len(training_pairs) + c.batch_size - 1) // c.batch_size
        if batches > c.max_model_calls:
            raise ValidationError("ReFormeR declared training batches exceed model-call budget")
        policy = policy or load_compatibility_policy()
        source = verify_checkout(Path(checkout), policy)
        if not {INDUCTION_PATH, PROMPT_PATH, PROMPT_BUILDER_PATH} <= set(policy.source_hashes):
            raise ValidationError("ReFormeR induction source closure is incomplete")
        if _canonical_json(backend.fingerprint()) != _canonical_json(backend_config.fingerprint):
            raise ValidationError("ReFormeR induction backend identity differs from config")
        supplied = Path(output_root).expanduser()
        if supplied.is_symlink():
            raise ValidationError("ReFormeR fit output must not be a symlink")
        root = supplied.resolve()
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValidationError("ReFormeR fit output must be separate from upstream source")
        root.mkdir(parents=True, exist_ok=False)
        tracker = UsageTracker(
            Controls(c.max_model_calls, c.max_tokens, c.max_cost_usd), backend_config
        )
        _save(
            root / "fit-start.json",
            {
                "release_id": RELEASE_ID,
                "adapter_revision": ADAPTER_REVISION,
                "controls": asdict(c),
                "backend": dict(backend_config.fingerprint),
                "backend_settings": {
                    "per_call_output_tokens": backend_config.max_output_tokens,
                    "temperature": backend_config.temperature,
                },
                "source_hashes": dict(policy.source_hashes),
                "split_audit": audit,
                "automatic_retry": False,
                "upstream_transport": "replaced_ollama_with_bounded_backend",
                "import_path_repairs": ["patterns.prompt_manager", "query_reformulation_prompts"],
            },
        )
        try:
            with _native_induction(source) as (extractor_class, pair_class):
                extractor = extractor_class.__new__(extractor_class)
                extractor.max_patterns = c.max_patterns
                extractor.consolidated_patterns = []
                extractor.individual_patterns = []
                for index in range(batches):
                    if _canonical_json(backend.fingerprint()) != _canonical_json(
                        backend_config.fingerprint
                    ):
                        raise ValidationError("ReFormeR training backend changed identity")
                    pairs = training_pairs[index * c.batch_size : (index + 1) * c.batch_size]
                    native_pairs = [
                        pair_class(pair.original_query, pair.reformulated_query, pair.query_id)
                        for pair in pairs
                    ]
                    journal = root / f"batch-{index + 1:04d}"
                    journal.mkdir()

                    def call_llm(messages, *, _journal=journal, _index=index):
                        _save(
                            _journal / "request.json",
                            {
                                "messages": messages,
                                "seed": c.seed + _index,
                                "max_output_tokens": backend_config.max_output_tokens,
                                "temperature": backend_config.temperature,
                                "backend": dict(backend_config.fingerprint),
                            },
                        )
                        text = tracker.complete(
                            backend,
                            stage="pattern_induction",
                            messages=messages,
                            seed=c.seed + _index,
                        )
                        _save(
                            _journal / "response.json",
                            {
                                "text": text,
                                "usage_cumulative": tracker.as_dict(),
                                "model_trace": tracker.stages,
                            },
                        )
                        return {"choices": [{"message": {"content": text}}]}

                    extractor.call_llm = call_llm
                    # Native parser and consolidation semantics, not a local replacement.
                    with redirect_stdout(sys.stderr):
                        result = extractor.extract_patterns_from_batch(native_pairs, index + 1)
                    extractor.consolidated_patterns = result
                    _save(
                        journal / "native-result.json",
                        {
                            "patterns": [asdict(pattern) for pattern in result],
                            "individual_patterns": extractor.individual_patterns,
                            "empty_native_result": not bool(result),
                        },
                    )
                _save(root / "patterns.json", [asdict(p) for p in extractor.consolidated_patterns])
            verify_checkout(source, policy)
            if _canonical_json(backend.fingerprint()) != _canonical_json(
                backend_config.fingerprint
            ):
                raise ValidationError("ReFormeR training backend changed identity")
            patterns = _load_patterns(root / "patterns.json")
            meta = {
                "schema_version": "agentinputlab.reformer.pattern_artifact.v1",
                "release_id": RELEASE_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "source_hashes": dict(policy.source_hashes),
                "split_audit": audit,
                "backend": dict(backend_config.fingerprint),
                "controls": asdict(c),
                "usage": tracker.as_dict(),
                "pattern_count": len(patterns),
                "pattern_sha256": _sha256_file(root / "patterns.json"),
                "held_out_acceptance_complete": False,
                "paper_result_parity_claimed": False,
            }
            meta["artifact_id"] = _canonical_hash(meta)
            _save(root / "artifact.json", meta)
            _save(
                root / "fit-terminal.json",
                {
                    "status": "candidate_generated",
                    "artifact_id": meta["artifact_id"],
                    "usage": tracker.as_dict(),
                    "model_trace": tracker.stages,
                    "automatic_retry": False,
                    "held_out_acceptance_complete": False,
                },
            )
            return ReFormeRPatternArtifact.load(root)
        except Exception as exc:
            terminal = {
                "status": "failed_fit" if tracker.model_calls else "entry_failure",
                "error_type": type(exc).__name__,
                "error_code": exc.code if isinstance(exc, DriverError) else None,
                "usage": tracker.as_dict(),
                "model_trace": tracker.stages,
                "provider_usage_unknown": tracker.model_calls > len(tracker.stages)
                or any(
                    "failure_code" in stage and stage.get("provider_usage") is None
                    for stage in tracker.stages
                ),
                "automatic_retry": False,
                "held_out_acceptance_complete": False,
            }
            if not (root / "fit-terminal.json").exists():
                _save(root / "fit-terminal.json", terminal)
            raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument(
        "--pairs", required=True, type=Path, help="JSON list of declared split pairs"
    )
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acknowledge-external-source", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-patterns", type=int, default=25)
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    pairs = [ReFormeRQueryPair(**row) for row in _loads_strict_json(args.pairs.read_text())]
    config = BackendConfig.from_json(args.backend_config.read_text(encoding="utf-8"))
    controls = ReFormeRInductionControls(
        batch_size=args.batch_size,
        max_patterns=args.max_patterns,
        max_model_calls=args.max_model_calls,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    artifact = ReFormeRPatternInduction(controls).fit(
        checkout=args.checkout,
        training_pairs=[p for p in pairs if p.split == "train"],
        held_out_pairs=[p for p in pairs if p.split != "train"],
        output_root=args.output,
        backend_config=config,
        backend=OpenAICompatibleBackend(config),
        source_license_acknowledged=args.acknowledge_external_source,
    )
    print(
        _canonical_json(
            {
                "status": "candidate_generated",
                "artifact_id": artifact.artifact_id,
                "pattern_sha256": artifact.pattern_sha256,
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Keep expected CLI failures machine-readable without exposing exception chains.

    The fit implementation still records its original terminal and charged usage.
    This wrapper neither retries nor turns a failed fit into a deployable artifact.
    Unexpected programming errors remain visible rather than being relabelled.
    """
    try:
        return _main(argv)
    except (DriverError, ValidationError, OSError) as exc:
        if isinstance(exc, DriverError):
            code = exc.code
            # Only publish a bounded internal code, never provider exception text.
            if (not isinstance(code, str) or not 1 <= len(code) <= 80
                    or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in code)):
                code = "FIT_EXECUTION_FAILED"
        elif isinstance(exc, ValidationError):
            code = "INVALID_FIT_CONFIGURATION"
        elif isinstance(exc, FileExistsError):
            code = "OUTPUT_ALREADY_EXISTS"
        else:
            code = "FIT_FILE_UNAVAILABLE"
        print(_canonical_json({
            "status": "failed", "error_code": code, "automatic_retry": False,
            "message": "Fit did not complete. Check the supplied configuration and any fit-terminal.json "
                       "in the output directory; retain recorded usage before deciding whether to retry.",
        }), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
