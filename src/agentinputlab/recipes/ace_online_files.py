"""Pinned file bindings for existing BFCL and retrieval online processors.

An online stream is explicitly ordered, not inferred from offline fit groups.
The existing processors retain ownership of prompts, feedback and scoring.
"""

from pathlib import Path
from threading import RLock

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.configuration import RunConfig, _known, _object
from agentinputlab.contracts import ValidationError
from agentinputlab.fit_configuration import _path, _reference
from agentinputlab.recipes.ace_retrieval import ACEOnlineRetrievalFeedback


class BudgetedOnlineRetrievalFeedback(ACEOnlineRetrievalFeedback):
    """Count uncached retrieval attempts before IO, including failed attempts."""

    def __init__(self, *, retrieval_limit_per_case, **kwargs):
        if type(retrieval_limit_per_case) is not int or retrieval_limit_per_case < 0:
            raise ValidationError(
                "online retrieval_limit_per_case must be an explicit nonnegative integer"
            )
        super().__init__(**kwargs)
        self.retrieval_limit_per_case = retrieval_limit_per_case
        self.retrieval_attempts = {e.example_id: 0 for e in self.stream}
        self._retrieval_lock = RLock()

    def score_query(self, predicted, target):
        with self._retrieval_lock:
            return self._score_query_locked(predicted, target)

    def _score_query_locked(self, predicted, target):
        from agentinputlab.benchmarks.ir_query import parse_ir_query
        from agentinputlab.contracts import AlgorithmOutputError

        if target not in self._examples:
            raise ValidationError("target outside frozen online stream")
        try:
            query = parse_ir_query(predicted, allow_json_fence=self.allow_json_fence)
        except AlgorithmOutputError:
            return super().score_query(predicted, target)
        if (target, query) not in self._cache:
            if self.retrieval_attempts[target] >= self.retrieval_limit_per_case:
                raise ValidationError("online per-case retrieval allowance exhausted before IO")
            self.retrieval_attempts[target] += 1
        return super().score_query(predicted, target)

    def fingerprint(self):
        return dict(super().fingerprint(), retrieval_limit_per_case=self.retrieval_limit_per_case)


def _stream(source):
    stream = _object(source.get("stream"), "online source stream")
    _known(stream, {"split", "ids"}, "online source stream")
    _path(stream.get("split"), "online source split")
    ids = stream.get("ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(i, str) or not i.strip() for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValidationError("online source stream requires explicit unique ordered IDs")
    return stream


def _select(adapter, stream):
    selected = []
    for key in stream["ids"]:
        try:
            selected.append(adapter.example(stream["split"], key))
        except (KeyError, StopIteration):
            raise ValidationError("online stream contains an unknown source ID") from None
    return tuple(selected)


def create_online_feedback(*, kind, source_input, benchmark, model_id=None):
    """Read pinned local assets only; never generate labels or select fit groups."""
    pin = _reference(source_input, "online source_input")
    source = load_pinned_json_mapping(
        pin["path"], expected_sha256=pin["sha256"], max_bytes=8 * 1024 * 1024
    )
    base = Path(pin["path"]).absolute().parent
    if source.get("benchmark") != benchmark:
        raise ValidationError("online source benchmark differs from configuration")
    stream = _stream(source)
    if kind == "bfcl":
        from agentinputlab.benchmarks import (
            SourceVerifiedBFCLV3ASTBridge,
            load_bfcl_v3_simple_adapter,
        )
        from agentinputlab.recipes.ace_bfcl import ACEOnlineBFCLFeedback

        _known(
            source,
            {"benchmark", "stream", "source_archive", "normalized_root", "prepared_file_hashes"},
            "BFCL online source",
        )
        if benchmark != "bfcl.v3.simple_python" or stream["split"] != "test":
            raise ValidationError("BFCL online requires its original Python-simple test split")
        bridge = SourceVerifiedBFCLV3ASTBridge(
            source_archive=base
            / Path(_path(source.get("source_archive"), "source_archive")).expanduser(),
            model_id=_path(model_id, "BFCL model_id"),
        )
        adapter = load_bfcl_v3_simple_adapter(
            base / Path(_path(source.get("normalized_root"), "normalized_root")).expanduser(),
            evaluation_bridge=bridge,
        )
        if dict(adapter.prepared.file_sha256) != source.get("prepared_file_hashes"):
            raise ValidationError("BFCL prepared files differ from the pinned online source")
        selected = _select(adapter, stream)
        return ACEOnlineBFCLFeedback(adapter=adapter, stream=selected)
    if kind != "retrieval":
        raise ValidationError("online file factory supports only BFCL or retrieval")
    from agentinputlab.recipes.ace_retrieval import (
        _CapturedRetrievalAdapter,
        _load_feedback_dataset,
    )

    _known(
        source,
        {
            "benchmark",
            "stream",
            "dataset",
            "retriever_config",
            "metric",
            "top_k",
            "allow_json_fence",
            "captured_states",
            "retrieval_limit_per_case",
        },
        "retrieval online source",
    )
    adapter = _load_feedback_dataset(source.get("dataset"), base)
    if adapter.benchmark_id != benchmark:
        raise ValidationError("online dataset benchmark differs from source")
    selected = _select(adapter, stream)
    if "captured_states" in source:
        capture = _reference(source["captured_states"], "captured_states")
        data = load_pinned_json_mapping(
            base / Path(capture["path"]).expanduser(),
            expected_sha256=capture["sha256"],
            max_bytes=16 * 1024 * 1024,
        )
        adapter = _CapturedRetrievalAdapter(adapter, {"stream": selected}, data, capture["sha256"])
        selected = tuple(adapter._examples[e.example_id] for e in selected)
    reference = _reference(source.get("retriever_config"), "retriever_config")
    path = base / Path(reference["path"]).expanduser()
    raw = load_pinned_json_mapping(path, expected_sha256=reference["sha256"], max_bytes=1024 * 1024)
    _known(raw, {"release_id", "seed", "retriever"}, "retriever-only run configuration")
    if raw.get("release_id") != "baseline.no_op" or not raw.get("retriever"):
        raise ValidationError(
            "retriever-only configuration requires baseline.no_op and a retriever"
        )
    config = RunConfig.from_file(path)
    load_pinned_json_mapping(path, expected_sha256=reference["sha256"], max_bytes=1024 * 1024)
    if (
        config.retriever.get("kind") == "prepared_corpus"
        and config.retriever.get("benchmark_id") != benchmark
    ):
        raise ValidationError("online retriever benchmark differs from dataset")
    return BudgetedOnlineRetrievalFeedback(
        retrieval_limit_per_case=source.get("retrieval_limit_per_case"),
        adapter=adapter,
        retriever=config._retriever(),
        stream=selected,
        metric=source.get("metric"),
        top_k=source.get("top_k"),
        allow_json_fence=source.get("allow_json_fence"),
    )
