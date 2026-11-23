"""Evaluator-bound retrieval feedback for ACE's unchanged offline training loop.

The generator produces query JSON. A caller-owned retriever and benchmark
adapter compute the feedback; qrels never enter the generated training input.
ACE's Boolean reflection branch uses a positive native score, while its
validation selection uses the mean of the declared native metric.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample
from agentinputlab.benchmarks.ir_query import parse_ir_query
from agentinputlab.benchmarks.retrieval import RetrievalPrediction
from agentinputlab.contracts import AlgorithmOutputError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable

from .ace import ACEExample


class _CapturedRetrievalAdapter:
    """Expose frozen visible contexts while scoring original owned examples."""

    def __init__(self, adapter, selected, captures, capture_sha256):
        from agentinputlab.api import _state

        if set(captures) != {'dataset_sha256', 'records'} or captures['dataset_sha256'] != content_hash(adapter.fingerprint()):
            raise ValidationError('captured retrieval dataset identity differs')
        originals = {e.example_id: e for group in selected.values() for e in group}
        records = captures['records']
        if not isinstance(records, dict) or set(records) != set(originals):
            raise ValidationError('captured retrieval IDs must match all declared groups exactly')
        self._originals, self._examples = originals, {}
        for key, original in originals.items():
            record = records[key]
            if not isinstance(record, dict) or set(record) != {'source_state_sha256', 'state', 'state_sha256'}:
                raise ValidationError('captured retrieval record requires source and captured state identities')
            state = _state(record['state'])
            before = original.visible_state()
            if record['source_state_sha256'] != before.state_hash or record['state_sha256'] != state.state_hash:
                raise ValidationError('captured retrieval state checksum differs')
            # The capture may add retrieval context and its cost, not edit a task.
            for field in ('task_id', 'query', 'instruction', 'messages', 'tool_names'):
                if getattr(before, field) != getattr(state, field):
                    raise ValidationError('captured retrieval changed the source task')
            metadata = dict(state.metadata)
            calls = metadata.pop('initial_capture_retrieval_calls', None)
            if type(calls) is not int or calls < 0 or metadata != dict(before.metadata):
                raise ValidationError('captured retrieval metadata must retain source and explicit retrieval cost')
            expected_budget = to_jsonable(before.budget)
            remaining = expected_budget['max_retrieval_calls']
            if remaining is not None:
                if calls > remaining:
                    raise ValidationError('captured retrieval cost exceeds the original allowance')
                expected_budget['max_retrieval_calls'] -= calls
            if to_jsonable(state.budget) != expected_budget:
                raise ValidationError('captured retrieval may only deduct its retrieval calls from the source budget')
            self._examples[key] = replace(original, visible_context=state.context,
                                           visible_metadata=state.metadata)
        self._native = adapter
        self._capture_sha256 = capture_sha256
        self.benchmark_id, self.pairing_mode = adapter.benchmark_id, adapter.pairing_mode

    def examples(self, split):
        return tuple(self._examples[e.example_id] for e in self._native.examples(split)
                     if e.example_id in self._examples)

    def score(self, example, prediction):
        if self._examples.get(example.example_id) is not example:
            raise ValidationError('captured retrieval example is not owned by this adapter')
        return self._native.score(self._originals[example.example_id], prediction)

    def fingerprint(self):
        return dict(native_dataset=self._native.fingerprint(),
                    captured_states_sha256=self._capture_sha256,
                    native_scorer_uses_original_owned_examples=True)


def _load_feedback_dataset(dataset, base):
    """Use the same verified native loaders for fit and optional heldout data."""
    from agentinputlab.configuration import _known, _object
    from agentinputlab.fit_configuration import _path, _reference

    dataset = _object(dataset, 'retrieval dataset')
    kind = dataset.get('kind')
    benchmark = _path(dataset.get('benchmark_id'), 'dataset.benchmark_id')
    if kind == 'production':
        _known(dataset, {'kind', 'benchmark_id', 'prepared_root'}, 'retrieval dataset')
    elif kind == 'beir_archive':
        _known(dataset, {'kind', 'benchmark_id', 'source_archive', 'license_acknowledged'}, 'retrieval dataset')
        _path(dataset.get('source_archive'), 'dataset.source_archive')
        if dataset.get('license_acknowledged') is not True:
            raise ValidationError('archive source terms must be explicitly acknowledged')
    elif kind in ('trec_dl_sample', 'beir_sample'):
        _known(dataset, {'kind', 'benchmark_id', 'root', 'manifest_sha256',
                         'license_acknowledged'}, 'retrieval dataset')
        _reference(dict(path=dataset.get('root'), sha256=dataset.get('manifest_sha256')),
                   'dataset manifest')
        if dataset.get('license_acknowledged') is not True:
            raise ValidationError('sample source terms must be explicitly acknowledged')
    else:
        raise ValidationError('retrieval dataset kind must be production, beir_archive, trec_dl_sample or beir_sample')
    # Use the public verified loaders, never reconstruct a scorer from gold rows.
    from agentinputlab.benchmarks import (
        BEIRSampleAdapter,
        TRECDLSampleAdapter,
        load_production_benchmark_adapter,
    )
    if kind == 'production':
        root = base / Path(_path(dataset.get('prepared_root'), 'prepared_root')).expanduser()
        adapter = load_production_benchmark_adapter(benchmark, root)
    elif kind == 'beir_archive':
        from agentinputlab.benchmarks.beir_archive import BEIRArchiveAdapter

        adapter = BEIRArchiveAdapter(benchmark_id=benchmark,
            source_archive=base / Path(dataset['source_archive']).expanduser(), license_acknowledged=True)
    else:
        root = base / Path(dataset['root']).expanduser()
        adapter = {'trec_dl_sample': TRECDLSampleAdapter, 'beir_sample': BEIRSampleAdapter}[kind](
            root, manifest_sha256=dataset['manifest_sha256'], license_acknowledged=True)
    if adapter.benchmark_id != benchmark:
        raise ValidationError('retrieval dataset benchmark identity differs')
    return adapter


class _RoleRetrievalAdapter:
    """Keep selected examples owned by their original corpus and native scorer."""

    def __init__(self, fit, heldout, groups, selected):
        self.benchmark_id, self.pairing_mode = fit.benchmark_id, fit.pairing_mode
        self._fit, self._heldout = fit, heldout
        self._groups = {role: dict(split=group['split'], ids=list(group['ids']))
                        for role, group in groups.items()}
        self._selected = selected
        self._owners = {e.example_id: (e, heldout if role == 'heldout' else fit)
                        for role, examples in selected.items() for e in examples}

    def examples(self, split):
        return tuple(e for role, group in self._groups.items() if group['split'] == split
                     for e in self._selected[role])

    def score(self, example, prediction):
        owned = self._owners.get(example.example_id)
        if owned is None or owned[0] is not example:
            raise ValidationError('retrieval example is not owned by this role adapter')
        return owned[1].score(example, prediction)

    def fingerprint(self):
        return dict(fit_dataset=self._fit.fingerprint(), heldout_dataset=self._heldout.fingerprint(),
                    groups=self._groups, native_scorer_uses_original_owned_examples=True)


def create_retrieval_feedback(*, source_input, output_root, mode='offline'):
    """Load native IR feedback from pinned assets and explicit fit groups.

    This is a file binding, not a new scorer or dataset split. Preparation is
    offline and performs only the original empty-ranking metric preflight.
    """
    from agentinputlab.artifacts import load_pinned_json_mapping
    from agentinputlab.configuration import RunConfig, _known, _object
    from agentinputlab.fit_configuration import _path, _reference

    if mode != 'offline':
        raise ValidationError('retrieval fit preparation supports offline mode only')
    pin = _reference(source_input, 'source_input')
    source = load_pinned_json_mapping(pin['path'], expected_sha256=pin['sha256'],
                                      max_bytes=8 * 1024 * 1024)
    _known(source, {'dataset', 'heldout_dataset', 'retriever_config', 'groups', 'metric', 'top_k',
                    'allow_json_fence', 'partition_label', 'captured_states'}, 'retrieval feedback source')
    base = Path(pin['path']).absolute().parent
    groups = _object(source.get('groups'), 'retrieval groups')
    if set(groups) != {'train', 'validation', 'heldout'}:
        raise ValidationError('retrieval groups require explicit train, validation and heldout')
    seen = set()
    for role, group in groups.items():
        group = _object(group, 'retrieval group')
        if set(group) != {'split', 'ids'}:
            raise ValidationError('retrieval group requires source split and IDs')
        _path(group['split'], 'source split')
        ids = group['ids']
        if (not isinstance(ids, list) or (role != 'heldout' and not ids)
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValidationError('retrieval group IDs must be explicit, unique and disjoint')
        seen.update(ids)
    adapter = _load_feedback_dataset(source.get('dataset'), base)
    benchmark = adapter.benchmark_id
    heldout = adapter
    if 'heldout_dataset' in source:
        if not groups['heldout']['ids']:
            raise ValidationError('heldout_dataset requires explicit nonempty heldout IDs')
        heldout = _load_feedback_dataset(source['heldout_dataset'], base)
        if heldout.benchmark_id != benchmark or heldout.pairing_mode != adapter.pairing_mode:
            raise ValidationError('fit and heldout dataset benchmark and pairing identities must match')
    selected = {}
    visible = set()
    for role, group in groups.items():
        try:
            owner = heldout if role == 'heldout' else adapter
            selected[role] = tuple(owner.example(group['split'], key) for key in group['ids'])
        except (KeyError, StopIteration):
            raise ValidationError('retrieval group contains an unknown source ID') from None
        for example in selected[role]:
            identity = content_hash(dict(query=example.query, instruction=example.instruction,
                                         context=example.visible_context, messages=example.visible_messages))
            if identity in visible:
                raise ValidationError('retrieval fit and heldout group inputs must be disjoint')
            visible.add(identity)
    if 'heldout_dataset' in source:
        adapter = _RoleRetrievalAdapter(adapter, heldout, groups, selected)
    if 'captured_states' in source:
        capture_pin = _reference(source['captured_states'], 'captured_states')
        captures = load_pinned_json_mapping(base / Path(capture_pin['path']).expanduser(),
            expected_sha256=capture_pin['sha256'], max_bytes=16 * 1024 * 1024)
        adapter = _CapturedRetrievalAdapter(adapter, selected, captures, capture_pin['sha256'])
        selected = {role: tuple(adapter._examples[e.example_id] for e in group)
                    for role, group in selected.items()}
    reference = _reference(source.get('retriever_config'), 'retriever_config')
    path = base / Path(reference['path']).expanduser()
    raw_config = load_pinned_json_mapping(path, expected_sha256=reference['sha256'], max_bytes=1024 * 1024)
    _known(raw_config, {'release_id', 'seed', 'retriever'}, 'retriever-only run configuration')
    if raw_config.get('release_id') != 'baseline.no_op' or not raw_config.get('retriever'):
        raise ValidationError('retriever-only configuration requires baseline.no_op and a retriever')
    config = RunConfig.from_file(path)
    if (config.retriever.get('kind') == 'prepared_corpus'
            and config.retriever.get('benchmark_id') != benchmark):
        raise ValidationError('retriever benchmark must match the feedback dataset')
    # Recheck the file read by RunConfig; its normal relative-path rules apply.
    load_pinned_json_mapping(path, expected_sha256=reference['sha256'], max_bytes=1024 * 1024)
    retriever = config._retriever()
    return ACERetrievalFeedback(adapter=adapter, retriever=retriever,
        train=selected['train'], validation=selected['validation'],
        metric=source.get('metric'), top_k=source.get('top_k'),
        allow_json_fence=source.get('allow_json_fence'), partition_label=source.get('partition_label'))


def _bind_feedback(
    processor, *, adapter, retriever, groups, metric, top_k, allow_json_fence, partition_label
):
    if not isinstance(adapter, BenchmarkAdapter) or not isinstance(retriever, Retriever):
        raise ValidationError("ACE retrieval feedback requires an adapter and retriever")
    if not callable(getattr(adapter, "fingerprint", None)):
        raise ValidationError("ACE retrieval adapter requires an explicit fingerprint")
    if not isinstance(metric, str) or not metric.strip():
        raise ValidationError("ACE retrieval metric must be explicit")
    if type(top_k) is not int or not 1 <= top_k <= 10_000:
        raise ValidationError("ACE retrieval top_k must be between 1 and 10000")
    if type(allow_json_fence) is not bool:
        raise ValidationError("ACE retrieval allow_json_fence must be boolean")
    if not isinstance(partition_label, str) or not partition_label.strip():
        raise ValidationError("ACE retrieval partition label must be explicit")
    cases = tuple(e for group in groups for e in group)
    if any(not group for group in groups):
        raise ValidationError("ACE retrieval groups must be nonempty")
    if any(not isinstance(e, BenchmarkExample) for e in cases):
        raise ValidationError("ACE retrieval groups require BenchmarkExample objects")
    if len({e.example_id for e in cases}) != len(cases):
        raise ValidationError("ACE retrieval group IDs must be disjoint and unique")
    if len(
        {
            content_hash(
                dict(
                    query=e.query,
                    instruction=e.instruction,
                    context=e.visible_context,
                    messages=e.visible_messages,
                )
            )
            for e in cases
        }
    ) != len(cases):
        raise ValidationError("ACE retrieval group inputs must be disjoint")
    if any(e.visible_tool_names for e in cases):
        raise ValidationError("ACE retrieval feedback does not execute tools")
    processor.adapter, processor.retriever = adapter, retriever
    processor.metric, processor.top_k = metric, top_k
    processor.allow_json_fence, processor.partition_label = allow_json_fence, partition_label
    processor._examples = {e.example_id: e for e in cases}
    processor._dataset = to_jsonable(adapter.fingerprint())
    processor._retriever = to_jsonable(retriever.fingerprint())
    processor._cache = {}
    processor.events = []
    for example in cases:
        processor._metric_value(adapter.score(example, RetrievalPrediction({})))


class ACERetrievalFeedback:
    """Bind one explicit IR task and disjoint caller-declared fit partitions.

    ``metric`` must name a scalar in ``adapter.score`` (for example
    ``"ndcg_at_10"``). The adapter retains its native exclusions and qrels.
    Invalid query JSON receives zero *training feedback*, not a repaired query
    or an extra model call. Held-out evaluation still uses its normal endpoint.
    """

    def __init__(
        self,
        *,
        adapter: BenchmarkAdapter,
        retriever: Retriever,
        train: Sequence[BenchmarkExample],
        validation: Sequence[BenchmarkExample],
        metric: str = "ndcg_at_10",
        top_k: int = 15,
        allow_json_fence: bool = False,
        partition_label: str = "caller_declared_compatibility_partitions",
    ) -> None:
        self.train, self.validation = tuple(train), tuple(validation)
        _bind_feedback(
            self,
            adapter=adapter,
            retriever=retriever,
            groups=(self.train, self.validation),
            metric=metric,
            top_k=top_k,
            allow_json_fence=allow_json_fence,
            partition_label=partition_label,
        )

    def _metric_value(self, metrics: Mapping[str, Any]) -> float:
        value = metrics.get(self.metric)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValidationError("ACE retrieval native metric is absent or outside [0,1]")
        return float(value)

    def fit_examples(self, split: str) -> tuple[ACEExample, ...]:
        """Build evaluator-blind query-generation inputs; targets are opaque IDs."""
        if split not in {"train", "validation"}:
            raise ValidationError("ACE retrieval exposes only train and validation groups")
        examples = self.train if split == "train" else self.validation
        return self._examples_for(examples, split)

    def _examples_for(self, examples, split):
        result = []
        for example in examples:
            state = example.visible_state()
            visible = dict(
                task_instruction=state.instruction,
                original_query=state.query,
                conversation=[dict(role=m.role, content=m.content) for m in state.messages],
            )
            result.append(
                ACEExample(
                    example_id=example.example_id,
                    input=dict(
                        question=(
                            "Generate a search query for this retrieval task. Preserve its "
                            "information need and task instruction. Your final answer must be "
                            'exactly one JSON object: {"query": "nonempty search query"}.\n'
                            + canonical_json(visible)
                        ),
                        context=canonical_json(
                            [
                                dict(
                                    document_id=b.block_id,
                                    text=b.text,
                                    source=b.source,
                                    score=b.score,
                                )
                                for b in state.context
                            ]
                        ),
                    ),
                    expected_output=example.example_id,
                    split=split,
                    metadata=dict(
                        task_processor="agentinputlab.ace.native_retrieval_feedback.v1",
                        benchmark_id=self.adapter.benchmark_id,
                        source_state_hash=state.state_hash,
                        partition_label=self.partition_label,
                        evaluator_target_is_opaque_id_not_reference_query=True,
                    ),
                )
            )
        return tuple(result)

    def validate_examples(self, train: Sequence[Mapping], validation: Sequence[Mapping]) -> None:
        """Reject unbound IDs, altered inputs or an attempted held-out fit before IO."""
        for supplied, split in ((train, "train"), (validation, "validation")):
            expected = tuple(e.as_dict() for e in self.fit_examples(split))
            if tuple(to_jsonable(e) for e in supplied) != expected:
                raise ValidationError("ACE retrieval examples differ from the frozen fit groups")

    def fingerprint(self) -> dict[str, Any]:
        return dict(
            processor_id="agentinputlab.ace.native_retrieval_feedback.v1",
            dataset=self._dataset,
            retriever=self._retriever,
            metric=self.metric,
            top_k=self.top_k,
            allow_json_fence=self.allow_json_fence,
            partition_label=self.partition_label,
            training_success_criterion="declared_native_metric_strictly_greater_than_zero",
            validation_selection="mean_declared_native_metric_not_text_exact_match",
            inputs={
                split: [
                    dict(
                        case_id=e.example_id,
                        visible_state_hash=e.visible_state().state_hash,
                        evaluator_gold_sha256=content_hash(e.evaluator_gold),
                    )
                    for e in cases
                ]
                for split, cases in (
                    ("train", self.train),
                    ("validation", self.validation),
                )
            },
            invalid_query_training_feedback=0.0,
            automatic_retry=False,
        )

    def score_query(self, predicted: str, target: str) -> float:
        if target not in self._examples:
            raise ValidationError("ACE retrieval target is outside the frozen fit groups")
        if to_jsonable(self.adapter.fingerprint()) != self._dataset:
            raise ValidationError("ACE retrieval dataset fingerprint changed during fitting")
        if to_jsonable(self.retriever.fingerprint()) != self._retriever:
            raise ValidationError("ACE retrieval retriever fingerprint changed during fitting")
        try:
            query = parse_ir_query(predicted, allow_json_fence=self.allow_json_fence)
        except AlgorithmOutputError:
            self.events.append(
                dict(
                    target=target,
                    status="invalid_query_training_feedback",
                    prediction_sha256=content_hash(predicted),
                    score=0.0,
                )
            )
            return 0.0
        key = target, query
        if key in self._cache:
            return self._cache[key]
        results = tuple(self.retriever.retrieve(query, top_k=self.top_k))
        if (
            len(results) > self.top_k
            or any(not isinstance(r, RetrievalResult) for r in results)
            or tuple(r.rank for r in results) != tuple(range(1, len(results) + 1))
            or len({r.document.doc_id for r in results}) != len(results)
        ):
            raise ValidationError("ACE retrieval received invalid or duplicate native ranks")
        prediction = RetrievalPrediction.from_ranked_ids([r.document.doc_id for r in results])
        metrics = dict(self.adapter.score(self._examples[target], prediction))
        score = self._metric_value(metrics)
        self._cache[key] = score
        self.events.append(
            dict(
                target=target,
                status="native_retrieval_scored",
                prediction_sha256=content_hash(predicted),
                query_sha256=content_hash(query),
                ranked_ids_sha256=content_hash([r.document.doc_id for r in results]),
                metrics=metrics,
                score=score,
            )
        )
        return score

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return self.score_query(predicted, ground_truth) > 0.0

    def evaluate_accuracy(self, predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValidationError("ACE retrieval predictions and targets must have equal length")
        return (
            sum(self.score_query(p, t) for p, t in zip(predictions, ground_truths, strict=True))
            / len(predictions)
            if predictions
            else 0.0
        )


class ACEOnlineRetrievalFeedback(ACERetrievalFeedback):
    """Native IR feedback for an explicitly transductive ordered online stream.

    Stream references are opaque case IDs; qrels remain evaluator-owned.
    This processor is not an offline fitting split or held-out evaluation.
    """

    def __init__(
        self,
        *,
        adapter,
        retriever,
        stream,
        metric="ndcg_at_10",
        top_k=15,
        allow_json_fence=False,
    ):
        self.stream = tuple(stream)
        _bind_feedback(
            self,
            adapter=adapter,
            retriever=retriever,
            groups=(self.stream,),
            metric=metric,
            top_k=top_k,
            allow_json_fence=allow_json_fence,
            partition_label="explicit_native_IR_feedback_transductive_online_stream",
        )

    def stream_examples(self):
        return self._examples_for(self.stream, "test")

    def validate_online_examples(self, examples):
        rows = tuple(ACEExample.from_value(row, split="test") for row in examples)
        if [r.as_dict() for r in rows] != [r.as_dict() for r in self.stream_examples()]:
            raise ValidationError("ACE online retrieval inputs/order differ from the frozen stream")

    def fit_examples(self, split):
        raise ValidationError("online retrieval feedback cannot be used as offline fit groups")

    def validate_examples(self, train, validation):
        raise ValidationError("online retrieval feedback cannot be used for offline fitting")

    def fingerprint(self):
        return dict(
            processor_id="agentinputlab.ace.native_online_retrieval_feedback.v1",
            dataset=self._dataset,
            retriever=self._retriever,
            metric=self.metric,
            top_k=self.top_k,
            allow_json_fence=self.allow_json_fence,
            partition_label=self.partition_label,
            stream=[
                dict(
                    case_id=e.example_id,
                    visible_state_hash=e.visible_state().state_hash,
                    evaluator_gold_sha256=content_hash(e.evaluator_gold),
                )
                for e in self.stream
            ],
            transductive_feedback_not_offline_or_heldout=True,
            success_criterion="declared_native_metric_strictly_greater_than_zero",
            evaluation_value="mean_declared_native_metric_not_text_exact_match",
            invalid_query_feedback=0.0,
            automatic_retry=False,
        )


__all__ = ["ACEOnlineRetrievalFeedback", "ACERetrievalFeedback"]
