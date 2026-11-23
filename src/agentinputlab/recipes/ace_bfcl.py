"""BFCL Python-simple AST feedback for ACE's native text generator.

The explicit task adapter asks ACE for function-call JSON in ``final_answer``;
it does not replace ACE's generator with a function-calling agent. Evaluation
uses the caller's source-verified BFCL adapter, never text exact match. Training
targets are opaque IDs, not the benchmark's hidden reference arguments.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable

from .ace import ACEExample


def create_bfcl_feedback(*, source_input, model_id, partition_label, output_root, mode='offline'):
    """Load the existing AST feedback from pinned assets and explicit fit groups.

    No model call, generated label, automatic partition or online adaptation is
    performed. The native BFCL release is a test split; these are caller-owned
    compatibility subdivisions, not an official training protocol.
    """
    from agentinputlab.artifacts import load_pinned_json_mapping
    from agentinputlab.benchmarks import SourceVerifiedBFCLV3ASTBridge, load_bfcl_v3_simple_adapter

    if mode != 'offline':
        raise ValidationError('file-configured BFCL feedback requires offline mode')
    source = load_pinned_json_mapping(source_input['path'],
        expected_sha256=source_input['sha256'], max_bytes=8 * 1024 * 1024)
    groups, seen = {}, set()
    for name in ('train_ids', 'validation_ids', 'heldout_ids'):
        ids = source.get(name)
        if (not isinstance(ids, (list, tuple)) or (name != 'heldout_ids' and not ids)
                or any(not isinstance(x, str) or not x.strip() for x in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValidationError('BFCL source requires explicit unique disjoint fit and heldout IDs')
        groups[name] = ids
        seen.update(ids)
    base = Path(source_input['path']).absolute().parent
    paths = {}
    for name in ('source_archive', 'normalized_root'):
        value = source.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError('BFCL source requires source_archive and normalized_root')
        paths[name] = base / Path(value).expanduser()
    bridge = SourceVerifiedBFCLV3ASTBridge(source_archive=paths['source_archive'], model_id=model_id)
    adapter = load_bfcl_v3_simple_adapter(paths['normalized_root'], evaluation_bridge=bridge)
    if dict(adapter.prepared.file_sha256) != source.get('prepared_file_hashes'):
        raise ValidationError('BFCL prepared-file hashes differ from the pinned fit source')
    return ACEBFCLFeedback(adapter=adapter,
        train=tuple(adapter.example('test', key) for key in groups['train_ids']),
        validation=tuple(adapter.example('test', key) for key in groups['validation_ids']),
        partition_label=partition_label)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate function-call JSON key")
        value[key] = item
    return value


def parse_bfcl_ace_answer(text: str) -> list[dict]:
    """Decode exactly one FC-JSON sequence; no fences, repair or code execution."""
    if not isinstance(text, str):
        raise ValueError("ACE BFCL final_answer must be a JSON string")
    value = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    canonical_json(value)  # Reject exponent overflow as well as explicit NaN/Infinity.
    if not isinstance(value, list) or not value:
        raise ValueError("ACE BFCL answer must be a nonempty function-call sequence")
    for call in value:
        if not isinstance(call, dict) or len(call) != 1:
            raise ValueError("each BFCL call must contain one function name")
        name, arguments = next(iter(call.items()))
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise ValueError("BFCL function name and argument object required")
    return value


class ACEBFCLFeedback:
    """Bind disjoint caller-declared groups to the native Python AST checker.

    This is a task-specific text-to-FC adapter, not a claim of native BFCL model
    handler prompt parity or a standard leaderboard training split.
    """

    processor_id = "agentinputlab.ace.BFCL_AST_feedback.v1"

    def __init__(
        self,
        *,
        adapter,
        train: Sequence[BenchmarkExample],
        validation: Sequence[BenchmarkExample],
        partition_label: str = "caller_declared_compatibility_partitions",
    ):
        self.train, self.validation = tuple(train), tuple(validation)
        self._initialize(
            adapter=adapter,
            groups=(("train", self.train), ("validation", self.validation)),
            partition_label=partition_label,
        )

    def _initialize(self, *, adapter, groups, partition_label):
        if getattr(adapter, "benchmark_id", None) != "bfcl.v3.simple_python":
            raise ValidationError("ACE BFCL requires the Python-simple adapter")
        if (
            not callable(getattr(adapter, "verify_integrity", None))
            or not callable(getattr(adapter, "score", None))
            or not callable(getattr(adapter, "function_tools", None))
        ):
            raise ValidationError("ACE BFCL requires integrity-checked native scoring")
        if not isinstance(partition_label, str) or not partition_label.strip():
            raise ValidationError("ACE BFCL partition label must be explicit")
        self.adapter = adapter
        self._groups = tuple(groups)
        cases = tuple(e for _, group in self._groups for e in group)
        if any(not group for _, group in self._groups) or any(
            not isinstance(e, BenchmarkExample) for e in cases
        ):
            raise ValidationError("ACE BFCL requires nonempty typed fit groups")
        if len({e.example_id for e in cases}) != len(cases):
            raise ValidationError("ACE BFCL fit IDs must be disjoint")
        for e in cases:
            if not isinstance(e.visible_metadata.get("function"), Mapping):
                raise ValidationError("ACE BFCL example lacks its visible function catalog")
        if len({content_hash(self._visible(e)) for e in cases}) != len(cases):
            raise ValidationError("ACE BFCL fit inputs must be disjoint")
        self.partition_label = partition_label
        self._examples = {e.example_id: e for e in cases}
        self._binding = self._native_binding()
        self.events = []

    def _visible(self, example):
        state = example.visible_state()
        return dict(
            query=state.query,
            instruction=state.instruction,
            messages=state.messages,
            context=state.context,
            tool_names=state.tool_names,
            function=example.visible_metadata.get("function"),
            native_function_tools=self.adapter.function_tools(example),
        )

    def _native_binding(self):
        self.adapter.verify_integrity()
        return to_jsonable(
            dict(
                prepared_files=dict(self.adapter.prepared.file_sha256),
                evaluation_provenance=self.adapter.evaluation_provenance,
            )
        )

    def fit_examples(self, split):
        if split not in {"train", "validation"}:
            raise ValidationError("ACE BFCL exposes only train and validation groups")
        cases = self.train if split == "train" else self.validation
        return self._examples_for(cases, split)

    def _examples_for(self, cases, split):
        return tuple(
            ACEExample(
                example_id=e.example_id,
                input=dict(
                    question=(
                        "Solve the supplied function-calling task. Preserve its task "
                        "instruction, user query and function schema. Put exactly a JSON "
                        "array of function calls in your final_answer string, using "
                        '[{"function_name": {"argument_name": "value"}}]. '
                        "Use the names in native_function_tools and their argument types, not this "
                        "illustrative name. Do not include code fences or commentary.\n"
                        + canonical_json(self._visible(e))
                    )
                ),
                expected_output=e.example_id,
                split=split,
                metadata=dict(
                    task_processor=self.processor_id,
                    source_state_hash=e.visible_state().state_hash,
                    partition_label=self.partition_label,
                    evaluator_target_is_opaque_id=True,
                ),
            )
            for e in cases
        )

    def validate_examples(self, train, validation):
        for supplied, split in ((train, "train"), (validation, "validation")):
            if tuple(to_jsonable(e) for e in supplied) != tuple(
                e.as_dict() for e in self.fit_examples(split)
            ):
                raise ValidationError("ACE BFCL examples differ from frozen fit groups")

    def fingerprint(self):
        return dict(
            processor_id=self.processor_id,
            native_binding=self._binding,
            metric="ast_accuracy",
            partition_label=self.partition_label,
            generator_output_adapter="native_ACE_text_final_answer_to_BFCL_FC_JSON",
            native_BFCL_handler_prompt_parity_claimed=False,
            training_success_criterion="native_ast_accuracy_strictly_greater_than_zero",
            validation_selection="mean_native_ast_accuracy",
            inputs={
                split: [
                    dict(
                        case_id=e.example_id,
                        visible_hash=content_hash(self._visible(e)),
                        evaluator_gold_sha256=content_hash(e.evaluator_gold),
                    )
                    for e in cases
                ]
                for split, cases in self._groups
            },
            automatic_retry=False,
        )

    def score_answer(self, predicted, target):
        if target not in self._examples:
            raise ValidationError("ACE BFCL target is outside frozen fit groups")
        if self._native_binding() != self._binding:
            raise ValidationError("ACE BFCL source/scorer fingerprint changed")
        try:
            calls = parse_bfcl_ace_answer(predicted)
        except (ValueError, TypeError):
            metrics, status = {"ast_accuracy": 0.0}, "invalid_FC_JSON_training_feedback"
        else:
            # The pinned OpenAI FC decoder calls json.loads on each argument
            # string. Preserve argument types while crossing that exact seam.
            native_calls = [
                {name: canonical_json(arguments)}
                for call in calls
                for name, arguments in call.items()
            ]
            metrics = dict(self.adapter.score(self._examples[target], native_calls))
            status = "native_AST_scored"
        value = metrics.get("ast_accuracy")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValidationError("ACE BFCL native ast_accuracy is absent or invalid")
        self.events.append(
            dict(
                target=target,
                prediction_sha256=content_hash(predicted),
                metrics=metrics,
                status=status,
            )
        )
        return float(value)

    def answer_is_correct(self, predicted, ground_truth):
        return self.score_answer(predicted, ground_truth) > 0

    def evaluate_accuracy(self, predictions, ground_truths):
        if len(predictions) != len(ground_truths):
            raise ValidationError("ACE BFCL predictions and targets differ in length")
        return (
            sum(self.score_answer(p, t) for p, t in zip(predictions, ground_truths, strict=True))
            / len(predictions)
            if predictions
            else 0.0
        )


class ACEOnlineBFCLFeedback(ACEBFCLFeedback):
    """AST feedback for an explicitly transductive native ACE online stream.

    Stream order is frozen before generation. This object is not an offline
    training split and its adapted artifact is not independent held-out evidence.
    """

    processor_id = "agentinputlab.ace.BFCL_AST_online_feedback.v1"

    def __init__(self, *, adapter, stream: Sequence[BenchmarkExample]):
        self.stream = tuple(stream)
        self._initialize(
            adapter=adapter,
            groups=(("stream", self.stream),),
            partition_label="explicit_native_BFCL_feedback_transductive_online_stream",
        )

    def stream_examples(self):
        return self._examples_for(self.stream, "test")

    def validate_online_examples(self, examples):
        supplied = tuple(ACEExample.from_value(row, split="test").as_dict() for row in examples)
        if supplied != tuple(row.as_dict() for row in self.stream_examples()):
            raise ValidationError("ACE online BFCL inputs/order differ from frozen stream")

    def fingerprint(self):
        return dict(
            super().fingerprint(),
            transductive_feedback_not_offline_or_heldout=True,
            stream_order_preserved=True,
        )

    def fit_examples(self, split):
        raise ValidationError("online BFCL feedback cannot be used as offline fit groups")

    def validate_examples(self, train, validation):
        raise ValidationError("online BFCL feedback cannot validate offline fit groups")
