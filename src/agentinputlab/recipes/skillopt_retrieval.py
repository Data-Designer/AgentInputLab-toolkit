"""IR environment for SkillOpt's source-native text-space optimizer.

Create the adapter inside a trusted environment factory after mounting the
pinned SkillOpt checkout. The trainer keeps its native six-stage update loop;
this environment owns query generation, retrieval and scalar native feedback.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

from agentinputlab.backends.llm import LLMBackend
from agentinputlab.backends.retrieval import Retriever
from agentinputlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample
from agentinputlab.benchmarks.ir_query import ContextIRQueryPredictor, IRQueryOutputError
from agentinputlab.benchmarks.retrieval import RetrievalPrediction
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, to_jsonable


def create_skillopt_retrieval_adapter(
    *, adapter: BenchmarkAdapter, retriever: Retriever, target_backend: LLMBackend,
    train, validation, cfg: dict, metric: str = "ndcg_at_10", top_k: int = 15,
    per_call_output_tokens: int = 512, seed: int = 17, allow_json_fence: bool = False,
    partition_label: str = "caller_declared_compatibility_partitions",
):
    """Return a native EnvAdapter with disjoint fit-only query groups.

    The backend must share the fit's metered provider with the optimizer.
    ``hard`` is native metric > 0 and ``soft`` is that unchanged metric. Invalid
    query JSON receives zero training feedback with its paid output retained.
    Held-out groups are evaluated separately and are never accepted here.
    """
    try:
        from skillopt.envs.base import EnvAdapter
    except ImportError as exc:
        raise ExternalDependencyError("mount the pinned SkillOpt checkout before creating its IR environment") from exc
    if not isinstance(adapter, BenchmarkAdapter) or not isinstance(retriever, Retriever):
        raise ValidationError("SkillOpt IR requires an explicit dataset adapter and retriever")
    if not isinstance(target_backend, LLMBackend):
        raise ValidationError("SkillOpt IR requires an explicit metered target backend")
    if not isinstance(cfg, dict) or not isinstance(metric, str) or not metric.strip():
        raise ValidationError("SkillOpt IR requires native configuration and an explicit metric")
    if not isinstance(partition_label, str) or not partition_label.strip():
        raise ValidationError("SkillOpt IR partition label must be explicit")
    groups = {"train":tuple(train), "validation":tuple(validation)}
    examples = (*groups["train"], *groups["validation"])
    if any(not group for group in groups.values()) or any(not isinstance(e, BenchmarkExample) for e in examples):
        raise ValidationError("SkillOpt IR requires nonempty train and validation examples")
    if len({e.example_id for e in examples}) != len(examples) or len({
        content_hash(dict(query=e.query, instruction=e.instruction, context=e.visible_context,
                          messages=e.visible_messages)) for e in examples
    }) != len(examples):
        raise ValidationError("SkillOpt IR train/validation IDs and visible inputs must be disjoint")
    if any(e.visible_tool_names for e in examples):
        raise ValidationError("SkillOpt IR environment does not execute tools")
    dataset = to_jsonable(adapter.fingerprint())
    corpus = to_jsonable(retriever.fingerprint())
    # Validate endpoint controls before the trainer can start a paid call.
    ContextIRQueryPredictor(target_backend, retriever, max_tokens=per_call_output_tokens,
                            top_k=top_k, seed=seed, allow_json_fence=allow_json_fence)

    def metric_value(metrics):
        value = metrics.get(metric)
        if type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValidationError("SkillOpt IR native metric must be present and within [0,1]")
        return float(value)

    for e in examples:
        metric_value(adapter.score(e, RetrievalPrediction({})))
    owned = {content_hash(e.example_id): e for e in examples}

    class RetrievalEnvAdapter(EnvAdapter):
        def __init__(self):
            self.analyst_workers = cfg.get("analyst_workers", 1)
            self.failure_only = cfg.get("failure_only", False)
            self.minibatch_size = cfg.get("minibatch_size", 2)
            self.edit_budget = cfg.get("edit_budget", 4)
            self._cfg = dict(cfg)

        def fingerprint(self):
            return dict(implementation="agentinputlab.skillopt.native_IR_environment.v1",
                        dataset=dataset, retriever=corpus, target_backend=to_jsonable(target_backend.fingerprint()),
                        metric=metric, hard_criterion="declared_native_metric_strictly_greater_than_zero",
                        top_k=top_k, per_call_output_tokens=per_call_output_tokens, seed=seed,
                        allow_json_fence=allow_json_fence, partition_label=partition_label,
                        groups={split:[dict(case_id=e.example_id, state_sha256=e.visible_state().state_hash,
                                gold_sha256=content_hash(e.evaluator_gold)) for e in cases]
                                for split,cases in groups.items()},
                        heldout_not_exposed=True, automatic_retry=False,
                        native_reflection_patch_merge_and_validation_gate=True)

        def _batch(self, split, count, batch_seed):
            if type(count) is not int or count < 1 or count > len(groups[split]):
                raise ValidationError("SkillOpt IR batch exceeds the declared group")
            cases = random.Random(batch_seed).sample(list(groups[split]), count)
            return [dict(id=content_hash(e.example_id), question=e.query,
                         task_type=adapter.benchmark_id) for e in cases]

        def build_train_env(self, batch_size, seed, **kwargs):
            return self._batch("train", batch_size, seed)

        def build_eval_env(self, env_num, split, seed, **kwargs):
            if split not in {"val", "validation", "valid_seen"}:
                raise ValidationError("SkillOpt IR fit environment exposes validation only; heldout evaluation is separate")
            return self._batch("validation", env_num, seed)

        def get_task_types(self):
            return [adapter.benchmark_id]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            if not isinstance(skill_content, str) or not skill_content.strip():
                raise ValidationError("SkillOpt IR candidate skill must be nonempty text")
            if any(kwargs.get(name) for name in ("diagnostic_mode", "diagnostic_instruction", "diagnostic_trace_context_by_id")):
                raise ValidationError("SkillOpt IR diagnostic extra attempts require a separate explicit protocol")
            if (to_jsonable(adapter.fingerprint()) != dataset
                    or to_jsonable(retriever.fingerprint()) != corpus):
                raise ValidationError("SkillOpt IR source dataset or corpus changed")
            root = Path(out_dir)
            root.mkdir(parents=True, exist_ok=True)
            results = []
            rows = list(env_manager)
            if any(not isinstance(item, dict) or item.get("id") not in owned for item in rows):
                raise ValidationError("SkillOpt IR rollout contains an undeclared fit case")
            if len({item["id"] for item in rows}) != len(rows):
                raise ValidationError("SkillOpt IR rollout contains duplicate cases")
            for item in rows:
                e = owned[item["id"]]
                if item != dict(id=content_hash(e.example_id), question=e.query, task_type=adapter.benchmark_id):
                    raise ValidationError("SkillOpt IR visible fit item changed")
                folder = root / "predictions" / item["id"]
                folder.mkdir(parents=True, exist_ok=False)
                state = e.visible_state()
                candidate = ContextBlock(
                    text="SKILLOPT CURRENT FITTING SKILL\n" + skill_content,
                    block_id="skillopt-fit-skill:" + content_hash(skill_content),
                    source="SkillOpt native fitting candidate")
                state = replace(state, context=(*state.context, candidate))
                row = dict(id=item["id"], source_case_id=e.example_id, question=e.query,
                           task_type=adapter.benchmark_id, hard=0, soft=0.0, response="",
                           agent_ok=False, n_turns=1, fail_reason="", native_metrics=None)
                predictor = ContextIRQueryPredictor(target_backend, retriever, max_tokens=per_call_output_tokens,
                    top_k=top_k, seed=seed, allow_json_fence=allow_json_fence)
                try:
                    reply = predictor.predict(state)
                    metrics = dict(adapter.score(e, reply.prediction))
                    value = metric_value(metrics)
                    row.update(hard=int(value > 0), soft=value, agent_ok=True, native_metrics=metrics,
                               response=reply.metadata["generation_response"]["text"],
                               endpoint_metadata=to_jsonable(reply.metadata), usage=to_jsonable(reply.usage))
                    if value == 0:
                        row["fail_reason"] = f"native {metric}=0"
                except IRQueryOutputError as exc:
                    if not isinstance(exc.__cause__, AlgorithmOutputError):
                        with (folder / "consumed-execution-error.json").open("x") as f:
                            json.dump(dict(error_type=type(exc).__name__, response=to_jsonable(exc.response),
                                           request=to_jsonable(exc.request), usage=to_jsonable(exc.usage),
                                           automatic_retry=False), f, sort_keys=True, allow_nan=False)
                        raise
                    row.update(response=exc.response.text, agent_ok=True, soft=0.0, hard=0,
                               fail_reason=str(exc), query_or_retrieval_failure=True,
                               generation_request=to_jsonable(exc.request), generation_response=to_jsonable(exc.response),
                               usage=to_jsonable(exc.usage))
                except Exception as exc:
                    with (folder / "execution-error.json").open("x") as f:
                        json.dump(dict(error_type=type(exc).__name__, automatic_retry=False), f)
                    raise
                conversation = [dict(type="message", turn=1, content=row["response"]),
                                dict(role="system", content="[NATIVE IR EVALUATION]\n" +
                                     json.dumps(dict(metric=metric, score=row["soft"],
                                                     valid_native_metric=row["native_metrics"] is not None)))]
                with (folder / "conversation.json").open("x") as f:
                    json.dump(conversation, f, sort_keys=True, indent=2, allow_nan=False)
                with (folder / "result.json").open("x") as f:
                    json.dump(row, f, sort_keys=True, indent=2, allow_nan=False)
                results.append(row)
            with (root / "results.jsonl").open("x") as f:
                for row in results:
                    f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            return results

    return RetrievalEnvAdapter()


__all__ = ["create_skillopt_retrieval_adapter"]
