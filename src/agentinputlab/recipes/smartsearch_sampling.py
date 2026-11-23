"""Pinned native SmartSearch policy sampling for curriculum training data.

ReCall.run/run_prefix, raw prompts and answer/F1 parsers stay upstream-owned.
Only model transport and the fixed Wikipedia-search seam are supplied by the
operator. No arbitrary Python sandbox or whole-trajectory retry is exposed.
"""
from __future__ import annotations

import ast
import json
import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from workers.smartsearch.driver import (
    BudgetGate,
    ModelTurn,
    SearchRecorder,
    SmartSearchDriver,
    _completion_stop,
)

from .qrag import _file, _sha
from .smartsearch import SMARTSEARCH_TOOL_SCHEMA
from .smartsearch_curriculum import SmartSearchTrainingCase, verify_smartsearch_preparation_source

_SOURCE = {
    "src/re_call/inference/re_call.py": "7d3c680cfa0186397c2970d1073d75ab2e587330ce107fdf632827b7857572ec",
    "src/flashrag/pipeline/active_pipeline.py": "20394e3418b6e30b43057ca96b97a423b4e4d37a3a04901e0a0a0dbc3af221fa",
    "src/verl/utils/reward_score/re_call.py": "2697ffa7479a0c9c4e6baf843164b7055b80350c627e01413584ecf4f76476bc",
    "src/flashrag/evaluator/metrics.py": "2fe5c0a592effb9d89e2704387e2b146d45c70e0ce9484d6e67bbb08664db098",
    "src/flashrag/evaluator/utils.py": "d7939725bf0dc4d7d0d425fabb4334ea31c469798ca1b2b794a3504788f27033",
}


@dataclass(frozen=True, slots=True)
class SmartSearchSamplingControls:
    seed: int = 17
    max_model_calls: int = 10
    max_output_tokens: int = 81920
    max_input_tokens: int = 262144
    max_retrieval_calls: int = 10

    def __post_init__(self):
        for name in ("seed", "max_model_calls", "max_output_tokens", "max_input_tokens", "max_retrieval_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValidationError("SmartSearch sampling controls require positive integer bounds")
        if self.max_model_calls > 10 or self.max_retrieval_calls > 10:
            raise ValidationError("SmartSearch native policy loop is at most ten model/search turns")


def verify_smartsearch_sampling_source(checkout):
    binding = dict(verify_smartsearch_preparation_source(checkout))
    root = Path(binding["root"])
    for relative, expected in _SOURCE.items():
        path = _file(root / relative, "SmartSearch native sampling source")
        if not path.is_relative_to(root) or _sha(path) != expected:
            raise ExternalDependencyError("SmartSearch native sampling/parser source changed")
    binding["sampling_source"] = dict(_SOURCE)
    return root, binding


def _native_surface(root, binding):
    # Reuse the existing verified-source loader, NOT the frozen-3B profile or
    # its immutable model identity. No inference manifest is weakened.
    loader = SimpleNamespace(checkout=root, manifest_sha256=content_hash(binding))
    recall, parse_answer = SmartSearchDriver._load_upstream_surface(loader)
    try:
        import jieba
        import langid
    except ImportError as exc:
        raise ExternalDependencyError("Install agentinputlab[smartsearch-sample] for the native multilingual F1 parser") from exc
    namespace = {"re": re, "string": string, "Counter": Counter, "langid": langid, "jieba": jieba}
    utils = ast.parse((root / "src/flashrag/evaluator/utils.py").read_text())
    normalize = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == "normalize_answer")
    metrics = ast.parse((root / "src/flashrag/evaluator/metrics.py").read_text())
    owner = next(n for n in metrics.body if isinstance(n, ast.ClassDef) and n.name == "F1_Score")
    score = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "token_level_scores")
    tree = ast.Module(body=[normalize, score], type_ignores=[])
    exec(compile(tree, str(root / "src/flashrag/evaluator/metrics.py"), "exec"), namespace)
    return recall, parse_answer, loader._upstream_module, namespace["token_level_scores"]


class SmartSearchPolicySampler:
    """Generate new native trajectories, then score against training references.

    Model: fingerprint(), count_prompt_tokens(prompt), generate(prompt,
    temperature=..., max_new_tokens=..., stop=tuple(...), seed=...) -> ModelTurn.
    Retriever: fingerprint(), search(query, top_n=5) -> native id/contents/score
    records. Its corpus_sha256/index_sha256 must identify a frozen real closure.
    Source native8192/turn and ten-turn limits are separately metered TRAINING
    costs, not a successful512/4096 test-budget claim. References reach only
    the scorer after generation, never the provider, tool or raw prompt.
    """

    def __init__(self, model, retriever, *, controls=None):
        self.controls = SmartSearchSamplingControls(**controls) if isinstance(controls, Mapping) else controls or SmartSearchSamplingControls()
        if not isinstance(self.controls, SmartSearchSamplingControls):
            raise ValidationError("SmartSearch policy sampler requires typed controls")
        for value, methods in ((model, ("fingerprint", "count_prompt_tokens", "generate")),
                               (retriever, ("fingerprint", "search"))):
            if any(not callable(getattr(value, name, None)) for name in methods):
                raise ValidationError("SmartSearch sampling runtime lacks an explicit native transport seam")
        self.model, self.retriever = model, retriever

    def sample(
        self, *, checkout, case: SmartSearchTrainingCase, golden_answers: Sequence[str],
        output_root, prefix: str | None = None, prefix_provenance: Mapping | None = None,
    ):
        if not isinstance(case, SmartSearchTrainingCase) or case.split not in {"train", "validation"}:
            raise ValidationError("SmartSearch policy construction samples only declared train/development cases")
        if not isinstance(golden_answers, (tuple, list)) or not golden_answers or any(
            not isinstance(a, str) or not a.strip() for a in golden_answers
        ):
            raise ValidationError("SmartSearch native post-generation scoring requires real reference answers")
        if prefix is not None and (not isinstance(prefix, str) or not prefix.strip() or not prefix_provenance):
            raise ValidationError("SmartSearch prefix regeneration requires a nonempty native prefix and provenance")
        source, binding = verify_smartsearch_sampling_source(checkout)
        recall, parser, module, scorer = _native_surface(source, binding)
        model_identity = to_jsonable(self.model.fingerprint())
        retrieval_identity = to_jsonable(self.retriever.fingerprint())
        for key in ("corpus_sha256", "index_sha256"):
            digest = retrieval_identity.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
                raise ValidationError("SmartSearch training retrieval requires frozen corpus/index hashes")
        root = Path(output_root)
        if root.exists() or root.is_symlink():
            raise ValidationError("SmartSearch sampling output must be create-only; no automatic retry")
        root = root.resolve(strict=False)
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValidationError("SmartSearch sampling output must not overlap native source")
        root.mkdir(parents=True, exist_ok=False)

        def save(name, value):
            with (root / name).open("x", encoding="utf-8") as stream:
                json.dump(to_jsonable(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)

        save("claim.json", {"case": case, "source": binding, "controls": self.controls,
                            "model": model_identity, "retrieval": retrieval_identity,
                            "golden_answers_sha256": content_hash(golden_answers),
                            "prefix_sha256": content_hash(prefix), "prefix_provenance": prefix_provenance,
                            "training_only": True, "no_automatic_retry": True})
        gate = BudgetGate({"model_calls": self.controls.max_model_calls,
                           "output_tokens": self.controls.max_output_tokens,
                           "retrieval_calls": self.controls.max_retrieval_calls,
                           "context_tokens": self.controls.max_input_tokens})
        recorder = SearchRecorder(self.retriever, gate, retrieval_identity)
        responses = 0

        class ModelSeam:
            def post(seam, url, json):
                nonlocal responses
                sampling = json.get("sampling_params")
                stops = ["</search>"] if prefix is not None else ["</search>", "</answer>"]
                if url != "agentinputlab://model/generate" or sampling != {
                    "temperature": 0.0, "max_new_tokens": 8192, "stop": stops,
                } or set(json) != {"text", "sampling_params"}:
                    raise ExternalDependencyError("SmartSearch native sampling transport contract changed")
                if to_jsonable(self.model.fingerprint()) != model_identity:
                    raise ExternalDependencyError("SmartSearch sampled model identity changed")
                prompt = json["text"]
                admitted = self.model.count_prompt_tokens(prompt)
                if type(admitted) is not int or admitted < 1:
                    raise ValidationError("SmartSearch sampling needs exact tokenizer admission")
                if gate.usage.output_tokens + 8192 > self.controls.max_output_tokens:
                    raise BudgetExceededError("SmartSearch sampling cannot admit the original8192 token cap")
                gate.before_model(admitted)
                number = gate.usage.model_calls
                save(f"call-{number:04d}-request.json", {"prompt": prompt, **sampling, "seed": self.controls.seed})
                turn = self.model.generate(prompt, temperature=0.0, max_new_tokens=8192,
                                           stop=tuple(stops), seed=self.controls.seed)
                if not isinstance(turn, ModelTurn) or any(type(n) is not int or n < 0 for n in (turn.input_tokens, turn.output_tokens)):
                    raise ExternalDependencyError("SmartSearch sampled completion lacks typed measured usage")
                save(f"call-{number:04d}-response.json", turn)
                responses += 1
                gate.after_model(turn.input_tokens, turn.output_tokens, admitted_input_tokens=admitted)
                if turn.output_tokens > 8192:
                    raise BudgetExceededError("SmartSearch sampler model exceeded native per-call cap")
                if to_jsonable(self.model.fingerprint()) != model_identity:
                    raise ExternalDependencyError("SmartSearch sampled model identity changed during generation")
                text, matched = _completion_stop(turn.text, {"finish_reason": turn.finish_reason, "stop_reason": turn.matched})
                return SimpleNamespace(json=lambda: {"text": text,
                    "meta_info": {"finish_reason": {"type": turn.finish_reason, "matched": matched}}})

        class FixedToolReCall(recall):
            def execute_tool_calls(agent, env, tool_calls):
                if env != "agentinputlab:no-exec" or to_jsonable(self.retriever.fingerprint()) != retrieval_identity:
                    raise ExternalDependencyError("SmartSearch training fixed retrieval identity changed")
                result = recorder.execute(tool_calls)
                if to_jsonable(self.retriever.fingerprint()) != retrieval_identity:
                    raise ExternalDependencyError("SmartSearch training retrieval identity changed during search")
                return result

        try:
            module.requests = ModelSeam()
            agent = FixedToolReCall("agentinputlab://model", "agentinputlab://disabled-executor")
            arguments = (agent, "agentinputlab:no-exec", json.dumps(to_jsonable(SMARTSEARCH_TOOL_SCHEMA)), case.question)
            transcript = recall.run.__wrapped__(*arguments) if prefix is None else recall.run_prefix.__wrapped__(*arguments, prefix)
            prediction = parser(transcript)
            metrics = scorer(None, prediction, list(golden_answers))
            for relative, expected in _SOURCE.items():
                if _sha(_file(source / relative, "sampling source")) != expected:
                    raise ExternalDependencyError("SmartSearch sampling source changed during execution")
            row = {"question": case.question, "golden_answers": list(golden_answers),
                   "output": {"final_response": transcript, "pred": prediction, "metric_score": metrics}}
            save("rollout.json", row)
            manifest = {"case_id": case.case_id, "split": case.split, "source": binding,
                        "model": model_identity, "retrieval": retrieval_identity,
                        "usage": gate.usage.as_mapping(), "model_responses": responses,
                        "unknown_output_token_reservations": 0,
                        "rollout_sha256": _sha(root / "rollout.json"),
                        "trajectory": recorder.trajectory, "evidence": recorder.evidence,
                        "prefix_provenance": prefix_provenance, "seed_adapter": self.controls.seed,
                        "native_loop_prompt_parser_scorer_called": True,
                        "unsafe_sandbox_used": False, "neural_training_performed": False,
                        "scored_is_not_correctness_or_full_curriculum_success": True}
            save("manifest.json", manifest)
            return manifest
        except Exception as exc:
            save("failed.json", {"error_type": type(exc).__name__, "error": str(exc),
                                 "usage": gate.usage.as_mapping(), "model_responses": responses,
                                 "unknown_output_token_reservations": (gate.usage.model_calls - responses) * 8192,
                                 "trajectory": recorder.trajectory, "evidence": recorder.evidence,
                                 "no_automatic_retry": True})
            raise
