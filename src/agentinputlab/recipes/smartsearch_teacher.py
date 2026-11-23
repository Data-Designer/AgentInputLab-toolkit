"""Native SmartSearch training-only process rewards and query refinement.

The pinned upstream prompts, fourth-think selection, parser and redundancy
rules run unchanged. Only the HTTP method is replaced by an operator-supplied,
metered completion provider. No generated code or upstream sandbox is run.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from .smartsearch_curriculum import (
    SmartSearchTrainingCase,
    _native_converter,
    validate_smartsearch_training_splits,
    verify_smartsearch_preparation_source,
)


@dataclass(frozen=True, slots=True)
class SmartSearchTeacherCompletion:
    text: str
    input_tokens: int
    output_tokens: int
    model_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or any(
            type(n) is not int or n < 0 for n in (self.input_tokens, self.output_tokens)
        ):
            raise ValidationError("SmartSearch teacher completion requires measured token counts")
        digest = self.model_fingerprint_sha256
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValidationError("SmartSearch teacher model identity must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SmartSearchTeacherControls:
    max_model_calls: int = 8
    max_total_output_tokens: int = 4096
    max_rollout_rows: int = 1000
    max_input_bytes: int = 16 * 1024 * 1024
    seed: int = 17

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (
            self.max_model_calls, self.max_total_output_tokens, self.max_rollout_rows,
            self.max_input_bytes,
        )) or type(self.seed) is not int or self.seed < 0:
            raise ValidationError("SmartSearch teacher controls must be positive integer limits")


class SmartSearchProcessTeacher:
    """Annotate real training rollouts before native SFT/DPO conversion.

    ``generate`` receives the exact native raw prompt, temperature0.6,
    max_new_tokens2048 and native stop list, plus the explicit operator seed.
    This is a separately metered TRAINING helper, not a 512-token test call.
    Budget shortage fails before a call rather than silently shrinking2048.
    Provider exceptions consume an unknown-usage reservation and stop the stage;
    neither the upstream empty-string fallback nor an automatic retry is used.
    """

    def __init__(
        self, generate: Callable[..., SmartSearchTeacherCompletion], *,
        model_fingerprint_sha256: str,
        controls: SmartSearchTeacherControls | None = None,
    ) -> None:
        if not callable(generate):
            raise ValidationError("SmartSearch teacher requires a completion provider")
        SmartSearchTeacherCompletion("", 0, 0, model_fingerprint_sha256)
        self.generate = generate
        self.model_fingerprint_sha256 = model_fingerprint_sha256
        self.controls = controls or SmartSearchTeacherControls()

    def annotate(
        self, *, checkout: str | Path, output_root: str | Path,
        train: Sequence[SmartSearchTrainingCase], validation: Sequence[SmartSearchTrainingCase],
        sampled_rollouts: Sequence[Mapping[str, Any]],
        rollout_provenance: Mapping[str, Any],
        test: Sequence[SmartSearchTrainingCase] = (), refine: bool = True,
    ) -> Mapping[str, Any]:
        audit = validate_smartsearch_training_splits(train, validation, test)
        if type(refine) is not bool or not isinstance(rollout_provenance, Mapping) or not rollout_provenance:
            raise ValidationError("SmartSearch teacher needs explicit rollout provenance/refinement choice")
        if not isinstance(sampled_rollouts, (list, tuple)) or not sampled_rollouts or (
            len(sampled_rollouts) > self.controls.max_rollout_rows
        ):
            raise ValidationError("SmartSearch teacher rollout count exceeds its bound")
        allowed = {case.question for case in train}
        for row in sampled_rollouts:
            if not isinstance(row, Mapping) or row.get("question") not in allowed:
                raise ValidationError("SmartSearch teacher may only label training questions")
            answers, output = row.get("golden_answers"), row.get("output")
            if not isinstance(answers, (list, tuple)) or not answers or any(
                not isinstance(a, str) or not a.strip() for a in answers
            ) or not isinstance(output, Mapping) or not isinstance(output.get("final_response"), str):
                raise ValidationError("SmartSearch teacher needs native sampled rollout fields")
            metric = output.get("metric_score")
            score = metric.get("f1") if isinstance(metric, Mapping) else None
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValidationError("SmartSearch teacher requires a measured native outcome F1")
            if "pred" not in output:
                raise ValidationError("SmartSearch teacher needs the recorded native prediction")
        inputs = {"rollouts": to_jsonable(sampled_rollouts),
                  "producer_provenance": to_jsonable(rollout_provenance)}
        encoded = json.dumps(inputs, allow_nan=False, ensure_ascii=False).encode()
        if len(encoded) > self.controls.max_input_bytes:
            raise ValidationError("SmartSearch teacher input exceeds its byte bound")
        rows = json.loads(encoded)["rollouts"]
        source = verify_smartsearch_preparation_source(checkout, include_teacher=True)
        source_root = Path(source["root"])
        root = Path(output_root)
        if root.is_symlink() or root.exists():
            raise ValidationError("SmartSearch teacher output must be create-only")
        root = root.resolve(strict=False)
        if root.is_relative_to(source_root) or source_root.is_relative_to(root):
            raise ValidationError("SmartSearch teacher output must not overlap native source")
        root.mkdir(parents=True, exist_ok=False)

        def save(name: str, value: Any) -> None:
            with (root / name).open("x", encoding="utf-8") as stream:
                json.dump(to_jsonable(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)

        usage = {"model_attempts": 0, "model_responses": 0, "input_tokens": 0,
                 "output_tokens": 0, "unknown_output_token_reservations": 0}
        save("claim.json", {"source": source, "split_audit": audit,
                            "input_sha256": content_hash(inputs), "controls": self.controls,
                            "model_fingerprint_sha256": self.model_fingerprint_sha256,
                            "native_helper_max_new_tokens": 2048, "native_temperature": 0.6,
                            "seed_adapter": self.controls.seed, "training_only": True})

        def complete(prompt: str, *, refinement: bool) -> str:
            cap = 2048
            if usage["model_attempts"] >= self.controls.max_model_calls or (
                usage["output_tokens"] + usage["unknown_output_token_reservations"] + cap
                > self.controls.max_total_output_tokens
            ):
                raise BudgetExceededError("SmartSearch teacher budget cannot admit the native2048-token cap")
            usage["model_attempts"] += 1
            usage["unknown_output_token_reservations"] += cap
            number = usage["model_attempts"]
            request = {"prompt": prompt, "temperature": 0.6, "max_new_tokens": cap,
                       "stop": ["</search>"] if refinement else [], "seed": self.controls.seed,
                       "training_only": True, "operation": "refinement" if refinement else "process_reward"}
            save(f"call-{number:04d}-request.json", request)
            response = self.generate(prompt, temperature=0.6, max_new_tokens=cap,
                                     stop=tuple(request["stop"]), seed=self.controls.seed)
            if not isinstance(response, SmartSearchTeacherCompletion):
                raise ExternalDependencyError("SmartSearch teacher provider omitted typed measured usage")
            usage["model_responses"] += 1
            usage["input_tokens"] += response.input_tokens
            usage["output_tokens"] += response.output_tokens
            usage["unknown_output_token_reservations"] -= cap
            save(f"call-{number:04d}-response.json", response)
            if response.model_fingerprint_sha256 != self.model_fingerprint_sha256:
                raise ExternalDependencyError("SmartSearch teacher model identity changed")
            if response.output_tokens > cap:
                raise BudgetExceededError("SmartSearch teacher provider exceeded its requested token cap")
            # Preserve upstream's exact append behavior, including malformed text;
            # no parser cleanup, replacement reward or second attempt is made.
            return response.text + ("</search>" if refinement else "")

        try:
            reward = _native_converter(source_root, "reward", source)
            reward.generate_response = lambda prompt: complete(prompt, refinement=False)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                scored = reward.evaluate_search(rows)
                scored = _native_converter(source_root, "redundancy", source).detect_redundancy(scored)
                if refine:
                    rewriter = _native_converter(source_root, "refinement", source)
                    rewriter.generate_response = lambda prompt: complete(prompt, refinement=True)
                    scored = rewriter.rewrite_search(scored)
            for relative, digest in source["files"].items():
                if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != digest:
                    raise ExternalDependencyError("SmartSearch teacher source changed during execution")
            save("scored-rollouts.json", scored)
            result = {"schema": "agentinputlab.smartsearch.native-process-teacher.v1",
                      "source": source, "split_audit": audit, "controls": self.controls,
                      "rollout_provenance": rollout_provenance, "usage": usage,
                      "input_rollouts": len(rows), "native_selected_rollouts": len(scored),
                      "refinement_enabled": refine, "training_only": True,
                      "neural_training_performed": False, "native_prompts_parsers_selection_unchanged": True,
                      "transport_adapter": "operator_provider_with_seed_and_usage_no_http_fallback",
                      "model_fingerprint_sha256": self.model_fingerprint_sha256,
                      "scored_rollouts_sha256": hashlib.sha256((root / "scored-rollouts.json").read_bytes()).hexdigest()}
            save("manifest.json", result)
            return freeze_mapping({**result, "root": str(root)})
        except Exception as exc:
            save("failed.json", {"error_type": type(exc).__name__, "error": str(exc),
                                 "usage": usage, "no_automatic_retry": True,
                                 "neural_training_performed": False})
            raise
