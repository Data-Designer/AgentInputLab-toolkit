"""Pinned native GRPO query refinement with measured, non-retrying transport."""
from __future__ import annotations
import __future__

import ast
import re
import threading
from collections.abc import Mapping
from pathlib import Path

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .qrag import _sha
from .smartsearch_grpo_data import verify_smartsearch_grpo_source
from .smartsearch_grpo_tools import _new_output, _save
from .smartsearch_teacher import SmartSearchTeacherCompletion, SmartSearchTeacherControls

_PATH = "src/verl/workers/rollout/vllm_rollout/generate_new_prompts.py"


class SmartSearchNativeGRPOPrefixBuilder:
    """Execute the native tensor/parser/refinement functions without raw HTTP.

    This TRAIN-only instance binds explicit references. Both native helper
    operations use their original 2048-token, temperature0.6, NO-stop request;
    unlike offline curriculum refinement, no closing tag is appended here.
    Quotas are per instance. A distributed executor must enforce an aggregate
    helper quota rather than silently allocating this allowance to every rank.
    """

    def __init__(self, generate, *, checkout, references: Mapping, model_fingerprint_sha256,
                 controls: SmartSearchTeacherControls, output_root):
        if not callable(generate) or not isinstance(controls, SmartSearchTeacherControls) or (
            not isinstance(references, Mapping) or not references
        ):
            raise ValidationError("SmartSearch GRPO prefixes require explicit TRAIN references and provider")
        clean = {}
        for question, answers in references.items():
            if not isinstance(question, str) or not question.strip() or not isinstance(answers, (list, tuple)) or (
                not answers or any(not isinstance(a, str) or not a.strip() for a in answers)
            ) or question.strip() in clean:
                raise ValidationError("SmartSearch GRPO prefix references are invalid or collide")
            clean[question.strip()] = tuple(answers)
        SmartSearchTeacherCompletion("", 0, 0, model_fingerprint_sha256)
        source, binding = verify_smartsearch_grpo_source(checkout)
        path = source / _PATH
        output = Path(output_root).resolve(strict=False)
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValidationError("SmartSearch GRPO prefix output must not overlap native source")
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise ExternalDependencyError("Install the native SmartSearch GRPO torch runtime") from exc
        self.root = _new_output(output_root)
        self.generate, self.controls, self.references = generate, controls, clean
        self.fingerprint, self.failed = model_fingerprint_sha256, False
        self._lock = threading.Lock()
        self.usage = dict(prefix_attempts=0, prefix_responses=0, prefixes_produced=0,
                          model_attempts=0, model_responses=0, input_tokens=0, output_tokens=0,
                          unknown_output_token_reservations=0)
        tree = ast.parse(path.read_bytes())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "generate_response"]
        namespace = dict(re=re, torch=torch, F=functional, generate_response=self._complete)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        self._native = namespace["generate_new_prompts"]
        self.source_path, self.source_sha = path, _sha(path)
        _save(self.root, "claim.json", dict(native_source=binding, native_prefix_sha256=self.source_sha,
            references_sha256=content_hash(clean), reference_split="train", controls=controls,
            model_fingerprint_sha256=self.fingerprint, native_tensor_parser_selection_unchanged=True,
            native_helper_stop=[], no_closing_tag_append=True, no_http_empty_fallback=True))

    def _complete(self, prompt):
        usage, cap = self.usage, 2048
        if not isinstance(prompt, str) or len(prompt.encode()) > self.controls.max_input_bytes:
            raise ValidationError("SmartSearch GRPO prefix helper input exceeds its byte bound")
        if usage['model_attempts'] >= self.controls.max_model_calls or (
            usage['output_tokens'] + usage['unknown_output_token_reservations'] + cap
            > self.controls.max_total_output_tokens
        ):
            raise BudgetExceededError("SmartSearch GRPO prefix helper cannot admit its native2048-token cap")
        usage['model_attempts'] += 1
        usage['unknown_output_token_reservations'] += cap
        number = usage['model_attempts']
        _save(self.root, f"helper-{number:06d}-request.json", dict(prompt=prompt, temperature=.6,
            max_new_tokens=cap, stop=[], seed=self.controls.seed, reference_split="train"))
        result = self.generate(prompt, temperature=.6, max_new_tokens=cap, stop=(), seed=self.controls.seed)
        if not isinstance(result, SmartSearchTeacherCompletion):
            raise ExternalDependencyError("SmartSearch GRPO prefix helper requires typed measured usage")
        usage['model_responses'] += 1
        usage['input_tokens'] += result.input_tokens
        usage['output_tokens'] += result.output_tokens
        usage['unknown_output_token_reservations'] -= cap
        _save(self.root, f"helper-{number:06d}-response.json", result)
        if result.model_fingerprint_sha256 != self.fingerprint:
            raise ExternalDependencyError("SmartSearch GRPO prefix helper model identity changed")
        if result.output_tokens > cap:
            raise BudgetExceededError("SmartSearch GRPO prefix helper exceeded the requested token cap")
        return result.text

    def __call__(self, prompts, tokenizer, ground_truth):
        with self._lock:
            if self.failed:
                raise ExternalDependencyError("SmartSearch GRPO prefix failed; no automatic helper retry")
            try:
                if self.usage['prefix_attempts'] >= self.controls.max_rollout_rows:
                    raise BudgetExceededError("SmartSearch GRPO prefix row quota exhausted")
                if _sha(self.source_path) != self.source_sha:
                    raise ExternalDependencyError("SmartSearch native prefix source changed")
                trajectory = tokenizer.decode(prompts.batch['input_ids'][0])
                if len(trajectory.encode()) > self.controls.max_input_bytes:
                    raise ValidationError("SmartSearch GRPO prefix trajectory exceeds its byte bound")
                match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', trajectory, re.DOTALL)
                answers = (ground_truth,) if isinstance(ground_truth, str) else tuple(ground_truth)
                if not match or self.references.get(match.group(1).strip()) != answers:
                    raise ValidationError("SmartSearch GRPO prefix input is not its bound TRAIN reference")
                self.usage['prefix_attempts'] += 1
                number = self.usage['prefix_attempts']
                _save(self.root, f"prefix-{number:06d}-request.json", dict(
                    trajectory_sha256=content_hash(trajectory), ground_truth_sha256=content_hash(answers)))
                result = self._native(prompts, tokenizer, ground_truth)
                if not isinstance(result, tuple) or len(result) != 2 or len(result[0]) != len(result[1]):
                    raise ExternalDependencyError("SmartSearch native prefix tensor collection differs")
                self.usage['prefix_responses'] += 1
                self.usage['prefixes_produced'] += len(result[0])
                _save(self.root, f"prefix-{number:06d}-response.json", dict(prefix_count=len(result[0]),
                    shapes=[[list(t.shape) for t in values] for values in result], usage=self.usage))
                return result
            except BaseException as exc:
                self.failed = True
                _save(self.root, "failed.json", dict(error_type=type(exc).__name__, error=str(exc),
                      usage=self.usage, no_automatic_retry=True, accept_checkpoint=False))
                raise

    def assert_healthy(self):
        with self._lock:
            if self.failed or self.usage['prefix_attempts'] != self.usage['prefix_responses'] or (
                self.usage['model_attempts'] != self.usage['model_responses']
                or self.usage['unknown_output_token_reservations']
            ):
                raise ExternalDependencyError("SmartSearch GRPO prefix has an unresolved/failed request")
            return dict(self.usage)
