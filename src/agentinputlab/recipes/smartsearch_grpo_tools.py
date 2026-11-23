"""Fixed retrieval and fail-closed native reward seams for SmartSearch GRPO.

This is not an arbitrary-code sandbox and does not implement a new reward.
The original pinned GRPO scorer owns all reward arithmetic. Only its helper
transport is replaced; failed requests cannot become empty positive rewards.
"""
from __future__ import annotations

import ast
import hashlib
import hmac
import json
import math
import re
import string
import threading
from collections import Counter
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Union

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from workers.smartsearch.driver import BudgetGate, SearchRecorder

from .smartsearch_sampling import verify_smartsearch_sampling_source
from .smartsearch_teacher import SmartSearchTeacherCompletion, SmartSearchTeacherControls

SMARTSEARCH_GRPO_ENVIRONMENT = "agentinputlab:smartsearch-fixed-wikipedia-v1"
_REWARD_PATH = "src/verl/utils/reward_score/re_call.py"
_REWARD_SHA = "2697ffa7479a0c9c4e6baf843164b7055b80350c627e01413584ecf4f76476bc"


def _save(root, name, value):
    # Encode before opening so serialization failures do not leave fake JSON.
    encoded = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    with (root / name).open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _new_output(output_root):
    root = Path(output_root)
    if root.is_symlink() or root.exists():
        raise ValidationError("SmartSearch GRPO seam output must be create-only")
    root = root.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=False)
    return root


class SmartSearchGRPORetrievalService:
    """Native /execute payload, but only one literal Wikipedia search is allowed.

    The environment is a fixed identifier, never Python. Native rollout code
    still formats its call; an AST allowlist parses it without exec/eval.
    Each request, including repeated queries, counts separately. Any service
    failure latches unhealthy: an executor must call assert_healthy before
    accepting a trained checkpoint. HTTP errors are never successful retrieval.
    """

    def __init__(self, retriever, *, output_root, max_retrieval_calls: int):
        if type(max_retrieval_calls) is not int or max_retrieval_calls < 1:
            raise ValidationError("SmartSearch GRPO retrieval requires a positive request quota")
        if any(not callable(getattr(retriever, name, None)) for name in ("search", "fingerprint")):
            raise ValidationError("SmartSearch GRPO retrieval requires a frozen retrieval provider")
        identity = to_jsonable(retriever.fingerprint())
        for key in ("corpus_sha256", "index_sha256"):
            digest = identity.get(key)
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValidationError("SmartSearch GRPO retrieval requires corpus/index hashes")
        self.root = _new_output(output_root)
        self.retriever, self.identity, self.limit = retriever, identity, max_retrieval_calls
        self._lock = threading.Lock()
        self.attempts = self.responses = 0
        self.failed = False
        _save(self.root, "claim.json", {"retrieval": identity, "max_retrieval_calls": self.limit,
              "environment": SMARTSEARCH_GRPO_ENVIRONMENT, "arbitrary_code_execution": False})

    @staticmethod
    def _query(body):
        if not isinstance(body, dict) or set(body) != {"env", "call"} or (
            body["env"] != SMARTSEARCH_GRPO_ENVIRONMENT
        ) or not isinstance(body["call"], str) or len(body["call"].encode()) > 8192:
            raise ValidationError("SmartSearch GRPO requires its fixed environment and one bounded search call")
        try:
            call = ast.parse(body["call"], mode="eval").body
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ValidationError("SmartSearch GRPO search call is not a literal expression") from exc
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or (
            call.func.id != "wikipedia_search" or call.args or len(call.keywords) != 2
        ) or {kw.arg for kw in call.keywords} != {"query", "top_n"} or any(
            not isinstance(kw.value, ast.Constant) for kw in call.keywords
        ):
            raise ValidationError("Only wikipedia_search(query=<string>, top_n=5) is supported")
        values = {kw.arg: kw.value.value for kw in call.keywords}
        query = values["query"]
        if not isinstance(query, str) or not query.strip() or len(query.encode()) > 4096 or (
            type(values["top_n"]) is not int or values["top_n"] != 5
        ):
            raise ValidationError("SmartSearch GRPO search requires a nonempty bounded query and top_n=5")
        return query

    def execute(self, body):
        try:
            query = self._query(body)  # Rejected syntax does not consume a retrieval.
        except ValidationError:
            self.reject_transport("invalid_native_request")
            raise
        with self._lock:
            if self.failed:
                raise ExternalDependencyError("SmartSearch GRPO retrieval has a retained failure; no automatic retry")
            if self.attempts >= self.limit:
                self.failed = True
                raise BudgetExceededError("SmartSearch GRPO retrieval request quota exhausted")
            if to_jsonable(self.retriever.fingerprint()) != self.identity:
                self.failed = True
                raise ExternalDependencyError("SmartSearch GRPO retrieval identity changed")
            self.attempts += 1
            number = self.attempts
            _save(self.root, f"request-{number:06d}.json", body)
            try:
                recorder = SearchRecorder(self.retriever, BudgetGate({"retrieval_calls": 1}), self.identity)
                recorder.execute([query])
                if to_jsonable(self.retriever.fingerprint()) != self.identity:
                    raise ExternalDependencyError("SmartSearch GRPO retrieval identity changed during search")
                result = {"result": "\n\n".join(row["text"] for row in recorder.evidence),
                          "output": "", "error": ""}
                _save(self.root, f"response-{number:06d}.json", {
                    "native_response": result, "trajectory": recorder.trajectory, "evidence": recorder.evidence})
                self.responses += 1
                return result
            except Exception as exc:
                self.failed = True
                _save(self.root, f"failed-{number:06d}.json", {"error_type": type(exc).__name__,
                      "error": str(exc), "attempt_consumed": True, "no_automatic_retry": True})
                raise

    def reject_transport(self, code):
        """An authenticated failed native request invalidates this training run."""
        with self._lock:
            if not self.failed:
                self.failed = True
                _save(self.root, "transport-rejected.json", {
                    "code": code, "retrieval_attempt_consumed": False,
                    "no_automatic_retry": True, "accept_checkpoint": False})

    def assert_healthy(self):
        with self._lock:
            if self.failed or self.responses != self.attempts:
                raise ExternalDependencyError("SmartSearch GRPO retrieval has an unresolved/failed request")
            return {"retrieval_attempts": self.attempts, "retrieval_responses": self.responses,
                    "retrieval_identity_sha256": content_hash(self.identity)}


def make_smartsearch_grpo_retrieval_server(service, *, capability: str, port: int = 0):
    """Loopback-only, ephemeral-port service compatible with native /execute.

    Set native sandbox_url to http://127.0.0.1:<port>/<capability>. The random
    operator capability belongs in private run config, not a published log.
    No bearer-header modification to upstream is necessary. Route/error/body
    limits apply before retrieval; server.close does not cancel scientific work.
    """
    if not isinstance(service, SmartSearchGRPORetrievalService) or not isinstance(capability, str) or (
        not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", capability)
    ):
        raise ValidationError("SmartSearch GRPO server requires a fixed service and random ASCII capability")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValidationError("SmartSearch GRPO server port is invalid")
    capacity = threading.BoundedSemaphore(8)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass

        def send_json(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def error(self, status, code):
            if getattr(self, "authenticated", False):
                service.reject_transport("http_" + code)
            self.send_json(status, {"error": {"code": code, "message": "Training tool request rejected"}})

        def do_POST(self):
            parts = self.path.split("/")
            if len(parts) != 3 or not hmac.compare_digest(parts[1].encode(), capability.encode()):
                self.error(401, "unauthorized")
                return
            self.authenticated = True
            if parts[2] != "execute":
                self.error(404, "not_found")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
                self.error(400, "invalid_body")
                return
            length = int(lengths[0])
            if not 0 < length <= 16384:
                self.error(413, "body_size")
                return
            if self.headers.get_content_type() != "application/json":
                self.error(415, "content_type")
                return
            if not capacity.acquire(blocking=False):
                self.error(429, "busy")
                return
            try:
                def unique(pairs):
                    value = {}
                    for key, item in pairs:
                        if key in value:
                            raise ValueError("Duplicate JSON key")
                        value[key] = item
                    return value

                body = json.loads(self.rfile.read(length), object_pairs_hook=unique,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                result = service.execute(body)
                self.send_json(200, result)
            except (ValueError, UnicodeDecodeError, RecursionError):
                self.error(400, "invalid_request")
            except BudgetExceededError:
                self.error(429, "quota_exhausted")
            except Exception:
                self.error(502, "retrieval_failed")
            finally:
                capacity.release()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


class SmartSearchNativeGRPOReward:
    """Call the original reward math with a metered, non-retrying helper.

    References are explicit TRAIN-only inputs to this instance, never rollout
    prompts. Validation requires a separate scoring instance with its own
    development references/cost ledger; neither instance accepts arbitrary
    test references. The native scorer receives unchanged solution/ground truth.
    Quotas are per instance/process; an execution coordinator must account for
    every rank/helper, not multiply an instance's allowance implicitly.
    """

    def __init__(self, generate, *, checkout, references: Mapping,
                 model_fingerprint_sha256: str, controls: SmartSearchTeacherControls,
                 output_root, reference_split: str = "train"):
        if reference_split not in {"train", "validation"} or not callable(generate) or (
            not isinstance(controls, SmartSearchTeacherControls)
        ) or not isinstance(references, Mapping) or not references:
            raise ValidationError("SmartSearch GRPO reward requires explicit training/development reference controls")
        clean = {}
        for question, answers in references.items():
            if not isinstance(question, str) or not question.strip() or not isinstance(answers, (list, tuple)) or (
                not answers or any(not isinstance(a, str) or not a.strip() for a in answers)
            ):
                raise ValidationError("SmartSearch GRPO reference question/answer collection is invalid")
            # The pinned scorer extracts the user's question with .strip().
            # Match that reference key only; do not rewrite rollout/data text.
            key = question.strip()
            if key in clean:
                raise ValidationError("SmartSearch GRPO reference questions collide after native whitespace extraction")
            clean[key] = tuple(answers)
        SmartSearchTeacherCompletion("", 0, 0, model_fingerprint_sha256)
        source, binding = verify_smartsearch_sampling_source(checkout)
        path = source / _REWARD_PATH
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != _REWARD_SHA:
            raise ExternalDependencyError("SmartSearch native GRPO reward source changed")
        self.source_path, self.source_sha = path, _REWARD_SHA
        output = Path(output_root).resolve(strict=False)
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValidationError("SmartSearch GRPO reward output must not overlap native source")
        self.root = _new_output(output_root)
        self.generate, self.controls = generate, controls
        self.references, self.split = clean, reference_split
        self.fingerprint = model_fingerprint_sha256
        self._lock = threading.Lock()
        self.failed = False
        self.usage = {"score_attempts": 0, "score_responses": 0, "model_attempts": 0,
                      "model_responses": 0, "input_tokens": 0, "output_tokens": 0,
                      "unknown_output_token_reservations": 0}
        tree = ast.parse(raw)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != "generate_response"]
        namespace = {"re": re, "string": string, "Counter": Counter, "Union": Union, "List": list,
                     "generate_response": self._complete}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self._score = namespace["compute_score_with_format"]
        _save(self.root, "claim.json", {"source": binding, "native_reward_sha256": _REWARD_SHA,
              "reference_split": reference_split, "references_sha256": content_hash(clean),
              "model_fingerprint_sha256": self.fingerprint, "controls": controls,
              "native_reward_math_changed": False, "failed_http_empty_fallback_disabled": True,
              "neural_optimizer_started": False})

    def _complete(self, prompt):
        usage, controls, cap = self.usage, self.controls, 2048
        if usage["model_attempts"] >= controls.max_model_calls or (
            usage["output_tokens"] + usage["unknown_output_token_reservations"] + cap > controls.max_total_output_tokens
        ):
            raise BudgetExceededError("SmartSearch GRPO helper cannot admit its native2048-token cap")
        usage["model_attempts"] += 1
        usage["unknown_output_token_reservations"] += cap
        number = usage["model_attempts"]
        _save(self.root, f"helper-{number:06d}-request.json", {"prompt": prompt, "temperature": .6,
              "max_new_tokens": cap, "stop": ["</answer>"], "seed": controls.seed,
              "reference_split": self.split})
        result = self.generate(prompt, temperature=.6, max_new_tokens=cap, stop=("</answer>",), seed=controls.seed)
        if not isinstance(result, SmartSearchTeacherCompletion):
            raise ExternalDependencyError("SmartSearch GRPO helper requires typed measured usage")
        usage["model_responses"] += 1
        usage["input_tokens"] += result.input_tokens
        usage["output_tokens"] += result.output_tokens
        usage["unknown_output_token_reservations"] -= cap
        _save(self.root, f"helper-{number:06d}-response.json", result)
        if result.model_fingerprint_sha256 != self.fingerprint:
            raise ExternalDependencyError("SmartSearch GRPO helper model identity changed")
        if result.output_tokens > cap:
            raise BudgetExceededError("SmartSearch GRPO helper exceeded the original per-call cap")
        if not result.text.strip():
            raise ExternalDependencyError("SmartSearch GRPO helper returned empty text; no positive reward fabricated")
        return result.text + "</answer>"  # Exact native GRPO helper append, unlike SFT process teacher.

    def __call__(self, tokenizer, solution_str, ground_truth):
        if not isinstance(solution_str, str) or len(solution_str.encode()) > self.controls.max_input_bytes:
            raise ValidationError("SmartSearch GRPO reward solution exceeds its explicit input bound")
        match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        answers = (ground_truth,) if isinstance(ground_truth, str) else tuple(ground_truth) if isinstance(ground_truth, (list, tuple)) else ()
        if match is None or self.references.get(match.group(1).strip()) != answers:
            raise ValidationError("SmartSearch GRPO reward references differ from the declared split")
        ground_truth = to_jsonable(ground_truth)
        with self._lock:
            if self.failed:
                raise ExternalDependencyError("SmartSearch GRPO reward has a retained failure; no retry")
            if self.usage["score_attempts"] >= self.controls.max_rollout_rows:
                raise BudgetExceededError("SmartSearch GRPO reward score quota exhausted")
            self.usage["score_attempts"] += 1
            number = self.usage["score_attempts"]
            _save(self.root, f"score-{number:06d}-request.json", {
                "solution_str": solution_str, "ground_truth": ground_truth, "reference_split": self.split})
            try:
                if hashlib.sha256(self.source_path.read_bytes()).hexdigest() != self.source_sha:
                    raise ExternalDependencyError("SmartSearch GRPO native reward source changed")
                score = self._score(tokenizer, solution_str, ground_truth)
                if hashlib.sha256(self.source_path.read_bytes()).hexdigest() != self.source_sha:
                    raise ExternalDependencyError("SmartSearch GRPO native reward source changed during scoring")
                if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                    raise ExternalDependencyError("SmartSearch GRPO native scorer returned a nonfinite reward")
                self.usage["score_responses"] += 1
                _save(self.root, f"score-{number:06d}-response.json", {"reward": score, "usage": self.usage})
                return score
            except Exception as exc:
                self.failed = True
                _save(self.root, f"score-{number:06d}-failed.json", {
                    "error_type": type(exc).__name__, "error": str(exc), "usage": self.usage,
                    "no_automatic_retry": True, "native_reward_math_changed": False})
                raise
