"""Supervised native DecomP with authenticated, budgeted host completions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import ClassVar

from agentinputlab.algorithms.decompose.decomp.worker_bridge import DecompWorkerRequest
from agentinputlab.backends.llm.decomp_completion import validate_native_parameters
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .base import JsonLineSubprocessWorker
from .llmcompiler_live import _server
from .supervisor import SupervisedWorkerError

_KEY_ENV = "AGENTINPUTLAB_DECOMP_CALLBACK_KEY"
_MODE = "controlled_native_completion_callbacks.v1"


class _Callbacks:
    def __init__(self, owner, request, identity):
        self.owner, self.request, self.identity = owner, request, identity
        self.budget = request.execution_budget
        self.input_bound = getattr(owner, "per_call_input_token_bound", None)
        if (self.budget is None or self.budget.max_cost_usd is not None
                or (self.budget.max_context_tokens is not None and self.input_bound is None)):
            raise ValidationError("live DecomP requires output/call budgets; finite input/cost ceilings need an input bound or pricing estimator")
        if self.input_bound is not None and (type(self.input_bound) is not int or self.input_bound < 1):
            raise ValidationError("DecomP input token bound must be a positive integer")
        if content_hash(request.backend_binding) != content_hash(owner.backend.fingerprint()):
            raise ValidationError("DecomP live backend differs from the context backend")
        self.request_id = content_hash(to_jsonable(request))
        self.events = []
        self.seen = set()
        self.lock = threading.RLock()
        self.attested = self.closed = self.fatal = False
        self.calls = self.input_tokens = self.output_tokens = self.reserved = self.inflight = 0
        self.input_reserved = 0
        self.unknown_usage_calls = 0

    def usage(self):
        return {"model_calls": self.calls, "retrieval_calls": 0, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "unpriced_model_calls": self.calls}

    def dispatch(self, envelope):
        if (not isinstance(envelope, dict) or set(envelope) != {"request_id", "call_id", "kind", "payload"}
                or envelope["request_id"] != self.request_id or not isinstance(envelope["call_id"], str)
                or not envelope["call_id"]):
            raise ValueError("invalid DecomP callback envelope")
        with self.lock:
            if self.closed or self.fatal or envelope["call_id"] in self.seen:
                raise ValueError("closed or duplicate DecomP callback")
            self.seen.add(envelope["call_id"])
        if envelope["kind"] == "attest":
            if self.attested or envelope["payload"] != self.identity["worker_attestation"]:
                raise ValueError("DecomP worker source or dependency attestation differs")
            self.attested = True
            self.events.append({"kind": "attestation", "matched": True})
            return {"attested": True}
        if envelope["kind"] != "model" or not self.attested:
            raise ValueError("DecomP worker must attest before native model calls")
        parameters = validate_native_parameters(envelope["payload"])
        cap = parameters["max_tokens"]
        input_reservation = self.input_bound or 0
        with self.lock:
            if (self.fatal or self.closed or self.calls >= self.budget.max_model_calls
                    or cap > self.owner.per_call_output_cap
                    or self.output_tokens + self.reserved + cap > self.budget.max_output_tokens
                    or (self.budget.max_context_tokens is not None and
                        self.input_tokens + self.input_reserved + input_reservation > self.budget.max_context_tokens)):
                raise ValueError("DecomP native budget exhausted before dispatch")
            self.calls += 1
            self.inflight += 1
            self.reserved += cap
            self.input_reserved += input_reservation
        event = {"kind": "model", "native_parameters": parameters, "seed": self.request.seed}
        self.events.append(event)
        try:
            response = to_jsonable(self.owner.backend.complete_native(parameters, seed=self.request.seed))
            event["response"] = response
            usage = response["usage"]
            counts = (usage["prompt_tokens"], usage["completion_tokens"])
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError("invalid native usage")
            with self.lock:
                self.input_tokens += counts[0]
                self.output_tokens += counts[1]
                self.reserved -= cap
                self.input_reserved -= input_reservation
                event["usage_verified"] = True
            if self.input_bound is not None and counts[0] > self.input_bound:
                raise ValueError("provider exceeded DecomP native input token bound")
            if counts[1] > cap:
                raise ValueError("provider exceeded DecomP native output cap")
            if response.get("model") != self.owner.backend.model_id:
                raise ValueError("native completion returned a different model")
            choices = response["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0]["text"], str):
                raise ValueError("native completion requires exactly one text choice")
            logs = choices[0].get("logprobs")
            if parameters["logprobs"] is not None:
                if not isinstance(logs, Mapping):
                    raise ValueError("native requested log probabilities are missing")
                probs, tokens = logs["token_logprobs"], logs["tokens"]
                if (not isinstance(probs, list) or not isinstance(tokens, list) or len(probs) != len(tokens)
                        or any(type(p) not in (float, int) or not math.isfinite(p) or p > 0 for p in probs)
                        or not all(isinstance(t, str) for t in tokens)):
                    raise ValueError("native token probabilities are invalid")
            return response
        except Exception as exc:
            with self.lock:
                self.fatal = True
                self.unknown_usage_calls += int(not event.get("usage_verified", False))
            event["error_type"] = type(exc).__name__
            raise RuntimeError("bound DecomP completion failed; no retry") from None
        finally:
            with self.lock:
                self.inflight -= 1


@dataclass
class DecompLiveRuntime:
    """Bind the original generator seam without granting child provider credentials.

    Use a dedicated legacy worker interpreter and explicit NLTK data directory.
    Backend I/O must have its own finite timeout. Every run gets an isolated HOME,
    source-checked participants, uncached completions and request-local accounting.
    """

    python_executable: str
    checkout: Path
    nltk_data: Path
    backend: object
    timeout_seconds: float = 300.0
    per_call_output_cap: int = 512
    # A caller-attested upper bound, e.g. the bound provider's context window.
    # Reserve it before each call, then replace it with actual reported input.
    # It never truncates prompts or changes the native generation parameters.
    per_call_input_token_bound: int | None = None
    callback_events: list = field(default_factory=list, init=False, repr=False)
    last_response: dict | None = field(default=None, init=False, repr=False)
    last_resource_receipt: object = field(default=None, init=False, repr=False)
    request_type: ClassVar[type] = DecompWorkerRequest
    worker_id: ClassVar[str] = "agentinputlab.decomp.controlled-live.v1"
    worker_flags: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        if (not isinstance(self.python_executable, str) or not self.python_executable
                or type(self.per_call_output_cap) is not int or self.per_call_output_cap < 1):
            raise ValidationError("DecomP worker interpreter and positive per-call cap required")
        if (type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0):
            raise ValidationError("DecomP finite positive worker timeout required")
        if self.per_call_input_token_bound is not None and (
                type(self.per_call_input_token_bound) is not int or self.per_call_input_token_bound < 1):
            raise ValidationError("DecomP input token bound must be a positive integer or None")
        if not callable(getattr(self.backend, "complete_native", None)) or not callable(getattr(self.backend, "fingerprint", None)):
            raise ValidationError("DecomP requires a source-parameter-preserving completion backend")
        self.checkout, self.nltk_data = Path(self.checkout), Path(self.nltk_data)

    def fingerprint(self):
        package = resources.files("workers").joinpath("decomp")
        policy = package.joinpath("compatibility_manifest.json").read_bytes()
        def sha(raw):
            return hashlib.sha256(raw).hexdigest()
        stopwords = self.nltk_data / "corpora" / "stopwords" / "english"
        attestation = {"schema": "agentinputlab.decomp.live-attestation.v1",
            "driver_sha256": sha(package.joinpath("driver.py").read_bytes()),
            "live_sha256": sha(package.joinpath("live.py").read_bytes()),
            "compatibility_manifest_sha256": sha(policy),
            "required_versions": json.loads(policy)["live_required_versions"],
            "english_stopwords_sha256": sha(stopwords.read_bytes()),
            "source_files_verified": True, "import_origins_verified": True}
        value = {"worker_id": self.worker_id, "profile_kind": "controlled_adapter",
            "upstream_revision": "8b157f6fe9ff5676ae51bd054f1df7b015adf278",
            "backend_binding_mode": _MODE, "worker_attestation": attestation,
            "backend": dict(self.backend.fingerprint()), "per_call_output_cap": self.per_call_output_cap,
            "timeout_seconds": self.timeout_seconds, "automatic_retry": False}
        if self.per_call_input_token_bound is not None:
            value["per_call_input_token_bound"] = self.per_call_input_token_bound
            value["input_budget_accounting"] = "pre_call_upper_bound_then_actual_provider_usage"
        return value

    def run(self, request: DecompWorkerRequest):
        if not isinstance(request, self.request_type):
            raise TypeError("DecompLiveRuntime requires DecompWorkerRequest")
        identity = self.fingerprint()
        callbacks = _Callbacks(self, request, identity)
        self.callback_events, self.last_response, self.last_resource_receipt = callbacks.events, None, None
        key = secrets.token_urlsafe(32)
        server = _server(callbacks, key, 4 * 1024 * 1024)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        raw = to_jsonable(request)
        payload = {"request_id": callbacks.request_id, "request": raw}
        try:
            with tempfile.TemporaryDirectory(prefix="agentinputlab-decomp-worker-") as home:
                worker = JsonLineSubprocessWorker(
                    command=(self.python_executable, "-m", "workers.decomp.live", "--checkout", str(self.checkout),
                        "--callback-url", f"http://127.0.0.1:{server.server_port}/callback", "--callback-key-env", _KEY_ENV,
                        *self.worker_flags),
                    worker_id=identity["worker_id"], identity=identity, timeout_seconds=self.timeout_seconds,
                    environment={"PATH": os.environ.get("PATH", ""), "HOME": home, "NLTK_DATA": str(self.nltk_data),
                        "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", _KEY_ENV: key})
                try:
                    response = dict(worker.call(payload))
                except SupervisedWorkerError:
                    response = {"status": "error", "failure_code": "LIVE_WORKER_EXECUTION_FAILED"}
                finally:
                    self.last_resource_receipt = worker.last_resource_receipt
            with callbacks.lock:
                callbacks.closed = True
                incomplete = callbacks.inflight > 0
                usage = callbacks.usage()
            if incomplete:
                response = {"status": "error", "failure_code": "CALLBACK_USAGE_INCOMPLETE"}
            if response.get("status") == "ok" and response["payload"]["usage"]["model_calls"] != usage["model_calls"]:
                response = {"status": "error", "failure_code": "CALLBACK_USAGE_MISMATCH"}
            response.update(usage=usage, consumed=bool(usage["model_calls"]), callback_diagnostics={"unknown_usage_calls": callbacks.unknown_usage_calls,
                "usage_incomplete": incomplete, "automatic_retry": False})
            self.last_response = response
            if response.get("status") != "ok":
                # The request's external-status hook raises only after the
                # monitoring boundary has charged this complete actual usage.
                return response
            value = response["payload"]
            value["usage"] = usage
            value["native_artifacts"].update(backend_binding_mode=identity["backend_binding_mode"],
                token_accounting="actual_native_provider_usage", provider_cache="disabled_controlled_profile")
            return self._parse_payload(value)
        finally:
            callbacks.closed = True
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def _parse_payload(self, value):
        from workers.decomp.runtime import _parse_response
        return _parse_response(value)
