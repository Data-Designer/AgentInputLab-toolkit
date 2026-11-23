"""Local-only bounded relay for observing every native training HTTP attempt.

Native training code retains its own retry policy. Every physical forwarded
attempt, including SDK retries, shares the same cap and has a durable receipt.
This helper is for operator acceptance, not a public or authenticated API.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from run_static_acceptance import save


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect forbidden", headers, fp)


class BoundedFitRelay:
    def __init__(self, *, model, root, tokenizer, max_attempts=32, max_tokens=200000):
        parsed = urllib.parse.urlsplit(model["base_url"])
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username
            or parsed.password
        ):
            raise ValueError("acceptance relay requires a local unauthenticated model service")
        if set(model.get("extra_body", {})) - {"chat_template_kwargs", "reasoning_effort"}:
            raise ValueError("relay backbone controls cannot override native budgets or messages")
        self.model, self.root, self.tokenizer = model, root, tokenizer
        self.max_attempts, self.max_tokens = max_attempts, max_tokens
        self.attempts = self.tokens = self.unknown_usage_attempts = self.rejected = 0
        self.provider_identity_mismatch = False
        self.input_tokens = self.output_tokens = 0
        self.lock = threading.Lock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/v1/chat/completions" or not 0 < length <= 8 * 1024**2:
                        raise ValueError("unsupported path or body size")
                    payload = json.loads(self.rfile.read(length))
                    status, body = relay.forward(payload)
                except (ValueError, KeyError, TypeError) as exc:
                    status, body = 400, json.dumps({"error": str(exc)}).encode()
                self.send(status, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.root.mkdir()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def summary(self):
        with self.lock:
            return {
                "physical_provider_attempts": self.attempts,
                "observed_input_tokens": self.input_tokens,
                "observed_output_tokens": self.output_tokens,
                "charged_tokens": self.tokens,
                "unknown_usage_attempts": self.unknown_usage_attempts,
                "rejected_before_model": self.rejected,
                "max_physical_attempts": self.max_attempts,
                "max_total_tokens": self.max_tokens,
                "per_call_output_cap": 512,
                "seed": 17,
                "native_retries_share_caps": True,
                "provider_identity_mismatch": self.provider_identity_mismatch,
            }

    def forward(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("native request must be one JSON object")
        if payload.get("model") != self.model["id"] or payload.get("stream", False):
            raise ValueError("wrong serving model or streaming request")
        caps = [payload[name] for name in ("max_tokens", "max_completion_tokens") if name in payload]
        if len(caps) != 1:
            raise ValueError("native training request must declare one output cap")
        cap = caps[0]
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= 512:
            raise ValueError("native training request exceeds fixed 512 output cap")
        if "seed" in payload and payload["seed"] != 17:
            raise ValueError("native training changed the declared seed")
        forwarded = {**payload, **self.model.get("extra_body", {}), "seed": 17}
        messages = forwarded["messages"]
        if not isinstance(messages, list):
            raise ValueError("native messages must be a list")
        # The actual server usage remains authoritative. This tokenizer-bound
        # reservation protects dispatch; unobserved errors retain the reservation.
        prompt_tokens = len(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **forwarded.get("chat_template_kwargs", {}),
            )
        )
        reservation = prompt_tokens + cap
        with self.lock:
            if (
                self.provider_identity_mismatch
                or self.attempts >= self.max_attempts
                or self.tokens + reservation > self.max_tokens
            ):
                self.rejected += 1
                return (
                    400,
                    b'{"error":"native fit aggregate budget exhausted before provider dispatch"}',
                )
            self.attempts += 1
            number = self.attempts
            self.tokens += reservation
        path = self.root / f"{number:04d}"
        path.mkdir()
        save(
            path / "request.json",
            {
                "payload": forwarded,
                "prompt_token_reservation": prompt_tokens,
                "output_token_reservation": cap,
                "started": time.time(),
                "authorization_forwarded": False,
            },
        )
        started = time.monotonic()
        status, body = 502, b'{"error":"local provider transport failed"}'
        observation = {}
        try:
            request = urllib.request.Request(
                self.model["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(forwarded).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.opener.open(request, timeout=180) as response:
                status, body = response.status, response.read(16 * 1024**2 + 1)
            if len(body) > 16 * 1024**2:
                raise ValueError("provider response exceeds acceptance byte limit")
            observation["response"] = json.loads(body)
            if observation["response"].get("model") != self.model["id"]:
                with self.lock:
                    self.provider_identity_mismatch = True
                observation["error"] = "actual provider serving alias differs from fixed model"
                status, body = 502, b'{"error":"provider serving identity mismatch"}'
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read(1024 * 1024)
            observation.update(
                error_type=type(exc).__name__,
                error=str(exc),
                response_body=body.decode(errors="replace"),
            )
        except Exception as exc:
            observation.update(error_type=type(exc).__name__, error=str(exc))
        usage = observation.get("response", {}).get("usage", {})
        incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
        known = all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (incoming, outgoing)
        )
        with self.lock:
            if known:
                self.input_tokens += incoming
                self.output_tokens += outgoing
                self.tokens += incoming + outgoing - reservation
            else:
                self.unknown_usage_attempts += 1
        save(
            path / "response.json",
            {
                **observation,
                "http_status": status,
                "seconds": time.monotonic() - started,
                "usage_observed": known,
                "unknown_usage_reservation_retained": not known,
            },
        )
        return status, body
