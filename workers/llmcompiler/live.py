"""Explicit controlled live model/tool callback transport for pinned LLMCompiler.

Only a startup-bound authenticated numeric-loopback endpoint is reachable. No
provider key, request-selected URL, import, shell command, or network tool exists
inside this worker. Native upstream planning/scheduling/join/replan is retained.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping
from types import SimpleNamespace
from uuid import uuid4

from . import driver


def runtime_attestation():
    from pathlib import Path
    manifest = driver._verify_static_manifests()
    return {"schema": "agentinputlab.llmcompiler.live-attestation.v1",
        "driver_sha256": driver._sha256_bytes(Path(driver.__file__).read_bytes()),
        "live_driver_sha256": driver._sha256_bytes(Path(__file__).read_bytes()),
        "compatibility_manifest_sha256": driver.COMPATIBILITY_MANIFEST_SHA256,
        "required_versions": {name: importlib.metadata.version(name)
            for name in manifest["legacy_dependencies"]["required_versions"]},
        "source_files_verified": True, "required_versions_verified": True, "import_origins_verified": True}


def load_prompts(path, request):
    value, _ = driver._load_json(path, label="LLMCompiler live prompt bundle")
    driver._expect_keys(value, {"schema_version", "upstream_commit", "prompts"}, label="prompt bundle")
    prompts = value["prompts"]
    keys = {"planner_example_prompt", "planner_example_prompt_replan", "planner_stop",
            "joinner_prompt", "joinner_prompt_final"}
    if (value["schema_version"] != "agentinputlab.llmcompiler.prompts.v1"
            or value["upstream_commit"] != driver.UPSTREAM_COMMIT
            or not isinstance(prompts, Mapping) or set(prompts) != keys
            or driver._canonical_hash(prompts) != request.prompt_hash
            or not all(isinstance(prompts[k], str) for k in keys - {"planner_stop"})
            or not isinstance(prompts["planner_stop"], list)
            or not all(isinstance(s, str) and s for s in prompts["planner_stop"])):
        raise driver.WorkerFailure("DEPENDENCY_MISMATCH", "live prompt bundle identity differs")
    return driver.ReplayBundle(prompts=dict(prompts), entries=())


class Callback:
    def __init__(self, url, key_env, request):
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None
                or parsed.path != "/callback" or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise driver.WorkerFailure("CONFIGURATION_INVALID", "callback URL must be numeric IPv4 loopback")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key_env) or not os.environ.get(key_env):
            raise driver.WorkerFailure("CONFIGURATION_INVALID", "callback credential unavailable")
        self.url, self.key, self.request = url, os.environ[key_env], request

    async def call(self, kind, payload):
        value = {"request_id": self.request.request_id, "call_id": uuid4().hex,
                 "kind": kind, "payload": payload}

        def send():
            request = urllib.request.Request(self.url, data=driver._canonical_json(value).encode(),
                headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                        request, timeout=self.request.controls.timeout_seconds) as response:
                    raw = response.read(driver._MAX_MANIFEST_BYTES + 1)
                if len(raw) > driver._MAX_MANIFEST_BYTES:
                    raise ValueError("callback response byte limit")
                result = json.loads(raw)
                if not isinstance(result, dict) or result.get("ok") is not True:
                    raise ValueError("callback failed")
                return result["payload"]
            except Exception:
                raise driver.WorkerFailure("HOST_CALLBACK_FAILED", "bound host callback failed; no retry",
                    consumed=True) from None

        return await asyncio.to_thread(send)


class LiveModel(driver._ReplayModel):
    def __init__(self, callback):
        super().__init__(())
        self.callback = callback

    async def complete(self, channel, prompt, stop):
        index = len(self.consumed)
        self.consumed.append(driver.ReplayEntry(channel, driver._sha256_text(prompt), "", 0, 0))
        value = await self.callback.call("model", {"prompt": prompt, "channel": channel,
            "stop": list(stop or ()), "seed": self.callback.request.seed})
        if (not isinstance(value, dict) or set(value) != {"text", "input_tokens", "output_tokens", "finish_reason"}
                or not isinstance(value["text"], str)):
            raise driver.WorkerFailure("BACKEND_OUTPUT_INVALID", "invalid live model response", consumed=True)
        entry = driver.ReplayEntry(channel, driver._sha256_text(prompt), value["text"],
            driver._nonnegative_int(value["input_tokens"], label="input_tokens"),
            driver._nonnegative_int(value["output_tokens"], label="output_tokens"))
        self.consumed[index] = entry
        if not entry.response.strip():
            raise driver.WorkerFailure("BACKEND_OUTPUT_INVALID", "live model returned empty final text", consumed=True)
        if channel == "planner":
            self.planner_responses.append(entry.response)
        return entry.response

    async def apredict(self, prompt, *, callbacks=None, stop=None, **kwargs):
        response = await self.complete("planner", prompt, stop)
        # This is buffered completion delivery, never native streaming parity.
        await self._notify(callbacks, response)
        return response

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        if len(prompts) != 1:
            raise driver.WorkerFailure("BACKEND_OUTPUT_INVALID", "one join prompt required")
        value = prompts[0]
        prompt = value.text if hasattr(value, "text") else value.to_string()
        response = await self.complete("join", prompt, stop)
        return SimpleNamespace(generations=[[SimpleNamespace(text=response)]])

    def assert_exhausted(self):
        # There is no frozen response count in an actual live execution.
        pass


class LiveTools:
    def __init__(self, callback):
        self.callback, self.calls = callback, 0

    async def invoke(self, name, arguments):
        self.calls += 1
        value = await self.callback.call("tool", {"name": name, "arguments": list(arguments)})
        if not isinstance(value, dict) or set(value) != {"output"}:
            raise driver.WorkerFailure("TOOL_EXECUTION_FAILURE", "invalid bound tool response", consumed=True)
        return value["output"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkout", "prompt-bundle", "callback-url", "callback-key-env"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    request_id = ""
    try:
        from pathlib import Path
        driver.verify_checkout(Path(args.checkout))
        raw = driver._read_request_line()
        if isinstance(raw, Mapping):
            request_id = raw.get("request_id", "")
        request = driver.validate_request(raw)
        bundle = load_prompts(Path(args.prompt_bundle), request)
        callback = Callback(args.callback_url, args.callback_key_env, request)
        upstream = driver.import_upstream(Path(args.checkout))
        asyncio.run(callback.call("attest", runtime_attestation()))
        response = asyncio.run(driver.execute_request(request, upstream, bundle, LiveTools(callback),
            runtime_model=LiveModel(callback)))
    except driver.WorkerFailure as exc:
        response = driver._failure_response(exc, request_id=request_id)
    except Exception as exc:
        response = driver._failure_response(driver.WorkerFailure("WORKER_INTERNAL_FAILURE",
            f"live worker failed closed: {type(exc).__name__}"), request_id=request_id)
    sys.stdout.write(driver._canonical_json(response) + "\n")
    sys.stdout.flush()
    # Preserve typed errors and usage for the supervised host, not only stderr.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
