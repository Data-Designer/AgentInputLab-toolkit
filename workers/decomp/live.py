"""Isolated pinned-native DecomP driver with an explicit host callback seam."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

from agentinputlab.contracts.serialization import content_hash

from . import driver


def failure_response(exc, checkout):
    """Expose source locations/types, never raw exception text or provider bodies."""
    result = {"status":"error","failure_code":exc.code,"message":exc.public_message}
    cause = exc.__cause__
    chain = []
    seen = set()
    # Native Controller.execute wraps participant errors in ValueError using
    # implicit exception context. Keep that source chain, without its messages.
    while cause is not None and id(cause) not in seen and len(chain) < 8:
        seen.add(id(cause))
        frames = []
        for frame in traceback.extract_tb(cause.__traceback__)[-16:]:
            try:
                relative = Path(frame.filename).resolve().relative_to(Path(checkout).resolve())
            except ValueError:
                continue
            if relative.as_posix().startswith("src/decomp/"):
                frames.append({"file":relative.as_posix(),"line":frame.lineno,"function":frame.name})
        chain.append({"error_type":type(cause).__name__, "source_frames":frames})
        cause = (cause.__cause__ if cause.__cause__ is not None else
                 None if cause.__suppress_context__ else cause.__context__)
    cause = exc.__cause__
    if cause is not None:
        result["upstream_error_type"] = chain[0]["error_type"]
        result["upstream_source_frames"] = chain[0]["source_frames"]
        result["upstream_error_chain"] = chain
    return result


class Callback:
    def __init__(self, url, key_env, request_id, checkout, policy):
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None
                or parsed.path != "/callback" or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise driver.DriverError("CONFIGURATION_INVALID", "callback must use numeric loopback")
        if not key_env.isidentifier() or not os.environ.get(key_env):
            raise driver.DriverError("CONFIGURATION_INVALID", "callback credential unavailable")
        self.url, self.key = url, os.environ[key_env]
        self.request_id, self.checkout, self.policy, self.attested = request_id, checkout, policy, False

    def attest(self):
        for name, module in tuple(sys.modules.items()):
            if name.startswith("decomp."):
                location = getattr(module, "__file__", None)
                if location and not Path(location).resolve().is_relative_to(self.checkout.root / "src"):
                    raise driver.DriverError("IMPORT_STATE_UNSAFE", "native module escaped checkout")
        versions = {}
        for name, expected in self.policy["live_required_versions"].items():
            distribution = importlib.metadata.distribution(name)
            if distribution.version != expected:
                raise driver.DriverError("DEPENDENCY_MISMATCH", "native worker dependency version differs")
            import_name = {"jsonnet": "_jsonnet", "python-dateutil": "dateutil"}.get(name, name.replace("-", "_"))
            dependency = importlib.import_module(import_name)
            origin = getattr(dependency, "__file__", None)
            if not origin or not any(Path(distribution.locate_file(file)).resolve() == Path(origin).resolve()
                                     for file in (distribution.files or ())):
                raise driver.DriverError("DEPENDENCY_ORIGIN_MISMATCH", "native dependency import escaped distribution")
            versions[name] = distribution.version
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        value = {"schema": "agentinputlab.decomp.live-attestation.v1", "driver_sha256": sha(Path(driver.__file__)),
            "live_sha256": sha(Path(__file__)), "compatibility_manifest_sha256": self.checkout.manifest_sha256,
            "required_versions": versions,
            "english_stopwords_sha256": sha(Path(os.environ["NLTK_DATA"])/"corpora/stopwords/english"),
            "source_files_verified": True, "import_origins_verified": True}
        self.call("attest", value)
        self.attested = True

    def call(self, kind, payload):
        raw = {"request_id": self.request_id, "call_id": uuid4().hex, "kind": kind, "payload": payload}
        request = urllib.request.Request(self.url, data=json.dumps(raw, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        result, failed = None, False
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=240) as response:
                body = response.read(4 * 1024 * 1024 + 1)
            if len(body) > 4 * 1024 * 1024:
                raise ValueError("callback response byte limit")
            result = json.loads(body)
            if result.get("ok") is not True:
                raise ValueError("host callback rejected")
        except Exception:
            failed = True
        if failed:
            raise driver.DriverError("HOST_CALLBACK_FAILED", "bound native completion failed; no retry")
        return result["payload"]

    def complete(self, **parameters):
        if not self.attested:
            self.attest()
        return self.call("model", parameters)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkout", "callback-url", "callback-key-env"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--planner-component", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.readline(driver.MAX_REQUEST_BYTES + 1)
        if len(raw) > driver.MAX_REQUEST_BYTES or not raw or sys.stdin.buffer.read(1):
            raise driver.DriverError("INVALID_REQUEST", "one bounded JSON line required")
        envelope = json.loads(raw)
        if set(envelope) != {"request_id", "request"} or content_hash(envelope["request"]) != envelope["request_id"]:
            raise driver.DriverError("INVALID_REQUEST", "native request identity differs")
        request = dict(envelope["request"])
        request.pop("execution_budget")
        request.pop("seed")
        native = driver.DecompDriver(environment={driver.UPSTREAM_ENV: args.checkout})
        checkout = native._checkout()
        callback = Callback(args.callback_url, args.callback_key_env, envelope["request_id"], checkout, checkout.manifest)
        native.native_completion = callback.complete
        result = native.run_planner(request) if args.planner_component else native.run(request)
        if not callback.attested:
            raise driver.DriverError("NATIVE_MODEL_NOT_CALLED", "no native model call was made")
        response = {"status": "ok", "payload": result}
    except driver.DriverError as exc:
        response = failure_response(exc, args.checkout)
    except Exception as exc:
        response = {"status": "error", "failure_code": "WORKER_INTERNAL_FAILURE", "error_type": type(exc).__name__}
    sys.stdout.write(json.dumps(response, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
