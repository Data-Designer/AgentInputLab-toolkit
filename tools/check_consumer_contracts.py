"""Audit installed user interfaces without model calls or dataset downloads.

Run from an unpacked source distribution after installing the wheel. Optional
--sdk-module points to the built SDK index.js (prefer a clean npm installation).
Reports are small, create-only JSON; they are not scientific acceptance results.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections import Counter
from importlib.metadata import distribution
from pathlib import Path


def audit(sdk_module: Path | None = None) -> dict:
    import agentinputlab
    from agentinputlab import FrozenAgentState, InterventionContext
    from agentinputlab.api import (
        ToolkitService,
        create_intervention,
        describe_intervention,
        make_server,
        run_intervention,
    )
    from agentinputlab.benchmarks import describe_benchmark, list_benchmarks
    from agentinputlab.registry import AlgorithmRegistry

    checks = []

    def check(name, operation):
        try:
            operation()
            checks.append({"check": name, "passed": True})
        except Exception as error:
            # Do not serialize arbitrary exception text, URLs, prompts or credentials.
            checks.append({"check": name, "passed": False, "error_type": type(error).__name__})

    def command(arguments, expected=0, env=None):
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=30, env=env)
        assert result.returncode == expected
        assert "Traceback" not in result.stderr
        return result.stdout

    registry = AlgorithmRegistry.default()
    profiles = registry.list_releases()
    for _method, profile in profiles:
        def inspect_profile(profile=profile):
            details = describe_intervention(profile.id)
            assert details["id"] == profile.id
            assert isinstance(details["parameters"], list)
            json.dumps(details, allow_nan=False)
            if not profile.entrypoint_available:
                assert details["blocker"]
        check("profile discovery: " + profile.id, inspect_profile)

    datasets = list_benchmarks()
    for dataset in datasets:
        def inspect_dataset(dataset=dataset):
            details = describe_benchmark(dataset["cell_id"])
            assert details["cell_id"] == dataset["cell_id"]
            assert isinstance(details["automatic_loader"], bool)
            assert details["loader_kind"]
            json.dumps(details, allow_nan=False)
        check("dataset discovery: " + dataset["cell_id"], inspect_dataset)

    for entry in distribution("agentinputlab").entry_points:
        if entry.group != "console_scripts":
            continue
        # Execute the installed entrypoint with the selected interpreter, even
        # when the environment's bin directory is absent from PATH.
        program = (
            "from importlib.metadata import distribution; "
            f"e=next(e for e in distribution('agentinputlab').entry_points if e.name=={entry.name!r}); "
            "raise SystemExit(e.load()())"
        )
        check("CLI help: " + entry.name,
              lambda program=program: command([sys.executable, "-c", program, "--help"]))

    for args in (["--version"], ["doctor", "--offline", "--json"], ["smoke", "--json"]):
        check("CLI " + " ".join(args),
              lambda args=args: command([sys.executable, "-m", "agentinputlab", *args]))
    no_token = {k: v for k, v in os.environ.items() if k != "AGENTINPUTLAB_API_TOKEN"}
    check("CLI missing server token is actionable", lambda: command(
        [sys.executable, "-m", "agentinputlab.api", "--port", "0"], 2, no_token))
    check("CLI invalid server port is actionable", lambda: command(
        [sys.executable, "-m", "agentinputlab.api", "--port", "70000"], 2, no_token))

    def prompt(release):
        state = FrozenAgentState(query="Consumer protocol check")
        result = run_intervention(create_intervention(release), state, InterventionContext(seed=17))
        assert result["state"]["query"] == state.query
        assert result["scored"] is False and result["benchmark_score"] is None
        assert result["record"]["usage"]["model_calls"] == 0
        if release != "baseline.no_op":
            assert result["state"]["instruction"]
        json.dumps(result, allow_nan=False)

    for release in ["baseline.no_op", "plan.plan_and_solve.ps_prompt", "plan.plan_and_solve.ps_plus_prompt"]:
        check("Python prompt execution: " + release, lambda release=release: prompt(release))

    def configuration():
        from agentinputlab import RunConfig

        raw = command([sys.executable, "-m", "agentinputlab", "config", "init"])
        with tempfile.TemporaryDirectory(prefix="ail-config-check-") as directory:
            path = Path(directory) / "run.json"
            path.write_text(raw, encoding="utf-8")
            parsed = json.loads(command([sys.executable, "-m", "agentinputlab",
                                         "config", "validate", str(path)]))
            assert parsed["scope"] == "configuration_only" and parsed["runtime_ready"] is None
            result = json.loads(command([sys.executable, "-m", "agentinputlab", "config",
                                         "run", str(path), "--query", "consumer config check"]))
            assert result["scored"] is False and result["record"]["usage"]["model_calls"] == 0
            assert RunConfig.from_file(path).to_dict() == json.loads(raw)
            path.write_text('{"release_id":"baseline.no_op","budegt":{}}', encoding="utf-8")
            command([sys.executable, "-m", "agentinputlab", "config", "validate", str(path)], 2)

    check("editable configuration CLI/Python roundtrip and typo rejection", configuration)

    def nested_parameters():
        details = describe_intervention("rewrite.rrr.common_support_prompt")
        fields = next(p["fields"] for p in details["parameters"] if p["name"] == "config")
        assert {f["name"] for f in fields} >= {"temperature", "max_reader_tokens", "top_k"}

    check("nested method configuration discovery", nested_parameters)

    def installed_guides():
        from agentinputlab.documentation import list_guides, read_guide

        topics = list_guides()
        assert {"getting-started", "configuration", "acceptance-coverage", "troubleshooting"} <= set(topics)
        for topic in topics:
            assert read_guide(topic).strip()
        listing = json.loads(command([sys.executable, "-m", "agentinputlab", "docs", "--list", "--json"]))
        assert listing["topics"] == topics and listing["offline"] is True
        guide = json.loads(command([sys.executable, "-m", "agentinputlab", "docs", "configuration", "--json"]))
        assert guide["content"] == read_guide("configuration")
        command([sys.executable, "-m", "agentinputlab", "docs", "../../etc/passwd"], 2)

    check("all installed offline guides and safe CLI topic selection", installed_guides)

    token = "consumer-contract-fixture-not-a-secret"
    service = make_server(ToolkitService(), token=token, port=0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{service.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, status, body=None, supplied_token=token, content_type="application/json"):
        req = urllib.request.Request(base + path, data=body, headers={
            "Authorization": "Bearer " + supplied_token, "Content-Type": content_type,
        })
        try:
            response = opener.open(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            assert response.status == status
            payload = json.load(response)
            assert isinstance(payload, dict)
            assert response.headers["Cache-Control"] == "no-store"
            if status >= 400:
                assert set(payload["error"]) == {"code", "message"}
            elif path == "/v1/interventions":
                assert payload["data"]["scored"] is False
                assert payload["data"]["benchmark_score"] is None

    try:
        check("HTTP authenticated health", lambda: request("/v1/health", 200))
        check("HTTP catalog", lambda: request("/v1/catalog", 200))
        check("HTTP unauthorized", lambda: request("/v1/health", 401, supplied_token="wrong"))
        check("HTTP unknown route", lambda: request("/absent", 404))
        valid = b'{"release_id":"baseline.no_op","state":{"query":"x"},"seed":17}'
        for label, body, code in [
            ("patch", valid, 200),
            ("missing state", b'{"release_id":"baseline.no_op"}', 400),
            ("malformed JSON", b'{', 400),
            ("unknown profile", b'{"release_id":"absent","state":{"query":"x"}}', 422),
            ("duplicate query", b'{"release_id":"baseline.no_op","state":{"query":"x","query":"y"}}', 400),
            ("nonfinite number", b'{"release_id":"baseline.no_op","state":{"query":"x","metadata":{"x":1e999}}}', 400),
            ("invalid budget", b'{"release_id":"baseline.no_op","state":{"query":"x","budget":{"max_model_calls":-1}}}', 400),
        ]:
            check("HTTP " + label, lambda body=body, code=code: request("/v1/interventions", code, body))
        check("HTTP media type", lambda: request("/v1/interventions", 415, valid, content_type="text/plain"))
        def oversized():
            # Send only headers: early rejection need not consume an oversized
            # body (a streaming sender may otherwise see a deliberate disconnect).
            connection = http.client.HTTPConnection("127.0.0.1", service.server_port, timeout=5)
            try:
                connection.putrequest("POST", "/v1/interventions")
                connection.putheader("Authorization", "Bearer " + token)
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", str(1024 * 1024 + 1))
                connection.endheaders()
                response = connection.getresponse()
                assert response.status == 413
                assert json.load(response)["error"]["code"] == "body_size"
            finally:
                connection.close()
        check("HTTP oversized declared body", oversized)
        if sdk_module is not None:
            def sdk():
                script = """
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const {AgentInputLab}=await import(pathToFileURL(process.argv[1]).href);
const client=new AgentInputLab({baseUrl:process.argv[2],token:process.argv[3]});
assert.equal((await client.health()).protocol,'agentinputlab.v1');
assert.equal((await client.catalog()).methods.length,27);
const result=await client.intervene({release_id:'plan.plan_and_solve.ps_prompt',state:{query:'consumer'},seed:17});
assert.equal(result.scored,false);assert.match(result.state.instruction,/devise a plan/);
const bad=new AgentInputLab({baseUrl:process.argv[2],token:'wrong'});
await assert.rejects(bad.health(),e=>e.status===401&&e.code==='unauthorized');
console.log('SDK-to-Python live loopback transport passed; no model calls');
"""
                command(["node", "--input-type=module", "-e", script, str(sdk_module.resolve()), base, token])
            check("installed SDK to Python actual HTTP roundtrip", sdk)
    finally:
        service.shutdown()
        service.server_close()
        thread.join()

    return {
        "schema": "agentinputlab.consumer-contracts.v1",
        "version": agentinputlab.__version__, "python": sys.version.split()[0],
        "installed_module": str(Path(agentinputlab.__file__).resolve()),
        "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
        "checks": checks,
        "inventory": {"methods": len(registry.list_methods()), "profiles": len(profiles),
                      "entrypoints": sum(p.entrypoint_available for _, p in profiles),
                      "declared_availability": dict(Counter(p.availability for _, p in profiles)),
                      "datasets": len(datasets), "automatic_loaders": sum(d["automatic_loader"] for d in datasets)},
        "real_model_calls": 0, "dataset_downloads": 0,
        "runtime_compatibility": "not_tested_by_this_audit",
        "sdk_transport": "tested" if sdk_module is not None else "not_requested",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional create-only JSON receipt outside source tree")
    parser.add_argument("--sdk-module", type=Path, help="Built/installed SDK dist/index.js for actual HTTP check")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run without -O: this checker requires assertions")
    if args.report is not None and args.report.exists():
        parser.error("report already exists; retain it and choose a new path")
    result = audit(args.sdk_module)
    if args.report is not None:
        content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(content)
        print(json.dumps({"report": str(args.report), "sha256": hashlib.sha256(content.encode()).hexdigest()}))
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, sort_keys=True))
    for row in result["checks"]:
        if not row["passed"]:
            print(json.dumps(row))
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
