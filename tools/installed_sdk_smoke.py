"""Exercise an installed npm tarball against the installed Python service.

This is a real cross-language transport/patch test, not model inference or a
benchmark measurement. Run from a directory containing the installed SDK.
"""
import os
import secrets
import subprocess
import threading

from agentinputlab.api import ToolkitService, make_server
from importlib import resources
from workers.official_arenas.retail_capture import _source_pins


def main():
    assert len(_source_pins()) == 7
    assert resources.files("workers.llmcompiler").joinpath("compatibility_manifest.json").is_file()
    token = secrets.token_urlsafe(32)
    server = make_server(ToolkitService(), token=token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
import assert from 'node:assert/strict';
import { AgentInputLab, AgentInputLabError } from '@agentinputlab/sdk';
const client = new AgentInputLab({baseUrl: process.env.AIL_TEST_URL, token: process.env.AIL_TEST_TOKEN});
assert.equal((await client.health()).status, 'ok');
const catalog = await client.catalog();
assert.equal(catalog.methods.length, 27);
assert.equal(catalog.benchmarks.length, 31);
assert.ok(catalog.benchmarks.some(cell => cell.cell_id === 'tooltalk.full'));
const result = await client.intervene({release_id: 'plan.plan_and_solve.ps_prompt', state: {query: 'Find a suitable flight'}, seed: 17});
assert.equal(result.scored, false);
assert.equal(result.benchmark_score, null);
assert.ok(result.state.instruction.length > 0);
await assert.rejects(client.intervene({release_id: 'unconfigured',state: {query: 'test'}}), e => e instanceof AgentInputLabError && e.status === 422);
console.log(JSON.stringify({transport: 'passed', installed_packages: true, benchmark_inference: false}));
'''
    try:
        subprocess.run(["node", "--input-type=module", "-e", script], check=True,
            env={**os.environ, "AIL_TEST_URL": f"http://127.0.0.1:{server.server_port}",
                 "AIL_TEST_TOKEN": token})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
