# LLMCompiler pinned worker

This directory contains a controlled external worker for the official
[`SqueezeAILab/LLMCompiler`](https://github.com/SqueezeAILab/LLMCompiler) implementation at
commit `a00c9d35507507da70e8c637eee64efc8c1857ae` (MIT). It is not a reimplementation of the
algorithm. The worker imports and invokes the upstream `Planner`, `TaskFetchingUnit`, and
`LLMCompiler` join/replan loop.

## Implementation level

- `plan.llm_compiler.planner_component` calls upstream `Planner.plan` or `Planner.aplan`, then
  returns the parsed dependency DAG without executing tools.
- `plan.llm_compiler.full` calls upstream `LLMCompiler._acall`; scheduling remains based on the
  upstream `TaskFetchingUnit`, with a traced subclass providing bounds and fail-closed error
  handling.
- The replay entry point uses SHA-256-bound artifacts; the explicit live entry point
  calls the host's bound model and read-only tools through authenticated loopback callbacks.
- Both releases therefore default to `profile_kind=controlled_adapter`. A full run may be labeled
  `official_native` only for an explicitly allowlisted worker whose fingerprint contains the
  complete source, dependency-version, import-origin, and native-transport attestation.

The live adapter preserves the one-request/one-response supervised process channel
and uses a separate, per-call authenticated callback channel. It does not give the
worker provider credentials, arbitrary URLs, commands, or side-effecting tools.
The worker verifies its source/dependency environment and attests its installed
driver hashes before the host permits any model or tool call. Install matching
AgentInputLab distributions in the host and worker environments.

## Integrity and compatibility

[`compatibility_manifest.json`](compatibility_manifest.json) pins the repository revision, MIT
license hash, relevant upstream source hashes, legacy dependency versions, and execution policy.
[`patch_manifest.json`](patch_manifest.json) separately records every worker-local compatibility,
safety, and observability patch and whether it changes algorithm logic.

The pinned checkout must be clean and match every recorded hash. The upstream release expects the
legacy dependency set headed by `langchain==0.2.6`; install that environment separately from the
AgentInputLab core environment. Before importing upstream code, the worker verifies **every**
distribution/version in `legacy_dependencies.required_versions` and checks that each imported
top-level package originates under that distribution. Missing, mismatched, or shadowed packages
produce typed dependency failures rather than silently falling back to substitute code.

Create a separate Python 3.10 worker environment (the validated legacy runtime)
and install `agentinputlab[llmcompiler]` there.
These legacy pins need not constrain the host application's model clients.

Check checkout and import readiness with:

```bash
python workers/llmcompiler/driver.py \
  --checkout /path/to/LLMCompiler \
  --smoke-import
```

## JSONL execution

Normal execution reads exactly one canonical `MethodWorkerRequest` JSON object from stdin and
writes exactly one response object to stdout:

```bash
python workers/llmcompiler/driver.py \
  --checkout /path/to/LLMCompiler \
  --model-replay /path/to/model-replay.json \
  --tool-replay /path/to/tool-replay.json
```

The request's backend fingerprint must contain the exact model replay file SHA-256 as
`replay_manifest_sha256`; its tools fingerprint must similarly contain
`tool_manifest_sha256`. Each model response binds its channel (`planner` or `join`), exact prompt
SHA-256, response SHA-256, and token counts. Each tool entry must exactly match the declared tool
schema and be marked `read_only: true`, `network: false`, and `implementation: replay_table`.
Every invocation must match a predeclared positional-argument case; a missing case fails closed.

## Live Python API

Export the pinned upstream HotpotQA prompt template without executing its config code:

```bash
python -m workers.llmcompiler.export_prompts \
  --checkout /path/to/LLMCompiler --template gpt --output compiler-prompts.json
```

Then bind your existing `LLMBackend` and read-only/no-network `ToolConnector`:

```python
import json
from pathlib import Path
from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.decompose.llm_compiler import LLMCompilerControls, LLMCompilerFullBridge
from agentinputlab.contracts import FrozenAgentState
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers import LLMCompilerLiveWorker
from agentinputlab.runner import CompositeInterventionRunner

prompts = Path("compiler-prompts.json")
worker = LLMCompilerLiveWorker(
    python_executable="/path/to/worker-env/bin/python",
    checkout=Path("/path/to/LLMCompiler"), prompt_bundle=prompts,
    backend=backend, tools=tools, per_call_output_cap=512,
)
method = LLMCompilerFullBridge(
    prompt_hash=content_hash(json.loads(prompts.read_text())["prompts"]),
    controls=LLMCompilerControls(streaming=False),
)
record = CompositeInterventionRunner().run(
    method, FrozenAgentState(query="Your question", budget=ExecutionBudget()),
    InterventionContext(backend=backend, tools=tools, external_runtime=worker, seed=17),
)
print(record.result.final_answer)
```

For a planner-only intervention, use `LLMCompilerPlannerBridge` with the same
bound worker, prompt hash and context, and call it through `InterventionRunner`.
Apply `record.patch` to the exact source state, then let your downstream agent
consume the added task DAG. This component executes neither planned tools nor
the joiner; it does not produce the full agent's final answer. Its model usage
must be deducted before the downstream agent runs.

`tools/run_llmcompiler_planner_acceptance.py` demonstrates a separate QA reader
and paired baseline on fixed prepared cases. The planner's query-only projection
is explicit; the reader regains the original visible instruction/history plus
the parsed DAG. It does not infer gold support/evidence labels from that DAG.
The model/reader calls share one output/call budget. No full retrieval corpus,
model download, generated-plan execution or automatic retry is part of the run.

The native interface accepts one query, not arbitrary instruction/history state.
Live callbacks enforce the state's model-call, tool-call and generated-token
budgets before dispatch. Finite input-token and priced-cost ceilings are rejected
until a pre-call estimator is bound; they are never silently ignored. Callback
backends/tools must configure their own bounded I/O timeouts. A worker timeout
with an outstanding callback is an explicit incomplete-usage failure, not zero cost.
The process resource receipt covers the isolated source process, not host callback CPU.

For multi-parameter tools, pass an explicit positional binding such as
`argument_order={"lookup": ("name", "year")}`. Dispatch and DAG traces use this
order rather than JSON property order. Buffered completion callbacks support the
upstream streaming parser but do not establish provider-streaming latency/parity.

`controls.timeout_seconds` bounds the complete run, including planner construction, planning,
scheduling, tools, join, and replan. Streaming planner exit always publishes a queue sentinel;
planner/tool tasks are cancelled and awaited on all exits. Each task batch is also validated
atomically before it reaches the upstream scheduler, so unknown dependencies, cycles, malformed
IDs, and task-limit violations fail before that batch executes.

Native replans use global task IDs while creating a new scheduler per round.
The traced scheduler keeps completed observations available for dependency
lookup within that invocation. Completed tasks are not rerun, included in a new
round's scratchpad twice, or shared between requests. This compatibility repair
does not rewrite plans, model responses, prompts, or call budgets.

## Deliberate limits

- No request-selected Python import, shell command, executable, URL, or network target.
- No write-capable or externally side-effecting tool.
- No fallback when a prompt, response, schema, case, checkout, license, source, or artifact hash
  differs.
- Replay token accounting reports the artifact's recorded logical calls; it does not establish
  latency, cost, or numerical parity with a live provider.
- A real upstream import smoke requires its pinned legacy environment. Fake pinned-checkout tests
  cover worker plumbing independently of those old dependencies.
