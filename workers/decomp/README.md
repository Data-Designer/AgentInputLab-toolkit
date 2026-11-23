# DecomP pinned worker

This JSONL worker runs the original `allenai/DecomP` controller and participant
implementations at commit `8b157f6fe9ff5676ae51bd054f1df7b015adf278`.
Set `AGENTINPUTLAB_UPSTREAM_ROOT` to that exact, clean checkout and install its
original optional dependencies in the worker environment.

The upstream project is Apache-2.0. The bounded search-loop compatibility code
is attributed to the pinned `BestFirstDecomposer.find_answer_decomp`; the
project-level third-party notice records the distribution boundary. The exact
upstream license text is distributed as `UPSTREAM_LICENSE.Apache-2.0.txt`.

Worker protocol v2 carries the complete input evidence as upstream-native
`paras`, alongside unique `paragraph_ids` and a SHA-256 `context_hash` over
both ordered arrays. The driver verifies that identity before importing
upstream code, inserts `paras` into the example passed to DecomP, and echoes the
IDs/hash in its response. The core bridge rejects any mismatch, so HotpotQA or
CommaQA evidence cannot be silently dropped or reordered at the worker
boundary. Non-empty context blocks require stable `block_id` values.

Only logical config artifact IDs declared in `compatibility_manifest.json` are
accepted. The driver never accepts a filesystem path, Python symbol, command,
host, credential, or Jsonnet external variable from a request. The upstream
`expr_eval` participant is rejected because its implementation uses Python
`eval`; this is a deliberate security incompatibility, not an algorithm result.

The compatibility manifest records execution, input and transport changes relative to upstream:

1. a bounded, traced form of the original best-first loop;
2. a request-bound cap on generator `max_steps`;
3. checksum-locked config loading and participant safety gates.
4. identity-checked pass-through of upstream `paras`.
5. the explicit, optional controlled native-completion transport described below.

`plan.decomp.full` is a composite-only profile: call `DecompFullBridge.run()`
through `CompositeInterventionRunner`. It returns the upstream terminal answer,
the exact input evidence, and the native execution trace as a `CompositeResult`.
There is intentionally no `propose()`/patch path, because feeding the full
system's answer back to a downstream solver would change the estimand and cause
a second answer generation.

The pinned runtime serializes all in-process executions with one process-global
`RLock`. This protects upstream's temporary CWD and `sys.path` mutations; its
fingerprint reports `concurrency_mode=process_global_serial_rlock`. Use separate
worker processes for parallel execution.

The `backend_binding` field is audit identity only in this first driver. DecomP's
original model clients remain responsible for generation. This limitation is
reported in every response and prevents claiming a controlled backend swap.
Fingerprint keys such as `model_revision`, `tokenizer_revision`, and
`max_output_tokens` are valid audit metadata. Only explicit credential key
names (for example `api_key`, `authorization`, `password`, `secret`, or
`access_token`) are rejected.

## Controlled live model binding

For caller-selected backbones, use `DecompLiveRuntime` instead of the legacy
audit-only `PinnedDecompRuntime`. It supervises an isolated worker and binds
only `GPT3Generator`'s provider-call seam to `DecompCompletionBackend` via
authenticated numeric-loopback callbacks. Native participants, prompt files,
sampling/stop/output settings, choice parsing, sequence scores, routing and
best-first search remain upstream code. The legacy engine name is explicitly
replaced by the caller's model, and the native response cache is disabled.
This is `controlled_adapter`, not original GPT-3 checkpoint/transport parity.

Install the validated legacy worker on Python 3.10, separately from the host.
Install the small CPU Torch distribution first to avoid fetching CUDA runtimes
for a source process that only calls the host's model service:

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==1.13.0+cpu'
python -m pip install 'agentinputlab[decomp]'
```

Upgrade pip inside this worker environment first. Older bundled pip versions
can reject the normalized `typing_extensions` wheel name and try to build it
using the PyTorch-only index, which does not provide its build dependencies.
The binary-only Torch step also fails explicitly on unsupported platforms
instead of starting an unexpected source build.

Provide the pinned clean checkout and an NLTK data directory containing
`corpora/stopwords/english`. Missing assets are explicit dependencies; this
runtime does not download models, datasets or NLTK resources during execution.
Host and worker must use matching native-worker protocol and source hashes. Source files,
prompt resources, worker hashes, dependency versions/import origins and the
English stopword checksum are attested before native model callbacks.

```python
from pathlib import Path
from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.decompose.decomp import DecompFullBridge
from agentinputlab.backends import DecompCompletionBackend
from agentinputlab.contracts import FrozenAgentState
from agentinputlab.integrations.external_workers import DecompLiveRuntime
from agentinputlab.runner import CompositeInterventionRunner

backend = DecompCompletionBackend(
    model_id="your-model", base_url="http://127.0.0.1:8000/v1",
    deployment_identity={"revision": "your-exact-model-revision"},
)
runtime = DecompLiveRuntime(
    python_executable="/path/to/legacy-env/bin/python",
    checkout=Path("/path/to/DecomP"), nltk_data=Path("/path/to/nltk_data"),
    backend=backend, per_call_output_cap=512,
)
record = CompositeInterventionRunner().run(
    DecompFullBridge(None, config_artifact_id="decomp.commaqa.coarse.p1"),
    FrozenAgentState(query="Your question", budget=ExecutionBudget()),
    InterventionContext(backend=backend, external_runtime=runtime, seed=17),
)
print(record.result.final_answer)
```

Config IDs are task-specific: reverse, letter concatenation and CommaQA prompts
are not interchangeable demonstrations or universal QA guarantees. Use the
appropriate source-pinned config and supply stable paragraph IDs for context.
The original native contract accepts query/context, not arbitrary instruction
or conversation history; an explicit caller projection is required for those.

The current controlled transport accepts one choice (`n=1`, `best_of=1`/`None`).
Native output caps are not enlarged or silently shrunk to fit the budget; a
dispatch is rejected if its original cap exceeds the remaining token/call
allowance or the caller's per-call ceiling. Actual native usage is retained on
failures; missing usage remains unknown, not free. For a cumulative input-token
ceiling, set `per_call_input_token_bound` to an upper bound enforced by the
bound provider (for example, its configured context-window size). The runtime
reserves that bound before every call, including concurrent calls, and settles
it to actual provider-reported input usage afterward. It never truncates the
prompt or changes generation parameters. Missing usage retains the reservation
and prevents another call; a provider exceeding its declared bound fails closed
while preserving actual consumption. Without an explicit bound, finite input
ceilings are rejected. Finite priced-cost ceilings still require a pricing
estimator and are not supported by this transport.
Backend I/O needs its own finite timeout in addition to the worker timeout.
`callback_events`, `last_response` and `last_resource_receipt` expose diagnostics;
the resource receipt covers the source process, not host callback CPU.
On failure, the runtime returns a typed error/usage envelope. The monitored
request hook raises after charging the complete actual usage, including failed
calls, rather than recording only the one-call pre-execution reservation.
