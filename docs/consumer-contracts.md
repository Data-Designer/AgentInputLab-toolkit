# User-facing feature contracts

AgentInputLab separates discovery, input intervention, and benchmark execution.
Use the layer that produces the output you need; an intervention record is not a
benchmark score or proof that a complete agent task succeeded.

| User goal | Supported path and expected output | Conditions / verification |
| --- | --- | --- |
| Install and inspect the toolkit | `python -m agentinputlab --version`, `doctor --offline --json`, `smoke --json` | The core has no mandatory third-party dependencies. Smoke checks do not call a model. |
| Find an algorithm | Registry / `describe_intervention` lists profiles, parameters, availability and blockers | 27 methods, 87 profiles, 86 execution entrypoints. Discovery does not instantiate or certify every runtime. |
| Find a dataset | `list_benchmarks` / `describe_benchmark` reports loader kind and setup requirements | 31 entries: 14 expose an automatic loader; 17 require a native adapter. Automatic loading can still require extras, runtime files, network access or access approval. |
| Change an agent input | `create_intervention` + `run_intervention` returns changed state, patch, provenance and usage | Prompt-only profiles need no model. Model-, retrieval-, artifact- and native-runtime profiles need their documented dependencies. |
| Obtain a real task score | Execute a benchmark case through its adapter and original scorer | Follow [real execution](real-execution.md) and [benchmark setup](benchmark-setup.md). `scored: false` / `benchmark_score: null` is not a successful scored run. |
| Call from JavaScript / TypeScript | Build `packages/sdk` from the source distribution, then call an explicitly configured Python HTTP service | Node 20+, Python service, bearer token. The SDK is not an independent implementation and is not yet published to npm. |
| Configure a run | `RunConfig` / `config init`, `config validate`, `config run`; [configuration guide](configuration.md) | Editable JSON for budgets, seed, method parameters, HTTP backend and in-memory BM25. Native runtimes and dense/custom retrievers remain explicit Python dependencies. |
| Configure execution budgets | Set cumulative limits under `budget`, per-call caps under the selected method's `parameters` | Invalid budgets are rejected. A profile can report a native-protocol conflict; it does not silently increase the budget or change its algorithm. |
| Handle failures safely | Inspect structured errors, terminal status and recorded usage | No implicit retry of model attempts. Failed executions remain failures; missing usage is not converted to zero. |

## Reproduce the consumer-interface checks

Install the Python wheel, unpack the matching source distribution, and run:

```sh
python tools/check_consumer_contracts.py
cd packages/sdk
npm ci
npm test
cd ../..
python tools/check_consumer_contracts.py --sdk-module packages/sdk/dist/index.js
```

The checker inspects every profile and dataset entry, all installed CLI entrypoints,
three prompt-only Python paths, authenticated HTTP success/error paths, and
optionally a real SDK-to-Python HTTP roundtrip. It makes **zero model calls and
zero dataset downloads**. The optional `--report PATH` writes a small, create-only
JSON receipt. Run without Python's `-O` flag, which disables assertion checks.

These checks complement the real dataset and model acceptance evidence; they do
not replace it or certify all algorithm × dataset × backbone combinations.
See [validation](validation.md) and [toolkit acceptance](toolkit-acceptance.md)
for the scope and exclusions of real acceptance.

## HTTP and SDK guardrails

- Server tokens must contain at least 24 printable ASCII characters without
  whitespace. Configure them through the environment; never put them in a URL.
- Requests reject duplicate JSON keys, non-finite numbers, invalid budgets,
  unsupported media types and bodies larger than 1 MiB before execution.
- The SDK checks response structure, rejects non-finite JSON numbers and bodies
  larger than 1 MiB, respects an already-aborted signal before dispatch, and
  validates timeout values against Node's timer range.
- The default server exposes a small allowlist of prompt-only profiles. Supply
  explicit factories and contexts to expose additional profiles; do not treat
  the entire catalog as automatically enabled on a public server.
- The service is a local integration component, not a hosted multi-tenant
  authentication, billing or model-serving platform.

Scientific success depends on the case and model. A valid API response establishes
that the requested operation completed under its contract, not that a model gave
the correct answer. Preserve the original scorer, failures and attempt costs when
comparing methods.
