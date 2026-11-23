# Pinned ACON worker

This directory is the isolated JSONL execution boundary for
[`microsoft/acon`](https://github.com/microsoft/acon) at commit
`d63f9ae18959dc7215ff62899c94c5e8c56847ae` (MIT).

The driver does **not** reproduce ACON in AgentInputLab. It verifies the pinned
checkout and prompt files, imports the upstream classes, and directly calls:

- `productive_agents.ctxopt.history_optimizer.HistoryOptimizer.process`
- `productive_agents.ctxopt.obs_optimizer.ObservationOptimizer.process`

The bridge owns only transport validation, artifact identity, the upstream
`MemoryManager` branch/assembly checks already represented in AgentInputLab,
OpenAI-compatible deployment plumbing, and usage accounting. A successful run
does not by itself claim paper-number or model-output parity.

The runtime uses a two-phase v2 protocol. `preflight` verifies the checkout,
target/optimizer/tokenizer attestations and exact upstream threshold branch with
zero model calls. Only a triggered branch receives an `execute` request, which
must reserve one model call before the worker is invoked and must carry the
hash-bound preflight decision token. This lets `BudgetLedger` reject execution
before the optimizer endpoint has any side effect.

## Environment

Use a separate Python 3.11+ environment. `pip install 'agentinputlab[acon]'`
supplies Requests, Jinja2 and the exact-token-counting Tiktoken dependency used
by the native history/observation classes. It does not include optional upstream
HistoryRetriever dependencies or replace the mounted source checkout.
Point the worker at the exact checkout with one of:

```bash
export AGENTINPUTLAB_ACON_CHECKOUT=/path/to/acon
# Or set a shared parent; the driver selects its `acon/` child.
export AGENTINPUTLAB_UPSTREAM_ROOT=/path/to/upstreams
```

Configure an explicitly identified OpenAI-compatible endpoint. This works with
a hosted API or a local vLLM server:

```bash
export AGENTINPUTLAB_ACON_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export AGENTINPUTLAB_ACON_OPTIMIZER_MODEL=served-model-name
# Authenticated endpoints must use HTTPS. Local HTTP vLLM must use no key.
export AGENTINPUTLAB_ACON_API_KEY='...'
export AGENTINPUTLAB_ACON_OPTIMIZER_MODEL_REVISION='optimizer-revision'
export AGENTINPUTLAB_ACON_TARGET_MODEL_REVISION='target-revision'
export AGENTINPUTLAB_ACON_TOKENIZER_REVISION='tiktoken-revision'
python workers/acon/driver.py
```

HTTP redirects are disabled, so an authorization header can never be forwarded
to a different origin. The endpoint response must echo the configured serving
model alias.

Optional, bounded settings are:

- `AGENTINPUTLAB_ACON_TEMPERATURE` (default `0`, range `0..2`)
- `AGENTINPUTLAB_ACON_TIMEOUT_SECONDS` (default `120`, range `0.1..3600`)
- `AGENTINPUTLAB_ACON_MAX_TOKENS` (positive integer)

The three revision environment variables above are required and must equal the
values supplied in `ACONArtifactIdentity`; absence or mismatch fails closed.
The artifact identity also carries SHA-256 hashes for:

- the optimizer deployment fingerprint (endpoint/alias/runtime controls),
- the actual target backend fingerprint supplied by `InterventionContext.backend`,
- the instantiated tokenizer fingerprint and deterministic token-ID probes.

Approximate character-count fallback is forbidden. ACON execution fails with
`TOKENIZER_ATTESTATION_UNAVAILABLE` if the pinned tokenizer cannot be inspected.

`AGENTINPUTLAB_ACON_OPTIMIZER_MODEL` is only the endpoint's serving alias. The
revision fields remain the auditable identities used by an experiment.

`acon_opt_args` cannot contain `task`, `observation`, or `history`; those names
are reserved by upstream ACON and collision attempts are rejected both in the
bridge and worker. History patches persist
`acon_original_first_user_prompt` and `acon_previous_history_summary` through a
`SET_METADATA` operation, so later rounds reproduce upstream cross-round state.
Because the remote optimizer is connected through a controlled adapter rather
than the paper's whole agent stack, emitted patches use
`profile_kind="controlled_adapter"`.

## Pinned prompt profiles

The request's `(prompt_artifact_hash, guideline_artifact_hash)` pair must select
exactly one profile in `compatibility_manifest.json`:

| Profile | Prompt bundle SHA-256 | Guideline SHA-256 |
|---|---|---|
| `appworld.history.v1` | `b806c2f7ee7126e9746ddf6d553815d36ec026c411c9d805499af1d2e7d7c2e7` | `0508caa837c50403be2c8545646359a0fb72009fb14df3a8acf85aedaf649834` |
| `appworld.observation.v1` | `e50fe6421387a3518695f6be72a6f577b458074599c4d326fab23c178f694363` | `f500677dfd6609aef26fe2dbb3f9c928d2bd7ea8c0e432114feeed0636c7b385` |
| `officebench.history.v1` | `c266950825c7daf0ec0f7cc94cb401e3b177fd47ce62cd225b0c0616467c3f8a` | `a6c84b76dabc49e735cd264d1fdfd7810b73d0830e21f1dc71c4a719ae83a6ec` |
| `officebench.observation.v1` | `a2b4e909519c1c2715cd9d1505568e96b9909b7c453514e8c19064e30d0874b0` | `cb91df9d652f4a399e4941483ed65ed8a337cec26fbdcffa424d0a18cb41d82f` |
| `smolagents.history.v1` | `f0e1c88671bc645716af713b592883ea9732997041459f32535953a70d4d82a8` | `a6c84b76dabc49e735cd264d1fdfd7810b73d0830e21f1dc71c4a719ae83a6ec` |
| `smolagents.observation.v1` | `7a98fe868ca09d3bc5455df7f7b1ac8aa18a292eb105207e9d30a3484d82e96e` | `0210e1bc686c1d601c96daddeac09281aa5ab1639e68e39b1209bb9ae8205e1c` |

The bundle digest is SHA-256 over canonical JSON mapping each selected upstream
prompt path to its raw-file SHA-256. The worker independently verifies every
source and prompt file before importing upstream code.

## Native guideline candidate pipeline

`pip install 'agentinputlab[acon-fit]'` supplies the separate Python3.11+
guideline-pipeline dependencies; it does not install models, benchmark data or
the source checkout. `ACONGuidelinePipeline` invokes the pinned upstream
`run_history_analysis` and `run_history_update`, with the original regression
selection, aggregation, sampling and Jinja prompts. The native LLM factory is
replaced by the caller's metered backend before constructing either optimizer.
Private-config discovery and the native transport's retries are never invoked.
This is a controlled backend binding, not original-cloud or paper-result parity.

```python
from agentinputlab.recipes import ACONGuidelineControls, ACONGuidelinePipeline

result = ACONGuidelinePipeline(ACONGuidelineControls(
    per_call_output_tokens=512, max_model_calls=32, max_output_tokens=16384,
    num_prompts=1, seed=17,
)).fit(
    checkout="/path/to/pinned/acon",
    training_root="/path/to/frozen-native-train-bundle",
    training_manifest_sha256="<64-hex-sha256>",
    output_root="/path/to/new-fit-output",
    backend=backend,  # explicit, real LLMBackend with actual token usage
    base_profile="smolagents.history.v1",
)
```

The train bundle's `manifest.json` uses schema
`agentinputlab.acon.training-input.v1`, `split="train"`, an upstream benchmark
name (`appworld`, `smolagents` or `officebench`), `train_ids`,
`excluded_evaluation_ids`, and a `files` mapping of relative paths to SHA-256.
It pins `baseline-evaluations.json`, `optimized-evaluations.json`, and every
file in the corresponding `baseline/` and `optimized/` native history trees.
Both evaluation inputs must cover exactly the frozen training IDs; native
directory-to-task mappings and all `llm_history.json` inputs are checked before
generation. This checks caller-supplied identities and exclusions, not the
authenticity of a dataset's declared train split. Acceptance operators must
independently bind the source dataset/scorer and separate heldout groups.

Run in a dedicated child process with a provider deadline. The in-API wall-time
check is cooperative; it is not process isolation. Every attempted request is
saved before transport, failures stop the fit without automatic retries, and
missing token usage is never reported as measured zero. Fit costs are separate
from later downstream evaluation. The default512-token request cap is an explicit
controlled-fit cap rather than the native script's16384-token maximum.

Generated Jinja candidates are saved as **opaque, untrusted text**. This API does
not render or deploy them, does not select candidates by heldout scores, and does
not certify downstream guideline compatibility. The training release remains
gated until real fitted-artifact and deployment acceptance is complete.

### Frozen fitted-history artifacts

`agentinputlab.recipes.ACONGuidelineArtifact.from_fit` promotes an explicitly
chosen candidate only after pinned fit-start, fit-terminal and aggregate-analysis
checksums, complete parsed training signals, train/evaluation ID separation,
upstream sources, and the exact fitting backend fingerprint have been checked.
Truncated analysis preserved as `raw_output` remains a diagnostic fit outcome,
not a deployment-accepted guideline. Selecting a candidate is an explicit input;
the loader never picks one by heldout performance.

The deployment profile accepts literal template text and the pinned source's
`task`, `history`, and `prev_summary` placeholders. Calls, imports, attribute
access, filters, loops and other executable/indirect Jinja constructs are
rejected before transport. This is a declared frozen-placeholder deployment
profile, not a general-purpose Jinja execution API. The original source
`HistoryOptimizer` still constructs its prompt, calls the backend and parses the
summary; its original system message is explicitly carried to that backend.

```python
from agentinputlab.recipes import ACONGuidelineArtifact

artifact = ACONGuidelineArtifact.from_fit(
    checkout="/path/to/pinned/acon", fit_root="/path/to/native-fit",
    training_root="/path/to/frozen-native-train-bundle", backend=backend,
    fit_start_sha256="<sha256>", fit_terminal_sha256="<sha256>",
    analysis_sha256="<sha256>",
    candidate_path="optimized_prompts/improved_history_prompt_samples_0.jinja",
)
artifact.save("/path/to/new-artifact.json")  # create-only, data-only JSON
result = artifact.compress_history(
    checkout="/path/to/pinned/acon", backend=backend, task=task,
    history=actual_history_text, previous_summary=None,
    evaluation_id="prefit-excluded-case-id", output_root="/path/to/new-runtime",
)
```

Evaluation IDs must belong to the exclusions frozen before fitting. For general
inference, omit the ID; the receipt then makes no heldout-evaluation claim.
ID declarations/checksums are not independent dataset/group attestations: the
experiment's dataset adapter must verify authentic splits and group separation.
Use a dedicated child process and an actual provider/process deadline. Each
runtime defaults to one512-token call, saves requests before transport and never
retries a failed generation. Callers must account for this usage in the shared
agent budget. Fit costs and earlier failed attempts remain separate, not free.

## Failure behavior

Stdout is reserved for one response JSON line per request; upstream import and
diagnostic prints are redirected to stderr. Missing ACON dependencies report
`UPSTREAM_DEPENDENCY_MISSING`. Commit, license, source, prompt, request identity,
deployment-binding, decision-token, tokenizer, and reserved-key mismatches each
have a distinct fail-closed code. Failures after an attempted model request
include model-call/token usage; failures before a side effect explicitly return
`consumed=false`.

Run the worker contract tests with:

```bash
PYTHONPATH=src:. python -m pytest -q tests/workers/test_acon_worker_driver.py
ruff check workers/acon tests/workers/test_acon_worker_driver.py
```
