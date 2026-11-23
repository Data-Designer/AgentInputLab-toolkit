# SmartSearch frozen-inference worker

This worker is a thin, pinned execution boundary for the released
`vvv111222/SmartSearch-3B` checkpoint. It runs the byte-verified upstream
`ReCall.run` loop and official answer parsers. It does **not** implement, load,
or expose SmartSearch SFT, DPO, GRPO, process-reward, ground-truth, or
test-time query-refinement paths.

This release is deliberately reported as `controlled_adapter`, not
`official_native`. Upstream did not publish immutable identities for its
Wikipedia corpus, retriever checkpoint, and index, so paper-level retrieval
parity cannot currently be established. Every locally selected retriever,
corpus, and index is nevertheless content-bound so experiments remain
repeatable and collisions become visible.

## Curriculum data preparation (separate from frozen inference)

Install `agentinputlab[smartsearch-prepare]` to use
`SmartSearchCurriculumPreparation`. This API loads the pinned native
`detect_redundancy`, `transfer_sft`, `transfer_dpo`, and `transfer_regenerate`
converters from a clean SmartSearch checkout. Source bytes are checked against
the pinned Git objects and compiled directly, without accepting a cached
`.pyc` or writing into the checkout. It does not import the full training stack,
start a service, generate rewards, or perform an optimizer update.

```python
from agentinputlab.recipes import (
    SmartSearchCurriculumPreparation, SmartSearchTrainingCase,
)

prepared = SmartSearchCurriculumPreparation().prepare(
    checkout="/path/to/pinned/SmartSearch", output_root="/path/to/new/curriculum",
    train=train_cases, validation=validation_cases, test=test_cases,
    initial_scored_rollouts=recorded_initial_rollouts,
    regenerated_scored_rollouts=recorded_regenerated_rollouts,
    rollout_provenance=recorded_producer_provenance,
)
```

Cases are typed `SmartSearchTrainingCase(case_id, question, split,
source_reference=...)`. IDs and normalized-question hashes must be disjoint
across training, validation and test collections. Every recorded rollout must
belong to a training question and retain its measured outcome F1 and native
process-score strings. References, rewards and prepared targets remain
training-only; this API never sends them to frozen inference.

Outputs include the unchanged native redundancy-annotated records, SFT rows,
DPO pairs, regeneration prefixes, hashes and producer provenance. An empty
native selection remains empty. It is not a fitted policy or evidence that
SFT, DPO, or GRPO succeeded. The full `training.smartsearch.sft_dpo_grpo`
entrypoint is not yet implemented; the preparation API must not be used to
advertise that profile as complete. The frozen-inference manifest and its
forbidden training paths are unchanged.

`SmartSearchProcessTeacher` separately calls the original training-only
`SearchEvaluation.evaluate_search` and `Rewrite.rewrite_search` code. A supplied
completion provider replaces only the HTTP generation method. Native prompts,
fourth-`<think>` selection, process-score parser, redundancy rules and refinement
stop behavior remain unchanged. The helper receives **2048 tokens per call and
temperature 0.6**, with a separately recorded seed adapter. These are training
helper calls, not 512-token test calls. The default admission budget is
4096 output tokens / eight calls; a call that cannot admit the original 2048
cap fails before dispatch rather than shrinking the native cap.

The provider returns `SmartSearchTeacherCompletion(text, input_tokens,
output_tokens, model_fingerprint_sha256)`. Requests, responses, measured costs
and failures are saved per call under a fresh output root. An exception stops
the stage without the upstream HTTP empty-string fallback or an automatic
retry. Usage that could not be measured remains an unknown-usage reservation,
not an actual zero. Only declared training questions may carry gold answers or
process rewards; validation and test questions are rejected before dispatch.
The resulting scored records feed curriculum preparation. They are not a
fitted policy, and no optimizer stage is claimed by either API.

### Native SFT and LoRA-DPO policy stages

`SmartSearchNativePolicyTraining.fit(stage="sft" | "dpo", ...)` consumes a
hash-bound curriculum artifact and a caller-mounted, code-free safetensors
policy snapshot. It invokes the pinned LLaMA-Factory trainer through a fresh
`torchrun` process group; the native trainer owns collation, loss, optimizer,
scheduler, three-epoch schedule, original batch sizes, cutoff and ZeRO-3 config.
Set `SmartSearchPolicyFitControls(gpu_processes=1..4)` and select physical GPUs
through the operator's `CUDA_VISIBLE_DEVICES`. The library does not discover,
reserve or signal other users' GPU workloads. Native training dependencies must
be installed in that dedicated runtime; the lightweight package does not
silently replace or upgrade the caller's PyTorch/Transformers stack.

Native configuration changes are explicit: local input/output/cache paths,
seed, disabled remote-code/telemetry/publishing and create-only outputs. No
lower learning rate, shorter epoch count, replacement objective or quantization
conversion is applied. Empty native selections, fixture-only rollouts, overlap,
unbound model/data/source bytes and quantized full-fit/merge bases are rejected.
These pinned configs use the Qwen template; applying them to another model
family requires a separately defined variant, not a hidden template switch.

Each rank retains a native log and source-bound terminal receipt. Native global
optimizer steps are checked for agreement, not summed across ranks. Observed
primary-policy forwards and token positions are recorded separately from
unmeasured reference-model work. A missing receipt is unknown usage, not zero;
failure does not automatically retry. The caller's optimizer-step ceiling
rejects a larger native schedule rather than silently truncating it.

`SmartSearchPolicyStageArtifact.load(...)` checks the resulting file closure.
`materialize_snapshot(...)` explicitly copies final SFT model/tokenizer files
into a new inference snapshot without importing pickle/optimizer state.
`SmartSearchNativePolicyTraining.merge(...)` calls the original CPU LoRA merge
against the exact SFT base used by the DPO artifact; a LoRA adapter alone is
not a complete policy. Source checkpoints, logs and original model assets stay
unchanged. A fitted or merged artifact is a candidate for heldout evaluation,
not evidence that the full SFT/DPO/GRPO curriculum or compatibility matrix passed.
GRPO and end-to-end policy sampling remain separate implementation gates.

## What remains upstream code

The driver verifies and source-loads these exact MIT-licensed files from
`RUC-NLPIR/SmartSearch@5e9e7ec0d6af34f45fad145d4bff442529e7e6ff`:

- `src/re_call/inference/re_call.py`: `ReCall`, prompt construction, ten-turn
  loop, `<search>` validation/extraction, response/result concatenation;
- `src/flashrag/pipeline/active_pipeline.py`:
  `extract_last_assistant_response` (AST-extracted without importing the heavy
  FlashRAG/Transformers module);
- `src/verl/utils/reward_score/re_call.py`: `last_boxed_only_string` and
  `remove_boxed` (AST-extracted without importing VERL).

The worker calls `ReCall.run.__wrapped__`. This disables only the upstream
five-attempt *whole-trajectory* retry, which could silently replay paid calls.
The disclosed loop contract remains 10 rounds, `top_n=5`, temperature 0,
8192 maximum new tokens, and `</search>`/`</answer>` stops. One non-nested
search tag per turn is enforced to give retrieval a finite budget bound.

## Security boundary

Do not start `scripts/serving/sandbox.py`. It executes request-provided Python
with same-process `exec`, has no effective timeout, and listens without
authentication. This worker never imports it and rejects
`AGENTINPUTLAB_SMARTSEARCH_SANDBOX_URL` if set. The upstream
`execute_tool_calls` method is overridden with one fixed operation:

```text
wikipedia_search(query=<model-produced text>, top_n=5)
```

No request may provide an endpoint, command, import, callable, credential,
environment, or arbitrary tool call. Retrieved documents containing protocol
tags are rejected before they can escape `<result>`. Source is loaded from
verified bytes with bytecode writing disabled; the checkout is never added to
`sys.path`. The pinned upstream file's `import requests` is satisfied by a
driver-local, fail-closed placeholder while the module loads, then replaced
only with the controlled model transport immediately before execution. The
loader neither depends on the third-party `requests` package nor mutates the
process-global `sys.modules` table.

## Required identities

The host supplies a `SmartSearchArtifactIdentity`, and the worker requires an
identical, operator-owned runtime manifest. It binds:

- Hugging Face repository and revision
  `vvv111222/SmartSearch-3B@82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8`;
- a local model snapshot file-manifest hash and tokenizer revision;
- served model alias and model-deployment fingerprint;
- retriever implementation/revision and deployment fingerprint;
- corpus, index, and retrieval-config SHA-256 identities;
- compatibility manifest, official tool schema, loop contract, and parser
  contract;
- model transport (`sglang_generate` or `openai_completions`).

Generate a model snapshot manifest after downloading the pinned revision. It
must be JSON and include at least:

```json
{
  "repository": "vvv111222/SmartSearch-3B",
  "revision": "82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8",
  "tokenizer_revision": "82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8",
  "snapshot_tree_sha256": "<canonical hash of the files object below>",
  "files": {
    "model-00001-of-00002.safetensors": {
      "sha256": "<locally verified sha256>",
      "bytes": 123
    }
  }
}
```

Hash the exact JSON bytes as `model_snapshot_manifest_sha256`; the worker also
hashes the canonical `files` object as `model_snapshot_tree_sha256`. Every
generation and tokenizer-admission response must echo the exact repository,
revision, tokenizer, snapshot-tree digest, served alias, and admission
fingerprint. A mutable alias alone is rejected, so the identity can safely
participate in run/cache keys. The worker does not load 6.8 GB of weights.

## Runtime manifest and environment

Set:

```text
AGENTINPUTLAB_SMARTSEARCH_CHECKOUT=/absolute/pinned/checkout
AGENTINPUTLAB_SMARTSEARCH_RUNTIME_MANIFEST=/absolute/runtime-manifest.json
```

A real Git checkout must have the pinned clean HEAD and no untracked files.
Git runs with a minimal environment, global/system config disabled, repository
fsmonitor disabled, and the resolved top-level checked. An extracted source
audit snapshot instead requires both:

```text
AGENTINPUTLAB_SMARTSEARCH_SOURCE_ARCHIVE=/absolute/smartsearch.tar.gz
AGENTINPUTLAB_SMARTSEARCH_SNAPSHOT_COMMIT=5e9e7ec0d6af34f45fad145d4bff442529e7e6ff
```

The archive must match the hash in `compatibility_manifest.json`; a folder
name alone is never accepted as commit evidence.

The runtime manifest contains the exact `artifacts` object, the model snapshot
manifest path, and operator-controlled service configuration. Endpoints never
come from benchmark samples or method requests. Example service fields:

```json
{
  "artifacts": {"...": "the exact SmartSearchArtifactIdentity mapping"},
  "model_snapshot_manifest_path": "/models/SmartSearch-3B.manifest.json",
  "model_transport": "openai_completions",
  "model_base_url": "http://127.0.0.1:8000",
  "tokenizer_base_url": "http://127.0.0.1:8000",
  "served_model_name": "smartsearch-3b-pinned",
  "model_snapshot_tree_sha256": "<canonical snapshot files hash>",
  "context_admission_fingerprint_sha256": "<pinned admission fingerprint>",
  "model_timeout_seconds": 120,
  "retriever_base_url": "http://127.0.0.1:9000",
  "retriever_timeout_seconds": 30,
  "retriever_revision": "<implementation@revision>",
  "corpus_sha256": "<sha256>",
  "index_sha256": "<sha256>",
  "retrieval_config_sha256": "<sha256>"
}
```

API keys, if needed, are read only from
`AGENTINPUTLAB_SMARTSEARCH_MODEL_API_KEY` and
`AGENTINPUTLAB_SMARTSEARCH_RETRIEVER_API_KEY`; keys are refused over plain
HTTP. Redirects are always refused.

### Model serving

`sglang_generate` forwards the original raw Qwen prompt and original SGLang
sampling object to `/generate`. The controlled endpoint must return the full
immutable identity envelope plus `meta_info.prompt_tokens` and
`meta_info.completion_tokens`.

`openai_completions` sends the same raw prompt to `/v1/completions`, which is
supported by vLLM's OpenAI-compatible server. Chat Completions is intentionally
unsupported because applying another chat template changes model treatment.
The immutable identity envelope and token usage are mandatory. This requires a
small operator wrapper; a bare endpoint that returns only a mutable alias fails
closed. When a context-token ceiling is active, the wrapper's `/tokenize`
endpoint must return an exact preflight count and the same identity. Generation
is rejected before its side effect when the prompt does not fit, and provider
usage must equal the admitted count. Without this endpoint, a budgeted run
fails closed. Admission is a network preflight, not a model call.

The retriever endpoint is a small identity-attested wrapper over FlashRAG's
`/search`: it accepts fixed `query`, `top_n=5`, `return_score=false` and returns
`{"identity": ..., "documents": [{"id": ..., "contents": ...}]}`. This
envelope preserves document lineage that the upstream loop normally discards.

For an operator-owned local vLLM service, `openai_completions_local_snapshot`
uses `local_model_root` and verifies both the `/v1/models` root and every file in
the pinned snapshot manifest. Its `/tokenize` endpoint supplies exact admission
counts. Weight checks are streamed rather than loading multi-gigabyte shards
into one Python byte string. Completion `stop_reason` metadata preserves the
upstream stop contract: ReCall receives exactly one closing stop marker, while
EOS or token-limit termination never invents one.

When using the audited runner, inject the configured worker through the context:

```python
method = SmartSearchCommonSupportQueries(artifacts=artifacts)
record = InterventionRunner().run(
    method, state, InterventionContext(external_runtime=worker, seed=17)
)
```

Do not hide the worker on `method.client` when calling `InterventionRunner`.
The frozen helper still ignores the audit seed; downstream target seeding is a
separate setting. Its fixed 8192-token call cap also requires an explicitly
adequate execution budget rather than a 512/4096-token compatibility label.

## Budget behavior

The host reserves one model call before starting the worker. Inside the
monolithic official loop, the worker checks the remaining model and retrieval
caps immediately before every corresponding side effect. It also requires
8192 output tokens to remain before each official model call, admits context
tokens before generation, and reconciles provider usage afterwards. A
provider/retriever attempt
that fails is still reported as consumed; checkout, identity, or schema failure
before any side effect reports `consumed=false`.

SmartSearch's upstream code ignores seed. The seed is retained in the audit
request but the runtime attestation always records `seed_applied=false`.

## Running and testing

Use the ordinary bounded `JsonLineSubprocessWorker` with
`python workers/smartsearch/driver.py`; package credentials and endpoint
configuration only in that worker's environment.

Compatibility, runtime, snapshot, request, and HTTP JSON is parsed strictly:
duplicate keys and non-finite `NaN`/`Infinity` values are rejected. Before any
model/retriever call, the runtime tool schema's canonical hash and frozen
loop/parser hashes must exactly match the artifact identity.

The deterministic test calls the real source-loaded `ReCall.run.__wrapped__`
from a fake pinned checkout and verifies two-turn search, the official parser,
prompt/result trajectory, lineage, identities, budgets, and security gates:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/workers/test_smartsearch_worker_driver.py
```

The real-checkout symbol smoke is opt-in and does not download weights:

```text
AGENTINPUTLAB_SMARTSEARCH_REAL_CHECKOUT=/path/to/checkout
AGENTINPUTLAB_SMARTSEARCH_REAL_RUNTIME_MANIFEST=/path/to/runtime.json
```

No native-parity claim is made by either test.
