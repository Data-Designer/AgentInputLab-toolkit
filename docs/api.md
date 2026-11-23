# API contract v1

Python functions: `catalog()`, `create_intervention(release_id, **parameters)` and `run_intervention(intervention, state, context)`. Use the native composite runner for profiles that produce complete trajectories, not `InputPatch` objects.

The HTTP transport exposes operator-configured patch profiles only:

| Method | Path | Response |
|---|---|---|
| GET | `/v1/health` | `{data: {status, protocol}}` |
| GET | `/v1/catalog` | `{data: {methods, benchmarks, configured_releases, note}}` |
| POST | `/v1/interventions` | `{data: {record, state, scored: false, benchmark_score: null}}` |

All requests require `Authorization: Bearer <token>`. POST bodies use `release_id`, `state`, optional nonnegative `seed`; unknown keys are rejected. State fields match the Python `FrozenAgentState` dataclass. Limits: 1 MiB JSON request body, one active intervention; additional attempts return 429 before execution. Catalogs are bounded packaged inventories, returned in full without pagination.

Ambiguous duplicate JSON keys and non-finite numbers (including overflowing
exponents) are rejected before dispatch. Server tokens require at least 24
printable ASCII characters without whitespace; invalid token/port configuration
fails at startup with an actionable CLI error. The SDK validates each success
payload's endpoint contract and rejects NaN/Infinity before serialization. A
client timeout may leave server work running; a signal aborted before dispatch
starts no call.

The response preserves the original patch, trace, usage and state hashes. `success` means patch execution succeeded, not task success. No official metric is invented. Model backend URLs, credentials, filesystem paths, constructors and commands cannot be supplied by HTTP clients; configure these in trusted Python code.

Errors use `{error: {code, message}}`: 400 invalid schema, 401 authentication, 404 route, 411 missing length, 413 oversized body, 415 media type, 422 unsupported profile/dependency or execution rejection, 429 busy, 500 unexpected failure. Error text omits raw upstream responses and secrets. No redirect following or automatic retry is performed by the SDK. A timeout/abort stops client waiting, **not** the server's active model call; inspect the operator before resubmitting.

The loopback service has no durable job store, cancellation or idempotency persistence. Shared production deployments require a separate authenticated TLS gateway, quotas and durable execution ownership. It must not be advertised as a production internet server.

Example of trusted registration:

```python
import os
from agentinputlab.api import ToolkitService, make_server
from agentinputlab import InterventionContext
from agentinputlab.algorithms.clarify import ActiveTaskDisambiguation
from agentinputlab.backends.llm import OpenAICompatibleBackend

backend = OpenAICompatibleBackend(
    model_id=os.environ['MODEL_ID'],
    base_url=os.environ['MODEL_BASE_URL'],
    deployment_identity={'revision': os.environ['MODEL_REVISION']},
    api_key_env='MODEL_API_KEY',
)
service = ToolkitService(
    factories={'clarify.active_task_disambiguation': ActiveTaskDisambiguation},
    context_factory=lambda seed: InterventionContext(backend=backend, seed=seed),
)
make_server(service, token=os.environ['AGENTINPUTLAB_API_TOKEN']).serve_forever()
```

The server's factory key is an operator alias. The returned `record.release_id` remains the adapter's original identity. In actual evaluations use explicit budgets and record both identities; model-output errors are legitimate outcomes, not reasons for hidden retries.

`OpenAICompatibleBackend.generate()` invokes its transport once. HTTP errors,
timeouts and malformed responses fail that attempt; there is no automatic
chat-to-completion retry, including for GPT-OSS Harmony errors. A failed request
can have consumed tokens even when the provider returns no usage: record that
cost as unknown, not zero. Any separately approved retry needs its own attempt
and budget accounting. The backend fingerprint records
`single_transport_no_implicit_retry.v1`; injected transports must likewise avoid
hidden retries because their internal requests are outside the backend's view.

## Q-RAG configurable reader budgets

Q-RAG's two selector actions remain fixed by its checkpoint profile. Its reader
output ceiling is configurable independently of the run's cumulative budget:

```python
from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.acquire.qrag import QRAGControls
from agentinputlab.api import create_intervention, run_composite_intervention

# Supply your pinned artifacts, mount acknowledgements and worker explicitly.
method = create_intervention(
    "acquire.qrag.reader_transfer_factorial",
    artifacts=artifacts,
    acknowledgements=acknowledgements,
    controls=QRAGControls(max_output_tokens=512),
)
state = FrozenAgentState(
    query=question, context=candidate_blocks,
    budget=ExecutionBudget(max_output_tokens=4096, max_model_calls=8),
)
result = run_composite_intervention(
    method, state, InterventionContext(external_runtime=worker, seed=17),
)
```

`QRAGControls()` defaults to the upstream reader ceiling of 4000. This is an
upper limit, not a minimum spend. Each reader request is capped at the smaller
of the configured ceiling and the remaining cumulative output-token budget.
Zero remaining tokens or model calls stops execution before invoking the worker.
Within a single observed run, earlier calls must debit that run's ledger.
Start a fresh ledger for each new run; a consumed ledger is not reusable.
Reader budget metadata records the configured ceiling, effective ceiling,
remaining tokens, and whether the configured value uses the upstream default
or a custom setting. Retaining the default does not establish paper parity;
the controlled reader and its deployment identity remain explicit.

The selector-plus-reader sidecar uses `agentinputlab.qrag.frozen.v3`. Its reader
callback must accept the keyword `max_output_tokens` and forward it to the
actual model request (for example, `GenerationRequest(max_tokens=...)`). Both
the worker and adapter reject usage beyond that effective ceiling. A v2 worker
or a reader with the old three-argument signature is rejected; there is no
silent fallback, hidden retry or automatic increase of the run budget. This
composite profile uses the Python API; the patch-only HTTP route does not
pretend to support a composite result.

### Task-instruction-preserving reader variant

Question-only Q-RAG profiles keep their original prompt and reject task
instructions/history. For instruction-bearing QA, explicitly configure
`QRAGArtifactIdentity(reader_profile=QRAG_INSTRUCTION_READER_PROFILE,
reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256, ...)` and use the same
`acquire.qrag.reader_transfer_factorial` factory. Import these constants from
`agentinputlab.algorithms.acquire.qrag`. The profile and prompt hash label this
as `instruction_preserving_v1`, not native-reader parity; old results are not
renamed. The frozen two-action selector still sees only the question and
ordered candidate texts, never answer/support labels.

This opt-in profile uses `agentinputlab.qrag.instruction_reader.v1`. Its callback
also accepts `task_instruction` and `passage_annotations` keywords. Use
`QRAG_INSTRUCTION_SYSTEM_PROMPT` and `render_instruction_reader_prompt(...)`
to build the actual model request; pass through the effective `max_output_tokens`.
The renderer preserves the exact instruction and selected visible IDs, source
and metadata. Nonempty conversation history/tools are rejected, not discarded.

The reader returns final JSON with `answer`, `support_ids`, `evidence_ids` and
`evidence_texts`. `parse_instruction_reader_answer(result.final_answer)` parses
those explicit predictions for the original QA scorer. Missing/invented support
is not backfilled from selected passages or reference answers; empty evidence
arrays and score-zero outcomes remain real results. Model/token/call limits and
actual costs apply unchanged, with no format-repair model calls or retries.

For IR tasks whose original prediction contract is exactly `{"query": "..."}`,
explicitly select `parse_native_query_reader_answer(result.final_answer)` instead
of the QA decoder. Record `QRAG_NATIVE_QUERY_DECODER_ID` and
`QRAG_NATIVE_QUERY_DECODER_SHA256` with the original raw output, prompt identity
and usage. This separately labelled decoder makes no model calls and does not
change the prompt, selector or budget. It accepts only a complete final query
object (or complete `json` fence), rejects duplicate/extra fields and empty
queries, and never converts a QA answer or reasoning into a query. Run that
query against the original benchmark corpus/ranker and score the resulting
ranking with its native adapter. Selected-passage IDs are not a replacement
for this retrieval/scoring step. Keep original decoding failures when recording
a separately decoded result; malformed and wrong-task outputs remain failures.

For a frozen policy receiving a single candidate, construct
`LocalQRAGFrozenPolicyRuntime(..., single_candidate_axis_compat=True)`. This
explicit, fingerprinted compatibility option fixes an upstream tensor-rank
error: it keeps the batch/candidate axes after the existing dot-product
reduction. It does not split, duplicate or truncate candidates, change Q-values,
weights, argmax, seed or budgets, or modify the mounted upstream source. The
default `False` retains the historical implementation for exact replay. Record
the runtime fingerprint and preserve earlier failed attempts and their costs.

## Query2doc author-cache replay

The official cached profile accepts the unmodified `trec_dl2019.jsonl` and
`trec_dl2020.jsonl` assets from
[the author's pinned release](https://huggingface.co/datasets/intfloat/query2doc_msmarco/tree/3019b760cd849830aec6d0654dbe8a4a03e791d6).
Only these two small split files are audited. Download them separately after
reviewing CC-BY-4.0 attribution and the underlying MS MARCO terms; the package
neither bundles them nor downloads the training cache or a retrieval index.

```python
from agentinputlab.algorithms.reformulate.query2doc import Query2DocCache
from agentinputlab.api import create_intervention

cache = Query2DocCache.from_official_jsonl(
    "trec_dl2019.jsonl", split="trec_dl2019", accept_license_review=True,
)
method = create_intervention(
    "rewrite.query2doc.official_cached", cache=cache, split="trec_dl2019",
)
```

The loader verifies fixed file size, SHA-256, row count and native schema offline.
State task ID and query text must match the source exactly; no text-lookup or
generation fallback is used. It applies five query repetitions plus the native
pseudo-document, then the caller runs its configured retriever and scorer.
The generations are the author's `text-davinci-003` outputs, not outputs from
the caller's target backbone. Replay makes zero new model calls; historical
generation cost is unknown, not free. Sample candidate-pool scores do not
establish full-corpus retrieval performance or Pyserini ranking parity.

## InfoCQR checksum-bound caches

Load a supplied JSONL cache explicitly, then configure it in Python:

```python
from agentinputlab.algorithms.reformulate.infocqr import InfoCQRCache
from agentinputlab.api import create_intervention

cache = InfoCQRCache.from_jsonl(
    "rw-zsl.jsonl", expected_sha256=cache_sha256,
    artifact_id="operator-capture-rw-zsl-v1", dataset_revision=dataset_revision,
)
method = create_intervention(
    "rewrite.infocqr.cached_rw_zsl", cache=cache, split="test",
)
```

Each record binds its mode, split, task ID, original query, conversation hash,
and post-processing identity. Supply rewrites generated by that exact mode:
zero-shot outputs do not establish few-shot or edit coverage. A matching byte
hash does not establish authorship or the generating backbone; preserve those
identities, source records, failures and historical generation costs separately.
Replay makes no new model call and does not itself produce a benchmark score.
Gold-derived first-turn rewrites remain disabled by default. The same configured
factory can be registered in `ToolkitService`; HTTP clients cannot supply cache
paths or constructor parameters.

## Self-Ask first-stage component

`create_intervention("plan.self_ask.planner_component", config=SelfAskConfig(...))`
exposes the first continuation of the existing controlled Self-Ask loop. Supply
the same frozen demonstrations and backend explicitly. Its prompt, first stop
marker, seed, temperature and token cap match the full loop's first request;
only the release audit ID differs. It makes one generation, suggests a first
follow-up or preserves an initial answer marker, and does not execute retrieval,
intermediate-answer calls or the iterative loop. An explicit native `No` is a
paid NO_OP, not a generated plan; the exact native prefix
`Are follow up questions needed here: No.` is accepted too. Empty or unparseable continuations remain paid
failures. The caller continues the original visible state with its downstream
reader and native scorer under the remaining shared budget. The separate full
loop profile and its results are unchanged.

For a chat backend that separates `message.content` from reasoning, opt into
`FinalContentStopBackend(OpenAICompatibleBackend(...))` from
`agentinputlab.backends`. Logical request stops apply to final content rather
than the provider's reasoning stream. This preserves seed, token cap and all
server-reported usage, including generation beyond a logical stop; it never
promotes reasoning or adds a recovery call. It requires the OpenAI-compatible
parser and rejects `stop` in `extra_body`. It is buffered-only and does not
change default backend behavior or historical results. Record its distinct
fingerprint as an execution variant; it is not wire-level stop parity.

## Context-aware information-retrieval endpoint

`ContextIRQueryPredictor` is an explicit downstream adapter for a caller-supplied
IR corpus. Use it after a context intervention when the desired output is a
search query, rather than a QA answer. It makes one configured generation and
one call to the original retriever; it does not choose a corpus or a scorer.

```python
from agentinputlab.benchmarks import ContextIRQueryPredictor

predictor = ContextIRQueryPredictor(
    backend=backend, retriever=locked_retriever,
    max_tokens=512, top_k=15, seed=17,
)
# remaining_state contains the intervention's patched visible context and
# the original budget minus the intervention's measured consumption.
output = predictor.predict(remaining_state)
native_prediction = output.prediction
```

The prompt preserves the visible task instruction, query, context and conversation;
task IDs, state/context metadata and evaluator labels are not sent. The final
response must already be `{"query": "..."}`. QA objects, duplicate keys, empty
queries and reasoning-only outputs are paid failures, never extraction/retry
requests. `IRQueryOutputError` retains that response, request and consumed usage.
Returned rank order is preserved; original retrieval scores and features remain
in the result metadata. Use the dataset adapter's original metric on
`output.prediction` (or its `.scores` mapping when that adapter expects a mapping).
This optional endpoint does not change the existing production predictor defaults
or any algorithm's native prompts, controls or cached outputs.

For a model that wraps its query JSON in one Markdown code block, explicitly set
`allow_json_fence=True`. The default remains strict. This option accepts one
`json` or unlabelled code block, then applies the same one-field JSON validation;
it does not extract answers, repair JSON, change the query value, or make an
additional model call. Its fingerprint and result metadata record
`decoder_policy="single_json_fence.v1"`; the original response and usage remain
unchanged. Keep strict and compatibility results separately labelled when
reusing saved responses.
