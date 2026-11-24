# Configure a run

Version 0.1.0a6 adds one editable JSON/Python run specification. Parsing and
validation do not download assets or call a model. Execution runs once, with no
implicit retries. Existing method-specific Python APIs continue to work.

## Start without a model

```sh
agentinputlab config init > run.json
agentinputlab config validate run.json
agentinputlab config run run.json --query "Find a suitable flight"
```

The generated profile is `plan.plan_and_solve.ps_prompt`. It changes the agent's
instruction; it does not call an LLM or produce a benchmark score. JSON files
are UTF-8, limited to 1 MiB, and reject duplicate keys, unknown fields and
non-finite numbers. `init` writes to stdout; choose a new filename to avoid
overwriting your own configuration.

```python
from agentinputlab import RunConfig

config = RunConfig.from_file("run.json")
print(config.validate())
result = config.run("Find a suitable flight")
print(result["state"]["instruction"])
```

`validate()` checks configuration shape and construction of the lightweight
backend/retriever. Native method/runtime constraints are additionally checked
at execution. It explicitly reports `runtime_ready: null` and
`real_acceptance_passed: null`; it is not a GPU, endpoint or benchmark check.

## Put each setting in the right place

| Setting | Location | Meaning |
| --- | --- | --- |
| Total output budget | `budget.max_output_tokens` | Cumulative output tokens across the run |
| Number of calls | `budget.max_model_calls`, `max_retrieval_calls`, `max_tool_calls` | Cumulative call ceilings, not retries |
| Per-call output length | Method `parameters.max_tokens`, or a nested native config's stage-specific cap | See `show RELEASE --parameters`; names differ by method |
| Shared chat request ceiling | `backend.max_output_tokens_per_call` | Clips each chat request before transport; `backend.stage_output_caps` provides explicit `method_id:phase` overrides |
| Sampling temperature | Method's native `parameters.temperature` or nested config | Forwarded to its generation requests; not every method samples |
| `max_length` | Only where the selected native profile explicitly declares it | Not a universal alias for output tokens; it can mean input plus output or compression length |
| Input accounting | `budget.max_context_tokens` | Cumulative provider-reported input tokens, **not** a model's context-window size |
| Seed | Top-level `seed` | Base seed; multi-stage algorithms may use documented offsets |
| Model endpoint | `backend` | Exact served model ID, API root, declared revision and key environment-variable name |
| Retrieval | `retriever` or a Python `InterventionContext` | Small inline BM25, checksum-bound corpus BM25/dense, or an injected custom retriever |

The per-call output ceiling and total output budget are separate. Requesting
512 tokens per call does not increase a total budget of 4096. A native algorithm
that requires more calls than your limit reports a conflict; the toolkit does
not silently alter the method. Some upstream profiles have fixed constraints.

Find the actual parameter names, defaults, required values and nested fields:

```sh
agentinputlab show rewrite.rrr.common_support_prompt --parameters
agentinputlab show rewrite.query2doc.querygym_live --parameters
```

For QueryGym's live Query2doc profile, `parameters` can contain
`{"temperature": 0.2, "max_tokens": 256}`. Execution still needs the pinned
QueryGym runtime; changing JSON does not install it. Unknown names such as
`temperatur` fail instead of silently using a default.

## A model-backed rewrite → retrieve → read run

This example makes real HTTP calls when you execute it. Supply your endpoint
and actual revision in the named environment variables. The tiny corpus and
prompts are a user example, not an official benchmark or author-paper protocol.

```python
import os
from agentinputlab import RunConfig
from agentinputlab.contracts.serialization import content_hash

rewrite = "Rewrite the question as a search query: {query}"
reader = "Answer {query} using these documents: {background}"
config = RunConfig.from_dict({
    "release_id": "rewrite.rrr.common_support_prompt",
    "seed": 17,
    "parameters": {"config": {
        "rewrite_prompt": rewrite,
        "reader_prompt": reader,
        "prompt_source_sha256": content_hash([rewrite, reader]),
        "rewrite_prompt_sha256": content_hash(rewrite),
        "reader_prompt_sha256": content_hash(reader),
        "temperature": 0.2,
        "max_rewriter_tokens": 128,
        "max_reader_tokens": 256,
        "top_k": 1,
    }},
    "budget": {"max_output_tokens": 384, "max_model_calls": 2,
               "max_retrieval_calls": 1},
    "backend": {
        "model_id": os.environ["AIL_MODEL"],
        "base_url": os.environ["AIL_BASE_URL"],
        "deployment_identity": {"revision": os.environ["AIL_MODEL_REVISION"]},
        "api_key_env": "AIL_API_KEY",
    },
    "retriever": {"kind": "bm25", "k1": 1.2, "b": 0.75,
                  "documents": [{"doc_id": "france",
                                 "text": "Paris is the capital of France."}]},
})
result = config.run("What is the capital of France?")
print(result)
```

`config.to_dict()` gives the JSON-serializable settings for your own storage.
From a16, inline BM25 documents without a `corpus_revision` receive a stable
`inline-sha256:` revision derived from the complete inline document collection.
Changing its content changes that identity. Explicit source revisions remain
unchanged. This supplies provenance required by RRR without another model call
or a user-invented version label.
Never put a plaintext key in that file: use `api_key_env`. Backend `extra_body`
can hold provider-specific options, but request-owned fields (`temperature`,
`max_tokens`, `seed`, model/messages, etc.) are rejected there. Set those on the
method/request instead; previously overlapping options could be overwritten.

### Hosted model identities, structured output and stage ceilings

These options are included in the published `0.1.0a15` package and the public
source. They do not require a research checkout.

Set the exact provider model ID and a revision in `backend`. Providers can return
a shorter family name in the response; declare verified names explicitly:

```json
{
  "backend": {
    "model_id": "deepseek-v4-1-flash-261017",
    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
    "deployment_identity": {"revision": "261017"},
    "api_key_env": "ARK_API_KEY",
    "extra_body": {"thinking": {"type": "disabled"}},
    "response_model_aliases": ["deepseek-v4-1-flash"],
    "max_output_tokens_per_call": 512
  }
}
```

Merge this fragment into your run configuration. The alias allowlist becomes
part of the backend fingerprint and reaches the ReFormeR and InfoCQR workers.
Responses retain the provider's original model label. A family label identifies
an accepted response name; the pinned request and deployment revision identify
the requested snapshot. Unknown aliases fail without an automatic retry.

For ACON's separate optimizer, set `runtime.response_model_aliases` in an
`acon_observation_worker` or `acon_history_worker` configuration. This is bound
to the optimizer fingerprint independently of the downstream backend.

Declare supported formats with `backend.capabilities`, for example
`["chat", "completion", "json_schema", "tool_calling", "seed"]` for an endpoint
that supports those features. A strict `json_schema` request requires that
capability before transport. The requested schema is preserved. Use the
capabilities verified for your endpoint and subscription.

For an endpoint that supports strict JSON Schema on `/responses` instead of
`/chat/completions` (such as DeepSeek direct), add
`"structured_output_transport": "responses_strict"` and include `"json_schema"`
in `backend.capabilities`. Set `base_url` to the API root, for example
`https://api.deepseek.com`, and configure the account key through `api_key_env`.
This option routes only strict schema requests, keeps the original output cap
and supplied seed, and does not retry or downgrade to JSON-object mode. It
currently supports text prompts and closed, required-field object schemas
with integer leaves, including Plan-and-Budget's assessment schema. Ordinary
chat requests still use Chat Completions. A provider accepting the seed field
does not guarantee deterministic output or equivalence across providers.

When a STATE-Bench endpoint supports a non-thinking Responses tool loop,
set `"agent_responses_reasoning_disabled": true` in its arena configuration.
For DeepSeek direct, this sends `reasoning: {"effort": "none"}` on the
evaluated agent's tool requests while preserving native temperature, tool
history and token limits. It defaults to false and does not change the
simulator or judge. This arena option is separate from the backend's
`responses_strict` option for the intervention's structured requests.

Stage ceilings are keyed by the algorithm's exact request metadata:

```json
{
  "backend": {
    "max_output_tokens_per_call": 512,
    "stage_output_caps": {
      "active_task_disambiguation:partition_hypotheses": 1024
    }
  },
  "budget": {"max_model_calls": 8, "max_output_tokens": 4096}
}
```

Merge the ceiling settings into a complete chat backend block. Each request
receives the smaller of its requested allowance and the matching stage ceiling;
the run's cumulative budget remains in force. This opt-in example changes the
named stage allocation and its fingerprint. Raw-completion Decomp/UDCS profiles
set their native per-call limits in method parameters.

### Active Task Disambiguation: joint H2/Q2

The default ATD layout samples hypotheses and questions separately, then
partitions the hypotheses for each question. Set `request_layout` to
`joint_h2q2` to request two hypotheses, then generate both questions and their
answer categories in one joint request. The adapter uses uniform empirical
information gain to select a clarification in either layout.

```json
{
  "release_id": "clarify.active_task_disambiguation.eig_uniform",
  "parameters": {
    "controls": {
      "request_layout": "joint_h2q2",
      "total_hypotheses": 2,
      "total_questions": 2,
      "max_tokens": 512,
      "temperature": 0.7
    }
  },
  "backend": {
    "base_url": "https://your-endpoint.example/v1",
    "model_id": "your-model",
    "deployment_identity": {"revision": "your-model-revision"},
    "api_key_env": "OPENAI_API_KEY",
    "max_output_tokens_per_call": 512,
    "stage_output_caps": {
      "active_task_disambiguation:partition_hypotheses": 1024
    }
  },
  "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
  "seed": 17
}
```

The endpoint must support chat, JSON-object output, seed, and the declared
stage sizes. The two stages reserve 512 and 1,024 output tokens within the
shared budget; allow at least two calls and 1,536 output tokens. If a configured
backend wrapper would shorten either stage, ATD fails before the first request.
An incomplete answer grid is reported as a method failure without retry.

## Per-case retrieval without hand-writing a document list

For tasks whose native visible state contains documents, opt in to a fresh
BM25 index for **each case**:

```json
{"retriever": {"kind": "visible_context_bm25", "k1": 1.2, "b": 0.75}}
```

Use this fragment inside a complete run configuration with the chosen method,
backend, budget and parameters. It works with `config run --state state.json`
and `config evaluate` over selected static cases. It does not add documents,
read evaluator gold, download a corpus, or make model calls during preparation.
The original state remains unchanged; methods still decide how to use retrieval.
Each evaluation record includes `case_retriever` with the case hash and actual
BM25 corpus fingerprint. No index is shared across cases.

By default, the visible title and metadata are retained, and the corpus revision
is the hash of the original visible context. Existing rank/provenance annotations
are kept under `visible_context_provenance`, not treated as new retrieval ranks.
To reproduce a previously explicit **text-only** document binding, choose
`"document_projection": "text_only"`; a fixed `"corpus_revision": "YOUR_REVISION"`
can also be declared. These settings are explicit configuration choices, not
automatic replacements for an existing experiment's retriever.

For long source documents, add `"chunk_words": 350` to index reversible
chunks from the same visible text. The default `chunk_boundary` is
`"word"`; choose `"line"` to keep source lines intact, including a line
that exceeds the target size. Each chunk carries its source block ID,
character offsets and block hash. Leave `chunk_words` unset to index one
document per visible block. Chunking uses the metadata-preserving projection.

For `plan.self_ask.full_loop` on static QA, an existing protocol may explicitly
expose its configured retriever as `search` to the downstream reader. To reproduce
that protocol, add this to the **predictor** JSON passed to `--predictor-config`:

```json
{"schema_version": 1, "retriever_capability": {"visible_name": "search"}}
```

This option is off by default. It requires a real configured retriever and only
declares Self-Ask's existing search capability; it does not install a new tool
executor, change demonstrations or budgets, add a search call, or provide hidden
documents. Each record keeps the original adapter-state hash, execution-state
hash, and bound retriever fingerprint. The original example and evaluator gold
are unchanged. Conflicting visible tools or an invalid visible-context corpus
fail before model execution. Do not enable it to silently change an existing
experiment that did not expose `search`.

The separate `plan.self_ask.planner_component` keeps strict response parsing by
default. Some models emit a standalone `No.` line followed by an explanation.
To accept that explicit no-follow-up decision, set
`parameters.allow_no_decision_explanation` to `true` alongside the existing
`parameters.config`. This does not change the prompt, demonstrations or budget.
The paid response produces an unchanged-input patch, not a generated plan or a
scored answer. Its metadata records
`response_compatibility: standalone_no_with_explanation.v1` and whether the option
was used. Inline ambiguous text, a bare `Yes`, and conflicting/incomplete protocol
markers remain errors. Keep strict failures and costs when comparing this
compatibility variant; it does not change `plan.self_ask.full_loop`.

Empty context or missing/duplicate block IDs fail before model execution. Short
candidate sets are not padded. Default Q-RAG/R2AG candidate-count checks apply;
the separately labelled exhaustion/short-pool options below can use fewer real
documents without duplicating them. This is visible-context retrieval, **not full-corpus IR**; use
`prepared_corpus` or `beir_archive` when the protocol requires a corpus index.
Native workers that attest a fixed index revision still require their matching
binding; this option does not bypass those checks.
`config evaluate-arena` rejects this option before execution; native arenas need
an explicit arena-compatible retriever rather than this static-case binding.

## Native IRCoT from a file (controlled loop or full QA)

The `ircot_native` runtime launches the existing pinned native implementation in
an isolated Python process with a private temporary HOME. It does not substitute
the common-support algorithm, mutate your shell environment, fit a model, or retry
failed requests. Install the same toolkit version in the host and an isolated
Python 3.10 worker with `agentinputlab[ircot]`; the worker also needs the matching
`en_core_web_sm` 3.4.1 pipeline, GPT-2 tokenizer closure and NLTK English stopwords
described in [the worker guide](../workers/ircot/README.md). Asset preparation and
downloads are not implied by configuration validation.

Start with your normal `backend`, `retriever`, `seed`, and `budget` configuration.
Both components must be file-defined (BM25 or a prepared corpus); Python-only
injected components use the existing native Python API instead. Add:

```json
{
  "release_id": "acquire.ircot.native_like_substituted",
  "parameters": {
    "controls": {"per_step_top_k": 6, "max_steps": 7,
                 "max_total_documents": 15, "max_output_tokens": 4096}
  },
  "runtime": {
    "kind": "ircot_native",
    "source_root": "assets/ircot",
    "python_executable": "env-ircot/bin/python",
    "tokenizer_root": "assets/gpt2-tokenizer",
    "tokenizer_manifest": "assets/gpt2-tokenizer-closure.json",
    "tokenizer_manifest_sha256": "REPLACE_WITH_TRUSTED_SHA256",
    "nltk_data": "assets/nltk_data",
    "corpus_revision": "YOUR_CORPUS_REVISION",
    "index_revision": "YOUR_INDEX_REVISION",
    "per_call_output_tokens": 512,
    "timeout_seconds": 600
  }
}
```

This fragment is not a complete run configuration: include the explicit backend,
retriever and positive model/retrieval/output budgets. Paths are relative to the
configuration file. The tokenizer manifest contains `file_sha256` for its complete
directory. Preparation checks these files, the pinned source and actual native
dependency identities without calling a model:

```bash
agentinputlab config prepare-ircot ircot.json --output-dir prepared-ircot --accept-substitutions
agentinputlab config run prepared-ircot/run.json --state visible-question.json
```

Preparation writes the full-QA substitution manifest and ready `run.json`; it never
overwrites an existing directory. Full QA accepts question-only visible state with
an explicit `task_id` and an external corpus. It retains all six native roles and
the shared final-answer budget; a 512 ceiling does not increase the native
generator's 300-token default. Changing a bound model, corpus, budget or dependency
requires a new explicit preparation receipt, not a silent fallback. Native input
constraints remain in effect; instructions are not stripped to force acceptance.

For the controlled retrieval/reasoning component, use
`acquire.ircot.controlled` with the same runtime kind. Preparation supplies its
source-bound prompt/corpus/index parameters. Its result is an input patch, **not**
the full-QA answer or a benchmark score. Neither profile claims the historical
Codex/Elasticsearch deployment or its published scores. Actual input-token/cost
usage is recorded, but this worker does not support pre-attested
`max_context_tokens`/`max_cost_usd` ceilings and rejects those before execution.
An existing backend configuration may explicitly set `extra_body.n` to integer
`1`; this preserves the single-response request. Other choice counts are rejected,
so it cannot expand the model budget or silently select one of several answers.

## Reuse frozen skills and plans from a configuration file

### Reference a real retrieval corpus instead of embedding it in JSON

Use `retriever.kind: prepared_corpus` for an existing normalized JSONL corpus.
Each line has `id`, `title`, and `text`; keep the corpus outside the 1 MiB run
configuration. The existing production loader verifies the SHA-256, document
count and unique IDs before execution. It does not download or replace a
missing corpus. Paths below are relative to the run configuration file:

```json
{
  "kind": "prepared_corpus",
  "benchmark_id": "beir.arguana",
  "profile_id": "sparse.bm25.memory.v1",
  "corpus_path": "data/corpus.jsonl",
  "corpus_sha256": "REPLACE_WITH_TRUSTED_CORPUS_SHA256",
  "corpus_revision": "REPLACE_WITH_SOURCE_REVISION",
  "expected_document_count": 8674,
  "options": {"k1": 1.2, "b": 0.75}
}
```

Put this object under `retriever`; use the count and digest from your verified
prepared dataset, not copied values for a different corpus. `config evaluate`
rejects a different declared benchmark. BM25 indexes the corpus in RAM; this
is not a streaming search engine for an arbitrarily large collection.

The same binding accepts the existing dense retriever profiles, for example
`dense.e5.base_v2`, with the `dense-retrieval` extra. Supply
`options: {"model_path": "models/e5", "device": "cpu", "local_files_only": true}`
and the existing production `artifact_files` list of `{name, path, sha256}`
entries. Dense weights stay local; model and artifact paths also follow the
configuration file. Validation checks structure without opening the corpus or
loading weights; execution verifies and builds the retriever. This does not
claim that every dense model/benchmark combination has passed real acceptance.

### Reuse an existing SQLite FTS5 index

For an index already built by `SQLiteFTS5Retriever.build`, use this explicit
binding under `retriever` instead of converting it to a different BM25 engine:

```json
{
  "kind": "sqlite_fts5",
  "index_path": "data/corpus.sqlite",
  "index_sha256": "REPLACE_WITH_TRUSTED_INDEX_SHA256",
  "corpus_identity": {"revision": "REPLACE_WITH_ORIGINAL_CORPUS_IDENTITY"},
  "expected_document_count": 8674,
  "benchmark_id": "beir.arguana"
}
```

Supply the **entire original** corpus identity, actual count and trusted digest;
the illustrative values above are not a prepared dataset. Relative paths follow
the configuration file. Validation checks shape without accessing the index;
execution streams its SHA-256 and verifies the stored identity and count before
calling a model. The index must remain unchanged while in use, with no WAL or
journal sidecars. Missing or mismatched indexes fail explicitly: this binding
does not download, rebuild, truncate or switch retrieval engines. Native SQLite
rankings and fingerprints are retained; it does not imply full-corpus coverage
if the original index was a sample. An optional `benchmark_id` binds evaluation
to the declared dataset, as with other configured retrievers.

### BEIR archives: prepare the corpus and evaluate from one configuration

FiQA, Quora, DBpedia-Entity and FEVER can use their existing checksum-pinned
BEIR archives directly. No manual corpus export or corpus hash entry is needed:

```json
{
  "release_id": "baseline.no_op",
  "retriever": {
    "kind": "beir_archive",
    "benchmark_id": "beir.fiqa_2018",
    "accept_license_review": true,
    "cache_dir": "data/beir",
    "engine": "sqlite_fts5"
  }
}
```

Save as `fiqa.json`, then run:

```bash
agentinputlab config evaluate fiqa.json --benchmark beir.fiqa_2018 --split test --limit 1 --output fiqa-results.jsonl
```

FiQA also supports explicit `--split dev` with its native development
judgments. The adapter verifies that the selected query IDs are disjoint
from train and test and records the selected split in its fingerprint.
If the retriever configuration declares a `split`, it must match the
evaluate command. Use a separate output and, when specifying one, a separate
`index_path` for each split. In a benchmark workflow, set `target.split` to
`"dev"` and keep the archive or cache setting under `run.retriever`; the
workflow uses that source for preparation and evaluation.

Install the package's `evaluation` extra for the native `ir-measures` scorer.

This prepares the bounded source cache, builds the complete corpus's native
SQLite FTS5/BM25 index once, and uses the original IR scorer. NO_OP retrieval
does not require a model endpoint; methods that generate text still require
their own backend/runtime configuration. `--offline` also forbids downloads
by this retriever. An existing archive can instead be mounted with
`source_archive: "data/fiqa.zip"` (omit `cache_dir`); `index_path` optionally
selects a reusable SQLite index. Paths follow the run configuration file.

Defaults cap downloads at 512 MiB, corpus documents at 100,000, and document
ID/title/text content at 256 MiB while building an index. Larger collections
require explicit `max_download_bytes`, `max_documents`, and `max_text_bytes`
values plus sufficient disk space. These are safety limits, not silent
sampling: an exceeded limit fails without producing a truncated index.
An existing index is checked against the source identity and document cap;
the text-byte build cap does not rescan an existing index. `memory_bm25` is an
explicit alternative engine with configurable `k1`/`b`, not an assertion of
identical rankings to SQLite. Do not combine this retriever with a second
`--cache-dir`/`--prepared-root` dataset source on the evaluate command.

### Score observed retrieval rankings

On BRIGHT Earth Science and FiQA, the supported Decomp, Self-Ask, RRR, IRCoT
and Q-RAG patch releases can supply their own ranked retrieval evidence to
the native scorer. Bind the real corpus retriever and pass this predictor
configuration:

```json
{"schema_version": 1, "trace_ranking": {"protocol": "observed_rank.v1"}}
```

This mode takes document IDs from the method's observed retrievals and adds
no reader call or second retrieval pass. Each release keeps its own model,
runtime and budget requirements. The evaluator records the ranked IDs and
rejects an ID that was absent from the case's retriever output. Leave
predictor `task`, `qa`, and `retrieval` unset in this mode: a separate reader
or rank cutoff does not apply to the method's observed ordering.

### HyDE: bind a dense encoder and corpus

HyDE needs a dense encoder and vector search, not just a lexical retriever.
Set top-level `vector_runtime` to the same `prepared_corpus` object described
above, selecting a dense `profile_id` and its matching local model files.
The resulting runtime is shared by the encoder and searcher. Corpus, model and
artifact paths follow the configuration file; sparse profiles are rejected.
For the existing multilingual E5 weights, use `dense.e5.multilingual_large`
(intfloat/multilingual-e5-large revision
`3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`), not the English E5 profile.
Install the `dense-retrieval` extra in the execution environment.

For `acquire.hyde.paper_reimplementation`, provide `parameters.config` with
the intended HyDE settings and run normally. For `acquire.hyde.controlled`,
provide those same settings, an explicit backend and `vector_runtime`, then:

```sh
agentinputlab config prepare-hyde hyde.json --output-dir prepared-hyde
agentinputlab config run prepared-hyde/run.json --state visible-state.json
```

Preparation verifies existing corpus/assets and captures dependency identities;
it does not generate hypotheses, encode documents, search, download or train.
It writes a new ready configuration and refuses to overwrite an existing one.
Generation and embedding happen during execution. Method settings and budgets
are unchanged: eight hypotheses use eight generation calls, so a subsequent
reader needs additional explicitly configured allowance. Preparation never
silently reduces hypotheses or increases the total call budget. A patch-only
run does not claim an end-to-end benchmark score.

### Controlled LongLLMLingua from a file

`curate.longllmlingua.controlled` can bind the existing native compressor without
a Python factory. Install `agentinputlab[compression]` and supply a local causal
LM snapshot with its complete model/tokenizer file hashes:

```json
{
  "release_id": "curate.longllmlingua.controlled",
  "runtime": {
    "kind": "longllmlingua_native",
    "snapshot": "./models/compressor",
    "model_id": "YOUR_ORIGINAL_HELPER_MODEL",
    "revision": "<40-hex model revision>",
    "file_sha256": {"<each loader file name>": "<its SHA-256>"},
    "device_map": "cpu",
    "torch_dtype": "float32"
  },
  "parameters": {"config": {"rate": 0.5, "context_budget": "+100"}},
  "seed": 17,
  "budget": {"max_output_tokens": 4096, "max_model_calls": 8}
}
```

Add the reader `backend` configuration described above and replace the identity
placeholders. `file_sha256` must include the actual config, tokenizer and
safetensors files; indexed weights require their index and all referenced
shards, plus loader sidecars present in the snapshot. Do not copy hashes from a
different model. Relative `snapshot` paths follow the configuration file.
Supported devices are `cpu`, logical `cuda:0` and `mps`; choose a floating dtype
supported by your hardware (`float16`, `bfloat16` or `float32`). Device acceptance
does not certify that a particular accelerator is available on your machine.

```sh
agentinputlab config prepare-longllmlingua compression.json --output-dir prepared-compression
agentinputlab config run prepared-compression/run.json --state visible-state.json
```

Preparation binds the original reader, compressor identity, settings, seed and
budget without loading model tensors, compressing, generating or fitting.
Execution verifies the native source and full local asset closure, then runs
the original compressor. The result is a patched input, **not** a benchmark
score: use `config evaluate` or `config evaluate-arena` with the appropriate
benchmark setup for downstream execution and scoring. Helper perplexity
computation is recorded separately from reader generation; no extra reader
calls or larger budget are granted. The existing `curate.longllmlingua.official`
constructor remains separate and unchanged.

### Frozen skills and playbooks

For ACE and SkillOpt evaluation, reference a previously saved
`artifact.as_dict()` JSON and its trusted file SHA-256. No fitting is performed:

To prepare a new artifact explicitly, use the separate
[file-based fitting command](fit-configuration.md). Its optimization controls
and development data are separate from the test-time configuration below.

```json
{
  "release_id": "recipe.skillopt.eval_only",
  "artifacts": {
    "artifact": {
      "path": "artifacts/skill.json",
      "sha256": "REPLACE_WITH_THE_TRUSTED_64_HEX_FILE_DIGEST"
    }
  },
  "budget": {"max_output_tokens": 4096, "max_model_calls": 8},
  "seed": 17
}
```

Use `recipe.ace.eval_only` with an ACE artifact instead. The path is relative to
the configuration file, not your shell's working directory. A checksum proves
which bytes were used, not that an artifact was trained on an appropriate split
or model: keep that provenance and select the correct artifact. The package
does not bundle private experiment results or train a replacement on a cache miss.

```sh
agentinputlab config validate run.json
agentinputlab config run run.json --state visible-state.json
```

`--state` accepts the complete visible state, for example:

```json
{
  "task_id": "case-1",
  "query": "Which option follows the policy?",
  "context": [{"text": "The visible policy.", "block_id": "policy"}],
  "tool_names": ["search"]
}
```

Do not put hidden labels in this state. Its budget is replaced by the explicit
run configuration. `--query` remains available for inputs that do not require
task IDs or captured context. Artifact validation and native identity checks
run before method execution; malformed files and hash mismatches stop the run.
The lightweight `config validate` command checks the reference's shape only,
without opening assets or contacting a model. These evaluation profiles return
a patched state, not an answer or benchmark score; the downstream agent and
native scorer are still separate steps.

### MCE: evaluate an existing text-only artifact

`recipe.mce.text_artifact_eval` accepts `artifacts.artifact` with a local `path`
and the raw file's `sha256`, just like ACE and SkillOpt evaluation. The file must
be the complete JSON export of `MCETextArtifact.as_dict()`. Relative paths are
resolved from the run configuration. Use `config run` for the input patch, or
`config evaluate-arena` with the native arena configuration for downstream
execution and scoring. Original fit costs remain recorded in the artifact;
evaluation does not fit again. The loader verifies both the file checksum and
the artifact's original content identity. Only frozen Markdown/text context is
accepted; it does not enable the separate generated-code or sandbox profiles.
An artifact whose validation selected iteration zero legitimately produces a
no-op patch, not a newly optimized context.

### Adaptive-RAG: use an existing frozen router

`acquire.adaptive_rag.common_support` accepts `artifacts.router` directly:

```json
{
  "release_id": "acquire.adaptive_rag.common_support",
  "artifacts": {"router": {
    "path": "./assets/router",
    "sha256": "<raw artifact-manifest.json SHA-256>",
    "device": "cpu"
  }},
  "parameters": {"config": {
    "single_retrieval_top_k": 6,
    "ircot": {
      "demonstrations": "<your fixed IRCoT demonstrations>",
      "prompt_sha256": "<SHA-256 of those exact demonstration bytes>",
      "max_steps": 2
    }
  }}
}
```

Add your backend, retriever, budget and seed, then use the same `config run` or
`config evaluate` entry. The path names the existing T5 router **directory**,
relative to the configuration file; its manifest and complete file closure are
checked by the original router loader. Device defaults to CPU. The optional
`adaptive-rag-fit` dependency extra supplies the native libraries; installing it
does not start fitting. Use its documented Python/dependency environment.

This binds the existing common-support classifier and its A/B/C selected-arm
execution, not the separate upstream-router profile. It does not fit on a cache
miss, download a substitute checkpoint, or claim original-checkpoint parity.
`config validate` checks shape without loading weights; runtime construction
verifies assets and execution loads the classifier. A working classifier does
not by itself establish downstream benchmark success.

### Adaptive-RAG: the complete controlled router and answer pipeline

The separate `acquire.adaptive_rag.controlled_reimplementation` profile now has
a file-based preparation entry. Start with an unbound template:

```json
{
  "release_id": "acquire.adaptive_rag.controlled_reimplementation",
  "parameters": {"controls": {"max_output_tokens": 512, "ircot_max_steps": 2}},
  "runtime": {
    "kind": "adaptive_rag_controlled",
    "source_root": "./upstream/Adaptive-RAG",
    "router_snapshot_path": "./assets/router",
    "router_weights_path": "./assets/router/model.safetensors",
    "router_tokenizer_path": "./assets/router/tokenizer.json",
    "prompt_root": "./prompts",
    "input_token_upper_bound": 65536,
    "device": "cpu"
  },
  "budget": {"max_output_tokens": 4096, "max_model_calls": 8},
  "seed": 17
}
```

Add the explicit `backend` and `retriever` configurations shown above. The
source must match the original pinned clean checkout. Supply the chosen router
snapshot and the fixed `no_retrieval.txt`, `single_retrieval.txt`, `ircot.txt`
prompts. Install the `adaptive-rag-fit` extra in the execution environment;
this does not start training. `input_token_upper_bound` must cover the serving
context capacity: it is a conservative reservation, not reported token usage.
The optional `max_input_tokens` explicitly bounds the native router input;
omitting it preserves the original tokenizer limit.

```sh
agentinputlab config prepare-adaptive template.json \
  --router-checkpoint your-actual-router-name --output-dir prepared-adaptive
agentinputlab config run prepared-adaptive/run.json --query "Your question"
```

Preparation verifies source, prompt and component closures and records the
actual configured backend/retriever fingerprints; it does not load the router
weights, classify, retrieve, generate or fit. It writes a new directory and
never overwrites an existing preparation. A backend fingerprint records the
declared deployment, not remote attestation of served weights. Changing the
backend, retriever or runtime requires a new preparation.

Execution uses the original controlled driver, native first-step A/B/C router,
selected generator/retrieval arm and budget supervisor. It runs locally in the
chosen Python environment, not an OS sandbox. It is not the common-support
profile or a claim of upstream FLAN/Elasticsearch arm parity. The original
**question-only** restriction remains: instructions, messages and preloaded
context are rejected rather than silently removed. `config run` returns the
composite answer/trace, not a benchmark score. `config evaluate` can use a
compatible question-only adapter and native scorer; it does not convert a rich
benchmark state into question-only input for you. Classifier and generator
calls are charged once, failures are retained, and no retry is automatic.

### Adaptive-RAG: replay saved router records

For `acquire.adaptive_rag.selector_replay`, put the saved record array in
`parameters.records`, together with `split`, `dependency_hash`, `artifact_id`
and `artifact_sha256`. Records use the JSON representation of
`AdaptiveRAGRecord`: each `router_prediction` is an object and `contexts` is an
array of context-block objects. You do not need to construct Python dataclass
objects or inject a runtime. Run the file with
`agentinputlab config run run.json --state visible-state.json`.

The original checks still reject gold/silver/oracle-derived routes, mismatched
strategies or dependency bundles, mixed splits, and duplicate IDs. Task IDs and
queries must match the saved records. This profile reuses recorded routes and
contexts; it does not run the classifier, regenerate missing records, or insert
the saved answer as evidence. Historical cost remains record metadata, not a
new model call. The inline identity fields are declarations, not verification
of an external file: verify the original artifact and preserve its provenance
before exporting its records into the configuration. The configuration's 1 MiB
file limit also applies to inline records.

Alternatively, keep the records in a checksum-pinned JSON file (up to 4 MiB and
10,000 records). Its top-level object contains `records` and `dependency_hash`;
each record has the same dependency hash. Replace the placeholders below with
the identities from your verified artifact receipt:

```json
{
  "release_id": "acquire.adaptive_rag.selector_replay",
  "parameters": {
    "split": "dev",
    "dependency_hash": "<dependency bundle SHA-256>",
    "artifact_id": "my-frozen-selector",
    "artifact_sha256": "<selector.json file SHA-256>"
  },
  "artifacts": {"records": {
    "path": "./selector.json",
    "sha256": "<same selector.json file SHA-256>"
  }}
}
```

The path is relative to the run configuration. File loading verifies the bytes
and applies the same typed-record guards as inline input. Do not also supply
`parameters.records`. Inspection does not open files; execution does. A missing
record produces an error instead of silently fitting or generating one.
Use `config evaluate` with the same artifact configuration and an explicit
backend/predictor to obtain downstream benchmark scores. `config run` alone
only applies the saved intervention. Replay does not establish live-classifier
compatibility or make the original router/strategy generation costs disappear.

### InfoCQR: reuse a saved rewrite cache

The `rewrite.infocqr.cached_rw_zsl`, `rewrite.infocqr.cached_rw_fsl` and
`rewrite.infocqr.cached_edit` profiles accept the original JSONL cache:

```json
{
  "release_id": "rewrite.infocqr.cached_rw_fsl",
  "parameters": {"split": "test"},
  "artifacts": {"cache": {
    "path": "artifacts/rewrites.jsonl",
    "sha256": "REPLACE_WITH_THE_TRUSTED_64_HEX_FILE_DIGEST",
    "artifact_id": "my-frozen-rewrites",
    "dataset_revision": "my-dataset-revision"
  }}
}
```

Run with `config run run.json --state visible-state.json`. The task ID,
query, conversation, mode and split must match the saved record. A cache miss
does not generate a replacement. These profiles return an input patch, not a
benchmark score; original cache-generation costs remain separate from replay
costs. This is cache reuse, not a newly executed teacher or trained student.

For an IR cache captured with initially retrieved context, explicitly reproduce
that preparation before the rewrite and rank the rewritten query directly:

```json
{
  "schema_version": 1,
  "retrieval": {"top_k": 100, "query_mode": "query_only", "source_mode": "retriever", "score_mode": "rank"},
  "retrieval_prefill": {"top_k": 3, "max_document_chars": 10000}
}
```

Pass this predictor file to `config evaluate --predictor-config predictor.json`
with the original cache, dataset and corpus retriever. Choose the limits and
score encoding from the cache's source protocol, not from the resulting score.
`retrieval_prefill` uses the visible query, refuses to overwrite existing context,
and charges one retrieval before the intervention; final ranking charges another.
It adds no model reader and does not regenerate missing cache records. InfoCQR
checks the original query, conversational messages and `new_context` metadata;
retrieved passages are not part of that conversation hash. Preparation receipts
separately record full state and context hashes. It is mutually
exclusive with `patch_retrieval` and `composite_retrieval`; those bindings retain
their own preparation options. The original generation costs remain separate.

### LongBench native text evaluation

The six registered LongBench v1 cells can be evaluated through `config evaluate`.
Install `agentinputlab[longbench]` for the original scoring dependencies. To prepare
missing official sources and evaluate using a shared cache:

```sh
agentinputlab config evaluate run.json --benchmark longbench.qasper \
  --split test --limit 1 --cache-dir ./toolkit-cache --accept-license-review \
  --output new-results.jsonl
```

The six subsets share the same three hash-locked source files (113,943,156 bytes
on a cold cache). Warm caches are verified and reused; `--offline` forbids
downloads. Corrupt files are rejected, not silently replaced. To prepare without
model calls, use `datasets prepare longbench.qasper --cache-dir ./toolkit-cache
--accept-license-review`. Alternatively, bind existing official files in a source
configuration; relative paths are resolved
beside that file, not beside the shell's working directory:

```json
{"data_zip":"data.zip","scorer_path":"metrics.py","prompts_path":"dataset2prompt.json"}
```

```sh
agentinputlab config evaluate run.json --benchmark longbench.qasper \
  --split test --limit 1 --longbench-config longbench-sources.json \
  --output new-results.jsonl
```

All three files must match the adapter's pinned SHA-256 values. This source-file
entry does not download them or accept arbitrary scoring code. Do not combine
`--longbench-config` with `--cache-dir`. This is an evaluation command,
not a new training split. Patch profiles use the original dataset prompt template
and the native text scorer, without converting answers into a QA JSON schema.
Composite profiles score their own final text, without an extra reader call;
their record explicitly does not claim baseline-template generation.

Optional `--predictor-config` JSON is
`{"schema_version":1,"native_text":{"max_output_tokens":512,"temperature":0.0}}`.
The per-call cap is also bounded by the remaining shared run budget. There is no
silent input truncation or automatic retry. Empty/wrong text is passed unchanged
to the original scorer; provider and scoring exceptions retain failure records.

For MultiFieldQA-en and QMSum, Query2doc's QueryGym live release and ReFormeR's
repository-native R-only release support rewrite-driven retrieval from the
original visible document. Configure a chunked `visible_context_bm25`
retriever and set `native_text.preserve_original_question` to `true` and
`native_text.rewrite_retrieval_top_k` to the desired passage count. The rewrite
selects evidence; the native reader answers the original question. ReFormeR
also takes `native_text.reformer_context_top_k` to select the initial passages
for its pattern selector; set it to the release's configured document count.
Both initial and rewrite retrievals count against the shared retrieval budget.

For an answer-producing profile whose declared response is the four-field QA
JSON object (for example Q-RAG's instruction-preserving reader), explicitly set
`native_text.composite_output_format` to `"structured_qa_json"`. Only its decoded
`answer` is then passed unchanged to the native LongBench text scorer, including
code indentation; support/evidence fields are not LongBench metrics. Set
`native_text.allow_json_fence` to `true`
only to accept one complete JSON code fence. Both options are recorded; the
default remains unmodified text. Invalid JSON/schema is a retained failure,
never a fallback answer. This adds no model calls and does not change the prompt,
budget, saved response, or native scorer. Patch profiles reject this composite
option before execution.

### Plan-and-Budget: choose cached or live execution explicitly

For `plan.plan_and_budget.native_cached_soft_budget`, use
`"parameters": {"total_word_budget": 200}` and bind `artifacts.records` to
`{"path": "plans.json", "sha256": "..."}`. The file uses the native
`{last_question_total, stats: [...]}` envelope. The visible state's exact task ID
and query must match a record. This profile produces a budgeted input patch.

For the answer-producing `plan.plan_and_budget.full`, configure its native
parameters and runtime:

```json
{
  "release_id": "plan.plan_and_budget.full",
  "parameters": {"output_format": "A concise plain-text answer."},
  "runtime": {
    "kind": "plan_and_budget_native",
    "source_root": "upstream/P-and-B"
  },
  "budget": {"max_output_tokens": 4096, "max_model_calls": 8},
  "seed": 17
}
```

Add your `backend` configuration as above, install the `plan-and-budget` extra,
and supply the pinned checkout described in [the full workflow](plan-and-budget-full.md).
Without `artifacts.records`, this explicitly runs live preparation and solving.
With `artifacts.records`, it reuses the exact matching plan. Missing cached
records never silently trigger new preparation calls. No dynamic Python imports
or shell commands are accepted in artifact/runtime configuration.

The separate `plan.plan_and_budget.hard_cap_extension` also accepts
`artifacts.records` in the same checksum-bound native file format. It uses the
existing per-step **output-token** caps and final synthesis, not the native
soft-word solver. Set its `output_format` and backend; it does not use
`runtime.kind=plan_and_budget_native`. An exact task ID/query match reuses the
plan without repeating preparation calls; a mismatch stops before generation.
Each cached plan must contain one to five nonempty steps. A larger upstream
cache can contain incompatible plans; the loader does not silently drop them.
Omit `artifacts.records` to explicitly choose live preparation. Reusing a plan
does not erase its original preparation cost or make cached and live runs an
identical scientific comparison.

#### Composite query output for retrieval (0.1.0a11)

Version `0.1.0a11` adds an explicit retrieval binding for answer-producing
profiles. Earlier versions do not expose this binding through `config evaluate`.
For example, set Plan-and-Budget's `parameters.output_format` to
`Return exactly one JSON object with a nonempty string field "query".` and
configure the source-bound retriever for the evaluated benchmark. Then pass
this predictor file with `--predictor-config ir-query.json`:

```json
{
  "schema_version": 1,
  "retrieval": {"top_k": 100, "query_mode": "query_only", "source_mode": "retriever"},
  "composite_retrieval": {"output_format": "json_query", "allow_json_fence": false}
}
```

The evaluator decodes the composite's final query, retrieves once, and uses the
benchmark's native scorer. It does not call a second reader or substitute the
original input query. Model and retrieval calls share the configured budget.
`allow_json_fence: true` explicitly accepts one JSON code fence and records
`single_json_fence.v1` in `downstream_metadata`; the default is strict JSON.
Malformed JSON, duplicate keys, prose, and answer objects are still failures.
Retain strict results separately when using the compatible view. Patch profiles,
QA tasks, and context-only retrieval cannot use this composite binding.

Plan-and-Budget `full` and `hard_cap_extension` can explicitly retrieve initial
context before planning. Add `"context_prefill": {"top_k": 3,
"max_document_chars": 10000}` inside `composite_retrieval`. This option is off
by default and requires a configured corpus retriever and an initially empty
stateless context. It uses the visible query, preserving document IDs, clipped
text and corpus revisions; it does not inject titles, scores, metadata or gold.
The prefill retrieval and final query retrieval both consume the existing
retrieval-call budget. Model/token limits are unchanged. Results record source,
prepared and execution-state hashes, clipping and retriever provenance. Missing
documents or insufficient budget produce a terminal failure without retry or
silently switching to an empty-context method. Preserve historical preprocessing
costs when comparing with older runs that did not count prefill in their ledger.

#### Paid query reader after an input patch (0.1.0a11)

An IR planner can append an advisory plan without changing `state.query`.
To have a model turn the complete patched visible state into a search query,
explicitly configure `patch_retrieval` instead of `composite_retrieval`:

```json
{
  "schema_version": 1,
  "retrieval": {"top_k": 100, "query_mode": "query_only", "source_mode": "retriever"},
  "patch_retrieval": {
    "output_format": "json_query",
    "max_tokens": 512,
    "temperature": 0.0,
    "allow_json_fence": false,
    "context_prefill": {"top_k": 3, "max_document_chars": 10000}
  }
}
```

Pass this file to `config evaluate --predictor-config`. The run configuration
must supply both a model backend and a corpus retriever. This is a separately
recorded, paid downstream call using the visible query, instruction, passages
and messages, never evaluator gold or task metadata. It uses the frozen
`fixed-source-blind-IR-query-reader-from-complete-patched-state.v1` protocol.
The helper and reader share the run's model/token budget; the reader cannot
exceed the remaining allowance. Optional prefill and final ranking also share
the retrieval budget. Remove `context_prefill` when the workflow needs no
initial retrieval. Prefill cannot replace existing context.

The reader must return a nonempty JSON `query`; malformed output is retained
with its cost and fails without retry or fallback. The native benchmark scorer
evaluates the resulting ranking. This option is off by default and applies only
to patch-producing retrieval workflows, not answer-producing composites, QA,
or native tool arenas. It does not alter the existing `query_only` predictor.

### Native DecomP from a file

`plan.decomp.full` and `plan.decomp.planner_component` accept the same explicit
native worker binding; the latter executes only the native planner transition.
They do not silently substitute the paper's controlled prompt adapter.

```json
{
  "release_id": "plan.decomp.full",
  "parameters": {"config_artifact_id": "decomp.commaqa.coarse.p1"},
  "backend": {
    "kind": "decomp_completion",
    "model_id": "your-model", "base_url": "http://127.0.0.1:8000/v1",
    "deployment_identity": {"revision": "your-model-revision"}
  },
  "runtime": {
    "kind": "decomp_live", "checkout": "upstream/DecomP",
    "python_executable": "decomp-worker/bin/python", "nltk_data": "assets/nltk_data",
    "per_call_output_cap": 512, "timeout_seconds": 300
  },
  "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
  "seed": 17
}
```

Prepare the separate Python 3.10 worker and pinned assets as described in the
[native worker instructions](../workers/decomp/README.md). Paths resolve relative
to the configuration file. The provider must implement `/completions`, including
the requested token log probabilities; a chat-only endpoint is not equivalent.
Native sampling, stop strings and output caps remain unchanged; `extra_body`
overrides and `final_content` stop adaptation are rejected. A finite cumulative
input-token budget additionally requires `runtime.per_call_input_token_bound`,
an upper bound enforced by the provider. No automatic retry or asset download
occurs during a run.

Use `config run` with a visible query/context state. Native config IDs are
task-specific, not interchangeable universal QA prompts. Instructions/history
are not silently discarded: incompatible inputs are rejected. Full execution
returns its own answer, while the planner produces an input patch; benchmark
scoring still requires the corresponding dataset's evaluation path.

### Native LLMCompiler from a file

`plan.llm_compiler.full` and `plan.llm_compiler.planner_component` can bind the
existing native `LLMCompilerLiveWorker` without constructing Python objects:

```json
{
  "release_id": "plan.llm_compiler.full",
  "parameters": {
    "prompt_hash": "CONTENT_HASH_OF_EXPORTED_PROMPTS",
    "controls": {"max_tasks": 8, "max_replans": 1, "concurrency": 4,
                 "timeout_seconds": 300, "streaming": false}
  },
  "runtime": {
    "kind": "llmcompiler_live",
    "checkout": "upstream/LLMCompiler",
    "python_executable": "compiler-env/bin/python",
    "prompt_bundle": "compiler-prompts.json",
    "prompt_bundle_sha256": "SHA256_OF_EXPORTED_BUNDLE_FILE",
    "per_call_output_cap": 512,
    "timeout_seconds": 400,
    "tools": {"kind": "visible_context_search", "source_revision": "YOUR_DATA_REVISION"}
  },
  "budget": {"max_model_calls": 8, "max_tool_calls": 8, "max_output_tokens": 4096},
  "seed": 17
}
```

Add the explicit `backend` configuration. Use the pinned native checkout,
exported prompt bundle and isolated worker dependencies described in
[LLMCompiler worker](../workers/llmcompiler/README.md). Paths are relative to the configuration file;
the worker Python path preserves virtual-environment symlinks. The prompt hash
is `content_hash(bundle["prompts"])`, distinct from the bundle file SHA-256.
Validation does not run a worker or model; execution checks the bundle and the
existing worker attests its native source and dependency versions.

Run `agentinputlab config run run.json --state visible-state.json`. The explicit
search binding uses only that state's visible context, the existing BM25
implementation and at most five passages; it is **not** live Wikipedia or a
full-corpus search. It executes no shell tools and has no network tool access.
Python callers may omit `runtime.tools` and supply a read-only `context.tools`
connector instead, but cannot configure both. Each case gets its own tool and
worker state. Model calls use the configured per-call cap and total case budget.
Instructions/history are not silently removed: the native profile's input
constraints still apply. `config run` returns the composite answer/trace, not a
benchmark score; any benchmark evaluation must additionally preserve its native
input and scoring contract.

### Reasoning models and stop strings

The default `backend.stop_policy` is `"provider"`: the provider receives the
method's stop strings unchanged. For a reasoning endpoint that separates
reasoning from final content, explicitly set `"stop_policy": "final_content"`
inside `backend` to apply those stops only to the parsed final text. This avoids
a logical stop inside reasoning terminating generation before the final answer.

This named compatibility policy changes the wire request and backend fingerprint.
It preserves seed and token caps, charges **all** generated tokens (including
tokens past the logical stop), and does not add a retry. It supports buffered
generation, not streaming. An empty or length-truncated final answer remains a
failure; reasoning text is never substituted for the answer. Record this policy
when comparing runs and keep previous results separate.

## Run through benchmark scoring

For saved workflows using the native `ContextIRQueryPredictor`, explicitly set
`patch_retrieval.protocol` to `agentinputlab.context_ir_query_endpoint.v1`
in the predictor JSON. This selects the original system/user prompt and native
rank-preserving retrieval predictor, with fixed temperature zero, the remaining
joint budget, and the existing explicit `allow_json_fence` decoder option.
Omitting `protocol` retains the existing patch-query reader unchanged; the two
prompt protocols are not interchangeable acceptance results.

Where the original input uses scored retrieval context, set
`patch_retrieval.context_prefill.projection` to `native_ir_context.v1` alongside
`top_k` and `max_document_chars`. This retains document IDs, clipped text, scores,
rank and corpus revision, using the visible dataset identity as the block source.
The retrieval is charged once before intervention. It does not inject document
titles, arbitrary metadata or evaluator gold. The default remains text-only
context; neither projection replaces an already populated state.

`config run` returns a patch or an architecture answer. For a complete static
benchmark run, use `config evaluate` with the same method/backend/artifact config:

```bash
python -m agentinputlab config evaluate ace-run.json \
  --benchmark 2wiki.multihop --split dev --limit 2 \
  --cache-dir ./benchmark-cache --offline --output ./ace-evaluation.jsonl
```

Use a new output path for every intentional run. Removing `--offline` enables
bounded dataset preparation when the cache is absent (512 MiB download default);
it never downloads model weights or fits a skill/playbook. Dataset terms may
require `--accept-license-review`. `--limit` selects a stable prefix, not a
statistical sample; repeat `--case-id ID` instead for exact cases. These options
limit execution, not the size of the underlying dataset download.

The JSONL file flushes a started marker before each case and a terminal record
afterwards. It includes selected IDs, native scorer provenance and metrics,
known usage, redacted monitoring, and captured model requests/responses for
later diagnosis/reuse. A parsing failure retains the model text and token cost.
Provider failure without usage is marked unknown/unpriced, not free. Existing
output files are never overwritten; there is no automatic retry or resume.

Intervention and downstream execution share the configured **per-case total**
budget. The reader is limited to the remaining output tokens/calls. A full
answer-producing architecture is scored without an additional reader call.
Failures remain in `selected_count` and per-case records; means are explicitly
`metrics_over_scored_cases`, not a fabricated all-case success rate. CLI exit
status is 1 if any case fails or awaits clarification, 2 for invalid configuration, and 0 if every
selected case reaches scoring (a valid score can be zero).

An explicit `REQUEST_CLARIFICATION` patch ends that static case with
`status: awaiting_clarification` and the actual questions in `clarification`.
No reader call or scoring occurs until an external answer is available; the
runner does not fabricate an answer from benchmark gold. The batch reports
`awaiting_clarification_count` separately from `failed_count` and `scored_count`,
retaining the original selected-case denominator and all consumed usage. Other
selected cases continue independently. Static evaluation does not automatically
resume a saved request; use the native interactive environment when evaluating
an interaction protocol, or retain this explicit boundary as the case outcome.

The native static binding covers `musique.answerable`, `2wiki.multihop`,
`hotpotqa.distractor`, `qasper.v0_3_dev`, `beir.arguana`, `bright.earth_science`,
`bright.economics`, the four BEIR archive entries described above, and
`bfcl.v3.simple_python` with its dedicated function-calling bridge described
below. The six `longbench.*` cells use the source-file/native-text binding
described above. Stateful arenas use `config evaluate-arena`, not the static QA reader.
Other entries still need their documented native runtime/scorer binding;
this list does not certify every algorithm/backbone combination.

For an existing source-locked FEVER or CQADupstack compatibility sample, use
`--beir-sample-config sample.json --accept-license-review --offline` instead of
`--cache-dir` or `--prepared-root`:

```json
{"root": "./existing-sample", "manifest_sha256": "<64-character SHA-256 of manifest.json>"}
```

The root is relative to this JSON file. The adapter verifies the manifest,
normalized data and source evidence; no data is downloaded or regenerated.
Only the sample's `test` split is supported. Its gold-enriched candidate pool
is for compatibility testing: output fingerprints retain
`full_dataset_score=false` and `metrics_scope=sampled_candidate_pool_only`.
The sample cannot be mixed with another dataset source or a full BEIR archive
retriever. Supply an explicitly matching retriever in the run configuration.

Optional `--predictor-config reader.json` controls the downstream reader, e.g.:

```json
{"schema_version": 1, "qa": {"max_tokens": 512, "temperature": 0.0, "use_seed": true,
 "output_format": "structured_json", "identifier_policy": "native_sets.v1"}}
```

If the final structured answer is wrapped in a single JSON Markdown code block,
add `"allow_json_fence": true` inside `qa`. This is off by default and is recorded
in the decoder audit and predictor fingerprint. It removes only the outer fence;
surrounding commentary, multiple objects, duplicate keys and invalid schemas
remain errors. It never retries the model or changes the generated answer.

The explicit Q-RAG instruction-preserving QA reader passes an empty answer
string to the benchmark scorer as returned. It does not turn an absent answer,
a non-string answer, or malformed JSON into a prediction.

Choose `answer_only` only for an explicitly answer-only protocol; it does not
invent support/evidence annotations. Retrieval settings use the existing
`retrieval` object (`top_k`, `query_mode`, `source_mode`, `score_mode`).
`score_mode` defaults to `"raw"`: pass the retriever's scores to the native scorer.
For a protocol defined by the retriever's ordered IDs, explicitly use
`"score_mode": "rank"` to encode that order as decreasing scores (N through 1).
This preserves the returned order even when raw scores tie; it does not rerank
documents or change relevance labels. The choice is recorded in the predictor
fingerprint and rank-mode result metadata. Choose it from the retrieval protocol,
not by comparing evaluation scores. Visible-context predictions under
`source_mode=context_or_retriever` already encode context order; `score_mode`
controls the retriever branch only. A retrieval run also
needs its configured retriever. Pure retrieval with a non-generating method
(for example `baseline.no_op` or `plan.plan_and_solve.ps_prompt`) needs no
`backend`, API key, or model server. Omit the `backend` block entirely; no dummy
endpoint is needed. A generating rewrite method or QA reader still requires its
model/runtime. Explicit saved-answer scoring can omit the backend when the
selected method itself requires no model generation.
The Python API is
`config.evaluate(benchmark_id, split=..., limit=2, cache_dir=...)`; it additionally
accepts a caller-owned native `adapter` and dependency `context`.

For `curate.acon.history_runtime` on static QA, add
`"history": {"kind": "acon"}` to the predictor configuration above. Prepare
the source-bound ACON worker with `config prepare-worker` as usual, then run
`config evaluate` with the prepared run file and `--predictor-config reader.json`.
This binds ACON to the reader's actual system/user messages, after rendering the
visible question and context, rather than to the adapter's initially empty
message list. It does not invent a system message or expose evaluator answers.
Each case has a fresh history and shares its total budget with the reader;
optimizer errors stop that case without silently falling back or retrying.

A static QA example has one user turn. ACON's native initial threshold skip
therefore preserves the original reader request and consumes no optimizer call.
The result includes `live_history_records` and its native skip decision; this is
an executable initial-boundary test, not a multi-turn compression measurement.
Use `config evaluate-arena` with its native history integration for actual
multi-turn environments. The static option requires the matching ACON profile,
a QA reader, and the reader's existing nonempty system prompt.

### BFCL Python-simple: native function calls and AST scoring

Use the same `config evaluate` command; missing BFCL data and the original,
checksum-verified v1.3 source archive are prepared automatically. The source bridge uses the
official Python AST checker and native function schema/decoder; it does not
execute generated code or claim aggregate leaderboard reproduction.

```sh
agentinputlab config evaluate run.json --benchmark bfcl.v3.simple_python \
  --split test --case-id simple_0 --case-id simple_1 \
  --cache-dir ./toolkit-cache \
  --output ./bfcl-first-run.jsonl
```

The cold source download is 44,552,456 bytes and is subject to the bounded
dataset loader. `--offline` requires cached sources. To reuse an existing
archive without copying or downloading, add
`--bfcl-source-archive ./data/official-v1_3.tar.gz --offline`. If you already
have normalized data, use `--prepared-root ./data/normalized` instead of
`--cache-dir`, together with that explicit archive. The original archive is
needed for scoring and must remain available; corrupt caches are not overwritten.

`run.json` selects the intervention, target backend, seed and total budget as
usual. The native visible function catalog is supplied automatically; evaluator
answers are not sent to the model or intervention. The backend must support
OpenAI-compatible `tool_calls`. The default downstream cap is 512 output tokens
at temperature 0, reduced if the intervention has consumed the shared budget.
An optional `--predictor-config` file can set
`{"schema_version":1,"function_calling":{"max_output_tokens":512,"temperature":0}}`.
QA-only final-answer architectures are rejected, not relabeled as function calls.

Missing tool calls are scored by the native checker (normally zero); transport
or malformed-envelope failures retain their recorded cost and are not retried.
Generated tool calls are saved alongside response text and usage for later
inspection/reuse. Existing output files are never overwritten. A custom Python
adapter can be supplied through `RunConfig.evaluate(..., adapter=...)`; the CLI
always constructs the source-verified BFCL bridge, not an arbitrary callback.

## Configure native Query2doc, Q-RAG and R2AG runtimes

### Query2doc author-released TREC caches

`rewrite.query2doc.official_cached` can read either of the two audited author
files directly, without a Python cache object or a model endpoint:

```json
{"release_id": "rewrite.query2doc.official_cached",
 "parameters": {"split": "trec_dl2019"},
 "artifacts": {"cache": {"path": "./trec_dl2019.jsonl",
   "split": "trec_dl2019", "accept_license_review": true}}}
```

Review the source and underlying dataset terms before setting the acceptance
flag. The original loader enforces the built-in file checksum, size, row count
and exact query match; no user-selected checksum can authorize a replacement.
The other supported split is `trec_dl2020`. Paths are relative to this config.
Use `config run FILE --state STATE` for expansion, or `config evaluate` with
the corresponding prepared TREC corpus/retriever for scoring. Expansion alone
is not a benchmark score. These pseudo-documents were generated by the author's
text-davinci-003, not the selected reader; original generation cost is unknown,
not zero. Missing cases do not trigger live generation. Use the separately
configured live profile for new queries.

For an existing bounded TREC DL compatibility bundle, create `trec-sample.json`
with its root and the exact `manifest.json` SHA-256:

```json
{"root": "./trec-sample", "manifest_sha256": "<64-character manifest SHA-256>"}
```

Configure the matching corpus retriever in `run.json` (for example the
checksum-bound `sqlite_fts5` retriever described above), then score explicit
sample IDs without downloading a full corpus:

```bash
agentinputlab config evaluate run.json --benchmark trec_dl.2019 --split test \
  --case-id CASE_ID --trec-sample-config trec-sample.json \
  --accept-license-review --offline --output results.jsonl
```

The sample root is relative to `trec-sample.json`. This path uses the native
TREC scorer, including grade-2 binary relevance, and verifies the source
queries, qrels and passages. Scores cover the fixed compatibility candidate
pool, not full-corpus TREC performance. Use `trec_dl.2020` with the matching
2020 bundle/cache. A wrong split, checksum or benchmark is rejected; no cache
miss generates a replacement answer. The sample source is mutually exclusive
with `--prepared-root`, `--cache-dir` and `--beir-sample-config`.

### Query2doc live execution

For `rewrite.query2doc.controlled_live`, use the same `querygym_query2doc`
runtime as the ordinary live profile, plus explicit `backend` and `retriever`
sections. You do not need to construct a Python runtime or copy fingerprints
by hand. In the preparation input, set `parameters.config` to the desired
QueryGym settings, for example:

```json
{"config": {"mode": "zs", "temperature": 0.7, "max_tokens": 256}}
```

Then prepare and run:

```bash
agentinputlab config prepare-query2doc controlled-query2doc.json --output-dir prepared-query2doc
agentinputlab config run prepared-query2doc/run.json --state visible-state.json
```

Preparation reads the pinned QueryGym source and the configured component
identities and writes a ready configuration. It makes no model/retrieval calls,
does not fit, and refuses an existing output directory. The ready file retains
the seed, total budget and method settings; changed model/retriever identities
require a new preparation. The shared retriever is bound but not called by the
rewriter; downstream retrieval and scoring use the normal evaluation path.
This file runtime supports native zero-shot (`zs`) and CoT (`cot`) modes.
Few-shot mode needs an additional training-data binding that this native runtime
does not implement; preparation rejects it rather than dropping demonstrations.

For interactive Tau tasks, use the native arena entry described at the end of
this page, not the static QA scorer.

`rewrite.query2doc.querygym_live` accepts a pinned QueryGym checkout directly:

```json
{"release_id": "rewrite.query2doc.querygym_live",
 "runtime": {"kind": "querygym_query2doc", "checkout": "./upstream/QueryGym"}}
```

Add your normal `backend` configuration to this file, then use `config run` or
`config evaluate`. No Python runtime object is needed. The existing source
revision and file checks remain enforced; this does not download an arbitrary
checkout or change QueryGym's prompts. Evaluation captures both the reformulator
and reader responses and charges their usage once against the shared budget.

`acquire.qrag.common_support` accepts its existing frozen-policy assets:

```json
{"release_id": "acquire.qrag.common_support",
 "runtime": {"kind": "qrag_common_support",
   "source_root": "./upstream/Q-RAG", "checkpoint_root": "./assets/policy",
   "embedder_root": "./assets/embedder",
   "compatibility_manifest_path": "./assets/compatibility_manifest.json",
   "compatibility_manifest_sha256": "<canonical manifest content hash>",
   "embedder_manifest_sha256": "<canonical manifest content hash>",
   "device": "cpu"}}
```

Replace the hash placeholders with the verified manifest identities; for this
native runtime they are `content_hash(json.loads(...))` from
`agentinputlab.contracts.serialization`, not the raw JSON file SHA. Add a
retriever supplying real candidate documents and, for QA evaluation, a backend.
Install `agentinputlab[qrag]` for the frozen-policy runtime. This is inference,
not new training; native checkpoint, source and embedder checks are unchanged.
Other Q-RAG profiles do not silently inherit this configuration.

For a real candidate pool containing exactly one document, opt in to the native
environment's candidate-exhaustion termination:

```json
"parameters": {"config": {"allow_candidate_exhaustion": true}},
"runtime": {"kind": "qrag_common_support",
  "single_candidate_axis_compat": true}
```

Merge these fields into the full configuration above. The selection cap stays
at two; the environment stops after its sole real candidate is selected. The
axis option fixes the pinned policy's singleton tensor shape without changing
weights. Empty pools remain errors; no document is padded or duplicated. The
default two-candidate boundary is unchanged. Record this opt-in separately from
earlier boundary failures; it does not recover a missing historical answer.

### Q-RAG selector plus reader from a configuration file

For `acquire.qrag.released_checkpoint_native` or
`acquire.qrag.reader_transfer_factorial`, select `runtime.kind="qrag_native"`.
These profiles produce their own final answer, rather than a context patch:
`config evaluate` scores that answer without making a second reader call.

```json
{"release_id": "acquire.qrag.reader_transfer_factorial",
 "seed": 17,
 "budget": {"max_output_tokens": 4096, "max_model_calls": 8},
 "parameters": {
   "artifacts": "<complete QRAGArtifactIdentity object for your assets and reader>",
   "controls": {"max_output_tokens": 4000},
   "acknowledgements": {
     "source_noassertion_acknowledged": true,
     "checkpoint_cc_by_4_0_acknowledged": true,
     "user_mounted_source": true, "user_mounted_checkpoint": true}},
 "backend": {"kind": "openai_compatible", "model_id": "your-reader",
   "base_url": "http://localhost:8000/v1",
   "deployment_identity": {"revision": "your-pinned-reader-revision"}},
 "runtime": {"kind": "qrag_native", "source_root": "./upstream/Q-RAG",
   "checkpoint_root": "./assets/policy", "embedder_root": "./assets/embedder",
   "compatibility_manifest_path": "./assets/compatibility_manifest.json",
   "python_executable": "./qrag-env/bin/python", "device": "cpu",
   "timeout_seconds": 600}}
```

Replace the `artifacts` placeholder with a JSON object containing the actual
`QRAGArtifactIdentity` fields, not the placeholder string. Its reader deployment
hash must match `content_hash(backend.fingerprint())`, where `backend` is the
public `OpenAICompatibleBackend` constructed with the same backend settings;
the identity also binds the selected reader prompt, original source, frozen
checkpoint, embedder, corpus and candidate index. Do not copy an unrelated
experiment's identities. Paths resolve relative to the configuration file.

If the frozen selector reaches a one-candidate input, including an input
that starts with exactly one candidate, set
`"single_candidate_axis_compat": true` in the `qrag_native` runtime block
to preserve the policy's candidate axis. This is an explicit live-policy
option. The default is unchanged, and the option cannot be combined with
`policy_replay_path`. Candidate order, selection budget and reader limits
are unchanged.

For a visible-state run, you can instead **remove** `parameters.artifacts`
from the template and generate it from your mounted assets and candidate texts:

```sh
agentinputlab config prepare-qrag template.json --state state.json \
  --reader-repository local://your-reader --reader-revision your-pinned-reader-revision \
  --reader-profile factorial_instruction_preserving_reader_transfer \
  --output-dir prepared-qrag
agentinputlab config run prepared-qrag/run.json --state prepared-qrag/state.json
```

Preparation verifies the original source, policy checkpoint and embedder manifest,
computes their identities and the configured reader fingerprint, and preserves
your controls, seed and budget. It reads the checkpoint for hashing, but makes
no policy, embedding or reader calls, downloads nothing and performs no fitting.
The reader repository/revision describe your explicit deployment; hashing a
configuration does not remotely attest the served weights. Acknowledgements
remain explicit. The generated configuration is bound to **that one visible
state**, using its existing candidate order with no live search; it rejects a
different state before starting a worker. Do not use it as an unrelated dataset's
identity or add a retriever that this profile does not execute. For a different
state, prepare a new directory, or use the batch entry below.
An explicitly supplied policy replay must match the
newly computed identity; preparation does not rewrite or silently replace it.

For a whole selected QA batch, keep the same **unbound** template (without
`parameters.artifacts` or `runtime.prepared_state_sha256`) and add `reader.json`:

```json
{"repository": "local://your-reader", "revision": "your-pinned-reader-revision",
 "profile": "factorial_instruction_preserving_reader_transfer"}
```

```sh
agentinputlab config evaluate template.json --qrag-reader-config reader.json \
  --benchmark 2wiki.multihop --split dev --limit 2 --offline \
  --prepared-root ./prepared-data --predictor-config predictor.json \
  --output new-results.jsonl
```

Use the normal QA predictor configuration described below (for example
`{"schema_version":1,"qa":{"output_format":"structured_json"}}`). The batch
verifies shared mounted assets and derives each case's identity from **its own
visible state**; gold remains with the scorer. The original worker rechecks
assets during execution. Each terminal row records `qrag_binding`, model
responses and usage. Failed cases remain in the selected denominator; they are
not retried. Without a replay selection this runs the frozen policy and reader
and incurs their normal costs. Python exposes the same route through
`RunConfig.evaluate(..., qrag_reader={...})`.

For **IR query generation**, explicitly use
`"profile": "factorial_native_query_reader_transfer"` in `reader.json` (or the
same `--reader-profile` value in `prepare-qrag`). This separately hashed reader
asks for exactly `{"query":"..."}`. Do not use the QA four-field instruction
profile for IR: its system instruction conflicts with a query-only task.
The frozen selector, visible passages, seed and configured budget are unchanged.
Keep the original benchmark's query decoder, corpus and scorer; a malformed
answer remains a failure, not a query recovered by dropping other fields.
This is an opt-in prompt variant, not an upgrade of old results or an assertion
that every backbone produces valid output. Old QA-reader responses cannot be
relabelled as outputs of this query-reader profile.

For an IR dataset without visible passages, add an explicit candidate source to
`reader.json` and a corpus `retriever` to the run configuration (for example the
BEIR archive configuration above):

```json
"candidate_source": {"kind": "retriever", "top_k": 6, "max_document_chars": 10000}
```

This retrieves once using the visible query, preserves returned order and IDs,
clips each passage to the declared character limit, and binds the resulting
per-case state before the frozen selector runs. The preparation is charged as
one retrieval call; the selector/reader receives the remaining budget, and
final query scoring uses the same corpus retriever. No gold labels are used.
The output records the source/prepared/execution state hashes, retriever
identity, candidate IDs, text hash and clipping count. Existing visible passages
are never overwritten: such a case fails and requires removing this option.
Without this option, the original given-passages behavior remains unchanged.
The optional `candidate_source.document_format` defaults to `"text"`, preserving
the document body exactly. Use `"title_text"` explicitly when the intended policy
input includes titles: it renders `Title: <stripped title>\n<stripped body>`
(or the stripped body for an empty title). The character limit applies to the
entire rendered passage, including its title. Both selector and reader see this
same rendered text. Saved policy responses must match these effective inputs
exactly; adding or dropping titles never bypasses the replay checksum.
For evaluation, also configure `composite_retrieval.output_format=json_query`
and retrieval `query_mode=query_only`, `source_mode=retriever` as shown in the
composite retrieval section. The declared candidate preparation is part of the
configuration being evaluated, not a silent replacement of an older protocol.

For LongBench's single visible long document, the instruction-preserving reader
can explicitly opt into this input preparation in `reader.json`:

```json
"longbench_context": {"kind": "lossless_words350_bm25_top6.v1"}
```

It constructs lossless 350-word chunks from **visible** context, selects at most
six by the visible query's BM25 ranking, and preserves the native task template
around the selected passages. It does not read gold answers or pad missing
candidates. This named projection requires `controls.max_candidates >= 6`, at
least two genuine chunks, and the instruction-preserving reader; do not combine
it with external `candidate_source`. The record includes original/prepared state
hashes, source character offsets, template hash and selected IDs. Corpus
construction is lossless, but top-6 selection is not a claim to use all context.
Preparation consumes one CPU retrieval call, recorded in total usage in addition
to the selector/reader's original logical usage; model-call and token caps and
the frozen input state's budget identity stay unchanged. Use LongBench's
`native_text.composite_output_format="structured_qa_json"` to score only this
reader's JSON answer, with `allow_json_fence=true` only when explicitly desired.
The same `config evaluate --qrag-reader-config ... --longbench-config ...`
command performs preparation, frozen selection, reading and native scoring.

To reuse saved policies, optionally add `"policy_replays": [{"path":
"./policy-a.json", "sha256": "<trusted file digest>"}, ...]` to `reader.json`.
Relative replay paths follow that file. Selection matches exact question,
ordered candidate IDs/text, seed and horizon, then verifies the full artifact
identity and trajectory. Duplicate keys are rejected; a missing match fails
that case with zero new calls and **never falls back to live inference**. Do
not also configure a second `runtime.policy_replay_path`. Replays preserve
their original logical usage; they are not new model generations.

Install this package and its `qrag` extra in the specified worker environment.
The installed worker runs as a separate process with a temporary home and
offline asset loading; it is not an OS/container security sandbox.

Then run `agentinputlab config run run.json --state state.json`, or the normal
`config evaluate` command with your prepared QA dataset. The native question-only
profiles reject task instructions they do not implement. To retain QA task
instructions and passage annotations, explicitly select
`factorial_instruction_preserving_reader_transfer` and its corresponding
`QRAG_INSTRUCTION_PROMPT_SHA256` in the artifact identity. This is a labelled
adapter, not a silent change to the original reader.

Live frozen-policy selection is the default. Optional `policy_replay_path` and
`policy_replay_sha256` select a saved trajectory explicitly; both are required.
The JSON schema is `agentinputlab.qrag.policy-replay.v1`, with `request`
(`question`, ordered `candidates` containing `document_id`/`text`, `seed`,
`max_steps`), `artifact_identity_sha256`, and the original policy `response`.
The file SHA is a raw SHA-256; identity hashes use canonical `content_hash`.
The worker checks the exact input, full artifact identity, trajectory, candidate
text hashes and attestations. A mismatch fails without live-policy fallback.
Replay retains original logical policy cost and is labelled in the worker
fingerprint; it is not a new independent policy execution. The reader still
runs once. Its effective output ceiling is the smaller of the configured
per-call cap and remaining total budget; failures and available raw responses
are retained without automatic retries.

`acquire.r2ag.common_support` can bind an existing backbone-matched R2AG artifact
without injecting a Python runtime object:

```json
{"release_id": "acquire.r2ag.common_support",
 "runtime": {"kind": "r2ag_common_support",
   "source_root": "./upstream/RRAG", "artifact_root": "./assets/r2ag",
   "manifest_sha256": "<raw artifact-manifest.json SHA-256>",
   "base_model_root": "./models/reader",
   "base_model_load_set_sha256": "<artifact manifest base_model_load_set_sha256>",
   "feature_model_root": "./models/e5",
   "feature_model_load_set_sha256": "<artifact manifest feature_model_load_set_sha256>",
   "reader_device": "cuda:0", "feature_device": "cuda:0"}}
```

Use the actual artifact manifest identities, not hashes from another backbone.
Add a retriever supplying candidate documents and a downstream backend for QA
evaluation. This binds the existing local reader and feature encoder; it does
not train a new artifact or enable the separate `controlled_same_candidates`
profile. Runtime dependencies are in `agentinputlab[r2ag-fit]` (installing the
extra does not run training). Existing explicit R2AG socket-worker environment
configuration still takes precedence over local execution; unset it to use the
local reader and encoder selected above.

The downstream backend can be an API model independent of the local reader:

```json
"backend": {
  "model_id": "your-api-model-id",
  "base_url": "https://your-provider.example/v1",
  "deployment_identity": {"revision": "your-pinned-snapshot"},
  "api_key_env": "DOWNSTREAM_API_KEY"
}
```

Keep the local reader's own artifact and load-set identities in `runtime`.
The runtime fingerprint identifies the reader and feature encoder; the backend
fingerprint identifies the API downstream. Reader calls and output remain part
of the method's allowance, with feature-encoder work recorded separately.

For a smaller real candidate pool, enable the explicit short-pool variant in
both the method and runtime blocks:

```json
"parameters": {"config": {"allow_short_candidates": true}},
"runtime": {"kind": "r2ag_common_support", "allow_short_candidates": true}
```

Merge this into the complete asset configuration above. It uses the existing
R²-Former position prefix and unchanged weights for the actual candidate count;
for a singleton, the absent-neighbor feature is explicitly zero. Empty pools
remain errors. This is a labelled compatibility variant, not a claim of the
original six-document profile's experimental parity. The pinned supported source
is verified before adaptation; no mounted source or checkpoint is rewritten.

For intentional reuse, `runtime.kind="r2ag_common_support_replay"` accepts
`response_path`, `response_sha256`, and `runtime_identity_sha256`. The saved
`agentinputlab.r2ag.common-response-replay.v1` envelope must contain the exact
request, response and original runtime identity. Query, ordered real documents,
provenance, seed and output cap must match. Original logical reader usage is
retained; physical reuse is marked and no live fallback is attempted.

### Explicit provider capabilities and UDCS decoding

Declare a server's supported capabilities using `backend.capabilities`, for
example `["chat", "completion", "seed", "json_schema"]`. This enables existing
schema-aware helpers, including Plan-and-Budget's complete step-credit schema.
It does not install server support or alter method/token budgets. Incorrect
capability declarations produce a provider error, without automatic retries.

For the two native UDCS routes on a GPT-OSS vLLM deployment whose Harmony chat
parser rejects otherwise generated responses, an explicit compatibility backend
is available:

```json
"backend": {"kind": "udcs_final_completion", "model_id": "gpt-oss-20b",
  "base_url": "http://localhost:8000/v1",
  "deployment_identity": {"revision": "your-pinned-deployment"},
  "api_key_env": "OPENAI_API_KEY"}
```

The server must expose `/tokenize` with `continue_final_message` and raw
`/v1/completions` with regex structured outputs. It renders its own final-channel
template with an opening `<think>` delimiter; one completion then uses the
native six XML fields. The grammar selects no action, probability or answer.
Model, seed, temperature and output cap remain caller-controlled; tokenization
is not a generative request. Re-entry into a model-private channel is rejected,
not promoted to an answer. Original HTTP failures and costs remain separate.
This decoding variant is opt-in, not a silent fallback or recovery of a prior
trial. A valid clarification request still waits for a real user response.
Ordinary downstream prediction continues through the normal chat endpoint,
without the UDCS XML grammar. Optional `extra_body.reasoning_effort` applies to
that downstream path; the helper uses the explicitly declared final prefill,
not a private-analysis channel. Both paths retain their actual API usage.

All runtime paths above resolve relative to the configuration file. Validation
checks configuration shape without loading weights or claiming runtime readiness;
execution verifies the assets. Missing assets are errors, not an automatic
substitution with another method or a new fit.

### RECOMP extractive execution

Both `curate.recomp.extractive` and
`curate.recomp.common_support_extractive` accept the existing extractive worker
through file configuration:

```json
{"release_id": "curate.recomp.common_support_extractive",
 "runtime": {"kind": "recomp_extractive_worker",
   "checkout": "./upstream/recomp", "checkpoint_root": "./assets/checkpoint",
   "checkpoint_manifest_path": "./assets/checkpoint-manifest.json",
   "checkpoint_manifest_sha256": "<raw checkpoint manifest SHA-256>",
   "compatibility_manifest_path": "./assets/compatibility_manifest.json",
   "compatibility_manifest_sha256": "<raw compatibility manifest SHA-256>",
   "device": "cpu"}}
```

Also supply the method's `checkpoint_revision`, `tokenizer_revision`,
`asset_manifest_hash` and explicit `asset_license_acknowledged` parameters from
the chosen artifact. The sentence-level `extractive` profile additionally
requires its `sentence_splitter_revision` and prepared sentence context; the
common-support profile splits visible context itself. Bind a retriever and
provide a visible state for `config run`; QA evaluation additionally needs its
downstream backend. This uses the existing frozen encoder and current-runtime
loader, not new training. Asset paths resolve relative to the configuration file,
and both manifest hashes are checked before constructing the worker.

For `curate.recomp.abstractive` and `curate.recomp.selective`, use the separate
native seq2seq worker binding (not the extractive loader):

```json
{"release_id": "curate.recomp.selective",
 "parameters": {"checkpoint_revision": "<from checkpoint manifest>",
   "tokenizer_revision": "<from checkpoint manifest>",
   "asset_manifest_hash": "<raw checkpoint manifest SHA-256>",
   "asset_license_acknowledged": true},
 "runtime": {"kind": "recomp_abstractive_worker",
   "python_executable": "./recomp-worker/bin/python",
   "checkout": "./upstream/recomp", "checkpoint_root": "./assets/checkpoint",
   "checkpoint_manifest_path": "./assets/checkpoint-manifest.json",
   "device": "cpu", "timeout_seconds": 900},
 "budget": {"max_model_calls": 8, "max_output_tokens": 4096}, "seed": 17}
```

Add a reader `backend` and the upstream `retriever` identity. Supply already
retrieved, uniquely identified context blocks; this entry does not invent or
retrieve evidence on a missing context. The selective profile runs the original
compressor and bounded reader, sharing the configured budget; an empty successful
summary disables augmentation, while a compressor failure remains a failure.
The abstractive profile returns a context patch instead of an answer and also
requires `parameters.input_template_hash` equal to the exported
`RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256` constant.

Prepare the native Python 3.10 worker dependencies and safetensors checkpoint
as described in [the worker instructions](../workers/recomp/README.md). The
file entry uses the installed package's bundled driver with the specified
interpreter. It verifies the manifest digest and revisions before spawning;
the original driver then checks source, dependencies and the complete asset
closure. Validation does not load weights. Runs do not train, download assets,
retry failures or replace the native compressor with the reader model.
`timeout_seconds` is a process limit, not extra model-token or call budget.

To avoid copying revision strings and hashes manually, start with the same
runtime, retriever and (for selective) reader backend, but keep only
`asset_license_acknowledged: true` and any chosen controls in `parameters`:

```sh
agentinputlab config prepare-recomp template.json --output-dir prepared-recomp
agentinputlab config run prepared-recomp/run.json --state visible-state.json
```

Preparation verifies the pinned source and every manifest-listed checkpoint
file, derives the revisions, manifest digest and native prompt identity, and
writes a new `run.json` without inference or fitting. Conflicting supplied
identities are rejected, not overwritten. Preparation does not attest worker
dependency versions; the original worker checks those at execution. An existing
output directory is never overwritten. No model or private data is bundled.

### Existing process-worker methods

The following public file bindings use the existing subprocess drivers:

| `runtime.kind` | Supported `release_id` |
| --- | --- |
| `ask_or_assume_worker` | `clarify.ask_or_assume.intent_monitor_component` |
| `acon_observation_worker` | `curate.acon.common_support_observation`, `curate.acon.observation_runtime` |
| `acon_history_worker` | `curate.acon.history_runtime` |
| `reformer_worker` | `rewrite.reformer.repo_native_r_only`, `rewrite.reformer.paper_spec_q_concat_r`, `rewrite.reformer.controlled_r_only`, `rewrite.reformer.controlled_q_concat_r` |
| `infocqr_worker` | `rewrite.infocqr.common_support_rw_zsl` |

Each requires `checkout`, `driver_path`, `driver_sha256`, `python_executable`,
`python_executable_sha256`, `compatibility_manifest_path`,
`compatibility_manifest_sha256`, `base_url`, and `model_id`. Use the corresponding
installed `workers/<method>/driver.py` and its manifest; hashes are raw-file
SHA-256 values. Paths resolve relative to your configuration file. The Python
executable retains its virtual-environment path, even when it is a symlink.
Only run drivers and interpreters you trust; hashes identify files, not their safety.

To avoid entering these file hashes manually, keep `runtime.kind`, `checkout`
and `python_executable` in your template and run:

```sh
agentinputlab config prepare-worker template.json --output-dir prepared-worker
agentinputlab config run prepared-worker/run.json --state visible-state.json
```

`prepare-worker` chooses this installation's bundled driver and manifest, checks
the manifest-listed upstream source files, hashes the explicit worker Python,
and derives the worker endpoint/model from `backend`. Existing conflicting
identities are rejected, not replaced. Virtualenv interpreter paths are retained.
No worker, model, download or training is started. Keep method parameters,
artifacts and (for ACON) revision fields explicit. This command prepares the
transport binding, not missing fitted artifacts or worker dependencies; the
worker still checks its full native compatibility contract when run. An
existing output directory is never overwritten.

Provide the normal `backend` section for the target model identity. For
Ask-or-Assume, ReFormeR and InfoCQR, specify `api_key_env` (an environment variable
name, never the key itself). ACON instead accepts `require_api_key`; when true,
it reads `AGENTINPUTLAB_ACON_API_KEY`. ACON also requires
`optimizer_model_revision`, `target_model_revision` and `tokenizer_revision`,
and its compatibility manifest must be next to its driver. Runtime
`max_output_tokens` and `timeout_seconds` configure the worker; they do not
increase the enclosing run budget. Keep the worker endpoint/model and backend
identity consistent.

For native ACON observation compression, use `acon_observation_worker` with
`curate.acon.observation_runtime` and a state containing the selected context
block. For history compression, use `acon_history_worker` with
`curate.acon.history_runtime` and the captured `messages` beginning with system
then user. Both accept `parameters.artifacts` and `parameters.controls`; the
same pinned driver performs their original preflight and compression. Retain
`acon_original_first_user_prompt` and `acon_previous_history_summary` in state
metadata when continuing an already summarized session. `config run` compresses
that supplied state once: it does not automatically install a live history hook
into every arena's agent loop.

The four ReFormeR profiles use the same pinned process worker, with the selected
profile recorded in its identity. `repo_native_r_only` and `paper_spec_q_concat_r`
select a pattern automatically; the two `controlled_*` profiles require an
explicit `parameters.manual_pattern_id` from the bound pattern library. The
`r_only` profiles replace the query with the rewrite, while `q_concat_r` retains
the original query followed by that rewrite. Selecting a profile does not refit
patterns or substitute a different pattern artifact.

These bindings do not supply method inputs automatically: for example, ReFormeR
still needs its pattern artifact and ACON its guideline identity. Inspect these
with `agentinputlab show RELEASE_ID --parameters`. They do not enable OpenHands,
student training, or another profile simply because it shares a method name.

## Run selected native arena cases

`config evaluate-arena` connects an input-patch profile to the original ToolTalk,
STATE-Bench, ToolSandbox, or Tau executor and scorer. It runs each selected case in a
fresh process, preserving the intervention and downstream costs under one agent
budget. It does not launch a research campaign or modify a paper.

For example, save a run configuration with your actual deployment identity:

```json
{
  "release_id": "baseline.no_op",
  "backend": {
    "model_id": "YOUR_SERVED_MODEL",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key_env": "MODEL_API_KEY",
    "deployment_identity": {"revision": "YOUR_MODEL_REVISION"}
  },
  "budget": {
    "max_model_calls": 8,
    "max_retrieval_calls": 8,
    "max_tool_calls": 8,
    "max_output_tokens": 4096,
    "max_context_tokens": 262144
  },
  "seed": 17
}
```

Save a separate native configuration, `arena.json`:

```json
{
  "arena": "tooltalk",
  "checkout": "./upstream/ToolTalk",
  "max_output_tokens_per_call": 512
}
```

For an **already-paid cached continuation**, the arena can explicitly
reserve the historical helper cost from the same original budget. Ordinary
cache use does not do this automatically: cache loading itself makes no model
calls. Add this optional reference to `arena.json`:

```json
"prior_intervention_usage": {"path": "prior-usage.json", "sha256": "SHA256_OF_RECEIPT"}
```

The receipt has `schema_version: 1`, the exact `release_id`, `model_id`, `seed`,
and full `original_budget` from the run, plus a `cases` array. Each case supplies
`case_id`, `input_state_sha256`, `output_state_sha256`, `cache_sha256`, a
`source: {"path": "original-terminal.json", "sha256": "..."}` reference, and
`usage` containing the actual historical counters (for example `model_calls`,
`input_tokens`, and `output_tokens`). Source paths are relative to the receipt.
Use counters from the original terminal; do not invent missing usage or copy a
training batch's cost into every test case. These are caller-supplied provenance
receipts, not cryptographic certification of the source's authenticity.

This opt-in supports `rewrite.infocqr.cached_rw_zsl`, `cached_rw_fsl`, and
`cached_edit`. It also supports `acquire.adaptive_rag.selector_replay` with a
pinned `artifacts.records` file: use `artifact_sha256` in each receipt case
instead of `cache_sha256`. This must equal both the records file SHA256 and
`parameters.artifact_sha256`. Include the original router and strategy cost in
the historical usage when they shared the original episode budget. Inline-only
selector records cannot use this receipt option.

It verifies source/artifact hashes, case and run identities, original
budget and materialized state; missing, mismatched, or over-budget records are
rejected. It cannot be combined with live history or a newly billed intervention.
The native episode receives the remaining original budget. Terminal `usage`
includes the historical debit, while `current_run_usage` excludes it and
`prior_intervention` identifies the source. A historical call is never labeled a
new call. This option supplies neither predictions nor scores, and failures
remain failures. Without it, existing cache behavior is unchanged.

To acquire just the pinned ToolTalk source (no methods, other arenas, model
weights, or experiments), run:

```sh
agentinputlab campaign bootstrap --arena tooltalk.full --root ./upstream --allow-network
```

Set `checkout` to `./upstream/_arenas/tooltalk.full/checkout` in `arena.json`
(relative to that JSON file). Repeat the command without `--allow-network` to
verify/reuse the existing checkout offline. For STATE or ToolSandbox use
`--arena state_bench.common_support` or `--arena toolsandbox.common_support`;
repeat `--arena` to select several. Existing local edits are preserved and
reported, not overwritten. Source readiness is not dependency or model readiness:
this command does not install upstream packages or scoring-model weights.

The checkout and its native dependencies must be installed in the same
Python environment as the toolkit. The provider checks its pinned source and
official inventory before executing any model calls. Paths are relative to the
configuration file. No model weights or private experiment artifacts are bundled.
Before executing a ToolTalk case, the original semantic scorer loads its pinned
DistilBERT weights using the standard Hugging Face cache configuration. Set
`HF_HOME` or `HF_HUB_CACHE` to reuse existing downloads. In offline mode, missing
scorer weights fail preflight before any agent call; in online mode the native
loader may fetch those weights. Inventory-only mode does not load this model.

### Isolate native arena dependencies

Native arenas need not share the Python environment that launches your CLI.
For example, the pinned STATE source requires Python 3.12 or newer, while
ToolSandbox pins different versions of several shared dependencies. Install
each native package and the **same toolkit wheel** in its own environment,
then set `worker_python` in its arena configuration:

```json
{
  "arena": "statebench",
  "checkout": "./upstream/_arenas/state_bench.common_support/checkout",
  "worker_python": "./state-env/bin/python",
  "environment_backend": {
    "model_id": "YOUR_ENVIRONMENT_MODEL",
    "base_url": "http://127.0.0.1:8001/v1",
    "api_key_env": "ENVIRONMENT_MODEL_API_KEY",
    "deployment_identity": {"revision": "YOUR_ENVIRONMENT_MODEL_REVISION"}
  }
}
```

`worker_python` is an executable path, not a shell command; relative paths are
resolved against the arena JSON file. Without it, execution uses the current
Python as before. With it, the intervention, native continuation and scoring
run in that environment. Install any selected method's dependencies there too.
Host `PYTHONPATH`, `PYTHONHOME` and user-site packages are not injected; API-key
environment variables and cache settings remain available. The worker checks
its toolkit implementation against the caller before native execution, records
its Python identity, and rejects a different build or missing executable without
falling back or retrying. This option selects an environment; it does not create
one or silently install or upgrade dependencies.

For STATE, the preparation command creates a separate dependency environment
using an explicitly selected Python 3.12 or newer:

```sh
agentinputlab config prepare-arena arena.json --environment ./state-worker \
  --python /absolute/path/to/python3.12 --allow-network
agentinputlab config evaluate-arena run.json --arena-config ./state-worker/arena.json \
  --list-cases --output-dir inventory-check
```

The command copies an existing clean pinned STATE checkout, or fetches the pinned
public source when it is absent. It installs the current toolkit version and the
native dependencies into `state-worker/venv`, checks dependencies and toolkit
identity, and writes an arena config with the same scientific settings. It does
not alter the source checkout, shared environments, model budget or fitted
artifacts. Development builds must supply `--toolkit-wheel /path/to/agentinputlab-...whl`
so the worker receives exactly the same implementation, not an older package
with the same version string. Method-specific dependencies are separate.

Repeat the same command without `--allow-network` to verify and reuse a completed
bundle; no installation is repeated. A changed, unrelated or incomplete bundle
is preserved and rejected. Failed preparation retains `failure.json` and bounded
installation diagnostics; choose a new directory after addressing the cause.
An installation receipt is not a model, dataset or scoring acceptance result.
For ToolSandbox, the same command selects its separate native dependency recipe
from `"arena": "toolsandbox"`; choose Python 3.10–3.12 (Linux Python 3.11 is
installation-tested). The recipe retains the upstream OpenAI 1.17.0 pin and adds
HTTPX 0.27.2: HTTPX 0.28 removed a constructor argument used by that SDK even
though dependency metadata allows that combination. Preparation checks actual
client construction without making an API request, as well as native scorer
import and `pip check`. It does not install Torch, CUDA or model weights.
Keep a separate bundle for each arena; STATE's dependencies are not downgraded.
ToolTalk uses the same preparation command with `"arena": "tooltalk"` on Linux
and Python 3.10-3.12 (clean-install verified on Python 3.11). Its isolated recipe
installs CPU PyTorch and the dependencies of the existing source-locked BERT-CLS
compatibility adapter; it does not install unused sent2vec gensim/spaCy extras
or CUDA runtimes. It retains the pinned official ToolTalk tools and scorer source.
The scorer uses DistilBERT revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
Preparation reuses the five exact cached files first and downloads missing files
only with `--allow-network` (weights are approximately 268 MB). It records their
hashes and tests actual semantic scoring before declaring readiness. This is a
scoring-model forward pass, not an agent generation or training attempt.

To reuse an existing Hugging Face **hub** cache, set `"scorer_cache": "/path/to/hub"`
in the ToolTalk arena JSON; relative paths resolve against that JSON. Without it,
preparation uses `./tooltalk-worker/scorer-cache` inside the selected bundle.
The generated config points to this cache and evaluates offline, so missing
scorer weights fail before agent calls rather than triggering a late download.
Repeat preparation offline to verify the environment and scorer hashes without
reinstalling. Other platforms can use an explicitly prepared `worker_python`.

```sh
agentinputlab config evaluate-arena run.json --arena-config arena.json \
  --list-cases --output-dir inventory-check
agentinputlab config evaluate-arena run.json --arena-config arena.json \
  --case-id ID_FROM_INVENTORY --output-dir first-evaluation
```

`--list-cases` captures the native inventory without model calls. Choose explicit
case IDs (repeat `--case-id`) or `--limit N` for execution. Output directories must
be new: receipts are never overwritten and failures are not retried. Every case
has a terminal receipt; zero scores remain valid scores, and failed cases remain
in the selected count. This is single-arm evaluation, not a paired comparison.

STATE (`arena: statebench`) additionally accepts `parameters.domains`, `split`,
`limit`, and `score_reasoning_effort`. ToolSandbox (`arena: toolsandbox`) accepts
`parameters.scenario_names`, `limit`, and `preferred_tool_backend`. Both require
an explicit `environment_backend` with `model_id`, `base_url`, `api_key_env`, and
`deployment_identity` for their native judge/user role. Those role costs remain
in the native trace separately from the tested agent budget. ToolTalk does not
take an environment model.

`backend.timeout_seconds` sets the evaluated agent's per-request transport
timeout for both intervention and native continuation calls. For STATE and
ToolSandbox, `environment_backend.timeout_seconds` separately sets the native
user/judge request timeout. Values must be finite positive seconds; omitted
values retain the existing clients' defaults. Neither option increases token,
tool or call budgets, changes the episode deadline, or enables retries.

The configured STATE entry also limits the native client's outer retry layer
to one attempt, including its nested JSON generation methods, for agent,
user and judge roles. A transport or parsing failure is returned without a
second generation; this setting does not change existing saved results or the
legacy native provider's default retry policy outside this configured entry.

`backend.extra_body` also reaches native continuation requests, and
`environment_backend.extra_body` independently configures STATE/ToolSandbox
user/judge requests. For example, `{"top_p": 0.9}` is forwarded to that endpoint.
ToolTalk/ToolSandbox retain their default `chat_template_kwargs.enable_thinking=false`
unless you explicitly replace `chat_template_kwargs`; STATE adds no such default.
For hosted APIs that require a different extension format, set
`"agent_extra_body_mode": "replace"` in `arena.json` to send exactly the
extensions from `backend.extra_body`, without inherited native defaults. STATE
and ToolSandbox support the separate `environment_extra_body_mode` for user/judge
requests. For example, combine this mode with
`"extra_body": {"thinking": {"type": "disabled"}}` for an endpoint that accepts
that option. Both modes default to `"merge"` and are included in the published
`0.1.0a15` package.
Omitting these options preserves existing requests. Extensions must be finite
JSON and cannot override native model identity, task input, tools, seed, temperature,
token caps, stop strings or response schema. Configure budgets through `budget`
and `max_output_tokens_per_call`, not `extra_body`. Endpoint support for a custom
extension is still required; a server rejection is returned, not retried.

For Tau Banking/Retail, `arena.parameters.agent_extra_body` can explicitly
separate the downstream agent's extensions from the input optimizer's
`backend.extra_body`. Omit it (or use `null`) to retain inheritance. Set it to
`{}` to send no helper-only extensions to the native agent, or provide a JSON
object to replace those extensions. For example, an optimizer can use
`backend.extra_body.chat_template_kwargs.enable_thinking=false` while
`"parameters": {"agent_extra_body": {}}` keeps the native agent's original
decoding configuration. This never changes the optimizer, environment-role
configuration, model, seed, token/call budgets, tools or scorer. Reserved native
request fields such as `max_tokens` and `stop` are rejected here too.

`backend.stop_policy` applies to intervention requests through the same backend
as `config run`: `provider` sends the method's stop strings to the endpoint;
`final_content` omits them on the wire and cuts only the parsed final text,
charging all returned tokens. Native agent/user/judge continuation requests do
not supply logical stop strings, so their tool/JSON decoding is unchanged.
This option does not promote reasoning text to an answer or add a retry.
`environment_backend` has no separate stop-policy option.

The entry rejects monetary caps that the
native executor cannot enforce. Composite profiles must use their own native
runner; they are not silently converted to an input-patch method. Recipe
evaluation profiles can reference frozen artifacts as described above; this
command never fits on the selected evaluation cases.

## Bring your own retriever or native runtime

For large corpora, dense retrieval or native workers, keep data out of the
1 MiB configuration file. Build the dependency in Python and inject it:

```python
from agentinputlab import InterventionContext, RunConfig

config = RunConfig.from_file("run-without-inline-retriever.json")
result = config.run(state, context=InterventionContext(
    backend=backend, retriever=my_retriever, external_runtime=my_native_runtime,
))
```

Here `state`, `backend`, `my_retriever` and `my_native_runtime` are your configured
objects; omit dependencies your profile does not need. The retriever implements
`retrieve` and `fingerprint`; see the retrieval backends and
[real execution guide](real-execution.md). Supplying a backend/retriever both in
the file and in the context is rejected. Config budget and seed are authoritative;
the original state is not mutated. Do not reuse an active budget ledger/monitor.

This local configuration interface does not enable arbitrary methods or backend
URLs on the SDK's HTTP server. Server operators still configure an explicit
allowlist and dependency contexts. A returned intervention/composite result is
not automatically a dataset score: use the benchmark's native adapter and scorer.

## Tau Banking Knowledge and Retail

Select `arena: tau_banking` or `arena: tau_retail` with the pinned Tau checkout.
`config prepare-arena` supports both using Python 3.12 or 3.13, as required by
the pinned upstream package. It installs the native text/knowledge dependencies
in an owned environment, verifies native tasks, environment construction and
scorer imports, and exports an arena configuration. It does not install voice
extras, model weights or training dependencies, and does not call a model.
Alternatively, select a prepared `worker_python` containing this same toolkit
build and the native dependencies.

```sh
agentinputlab config prepare-arena tau.json --environment ./tau-worker \
  --python /absolute/path/to/python3.12 --allow-network
agentinputlab config evaluate-arena run.json --arena-config ./tau-worker/arena.json \
  --list-cases --output-dir ./tau-ids
```

Use the generated arena file for capture and evaluation. Preparation verifies
the bundled BM25 path for Banking; a different retriever may need its own
explicit assets or service configuration. A readiness receipt does not certify
model answers. As with other arenas, development builds need `--toolkit-wheel`
and completed environments can be verified/reused without `--allow-network`.

```json
{
  "arena": "tau_retail",
  "checkout": "./upstream/tau2-bench",
  "worker_python": "./tau-env/bin/python",
  "environment_backend": {
    "model_id": "your-model",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "deployment_identity": {"revision": "your-model-revision"}
  },
  "parameters": {"split": "test", "limit": 40, "max_steps": 16,
    "max_errors": 3, "timeout_seconds": 600,
    "user_max_tokens": 1024, "judge_max_tokens": 512},
  "max_output_tokens_per_call": 512
}
```

Use the same model ID and deployment identity in `run.json.backend`. These
native profiles bind the user simulator and assertion judge to the evaluated
backbone, use local endpoints, and report environment calls separately from the
agent budget. They are not independent-judge leaderboard evaluations. Retail
keeps native seed 17, temperature 0 and zero retries; set `run.json.seed` to 17.
The evaluated agent and intervention share `run.json.budget`; explicitly set
`max_context_tokens`. Environment caps above do not increase the agent allowance.
For Banking, omit `judge_max_tokens` (the native judge uses the agent cap),
use `retrieval_config: "bm25"` and `retrieval_config_kwargs: {}` as applicable.
Banking's `train` selector does not establish a disjoint official training split;
declare development/heldout groups when preparing a fitter.

```sh
# List official IDs without user simulation or model requests.
agentinputlab config evaluate-arena run.json --arena-config tau.json \
  --list-cases --output-dir ./tau-ids
# Capture ONLY the selected initial user state, for later fit preparation.
# Unlike listing, this step calls the user simulator and records its usage.
agentinputlab config evaluate-arena run.json --arena-config tau.json \
  --case-id 5 --capture-only --output-dir ./tau-capture
# Capture the selected case, execute the intervention + native agent, and score.
agentinputlab config evaluate-arena run.json --arena-config tau.json \
  --case-id 9 --output-dir ./tau-evaluation
```

IDs above illustrate Retail; obtain valid IDs from your selected domain/split.
Capture-only output includes `capture/inventory.json`, per-case
`capture/native-configurations.json`, and `capture/capture-bundle.json` with
hash-pinned member files and original success/failure receipts. To reuse it,
add `"capture_bundle": {"path": "./tau-capture/capture/capture-bundle.json",
"sha256": "<bundle file SHA-256 from capture/terminal.json>"}` to `tau.json`.
Paths in the arena config resolve relative to that config file. Then select the
same case with `config evaluate-arena`; the initial user state is reused and
only the evaluated intervention/agent runs. The model, role caps, seed, split,
source/scorer and budget must match; local endpoint addresses may change.
An absent capture is an error, never an implicit model request. Saved capture
failures stay failures without retry. Original capture costs remain referenced,
not charged as new calls. Keep the bundle and its relative member files together.
Without `capture_bundle`, evaluation explicitly captures its selected cases.
Banking uses the native single-case
initial-capture seed 0; Retail uses 17, and both record that preparation seed
separately from the evaluated-agent seed. Capture failures are retained in the
selected-case denominator, never silently dropped or retried. A captured state
is not a scored result. New output directories are required throughout.

### Reuse an existing common-support Q-RAG policy draw

`acquire.qrag.common_support` can reuse an explicitly saved policy trajectory
without loading the policy weights again. This is separate from the composite
Q-RAG reader profiles and does not produce a new model response.

```json
{
  "kind": "qrag_common_support_replay",
  "policy_replay_path": "./policy.json",
  "policy_replay_sha256": "<SHA-256 of policy.json>",
  "policy_identity_sha256": "<content_hash of the original policy fingerprint>"
}
```

Use this as `run.json.runtime`. The saved JSON must contain exactly `schema`
(`agentinputlab.qrag.common-policy-replay.v1`), `request`, `response`, and
`policy_identity`. Save `request` using `to_jsonable(QRAGPolicyRequest(...))`,
the original `select(...)` response as `response`, and the original policy's
`fingerprint()` as `policy_identity`. Keep the original source evidence and
physical inference costs alongside this artifact; a checksum is a content
binding, not proof of provenance.

The request must match the question, seed, horizon, and full ordered retrieval
results, including document text, title, metadata, ranks and scores. The normal
trajectory validator still applies. Mismatches fail without a live fallback.
Trace metadata identifies saved reuse; reported trajectory usage is its original
logical cost, not new physical inference. `config run` returns the selected
state; use the appropriate benchmark evaluation command for downstream scoring.
