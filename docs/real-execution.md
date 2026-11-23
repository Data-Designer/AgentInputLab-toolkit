# Run with real backends and data

1. Choose an exact registered release profile, not only a paper method name. Inspect its entrypoint, constructor, required worker/assets and license. In Python, `inspect.signature(registry.resolve_entrypoint(release_id))` exposes constructor parameters.
2. Configure a real `OpenAICompatibleBackend` for a hosted API or self-hosted server, or an optional Transformers backend. Credentials come from the operator's environment, never package files. Preserve backend identity, model revision and actual usage.
3. Use the method's real dependencies: prepared retrieval corpus, tokenizer/compressor checkpoint or external worker. Do not replace missing artifacts with test fixtures. Minimal `pip install` intentionally does not download models or datasets.
4. Load a benchmark through its production adapter. Stateful benchmarks require the native environment, checkpoint restore/continuation and official scorer; a string rewrite alone is not benchmark execution. BFCL requires the official AST scorer bridge. Unsupported dataset families must be marked ineligible, not silently coerced.
5. Run paired NO_OP/intervention on the same frozen state, seed, model and declared budget. Record genuine failures, total cost and any retries separately. Synthetic smoke output must remain labeled synthetic.

Python extras include `benchmarks`, `evaluation`, `compression`, `dense-retrieval`, `selectors`, `local`, `qrag`, `clariq` and `acon`. Install only what your selected adapter needs, for example `python -m pip install '.[benchmarks,evaluation]'` in a checkout. Worker adapters and their compatibility manifests ship in the wheel and source distribution under the `workers` package; their upstream checkouts and checkpoints are not vendored. For example, `python -m workers.official_arenas.driver --help` works outside the source checkout after installation. An importable driver is not proof that its upstream runtime is configured or verified.

## Structured QA identifiers

The production QA predictor defaults to `qa.identifier_policy="native_sets.v1"`.
HotpotQA and 2Wiki support IDs accept canonical JSON strings, nested
`[title, sentence_index]` pairs, and flat title/non-negative-integer pairs.
2Wiki evidence IDs accept JSON strings or nested `[subject, relation, object]`
triples. The parser canonicalizes these representations and retains the first
occurrence of each ID. Other benchmarks use opaque string IDs.

Support IDs are checked against the visible context; 2Wiki triples retain the
native scorer's evidence semantics. Malformed typed entries and unknown support
IDs are excluded and hash-audited. Titles alone, stringified indices and flat
triple streams are never completed or inferred. The predictor records the
policy, raw-array hashes, normalization counts, duplicate counts and invalid-ID
counts. Strict JSON, the four required response fields and a non-empty answer
remain required.

Set `qa.identifier_policy="strict_strings.v1"` to reproduce the earlier parser's
string-only, duplicate-rejecting behavior and canonical-ID filtering. The policy
is part of the predictor fingerprint; generation requests, budgets and native
scorers are unchanged. Replay analysis under a new policy is recorded separately
from the original model attempt and its original outcome.

For a backend that wraps its final JSON in one Markdown code block, explicitly
set `qa.allow_json_fence=true`. This optional decoder removes only a complete
outer `json` or unlabelled fence. It does not extract answers from surrounding
prose, repair JSON, relax required fields, or accept duplicate object keys.
The default remains strict. Enabled runs record the envelope policy, whether a
fence was removed, and the original response hash; token costs and native
scoring stay unchanged. This option also applies to composite-method answers
scored through `config evaluate`, without adding a reader call.

## Dependency-locked live Query2doc

`rewrite.query2doc.controlled_live` calls the pinned QueryGym community
implementation, preserving its prompts, five-query-plus-pseudo-document
composition and explicit native fallback. It adds an immutable construction
binding for the shared generator, retriever, native runtime, method settings
and execution budget:

```python
from agentinputlab.algorithms.reformulate.query2doc import (
    PinnedQueryGymRuntime, Query2DocLiveBindings, Query2DocLiveConfig,
)
from agentinputlab import InterventionContext
from agentinputlab.api import create_intervention, run_intervention

runtime = PinnedQueryGymRuntime(querygym_checkout, backend)
context = InterventionContext(
    backend=backend, retriever=retriever, external_runtime=runtime, seed=17,
)
bindings = Query2DocLiveBindings.from_context(
    context, budget=state.budget, config=Query2DocLiveConfig(max_tokens=512),
)
method = create_intervention("rewrite.query2doc.controlled_live", bindings=bindings)
result = run_intervention(method, state, context)
```

The checkout must be the runtime's pinned revision; dependency and source
fingerprints must match before and after execution. Runtime identity declares
the same generator as the shared context. Bindings contain data, not executable
dependencies; `to_jsonable(bindings)` is also accepted as a factory/CLI registry
builder recipe. Capture bindings before observing test outcomes. A few-shot
configuration requires fixed demonstration IDs and a runtime that actually
implements that training-data binding; the built-in pinned runtime currently
implements zero-shot and CoT, and fails explicitly for unsupported few-shot use.

The rewriter does not perform retrieval. The benchmark driver must consume the
rewritten query using the bound shared retriever, its declared retrieval depth
and the remaining budget. The binding checks identity, not downstream execution
or independent weight-file verification. A native empty-generation fallback is
recorded as NO_OP with its attempted call and cost; it is not an unreported
retry or a benchmark success. Native worker requests retain their original
QueryGym source-release label; the outer trace records the distinct controlled
release. The adapter does not claim original-paper result parity.

## Dependency-locked HyDE

`acquire.hyde.controlled` uses the same HyDE formula as the clean-room profile:
generate hypothetical passages, average their embeddings with the query's
embedding, and search real corpus documents. It additionally binds the shared
generator, embedding model, index, candidate count, settings and execution budget
before dispatch. Supply runtime dependencies through the runner context:

```python
from agentinputlab.algorithms.acquire import HyDEConfig, HyDEControlledBindings
from agentinputlab.api import create_intervention, run_intervention

bindings = HyDEControlledBindings.from_context(
    context,
    budget=state.budget,
    config=HyDEConfig(num_hypotheses=4, max_tokens=512, top_k=5),
)
method = create_intervention("acquire.hyde.controlled", bindings=bindings)
result = run_intervention(method, state, context)
```

Here `context` contains the operator's real `backend`, `dense_embedder` and
`vector_searcher`. Capture their identities before dispatch, not after seeing
test outputs. The fingerprints and budget must still match at execution; an
identity change before or during the method fails rather than silently switching
dependencies. Generation, embedding and retrieval remain instrumented by the
runner. Empty hypotheses, provider errors and exhausted budgets are not retried
or converted into a successful NO_OP. Hypothetical text is never labeled corpus
evidence. Continue the patched state with the benchmark reader and native scorer;
the returned patch is not an endpoint score.

`to_jsonable(bindings)` is a data-only recipe accepted as `bindings=` by the
Python factory and the registry intervention builder used by CLI campaigns.
Its `generator`, `embedder`, `searcher`, `budget` and `config` fields can also be
specified directly in frozen campaign JSON. No executable object, credential or
model download belongs in that recipe. Fingerprint equality records the
operator-declared dependency identity; it does not independently hash weights,
prove original-paper parity or certify a factorial experiment. Declare
`profile_kind="factorial_adapter"` explicitly only with the corresponding shared
comparison protocol.

## Reloading frozen ACE and SkillOpt artifacts

Fit an ACE playbook or SkillOpt skill on explicitly isolated training/validation
data, save `artifact.as_dict()` as JSON, and record the file's SHA-256 in a
trusted receipt. Later processes can reload it without fitting again:

```python
from agentinputlab.recipes import ACEContextArtifact, SkillOptArtifact

ace_artifact = ACEContextArtifact.from_json(
    "ace-artifact.json", expected_sha256=trusted_ace_receipt["artifact_file_sha256"]
)
skill_artifact = SkillOptArtifact.from_json(
    "skillopt-artifact.json", expected_sha256=trusted_skillopt_receipt["artifact_file_sha256"]
)
```

These loaders require a checksum from a trusted receipt, reject symlinks and
files larger than 4 MiB by default, and parse JSON without executable imports or
pickle. Duplicate keys, non-finite numbers, malformed schemas and inconsistent
content IDs are rejected. Loaded mappings are detached and recursively frozen.
Use `max_bytes=` to declare a different limit explicitly, not an unbounded read.

`from_mapping()` validates content identity but has no independent file pin.
The historical ACE content ID excludes usage, so the whole-file checksum is
also needed to preserve its recorded training cost. SkillOpt's
`load_skillopt_artifact(output_root, ...)` additionally rehashes the required
native output files; loading a JSON manifest alone does not inspect that output
directory. A valid checksum/ID proves consistency with the trusted receipt, not
authorship, licensing, split isolation or compatibility with every target model.
Keep upstream/model fingerprints, train/validation manifests, native outputs
and actual fit costs with the receipt, and declare evaluation-time transfers.
Neither loader downloads assets, contacts a model or reruns optimization.

## SkillOpt environments and native IR feedback

`SkillOptOfflineRecipe` accepts a caller-owned native `EnvAdapter` through an
explicit `SkillOptEnvironmentBinding`. Pin the factory Python file and its JSON
configuration, then supply the binding to the recipe:

```python
from agentinputlab.recipes import SkillOptEnvironmentBinding, SkillOptOfflineRecipe

binding = SkillOptEnvironmentBinding(
    factory_path="/path/to/ir_factory.py",
    factory_sha256=trusted_factory_hash,
    configuration_path="/path/to/ir_task.json",
    configuration_sha256=trusted_task_hash,
)
recipe = SkillOptOfflineRecipe(environment_binding=binding)
# Call recipe.fit(...) with the pinned checkout, YAML configuration, model
# fingerprints and isolated output directory, as for the native recipe.
```

The trusted factory defines `create(*, cfg, configuration)` and returns an
upstream `EnvAdapter` with a nonempty JSON-serializable `fingerprint()`. The
original CLI parses all training controls and runs the original trainer,
reflection, patch merging and validation gate. The factory and task checksums,
environment fingerprint and binding receipt enter the fitted artifact's
identity. With no binding, the existing native CLI command stays unchanged.
Factories execute trusted Python code; pin their dependencies and keep
credentials in environment variables rather than configuration or fingerprints.

For retrieval tasks, the factory can call
`create_skillopt_retrieval_adapter(adapter=..., retriever=..., target_backend=...,
train=..., validation=..., cfg=cfg, metric="ndcg_at_10")`. Supply disjoint native
examples, their original corpus/retriever and their original scalar metric.
The target backend must share the optimizer's metered fit budget. The default
query decoder is strict one-field JSON, with single-fence compatibility available
through explicit `allow_json_fence=True`. Qrels stay outside target prompts;
reflection receives the measured scalar score. Malformed query outputs retain
their cost and receive zero training feedback. Provider or corpus execution
errors propagate without a fallback or retry.

This fit environment exposes training and validation groups. Evaluate held-out
queries separately with the native dataset scorer and record their inference
budget separately from fitting. A held-out request to the fitting environment
is rejected before inference.

## Presegmented RECOMP extraction

The standalone `curate.recomp.extractive` interface consumes sentence candidates.
For ordinary visible context blocks, call
`prepare_recomp_sentence_state(state, max_candidates=4096)` from
`agentinputlab.algorithms.curate.recomp` first, then use that same prepared state
for both NO_OP and compression arms. It preserves all query/history/budget
fields and records each exact source substring and offsets. Candidate overflow
fails without silent truncation; it does not download, retrieve or generate.
This is the disclosed controlled punctuation/newline splitter, not the original
paper's dataset-specific preprocessing. Bind the actual external scorer,
checksum-locked safe checkpoint and retriever provenance separately; preparation
does not run or certify them. Do not relabel paragraph-input common-support
results as standalone sentence-input results.

## Native τ³ checkpoint preparation

The Banking Knowledge provider now defaults to one initial simulator capture
attempt. Extra preparation attempts require explicit `capture_max_attempts`
(at most 16) and use recorded deterministic seeds. They are not evaluated-agent
attempts or free setup: retain every native call's actual or unknown usage.
The initial-user attempt counter alone is not token or cost telemetry.

The legacy GPT-OSS text-only preparation variant additionally requires
`capture_text_only_fallback=true` and `capture_max_attempts=17`. It changes the
simulator's tool availability on the last preparation attempt and is not native
capture parity; both visible/protected checkpoint metadata record that policy.
It is off by default and is not used in toolkit acceptance. Previously saved
checkpoints and paper results must not be relabeled with the new policy.
Retail retains its separate single fixed-seed capture; it never uses this
Banking Knowledge fallback.

## Loading native Plan-and-Budget caches

The cached profile accepts the upstream `decomposed_questions.json` envelope
directly, using a checksum from a separately trusted source receipt:

```python
from agentinputlab.algorithms.decompose.plan_and_budget import PlanAndBudgetCache

cache = PlanAndBudgetCache.from_native_json(
    "private/decomposed_questions.json",
    expected_sha256=trusted_receipt["cache_sha256"],
)
method = create_intervention(
    "plan.plan_and_budget.native_cached_soft_budget",
    records=cache, total_word_budget=512, schedule="constant",
)
```

The loader limits reads to 8 MiB by default, rejects duplicate IDs/JSON keys,
malformed records, symlinks and non-finite numbers, and returns immutable records
without reference, solution or answer fields. The exact task ID **and** source
query must match before a patch can be applied. Existing anomalous credit sums
and step counts remain explicit warnings rather than silently repaired data.
The patch carries the cache checksum. Byte identity is not authorship, license
permission or proof of the original generating model; retain those in the source
receipt. Historical cache-generation cost remains unknown, not zero.

These allocations are soft **word** instructions, not hard decoder-token caps.
Equal-weight remainder allocations use stable source-index ordering, recorded
in the trace as `allocation_tie_policy=stable_source_index`. NumPy's native
default sort can order tied indices differently, so a remainder word can go to
a different tied step without changing the total budget. The port does not
claim bitwise parity with that version-dependent tie order; this same allocation
rule is used by the controlled regenerated profile.
The caller still supplies and budgets the downstream model and native scorer.
An author cache for MATH-500, NaturalInstructions or TravelPlanner must not be
relabelled as a cache for a different benchmark. This API neither downloads the
cache nor generates missing plans, and does not claim full upstream solver
prompt or paper-result parity. Use the separate controlled regenerated profile
when explicitly preparing a new benchmark's cache fields.

## Plan-and-Budget decomposition-only component

`plan.plan_and_budget.decomposer_component` exposes the first stage of the
existing controlled, general-QA Plan-and-Budget adaptation. It shares that
profile's decomposition prompt, parser, seed and first-stage token reservation.
The optimizer sees the query; the returned patch appends suggested sub-questions
while preserving the reader's original query, evidence, instruction and history.

```python
from agentinputlab import InterventionContext
from agentinputlab.api import create_intervention, run_intervention

method = create_intervention(
    "plan.plan_and_budget.decomposer_component", decomposition_max_tokens=512
)
result = run_intervention(method, state, InterventionContext(backend=backend, seed=17))
```

Supply `state` and an explicitly configured real `backend`.
This component makes one generation call. It does not assess credits,
allocate soft-word budgets or run a solver; it is not the full P&B system or a
native released-cache/paper-parity claim. A separate benchmark reader consumes
the patch and its actual remaining budget. Nonempty parser fallback and its
warnings remain explicit; empty/invalid generations retain their actual cost.
`tools/run_plan_budget_component_acceptance.py` runs fixed real QA pairs using
prepared sources and the native scorers, without downloading data or models.

## Plan-and-Solve prompt profiles

`plan.plan_and_solve.ps_prompt` and `plan.plan_and_solve.ps_plus_prompt` are
dependency-free input interventions. They append the fixed PS 201 or PS+ 302
instruction while preserving the query, evidence and remaining budget. The
caller then executes its downstream model and benchmark-specific parser/scorer;
the intervention does not generate an answer or charge a hidden helper call.

```python
method = create_intervention("plan.plan_and_solve.ps_plus_prompt")
```

The trace identifies the exact prompt and upstream revision. PS+ 302 is one
explicit variant from the [authors' prompt table](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting/tree/9d5a176f5b4de1011d426009ab6398dc76046e88#cook-prompts),
not an automatically tuned or dataset-selected template. Modern chat-model
execution is controlled model transfer of this prompt intervention, not a
reproduction of the original GPT-3 arithmetic pipeline or its separate answer-
extraction pass. No arithmetic regex replaces QA, tool or environment scoring.

### Planning-clause component

`plan.plan_and_solve.plan_component` is a distinct controlled component, not an
alias of either prompt-only profile. It submits `Q:{query}\nA: {planning_clause}`
to the shared `context.backend` once, using the first sentence of fixed PS 201
and omitting its subsequent solve clause. The prompt sees only the visible
query, not the reader's evidence, instruction, history, gold labels or scorer.
The generation defaults to 512 output tokens, seed from the context, no stop
strings and no JSON/answer-extraction pass. The actual attempt and token/cost
usage count against the caller's shared budget.

```python
method = create_intervention("plan.plan_and_solve.plan_component", max_tokens=512)
result = run_intervention(method, state, context)  # context.backend is required
```

The nonempty final text is appended as `Suggested plan:` to the downstream
instruction; query, context, messages and budget fields remain unchanged.
The runner accounts for consumed resources; continue the patched state with
the remaining budget in the original reader/environment and official scorer.
No solver or tool action is executed by the component itself. The raw advisory
text is preserved apart from outer whitespace: it is not judged to be a good
or answer-free plan, and a partial nonempty generation is not promoted into a
quality score. Empty final output is a paid failure, without retry or fallback.
This declared planning-clause adaptation is not an upstream planner API or a
reproduction of the full PS generation/answer-extraction pipeline. Both
`controlled_adapter` and explicitly chosen `factorial_adapter` are available;
`official_native` is not claimed for this component. It is constructible from
the same JSON registry-builder recipe and operator-configured SDK service.

## RQ-RAG full-path leaf selection

`RQRAGCompletionBackend` (from `agentinputlab.recipes`) is an explicit opt-in
raw `/completions` backend for `RQRAGMethodTreeSearch`. It requests echoed prompt
and answer log probabilities in one generation, enabling `confidence`, `ppl`
and `ensemble` selection without an unreported second scoring call. Input
messages, tools, JSON mode, incomplete prompt probabilities and unsafe payload
overrides are rejected. It does not insert a chat template, apply a fallback or
retry an output. The fingerprint and raw response retain this scoring profile;
it is a controlled method configuration, not released-7B checkpoint parity.

The PPL covers all returned prompt/path and completion tokens except the first
unconditional token; any unreturned stop-token count is explicitly reported.
The confidence is the mean maximum token probability over the generated answer.
Use a server that supports native echoed prompt log probabilities (such as the
verified vLLM completion route), and account for every request's actual usage.
Post-hoc leaf selectors can reuse a fixed generated tree only when all native
request payloads match exactly. Record those CPU derivations separately from
physical model calls and preserve failed trees; do not retry until a tree passes.

Prompt log probabilities also need transient GPU memory beyond ordinary decoding.
A short interface probe does not establish long-context capacity. The acceptance
run's restored 48-GiB 27B-FP8 service used `--max-num-batched-tokens 512`,
`--max-num-seqs 2`, `--gpu-memory-utilization 0.84` and chunked prefill, while
keeping the model and 32,768-token context limit unchanged. Record such execution
changes and their extra costs separately from method/token budgets. A connection
refused before HTTP is not a model generation; an engine OOM can have unknown
consumed compute. Preserve both rather than turning infrastructure failures into
quality scores or silently restarting unrelated healthy services.

## Native ToolSandbox call IDs

OpenAI-compatible tool-call IDs are opaque correlation strings. The pinned
ToolSandbox converter also uses them as Python variable names, so IDs containing
hyphens or punctuation need a deterministic, identifier-safe encoding. The native
provider applies that encoding before the upstream converter stores calls and
their matching results, for both agent and simulated-user roles. It records each
original/encoded pair in `compatibility_repairs`; raw HTTP receipts stay unchanged.
Tool names, arguments, call order, model output text, usage, budgets and official
scoring are not changed. Missing IDs are not fabricated and duplicate logical
calls are not deduplicated. This is a protocol representation adjustment, not a
new model request, output retry or repair of malformed function names/arguments.

## Native live history

The file-based `config evaluate-arena` entry can select the same native hook.
Use a run configuration for `curate.acon.history_runtime`, including its
`acon_history_worker` runtime and pinned artifacts, and add this field to the
normal arena configuration:

```json
"history": {"kind": "acon", "prefix_policy": "strict_system_user"}
```

Then run:

```sh
agentinputlab config evaluate-arena run.json --arena-config arena.json --case-id CASE_ID --output-dir NEW_OUTPUT
```

This creates a separate history hook for each episode. It does not also apply a
one-time initial compression. Native provider usage includes both the optimizer
and downstream agent; `live_history_records` retains compression decisions,
including on failed episodes. Do not add those records' costs to the total a
second time. Omitting `history` keeps the existing initial-patch execution mode.
For a native τ³ conversation with an assistant greeting before the first user,
explicitly select `preserve_pre_user_assistant`; the default never silently
reorders the conversation. This option does not enable a paused benchmark or
supply missing model endpoints, source assets or optimizer artifacts.

Use Python 3.11+ and the `acon` extra for the source-mounted ACON history and
observation optimizers. Mount the exact upstream checkout and bind its prompt,
tokenizer, optimizer and target identities as described in
[`workers/acon/README.md`](../workers/acon/README.md). The extra supplies the
dependencies of those two optimizer classes, not optional upstream
`HistoryRetriever`, guideline training or the full ACON agent stack.

For a configured `curate.acon.history_runtime` method and its real worker context:

```python
from agentinputlab.integrations.live_history import ACONLiveHistory
from workers.official_arenas.providers.toolsandbox import ToolSandboxProvider

live = ACONLiveHistory(acon_method, intervention_context, per_call_output_tokens=512)
provider = ToolSandboxProvider(history_transform=live)
# Use the provider's normal native capture / run_arm interfaces.
```

Each episode needs its own `live` object. It compresses the agent-visible view
at real user-turn boundaries and persists ACON summary metadata across turns.
Intermediate tool calls/results keep their original protocol fields and IDs;
the native environment, user simulator and official scorer are unchanged.
Optimizer calls share the evaluated agent's run budget, with a downstream call
and output allowance reserved first. Initial/threshold/budget skips are recorded
as skips, not successful compression. Genuine optimizer failures retain their
usage and propagate. `live.records` exposes decisions and source-bound traces.
The default provider without `history_transform` keeps its existing behavior.

The native τ³ Banking/Retail and STATE-Bench providers also expose this explicit
constructor option. See [native history adapters](acon-live-tau.md) for immutable
τ³ greeting prefixes and STATE's typed Responses Item/usage contract. Neither
adapter rewrites simulator state or official scorer inputs.

You can check a native history prefix before constructing any model service:

```python
from agentinputlab.integrations.live_history import (
    NativeHistoryInputError, validate_native_history_prefix,
)

try:
    first_user_index = validate_native_history_prefix(native_request_messages)
except NativeHistoryInputError as error:
    print(error.code)  # e.g. native_system_user_boundary_missing
    raise
```

This validates framing, not algorithm performance or complete tool history.
The source-native LongBench reader sends a single user prompt without a system
frame. That endpoint cannot enter the strict ACON history interface; adding a
fake system frame would change the native request. Its long context can instead
be used with the separately identified ACON observation profiles, which have
their own budgets and evidence. No automatic profile substitution, model retry,
synthetic score, or benchmark-wide claim follows from a prefix rejection.

## Compact BRIGHT retrieval domains

The source-locked BRIGHT loaders cover `bright.earth_science`, `bright.economics`,
`bright.robotics`, and `bright.stackoverflow`. Query/qrel and corpus components
have independent immutable revisions and checksums; answers and reasoning stay
outside optimizer-visible inputs. Robotics needs about 8 MB of source files and
Stack Overflow about 40 MB, plus decoded data and indexing space.

```python
from agentinputlab.benchmarks import load_production_benchmark_adapter, prepare_benchmark

cache = prepare_benchmark("bright.robotics", cache_dir="toolkit-cache")
adapter = load_production_benchmark_adapter("bright.robotics", cache["prepared_root"])
example = sorted(adapter.examples("test"), key=lambda item: item.example_id)[0]
state = example.visible_state()
```

`prepare_benchmark` reuses verified caches and streams missing pinned files with
a default 512 MiB total download limit. Install the `benchmarks` extra for Parquet
decoding. A fixed query sample tests compatibility; it is not a full-domain
performance result. Very large sources can remain deferred instead of downloading
an entire corpus just to sample it.

The preparation budget is separate from the download budget. Set
`max_prepared_bytes` in Python or `--max-prepared-mib` in `agentinputlab datasets
prepare` (default 512 MiB per source asset). Planned source and normalized-output
bytes are summed on each actual destination filesystem before downloading, with
a further 1 GiB kept free. Normalized files, including metadata, are bounded
during writes; failed partial preparation is removed, not exposed as a cache.
Verified existing caches need no fresh output allowance. These are preflight
snapshots, not reservations against other users' concurrent disk writes.
External model caches and user-created retrieval indexes require their own
capacity budgets; these limits do not claim to bound those separate tools.

## Source-side BEIR compatibility samples

For FEVER or CQADupStack without a full-corpus download, `BEIRSampleAdapter`
loads a caller-mounted, manifest-pinned small bundle. It never accesses the
network itself. The bundle keeps `inputs.jsonl`, evaluator-only `gold.jsonl`,
`corpus.jsonl`, and byte-locked acquisition receipts under `resources/`.
Its manifest records exact upstream repository revisions, query/background
selection rules, counts, file hashes and upstream data license. All bundle
reads are bounded to 16 MiB, with no symlink or path-traversal sources.
The acquisition scope is explicit: a small complete subforum source file may
be mounted while the other forums remain deferred. This is distinct from
downloading the complete multi-forum benchmark or a large FEVER corpus.

```python
from agentinputlab.benchmarks import BEIRSampleAdapter

adapter = BEIRSampleAdapter(
    "mounted-beir-sample",
    manifest_sha256="<sha256-of-your-source-bound-manifest>",
    license_acknowledged=True,
)
example = adapter.examples("test")[0]
state = example.visible_state()  # does not expose relevance judgments
documents = tuple(adapter.iter_documents(forum=example.visible_metadata["forum"]))
# Run your actual retriever over documents; score its actual returned document IDs.
# metrics = adapter.score(example, ranked_document_ids)
```

This profile deliberately combines fixed source-background rows with the selected
queries' relevant documents. It exercises compatibility on a gold-enriched
candidate pool, **not full-corpus retrieval quality**; reported nDCG@10/Recall@100
use `ir-measures` on that declared pool and must not be compared with full BEIR
leaderboards. The fingerprint retains this distinction. CQADupStack query and
document IDs include the forum namespace; cross-forum or absent candidates are
rejected. A sample of some forums does not establish coverage of all 12 forums.

## Compact ClariQ dev checks

Install the `clariq` extra and mount the official
[dev TSV](https://github.com/aliannejadi/ClariQ/blob/46885a544581a0af8aff0681d29e4971807e2912/data/dev.tsv),
[question bank](https://github.com/aliannejadi/ClariQ/blob/46885a544581a0af8aff0681d29e4971807e2912/data/question_bank.tsv),
and [native scorer](https://github.com/aliannejadi/ClariQ/blob/46885a544581a0af8aff0681d29e4971807e2912/src/clariq_eval_tool.py)
under their original `data/` and `src/` paths. These three pinned files total
838,652 bytes; no ClueWeb corpus or multi-turn archive is required. Their upstream
license is `NOASSERTION`: review local-use terms explicitly; source rows and
scoring code are not vendored or redistributed by this package.

```python
from agentinputlab.benchmarks.clariq import ClariQDevBundle, ClariQPrediction

bundle = ClariQDevBundle("mounted-clariq", accept_license_review=True)
examples = bundle.examples("dev")[:2]  # freeze selection before seeing scores
state = examples[0].visible_state()  # no facet descriptions, answers or need labels
ranked_ids = bundle.rank_questions(state.query, top_k=30)
# Supply actual model need predictions, never reference labels or invented values.
# predictions[example.example_id] = ClariQPrediction(model_need, ranked_ids)
# bundle.evaluate(predictions, topic_ids=[e.example_id for e in examples], output="new-score-dir")
```

`question_retriever` exposes the visible full small question bank for separately
configured rewriting methods. This BM25 ranker is a declared toolkit baseline,
not an upstream ClariQ model. `evaluate` calls the unchanged source-verified RQ1
weighted precision/recall/F1 and RQ2 Recall@5/10/20/30 routines, retaining native
run files and stdout. Missing predictions stay in the frozen topic denominator.
Topic subsets are labeled compatibility subsets, not full-dev performance.
RQ3 document retrieval utility requires a different evaluation bundle and is
not synthesized from question overlap. The standalone bundle does not expand
the frozen seven-cell research campaign.

## LongBench v1

The `longbench` extra supplies the native scorer's dependencies. Mount the
[pinned official data archive](https://huggingface.co/datasets/zai-org/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip),
[scorer](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py),
and [prompt templates](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json).
Their checksums are verified before external scorer code is loaded. The six
available cells are `longbench.qasper`, `longbench.multifieldqa_en`,
`longbench.govreport`, `longbench.qmsum`, `longbench.passage_retrieval_en`, and
`longbench.repobench_p`.

```python
from agentinputlab.benchmarks import LongBenchAdapter
from agentinputlab.backends import GenerationRequest

adapter = LongBenchAdapter(
    benchmark_id="longbench.multifieldqa_en",
    data_zip="mounted-longbench/data.zip",
    scorer_path="mounted-longbench/metrics.py",
    prompts_path="mounted-longbench/dataset2prompt.json",
)
example = adapter.examples("test")[0]
state = example.visible_state()  # excludes all reference answers
# Optionally apply a separately configured context intervention to state.
response = backend.generate(GenerationRequest(
    prompt=adapter.render(state), max_tokens=512, seed=17,
))
metrics = adapter.score(example, response.text)
```

Here `backend` is an explicitly configured real backend as described above.
Choose and record a model-appropriate input window and truncation policy before
generation; `render()` does not silently truncate or change the native prompt.
Native metrics are returned on a 0–1 scale. Constraint preservation and query
fidelity are separate study-level measurements, not invented LongBench labels.
The RepoBench-P exact-match diagnostic is separate from its native code-similarity
metric. `fingerprint()` exposes all data, scorer and template bindings.

The built-in LLMLingua adapters cache native weights by their full loading
configuration so repeated calls do not reload them. Custom compressor factories
stay visible to dependency inspection and cannot use that internal cache to hide
an unmonitored backend.

The existing campaign and benchmark APIs remain available in Python. Research-host launch scripts and local experiment archives are intentionally not distributed. The public convenience HTTP API handles patch execution, not arbitrary remote campaign commands. For real benchmark execution use the native benchmark runtime interfaces; the SDK cannot manufacture downstream task scores.
