# Full-toolkit acceptance protocol

Acceptance is independent of paper selection and paper scores. It includes every
registered method and benchmark, including ToolTalk, Retail, and methods omitted
from the paper. A catalog entry, import test, synthetic fixture or historical
research run is not evidence that the installed distribution works.

## Coverage and eligibility

Review each method release separately: prompt patches, composite agents, external
workers, artifact replay, components and training recipes have different contracts.
Training recipes require an isolated training/artifact pipeline check, not a dummy
runtime invocation. Do not substitute a different algorithm to make a row pass.

For each method × benchmark × backbone, record an explicit eligibility decision.
Family declarations are candidates, not approvals. A negative decision needs a
semantic reason and source evidence. Missing code, credentials, data, checkpoints,
licenses or hardware are unresolved prerequisites, not “not applicable” passes.
Keep all rows visible and block release while required rows remain unresolved.

## Fixed cases before dispatch

Bind exact release ID, package hash, model revision/quantization, dataset split and
revision, official scorer revision, case IDs, seed, token/call/wall-time budgets
and dependency hashes before execution. Choose real cases without looking at
their scores. Include a baseline and representative boundary cases per applicable
combination; record extra attempts and both arms in actual cost. Never relabel a
synthetic fixture as a real case or silently import paper results as acceptance.

Boundary coverage includes multi-turn state reset, tool schema/name/arguments,
reasoning versus final channels, truncation and stop tokens, native model templates,
retrieval corpus identity, compressor/tokenizer compatibility, missing assets,
timeouts, HTTP errors, memory pressure and SDK error propagation. Offline injected
errors check handling; they do not replace real model/official-scoring cases.

## Bounded data acquisition

Compatibility acceptance does not require full benchmark evaluation. Defer very
large source downloads and full-corpus indexes when a fixed, source-bound sample
can exercise the installed interface. Select cases before inspecting outcomes,
record the source revision, case IDs, corpus subset and selection rule, and label
the evidence as sampled compatibility coverage rather than a full-dataset score.
Do not download an entire large corpus merely to extract a small sample.
If a small independent source component is used (for example one bounded CQA
forum file), record that component's complete download explicitly, distinguish
it from row-level API sampling, and leave other components unclaimed. A
gold-enriched candidate pool is compatibility evidence, not a full-corpus
retrieval-quality result. Caller-pinned sample manifests are not canonical
upstream dataset locks supplied by the package.

`TRECDLSampleAdapter` accepts a separately acquired, manifest-pinned TREC DL
2019/2020 passage sample. It verifies normalized queries, passages and retained
judgments against the saved official query/qrel components and revision-pinned
source-side passage rows. It never downloads the MS MARCO collection. Candidate
enrichment and sampling stay explicit; full-corpus scores remain unclaimed.
The native `ir_measures` providers compute graded nDCG@10, AP and Recall@100;
binary measures use relevance **at least 2**, because passage grade 1 means
related but not relevant ([NIST task definition](https://trec.nist.gov/data/deep2019.html)).
The manifest locks this threshold and both upstream data-term identities.
Both official unused-column spellings are accepted (`Q0` in the 2019 qrels and
`0` in the 2020 qrels); query IDs, passage IDs and graded labels remain unchanged.

```python
from agentinputlab.benchmarks import TRECDLSampleAdapter

adapter = TRECDLSampleAdapter(
    "private/trec-sample", manifest_sha256="<64-hex-sha256>", license_acknowledged=True
)
example = adapter.examples("test")[0]
state = example.visible_state()  # no qrels in predictor-visible state
metrics = adapter.score(example, {"native-passage-id": 1.0})
```

The supplied ranked IDs must belong to that pinned candidate pool. The acceptance
operator supports `--benchmark trec_dl.2019 --sample-root ...
--sample-manifest-sha256 ...`; no archive or full-index fallback is performed.
Unavailable bounded source access is deferred, not silently called complete.
The selected retrieval profiles require their declared extras (`querygym` and
`reformer` as applicable) in addition to `benchmarks,evaluation`. The operator
checks these before dispatch; `--release` selects exact reviewed profiles.
Entry-only continuations may pin `expected_job_ids` in the private configuration
to reject any drift in fixed case, model, scorer, retriever or budget identities.

Before acquiring assets, check free space on the actual destination filesystem
and account for source files, extraction, indexes and temporary caches. Set an
explicit download-size limit; unknown size or insufficient space requires deferral
or a bounded source interface, not an unbounded fallback. Cleanup is restricted
to isolated acceptance downloads with no active consumer. Shared research data,
model weights, results and caches are not disposable acceptance artifacts.

Download-free configuration checks should reuse already prepared, bounded real
cases. For ReFormeR's four inference profiles, automatic and fixed-manual pattern
paths are separate native executions. Their respective `r` and `q ⊕ r` arms may
share exact saved model-seam completions: re-execute the pinned selector and
application code, require identical messages/seed/temperature/output cap, and
forbid network fallback. Execute the downstream reader for each materialized
query. Record physical helper cost once and logical helper usage in each arm;
do not forge a new worker response by editing the release or request ID.

The public source-locked preparation API checks the combined planned source and
normalized bytes per destination device, keeps a 1 GiB reserve, and bounds each
asset's normalized files separately (`max_prepared_bytes`, default 512 MiB).
Space checks are snapshots rather than reservations; a valid cache is reused
without allocating a second copy. External indexes and model downloads are not
covered by this preparation limit and need separate explicit budgets.

## Outcomes

Separate task quality from implementation correctness. An official score of zero
can be a valid measured endpoint; a crash, missing scorer or empty fabricated
result cannot. Retain budget-exhausted/timeout outcomes and failure denominators.
Do not retry real scientific failures merely to obtain positive scores. Code
repairs require new version-bound evidence and explicitly recorded extra attempts.

## Installed-product gate

Test clean wheel and sdist installation outside the checkout, optional extras,
worker discovery, native runtime dependencies, Python API/CLI, npm tarball and
TypeScript-to-Python execution. Preserve composite-agent semantics and official
scoring; the patch-only HTTP endpoint must not masquerade as benchmark execution.
Test supported Python/Node versions and publish only after distribution/license/
secret scans and complete acceptance review. Keep raw data, traces and results
outside the public source branch and packages.

`tools/acceptance_plan.py` produces a **non-executable review inventory**, not a
dispatch queue or passing acceptance report. All case selections and eligibility
decisions start unresolved. Its output must be saved outside the source tree.
`benchmark_reference_revision` retains the catalog's benchmark/scorer reference;
it is not automatically the data asset revision. `dataset_revision` and
`dataset_source_binding` start unset and require a reviewed repository namespace,
revision and content checksum. For example, HotpotQA's upstream GitHub
code/scorer commit and the official Hugging Face Parquet revision are different
identities, both retained. Comparing those hex strings as if they were revisions
of one repository creates a false data-version mismatch. Bind source assets
through their actual preparation manifest; do not overwrite either identity,
relabel old evidence or trigger reruns solely because those namespaces differ.
The inventory embeds a `data_acquisition_policy` that forbids full-corpus
download/index fallbacks and requires storage budgets for source, extracted,
normalized, index, temporary and cache bytes. Each row's
`data_acquisition_review` starts unset; an operator must bind a reviewed existing
cache or a bounded sample before dispatch. This review inventory does not itself
enforce a downloader's limits or turn deferred prerequisites into acceptance passes.

Pass `--observed <saved-receipt-inventory.json>` to reconcile the complete review
matrix with exact release × benchmark × backbone receipt groups. All attempts,
including real failures and pre-call/dependency guards, remain visible. Baseline
groups are separate from the 27 algorithm families. No family aliases, best-run
selection or automatic passes are inferred; observed groups still require
configuration, applicability and case-level review. Catalog availability and
missing prerequisites are included to distinguish runnable, gated and locked
profiles without declaring unimplemented profiles “not applicable.”
