# Pinned upstream workers

This directory contains auditable drivers for algorithms whose original runtime
cannot safely live inside the dependency-free AgentInputLab wheel. A worker is
not a vendored copy of an upstream repository: users provide the exact checkout
and optional weights, while the driver verifies the declared commit, license,
source files, configuration artifacts, and protocol identity before execution.

The worker boundary standardizes transport, limits, trace projection, and
fair-comparison settings. It does not replace the upstream algorithm. Any
compatibility or security patch is listed in the method's manifest and included
in its runtime fingerprint.

The following claims are deliberately separate:

1. **Protocol tested** — malformed inputs and outputs fail closed.
2. **Pinned plumbing tested** — a small checkout exercises the real upstream
   symbols through the AgentInputLab bridge.
3. **Upstream boot tested** — the audited checkout imports in its locked
   dependency environment.
4. **Result parity tested** — official fixtures or benchmark outputs match the
   upstream reference within a declared tolerance.

A worker remains `external_only`, `planned`, or `retrain_required` until the
applicable upstream boot and parity gates pass. Fake fixtures are never evidence
of paper-result reproduction.

Current worker directories are documented independently. Each README lists its
upstream revision, supported profile, required environment variables, execution
mode, and remaining parity limitations.

The `ask_or_assume/` worker is intentionally narrower than a full OpenHands
deployment: it runs only the pinned IntentAgent decision and forced-question
component path and labels all outputs as component-only.

The `reformer/` worker executes source only from a user-mounted pinned checkout.
Its upstream has no license file, so it requires operator and per-request
`NOASSERTION` acknowledgements and never vendors upstream code.

The `recomp/` worker imports the pinned official sentence-scoring API and
replays the official abstractive prompt/generate path against user-mounted,
hash-attested checkpoints. It fixes each input block as one candidate sentence,
requires a retriever fingerprint without performing retrieval, and keeps
paper-native top-N assembly separate from controlled assembly.

The `rrr/` worker verifies the unlicensed, credential-risk pinned tarball
without extracting/importing it, then runs a hash-bound compatible inference
chain over user-mounted prompts, frozen search artifacts, and local checkpoints.
Warm-up/PPO checkpoints are inference-only; controlled retrieval is explicitly
separate from the frozen Bing-cache profile.

The `infocqr/` worker directly calls pinned official teacher prompt functions
or replays checksum-locked author caches. Because neither source/cache licensing
nor a released T5 student checkpoint is available, it is dual-gated and never
vendors upstream material or fabricates student weights.

The `smartsearch/` worker runs only frozen inference with the upstream
10-round/top-5/8192-token loop. It replaces the unsafe code-execution search
sandbox with a fixed retriever shim and labels the result as a controlled
adapter; training and paper-result parity are outside this worker.

The `qrag/` worker runs a user-mounted, hash-attested frozen checkpoint through
the two-step E5-Hotpot action path and records every available Q-vector, action,
and selected candidate. Source and checkpoint licenses are acknowledged
separately; reward, gold labels, and training are unavailable at evaluation
time.

The `adaptive_rag/` worker verifies a clean pinned checkout and exact router
snapshot/weights/tokenizer, generator, retriever, corpus, index, and prompt
closures. A supervisor applies the upstream first-step A/B/C routing rule,
reserves budget before every call, executes only the selected **controlled
reimplementation** arm, and emits a hash-linked event log that the host
independently recomputes. Its hermetic runtime is fixture-only; neither the
source-checkout smoke nor this controlled arm is evidence of upstream arm,
batch/postprocess-script, or paper-result parity.

The `r2ag/` worker verifies a clean pinned RRAG checkout plus separately
hash-bound R²-Former, projection, feature schema, and complete reader/tokenizer/
feature-embedder file closures. It is designed to execute RFormer → projection
→ retrieval-marker-only embedding replacement → reader generation on an
allowlisted, fixed candidate DTO and is labeled `controlled_source_informed`.
CI exercises the complete host/real-driver boundary with a fixture runtime; it
does not run a production neural runtime, official paper weights, or
paper-result parity.

The `mce/` worker loads the hash-pinned upstream `run_iteration` orchestration
but applies a hash-bound semantic patch at four unsafe/unimplemented seams:
meta/base agents run through an isolated manager, interface validation is
remote/static without local import, and evaluation runs in a separate trusted
oracle supervisor. Generated code remains in a verified worker-side
content-addressed store; the host receives only an opaque handle, hashes,
scores, usage, and a closed regular-file manifest. This is a controlled thin
wrapper—not `official_native` E2B or paper-result parity—and production
attestation/credentials remain operator supplied.
