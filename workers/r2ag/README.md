# R²AG pinned sidecar

This directory provides an inference-only compatibility worker for
`yefd/RRAG@61cfd1027c0101172fda701ec816bc18360b5bab`. It does not vendor or
download the upstream checkout, R²-Former/projection checkpoint, reader,
tokenizer, retrieval encoder, or datasets.

## Claim boundary

The public label is `controlled_source_informed`. The host implements the three
retrieval-feature formulas after auditing the pinned source on an already fixed
candidate pool (or accepts a hash-bound upstream-shaped feature artifact). The
worker code is designed to call the pinned `RFormer`, its learned projection,
marker-only embedding replacement, and reader generation in that order.

This is not an assertion that a mounted checkpoint is an official paper
checkpoint, that the production neural runtime has been exercised in CI, or that
benchmark result parity exists. Every result records
`paper_checkpoint_parity=false`, plus `fixture_mode` and
`execution_evidence`. Component and file-closure SHA-256 values hash-bind a
successfully verified mounted treatment; they do not establish provenance beyond
the supplied repository/revision records.

## Required read-only mounts

- A clean Git checkout at the fixed upstream commit and tree. The driver checks
  the commit, tree, tracked and untracked cleanliness, and every source file in
  `compatibility_manifest.json`.
- A checkpoint directory with `config.json` and
  `RRAGLlama_pytorch_model.bin`. The production runtime uses
  `torch.load(..., weights_only=True)`, accepts only the `r_former` and
  `llama_proj` state dictionaries, hashes each tensor component
  deterministically, and loads both with `strict=True`.
- A local reader snapshot and a local tokenizer snapshot. Each contains a closed
  `agentinputlab-manifest.json` with the repository, immutable revision, and
  every non-manifest file's relative path, byte size, and SHA-256. The worker
  rejects missing/extra files, symlinks, size drift, and content drift before
  constructing the runtime.
- For `feature_source=worker`, a local sentence-transformer snapshot at
  `<checkpoint>/feature-embedder` with the same closed manifest schema. Its
  repository, revision, and manifest hash must equal the strictly typed
  `candidate_pool_fingerprint.retriever` identity.

The closure manifest schema is:

```json
{
  "schema": "agentinputlab.artifact-closure.v1",
  "repository": "organization/model",
  "revision": "immutable-revision",
  "files": {
    "config.json": {
      "size": 1234,
      "sha256": "lowercase-64-character-sha256"
    },
    "model.safetensors": {
      "size": 5678,
      "sha256": "lowercase-64-character-sha256"
    }
  }
}
```

`agentinputlab-manifest.json` does not list or hash itself. Its own file hash is
carried in `R2AGArtifactIdentity`, while the `files` mapping must equal the
complete set of all other regular files below the snapshot root.

Run the worker as a non-root sidecar with no network, read-only mounts, no host
secrets, and explicit CPU/GPU, memory, and time limits. A custom
`runtime_factory` is accepted only in explicit fixture mode; production always
uses `OfficialR2AGRuntime`.

## Preserved execution and fairness gates

The request contains only the allowlisted candidate DTO fields
`document_id`, `text`, `title`, `source`, `score`, and `corpus_revision`.
Arbitrary benchmark metadata never crosses the worker boundary. Candidate
content/order and retriever/corpus/index identities are hash-bound. Candidate
count, checkpoint `config.num_k`, worker `num_k`, and feature row count must be
identical (2–30 documents). The driver checks model/output/context budget before
reader generation, records actual response usage, and rejects training, tools,
retrieval, reward, labels, and gold metadata.

The candidate-pool sub-identities use strict schemas:

- retriever/corpus: `repository`, `revision`, optional `manifest_sha256`;
- index: `algorithm`, `revision`, optional `manifest_sha256`;
- precomputed feature artifact: exactly `repository`, `revision`, `sha256`.

Nested oracle, reward, label, answer, reference, target, credential, and secret
keys are rejected by both host and worker. The worker also recomputes the full
request content hash before any source or artifact work.

## JSONL launch

The driver implements the in-process `WorkerClient` protocol and also exposes a
one-object-per-line JSONL boundary compatible with
`JsonLineSubprocessWorker`:

```bash
python -m workers.r2ag.driver \
  --source-root /read-only/RRAG \
  --checkpoint-root /read-only/r2ag-checkpoint \
  --base-model-root /read-only/reader \
  --tokenizer-root /read-only/tokenizer
```

Run this command in the separately pinned, non-root, network-disabled worker
environment that provides the reviewed Torch/Transformers dependencies.

The production worker also verifies the exact upstream runtime distributions
before loading weights: `torch==2.1.2`, `transformers==4.40.2`, and
`sentence-transformers==2.6.1`. Distribution metadata is queried using those
exact package names; missing packages, local-version suffixes, and version
drift fail closed. The versions are included in both the compatibility
manifest and worker fingerprint.

Reader configuration compatibility is checked before deserializing neural
checkpoints or constructing a reader; integrity hashing may still read files.
The pinned Transformers 4.40.2 runtime does not recognize
modern `qwen3`, `qwen3_5`, `gemma4_unified`, or `gpt_oss` model types. Mounting
one of those snapshots therefore reports `READER_CONFIG_INCOMPATIBLE`,
not a failed inference or a benchmark score of zero. Do not upgrade this
profile's dependencies, change the reader, or unwrap a conditional-generation
configuration to make it pass. A modern-reader adaptation is a separately
configured and validated execution profile. A successful configuration check
does not replace complete snapshot, trained-checkpoint, or neural validation.

The pinned upstream reader passes `inputs_embeds` (not prompt `input_ids`) into
`generate` and decodes the complete returned token tensor. The compatibility
runtime intentionally preserves that behavior instead of guessing a prompt
prefix to strip. It requires a rank-2, single-batch result containing between
one and `max_new_tokens` tokens and a non-empty decoded answer.

## Checks

```bash
PYTHONPATH=src:. python -m pytest -q tests/workers/test_r2ag_worker_driver.py
PYTHONPATH=src:. python -m pytest -q tests/algorithms/unit/test_r2ag_method.py
```

These tests include a complete host → typed request → real driver → fixture
runtime → host round-trip, JSONL framing, metadata-leak rejection, request-hash
recomputation, and reader/tokenizer/embedder closure-tamper rejection. They do
not load a real R²AG checkpoint.

To verify a real source checkout without loading a checkpoint:

```bash
AGENTINPUTLAB_RUN_REAL_R2AG_SOURCE_SMOKE=1 \
AGENTINPUTLAB_R2AG_CHECKOUT=/read-only/RRAG \
PYTHONPATH=src:. python -m pytest -q \
  tests/workers/test_r2ag_worker_driver.py -k real_checkout
```
