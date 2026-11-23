# Rewrite-Retrieve-Read quarantined inference worker

The toolkit's recommended RRR entrypoint is
[`rewrite.rrr.common_support_prompt`](../../docs/rrr-controlled.md), which uses
the configured backbone and shared retriever. It runs through the Python core
without this external worker or an author warm-up/PPO checkpoint. This document
covers the separately selected external-worker profiles.

This worker supports the separable inference chain from
[`xbmxb/RAG-query-rewriting`](https://github.com/xbmxb/RAG-query-rewriting) at
commit `6bab6243e60c7f3e526b6ee44f95381b1024d471`:

`rewrite prompt/model -> parsed query -> fixed search artifact -> snippets -> frozen reader`

It remains `external_only`. The repository has no root license file, its
published `complete()` function is an empty stub, and a redacted security scan
of the pinned tarball reports credential-risk assignments in seven source
paths. AgentInputLab therefore never vendors, extracts, imports, or echoes the
upstream source or any suspected value. The tarball is used only for commit,
byte-hash, absent-license, AST-symbol, and risk-count evidence. Output evidence
is `pinned_tarball_compatible_chain`, not native execution or paper parity.

## Mandatory gates

Execution requires all of the following:

1. the adapter sets `source_license_acknowledged=True`;
2. the adapter sets `checkpoint_license_acknowledged=True`;
3. the worker receives `--acknowledge-noassertion-source`;
4. the worker receives `--acknowledge-noassertion-checkpoints`;
5. the operator explicitly keeps the risky tarball quarantined with
   `--acknowledge-credential-quarantine`.

Source, rewriter, reader, prompt, search cache, and dataset licenses are
independent. Both checkpoint manifests must retain `license: "NOASSERTION"`.
No upstream data, cache, source, or weights are included in wheels/sdists.

## Frozen comparison identity

Every request binds the two prompt byte hashes; prompt/warm-up/PPO rewriter
checkpoint revision; frozen-reader revision; model-deployment fingerprints;
search-cache or controlled-retriever artifact; corpus/index revisions; top-k;
search time window; seed; input character limits; generation token limits; and
the complete input hash. Successful requests report two model calls and one
search; failures retain only the calls and usage actually consumed. Local
encoder-decoder output accounting excludes the supplied decoder-start prefix,
while decoded output and generation limits stay unchanged.
Requests cannot choose paths, modules, URLs, environment variables, or
credentials.

The official single-query parser boundary is preserved: generation ends at
`**`, queries use `;`, and this release accepts exactly one query. Search uses
the rewritten query; the frozen reader receives the original task query plus
the joined snippets, matching the upstream `rewrite -> rewrite2` data flow.

`bing_cache` is a `controlled_adapter` with `search_profile=bing_cache`. It
replays a frozen Bing result artifact; the worker never calls the
credential-bearing Bing code.
`controlled_static` requires a matching retriever fingerprint and is always
labelled `controlled_adapter`, never native.

## Checkpoint inference only

The rewriter checkpoint manifest supports `prompt`, `warmup_checkpoint`, and
`ppo_checkpoint`; these are inference profiles. The reader uses
`frozen_reader`. Local loading uses Hugging Face with
`local_files_only=True`, `trust_remote_code=False`, and deterministic greedy
generation. There is no PPO training command or code path.

```json
{
  "schema_version": "agentinputlab.rrr.checkpoint.v1",
  "role": "rewriter",
  "profile": "ppo_checkpoint",
  "revision": "immutable-checkpoint-revision",
  "model_type": "seq2seq",
  "license": "NOASSERTION",
  "files": {
    "config.json": "<sha256>",
    "model.safetensors": "<sha256>"
  }
}
```

## Search cache and commands

The JSON search artifact uses schema `agentinputlab.rrr.search-cache.v1` and
records `mode`, corpus/index revisions, ISO date window, and exact query-to-row
entries. Its raw SHA-256 is `search_artifact_hash`.

Source/symbol/security smoke test (never imports source):

```bash
python workers/rrr/driver.py \
  --source-tarball /mounted/6bab6243.tar.gz \
  --smoke-source
```

Inference:

```bash
python workers/rrr/driver.py \
  --source-tarball /mounted/6bab6243.tar.gz \
  --prompt /mounted/rewrite-prompt.txt \
  --reader-prompt /mounted/reader-prompt.txt \
  --search-cache /mounted/search-cache.json \
  --rewriter-checkpoint /mounted/rewriter \
  --rewriter-manifest /mounted/rewriter-manifest.json \
  --reader-checkpoint /mounted/reader \
  --reader-manifest /mounted/reader-manifest.json \
  --acknowledge-noassertion-source \
  --acknowledge-noassertion-checkpoints \
  --acknowledge-credential-quarantine
```

Fake-tarball tests prove transport, trajectory, redaction, identity, fairness,
budgets, and license gates only. The opt-in real-tarball test proves byte and
source-symbol evidence. Neither is a benchmark-result parity claim.

An existing pinned Git checkout can replace the archive for the same inference
chain. Use `--source-checkout /mounted/RAG-query-rewriting` instead of
`--source-tarball`. This verifies the exact commit/root, the manifest's required
file SHA-256 locks, tracked Python Git blobs, AST symbols and credential-risk
inventory without reading bundled datasets or downloading the archive. It never
imports the checkout. Evidence is `pinned_checkout_compatible_chain`; the complete
archive checksum and historical Bing/checkpoint parity are not claimed.
