# InfoCQR pinned teacher/cache worker

This worker integrates the inference-time material in
[`smartyfh/InfoCQR@2953c461`](https://github.com/smartyfh/InfoCQR/tree/2953c461aa0c6817b1b7656dbd5168050227b31d)
without copying upstream source into AgentInputLab.

It supports two bounded paths:

1. checksum-locked replay of the author's QReCC JSONL caches (`rw_zsl`,
   `rw_fsl`, and the T5-candidate teacher editor cache);
2. live teacher inference by dynamically loading and directly calling the
   pinned official `generate_rewrite()` function for zero-shot, four-shot,
   self-edit, or editor-only prompts.

`self_edit` follows the repository trajectory: official four-shot generation,
pinned post-processing, then official self-editor generation. `t5_edit` only
accepts an explicitly supplied initial candidate whose artifact SHA is bound in
the request; the worker does not claim that candidate came from released
InfoCQR weights.

## License and source gates

The audited checkout has no `LICENSE`, `COPYING`, or `NOTICE` file. Its source
and included generation caches therefore remain `NOASSERTION` and are never
vendored. Execution requires both:

- worker startup with `--acknowledge-noassertion-license`;
- `source_license_acknowledged=true` in every hashed request.

The worker verifies the exact commit when Git metadata is present, every
relevant source-file hash in all cases, the README evidence hash, and the
continued absence of a license file. A GitHub source tarball is identified by
the complete audited source-hash set.

## Bound experimental identity

Every request binds the QReCC revision/split, conversation source/number/turn,
ordered `NewContext`, current query, prompt source SHA, teacher deployment
fingerprint, retriever/corpus fingerprint, corpus ID, cache byte SHA when
replaying, seed, and model-call/token/cost ceilings. Endpoint and credentials
are startup-only and cannot be injected through JSONL.

The upstream first-turn scripts copy `Truth_rewrite` when `NewContext` is empty.
Cached first turns therefore fail closed unless the caller deliberately enables
`allow_first_turn_truth_rewrite`; live teacher mode calls the official prompt
function instead and never receives `Truth_rewrite`.

## Verify and run

Verify source plus a manifest-known author cache without importing provider
SDKs or making a model call:

```bash
python workers/infocqr/driver.py \
  --checkout /opt/upstreams/InfoCQR \
  --known-cache test_rw_fsl \
  --verify-only
```

Run live generation against either a hosted OpenAI-compatible API or local
vLLM endpoint:

```bash
python workers/infocqr/driver.py \
  --checkout /opt/upstreams/InfoCQR \
  --acknowledge-noassertion-license \
  --backend-config-json "$INFOCQR_BACKEND_CONFIG"
```

Example configuration:

```json
{
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key_env": "INFOCQR_API_KEY",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "fingerprint": {
    "provider": "vllm",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "deployment-or-weight-revision"
  },
  "timeout_seconds": 60,
  "max_output_tokens": 2560,
  "temperature": 0,
  "input_cost_per_million": 0,
  "output_cost_per_million": 0
}
```

A custom cache requires all of `--cache-path`, `--cache-artifact-id`,
`--cache-sha256`, `--cache-profile`, and `--cache-split`.

## Deliberate limits

- The repository publishes T5 training/test scripts but no student checkpoint.
  This legacy teacher/cache worker keeps its `T5_CHECKPOINT_UNAVAILABLE` gate.
  The separate [trained student adapter](../../docs/infocqr-student.md) now
  accepts a caller-mounted, checksum-pinned trained checkpoint and provenance
  receipt. It executes native source serialization/collation and local T5
  inference; no generic t5-base, teacher or author cache is substituted.
- The legacy OpenAI SDK call is replaced only at the provider transport seam.
  Prompt construction still executes the exact pinned official function.
- `post_processing.py` exposes only a file-oriented `main`; its exact pinned
  branch order is transparently reproduced and hash-labeled in the response.
- Cache/live plumbing and source prompt parity are tested. Paper-result parity
  is not claimed.
