# ReFormeR pinned inference worker

This worker executes only the separable inference path from
[`aminbigdeli/ReFormeR`](https://github.com/aminbigdeli/ReFormeR/tree/72e52450a922bc0051b5b39c3a6307186a555137).
It loads the official `Reformulator` class from a user-mounted checkout and
directly calls `select_best_pattern()` and `apply_pattern()`. It does not vendor
or copy upstream source, run pattern induction, or claim paper-result parity.

## License gate

The pinned repository has no root `LICENSE`, `COPYING`, or `NOTICE` file. Its
README contains an MIT badge, but a badge is not a complete license grant, so
AgentInputLab records the source license as `NOASSERTION`.

Execution requires two explicit acknowledgements:

1. the adapter sets `source_license_acknowledged=True` per request;
2. the operator starts the worker with `--acknowledge-noassertion-license`.

The worker verifies the exact commit, README evidence hash, every declared
source hash, the continued absence of a license file, and a clean Git checkout.
If the upstream project later adds a license, the current manifest deliberately
fails closed until it is re-audited.

## Four non-interchangeable profiles

| Release | Pattern choice | Materialized query |
|---|---|---|
| `rewrite.reformer.repo_native_r_only` | official automatic selection | repository `r` only |
| `rewrite.reformer.paper_spec_q_concat_r` | official automatic selection | paper `q ⊕ r` |
| `rewrite.reformer.controlled_r_only` | manually fixed pattern, official application | `r` only |
| `rewrite.reformer.controlled_q_concat_r` | manually fixed pattern, official application | `q ⊕ r` |

Automatic selection calls the official selector and application methods.
Manual profiles are a controlled ablation: AgentInputLab fixes a named pattern
from the exact hashed pattern set, then calls only the official application
method. The `q ⊕ r` profiles concatenate outside upstream code because the
repository returns `r` while the paper specifies concatenation. These axes are
recorded in the release ID, request identity, response, trace, and patch hash.

## Bound comparison state

Every request binds:

- source commit and all relevant source hashes;
- prompt JSON SHA-256 and pattern-file byte SHA-256;
- the complete selected pattern, including its examples;
- model deployment fingerprint and retrieval fingerprint;
- ordered document IDs/text and their canonical retrieval-state hash;
- manual pattern ID when applicable;
- seed plus model-call, provider-token, and priced-cost ceilings.

Endpoint, model, API-key environment variable, checkout, and pattern path are
startup-only settings. JSONL requests cannot override them.

## Start the worker

Install the inference worker's source-import dependency in its isolated Python
environment. vLLM and upstream training dependencies are not required by this
remote-model seam:

```bash
pip install 'agentinputlab[reformer]'
```

First verify a clean checkout without importing vLLM or making a model call:

```bash
python workers/reformer/driver.py \
  --checkout /opt/upstreams/ReFormeR \
  --verify-only
```

Then serve it through an OpenAI-compatible endpoint. This endpoint may be a
hosted API or a local vLLM server exposing the OpenAI protocol.

```bash
python workers/reformer/driver.py \
  --checkout /opt/upstreams/ReFormeR \
  --patterns /opt/artifacts/extracted_patterns.json \
  --acknowledge-noassertion-license \
  --backend-config-json "$REFORMER_BACKEND_CONFIG"
```

Example backend configuration:

```json
{
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key_env": "REFORMER_API_KEY",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "fingerprint": {
    "provider": "vllm",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "deployment-or-weight-revision"
  },
  "extra_body": {},
  "timeout_seconds": 60,
  "max_input_tokens_per_call": 28000,
  "max_output_tokens": 500,
  "temperature": 0.1,
  "input_cost_per_million": 0,
  "output_cost_per_million": 0
}
```

The worker's `max_tokens` control covers provider **input plus output** tokens;
the shared runner's `max_output_tokens` budget covers generated output. For the
startup limits above, configure the automatic profile with
`ReFormeRControls(max_model_calls=2, max_tokens=2 * (28000 + 500))`. Manual
profiles need one such call allowance and an explicit `manual_pattern_id` from
the hashed pattern library. Do not raise the shared output budget to account for
input tokens or count unpriced local generation as zero compute cost.

## Deliberate limits

- Pattern induction symbols and hashes are audited, but induction is not
  executed by this inference worker.
- The official vLLM chat-template serialization is replaced by a transparent
  OpenAI-compatible message seam; prompt content still comes from the pinned
  official `PromptManager` and `prompts.json`.
- The repository's invalid-selection and no-document first-pattern fallbacks
  are preserved and surfaced explicitly.
- No official output fixture or benchmark-result parity is available yet, so
  all four profiles remain `external_only`.
