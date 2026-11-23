# RRR with a configured backbone and retriever

Use `rewrite.rrr.common_support_prompt` as the toolkit's recommended RRR profile.
It preserves the source-locked rewrite prompt, single-query parser, retrieval,
snippet assembly and reader stages while using the backend and retriever from
your `InterventionContext`. Keep the prompt artifacts and their source revision
fixed for each comparison.

```python
from pathlib import Path

from agentinputlab import InterventionContext
from agentinputlab.algorithms.reformulate.rrr import RRRCommonSupportConfig
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab.contracts.serialization import content_hash

# Operator-supplied, source-locked prompt artifacts and trusted source digest.
rewrite = Path(rewrite_prompt_path).read_text(encoding="utf-8")
reader = Path(reader_prompt_path).read_text(encoding="utf-8")
config = RRRCommonSupportConfig(
    rewrite_prompt=rewrite,             # Contains {query}.
    reader_prompt=reader,               # Contains {query} and {background}.
    prompt_source_sha256=prompt_source_sha256,
    rewrite_prompt_sha256=content_hash(rewrite),
    reader_prompt_sha256=content_hash(reader),
    top_k=5,
    max_rewriter_tokens=100,
    max_reader_tokens=300,
)
method = create_intervention("rewrite.rrr.common_support_prompt", config=config)
result = run_intervention(
    method,
    state,
    InterventionContext(backend=backend, retriever=retriever, seed=17),
)
```

The backend and retriever are real operator-configured dependencies. The core
runner meters both generations and retrieval against `state.budget`; configure
the per-stage caps to fit that budget. The reader uses the original query with
the retrieved snippets. Its answer is retained in the trace; the returned input
patch continues through the benchmark's native agent and scorer.

Results identify this implementation as `controlled_adapter`, with
`paper_checkpoint_used=False` and `ppo_training_used=False`. This is the
prompt-adapted RRR comparison, distinct from official PPO checkpoint inference.

For an explicitly configured external worker, select
`rewrite.rrr.prompt_external`. The optional `rewrite.rrr.warmup_external` and
`rewrite.rrr.ppo_external` profiles additionally require their respective
user-mounted checkpoints. Profile identities, acceptance evidence and costs
remain separate; selecting the prompt adapter never relabels a missing or
untested checkpoint profile as passed. See the [external worker guide](../workers/rrr/README.md).

## Checkpoint profile through a run configuration

The `rrr_checkpoint` file runtime binds an existing warm-up or PPO rewriter,
your configured reader backend, and the visible-context retriever. It performs
inference, not SFT or RL training. It never imports the quarantined upstream
source or switches a missing checkpoint to the prompt adapter.

```json
{
  "kind": "rrr_checkpoint",
  "checkout": "./RAG-query-rewriting",
  "checkpoint_root": "./rewriter/checkpoint",
  "checkpoint_manifest_path": "./rewriter/checkpoint-manifest.json",
  "checkpoint_manifest_sha256": "<trusted manifest SHA-256>",
  "rewrite_prompt_path": "./rewrite-prompt.txt",
  "reader_prompt_path": "./reader-prompt.txt",
  "acknowledge_credential_quarantine": true
}
```

Use this as `runtime` in `run.json`, with `release_id` set to
`rewrite.rrr.warmup_external` or `rewrite.rrr.ppo_external`. Configure `backend`
as the reader and `retriever` as the declared visible corpus. `checkpoint_root`
is the directory containing the manifest's model files, not necessarily the
directory containing the manifest itself. Local checkpoint inference needs the
worker's PyTorch/Transformers dependencies and mounted weights; assets are not
bundled in the wheel.

Keep the method's explicit `parameters`: source/checkpoint acknowledgements,
prompt file hashes, checkpoint revisions, and controls with
`search_mode: "controlled_static"` and the matching
`rewriter_profile: "warmup_checkpoint"` or `"ppo_checkpoint"`. The configured
reader deployment revision must equal `reader_checkpoint_revision`. Search and
index hashes use `content_hash(retriever.fingerprint())`; `corpus_revision` uses
`content_hash({"original_visible_state": state.state_hash,
"context": to_jsonable(state.context)})`. These bindings are case-specific and
must be prepared for the actual full visible state and budget, not copied from
another input. `config validate` checks configuration shape; `config run` also
verifies mounted assets, profile and state bindings before inference.

For an existing rewrite, add `rewriter_replay_path` and
`rewriter_replay_sha256` to the runtime. Its pinned JSON contains exactly:

```json
{
  "schema": "agentinputlab.rrr.rewriter-replay.v1",
  "request": {
    "prompt": "<exact expanded rewrite prompt>",
    "seed": 17,
    "max_tokens": 100,
    "checkpoint": {"role": "rewriter", "profile": "warmup_checkpoint",
      "revision": "<original revision>", "manifest_sha256": "<trusted hash>"}
  },
  "response": {"text": "<original generated text>", "input_tokens": 10,
    "output_tokens": 5}
}
```

Token counts above illustrate the schema, not measured results. Preserve the
original generation receipt and costs. A mismatch fails without live fallback;
the trace identifies reuse and retains the original logical rewrite usage.
Only the configured reader then generates a response. Without the replay fields,
the original local checkpoint generates the rewrite once. The two stages share
the run budget and there is no implicit third reader. `config run` returns an
input patch, not a benchmark score; continue through the native benchmark agent
and scorer for task results.
