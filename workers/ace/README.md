# ACE generic text-task worker

This directory contains the real upstream worker for
`recipe.ace.offline`. It does not reimplement ACE: every successful request
constructs the pinned upstream `ace.ace.ACE` object and calls
`ACE.run(mode="offline")`.

## Pinned upstream

- Repository: `https://github.com/ace-agent/ace.git`
- Commit: `bcb7cea0504afad6f55fec4845dd4864c9f9eee7`
- License: Apache-2.0
- License SHA-256:
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

`compatibility_manifest.json` pins the license, package metadata, ACE
orchestrator, Generator, Reflector, Curator, prompts, LLM helper, logging, and
playbook helper sources. The worker verifies the Git HEAD and every listed hash
before importing anything from the mounted checkout. Symlinked pinned files,
source drift, a different commit, a dirty checkout (including untracked files),
and AppWorld imports fail closed.

The checkout is an operator startup argument. A JSONL request cannot select a
checkout, path, module, callable, command, endpoint, or environment variable.
The driver never invokes a shell and only invokes a fixed, argument-list
`git status --porcelain` subprocess during startup verification; requests
cannot influence that command. It never adds `ace-appworld` to the import path.
BulletpointAnalyzer is disabled.

## Install upstream dependencies

Use an isolated Python 3.10+ worker environment:

```bash
pip install 'agentinputlab[ace]'
python -m workers.ace.driver --checkout /opt/upstreams/ace --verify-only
```

The `ace` extra installs the dependencies used by this generic-text worker:
NumPy, the OpenAI-compatible client, Python-dotenv and Tiktoken. It does not
install the upstream checkout or unrelated embedding models, FAISS, AppWorld,
or the optional BulletpointAnalyzer stack. BulletpointAnalyzer remains disabled
in this profile. Mounted source and license hashes are still verified before
imports; installing dependencies is not a substitute for that verification.
The worker itself adds no hidden model client. Credentials must already be
exported in the worker environment using the upstream provider variable:

- `OPENAI_API_KEY`
- `SAMBANOVA_API_KEY`
- `TOGETHER_API_KEY`
- `COMMONSTACK_API_KEY`

Loading credentials from a checkout `.env` is disabled. The default path keeps
the endpoints implemented by the pinned upstream. An operator may instead bind
an OpenAI-compatible deployment once at worker startup with `base_url` and
`api_key_env`; JSONL requests still cannot choose or mutate an endpoint,
credential name, model, or deployment option.

Verify the checkout and import without starting JSONL service:

```bash
python -m workers.ace.driver \
  --checkout /opt/upstreams/ace \
  --verify-only
```

## Explicit LLM configuration

Serve mode requires `--llm-config-json`. There are no implicit model or pricing
defaults. The three fingerprints must exactly equal the
`ace_model_fingerprints` used by `ACEOfflineRecipe.fit`.

```json
{
  "api_provider": "openai",
  "max_tokens_per_call": 4096,
  "json_mode": true,
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key_env": "OPENAI_API_KEY",
  "extra_body": {
    "chat_template_kwargs": {"enable_thinking": false}
  },
  "roles": {
    "target": {
      "model": "generator-model-id",
      "fingerprint": {
        "provider": "openai",
        "model": "generator-model-id",
        "revision": "deployment-revision"
      },
      "input_cost_per_million": 1.0,
      "output_cost_per_million": 2.0
    },
    "reflector": {
      "model": "reflector-model-id",
      "fingerprint": {
        "provider": "openai",
        "model": "reflector-model-id",
        "revision": "deployment-revision"
      },
      "input_cost_per_million": 1.0,
      "output_cost_per_million": 2.0
    },
    "curator": {
      "model": "curator-model-id",
      "fingerprint": {
        "provider": "openai",
        "model": "curator-model-id",
        "revision": "deployment-revision"
      },
      "input_cost_per_million": 1.0,
      "output_cost_per_million": 2.0
    }
  }
}
```

Zero prices are valid for explicitly unpriced local deployments; they must
still be stated. `base_url`, `api_key_env`, and `extra_body` are optional,
startup-only deployment controls and are accepted only for `api_provider` set
to `openai`. For an operator-bound endpoint, the worker sends the fit request's
`controls.seed` to all three role models as well as seeding local algorithm
randomness. A different seed in `extra_body` is rejected before a provider call;
the worker does not silently override either declaration. Model/server seed
support determines reproducibility. Start the worker with the serialized object
as one argument:

```bash
python workers/ace/driver.py \
  --checkout /opt/upstreams/ace \
  --llm-config-json "$ACE_LLM_CONFIG"
```

The process reads one strict JSON object per stdin line and writes exactly one
JSON object per stdout line. Upstream stdout is discarded into a one-MiB
bounded hash sink; the response retains only its byte count and SHA-256 digest,
never raw prompts or responses. Exceeding the bound fails the request closed.
Duplicate JSON keys, non-finite values, unknown protocol fields, and model
fingerprint mismatches are rejected.

## Generic text-task mapping

The existing recipe sends `ACEExample` records. The worker maps each record to
the upstream task-neutral shape:

```text
ACEExample.input          -> question and optional context
ACEExample.expected_output -> target
```

A string input becomes the question directly. A mapping may contain exactly one
of `question`, `query`, `prompt`, `instruction`, or `q`, plus optional
`context`. With no recognized question field, the whole JSON value is rendered
canonically as text. Non-string targets are also rendered as canonical JSON.
By default, evaluation is stripped, case-insensitive exact match.

### Native retrieval feedback

For retrieval fitting, `ACERetrievalFeedback` binds an explicit benchmark
adapter, retriever, and disjoint training/validation groups. It preserves the
adapter's native qrels and exclusion rules. Training inputs contain the visible
information need and evidence; their evaluator target is an opaque case ID,
not a gold query or an answer. The upstream ACE reflection branch tests whether
the declared native metric is positive; validation selection uses its mean
(for example, nDCG@10), not text exact match. Invalid query JSON has zero
training feedback without extraction or another model call. Infrastructure
errors remain errors. This controlled task mapping is not a paper-parity claim.

```python
from agentinputlab.recipes import ACERetrievalFeedback

feedback = ACERetrievalFeedback(
    adapter=adapter, retriever=retriever,
    train=train_examples, validation=validation_examples,
    metric="ndcg_at_10", top_k=15,
)
train = feedback.fit_examples("train")
validation = feedback.fit_examples("validation")
```

An in-process service can pass `data_processor=feedback` to `ACEWorkerRuntime`.
For a subprocess worker, provide an operator-owned factory function that
constructs this object from verified local dataset and corpus configuration:

```bash
python -m workers.ace.driver \
  --checkout /opt/upstreams/ace \
  --llm-config-json "$ACE_LLM_CONFIG" \
  --task-processor-factory my_project.ir_feedback:make_feedback \
  --task-processor-config-json '{"dataset_config":"/opt/data/ir.json"}'
```

The factory receives the startup JSON as keyword arguments. It must construct
the same feedback binding as the fit caller, perform no model calls, and expose
`answer_is_correct`, `evaluate_accuracy`, and `fingerprint`. Dataset/corpus
acquisition belongs before fitting. The driver records the factory module and
configuration hashes and the processor fingerprint; factory stdout is bounded
and excluded from JSONL. A processor may expose `validate_examples` to reject
altered or held-out fit inputs before any provider attempt. Requests cannot
select or change this factory. Put credentials only in the environment, never
in the factory configuration. Without these options the original worker and
its text-scoring behavior are unchanged.

The worker independently recomputes the split-audit hash and recipe request ID.
The request ID binds the complete canonical train and validation records,
including expected outputs and metadata, as well as controls, model
fingerprints, playbook hash, split-audit hash, and upstream commit. The worker
requires disjoint train/validation IDs and never accepts test examples in the
offline request.

## Upstream execution and playbook fidelity

For every request the worker:

1. renders native bullets as upstream lines of the form
   `[id] helpful=X harmful=Y :: content`;
2. initializes the pinned `ACE` with the configured Generator, Reflector, and
   Curator;
3. derives `next_global_id` with the pinned upstream helper;
4. calls `ACE.run(mode="offline")` once, with one epoch, curator frequency one,
   and validation at the final training step;
5. observes, but does not reorder, the upstream Generator → Reflector → Curator
   calls and their shared mutable playbook state;
6. parses upstream `best_playbook`, preserving existing IDs, content, counters,
   and metadata;
7. emits replayable `add`, `edit`, and `delete` operations plus the exact
   AgentInputLab `ACEPlaybook` candidate hash.

To remain compatible with the current diff protocol, structured order is
canonicalized as existing bullet order followed by upstream-created bullet
order. This does not change ACE's text supplied during execution; it only makes
the returned diff deterministic and exactly replayable by
`ACEWorkerResponse.from_mapping`.

Run artifacts and detailed upstream logs live in a per-request temporary
directory and are deleted after the response. Only validated usage totals and
the selected playbook cross the JSONL boundary.

## Current parity limits

- The pinned `ACE.run` exposes only its whole-run validation-best playbook, not
  validation scores for every curator step. Therefore one worker `iteration`
  represents one complete real upstream offline run. Internal sample-level
  Generator/Reflector/Curator state transitions still execute unchanged.
- Request `max_rounds` maps to upstream `max_num_rounds`, i.e. reflection and
  regeneration rounds per incorrect sample. It is not the number of curator
  steps.
- At the pinned commit, upstream `apply_curator_operations` fully applies only
  `ADD`; counter changes appear as replayable `edit` operations in the worker
  response. The worker does not invent unsupported merge semantics.
- `max_model_calls` counts actual calls to the pinned components'
  `api_client.chat.completions.create` seam. Retry attempts share the same
  request counter and are rejected before the next provider attempt when the
  ceiling is reached. Successful agent results must still expose token metadata
  or the request fails rather than silently undercounting it.
- Token and cost ceilings are checked from provider-reported successful-call
  usage and explicit startup pricing. A call that itself crosses a ceiling is
  rejected after that call returns; the upstream API has no preflight token
  estimate. Failed provider attempts count toward `max_model_calls`, but their
  billable tokens or cost cannot be added unless the provider returns usage
  metadata for that failed attempt.
