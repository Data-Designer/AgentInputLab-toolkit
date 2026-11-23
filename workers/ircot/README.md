# Pinned IRCoT worker

This directory adapts the Apache-2.0 IRCoT repository at commit
`3c1820f698eea5eeddb4fba3c56b64c961e063e4` to AgentInputLab's existing
`IRCoTControlledWorker` contract. The upstream
`BestFirstDecomposer.find_answer_decomp` controller remains responsible for
the initial retrieval → CoT generation → exit check → follow-up retrieval
loop. AgentInputLab only applies bounded controls, freezes external bindings
for the test profile, observes controller transitions, and validates the
resulting trajectory.

## Identity profiles

- `fixture.tiny.frozen.v1` is a manifest-locked `controlled_fixture`. It runs
  the real upstream IRCoT loop against the local, checksum-pinned prompt,
  retriever responses, and generator responses. It is deterministic, offline,
  and may be called in-process through `PinnedIRCoTWorker`.
- `upstream.hotpotqa.codex.v1` and
  `upstream.hotpotqa.flan_t5_xxl.v1` are `upstream_native` identities. They
  retain the pinned upstream HotpotQA config/prompt and require exact
  generator, retriever, corpus, and index attestations. They must run in the
  isolated JSONL process; the in-process adapter fails with
  `ISOLATION_REQUIRED`.

The historical clients do not expose complete token telemetry through the
original generator interface. Their JSONL execution currently rejects with
`TOKEN_USAGE_OBSERVER_REQUIRED` before any model request, rather than reporting
unobserved token counts as zero. Source/identity inspection remains available.

The fixture identity must not be reported as an upstream-native reproduction.
Both identities, the upstream commit, manifest/config/prompt hashes, binding
attestation, driver hash, and runtime-adapter hash are exposed in the worker
fingerprint.

## Fail-closed compatibility gate

`compatibility_manifest.json` pins the upstream license and every source file
used by the adapter. The driver rejects a wrong commit, dirty checkout,
changed license/source/config/prompt/bindings, an unlisted profile, or an
unattested production binding before execution.

The historical upstream environment also needs optional packages and model
assets that are deliberately not hidden by the adapter. Identity inspection
reports `full_api_status=dependency_missing:<package>` when the complete
upstream API cannot import. A production call returns a structured
`DEPENDENCY_MISSING` response (for example when `_jsonnet` is absent) rather
than substituting another implementation. The frozen fixture supplies only
small compatibility shims needed to execute its offline trajectory; these do
not apply to an upstream-native profile.

Production endpoints and fingerprint hashes come only from the environment
keys named in the manifest. Request JSON cannot provide a path, module,
command, endpoint, or credential, and fingerprints containing credential
fields are rejected.

## Usage

For the offline bridge fixture, point the runtime at an exact pinned checkout:

```python
from pathlib import Path

from workers.ircot import PinnedIRCoTWorker

worker = PinnedIRCoTWorker(
    environment={
        "AGENTINPUTLAB_UPSTREAM_ROOT": str(Path("/path/to/pinned/ircot")),
    }
)
```

Pass `worker` to `IRCoTControlledWorker`; its `call_controlled` method accepts
the bridge's immutable `IRCoTWorkerRequest` directly.

For a production profile, start one JSONL worker process with its environment
prepared outside the request boundary:

```bash
AGENTINPUTLAB_UPSTREAM_ROOT=/path/to/pinned/ircot \
AGENTINPUTLAB_IRCOT_PROFILE=upstream.hotpotqa.codex.v1 \
python -m workers.ircot.driver
```

Use AgentInputLab's `JsonLineSubprocessWorker` as the client for that process.
The production environment must additionally provide the exact index and
binding hashes plus the model/retriever settings named by the selected
manifest profile. Secrets belong only in the isolated process environment;
they must never be included in request JSON or in a public fingerprint.

## Startup-bound controlled reader transfer

`PinnedIRCoTControlledRuntime` binds a toolkit backend, retriever, actual GPT-2
tokenizer and dependency fingerprints once at process startup. It executes the
pinned HotpotQA Codex configuration through the original
`build_decomposer_and_models` and `BestFirstDecomposer.find_answer_decomp`, with
the original retrieval, CoT sentence and exit participants. Native prompt
selection/fitting, spaCy sentence parsing, query preprocessing, deduplication
and exit decisions are preserved. The generator and retriever are controlled
replacements, so this is **not** historical Codex/Elasticsearch result parity.
It is also distinct from the one-pass common-support adapter.

Run each worker in its own process with an explicit private `HOME` equal to
`cache_home`; upstream import-time caches then cannot touch research caches.
The mounted source must remain at the manifest-pinned commit with a clean tree.
Provide real native dependencies, the native `en_core_web_sm` package, and a
hash-closed GPT-2 tokenizer; there are no fixture dependency shims or implicit
asset downloads in this adapter. Those startup fingerprints describe supplied
artifacts; they do not establish authorship, license clearance or real-model
acceptance merely by being present.

Use an isolated **Python 3.10** environment and install `agentinputlab[ircot]`.
This legacy worker extra pins spaCy 3.4.1 and Transformers 4.24.0; it is not a
promise that this optional environment supports every Python version supported
by the dependency-light toolkit core. It is separate from the ACE/LLMCompiler
environments, whose native client versions differ. The controlled reader is a
remote backend, so this extra does not install Torch or backbone weights. Mount
the matching `en_core_web_sm` 3.4.1 pipeline, GPT-2 tokenizer files and NLTK
English stopwords separately, recording their sources/checksums. No corpus or
model-weight download is triggered by importing the controlled adapter.

```python
from workers.ircot import PinnedIRCoTControlledRuntime
from agentinputlab.algorithms.acquire.ircot import IRCoTControlledWorker, IRCoTControls

worker = PinnedIRCoTControlledRuntime(
    source_root=pinned_checkout, backend=backend, retriever=retriever,
    tokenizer=gpt2_tokenizer, tokenizer_fingerprint=trusted_tokenizer_manifest,
    dependency_fingerprint=native_dependency_manifest,
    corpus_revision=corpus_sha256, index_revision=index_sha256,
    cache_home=private_worker_home, seed=17,
    max_model_calls=7, max_output_tokens=3584, per_call_output_tokens=512,
)
method = IRCoTControlledWorker(
    client=worker, corpus_revision=corpus_sha256, index_revision=index_sha256,
    prompt_hash=worker.fingerprint()["prompt_sha256"],
    controls=IRCoTControls(max_steps=7, max_output_tokens=3584),
)
```

This example reserves a separate downstream call/output allowance within an
overall eight-call/4096-output-token application budget. The caller must debit
both stages; the worker does not make the final downstream prediction. Generation
seeds are startup `seed + call_index`, not a request-selected seed. For reasoning
backends, explicitly choose `FinalContentStopBackend` to apply the native logical
newline stop to final content while charging all provider-generated tokens.
No retry, answer extraction call, or reasoning-to-final promotion is added.

Each response retains native controller events, exact actual generation and
retrieval records, and provider-observed tokens. If a transport fails without
usage, the error response marks `usage_is_complete=false` and records the number
of unknown-usage attempts separately; reported token sums are observed lower
bounds, not claims of zero consumption. Callers must preserve those error fields
and raw records. Default CPU/GPU counters are not measured compute costs.

## Verification

```bash
python -m pytest -q \
  tests/unit/test_ircot_worker.py \
  tests/workers/test_ircot_worker_driver.py
python -m ruff check workers/ircot tests/workers/test_ircot_worker_driver.py
```

The worker test builds a fake pinned checkout for hermetic bridge E2E coverage.
When `/private/tmp/agentinputlab-upstreams/ircot` is available, it additionally
checks the real pinned checkout's API and executes the frozen fixture through
the original upstream loop.
