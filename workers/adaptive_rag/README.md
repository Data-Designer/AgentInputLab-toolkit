# Adaptive-RAG controlled router-parity sidecar

This directory implements the controlled execution boundary for
`starsuzi/Adaptive-RAG@0c88670af8707667eb5c1163151bb5ce61b14acb`. It does not
vendor a classifier checkpoint, FLAN model, Elasticsearch index, corpus, or
paper predictions.

## What the worker proves

Production startup verifies an exact clean Git checkout, commit, tree, pinned
source closure, all three prompt hashes, and content closures for the router
snapshot/weights/tokenizer, generator artifact/config, retriever
implementation/config, corpus manifest, and index manifest. The host also pins
the worker fingerprint. The worker then:

1. invokes the router on the question only;
2. extracts the real first decoder-step logits for `A`, `B`, and `C`;
3. applies deterministic `A`-first argmax tie breaking;
4. executes only the selected no-retrieval, single-retrieval, or bounded IRCoT
   **controlled reimplementation** arm; and
5. records every model/retrieval call through a supervisor that reserves budget
   before dispatch.

Each strategy backend must expose an exact component manifest that matches the
independently hashed deployment closures. Hermetic component classes are
rejected in production. The driver builds a hash-linked event log from calls it
supervised, and the AgentInputLab host recomputes the chain, trajectory, usage,
and selected-arm claim before setting `controlled_method_execution=true` and
`controlled_upstream_router_parity=true`. It always records
`upstream_arm_execution_parity=false`.

Evaluation requests have an exact schema and cannot carry gold, silver,
oracle, reward, target, source path, endpoint, or attestation fields. Retrieved
document metadata is allowlisted before both generation and host evidence
materialization. Paths, endpoints, credentials, and runtime components are
deployment-time inputs. Typed worker failures return partial usage and completed
events so the host ledger charges every attempted model/retrieval call.

## Production and fixture boundaries

`runtime.py` contains a local-files-only Hugging Face router. Generator and
retriever deployments implement the typed interfaces in `driver.py` and are
injected only at worker startup; their paths, endpoints, and credentials never
come from an evaluation request. The production driver defaults to
`fixture_mode=false` and rejects non-Git or dirty source trees.

The same module supplies `OpenAICompatibleGenerator` for an explicitly configured
toolkit backend and `ToolkitRetriever` for a separately bound real retriever.
They keep the driver's seed and output cap, forward final text without a
reasoning-to-answer fallback, strip document metadata, and make no automatic
retry. Declare a conservative `input_token_upper_bound` for the serving context
capacity; the driver records actual response usage, not that reservation.
The local router serializes its seed-setting/model call when shared by concurrent
workers. These are controlled arm adapters, not the upstream FLAN/Elasticsearch
pipeline. `tools/run_adaptive_controlled_acceptance.py` demonstrates a complete
installed-package run with existing T5-base transfer weights, source-bound small
QASPER paragraph corpora, all four configured readers, and native QA scoring.
Its question-only preparation applies to both arms; it is a different input and
output profile from the structured-JSON common-support cohort.

`tools/prepare_adaptive_boundary_cases.py` freezes a small source-only candidate
bank before native classifier inference, then chooses the first source-order
case per A/C route. It retains all screening attempts and costs, does not read
target-reader scores or supply references to the classifier, and leaves absent
routes missing rather than changing logits or growing the bank until a route
appears. This route-conditioned sample tests branches; it is not a benchmark
quality estimate. The generated selection receipt and source bank must be pinned
by checksum in the acceptance config:

```json
{"boundary_selection": {"path": "private/screen/selection.json", "sha256": "<trusted-64-hex-digest>"}}
```

The full-worker operator then verifies the source identities and executes the
selected cases on every configured reader. Native routing runs again during
each actual method attempt, so screening is recorded as extra preparation, not
as free work or as additional reader-budget allowance. The original fixed
102:104 cohort remains the default when no boundary receipt is supplied.

`fixture_runtime.py` is a small hermetic runtime for CI. Its component manifests
are permanently fixture-marked, its output is `hermetic_fixture`, and the host
accepts it only when a test explicitly sets `allow_fixture_runtime=True`. A
fixture proves protocol, controlled algorithm flow, and failure gates. It is
not official T5-large, Elasticsearch, FLAN, upstream IRCoT semantics, or
paper-result parity.

## Checks

```bash
PYTHONPATH=src:. python -m pytest -q tests/workers/test_adaptive_rag_worker_driver.py
PYTHONPATH=src:. python -m pytest -q tests/algorithms/unit/test_adaptive_rag_method.py
```

Set `AGENTINPUTLAB_ADAPTIVE_RAG_CHECKOUT` to an existing checkout at the pinned
commit to enable the opt-in real-source closure smoke. The smoke does not load
or download model weights.
