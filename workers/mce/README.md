# MCE isolated worker

This sidecar integrates
`metaevo-ai/meta-context-engineering@c4b7a7c2ce3ffc4bf4a74c52d2dd8a9a8fb14c30`
without moving generated Python or context contents into the AgentInputLab
host.

## Controlled semantic-patch path

`MCEOfficialRuntime` calls the upstream `mce.main.run_iteration` function for
baseline iteration zero and every configured evolution iteration. The function
must originate from the pinned `mce/main.py` SHA-256; preloaded `sys.modules`
substitutes are discarded.

This is deliberately **not** an `official_native` adapter. A hash-bound semantic
patch replaces exactly four globals already imported by `mce.main`:

- `run_meta_agent` and `run_base_agent` delegate to an isolated manager's
  remote `run_agent`;
- `load_interfaces` performs remote static validation and returns an opaque
  receipt, never a locally imported callable;
- `batch_evaluate` delegates to a separate trusted oracle supervisor.

The pinned `run_iteration` orchestration, baseline/sub-iteration loop, batching,
aggregation, and returned validation metric remain upstream code. The patch
manifest and digest are bound through worker identity, request, runtime
binding, artifact bundle, and response attestation.

The host freezes:

- train and validation example identities and input hashes;
- the interface and initial-context hashes;
- target, meta, and base model fingerprints;
- seed, iteration, batch, call, wall, and artifact limits;
- source commit and source-file hashes;
- sandbox image, policy, test report, and resource limits.

## Security boundary

The driver must run inside the container or microVM named by the
`MCESandboxAttestation`. The source mount is read-only and the workspace is
new for every request. Generated code may execute only inside that worker.
The validation oracle is mounted only in a trusted, out-of-process supervisor;
the generated workspace cannot see it, and only the supervisor returns metric
values. Provider calls pass through the same trusted boundary, which reserves
model/tool budget before transport. A production runtime must supply the exact
isolated-manager fingerprint and implement `bind_run`, `run_agent`,
`validate_interfaces_static`, `batch_evaluate`, and `usage_receipt`; any
missing or altered field fails closed. This is an operator-attested deployment
boundary, not a claim that a Python object can cryptographically prove its own
container or microVM.
Only path, media type, byte size, SHA-256, iteration score, and normalized
usage leave the worker. Symlinks, paths outside the workspace, test-split
fields, and inline file contents are rejected.

The public upstream currently raises `NotImplementedError` for its own E2B
flag, so AgentInputLab does not describe that flag as a working sandbox.
Isolation is supplied by the external worker deployment.

Upstream does not expose complete model-call/token accounting. The isolated
manager therefore binds target/meta/base model identities to a trusted
pre-call supervisor and must return an exact usage receipt. Missing or
different accounting fails closed. The runtime seeds Python once per run.
At the three stochastic patched seams it assigns deterministic logical seeds as
`base_seed + logical_call_index`, in upstream invocation order and before each
await. A batch evaluator may fan out into multiple provider calls; the attested
manager contract therefore requires each actual provider call to derive a
non-negative 63-bit seed from SHA-256 over the canonical, versioned
`{logical_seed, stable_subcall_index}` identity in stable sample/call order.
This avoids the cross-call collisions caused by simple integer addition. The
manager passes that seed through `before_seeded_model_call`, which verifies the
derivation before reserving the provider call. These versioned seed modes are bound into the
semantic-patch manifest, manager fingerprint, run binding, sandbox attestation,
and response attestation. This remains an operator-attested boundary rather
than cryptographic proof of a remote provider's sampling implementation. The
configured wall deadline applies to the complete run.

## Status boundary

The checked path is an `external_only`, controlled thin-wrapper L2 integration.
It is not an official E2B path or a paper-result reproduction: model
credentials/local weights, a production isolated-manager deployment, a
task-specific upstream environment, and generated artifacts remain operator
supplied. The fixture runtime is rejected unless explicitly enabled.

An operator constructs `MCEWorkerDriver` with the trusted runtime, artifact
store, and sandbox attestation, then exposes it with
`workers.mce.serve_jsonl(driver, input_stream, output_stream)`. The module CLI
intentionally refuses to invent those trust-boundary dependencies.

## Text-artifact fitting API

The separate text-only profile produces Markdown context instead of executable
interfaces. Install `agentinputlab[mce-text]` and mount the pinned upstream
checkout. In a dedicated child process, set `PYTHON_DOTENV_DISABLED=1`, give the
unused upstream client a non-secret dummy `OPENAI_API_KEY`, and set its
`OPENAI_API_BASE=http://127.0.0.1:1/v1`. Actual inference uses the explicitly
supplied backend; configure its credentials under a different environment key.

```python
from agentinputlab.recipes import (
    MCETextFitControls, MCETextOfflineRecipe, MCETextArtifactCommonSupport,
)

# train and validation are disjoint sequences of MCEExample objects, with
# input={"question": ..., "context": ...}, a text expected_output, and the
# corresponding split. backend is a real, metered LLMBackend.
artifact = MCETextOfflineRecipe(MCETextFitControls(
    iterations=1, train_batch_size=2, per_call_output_tokens=512,
    max_model_calls=32, max_output_tokens=16384, seed=17,
)).fit(
    checkout="/path/to/pinned/meta-context-engineering",
    output_root="/path/to/new/fit-directory",
    train=train, validation=validation, backend=backend,
)
method = MCETextArtifactCommonSupport(artifact)
```

This API runs pinned `run_iteration` for the baseline and evolution iterations.
Validation references remain scorer-only; first-best validation selects the
artifact. Baseline selection yields an explicit NO_OP. Fit costs and seed
`base_seed + logical_call_index` are recorded separately from downstream use.
Record provider requests before transport and enforce a child-process/provider
deadline: the API's wall check is cooperative. Claude-agent and embedding SDK
imports use sentinels that raise if invoked; their execution and the full
generated-code sandbox are separate profiles, not covered by this text fit.

## Frozen executable-artifact evaluation

`recipe.mce.final_artifact_eval` uses `MCEFinalArtifactEvaluator`, not the
text-only profile. Keep the `MCESplitAudit` computed from the original fitting
and validation examples together with the returned `MCEContextArtifact`:

```python
from agentinputlab.recipes import (
    MCEFinalArtifactEvaluator, MCEFinalControls, MCEFinalExample,
    validate_mce_split_isolation,
)

# artifact is returned by MCEOfflineRecipe.fit; client is the same isolated
# sidecar with its original persistent artifact store and model identities.
fit_audit = validate_mce_split_isolation(train_examples, validation_examples)
evaluator = MCEFinalArtifactEvaluator(
    artifact=artifact, split_audit=fit_audit,
    interface_signature=interface_signature, client=client,
    controls=MCEFinalControls(
        max_model_calls=8, max_tool_calls=8,
        max_output_tokens=4096, max_output_tokens_per_call=512,
    ),
)
result = evaluator.evaluate(
    (MCEFinalExample("heldout-id", {"question": "..."}, expected_output="..."),),
    context=context,
)
```

These budgets apply to the **whole evaluate call**. Use one example per call
when each benchmark position has its own 4096-token/8-call budget. Increasing
the collection size does not silently multiply a budget. Fitting costs remain
separate. Model/source/interface/environment/storage identities and the
original fit split audit are checked before dispatch. Test IDs and input
hashes must be disjoint from both fitting and validation.

The sidecar materializes the frozen regular-file closure into a fresh generated
workspace. Native task loading and the existing isolated-evaluation seam run
without meta/base-agent evolution; generated interfaces are never imported by
the sidecar or caller host. The manager must provide
`final_evaluation_fingerprint()` equal to
`workers.mce.driver.final_evaluation_manager_contract()`. Its actual provider
transport must pass the requested cap to `before_seeded_model_call`, retain the
returned reservation ID, and pass that ID to `after_model_call`. Concurrent
reservations cannot exceed the aggregate token/call budget. Use
`failed_model_call(reservation_id=...)` for unknown failed-response usage;
that reservation is quarantined, not reported as zero actual token usage.
This is an operator-attested boundary, not proof that a remote provider obeys
its requested generation cap. Measured provider overruns are retained and
rejected.

Configure a persistent, private `oracle_root` on `MCEWorkerDriver`. Each final
request gets a create-only `mce-final-<request_id>` evidence directory there,
outside the generated workspace, containing the claim, oracle, raw native
evaluation and terminal/cost record. A repeated identical request is rejected
before another model call; inspect the existing evidence instead of retrying.
The host receives per-position scored/failed rows, the unchanged native
summary, and a separately labeled all-position primary-metric mean that keeps
failed positions in its denominator. Neither fitting nor text-artifact smoke
tests certify this full generated-code runtime; production acceptance also
requires an available isolated manager and real model/task execution.

## Checks

```bash
PYTHONPATH=src:. pytest -q tests/workers/test_mce_worker_driver.py
PYTHONPATH=src:. pytest -q tests/unit/test_remaining_recipes.py -k mce
AGENTINPUTLAB_MCE_PINNED_SOURCE=/path/to/pinned/checkout \
  PYTHONPATH=src:. pytest -q \
  tests/workers/test_mce_worker_driver.py::MCEOfficialRuntimeTests::test_real_pinned_run_iteration_with_semantic_patch
```
