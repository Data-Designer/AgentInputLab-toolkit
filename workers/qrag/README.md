# Q-RAG frozen-checkpoint sidecar

This directory is an **inference-only compatibility bridge** for
`griver/Q-RAG@e7598130b63eaab868469877e88bcacc59a44efb`. It does not vendor the
upstream checkout, checkpoint, tokenizer, base embedder, datasets, or reader.
It does not expose RL/PQN training.

The bounded common-support profile is executable with the repository-level
driver below. It uses the same six BM25 candidates for both paired arms, lets
the released frozen policy select exactly two real candidate documents, and
passes only those documents to the shared downstream LLM. This is the profile
used for cross-family comparison; it remains distinct from paper-result parity.

## Release boundary

- Source: user-mounted, read-only, repository license `NOASSERTION`. The
  upstream README claims CC BY 4.0, but there is no root `LICENSE`; the source
  acknowledgement is a risk acknowledgement, not legal clearance.
- Checkpoint: user-mounted
  `Q-RAG/qrag-ft-e5-on-hotpotqa@e0e48389...`, separately acknowledged as
  CC BY 4.0. `model_best.pt` and `config.yaml` are hash-pinned.
- Embedder: user-mounted
  `intfloat/multilingual-e5-large@3d7cfbda...`; mutable `main` is rejected.
- Reader: a separately pinned controlled factor. The base profile records the
  upstream QwQ-32B family, but does not claim that the paper's exact reader
  snapshot is recoverable.

Accordingly, the public result is `controlled_adapter`, not end-to-end
`official_native` parity. The registry's `native_only` status means only the
released checkpoint treatment is eligible; it is not a parity claim.

## Preserved algorithm path

The sidecar runtime must call the official candidate/environment construction,
`PQN.select_action(..., random=False, evaluate=True)`, and `QAEnv.step`. Every
step records the available candidate indices, full finite Q-values, selected
index/Q-value, and accumulated selected document IDs. Q-RAG has no stop action:
termination is `fixed_step_budget`, `candidate_exhausted`, or
`empty_candidates`.

To avoid evaluation-label leakage, the runtime must construct `answer=""` and
`sf_idx=[]` internally and use upstream `DummyFeedbackModel`. It must never
load `GroundTruthFeedback`, compute EM/F1/reward, import `train_q_rag.py`, or
call optimizer/scheduler/update/save methods. This changes the paper's
ground-truth-assisted early termination and is why parity is not claimed.

## Security contract

The published 20 GB `model_best.pt` is a pickle checkpoint and upstream
`PQN.load` calls `torch.load(weights_only=False)`. Do **not** call that loader.
The dedicated runtime must use `torch.load(..., weights_only=True)`, validate
the expected inference keys, and call inference-module `load_state_dict` with
`strict=True`; optimizer and scheduler state are ignored. Run as non-root with
no network during selector construction, read-only mounts, no host secrets,
and CPU/GPU/memory/time limits. For production, a one-time isolated conversion
to a tensor-only artifact is preferred and must bind both source and converted
manifests.

The released file also serializes an OmegaConf `ListConfig` inside ignored
scheduler state. The common-support runtime therefore wraps the weights-only
load in a scoped allowlist containing only the inert OmegaConf metadata/node
constructors and built-in containers required by this exact pinned artifact.
Any other Python global still fails closed; the unsafe upstream loader is never
called.

`QRAGFrozenDriver` performs all license, source, checkpoint, embedder,
candidate order, profile, horizon, Q-value, usage, and response-attestation
checks before returning a successful envelope. `runtime_factory` and `reader`
are explicit dependency seams; the CI fake uses them without weakening the
production manifest.

### Reader budget configuration (protocol v3)

`QRAGControls(max_output_tokens=...)` sets the reader's per-call output ceiling;
4000 is the upstream default, not a hard minimum. `ExecutionBudget` independently
sets cumulative run limits. The effective reader cap is the smaller of the
configured ceiling and remaining output tokens. With no remaining tokens or
model calls, execution stops before constructing the selector runtime.

The reader callable has the signature
`reader(question, passages, artifacts, *, max_output_tokens) -> ReaderResult`.
It must send that effective ceiling to its actual model request, use its pinned
deployment, and avoid hidden retries. A legacy three-argument reader is rejected
before selector work. Return actual provider input/output usage, including
reasoning tokens where reported; do not replace missing usage with zero.

Both ends bind and verify the requested/effective budget in the v3 envelope.
The response records `reader_budget`, including `preset="upstream_default"` or
`"custom"`; actual usage above the effective ceiling is a consumed failure.
v2 requests/responses are not silently accepted under this changed contract.
No checkpoint, selector horizon, prompt or historical result changes with this
configuration support. End-to-end paper parity is not claimed for either budget.

### Explicit instruction-preserving QA adapter

`reader_profile="factorial_instruction_preserving_reader_transfer"` is an
opt-in transfer variant, with `QRAG_INSTRUCTION_PROMPT_SHA256` and protocol
`agentinputlab.qrag.instruction_reader.v1`. The original two question-only
profiles retain protocol v3 and their original prompts. They are not silently
converted into this adapter.

Its callback must accept `max_output_tokens`, `task_instruction` and
`passage_annotations` as keyword arguments. Call the shipped
`render_instruction_reader_prompt(question, passages, task_instruction=...,
passage_annotations=...)` and use `QRAG_INSTRUCTION_SYSTEM_PROMPT` in the
actual model request. Task instruction and visible passage annotations are
hash-bound, preserved in selection order, and included only in the reader;
the frozen selector's question/candidate input is unchanged. This variant
rejects conversation history and tool execution. No labels reach the policy.

The final JSON schema contains `answer` and the three string arrays
`support_ids`, `evidence_ids`, `evidence_texts`. The parser accepts explicit
final JSON, not reasoning salvage or automatic evidence inference. Score the
model's predictions with the original benchmark adapter, retain zero scores
and malformed outputs, and record actual costs without retries. This is a
labelled prompt/input adaptation, not upstream native parity.

For IR, select `factorial_native_query_reader_transfer` explicitly and bind
`QRAG_QUERY_PROMPT_SHA256`. It uses the same task-input envelope and frozen
selector, but the model callback must use `QRAG_QUERY_SYSTEM_PROMPT` with
`render_instruction_reader_prompt`. Its sole output field is `query`, decoded
by `parse_native_query_reader_answer`; do not use the four-field QA system prompt
or salvage a mixed QA/query response. The configured file worker selects this
prompt automatically from the explicit profile. Existing profiles and their
prompt hashes remain unchanged; old responses are not new-profile evidence.

IR native-query decoding is a separate opt-in postprocessor:
`parse_native_query_reader_answer(final_text)`, identified by
`QRAG_NATIVE_QUERY_DECODER_ID`/`QRAG_NATIVE_QUERY_DECODER_SHA256`. It accepts
exact final `{"query": "nonempty text"}` JSON, not a QA wrapper or inferred
query. The worker's generated text, prompts, budgets and usage stay unchanged.
Use the returned query with the original corpus/ranker/native scorer, retain
the raw output and any original QA-decoding rejection, and record the selected
decoder alongside the new score. This path does not retry generation.

The pinned upstream `DummyFeedbackModel` omits the abstract `reset` method and
therefore cannot be instantiated with current Python ABC enforcement. The
runtime supplies only that missing lifecycle method by delegating to the
upstream `AFeedbackModel.reset`; it does not alter feedback, reward,
termination, candidate construction, Q-values, or the two-step PQN policy.

## Installed common-support API

Install the `qrag` extra in the dedicated non-root environment. Source,
checkpoint and embedder mounts remain operator-provided; acknowledge their
separate licenses before execution. The public distribution does not include
research campaign scripts or previous experiment results.

The installed API is
`agentinputlab.algorithms.acquire.qrag.LocalQRAGFrozenPolicyRuntime`. Construct
it with `source_root`, `checkpoint_root`, `embedder_root`,
`compatibility_manifest_path`, `compatibility_manifest_sha256`,
`embedder_manifest_sha256` and an explicit `device` (`cpu` or the operator's
authorized CUDA device). Both manifest hashes use
`agentinputlab.contracts.serialization.content_hash` on their parsed JSON
objects. The compatibility policy ships as
`workers/qrag/compatibility_manifest.json`; the embedder manifest belongs to
the mounted snapshot.

Pass this runtime as `InterventionContext.external_runtime` and a real
candidate retriever as `InterventionContext.retriever` when running
`QRAGCommonSupport` through `InterventionRunner`. The runtime loads only the
frozen selector; the caller runs and scores its separately bound downstream
model. Construct one runtime per isolated process to avoid sharing upstream
global PyTorch device or random state with unrelated applications.

`QRAGFrozenDriver` is a separate selector-plus-reader integration class, not a
ready-made command-line server. `python -m workers.qrag.driver` intentionally
exits with deployment guidance until an operator supplies its runtime and
reader. Do not point users at unpublished research-only script paths.

For a file-based selector-plus-reader entry, use `agentinputlab config run` or
`config evaluate` with `runtime.kind="qrag_native"`; see
[configuration](../../docs/configuration.md#q-rag-selector-plus-reader-from-a-configuration-file).
This binds the existing driver to the frozen policy and configured reader in
the installed `workers.qrag.configured` subprocess. It retains the original
asset, profile, prompt and budget checks. An explicitly hash-bound saved-policy
mode is available for replay; it never falls back to new policy execution.
The subprocess is process separation, not a replacement for a production OS
sandbox or the read-only mount/security controls described above.

## Checks

```bash
PYTHONPATH=src:. python -m pytest -q tests/workers/test_qrag_worker_driver.py
PYTHONPATH=src:. python -m pytest -q tests/algorithms/unit/test_remaining_runtime.py -k qrag
```

Set `AGENTINPUTLAB_RUN_REAL_QRAG_IMPORT_SMOKE=1` and
`AGENTINPUTLAB_QRAG_CHECKOUT=/read-only/Q-RAG` to run the opt-in symbol/hash
smoke. It does not deserialize or download a checkpoint.
