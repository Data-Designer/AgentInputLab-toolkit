# SmartSearch training interfaces

These training interfaces are separate from the frozen released-policy profile.
They use the pinned SmartSearch checkout at commit
`5e9e7ec0d6af34f45fad145d4bff442529e7e6ff` and keep its native prompts,
selection thresholds, SFT/DPO trainers and reward arithmetic. The complete
`training.smartsearch.sft_dpo_grpo` resolves to `SmartSearchNativeTrainingRecipe`,
a stage-oriented API over those native implementations. The entry remains
`retrain_required`: code availability and a closed optimizer chain do not
certify the complete data-construction recipe or benchmark compatibility.

## Data and policy stages

### Registered stage-oriented recipe

The Python registry/API returns the same concrete native implementations through
one entry point:

```python
from agentinputlab.api import create_intervention
from agentinputlab.recipes import SmartSearchPolicyFitControls

recipe = create_intervention(
    "training.smartsearch.sft_dpo_grpo",
    policy_controls=SmartSearchPolicyFitControls(gpu_processes=2, max_optimizer_steps=100),
)
# In the dedicated policy environment, with operator-supplied frozen inputs:
sft = recipe.fit(
    stage="sft", checkout=checkout, **bootstrap_snapshot,
    curriculum_root=initial_curriculum.root, output_root="new-run/sft",
    native_data_root=native_data_root, source_and_data_acknowledged=True,
)
sft_snapshot = sft.materialize_snapshot("new-run/sft-inference")
# Sample/refine/regenerate with THAT fitted policy, then prepare its DPO data.
dpo = recipe.fit(
    stage="dpo", checkout=checkout, **sft_snapshot,
    curriculum_root=dpo_curriculum.root, output_root="new-run/dpo",
    native_data_root=native_data_root, source_and_data_acknowledged=True,
)
merged = recipe.fit(
    stage="merged_dpo", checkout=checkout, **sft_snapshot,
    dpo_artifact_root=dpo.root, output_root="new-run/merged",
    native_data_root=native_data_root, source_and_data_acknowledged=True,
)
merged_snapshot = merged.materialize_snapshot("new-run/merged-inference")
```

`recipe.sample`, `recipe.annotate`, `recipe.prepare_curriculum` and
`recipe.prepare_grpo_data` expose the sampling, metered teacher and native
converter operations described below. Every stage uses its declared seed and
cost controls; empty selections and real failures stay recorded. Run the GRPO
stage in its separate native environment with explicit `grpo_controls` and
`grpo_cost_limits`, then call `recipe.fit(stage="grpo", ...)` using the same
arguments as `SmartSearchNativeGRPOTraining.fit` below. There is no implicit
`stage="all"` or automatic retry across incompatible native environments.

Once native stages have finished, `recipe.assemble` accepts `checkout`,
`sft_root`, `sft_snapshot`, `initial_curriculum_root`, `dpo_root`,
`dpo_curriculum_root`, `merged_root`, `merged_snapshot`, `grpo_root`,
`grpo_data_root` and a fresh `output_root`. It checks safe checkpoint bytes,
source revision, stage seeds, SFT→DPO→merge→GRPO identities, identical disjoint
case/question collections, and completed GRPO optimizer/usage receipts.
Snapshots use the three-field mappings returned by `materialize_snapshot`.
`SmartSearchTrainingChainArtifact.load` rechecks referenced inputs. Assembly
performs no model calls or fits and does not copy checkpoint weights. Its
receipt closes the native optimizer/export chain; sampling/teacher provenance,
held-out evaluation and complete benchmark compatibility have their own gates.

`SmartSearchTrainingCase` and `validate_smartsearch_training_splits` bind IDs,
normalized question hashes and provenance across disjoint train/development/test
sets. `SmartSearchPolicySampler.sample` calls the original ReCall loop with a
fixed retrieval provider, records each completion and usage, then applies the
original answer parser/F1 scorer. References do not enter generation prompts.
Its native 8,192-token calls and ten-turn limit are training-sampling costs,
not the toolkit's 512/4,096/8 downstream test budget.

`SmartSearchProcessTeacher.annotate` uses the original training process-reward
and redundancy code. Use `refine=False` for the initial SFT selection; native
query refinement belongs to subsequent policy sampling. Its helper keeps the
native 2,048-token per-call cap. Set explicit `SmartSearchTeacherControls` for
the whole stage; the adapter rejects insufficient allowance before a call
rather than shrinking that cap. Provider failures retain unknown usage and
stop the stage, with no retry or fabricated score.

`SmartSearchCurriculumPreparation.prepare` runs the original SFT/DPO/prefix
converters. Empty native selections remain empty. `SmartSearchNativePolicyTraining`
executes the pinned full-SFT/LoRA-DPO stages and CPU merge in fresh processes,
with immutable base/data/source bindings and observed optimizer/forward costs.
The native Qwen template is not silently replaced for other model families;
quantized full-fit/merge bases are rejected explicitly.

### Native policy environment

Use a dedicated Linux/Python 3.11 environment with compatible CUDA drivers for
the pinned native SFT/DPO stages. Do not upgrade a benchmark server's existing
environment in place. The toolkit's base installation stays dependency-free;
this optional training environment is separate from released-policy inference
and the GRPO runtime.

```bash
python3.11 -m venv .venv-smartsearch-policy
.venv-smartsearch-policy/bin/python -m pip install 'agentinputlab[smartsearch-policy]'
DS_BUILD_OPS=0 .venv-smartsearch-policy/bin/python -m pip install --no-build-isolation deepspeed==0.16.9
.venv-smartsearch-policy/bin/python -m pip check
```

DeepSpeed is installed after PyTorch so its build metadata can inspect the
selected Torch/CUDA runtime. `DS_BUILD_OPS=0` avoids an eager build of every
optional operation; native training may compile the operation it actually uses.
The caller still supplies pinned model snapshots and the two source mounts
below. A successful import alone is not a completed neural training stage.

### Completing the vendored data dependency

The pinned SmartSearch Git snapshot omits `llamafactory.data`. Its other 171
framework Python files exactly match official LLaMA-Factory commit
`3f0c3dc84d95e0ebdb632fd427ad30f7217c32e2`. Complete that dependency once with
`prepare_smartsearch_native_data_source(checkout=smartsearch_checkout,
official_checkout=llamafactory_git_repository, output_root=new_data_mount)`.
This copies only the 17 hash-pinned official data files to a separate directory.
It does not modify either checkout, load a model, or replace any trainer/loss.
The Git repository must contain the named historical objects; its working-tree
version is not used. This source completion is recorded in every stage plan.

Pass `native_data_root=new_data_mount` to both `fit` and `merge`. Each fresh
distributed worker verifies the original 171 files and the supplemental 17
files before mounting just `llamafactory.data`. There is no fallback to a newer
installed framework. Use a fresh dedicated process for every native stage.

## Fixed GRPO retrieval service

Prepare GRPO data with `SmartSearchGRPOExample` and
`prepare_smartsearch_grpo_data`. Each example wraps a split-labeled
`SmartSearchTrainingCase` and its reference-answer tuple. Explicit train and
development collections are required and checked against optional test IDs and
question hashes. No test set is sampled into development. The exported parquet
rows retain the native `data_source`, `question`, `ability`, `reward_model` and
`extra_info` fields; references are separate from the question. The tool schema
matches the pinned source, while the executable environment string is replaced
with the fixed retrieval identifier below. `SmartSearchGRPODataArtifact.load`
checks file/manifest hashes, parquet row counts, logical fields and split hashes.

`SmartSearchGRPOControls` defaults to native batch256, PPO minibatch64, eight
rollouts and two epochs. Changing these requires an explicit
`bounded_training_variant=True`; it is recorded as a separate training variant,
not identical full-scale training. The planner rejects an empty `drop_last`
loader or an excessive optimizer-step plan rather than duplicating rows or
truncating training. Hardware/cache limits are explicit operator controls.
These preparation/configuration interfaces do not run a GRPO optimizer.

`planned_steps(train_rows)` counts native dataloader/global steps.
`planned_actor_optimizer_steps(train_rows, ppo_epochs=...)` uses the native
PPO minibatch and epoch count to bound synchronized optimizer updates per rank.
Configuration admission checks both against `max_optimizer_steps`; it does not
truncate training to fit the cap. For example, batch4/minibatch2/one PPO epoch
has two actor updates per global step, not one. Runtime receipts must retain
actual rank counters and distinguish synchronized updates from their rank sum.

```python
import secrets
import threading
from agentinputlab.recipes import (
    SMARTSEARCH_GRPO_ENVIRONMENT,
    SmartSearchGRPORetrievalService,
    make_smartsearch_grpo_retrieval_server,
)

# retriever supplies fingerprint() with corpus_sha256/index_sha256, plus
# search(query, top_n=5) -> dictionaries containing id, contents and score.
tools = SmartSearchGRPORetrievalService(
    retriever, output_root="new-run/grpo-tools", max_retrieval_calls=400,
)
capability = secrets.token_urlsafe(32)
server = make_smartsearch_grpo_retrieval_server(tools, capability=capability)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
native_sandbox_url = f"http://127.0.0.1:{server.server_port}/{capability}"
# Put SMARTSEARCH_GRPO_ENVIRONMENT in each native dataset row's extra_info.env.
# Native rollout POSTs {env, call} to native_sandbox_url + "/execute".
```

Only the literal expression `wikipedia_search(query=<string>, top_n=5)` is
accepted. The environment is an identifier, not Python code. No `exec`, `eval`,
shell, imports or client-supplied URL is permitted. Native response keys remain
`result`, `output`, `error`; request/evidence/response records are create-only.
Repeated queries consume separate retrievals. A random capability, loopback
binding, bounded JSON bodies and an eight-request concurrency limit protect the
local transport. Keep the capability in private run configuration.

HTTP status codes distinguish malformed schema (400), missing capability
(401), missing route (404), body/media limits (413/415), quota/overload (429)
and retrieval failure (502). There is no automatic retry. Before accepting any
checkpoint, the executor must call `tools.assert_healthy()` and check every
native worker/optimizer receipt. A native rollout that absorbs an HTTP error
as observation text does not make that checkpoint valid. Shutdown of the local
server does not cancel an in-flight model attempt.

## Native GRPO reward seam

`SmartSearchNativeGRPOReward` loads the original reward functions and binds a
metered completion callable instead of the upstream hardcoded HTTP helper.
Call it with the native `(tokenizer, solution_str, ground_truth)` signature.
The provider returns `SmartSearchTeacherCompletion` and receives the original
temperature 0.6, cap 2,048, stop `</answer>`, plus the recorded operator seed.
The original helper's closing-tag append and reward formula are preserved.

Construct it with explicit references, `reference_split="train"`, a model
fingerprint, create-only output path and `SmartSearchTeacherControls`.
Development evaluation uses a separate instance with
`reference_split="validation"`, disjoint references and separate costs. Test
references are not accepted. Failed/empty/unmetered helper responses stop the
stage instead of becoming positive process rewards. Measured usage and unknown
reservations remain recorded after a failure. Quotas are per instance/process:
the execution coordinator must account for all workers and helpers explicitly.

These service/reward interfaces are prerequisites for the native GRPO executor,
not a replacement GRPO optimizer or a full-recipe compatibility certificate.

The execution worker also uses `NativeGRPORewardManagerContract` from
`agentinputlab.recipes.smartsearch_grpo_contracts` around the pinned native
manager. It completes the missing `reward_extra_info` dictionary field while
preserving the exact native reward tensor. A failed batch latches closed, so
the native driver's catch/fallback branch cannot re-score already consumed
items or pay the helper twice. This is an execution-contract repair, not a
new reward formula or an end-to-end GRPO acceptance result.

`SmartSearchNativeGRPOPrefixBuilder` runs the pinned native query-evaluation,
refinement, parser and tensor functions against explicit TRAIN references.
Its provider must return `SmartSearchTeacherCompletion` with measured usage.
Both native helper operations request 2,048 tokens at temperature 0.6 with no
stop string, and the adapter does not append a closing tag. These requests
differ from offline curriculum refinement; do not reuse its stop settings.
The instance records every request and response, reserves unknown usage on
transport failure, and latches failures so there is no automatic helper retry.
Distributed workers need an aggregate helper budget across all instances.

In a fresh worker, `install_native_rollout_batch_contracts` from
`agentinputlab.recipes.smartsearch_grpo_rollout` binds that explicit metered
prefix builder to the verified `vLLMRolloutWithTool` class. The execution
adapter builds one combined prefix batch outside the per-input loop, skips an
empty prefix batch, and selects non-tensor metadata by the same original row
indices as the tensors. It leaves the native single-trajectory generator,
group size and truncation intact. All generated outputs count toward costs,
including any subsequently discarded by native group truncation. Keep the
returned source/adapter binding with the run; this execution variant is not a
claim of bitwise equivalence to the buggy assembly or a completed GRPO fit.

## Aggregate execution costs

Create one `SmartSearchGRPOCostLedger` at the coordinator with explicit
`SmartSearchGRPOCostLimits`. All ranks must use that same ledger through a
synchronous coordinator client; a separate ledger per rank multiplies the
allowance and is not supported as aggregate enforcement.

`MeteredNativeGRPOEngine` observes the pinned token-only vLLM `generate` call.
It reserves the full batch × samples × requested output cap before execution,
then counts every returned sample **before** native prefix/group selection.
The original prompt objects, sampling parameters and output objects pass
through unchanged. Failed or malformed responses retain their unknown-usage
reservation and invalidate checkpoint acceptance, with no automatic retry.

`meter_smartsearch_grpo_helper` puts both native prefix and reward helper
requests behind the same helper allowance. It preserves each operation's
native stop rule, checks the bound model identity and measured usage, and
serializes calls to the shared teacher provider. `observe_native_grpo_optimizer`
observes the original optimizer's `step` without replacing the loss, gradients
or schedule. Check every actor rank's actual updates against the admitted
plan; do not add synchronized rank counts together.

The ledger reports logical input/output token counts. Per-rank worker receipts
also observe the outer FSDP forward boundary; initialization and internal
gradient-checkpoint recomputation are not included in that forward counter.

## Native GRPO execution

### Dedicated GRPO environment

The native GRPO worker uses a separate Linux/Python 3.11 CUDA environment, not
the SFT/DPO environment or the dependency-free base install. The
`smartsearch-grpo` extra pins the native Torch, Transformers, vLLM, Ray,
TensorDict, dataset and parser stack. Install FlashAttention **after** PyTorch:
its extension must be built against that environment's actual Torch/CUDA ABI.

```bash
python3.11 -m venv .venv-smartsearch-grpo
.venv-smartsearch-grpo/bin/python -m pip install 'agentinputlab[smartsearch-grpo]'
MAX_JOBS=2 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  .venv-smartsearch-grpo/bin/python -m pip install --no-build-isolation --no-deps \
  --no-binary=flash-attn flash-attn==2.8.3
.venv-smartsearch-grpo/bin/python -m pip check
.venv-smartsearch-grpo/bin/python -c \
  'import torch, flash_attn, flash_attn_2_cuda; print(torch.__version__, flash_attn.__version__)'
```

A compatible NVIDIA CUDA toolkit, compiler and sufficient build memory are
required. `MAX_JOBS` bounds compiler parallelism; it does not change model
budgets. A prebuilt extension with the same version label can still have a
different C++ ABI. An undefined-symbol error is an installation failure, not a
model result; build against the selected Torch instead of changing training
parameters or rerunning consumed model attempts. Verify a small real CUDA
FlashAttention forward/backward and the pinned native worker imports before
starting a paid training attempt. Import or kernel checks are not GRPO training
acceptance. Keep build logs and the installed environment's package versions.

Use `SmartSearchNativeGRPOTraining.fit` in a fresh process with the operator's
selected actor GPUs in `CUDA_VISIBLE_DEVICES`. It creates its own local Ray
cluster, binds all actor ranks to one loopback coordinator, and wires the
native prefix builder, native reward manager, fixed retrieval service and
aggregate costs. The helper callback runs outside that actor mesh. It returns
`SmartSearchTeacherCompletion`; the accompanying tokenizer callback counts the
same raw prompt's input tokens.

For colocated actor/reference workers, the execution adapter preserves the
native shared `Worker` container base while forwarding each role's original
configuration and constructor arguments to its accounting subclass. The
checksum-bound factory adjustment is process-local: it does not edit the
upstream checkout, change worker methods or replace the native optimizer.
Container actors explicitly disable automatic restarts and task retries.
The executor also prepares its new native checkpoint and rollout-journal
directories before any generation, so the pinned reward manager can save its
initial validation without depending on a prior checkpoint write.

The example below is an explicitly labeled small-training variant. Batch4,
minibatch2, eight rollouts and two epochs over eight training rows imply four
global steps and eight synchronized actor updates per rank. Output/call caps
are aggregate admission limits, not permission to silently truncate a native
request or retry a failed one.

```python
from agentinputlab.recipes import (
    SmartSearchGRPOControls, SmartSearchGRPOCostLimits,
    SmartSearchNativeGRPOTraining,
)

training = SmartSearchNativeGRPOTraining(
    controls=SmartSearchGRPOControls(
        train_batch_size=4, ppo_mini_batch_size=2,
        gpu_processes=2, max_optimizer_steps=8,
        bounded_training_variant=True,
    ),
    cost_limits=SmartSearchGRPOCostLimits(
        policy_requests=1024, policy_output_tokens=4_194_304,
        helper_requests=4096, helper_output_tokens=8_388_608,
        actor_ranks=2, optimizer_steps_per_rank=8,
    ),
)
# Operator-owned inputs from the preceding native SFT/DPO/merge/data stages:
artifact = training.fit(
    checkout=smartsearch_checkout,
    model_root=merged_dpo_model,
    model_manifest=merged_model_manifest,
    model_manifest_sha256=merged_manifest_sha256,
    data_root=grpo_data.root,
    output_root=new_run_directory,
    retriever=frozen_retrieval,
    generate_helper=raw_text_teacher_callback,
    count_helper_input_tokens=teacher_tokenizer_count,
    helper_fingerprint_sha256=teacher_fingerprint,
    model_provenance={"merged_dpo_artifact_id": merged_dpo_artifact.artifact_id},
    source_and_data_acknowledged=True,
)
print(artifact.checkpoint)
```

Native vLLM rank-derived engine seeds remain intact; the explicit recipe seed
binds data shuffling, helper requests and PyTorch RNG state. Native optimizer,
loss, scheduler, KL and group-selection behavior remain upstream. Source and
execution-adapter bindings are saved with the run.

The pinned native checkpoint manager's `hf_model` branch saves configuration
and tokenizer metadata, not HF weights. This executor additionally gathers
the actual live trained FSDP actor into create-only safe HF tensors, while
retaining the original native checkpoint. `SmartSearchGRPOArtifact.load`
verifies the learned export and its plan/terminal closure. Failed runs keep
their costs and unresolved reservations and cannot produce an accepted artifact.

## Held-out inference from the learned policy

After all native fitting stages close, `recipe.assemble(...)` verifies the
SFT → DPO → merged-DPO → GRPO artifacts, their safe model exports, source
revision, costs and training/development split identities. Bind the returned
chain to a provider serving **that learned GRPO export**, not the original
released-3B weights. The provider fingerprint includes
`training_chain_artifact_id`, `trained_policy_artifact_id`,
`model_manifest_sha256`, and the exact exported `model_root`.

```python
from agentinputlab.recipes import (
    SmartSearchInferenceControls, SmartSearchTrainedLocalModel, SmartSearchTrainingCase,
)

# recipe and verified_chain come from the completed native stages above.
# Start your own vLLM service at this URL with the learned exported-policy root.
learned_policy_provider = SmartSearchTrainedLocalModel(
    training_chain_root=verified_chain.root,
    base_url="http://127.0.0.1:8800",
    served_model_name="my-trained-smartsearch-policy",
)
predictor = recipe.inference(
    training_chain_root=verified_chain.root,
    model=learned_policy_provider,
    retriever=frozen_test_retriever,
    controls=SmartSearchInferenceControls(seed=17),
)
result = predictor.predict(
    case=SmartSearchTrainingCase("heldout-case-id", "Question from the test input", "test"),
    output_root=new_prediction_directory,
)
print(result["prediction"], result["usage"])
```

The model seam implements `fingerprint()`, `count_prompt_tokens(prompt)` and
`generate(prompt, temperature=..., max_new_tokens=..., stop=..., seed=...)`,
returning a measured `workers.smartsearch.driver.ModelTurn`. The retriever uses
the same fingerprint/search contract as the training sampler, with its own
frozen test corpus and index. These are explicit operator-controlled runtime
seams; a caller-provided fingerprint does not attest a remote deployment by
itself. A local loader or service handshake must verify the actual export.
An explicit `api_key` requires HTTPS for both inventory and generation requests.
Loopback clients bypass outbound proxies without changing process-wide proxy
configuration; API keys are excluded from fingerprints and request journals.

This inference API executes the pinned native `ReCall` loop and answer parser
without the whole-trajectory retry decorator. It never accepts answers,
labels or test rewards. Score its returned prediction with the benchmark's
evaluator **after** inference; `scored` remains false until that separate step.
Test question hashes and case IDs must be disjoint from the fitted training
and development sets. The curriculum sampler continues to reject test inputs;
calling it with a different split label is not a test-inference workaround.

The native 8192-token turn cap and ten-turn loop are metered separately from
512-per-call/4096-total controlled comparisons. Empty native predictions and
paid failures remain visible, with measured costs or conservative unresolved
output reservations; they are not automatically retried or called successes.
The immutable released-checkpoint inference profile is unchanged and cannot
be relabeled to claim that it used these newly trained weights.
