# Train a Q-RAG value policy

`training.qrag.value_policy` runs the pinned upstream training script with its
PQN, TD(lambda) targets, dual-critic loss, AdamW, target-network updates and
ground-truth support-fact reward. Training uses mounted source and a closed,
full-sized multilingual E5 initialization.

Install the runtime in a separate environment:

```sh
pip install 'agentinputlab[qrag-fit]'
```

Supply separate native-format train and development JSON arrays. Supported task
formats are HotpotQA, 2WikiMultihopQA, MuSiQue and Babilong. The upstream
`QADatasetAdapter` projects their existing support labels; the toolkit checks
IDs, label references and cross-split query overlap. Keep test cases outside
these files.

```python
from agentinputlab.recipes import QRAGFitControls, QRAGValuePolicyTraining

controls = QRAGFitControls(
    steps_count=2, batch_size=2, accumulate_grads=1, environments=1,
    eval_interval=1, eval_episodes=1, warmup_steps=1000,
    max_optimizer_steps=2, seed=17, device="cuda:0",
)
artifact = QRAGValuePolicyTraining(controls).fit(
    checkout="/mounted/Q-RAG",
    initial_checkpoint="/mounted/e5-snapshot",
    initial_manifest="/mounted/e5-manifest.json",
    initial_manifest_sha256="<SHA256 of the manifest bytes>",
    train_file="/mounted/train.json",
    dev_file="/mounted/dev.json",
    output_root="/new/value-policy-fit",
    source_and_asset_acknowledged=True,
)
print(artifact.artifact_id, artifact.checkpoint)
```

This example declares a two-step compatibility configuration. Defaults retain
the upstream 50,000-step schedule and 1,000-step warmup; every override is saved
with the fit. Native checkpoints include the optimizer and several encoder
copies: reserve at least 128 GiB of output space and size GPU capacity for the
complete E5 policy. CPU/wall time, peak allocated CUDA memory, native loss and
actual optimizer steps are recorded separately from LLM calls.

The initial-model manifest has schema `agentinputlab.qrag.initial_model.v1` and
a `files` object mapping every relative snapshot file to its SHA-256. Include
the complete tokenizer and safe model closure. The initial E5 architecture is
24 layers with a 1,024-dimensional hidden state.

The output contains `fit-plan.json`, native training logs and checkpoints,
`parameter-proof.json`, `fit-terminal.json` and a content-addressed artifact.
`native-final.pt` is the final fitted candidate; upstream best/last snapshots
retain their own selection semantics. Unchanged snapshots share one native
serialization through hardlinks, reducing duplicate checkpoint storage. A failed fit keeps its terminal record
and consumed cost. Use a new, explicitly authorized attempt for any subsequent
training; the recipe never retries automatically.

Deploy fitted policies under their training artifact identity. The frozen
released-checkpoint adapter continues to require its original checkpoint.

## Use the trained selector

The trained runtime uses the upstream PQN/QAEnv two-document selection loop.
It loads all four network state dictionaries with `weights_only=True`, restores
the selected snapshot's alpha and keeps inference in evaluation mode. The
optimizer and scheduler remain outside the inference path.

```python
from agentinputlab.recipes import QRAGTrainedPolicyIntervention

runtime = artifact.runtime(
    checkout="/mounted/Q-RAG",
    initial_checkpoint="/mounted/e5-snapshot",
    initial_manifest="/mounted/e5-manifest.json",
    initial_manifest_sha256="<same initialization manifest SHA256 used for fit>",
    checkpoint_sha256="<SHA256 of the saved native-best snapshot>",
    source_and_asset_acknowledged=True,
    device="cuda:0",
)
method = QRAGTrainedPolicyIntervention(runtime)
# Pass method to InterventionRunner with your Retriever in InterventionContext.
print(runtime.fingerprint())
```

`selection="native_best"` is the default and follows the development-set
selection made during native training. Supply the checksum from that retained
snapshot's evidence. To use the final update, declare `selection="final"` and
its checkpoint checksum explicitly. These choices produce distinct deployment
identities; fix the choice before evaluating test cases.

The runtime consumes a question and retrieved documents. Record its embedding
work and CPU/GPU time separately from downstream reader generations. The common
support intervention retains the visible context and adds the selected real
documents, using the same placement rule as the released-policy adapter.
