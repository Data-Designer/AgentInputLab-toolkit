# Training a RECOMP abstractive candidate

`training.recomp.abstractive` executes the unchanged native
`train_hf_summarization_model.py` from
`carriex/recomp@51d4432151efb3275257a9407dc71d1e5ec6634d`.
It supplies declared local assets and observes the original `Seq2SeqTrainer`;
it does not replace the tokenization, loss, AdamW, scheduler or training loop.

Use a dedicated Python 3.10 environment with `agentinputlab[recomp-seq2seq-fit]`.
Install the official matching Torch 1.13.0 CPU or cu117 build first. Do not mix
these legacy dependencies into a model-server or research environment. CUDA
requires exactly one operator-selected visible device; CPU is also supported.

```python
from agentinputlab.api import create_intervention

recipe = create_intervention(
    "training.recomp.abstractive",
    controls=dict(epochs=1, max_steps=1, max_optimizer_steps=1, device="cuda"),
)
candidate = recipe.fit(
    checkout="/mounted/recomp",
    initial_checkpoint="/mounted/safe-t5",
    initial_checkpoint_manifest="/mounted/t5-manifest.json",
    train_file="/mounted/official-train-sample.json",
    dev_file="/mounted/official-dev-sample.json",
    nltk_root="/mounted/nltk-data",
    nltk_manifest="/mounted/nltk-resource-manifest.json",
    nltk_manifest_sha256="<sha256-of-resource-manifest>",
    output_root="/new/candidate-run",
    source_and_asset_acknowledged=True,
)
```

Training files use the native JSON array or JSONLines format (`.json` extension)
with original `question`, `passages` and `summary` strings. Empty summaries are
valid native labels and remain unchanged. Repeated training queries are valid;
normalized train/dev query overlap is rejected. Dataset provenance and a fixed
sampling rule should be captured before training. No test answers become labels.
The declared optimizer-step cap rejects an oversized plan rather than silently
truncating data. An explicit `max_steps` is recorded as a training variant.

The initialization uses the closed safe-checkpoint format of the
[RECOMP worker](../workers/recomp/README.md). Training from a released NQ
compressor is a user-selected warm start, not the paper's base-model training.

The pinned source checks for English NLTK `punkt` data on import. Provide genuine
resources from the official NLTK data repository with a hash-pinned manifest:
`schema: agentinputlab.recomp.seq2seq.nltk.v1`, `files: {relative_path: sha256}`.
Include both original English punkt files and safe English punkt_tab tables.
The adapter verifies those bytes; it does not create an empty directory to bypass
the dependency check or deserialize the legacy pickle. Native teacher-forced
evaluation does not invoke sentence tokenization or generate placeholders.

Both native pre-training and post-training validation are retained. Evaluation
uses teacher-forced loss (`predict_with_generate=False`); it is not generated
summary quality, ROUGE, best-model selection or downstream QA certification.
No reporting service, Hub push, resumed fit or automatic retry is enabled.

The artifact records actual optimizer steps, finite loss events, input/label
token positions, Trainer history, wall/CPU time and changed parameter tensors.
Training compute is separate from the 4096-token/8-call test inference budget;
GPU work is not fully metered and must not be presented as free. Native output
is retained. The trained model is exported to safe-only tensors with exact
live-model equality, including tied keys. Artifact loading rechecks all hashes
and terminal status. `candidate.inference_parameters()` configures the existing
abstractive worker. Its inference budget still caps each generation at 512.

Checkpoint weights, datasets and experiment records are not bundled into the
distribution. An actual trained artifact and independent downstream reader
acceptance remain required before this release profile can be certified.
