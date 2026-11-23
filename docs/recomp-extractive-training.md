# Training a RECOMP extractive candidate

`training.recomp.extractive` executes the actual pinned
`carriex/recomp@51d4432151efb3275257a9407dc71d1e5ec6634d`
`train_extractive_compressor.py`. It uses the original `RALMDataset`,
`MultipleNegativesRankingLoss(dot_score, scale=1)`, mean pooling, batching,
optimizer, scheduler and fit/save loop. The script is not copied into the wheel.

Use a dedicated Python 3.10 process with the optional `recomp-fit` dependencies.
The upstream requirements omit sentence-transformers; this bridge explicitly
fixes it to 2.2.2 and records the complete declared dependency versions. For CPU,
install the official matching CPU torch 1.13.0 and torchvision 0.14.0 builds
before installing `agentinputlab[recomp-fit]`; otherwise the PyPI torch wheel can
include unused CUDA libraries. Do not install these legacy dependencies into a
model-server or existing research environment.

```python
from agentinputlab.api import create_intervention
from agentinputlab.recipes import RECOMPExtractiveFitControls

recipe = create_intervention(
    "training.recomp.extractive",
    controls=RECOMPExtractiveFitControls(
        epochs=1, batch_size=64, warmup_steps=0, max_training_steps=1,
    ),
)
candidate = recipe.fit(
    checkout="/mounted/recomp",
    initial_checkpoint="/mounted/safe-encoder",
    initial_checkpoint_manifest="/mounted/encoder-manifest.json",
    train_file="/mounted/official-train-sample.json",
    dev_file="/mounted/official-dev-sample.json",
    output_root="/new/candidate-run",
    source_and_asset_acknowledged=True,
)
```

Initialization uses the same closed safe checkpoint format as the
[inference worker](../workers/recomp/README.md). The caller explicitly chooses
the initial encoder. Continuing training from a released NQ compressor is a
warm-start acceptance variant, not a claim to reproduce the paper's original
base-model training. Native source and initializer bytes are hash-verified.

Training files must contain the original record-array format with
`dpr_instance.query`, `positive_ctxs`, `negative_ctxs` and `retrieved_docs` text /
EM labels. Labels must come from declared training data, not held-out toolkit
answers. Source train/dev provenance should be recorded in a sampling manifest.
The adapter rejects cross-split query overlap, malformed indices and a zero-pair
training set. It computes the exact active native condition
`negative.em < positive.em`; the upstream `num_negatives` option does not limit
that active loop. Planned optimizer steps are bounded before execution. No data,
documents, pairs or epochs are silently dropped to meet a budget.

Native blank-document candidates are retained exactly, including their EM
labels and positive/negative membership. They are not malformed input and are
not filtered out by the wrapper.

The native script constructs its dev dataset but supplies no evaluator to
`model.fit`. The artifact therefore records **no native held-out validation**.
The fit runs the real loss and optimizer, observes finite losses and actual
optimizer steps, and verifies changed encoder tensors. The live trained encoder
is exported to a safe-only checkpoint and every exported tensor is checked for
equality. Native output and logs remain in the create-only run folder. The
candidate is not automatically certified for downstream inference.

`candidate.checkpoint` and `candidate.checkpoint_manifest` can configure the
existing pinned worker. `candidate.inference_parameters(sentence_splitter_revision=...)`
returns the matching data-only component parameters. Candidate loading rechecks
the terminal, manifest and complete checkpoint closure; a raw native directory
alone is not a verified artifact.

Training optimizer steps, finite loss observations, CPU/wall time and provenance
are separate from the 4096-token / 8-call **test inference** budget. No generated
LLM response does not mean zero training compute. Nothing is bundled into PyPI,
the paper is not modified, and a failed consumed fit is retained without retry.
