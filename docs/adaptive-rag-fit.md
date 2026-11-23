# Adaptive-RAG native router training

`acquire.adaptive_rag.rebuilt_offline` reconstructs native silver labels and fits
a local T5 router through the unchanged author classifier at
`starsuzi/Adaptive-RAG@0c88670af8707667eb5c1163151bb5ce61b14acb`.
Install `agentinputlab[adaptive-rag-fit]` in a dedicated Python 3.10 environment.
This legacy training extra is separate from modern local inference extras.

The release invokes the original question-to-A/B/C seq2seq loss, native
AdamW/Accelerate loop, tokenizer preprocessing and post-fit validation.
It does not replace training with a three-class cross-entropy approximation.
The dependency lock uses Transformers 4.28.1 rather than the author's Git
development build; this deployment adaptation is recorded in each fit plan.

## Rebuild labels, then fit

Mount the exact clean author checkout, original processed JSONLines and native
**correct-prediction ID mappings** for the no-retrieval, single-retrieval and
multi-step teacher arms. `rebuild_adaptive_silver(...)` calls the author's
`label_complexity` unchanged. Its effective priority is A > B > C; IDs for which
no arm has a correct prediction are excluded. It writes create-only
`silver.json` and `provenance.json`, including input/source hashes and the
declared teacher revision. It makes no teacher calls itself.

Rebuild train and validation independently, with no shared query or ID. Bind
their final JSON files in a caller-owned provenance file:

```json
{
  "schema": "agentinputlab.adaptive-rag.training_data.v1",
  "teacher_revision": "an-immutable-teacher-and-predictions-revision",
  "files": {
    "train": {"sha256": "SHA256_OF_NATIVE_TRAIN_JSON"},
    "dev": {"sha256": "SHA256_OF_NATIVE_DEV_JSON"}
  },
  "source_provenance": ["HASH_OF_TRAIN_LABEL_PROVENANCE", "HASH_OF_DEV_LABEL_PROVENANCE"]
}
```

These are schema placeholders, not real data or acceptance evidence. Use an
operator-verified manifest binding your actual source and prediction files.

```python
from agentinputlab.api import create_intervention

recipe = create_intervention(
    "acquire.adaptive_rag.rebuilt_offline",
    controls={"epochs": 2, "batch_size": 8, "max_optimizer_steps": 10000,
              "device": "cpu", "seed": 17},
)
artifact = recipe.fit(
    checkout="/assets/Adaptive-RAG",
    initial_checkpoint="/assets/t5-router",
    initial_checkpoint_manifest="/assets/t5-router-manifest.json",
    initial_manifest_sha256="ACTUAL_MANIFEST_SHA256",
    train_file="/data/train.json", dev_file="/data/validation.json",
    data_provenance="/data/provenance.json",
    data_provenance_sha256="ACTUAL_PROVENANCE_SHA256",
    nltk_root="/assets/nltk-data", nltk_manifest="/assets/nltk-manifest.json",
    nltk_manifest_sha256="ACTUAL_NLTK_MANIFEST_SHA256",
    output_root="/runs/router-fit-unique", source_and_asset_acknowledged=True,
)
print(artifact.checkpoint, artifact.router_bindings())
```

For CPU, set `CUDA_VISIBLE_DEVICES=""` **before starting the process**. For CUDA,
select exactly one authorized device. The NLTK manifest uses the genuine English
resource format documented in `recomp-abstractive-training.md`; no empty resource
directory bypass is accepted. Work on a local scratch filesystem when possible,
then archive the closed artifact; active output directories must not be moved.

The input checkpoint manifest schema is
`agentinputlab.adaptive-rag.router_checkpoint.v1`, with a `files` mapping of
every flat checkpoint filename to its SHA256. A standard T5 `config.json`,
`model.safetensors` and `tokenizer.json` are required. Remote code, pickle and
unlisted files are rejected. Initialization can be an explicitly identified
warm start; that is not original-paper initialization parity.

`max_steps` optionally bounds the native optimizer updates; it never silently
truncates input files. Tokenizer overflow is accounted for before the first
training update. Observers forward original calls and record real optimizer
steps, losses and validation generation. Compute is separate from test budgets
and is not free. Failed fits remain terminal, with no automatic repeat.

The trained artifact includes native logs/checkpoint/validation files, a safe
export exactly matching live trained states, the split and dependency plan,
changed-tensor counts, runtime costs and content-addressed identity. The loader
checks that closure. `router_bindings()` supplies the new router revision and
snapshot hashes. Safe checkpoints may omit the three standard tied T5 embedding
aliases; those names are expanded only from the same native shared weight,
with live storage identity checked. No missing untied weight is invented.
The returned bindings supply the
snapshot/weights/tokenizer hashes for the existing controlled worker;
generator/retriever/corpus/index bindings remain separate.
Pass `artifact.router_configuration()` to `LocalTransformersRouter` so native
fit/validation question truncation is retained (384 by default, capped by the
tokenizer's model limit). Record that configuration in the deployment fingerprint;
it is operator-owned and not an evaluation-request override.

Native classification validation is not reader benchmark quality. A bounded
fit proves execution, not published-score parity. Real acceptance must separately
check that the trained snapshot loads, routes and drives the selected arm on
held-out cases for each intended reader. Released FLAN-teacher labels do not
establish that labels were rebuilt from the toolkit's four current readers.
