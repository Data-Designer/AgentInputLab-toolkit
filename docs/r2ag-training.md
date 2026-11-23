# Train and use a reader-aligned R²AG artifact

`acquire.r2ag.official_retrained` trains the pinned upstream RFormer and learned
projection against a caller-mounted frozen reader. The recipe uses native
retrieval features, HotpotQA prompts, marker replacement and the combined
language-model plus retrieval BCE loss.

```sh
pip install 'agentinputlab[r2ag-fit]'
```

## Prepare native features

Supply separate train and development JSON arrays in the original HotpotQA
format: `_id`, `question`, `answer`, ten `context` entries and real
`supporting_facts`. Keep evaluation queries outside both files.

```python
from agentinputlab.recipes import prepare_r2ag_native_features

features = prepare_r2ag_native_features(
    checkout="/mounted/RRAG",
    encoder_root="/mounted/multilingual-e5-large",
    encoder_manifest="/mounted/encoder-manifest.json",
    encoder_manifest_sha256="<manifest SHA256>",
    train_file="/mounted/train.json",
    dev_file="/mounted/development.json",
    output_root="/new/native-features",
    device="cuda:0",
    source_and_asset_acknowledged=True,
)
```

The extractor retains upstream bare query and `Title: ...` document inputs.
It saves the ranked documents, three cosine-derived features, file hashes and
embedding work. Model manifests use schema
`agentinputlab.r2ag.model_snapshot.v1` with a complete `files` mapping from
relative snapshot paths to SHA-256 hashes. Mount the complete standard
tokenizer and safetensors model files.

Quantized snapshots keep their declared quantization: the loader does not
silently dequantize or fall back to dense weights. It loads safetensors only.
The ancillary `metal/model.bin` in a GPT-OSS MXFP4 snapshot is checksum-bound
but is not deserialized. Preload the matching native kernels into
`KERNELS_CACHE` for offline execution; keep complete pinned snapshots so the
Hub's offline integrity checks can resolve them.

## Fit the native retrieval modules

```python
from agentinputlab.recipes import R2AGFitControls, R2AGOfflineTraining

artifact = R2AGOfflineTraining(R2AGFitControls(
    batch_size=1, max_steps=2, max_optimizer_steps=2, device="cuda:0",
)).fit(
    checkout="/mounted/RRAG",
    base_model_root="/mounted/reader-snapshot",
    base_model_manifest="/mounted/reader-manifest.json",
    base_model_manifest_sha256="<reader manifest SHA256>",
    features_root=features,
    feature_manifest_sha256="<feature-manifest.json SHA256>",
    output_root="/new/reader-aligned-fit",
    source_and_asset_acknowledged=True,
)
```

This example declares a two-step compatibility configuration. Defaults use
two epochs, batch size two, gradient accumulation two, PagedAdamW32bit at
2e-4, 3% warmup, constant scheduling and gradient clipping at 0.3. Training
uses all non-padding token labels, BF16 autocast and TF32 on CUDA. The explicit
modern-reader loop imports the native model and loss unchanged. A dedicated
`<R>` vocabulary row transports retrieval markers; the reader is frozen after
resizing. This configuration has its own recorded artifact identity.

The output retains the fit plan, exact native source revision, parameter
hashes, native component checkpoint, tokenizer, frozen marker rows and terminal
costs. Training and feature-extraction costs are separate from answer generation.
Every output directory is create-only; a failure retains its actual evidence
and consumed cost.

Reader-aligned fitting requires input gradients through the frozen reader.
The native finegrained FP8 kernel currently has no input backward formula, so
this recipe first verifies the manifest and model-config checksums, then rejects
FP8 fitting before reading large weight files or starting a reader forward.
This config-only rejection does not certify the remaining snapshot; supported
configurations still undergo complete file-closure and payload validation.
Inference support is
not evidence of training support. No alternative gradient estimator, precision
change or automatic retry is applied. Other reader configurations must pass
their own native fit and trained-inference checks.

## Generate with the matching reader

```python
from agentinputlab.recipes import R2AGReaderDocument

runtime = artifact.runtime(
    checkout="/mounted/RRAG",
    base_model_root="/mounted/reader-snapshot",
    base_model_manifest="/mounted/reader-manifest.json",
    base_model_manifest_sha256="<same reader manifest SHA256 used for fit>",
    device="cuda:0",
    source_and_asset_acknowledged=True,
)
documents = [
    R2AGReaderDocument(document_id, title, text, (rerank, neighbor, precedent))
    for document_id, title, text, rerank, neighbor, precedent in ranked_records
]
result_directory = runtime.read(
    query="Your question",
    documents=documents,
    output_root="/new/answer-attempt",
    seed=17,
    max_new_tokens=512,
)
runtime.close()
```

Supply ten unique documents in native descending retrieval-score order, with
their saved three-column feature vectors. Preserve the corresponding embedding
cost and input identity when reusing features. Inputs contain the query and
retrieved documents; reference answers and support labels belong to the scorer.

For new visible candidate pools, compute those features without labels:

```python
from agentinputlab.recipes import R2AGNativeFeatureEncoder, R2AGRetrievalDocument

encoder = R2AGNativeFeatureEncoder(
    checkout="/mounted/RRAG",
    encoder_root="/mounted/multilingual-e5-large",
    encoder_manifest="/mounted/encoder-manifest.json",
    encoder_manifest_sha256="<encoder manifest SHA256>",
    device="cuda:0",
    source_and_asset_acknowledged=True,
)
ranked = encoder.encode(
    query="Your question",
    documents=[R2AGRetrievalDocument(d.id, d.title, d.text) for d in visible_candidates],
    output_root="/new/query-features",
)
encoder.close()
# Pass ranked[:10] to runtime.read(..., documents=ranked[:10], ...).
```

Supply 10-100 visible candidates, at most 4 MiB of document text. Choose any
candidate retrieval explicitly before this call. The encoder runs the native
formula on the entire supplied pool, retains its ranking and records actual
embedding attempts and input-token usage separately from reader generation.
It does not receive reference answers or support labels. Feature outputs are
create-only and can be saved for reuse by matching trained readers.

The runtime restores both native trained modules and the marker rows using
tensor-only loading, then calls native `generate`. It saves the raw answer,
generated token IDs, conditioning tensor hashes and attempt costs. Pass that
answer to the benchmark's scorer. A generated candidate is a model response;
benchmark success is determined by the scorer.
