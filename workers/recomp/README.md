# RECOMP pinned inference worker

This worker executes the released inference paths from
[`carriex/recomp`](https://github.com/carriex/recomp) at commit
`51d4432151efb3275257a9407dc71d1e5ec6634d`. The upstream source is MIT;
checkpoints and data are separate assets whose upstream distribution does not
declare a license. AgentInputLab therefore never bundles or downloads them.
Execution requires a user-mounted checkout and checkpoint plus a byte-locked
checkpoint manifest whose `license` remains `NOASSERTION`; the bridge also
requires an explicit license acknowledgement.

## Supported paths and claim boundary

- **Extractive:** imports the pinned
  `run_extractive_compressor.get_contriever_scores` function and calls it with
  `top_k=-1`. One input `ContextBlock` is one already-segmented sentence. The
  worker neither segments documents nor retrieves candidates and rejects any
  changed ID, order, text, or score-vector length.
- **Abstractive:** uses the pinned prompt
  `Question: {question}\n Document: {passages}\n Summary: `, local Hugging Face
  `generate`, the upstream `batch_decode` options, and `.strip()`. A successful
  empty summary is preserved and clears augmentation context.
- **Paper-native extractive assembly:** NQ and TQA select the highest-scoring
  sentence; HotpotQA selects the two highest-scoring sentences. Use
  `RECOMPExtractiveControls.paper_native("nq"|"tqa"|"hotpotqa")`.
- **Controlled assembly:** arbitrary top-N/threshold/output-order controls are
  separately labelled `controlled_adapter`; they are not paper-native claims.

RECOMP assumes retrieved documents are provided. The bridge requires
`InterventionContext.retriever.fingerprint()` and binds that identity, the
candidate input IDs/order/content, checkpoint manifest, and usage to each
request. Changing the retriever is therefore visible and cannot silently share
an experimental cell or cache identity.

## Checkpoint manifest

Create a local JSON file outside the package:

```json
{
  "schema_version": "agentinputlab.recomp.checkpoint.v1",
  "mode": "extractive",
  "checkpoint_revision": "your-immutable-revision",
  "tokenizer_revision": "your-immutable-revision",
  "license": "NOASSERTION",
  "files": {
    "config.json": "<sha256>",
    "model.safetensors": "<sha256>",
    "tokenizer.json": "<sha256>"
  }
}
```

Every listed path must be a regular, non-symlink file below the mounted root.
At least one recognized weight file is required. `asset_manifest_hash` passed
to the adapter is the SHA-256 of the manifest bytes.

## Run and smoke-test

The official repository pins a legacy environment (`torch==1.13.0`,
`transformers==4.24.0`, `tokenizers==0.13.2`; the extractive script also imports
unpinned `pandas`, `rank_bm25`, and `tqdm`). Use an isolated container or virtual
environment, then run:

The official `torch==1.13.0+cpu` build is accepted only with `--device cpu`.
The official CUDA 11.7 build, `torch==1.13.0+cu117`, is accepted with `--device
cpu`, `--device cuda`, or `--device cuda:N`; other local build tags and newer
public versions are not admitted by this pinned worker. Full build versions
are retained in every runtime fingerprint. Use a Python 3.10 environment for the legacy wheels rather than
installing these dependencies into the current toolkit or model-server
environment. The worker and current toolkit communicate over JSON lines, so
they do not need to share an interpreter.

For abstractive/selective **inference only**, install the `recomp` extra in that
Python 3.10 environment. It does not install the training datasets/metrics or
run training. For a CPU worker, install the official CPU Torch build first so
the subsequent extra does not pull CUDA libraries:

```sh
python3.10 -m venv recomp-worker
recomp-worker/bin/python -m pip install 'torch==1.13.0+cpu' --index-url https://download.pytorch.org/whl/cpu
recomp-worker/bin/python -m pip install 'agentinputlab[recomp]'
recomp-worker/bin/python -m pip check
```

This extra does not prepare source/checkpoint files. After mounting them, the
host toolkit's `config prepare-recomp` command derives the ready file described
in [the configuration guide](../../docs/configuration.md). Set its
`runtime.python_executable` to this worker's Python. The extractive path has
additional upstream imports listed above and is not covered by this seq2seq
inference-only extra.

The production loader freezes all model parameters and selects evaluation
mode. This worker never trains: disabling gradient tracking avoids retaining
an unnecessary autograd graph in the upstream extractive scorer's full input
batch, without changing checkpoint values, precision, tokenization, candidate
order, sentence scores or decoding controls. Large batches can still exceed
device capacity; such failures retain their consumed usage and are not retried
or silently truncated.

```bash
python workers/recomp/driver.py \
  --checkout /mounted/recomp \
  --checkpoint /mounted/checkpoint \
  --checkpoint-manifest /mounted/checkpoint-manifest.json \
  --acknowledge-noassertion-checkpoint
```

The process accepts exactly one typed JSON request line. Model loading is
`local_files_only=True` and `trust_remote_code=False`; request data cannot pick
modules, paths, URLs, retrievers, or checkpoints.

The legacy production loader additionally requires `model.safetensors` (or a
safe-only shard index), requires the native safetensors availability check to
pass, and rejects `.bin`/`.pt`/`.pth` files in its closed mount. Transformers
4.24 predates the `use_safetensors` keyword; its verified local-file branch is
used without that unsupported argument. A hash-pinned `.bin` file alone
is not a safe pickle-loading policy. If upstream supplies only `.bin`, use a
separate modern PyTorch 2.6+ conversion environment with `weights_only=True`,
verify every tensor key, shape, dtype and value against the derived safetensors,
and retain both source and derived hashes. Then mount only the verified derived
checkpoint and tokenizer under a new closed manifest. This changes storage
format, not model parameters, quantization or inference settings; it does not
grant a missing asset license. Install the tested `safetensors==0.5.3` in the
isolated legacy worker environment, not the dependency-free core. Existing
extractive and abstractive checkpoints both require this safe-only mount.

To test a real pinned checkout without inference:

```bash
python workers/recomp/driver.py --checkout /mounted/recomp --smoke-import extractive
python workers/recomp/driver.py --checkout /mounted/recomp --smoke-import abstractive
```

The deterministic test suite uses a synthetic fake checkout and dummy weights
only to prove protocol plumbing. It reports `fake_checkout_plumbing`, never
paper parity. A successful real-checkout import reports
`pinned_upstream_source`; model/result parity remains an opt-in user gate that
requires the separately governed upstream checkpoint and benchmark artifacts.
