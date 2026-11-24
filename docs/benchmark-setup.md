# Find, prepare and use a dataset

These discovery commands (0.1.0a3+) read packaged metadata. They do not create
caches, download data or call a model:

```sh
agentinputlab datasets list
agentinputlab datasets list --search beir
agentinputlab datasets list --family reformulate --json
agentinputlab datasets show beir.arguana
```

`automatic_loader` tells you whether `load_benchmark()` supports preparation for
that entry. `source_download_bytes` is the size of **all source files for a cold
cache**, not the currently missing bytes, unpacked disk usage or GPU memory.
`exceeds_default_download_limit` flags sources larger than the default 512 MiB
download ceiling. Review upstream terms before using `--accept-license-review`.
Do not raise the ceiling just to inspect a dataset; use a documented source-side
sample for large datasets when appropriate.

## Cache-first data loading

Install the optional dependencies required by the selected adapter:

```sh
python -m pip install 'agentinputlab[benchmarks,evaluation]==0.1.0a19'
agentinputlab datasets prepare beir.arguana --cache-dir ./toolkit-cache
```

Preparation verifies pinned source hashes and reuses valid cached files. It never
silently overwrites a corrupt cache. To prohibit network access, pass `--offline`.
Set `AGENTINPUTLAB_CACHE` or `cache_dir` to choose a storage location; no changes
are made to your other research or model caches.

```python
from agentinputlab.benchmarks import describe_benchmark, list_benchmarks, load_benchmark

print([item['cell_id'] for item in list_benchmarks(search='arguana')])
info = describe_benchmark('beir.arguana')
adapter = load_benchmark('beir.arguana', cache_dir='./toolkit-cache', offline=True)
example = adapter.examples(info['split'])[0]
state = example.visible_state()
print(state.query)
```

Pass `visible_state()` to the model, **not** the entire example: the full example
also carries evaluator-only gold. Use the adapter's typed prediction object and
`score()` after real downstream execution. A successful input rewrite is not a
task score, and a retrieval query is not a QA answer. The
[real-execution guide](real-execution.md) covers readers, retrieval and native
scorers.

## Native and interactive runtimes

`native_adapter` means there is no automatic loader in `load_benchmark()` for
that entry; it does not mean the dataset has no adapter. Do not substitute a
different dataset to bypass its prerequisites.

| Dataset type | Preparation route |
| --- | --- |
| Prepared QA / retrieval | `datasets show` → `datasets prepare` → `load_benchmark`. |
| BFCL simple | `datasets prepare` → `load_benchmark(..., model_id=...)`; the pinned source archive supplies both 400 cases and the original Python AST scorer. |
| LongBench v1 | Install the `longbench` extra, then `datasets prepare` or `config evaluate --cache-dir` with license acknowledgement. Six subsets share three verified source files (113,943,156 bytes on a cold cache). Existing files can instead use `--longbench-config`; see [native text evaluation](configuration.md#longbench-native-text-evaluation). |
| STATE, ToolSandbox, ToolTalk and interactive arenas | Native provider/environment, checkpoint or task selection, and official continuation/scoring. Inspect `python -m workers.official_arenas.driver --help` and [real execution](real-execution.md). |
| τ³ banking checkpoints | `load_benchmark` additionally requires `checkpoints` and `evaluation_bridge` before downloading. |
| Large BEIR / TREC collections | Use the documented `BEIRSampleAdapter` or `datasets prepare-trec-sample` route when full sources exceed your storage budget. |

Algorithm eligibility and runtime compatibility remain separate from dataset
availability. Check `eligibility_gate`, `family_support` and the chosen algorithm
profile; a catalog entry is not a universal compatibility guarantee.

### BFCL simple: cache-first preparation and scoring

```sh
agentinputlab datasets prepare bfcl.v3.simple_python --cache-dir ./toolkit-cache
```

This acquires the checksum-pinned v1.3 source archive (44,552,456 bytes) and
normalizes the 400 Python simple cases. The archive remains necessary for
scoring, not just preparation. If it already exists elsewhere, use
`--source-archive /path/to/official-v1_3.tar.gz --offline` to verify and use it
without another copy or download. Keep that mounted source available.

```python
adapter = load_benchmark(
    "bfcl.v3.simple_python", model_id="your-actual-served-model-id",
    cache_dir="./toolkit-cache", offline=True,
)
```

The bridge uses the pinned native Python AST checker and function-call decoder;
this is per-case scoring, not the complete BFCL leaderboard runner. For an
end-to-end model request and score, use the [BFCL configuration entry](configuration.md).

## Debug without repeating model requests

Use `agentinputlab --debug COMMAND ...` for toolkit-error tracebacks locally.
Normal mode prints a short error and exits with status 2. Keep debug traces
private until reviewed for paths or application details. Fix configuration before
resubmitting: the CLI does not automatically retry consumed model calls.
