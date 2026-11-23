# Controlled LongLLMLingua

`curate.longllmlingua.controlled` runs the original LongLLMLingua compression
under a frozen comparison binding. It binds the auxiliary compressor's model,
tokenizer, source files and loading policy; the downstream reader fingerprint;
the visible state's budget; the method settings; and the seed. Changing a
binding requires a new configuration. Identities are checked before and after
compression. No executable model object is stored on the intervention.

Install a local prerelease wheel with the `compression` extra. Mount the helper
checkpoint separately; model weights and experiment records are not distributed
with this library. A reviewed `model_files` mapping contains relative filenames
and their SHA-256 digests, including config, tokenizer and safetensors shards.
Checksums identify the operator's assets; they do not establish model authorship.

```python
from agentinputlab import InterventionContext
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab.algorithms.curate import LongLLMLinguaControlledBindings
from agentinputlab.integrations.external_workers.longllmlingua import LongLLMLinguaNativeRuntime

# state, reader, checkpoint_revision and model_files are operator-provided.
runtime = LongLLMLinguaNativeRuntime(
    "/path/to/reviewed/snapshot",
    model_id="NousResearch/Llama-2-7b-hf",
    revision=checkpoint_revision,
    file_sha256=model_files,
    device_map="cpu",
    torch_dtype="float16",
)
context = InterventionContext(backend=reader, external_runtime=runtime, seed=17)
bindings = LongLLMLinguaControlledBindings.from_context(context, budget=state.budget)
method = create_intervention("curate.longllmlingua.controlled", bindings=bindings)
result = run_intervention(method, state, context)
```

The runtime requires the pinned LLMLingua 0.2.2 source closure and verifies the
full model/tokenizer file closure before a live load. No assets are downloaded
implicitly, no remote model code runs, and no alternative checkpoint is chosen.
Injected compressor factories are labeled caller runtimes; they are not proof
of official weights or fresh native execution. The pure-data binding can also
be serialized for `RegistryInterventionBuilder`. A configured `ToolkitService`
exposes the same method to the SDK; HTTP clients cannot inject model paths/code.

Compression preserves the native question/instruction handling, context
ordering, iterative perplexity passes, token filtering and context-budget
semantics. This differs from LLMLingua-2's query-agnostic token classifier.
Settings are frozen per binding, with explicit `controlled_adapter` or
`factorial_adapter` labels. The reader is **bound, not invoked** by the curator;
the downstream benchmark driver must enforce the shared generation policy and
use the original scorer. A successful patch is not a benchmark score or a claim
that a complete paired experiment has been accepted.

The helper performs perplexity forward passes, not sampled text generation.
Its sampled input/output-token usage is not the same as its computational cost.
Successful calls record process CPU time and helper wall time separately from
reader generation; internal forward counts are explicitly unmetered. Preserve
failed helper diagnostics and supervisor/resource measurements as well. Native
compression statistics use the upstream GPT-3.5 statistics tokenizer, not the
helper or reader tokenizer. Strict replay of saved compressed bodies is analysis
reuse: retain the original helper/reader costs and failures, and do not report
replayed responses as new model calls or independent benchmark measurements.
