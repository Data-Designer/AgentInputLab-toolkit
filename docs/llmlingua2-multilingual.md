# LLMLingua-2: multilingual BERT

For a local prerelease build, install
`pip install '/path/to/agentinputlab-0.1.0a1-py3-none-any.whl[compression]'`.
The PyPI release remains gated by the separate compatibility acceptance.
The multilingual profile uses Microsoft's token-classification checkpoint,
not the larger XLM-R checkpoint and not the downstream reader's weights.

```python
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab import ContextBlock, FrozenAgentState

compressor = create_intervention(
    "curate.llmlingua2.bert_multilingual_official",
    device_map="cpu",
)
result = run_intervention(compressor, FrozenAgentState(
    query="Which meeting approved the project?",
    context=(ContextBlock(text="Paste the visible source passages here."),),
))
```

The default loader downloads only the pinned config, safetensors weights and
tokenizer files. For offline execution, pass a complete snapshot directory as
`model_name` and `model_config={"local_files_only": True}`. The six files are
checksum-locked to revision `5f0c82792b7ea14c6484e015b6a072009496b7f2` of
[`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`](https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank).
An existing incompatible snapshot fails before compression. No remote code or
alternative checkpoint is silently loaded. Injected factories are explicitly
caller-owned runtimes, not a certificate of official weights.

Compression uses LLMLingua 0.2.2's native LLMLingua-2 dispatcher and defaults
(`LLMLingua2Config`). It processes visible contexts, not the query or instruction;
the latter remain available to the downstream task runner. Native initialization
seeds the auxiliary compressor with 42; downstream generation has its own seed.
Native token counts use its GPT-3.5 statistics tokenizer, not the BERT classifier
tokenizer or the downstream reader tokenizer. Record helper time/memory and
downstream model usage separately. A compressed patch is not a benchmark score.

`official_native` identifies the original compression/checkpoint behavior;
`factorial_adapter` permits explicitly recorded experimental factor settings.
Neither mode certifies the historical paper environment or all benchmarks and
backbones. Continue the returned state through the benchmark's real workflow
and original scorer. The same profile is available via the JSON registry builder
and an operator-configured `ToolkitService`; HTTP clients cannot select arbitrary
paths, model code, or backend URLs. The SDK uses that Python execution core.

The isolated CPU acceptance environment uses LLMLingua 0.2.2, Transformers
4.38.2, Accelerate 0.28.0, PyTorch 1.13.0+cpu and NumPy 1.26.4. This records
one tested runtime binding; it is not a claim about every dependency combination
permitted by upstream LLMLingua's broad requirements.
