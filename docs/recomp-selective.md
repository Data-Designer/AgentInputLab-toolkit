# RECOMP selective output modes

`curate.recomp.selective` runs the pinned abstractive compressor, applies its
learned empty-summary gate, and calls one reader. The default `answer_text`
mode retains the original QA prompt and result format.

For a retrieval benchmark, explicitly request a query instead of treating an
answer as a search query:

```python
method = create_intervention(
    "curate.recomp.selective",
    checkpoint_revision=checkpoint_revision,
    tokenizer_revision=tokenizer_revision,
    asset_manifest_hash=asset_manifest_hash,
    asset_license_acknowledged=True,
    output_format="ir_query_json",
)
result = run_composite_intervention(method, state, context)
query = json.loads(result["final_answer"])["query"]
```

Supply `create_intervention` and `run_composite_intervention` from
`agentinputlab.api`, and `json` from the standard library. The context needs
the pinned compressor worker, a reader backend, and the captured retriever
identity. Use the returned query with the benchmark's original corpus ranker
and scorer; compressor-selected evidence is not a replacement retrieval score.

The query mode has its own assembly identity,
`agentinputlab.recomp.learned_empty_gate_ir_query_reader.v1`. It retains the
input query, task instructions, compressor settings, and empty-summary policy.
It changes only the declared reader output protocol. Both stages share the
configured total budget; neither helper nor reader may exceed 512 output
tokens per call. Invalid query JSON remains a consumed failure without a
fallback answer, repair call, or retry. This is a controlled IR assembly, not
a claim of the original paper's QA prompt or score parity.

To accept a complete JSON code fence as a lossless envelope, explicitly set
`query_envelope="complete_json_or_fence"`. The default remains `strict_json`.
This option keeps the reader request, compressor, and raw final text unchanged;
it accepts only a complete JSON object or a single complete lowercase `json`
fence, rejecting prose, duplicate keys, extra fields, and multiple objects.
Decode its returned raw text with
`agentinputlab.algorithms.curate.recomp.parse_recomp_query_reader_answer` and
the same `envelope_policy`. The result policy records the decoder ID and hash.
When applying it to saved outputs, retain the original strict rejection and
costs, label the derivative policy, and do not count it as a new model trial.
