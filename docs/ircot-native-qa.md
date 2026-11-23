# Complete native-like IRCoT QA

`acquire.ircot.native_like_substituted` runs the pinned upstream
`ircot_qa_codex_hotpotqa.jsonnet` graph, not the retrieval-only configuration.
All six original participants execute: retrieve/reset paragraphs, generate a
CoT sentence, decide whether to exit, copy the original question, generate the
final answer, and extract it. The original best-first controller, prompt
selection/fitting, sentence parser and `return_qid_prediction` extraction remain
in the caller-mounted source checkout.

The source revisits the copy-question participant after extraction to drain its
task stack. That terminal visit makes no additional model call and is retained
in the controller trace.

The reader model and retriever/index are explicit substitutions for the
historical Codex/Elasticsearch deployment. A create-only substitution manifest
binds the source, configuration, prompt, tokenizer, native dependencies, model,
index and startup budgets. It is not a claim of historical or published-score
parity. The existing retrieval-only controlled worker is unchanged.

Run the worker in a dedicated Python 3.10 process with `HOME` set to its private
`cache_home`. Install `agentinputlab[ircot]` and mount the checksum-locked source,
GPT-2 tokenizer, native spaCy `en_core_web_sm` 3.4.1 and NLTK English stopwords as
described in the [worker guide](../workers/ircot/README.md). Importing the toolkit
does not fetch these assets or start a model service.

```python
from pathlib import Path
from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab.algorithms.acquire import write_ircot_substitution_manifest
from workers.ircot import PinnedIRCoTNativeQARuntime

runtime = PinnedIRCoTNativeQARuntime(
    source_root=source_checkout,
    backend=backend, retriever=retriever, tokenizer=gpt2_tokenizer,
    tokenizer_fingerprint=tokenizer_manifest,
    dependency_fingerprint=native_dependency_manifest,
    corpus_revision=corpus_revision, index_revision=index_revision,
    cache_home=Path(private_worker_home), seed=17,
    max_model_calls=8, max_output_tokens=4096,
    max_retrieval_calls=8, per_call_output_tokens=512,
)
receipt = write_ircot_substitution_manifest(
    runtime, Path(private_worker_home) / "substitutions.json",
    accept_substitutions=True,
)
method = create_intervention(
    "acquire.ircot.native_like_substituted",
    manifest_path=receipt["path"], expected_manifest_sha256=receipt["sha256"],
)
result = run_composite_intervention(
    method, FrozenAgentState(query=source_question, task_id=source_id),
    InterventionContext(
        backend=backend, retriever=retriever, external_runtime=runtime, seed=17,
    ),
)
```

The input is deliberately question-only: bind the corpus separately rather than
silently ignoring supplied context, policy instructions, tools or conversation
history. Evaluation IDs appearing among the pinned prompt demonstrations are
rejected before dispatch. Check benchmark eligibility and source/corpus scope
for the intended run; registering this entrypoint does not certify every arena.

Defaults preserve native top-6 retrieval, at most 15 active paragraphs, and
seven reasoning generations plus one native final-answer generation inside the
same eight-call/4096-output-token budget. The source generator's default
300-token cap remains 300 under the caller's 512-token ceiling; unused tokens
are not consumed. The final call is reserved, not added afterwards. Each actual
request uses the startup seed plus its call index. There is no retry, source
cache, independent downstream answer, or reasoning-to-final promotion.

The source backend defaults are temperature 0, top-p 1, zero presence/frequency
penalties and one sample. Configure and fingerprint those sampling defaults on
the supplied backend as well as any backbone-specific reasoning options.
`GenerationRequest` carries the source stage/options for inspection; arbitrary
backends are not assumed to interpret metadata as sampling controls. Reasoning
models can use the explicit `FinalContentStopBackend` for native logical stops,
with all provider-generated tokens still charged.

Successful results retain real corpus evidence, all native controller stages,
the original final prediction, actual provider usage and binding identities.
Empty or non-text native predictions and transport/budget failures remain
failures with available cost records; they do not trigger replacement attempts.
Run the independent benchmark scorer on `final_answer`. The toolkit API's
`scored=False` is not a benchmark score.

### Hotpot benchmark evaluation

For `config evaluate`, explicitly select the native answer-only variant in the
predictor JSON (this is not an instruction-following support-fact generator):

```json
{
  "schema_version": 1,
  "qa": {"output_format": "answer_only"},
  "ircot_input": {
    "projection": "hotpot_question_with_external_context.v1",
    "acknowledge_answer_only": true
  }
}
```

Use the prepared `run.json` with `--benchmark hotpotqa.distractor`, validation
split and the corresponding example ID. Bind that example's full visible
paragraph collection as the explicit BM25 corpus before `prepare-ircot`.
Evaluation checks paragraph text, titles, IDs, corpus and index fingerprints;
a different example needs its own index and prepared substitution manifest.
It rejects extra policy instructions, messages or tools rather than dropping
them. The receipt retains the original input hash and instruction, and marks
the declared answer-only projection. All original Hotpot metrics are computed:
supporting-fact predictions remain absent and are scored accordingly. There is
no additional reader call or inferred gold supporting evidence.
