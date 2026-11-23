# InfoCQR T5 student

`rewrite.infocqr.t5_student_external` is a trained T5 query rewriter. It is
separate from InfoCQR's teacher prompts, author rewrite caches, and the four
downstream backbone models. A fixed auxiliary student can feed different
downstream readers without being relabeled as those readers.

Install `agentinputlab[infocqr-student]`. Mount the upstream InfoCQR source at
revision `2953c461aa0c6817b1b7656dbd5168050227b31d` and a trained T5 student
checkpoint with its trusted provenance manifest. Nothing in this adapter
downloads or substitutes generic t5-base weights. The pinned source contains
training/test scripts, not the trained student checkpoint itself.

```python
from agentinputlab.api import create_intervention
from agentinputlab.contracts import InterventionContext
from agentinputlab.integrations.external_workers.infocqr_student import InfoCQRStudentRuntime
from agentinputlab.runner import InterventionRunner

runtime = InfoCQRStudentRuntime(
    checkout="upstream/InfoCQR",
    checkpoint="private/trained-student",
    manifest="private/student-manifest.json",
    expected_manifest_sha256=trusted_receipt["student_manifest_sha256"],
    source_license_acknowledged=True,
    device="cpu",  # Or the operator's explicitly assigned cuda:N device.
)
student = create_intervention(
    "rewrite.infocqr.t5_student_external",
    checkpoint_manifest_sha256=trusted_receipt["student_manifest_sha256"],
    dataset_revision=qrecc_revision,
    profile_kind="controlled_adapter",
    source_license_acknowledged=True,
)
record = InterventionRunner().run(
    student, qrecc_visible_state,
    InterventionContext(external_runtime=runtime, seed=17),
)
rewritten_state = record.patch.apply(qrecc_visible_state)
```

The state must expose QReCC's `conversation_source`, `conversation_no`,
`turn_no` and complete Q/A `new_context` in metadata. It does not expose gold
rewrites or evaluator labels. The native notebook function constructs the
`<Que>`/`<Ans>` input; the native T5 collator left-truncates the source at 384
tokens. An empty placeholder is used only for the collator's unused target.
Both labels and decoder input IDs are removed before inference, as in the
source test script. Generation uses `max_length=64`, including the decoder
start token, with the checkpoint's generation defaults. At most 63 new decoder
tokens count toward the run budget. No teacher/cache fallback or implicit retry
occurs. An empty rewrite remains a costed failure.

Weight loading and input tokenization run as a **zero-generation preflight**.
A missing library, invalid checkpoint or input-budget rejection before generation
does not become a consumed scientific attempt. The subsequent generation has one
reserved model call. `runtime.last_attempt` retains known input usage and marks
output usage unknown if generation raises; persist it alongside failed run
records rather than interpreting unknown tokens or local inference as free.

The modern optional runtime is a declared dependency binding, not the original
Torch 1.12.1 / Transformers 4.25.1 environment. Source serialization, collation
and generation logic are retained; historical environment and paper-result
parity are not claimed. The old `workers.infocqr.driver` teacher/cache protocol
is unchanged; it cannot silently route a teacher request into this student.

## Trusted checkpoint manifest

The manifest is outside the checkpoint directory and checksum-locked by a
separately trusted receipt. Required fields are:

```json
{
  "schema_version": "agentinputlab.infocqr.student-artifact.v1",
  "upstream_revision": "2953c461aa0c6817b1b7656dbd5168050227b31d",
  "dataset_id": "qrecc",
  "dataset_revision": "the-exact-training-data-revision",
  "provenance_kind": "source_recipe_finetuned",
  "training_receipt_sha256": "SHA256_OF_THE_TRUSTED_TRAINING_RECEIPT",
  "training_source_sha256": "d40bc17e365d9f626fd5bf1fd14fffd3073c0b6772434a78eac6412d6fd2f4de",
  "files": {
    "config.json": "SHA256_OF_CONFIG",
    "model.safetensors": "SHA256_OF_TRAINED_WEIGHTS",
    "spiece.model": "SHA256_OF_TOKENIZER"
  }
}
```

List **every** checkpoint/tokenizer/generation file, including shard indexes
and shards. Closed-tree checks reject missing/unlisted files, links, changed
bytes and pickle weights. Use a separate tensor-equality-verified conversion
when a trusted checkpoint is only available as `.bin`; it is never deserialized
implicitly. Native `<Que>`/`<Ans>` tokens are added to the tokenizer, but the
trained embedding table is never resized or filled with new random vectors.

`source_recipe_finetuned` requires the controlled profile. `author_student`
allows an explicitly declared author-checkpoint profile, but checksums and a
caller-provided receipt are not an authorship certificate or a proof of training
quality. Those require independent acceptance evidence. Generic t5-base is not
a trained student, and no fabricated training receipt belongs in real runs.

Python factory, registry builder and operator-configured `ToolkitService` can
use the profile. The runtime belongs in `InterventionContext.external_runtime`,
not on the intervention. Shared library instances serialize local T5 operations
to keep tokenizer truncation state and Torch RNG state isolated. Benchmark
retrieval and official scoring remain separate downstream operations.
