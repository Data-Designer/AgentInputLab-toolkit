# Train a LLMLingua-2 token classifier

Install `agentinputlab[llmlingua2-fit]` in a dedicated process/environment. Mount
the clean Microsoft LLMLingua checkout at the registry's pinned commit, a safe
local two-label BERT/XLM-R checkpoint, and the English `en_core_web_sm` model.
Source, data and model assets are external; none ship in the distribution.

`label_llmlingua_compressed_records` executes the official word alignment and,
optionally, its percentile filters on JSON records containing paired
`prompt_list` and `compressed_prompt_list`. Supply real, provenance-bound teacher
outputs (for example, a bounded sample of the author's published MeetingBank
compression corpus). This function never calls or pays a teacher, and never
manufactures labels from benchmark answers. It retains native alignment, lemma
tokenization and filtering, including upstream edge cases.

```python
from agentinputlab.api import create_intervention

recipe = create_intervention(
    "training.llmlingua2.distillation",
    controls={"epochs": 1, "batch_size": 4, "max_optimizer_steps": 1},
)
# artifact = recipe.fit(
#     checkout=..., initial_checkpoint=..., initial_checkpoint_manifest=...,
#     initial_manifest_sha256=..., training_file=..., data_provenance=...,
#     data_provenance_sha256=..., nlp_root=..., nlp_manifest=...,
#     nlp_manifest_sha256=..., output_root=...,
#     source_and_asset_acknowledged=True,
# )
```

The training JSON contains native `origin` and Boolean word `labels` arrays.
The provenance JSON binds its SHA-256 as `training_sha256` and declares a
`teacher_revision`. Input chunks must be unique; the native seeded random
80/20 split is retained. This is a chunk-level split, not a promise of
meeting/document-level independence. Initial checkpoint and spaCy model
manifests enumerate every file with SHA-256, under schemas
`agentinputlab.llmlingua2.token_checkpoint.v1` and
`agentinputlab.llmlingua2.spacy_model.v1`. Initial checkpoint paths must contain
the native `bert-base-multilingual-cased` or `xlm-roberta-large` substring.

The original source owns the token loss, padding labels, Adam, clipping order,
training and validation. Checked local loaders replace only file acquisition;
observer wrappers forward native arguments/results. Outputs are create-only.
The result records actual steps, changed tensors, losses, validation accuracy,
runtime versions and time/memory costs. A safe final-candidate checkpoint is
distinct from the author's best-accuracy checkpoint; a nonpositive native best
score does not become a fictitious successful selection. Preserve real failures
and consumed steps; do not rerun a failed training attempt to obtain a pass.

`LLMLingua2TrainedArtifact.load` verifies the closed artifact before deployment.
Deploy it through the typed trained-candidate path:

```python
from agentinputlab.recipes import LLMLingua2TrainedArtifact

artifact = LLMLingua2TrainedArtifact.load("/mounted/native-fit-output")
method = artifact.intervention(config={"rate": 0.5}, device_map="cpu")
# InterventionRunner().run(method, visible_state, InterventionContext(seed=17))
```

This uses the native compression dispatcher, with exact fitted-weight/tokenizer
hashes, a `native-fit-<artifact_id>` revision and `controlled_adapter` profile.
The pipeline retains `training.llmlingua2.distillation` as its identity; it is
not the released official BERT or XLM-R treatment. File changes are rejected;
reusing an immutable model avoids repeated full checkpoint reads. Helper CPU
and wall time are recorded separately from generative calls. LLMLingua 0.2.2
uses a model-name substring to route word-boundary handling. After loading the
verified fitted checkpoint, this bridge sets that native dispatch field from
the actual model configuration (BERT or XLM-R); it does not reload or replace
weights, change token probabilities, or change compression controls. The public
artifact identity remains the fitted path and its content-bound revision.
Warm-starting from the
published MeetingBank model is not from-scratch distillation or paper-result
parity. Training costs are separate from downstream model budgets; data terms
and teacher provenance must be retained in derived artifacts.
