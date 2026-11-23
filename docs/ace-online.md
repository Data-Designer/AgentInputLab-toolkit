# Native ACE online adaptation

`recipe.ace.online_transductive` executes the unchanged pinned
`ace.ace.ACE.run(mode="online")` at
`ace-agent/ace@bcb7cea0504afad6f55fec4845dd4864c9f9eee7`.
Use a dedicated process with `agentinputlab[ace]` and the verified upstream
checkout. No native source is bundled or rewritten.

```python
from agentinputlab.api import create_intervention

recipe = create_intervention("recipe.ace.online_transductive")
result = recipe.run_stream(
    ordered_feedback_examples,
    checkout="/mounted/ace",
    llm_config=explicit_three_role_deployment,
    output_root="/new/online-run",
    allow_test_feedback=True,
    source_and_asset_acknowledged=True,
)
```

The deployment mapping is the same explicit target/reflector/curator structure
as the [ACE worker](../workers/ace/README.md). This profile requires a startup-
bound OpenAI-compatible endpoint, matching configured per-call cap and
declared fingerprints/pricing. API keys stay in the configured environment
variable; the result stores its name, not its value. SDK retries are disabled.
Provider requests receive the declared seed (17 by default). Native component
retries, if any, consume the same finite budget; the toolkit never reruns the
whole stream after a failure.

The defaults are 512 output tokens per call and a shared 4096 output tokens /
8 calls per case. They are defaults, not library-wide ceilings. Set
`ACEOnlineControls(per_call_output_tokens=1024, per_case_output_tokens=4096,
per_case_model_calls=8)` and the deployment's `max_tokens_per_call=1024` together
to use a different explicit allocation. The per-call cap must fit the shared
case budget; a new call is rejected before dispatch if its maximum allocation
does not fit the remaining allowance. Provider context/output limits still
apply. Changing these controls defines a different run; existing acceptance
records retain their original 512 / 4096 / 8 configuration and are not
relabeled as tests of another budget.

Examples contain stable IDs, visible `input` question/context and explicit
`expected_output` feedback. They are a **transductive stream**, not an offline
training/validation split. Feedback can influence later windows. Never import
this artifact or its post-update accuracy as independent held-out evidence.
The result type is deliberately different from an offline ACE playbook artifact.

Task-specific processors can be supplied through `data_processor=`. The shipped
[STATE-Bench feedback adapter](ace-native-statebench.md) uses native tools and
the original judge on official training captures, rather than generic text
exact match. Its explicit online mode preserves the original split labels.
The [ToolSandbox feedback adapter](ace-native-toolsandbox.md) uses native
Agent-role tools and state-based scoring on explicitly caller-declared fitting
groups; it does not claim an official training split.
The [Tau feedback adapter](ace-native-tau.md) reuses initial-user captures and
native tools/reward scoring, distinguishing Retail's official training split
from Banking's caller-defined groups.

The full native online path is retained: initial evaluation of the entire
stream, evaluation of each window before its updates, native Generator /
Reflector / Curator adaptation, and native post-update generations. Window
size defaults to one and the source order is preserved. Initial, window and
adaptation calls all share each case's default **8 calls / 4096 output tokens**;
the budget does not reset between phases. All actual input tokens are recorded
separately. No extra allowance is granted for the initial evaluation. A stream
may stop under the cap; records distinguish consumed partial cases from
never-started or unreached primary evaluations.

Repeated question text with distinct source contexts (for example, the same
summary instruction for different documents) is supported. Native inputs stay
unchanged; sample observers bind usage to the exact question/context pair and
stable case ID. Duplicate full inputs remain rejected to avoid ambiguous cost
attribution.

Only the window prediction made before that window's feedback is a primary
online prediction. Initial and post-update answers remain separate. Native
`"No final answer found"` placeholders and swallowed evaluation errors are not
promoted to valid predictions. The original denominator is retained even when
the native summary excludes errors. No answer is filled, repaired or retried.

The receipt contains a hash-chained event journal, actual provider requests and
responses, per-case and total usage, each window's pre-update playbook hash,
native update tracking, primary outcomes and all native output-file hashes.
Observer wrappers forward original arguments and return values unchanged;
they do not replace the online loop, parser or curator. Pre-model source/import
errors are marked as execution prerequisites, not scientific failures.
Local zero pricing is flagged as unpriced, not zero compute cost. GPU work and
provider-side cost still require external resource metering.
