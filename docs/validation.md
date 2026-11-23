# Validation policy

The compatibility report separates dimensions that must not be conflated:

- Registry resolution: every declared entrypoint is imported and its signature inspected.
- Structural eligibility: method family versus benchmark-declared family, including checkpoint-conditional gates. This is a necessary condition, not full method compatibility.
- Offline conformance: synthetic unit/integration cases exercise contracts and failure handling.
- Installed-package acceptance: build, install into a fresh environment outside the checkout, load package data, run CLI and Python API, install the npm tarball and call the Python service.
- Real acceptance: actual model × benchmark × method, with native scorer and identity/usage/terminal evidence.

The first four do not establish the fifth. The generated cross-product inventory defaults real acceptance to `not_run`. It must not promote paper results from a different release/profile, asset hash or budget. Real failure remains failure; unsupported and blocked cells are not successes. No historical experimental outcomes are shipped in this branch.

A declared external algorithm/output failure can include valid actual usage.
Monitoring settles that usage before raising the typed failure, including any
irreversible budget overage. It does not replace known token counts with a
minimum one-call reservation. Missing or malformed telemetry remains explicitly
unknown; only an explicit unconsumed rejection cancels the reservation.

Before running a large matrix, freeze exact dataset splits, model revisions, method profiles, dependencies, budgets, limits and a finite task manifest. Start with one real case per eligible supported cell; classify setup failures separately from scientific outcomes. Report native metrics even when zero. Larger runs require explicit resource/attempt authorization. A method × dataset × backbone cross-product includes structural ineligibilities and must not be represented as all tested by testing only the easiest cells.

Research-only orchestration tests that import private launch scripts stay with the research branch. Public tests cover reusable library, adapters, workers, package boundary and SDK. Removing those private fixtures is distribution isolation, not a claim that their workflows were revalidated.

## Configure a compatible execution profile

Budgets belong to each configured run. Set cumulative calls and tokens on the
state, and per-stage caps on the selected method or worker configuration. Every
stage consumes the shared budget. Defaults are convenient starting points, not
permission for hidden retries or a larger effective attempt budget.

| Profile | Required setup and scope |
| --- | --- |
| RRR controlled prompt | Configured backbone, retriever and pinned prompts; start with `rewrite.rrr.common_support_prompt`. Warm-up/PPO checkpoint profiles remain separate optional integrations. |
| Native HyDE with a downstream reader | Eight hypothesis generations plus one reader require at least nine calls. An eight-call run retains that protocol conflict; it does not silently drop a hypothesis or add a call. |
| ACON fitted compression | Supply a complete, validated fitted artifact. A truncated or failed fit is retained as a failure and cannot initialize compression. |
| InfoCQR / Plan-and-Budget cached profiles | Supply a generated cache matching the declared input, model and artifact identities. Cache replay preserves missing-entry errors instead of drawing replacements. |
| IRCoT prediction replay | Use native answer-bearing tasks. Tool-task retrieval enhancement uses its separate profile and task-appropriate native scorer. |
| ACE frozen transfer | Fit on declared, disjoint development data, then apply the same-backbone frozen playbook to heldout inputs. Keep transfer inference distinct from fitting on the target benchmark. |

The release validation protocol keeps ToolTalk globally test-only. Frozen
BFCL-to-ToolTalk ACE transfer uses no ToolTalk fitting or online feedback.
Offline prompt/playbook optimization is distinct from additional SFT or RL;
recipes requiring those training procedures have separate prerequisites.

Interpret each compatibility record by its exact profile, backbone, benchmark,
source revision and budget. Real model failures retain their terminal state,
native metrics when available and actual costs. Restricted, deferred,
inapplicable and untested configurations remain distinct from verified execution.
