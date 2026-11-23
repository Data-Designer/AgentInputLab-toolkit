# ACON history in a native τ³ Banking episode

The optional history interface leaves the pinned simulator, tools, task state,
agent generation controls and scorer in charge. Configure one transformer per
episode; do not share its mutable history across tasks.

~~~python
from agentinputlab.integrations.live_history import ACONLiveHistory
from workers.official_arenas.providers.tau3 import Tau3Provider

history = ACONLiveHistory(
    history_method, optimizer_context,
    per_call_output_tokens=512,
    prefix_policy="preserve_pre_user_assistant",
)
provider = Tau3Provider(history_transform=history)
# Use the usual captured checkpoint, executable payload and run_arm().
~~~

The Retail provider accepts the same explicit constructor option. Use a new
transformer for each Retail episode. Its native no-retry capture, separate
backbone-bound user/judge controls and exact assertion-coverage guard remain
unchanged; unresolved judge output does not become a successful or zero-scored
agent episode.

The prefix policy is explicit and non-default. Native τ³ starts with a system
message, a fixed assistant greeting, and the first user request. The greeting
is immutable: it is excluded from the mutable optimizer view and reinserted
at the same downstream position with all original fields. The upstream ACON
worker still sees its system/first-user framing. No greeting, user request,
tool argument, result or simulator state is fabricated to make a history
input pass. Tool-call/result pairs retained by ACON keep their original native
objects and wire representation. Unsupported prefix frames fail before a
downstream model call.

This is a declared native-prefix adapter, not a change to the strict history
profile or a claim of native prompting parity. The default strict prefix
policy, default provider path and ordinary ToolTalk interface are unchanged.
The initial native user turn may skip compression; that branch alone is not
evidence of a compression effect.

Every live optimizer call, both token directions and measured cost count in the
same run budget as evaluated-agent calls. The adapter reserves downstream
capacity, never performs tools or retrieval, and records helper and target
usage separately in create-only live-history.json. Simulator/judge costs
retain their existing native accounting and are not hidden in optimizer usage.
Native environment history and scorer inputs are never replaced by the
compressed model-visible view. Real malformed outputs, budget stops and
failure records remain outcomes; they are not retried to obtain a better score.

The adapter and its fixtures are not real-model acceptance by themselves.
Deployment acceptance also needs source-bound saved-response qualification
and any genuinely unspent model calls, with original failure denominators and
all attempts retained.

## STATE-Bench Responses transport

`StateBenchProvider(history_transform=history)` is also opt-in. Use the strict
prefix policy for this provider. The session exposes real top-level instructions
and native injected policy Items as one immutable optimizer policy view. It
groups each complete assistant/function/result episode into an atomic text view
between actual user turns. This is an optimizer view, not a Chat Completions
conversion: retained Responses Items, call IDs, reasoning, annotations and tool
outputs keep their original native objects and ordering on the downstream wire.
Only the first user text can receive ACON's summary; the current user request
must remain. Native system/developer Items and instructions cannot be rewritten.

The hook runs at new user turns, not inside `previous_response_id` tool loops.
Existing stateless replay and model-specific transport normalization still apply.
Helper and target usage share the original caps, including all tool follow-ups;
`live-history.json` records both usage and retained source Item indices even if
the native episode fails. Default provider calls do not enable this adapter.
Unsupported non-text user content or incomplete function episodes fail before
the optimizer is called. The official environment, simulator, judge and scorer
are unchanged. An initial compression skip is not evidence of compression gain.
