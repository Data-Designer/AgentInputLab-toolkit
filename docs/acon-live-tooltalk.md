# ACON history in a native ToolTalk episode

`ToolTalkProvider` accepts an optional caller-owned history transformer. Its
default path keeps the existing ToolTalk messages, tools, turn loop and scorer.
For ACON, construct one `ACONLiveHistory` object per episode and pass it when
constructing the provider:

```python
from agentinputlab.integrations.live_history import ACONLiveHistory
from workers.official_arenas.providers.tooltalk import ToolTalkProvider

history = ACONLiveHistory(
    method=history_method,       # curate.acon.history_runtime
    context=optimizer_context,  # pinned ACON worker and actual model identities
    per_call_output_tokens=512,
)
provider = ToolTalkProvider(history_transform=history)
# Use the normal source-verified capture, executable arm payload and run_arm().
```

The callback receives the actual committed native conversation and remaining
run budget at each user-turn boundary. ToolTalk restores its official history
at that boundary: predicted tool actions from the preceding turn are temporary
rollouts, not committed history. Within a turn, the provider reuses the current
compressed view and appends the exact temporary tool frames without changing
the transformer's committed history. Retained call/result pairs stay intact.
The initial turn can legitimately take the native skip branch. Do not
concatenate unrelated tasks or fabricate user follow-ups to force compression.

Dynamic history calls, both token directions and costs are added to the same
episode usage as downstream calls. An initial intervention's costs are debited
once before the episode; a live callback's costs are debited as they occur. The
callback must reserve the downstream call/output allowance and must not perform
tool execution or retrieval. The target's native seed sequence and per-call
output limit remain unchanged. Invalid usage and orphaned tool frames fail
before the next target call.

`live-history.json` preserves the callback fingerprint, decisions and measured
usage, including when native execution fails. A successful trajectory binds
this receipt by SHA-256. These files are create-only: a consumed or partial run
must not be silently retried in its old output directory.

ToolTalk remains evaluation-only. This runtime hook neither fits guidelines nor
provides benchmark feedback to a training recipe. Compatibility tests and a
valid artifact hash are not an independent benchmark-quality or full-profile
certification.
