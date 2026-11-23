"""Bind ACON to the QA reader's actual message boundary, not an empty state.

A static QA case has one real user request. Its native initial threshold skip
is a valid execution outcome, not evidence of multi-turn compression quality.
"""

from dataclasses import replace

from agentinputlab.contracts import Message, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability.models import ResourceUsage


class QAHistoryBackend:
    def __init__(self, inner, hook, *, state, ledger):
        self.inner, self.hook = inner, hook
        self.state, self.ledger = state, ledger

    @property
    def model_id(self):
        return self.inner.model_id

    @property
    def capabilities(self):
        return self.inner.capabilities

    def fingerprint(self):
        return self.inner.fingerprint()

    def generate(self, request):
        # Production QA uses this exact system/prompt serialization. Do not
        # invent a system message, append a second user prompt or touch gold.
        if request.messages or not request.system or not request.prompt or request.tools:
            raise ValidationError("static ACON requires the QA reader's system/prompt text request")
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.prompt},
        ]
        usage = self.ledger.usage
        remaining = {
            "max_model_calls": self.state.budget.max_model_calls - usage.model_calls,
            "max_output_tokens": self.state.budget.max_output_tokens - usage.output_tokens,
        }
        before = len(self.hook.records)
        try:
            value = self.hook.transform(
                messages, state=to_jsonable(self.state), remaining_budget=remaining
            )
        except Exception:
            # Keep irreversible optimizer usage if its worker failed; no reader
            # fallback and no retry. The original failure receipt stays visible.
            for record in self.hook.records[before:]:
                for snapshot in record.get("monitor_snapshots", []):
                    self.ledger.charge_actual(ResourceUsage(**snapshot["usage"]))
            raise
        self.ledger.charge_actual(ResourceUsage(**value["usage"]))
        self.ledger.require_within_budget()
        if value["messages"] == messages:
            # Preserve typed request and wire payload exactly on native skip.
            return self.inner.generate(request)
        transformed = replace(
            request, prompt="", system="", messages=tuple(Message(**m) for m in value["messages"])
        )
        return self.inner.generate(transformed)
