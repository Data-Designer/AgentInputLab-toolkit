"""One coordinator-owned, fail-closed cost ledger for native GRPO workers.

Workers must send reservations and settlements to this SAME instance (for
example through a serial Ray actor). Instantiating a ledger per rank is not
aggregate enforcement. No model, optimizer or Ray cluster is started here.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError

from .smartsearch_grpo_tools import _new_output, _save
from .smartsearch_teacher import SmartSearchTeacherCompletion


@dataclass(frozen=True, slots=True)
class SmartSearchGRPOCostLimits:
    policy_requests: int
    policy_output_tokens: int
    helper_requests: int
    helper_output_tokens: int
    actor_ranks: int
    optimizer_steps_per_rank: int

    def __post_init__(self):
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in self.__dataclass_fields__) or self.actor_ranks > 4:
            raise ValidationError("SmartSearch GRPO needs positive aggregate cost limits and one to four actor ranks")


class SmartSearchGRPOCostLedger:
    """Reserve before neural work; retain unknown usage and every rank's costs.

    Input token counts are logical prompt tokens, not measured physical model
    forwards. Returned output counts include ALL generated samples, before the
    native selector drops any. Optimizer counters are per rank; synchronized
    updates are never computed by summing ranks.
    """

    def __init__(self, *, output_root, limits: SmartSearchGRPOCostLimits):
        if not isinstance(limits, SmartSearchGRPOCostLimits):
            raise ValidationError("SmartSearch GRPO requires typed aggregate cost limits")
        self.root = _new_output(output_root)
        self.limits, self._lock = limits, threading.RLock()
        self._failed, self._number = False, 0
        self._pending = {}
        self._invalid_usage_reports = []
        self._costs = {kind: dict(requests=0, responses=0, logical_input_tokens=0,
                                 output_tokens=0, reserved_output_tokens=0, generated_samples=0)
                       for kind in ("policy", "helper")}
        self._steps = {str(rank): dict(attempts=0, responses=0) for rank in range(limits.actor_ranks)}
        _save(self.root, "claim.json", dict(limits=limits, coordinator_owned=True,
            all_generated_samples_counted=True, physical_forwards_not_claimed=True,
            synchronized_updates_not_sum_of_rank_steps=True, no_automatic_retry=True))

    def _healthy(self):
        if self._failed:
            raise ExternalDependencyError("SmartSearch GRPO run has a retained failure; no automatic neural retry")

    def _event(self, name, value):
        self._number += 1
        try:
            _save(self.root, f"event-{self._number:08d}-{name}.json", value)
        except BaseException:
            self._failed = True
            raise

    def _reject(self, message, *, budget=False):
        self._failed = True
        self._event("rejected", dict(error=message, accept_checkpoint=False, no_automatic_retry=True))
        error = BudgetExceededError if budget else ExternalDependencyError
        raise error(message)

    def reserve(self, *, kind, worker, input_token_counts, samples, max_new_tokens):
        """Return an opaque ticket, shared across all worker/rank clients."""
        with self._lock:
            self._healthy()
            if not isinstance(kind, str) or kind not in self._costs or not isinstance(worker, str) or not worker.strip() or (
                len(worker) > 128 or not isinstance(input_token_counts, (tuple, list))
                or not input_token_counts or len(input_token_counts) > 10000
                or any(type(n) is not int or n < 0 for n in input_token_counts)
                or type(samples) is not int or not 1 <= samples <= 64
                or type(max_new_tokens) is not int or max_new_tokens < 1
                or max_new_tokens > (2048 if kind == "helper" else 8192)
                or (kind == "helper" and (samples != 1 or len(input_token_counts) != 1))
            ):
                raise ValidationError("SmartSearch GRPO generation reservation is invalid")
            number = len(input_token_counts) * samples
            reservation = number * max_new_tokens
            costs = self._costs[kind]
            if costs["requests"] >= getattr(self.limits, kind + "_requests") or (
                costs["output_tokens"] + costs["reserved_output_tokens"] + reservation
                > getattr(self.limits, kind + "_output_tokens")
            ):
                self._reject("SmartSearch GRPO aggregate generation budget exhausted", budget=True)
            ticket = f"generation-{self._number + 1:08d}"
            item = dict(ticket=ticket, kind=kind, worker=worker, input_token_counts=list(input_token_counts),
                        samples=samples, output_samples=number, max_new_tokens=max_new_tokens,
                        reserved_output_tokens=reservation)
            self._event("generation-reserved", item)
            self._pending[ticket] = item
            costs["requests"] += 1
            costs["logical_input_tokens"] += sum(input_token_counts)
            costs["reserved_output_tokens"] += reservation
            return ticket

    def settle(self, ticket, *, output_token_counts):
        with self._lock:
            # Another rank may have failed while this paid request was in
            # flight. Still retain its measured cost; never clear the latch.
            if not isinstance(ticket, str) or ticket not in self._pending:
                self._reject("SmartSearch GRPO ticket is unknown or already settled")
            item = self._pending[ticket]
            # Validation failures keep the entire reservation: usage is not
            # assumed zero when the provider omitted or malformed its report.
            if not isinstance(output_token_counts, (list, tuple)) or (
                len(output_token_counts) != item["output_samples"]
                or any(type(n) is not int or n < 0 or n > item["max_new_tokens"]
                       for n in output_token_counts)
            ):
                report = dict(ticket=ticket, kind=item["kind"],
                    reported_output_tokens=sum(n for n in output_token_counts if type(n) is int and n >= 0)
                    if isinstance(output_token_counts, (list, tuple)) and len(output_token_counts) <= 10000 else None,
                    reservation_retained=True)
                self._invalid_usage_reports.append(report)
                self._event("invalid-usage-report", report)
                self._reject("SmartSearch GRPO provider token/sample counts differ from its reservation")
            self._event("generation-settled", dict(ticket=ticket, output_token_counts=list(output_token_counts)))
            costs = self._costs[item["kind"]]
            costs["responses"] += 1
            costs["output_tokens"] += sum(output_token_counts)
            costs["generated_samples"] += len(output_token_counts)
            costs["reserved_output_tokens"] -= item["reserved_output_tokens"]
            del self._pending[ticket]

    def fail(self, ticket, *, error_type):
        """Retain the pre-call reservation after a failed/unknown provider call."""
        with self._lock:
            if not isinstance(error_type, str) or not error_type or len(error_type) > 128:
                raise ValidationError("SmartSearch GRPO failure requires a bounded exception type")
            self._failed = True
            self._event("provider-failed", dict(ticket=ticket, error_type=error_type,
                ticket_pending=ticket in self._pending, accept_checkpoint=False, no_automatic_retry=True))

    def optimizer_begin(self, rank):
        with self._lock:
            self._healthy()
            if type(rank) is not int or str(rank) not in self._steps:
                raise ValidationError("SmartSearch GRPO optimizer rank is outside its admitted actor mesh")
            cost = self._steps[str(rank)]
            if cost["attempts"] != cost["responses"]:
                self._reject("SmartSearch GRPO optimizer already has an unresolved step")
            if cost["attempts"] >= self.limits.optimizer_steps_per_rank:
                self._reject("SmartSearch GRPO actor optimizer step quota exhausted", budget=True)
            self._event("optimizer-begin", dict(rank=rank, step=cost["attempts"] + 1))
            cost["attempts"] += 1

    def optimizer_end(self, rank):
        with self._lock:
            # Account for a concurrent rank's already-completed native step
            # even when a different worker has invalidated this run.
            if type(rank) is not int or str(rank) not in self._steps:
                raise ValidationError("SmartSearch GRPO optimizer rank is outside its admitted actor mesh")
            cost = self._steps[str(rank)]
            if cost["attempts"] != cost["responses"] + 1:
                self._reject("SmartSearch GRPO optimizer settlement has no pending step")
            self._event("optimizer-end", dict(rank=rank, step=cost["attempts"]))
            cost["responses"] += 1

    def snapshot(self):
        with self._lock:
            return dict(failed=self._failed, costs={kind: dict(value) for kind, value in self._costs.items()},
                        optimizer_rank_steps={rank: dict(value) for rank, value in self._steps.items()},
                        pending_generation_tickets=sorted(self._pending),
                        invalid_usage_reports=[dict(row) for row in self._invalid_usage_reports],
                        physical_model_forwards_measured=False)

    def assert_healthy(self, *, expected_optimizer_steps=None):
        with self._lock:
            self._healthy()
            if self._pending or any(n["attempts"] != n["responses"] for n in self._steps.values()):
                raise ExternalDependencyError("SmartSearch GRPO has unresolved paid work; checkpoint not accepted")
            steps = [n["responses"] for n in self._steps.values()]
            if len(set(steps)) != 1:
                raise ExternalDependencyError("SmartSearch GRPO actor rank step counts differ")
            if expected_optimizer_steps is not None and (
                type(expected_optimizer_steps) is not int or expected_optimizer_steps < 1
                or steps[0] != expected_optimizer_steps
            ):
                raise ExternalDependencyError("SmartSearch GRPO actual actor steps differ from the admitted plan")
            return {**self.snapshot(), "synchronized_optimizer_steps": steps[0]}


def observe_native_grpo_optimizer(optimizer, *, ledger, rank):
    """Observe native AdamW.step without replacing loss, gradients or schedule.

    ``ledger`` may be a synchronous RPC client for the sole coordinator. A
    failed step is unresolved and the wrapper is latched; it cannot be retried.
    """
    if not callable(getattr(optimizer, "step", None)) or any(
        not callable(getattr(ledger, name, None)) for name in ("optimizer_begin", "optimizer_end", "fail")
    ) or type(rank) is not int or rank < 0 or getattr(optimizer, "_agentinputlab_grpo_observed", False):
        raise ValidationError("SmartSearch GRPO optimizer observer needs one native optimizer/rank and shared ledger")
    original, failed = optimizer.step, False

    def step(*args, **kwargs):
        nonlocal failed
        if failed:
            raise ExternalDependencyError("SmartSearch GRPO optimizer failed; no automatic step retry")
        try:
            ledger.optimizer_begin(rank)
            result = original(*args, **kwargs)
            ledger.optimizer_end(rank)
            return result
        except BaseException as exc:
            failed = True
            ledger.fail(f"optimizer-rank-{rank}", error_type=type(exc).__name__)
            raise

    optimizer.step = step
    optimizer._agentinputlab_grpo_observed = True
    return optimizer


def meter_smartsearch_grpo_helper(generate, *, ledger, worker, fingerprint, count_input_tokens):
    """Place native prefix AND reward helper clients behind one shared quota.

    Host this callback at the sole teacher coordinator. Keep the provider's
    original text, sampling and stop behavior. This does not construct a model
    or replace either native parser. A distributed executor supplies an RPC
    client to this callback, not a new independent quota per rank.
    """
    if not callable(generate) or not callable(count_input_tokens) or any(
        not callable(getattr(ledger, name, None)) for name in ("reserve", "settle", "fail")
    ) or not isinstance(worker, str) or not worker.strip():
        raise ValidationError("SmartSearch GRPO helper requires a measured provider and sole shared ledger")
    SmartSearchTeacherCompletion("", 0, 0, fingerprint)
    lock, failed = threading.Lock(), False

    def complete(prompt, *, temperature, max_new_tokens, stop, seed):
        nonlocal failed
        with lock:
            if failed:
                raise ExternalDependencyError("SmartSearch GRPO helper failed; no automatic provider retry")
            if not isinstance(prompt, str) or temperature != .6 or type(max_new_tokens) is not int or max_new_tokens != 2048 or (
                type(seed) is not int or seed < 0 or not isinstance(stop, (tuple, list))
                or tuple(stop) not in ((), ("</answer>",))
            ):
                raise ValidationError("SmartSearch GRPO helper request differs from its native contract")
            tokens = count_input_tokens(prompt)
            ticket = ledger.reserve(kind="helper", worker=worker, input_token_counts=[tokens],
                                    samples=1, max_new_tokens=max_new_tokens)
            try:
                result = generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens, stop=stop, seed=seed)
                if not isinstance(result, SmartSearchTeacherCompletion):
                    raise ExternalDependencyError("SmartSearch GRPO helper omitted typed usage")
                ledger.settle(ticket, output_token_counts=[result.output_tokens])
                if result.input_tokens != tokens or result.model_fingerprint_sha256 != fingerprint:
                    raise ExternalDependencyError("SmartSearch GRPO helper identity/input accounting differs")
                return result
            except BaseException as exc:
                failed = True
                ledger.fail(ticket, error_type=type(exc).__name__)
                raise

    return complete


class MeteredNativeGRPOEngine:
    """Transparent observer of the pinned rollout's vLLM.generate seam.

    Inputs and returned RequestOutput objects are passed through unchanged.
    All engine outputs are settled before native prefix/group selection, so
    later truncation cannot hide their cost. No token/stop/n/temperature value
    is rewritten. Use only with the pinned native rollout and its TP=1 config.
    """

    def __init__(self, engine, *, ledger, worker):
        if not callable(getattr(engine, "generate", None)) or any(
            not callable(getattr(ledger, name, None)) for name in ("reserve", "settle", "fail")
        ) or not isinstance(worker, str) or not worker.strip():
            raise ValidationError("SmartSearch GRPO engine observer requires native generation and a shared ledger")
        self._engine, self._ledger, self._worker = engine, ledger, worker
        self._failed = False

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def generate(self, *args, **kwargs):
        if self._failed:
            raise ExternalDependencyError("SmartSearch GRPO engine failed; no automatic generation retry")
        prompts, params = kwargs.get("prompts"), kwargs.get("sampling_params")
        if args or not isinstance(prompts, list) or not prompts or any(
            not isinstance(p, dict) or set(p) != {"prompt_token_ids"}
            or not isinstance(p["prompt_token_ids"], list) or not p["prompt_token_ids"]
            for p in prompts
        ):
            raise ValidationError("SmartSearch GRPO engine seam differs from the pinned token-only native call")
        samples, cap = getattr(params, "n", None), getattr(params, "max_tokens", None)
        ticket = self._ledger.reserve(kind="policy", worker=self._worker,
            input_token_counts=[len(p["prompt_token_ids"]) for p in prompts],
            samples=samples, max_new_tokens=cap)
        try:
            result = self._engine.generate(*args, **kwargs)
            if not isinstance(result, list) or len(result) != len(prompts):
                raise ExternalDependencyError("SmartSearch native engine returned a different prompt count")
            counts = []
            for row in result:
                outputs = getattr(row, "outputs", None)
                if not isinstance(outputs, (tuple, list)) or len(outputs) != samples:
                    raise ExternalDependencyError("SmartSearch native engine returned a different sample count")
                for output in outputs:
                    ids = getattr(output, "token_ids", None)
                    if not isinstance(ids, (tuple, list)):
                        raise ExternalDependencyError("SmartSearch native engine omitted measured output token IDs")
                    counts.append(len(ids))
            self._ledger.settle(ticket, output_token_counts=counts)
            return result
        except BaseException as exc:
            self._failed = True
            self._ledger.fail(ticket, error_type=type(exc).__name__)
            raise
