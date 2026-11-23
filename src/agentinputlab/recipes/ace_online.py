"""Explicitly transductive, source-pinned ACE online execution.

The unchanged native online loop evaluates a window before updating on its
feedback. Initial evaluation, adaptation and post-update generations all count.
This is not an offline or held-out evaluation protocol.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .ace import ACEExample

RELEASE = "recipe.ace.online_transductive"
PROTOCOL = "agentinputlab.ace.native-online.v1"
_LOCK = threading.RLock()


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ACEOnlineControls:
    max_rounds: int = 1
    per_case_model_calls: int = 8
    per_case_output_tokens: int = 4096
    per_call_output_tokens: int = 512
    window_size: int = 1
    max_stream_examples: int = 128
    playbook_token_budget: int = 4096
    timeout_seconds: int = 3600
    seed: int = 17

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValidationError(f"ACE online {name} must be an integer in range")
        if self.per_call_output_tokens > self.per_case_output_tokens:
            raise ValidationError("ACE online per-call cap must fit its shared case budget")


def audit_online_stream(examples, controls):
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise ValidationError("ACE online needs an ordered example sequence")
    rows = tuple(ACEExample.from_value(row, split="test") for row in examples)
    if not rows or len(rows) > controls.max_stream_examples or controls.window_size > len(rows):
        raise ValidationError("ACE online stream size or window exceeds its explicit bound")
    if any(row.expected_output is None for row in rows):
        raise ValidationError("ACE online requires explicitly acknowledged feedback labels")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValidationError("ACE online stream IDs must be unique")
    from workers.ace.driver import adapt_example

    adapted = tuple(adapt_example(row.as_dict()) for row in rows)
    # Summarization/code tasks can repeat a question for different documents.
    # Native sample callbacks expose both original fields; neither field needs
    # rewriting just to attribute usage to its stable source example.
    if len({(row["question"], row["context"]) for row in adapted}) != len(rows):
        raise ValidationError(
            "ACE online callback attribution requires unique question/context inputs"
        )
    return rows, adapted


class _Journal:
    def __init__(self, root):
        self.stream = (root / "events.jsonl").open("x", encoding="utf-8")
        self.previous = "0" * 64
        self.count = 0
        self.lock = threading.RLock()

    def event(self, kind, **data):
        with self.lock:
            row = dict(index=self.count, previous_sha256=self.previous, kind=kind, **data)
            digest = content_hash(row)
            self.stream.write(
                json.dumps({**row, "sha256": digest}, sort_keys=True, allow_nan=False) + "\n"
            )
            self.stream.flush()
            self.previous, self.count = digest, self.count + 1

    def close(self):
        self.stream.close()


def _tracker_type():
    from workers.ace.driver import UsageTracker, WorkerFailure

    class OnlineTracker(UsageTracker):
        def __init__(self, controls, config, online, ids, journal):
            super().__init__(controls, config)
            self.online, self.journal = online, journal
            self._lock, self.local = threading.RLock(), threading.local()
            self.started = time.monotonic()
            self.phase = "initial_evaluation"
            self.input_to_id = {
                (key if isinstance(key, tuple) else (key, "")): value for key, value in ids.items()
            }
            self.question_to_ids = {}
            for (question, _), case in self.input_to_id.items():
                self.question_to_ids.setdefault(question, set()).add(case)
            # Budget keys remain stable source IDs, including repeated queries.
            self.by_case = {
                key: dict(
                    model_calls=0, input_tokens=0, output_tokens=0, provider_usage_responses=0
                )
                for key in ids.values()
            }
            self.budget_rejections = []

        def select(self, question, context=None):
            candidates = self.question_to_ids.get(question, set())
            if context is not None:
                case = self.input_to_id.get((question, context))
            elif len(candidates) == 1:
                case = next(iter(candidates))
            else:
                # Component callbacks inside an already selected native sample
                # may expose only question; never guess between distinct inputs.
                case = getattr(self.local, "case_id", None)
                if case not in candidates:
                    case = None
            if case is None:
                raise WorkerFailure(
                    "EXAMPLE_ATTRIBUTION_FAILED",
                    "native ACE question/context differs from the frozen stream or is ambiguous",
                )
            self.local.case_id = case

        def case_for(self, item):
            key = (item["question"], item.get("context", ""))
            if key not in self.input_to_id:
                raise WorkerFailure(
                    "EXAMPLE_ATTRIBUTION_FAILED", "native sample differs from frozen stream"
                )
            return self.input_to_id[key]

        def _wrap(self, component, method_name, role):
            original = getattr(component, method_name)

            @functools.wraps(original)
            def observed(*args, **kwargs):
                if "question" in kwargs:
                    self.select(kwargs["question"])
                return base(*args, **kwargs)

            super()._wrap(component, method_name, role)
            base = getattr(component, method_name)
            setattr(component, method_name, observed)

        def before_provider_attempt(self, role):
            with self._lock:
                case = getattr(self.local, "case_id", None)
                if case not in self.by_case:
                    raise WorkerFailure(
                        "EXAMPLE_ATTRIBUTION_FAILED", "provider attempt has no source example"
                    )
                usage = self.by_case[case]
                if (
                    usage["model_calls"] >= self.online.per_case_model_calls
                    or usage["output_tokens"] + self.online.per_call_output_tokens
                    > self.online.per_case_output_tokens
                    or time.monotonic() - self.started >= self.online.timeout_seconds
                ):
                    self.budget_rejections.append(dict(case_id=case, role=role, phase=self.phase))
                    raise WorkerFailure(
                        "ONLINE_BUDGET_EXHAUSTED",
                        "ACE online shared case budget exhausted before dispatch",
                    )
                super().before_provider_attempt(role)
                usage["model_calls"] += 1
                self.journal.event(
                    "provider_attempt",
                    case_id=case,
                    role=role,
                    phase=self.phase,
                    per_case_attempt=usage["model_calls"],
                    seed=self.online.seed,
                )

        def after_provider_response(self, role, response):
            usage = (
                response.get("usage")
                if isinstance(response, Mapping)
                else getattr(response, "usage", None)
            )
            if usage is None:
                raise WorkerFailure(
                    "UPSTREAM_USAGE_UNAVAILABLE", "online provider omitted actual usage"
                )
            getter = (
                usage.get if isinstance(usage, Mapping) else lambda key: getattr(usage, key, None)
            )
            incoming, outgoing = getter("prompt_tokens"), getter("completion_tokens")
            if (
                type(incoming) is not int
                or incoming < 0
                or type(outgoing) is not int
                or outgoing < 0
            ):
                raise WorkerFailure(
                    "UPSTREAM_USAGE_UNAVAILABLE", "online provider usage is invalid"
                )
            with self._lock:
                row = self.by_case[self.local.case_id]
                row["input_tokens"] += incoming
                row["output_tokens"] += outgoing
                row["provider_usage_responses"] += 1
                super().after_provider_response(role, response)
                if (
                    outgoing > self.online.per_call_output_tokens
                    or row["output_tokens"] > self.online.per_case_output_tokens
                ):
                    raise WorkerFailure(
                        "PROVIDER_BUDGET_BREACH", "online provider exceeded the declared output cap"
                    )

    return OnlineTracker


class _RecordingCompletions:
    def __init__(self, delegate, tracker, role, calls):
        self.delegate, self.tracker, self.role, self.calls = delegate, tracker, role, calls

    def create(self, *args, **kwargs):
        if args:
            raise ValidationError("native ACE completion must use named arguments")
        if "seed" in kwargs and kwargs["seed"] != self.tracker.online.seed:
            raise ValidationError("native ACE supplied a different provider seed")
        kwargs = dict(kwargs, seed=self.tracker.online.seed)
        row = dict(
            case_id=self.tracker.local.case_id,
            role=self.role,
            phase=self.tracker.phase,
            request=to_jsonable(kwargs),
            started=time.time(),
        )
        self.calls.append(row)
        try:
            response = self.delegate.create(**kwargs)
            row["response"] = (
                response.model_dump(mode="json")
                if callable(getattr(response, "model_dump", None))
                else to_jsonable(response)
            )
            return response
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["wall_seconds"] = time.time() - row["started"]
            self.tracker.journal.event("provider_return", call=row)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _RecordingClient:
    def __init__(self, delegate, tracker, role, calls):
        self.delegate = delegate
        self.chat = SimpleNamespace(
            completions=_RecordingCompletions(delegate.chat.completions, tracker, role, calls)
        )

    def __getattr__(self, name):
        return getattr(self.delegate, name)


@dataclass(frozen=True, slots=True)
class ACEOnlineResult:
    root: Path
    artifact_id: str
    terminal: Mapping

    @classmethod
    def load(cls, root):
        folder = Path(root).expanduser()
        if folder.is_symlink() or not folder.is_dir():
            raise ValidationError("ACE online result requires a regular run directory")
        names = (
            "plan.json",
            "terminal.json",
            "provider-calls.json",
            "events.jsonl",
            "final-playbook.txt",
            "native-files.json",
            "manifest.json",
        )
        if any((folder / name).is_symlink() or not (folder / name).is_file() for name in names):
            raise ValidationError("ACE online result closure is incomplete")
        metadata = json.loads((folder / "manifest.json").read_text())
        identity = metadata.pop("artifact_id", None)
        if identity != content_hash(metadata) or metadata.get("protocol") != PROTOCOL:
            raise ValidationError("ACE online result identity differs")
        if set(metadata.get("files", {})) != set(names) - {"manifest.json"} or any(
            _sha(folder / name) != sha for name, sha in metadata["files"].items()
        ):
            raise ValidationError("ACE online result file hashes differ")
        previous = "0" * 64
        for index, line in enumerate((folder / "events.jsonl").read_text().splitlines()):
            row = json.loads(line)
            digest = row.pop("sha256")
            if (
                row["index"] != index
                or row["previous_sha256"] != previous
                or content_hash(row) != digest
            ):
                raise ValidationError("ACE online event chain is invalid")
            previous = digest
        terminal = json.loads((folder / "terminal.json").read_text())
        if (
            terminal.get("protocol") != PROTOCOL
            or terminal.get("event_chain_final_sha256") != previous
        ):
            raise ValidationError("ACE online terminal event binding differs")
        native_files = json.loads((folder / "native-files.json").read_text())
        from pathlib import PurePosixPath

        for name, digest in native_files.items():
            path = PurePosixPath(name)
            target = folder / name
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or target.is_symlink()
                or not target.is_file()
                or not target.resolve().is_relative_to(folder.resolve())
                or _sha(target) != digest
            ):
                raise ValidationError("ACE native output closure differs")
        return cls(folder.resolve(), identity, terminal)


class ACEOnlineRecipe:
    release_id = RELEASE

    def __init__(self, controls=None):
        self.controls = (
            ACEOnlineControls(**controls)
            if isinstance(controls, Mapping)
            else controls or ACEOnlineControls()
        )
        if not isinstance(self.controls, ACEOnlineControls):
            raise ValidationError("ACE online requires typed controls")

    def run_stream(
        self,
        examples,
        *,
        checkout,
        llm_config,
        output_root,
        allow_test_feedback=False,
        source_and_asset_acknowledged=False,
        data_processor=None,
    ):
        if allow_test_feedback is not True or source_and_asset_acknowledged is not True:
            raise ValidationError(
                "ACE online requires explicit transductive feedback and source acknowledgement"
            )
        from workers.ace.driver import (
            ACEWorkerRuntime,
            BoundedHashSink,
            Controls,
            GenericTextProcessor,
            LLMConfig,
        )

        c = self.controls
        rows, adapted = audit_online_stream(examples, c)
        task_fingerprint = None
        if data_processor is not None:
            for name in (
                "answer_is_correct",
                "evaluate_accuracy",
                "fingerprint",
                "validate_online_examples",
            ):
                if not callable(getattr(data_processor, name, None)):
                    raise ValidationError(f"ACE online task processor lacks {name}()")
            value = data_processor.fingerprint()
            if not isinstance(value, Mapping) or not value:
                raise ValidationError(
                    "ACE online task processor fingerprint must be a nonempty object"
                )
            task_fingerprint = to_jsonable(value)
            data_processor.validate_online_examples(tuple(row.as_dict() for row in rows))
        if not isinstance(llm_config, Mapping):
            raise ValidationError("ACE online needs an explicit data-only LLM deployment mapping")
        config = LLMConfig.from_json(json.dumps(to_jsonable(llm_config), allow_nan=False))
        if config.max_tokens_per_call != c.per_call_output_tokens or config.base_url is None:
            raise ValidationError(
                "ACE online requires an explicit no-retry OpenAI-compatible deployment and matching cap"
            )
        root = Path(output_root).expanduser()
        if root.exists() or root.is_symlink():
            raise ValidationError(
                "ACE online output must be create-only; consumed streams are not retried"
            )
        root = root.absolute()
        root.mkdir(parents=True)
        controls = Controls(
            max_rounds=c.max_rounds,
            max_model_calls=c.per_case_model_calls * len(rows),
            max_tokens=2_000_000_000,
            max_cost_usd=1e12,
            seed=c.seed,
        )
        plan = dict(
            protocol=PROTOCOL,
            release_id=RELEASE,
            controls=asdict(c),
            stream=[row.as_dict() for row in rows],
            deployment=to_jsonable(llm_config),
            transductive_not_offline_or_heldout=True,
            order_preserved=True,
            initial_evaluation_and_all_adaptation_costs_in_shared_case_budget=True,
            generated_artifact_must_not_be_reused_as_offline_heldout_evidence=True,
            automatic_retry=False,
        )
        if task_fingerprint is not None:
            plan["native_task_processor"] = task_fingerprint
        _save(root / "plan.json", plan)
        journal = _Journal(root)
        calls, predictions, windows, updates = [], [], [], []
        tracker = runtime = ace_system = None
        started, cpu = time.monotonic(), time.process_time()
        error = None
        diagnostics = BoundedHashSink()
        try:
            with _LOCK:
                runtime = (
                    ACEWorkerRuntime(Path(checkout), config, data_processor=data_processor)
                    if data_processor is not None
                    else ACEWorkerRuntime(Path(checkout), config)
                )
                roles = config.roles
                if not os.environ.get(config.api_key_env, ""):
                    raise ValidationError(
                        "ACE online configured API-key environment variable is unset"
                    )
                bootstrap = "OPENAI_API_KEY" not in os.environ
                if bootstrap:
                    # Native construction creates clients but performs no API
                    # call; immediately bind the caller's actual deployment.
                    os.environ["OPENAI_API_KEY"] = "native-constructor-bootstrap-not-a-secret"
                with contextlib.redirect_stdout(diagnostics):
                    try:
                        ace_system = runtime.ace_class(
                            api_provider=config.api_provider,
                            generator_model=roles["target"].model,
                            reflector_model=roles["reflector"].model,
                            curator_model=roles["curator"].model,
                            max_tokens=c.per_call_output_tokens,
                            use_bulletpoint_analyzer=False,
                        )
                        runtime._bind_openai_compatible_clients(ace_system, seed=c.seed)
                    finally:
                        if bootstrap:
                            os.environ.pop("OPENAI_API_KEY", None)
                tracker = _tracker_type()(
                    controls,
                    config,
                    c,
                    {
                        (row["question"], row["context"]): item.example_id
                        for item, row in zip(rows, adapted, strict=True)
                    },
                    journal,
                )
                for role, component in (
                    ("target", ace_system.generator),
                    ("reflector", ace_system.reflector),
                    ("curator", ace_system.curator),
                ):
                    component.api_client = _RecordingClient(
                        component.api_client, tracker, role, calls
                    )
                tracker.install(ace_system)
                ace_module = sys.modules[runtime.ace_class.__module__]
                utils = sys.modules["utils"]
                native_eval, native_sample = (
                    ace_module.evaluate_test_set,
                    utils.evaluate_single_test_sample,
                )
                native_train = ace_system._train_single_sample
                native_initial = ace_system._run_test
                evaluations = 0

                def observed_sample(args_tuple, data_processor):
                    if (
                        task_fingerprint is not None
                        and to_jsonable(data_processor.fingerprint()) != task_fingerprint
                    ):
                        raise ValidationError("ACE online task processor fingerprint changed")
                    item = args_tuple[1]
                    tracker.select(item["question"], item.get("context", ""))
                    result, sample_error = native_sample(args_tuple, data_processor)
                    record = dict(
                        case_id=tracker.case_for(item),
                        phase=tracker.phase,
                        native_result=to_jsonable(result),
                        native_error=sample_error,
                    )
                    predictions.append(record)
                    journal.event("native_sample_evaluated", **record)
                    return result, sample_error

                def evaluate(*args, **kwargs):
                    nonlocal evaluations
                    phase = "initial_evaluation" if evaluations == 0 else "pre_window_evaluation"
                    tracker.phase = phase
                    evaluations += 1
                    playbook = args[2]
                    window = dict(
                        phase=phase,
                        case_ids=[tracker.case_for(x) for x in args[3]],
                        before_playbook_sha256=content_hash(playbook),
                    )
                    journal.event("evaluation_window_started", **window)
                    result = native_eval(*args, **kwargs)
                    window["native_statistics"] = to_jsonable(result)
                    windows.append(window)
                    journal.event("evaluation_window_finished", **window)
                    return result

                def train(*args, **kwargs):
                    item = kwargs["task_dict"]
                    tracker.select(item["question"], item.get("context", ""))
                    tracker.phase = "feedback_adaptation"
                    record = dict(
                        case_id=tracker.local.case_id,
                        before_playbook_sha256=content_hash(ace_system.playbook),
                    )
                    journal.event("feedback_update_started", **record)
                    result = native_train(*args, **kwargs)
                    record.update(
                        after_playbook_sha256=content_hash(ace_system.playbook),
                        native_tracking=to_jsonable(result),
                    )
                    updates.append(record)
                    journal.event("feedback_update_finished", **record)
                    return result

                # Observation only: every native call receives its original
                # arguments and every native return value is passed unchanged.
                ace_module.evaluate_test_set = evaluate
                utils.evaluate_single_test_sample = observed_sample
                ace_system._train_single_sample = train
                state = random.getstate()
                try:
                    random.seed(c.seed)
                    with contextlib.redirect_stdout(diagnostics):
                        native_result = ace_system.run(
                            mode="online",
                            train_samples=None,
                            val_samples=None,
                            test_samples=list(adapted),
                            data_processor=(
                                data_processor
                                if data_processor is not None
                                else GenericTextProcessor()
                            ),
                            config=dict(
                                num_epochs=1,
                                max_num_rounds=c.max_rounds,
                                curator_frequency=1,
                                online_eval_frequency=c.window_size,
                                eval_steps=len(rows),
                                save_steps=len(rows) + 1,
                                playbook_token_budget=c.playbook_token_budget,
                                task_name="toolkit_explicit_online",
                                json_mode=config.json_mode,
                                no_ground_truth=False,
                                save_dir=str(root / "native"),
                                test_workers=1,
                                use_bulletpoint_analyzer=False,
                            ),
                        )
                finally:
                    random.setstate(state)
                    ace_module.evaluate_test_set = native_eval
                    utils.evaluate_single_test_sample = native_sample
                    ace_system._train_single_sample = native_train
                    ace_system._run_test = native_initial
                _save(root / "native-result.json", to_jsonable(native_result))
        except Exception as exc:
            error = dict(error_type=type(exc).__name__, message=str(exc))
        primary = []
        for row in rows:
            records = [
                p
                for p in predictions
                if p["case_id"] == row.example_id and p["phase"] == "pre_window_evaluation"
            ]
            if len(records) > 1:
                raise ValidationError("ACE online primary prediction was evaluated twice")
            record = records[0] if records else None
            result = record["native_result"] if record else None
            valid = (
                isinstance(result, dict)
                and result.get("success") is True
                and isinstance(result.get("final_answer"), str)
                and bool(result["final_answer"].strip())
                and result["final_answer"] != "No final answer found"
            )
            primary.append(
                dict(
                    case_id=row.example_id,
                    status=(
                        "native_prediction_returned"
                        if valid
                        else "failed_primary_prediction"
                        if record
                        else "not_reached_after_native_failure"
                        if tracker and tracker.by_case[row.example_id]["model_calls"]
                        else "never_started"
                    ),
                    native_prediction=result,
                    raw_primary_observation=record,
                    feedback_completed=any(u["case_id"] == row.example_id for u in updates),
                )
            )
        journal.event("stream_terminal", native_returned=error is None, error=error)
        journal.close()
        _save(root / "provider-calls.json", calls)
        with (root / "final-playbook.txt").open("x", encoding="utf-8") as stream:
            stream.write(ace_system.playbook if ace_system is not None else "")
        terminal = dict(
            protocol=PROTOCOL,
            release_id=RELEASE,
            status=(
                "native_online_returned"
                if error is None
                else "native_failed_terminal"
                if tracker and tracker.model_calls
                else "execution_failed_before_model"
            ),
            error=error,
            expected_positions=len(rows),
            primary_outcomes=primary,
            predictions_before_current_window_feedback=True,
            post_update_predictions_are_not_primary=True,
            upstream_results_may_exclude_failures_denominator_preserved=len(rows),
            windows=windows,
            updates=updates,
            usage=tracker.as_dict() if tracker else {},
            per_case_usage=tracker.by_case if tracker else {},
            budget_rejections=tracker.budget_rejections if tracker else [],
            event_chain_final_sha256=journal.previous,
            event_count=journal.count,
            wall_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - cpu,
            monetary_prices_unconfigured=all(
                not r.input_cost_per_million and not r.output_cost_per_million
                for r in config.roles.values()
            ),
            automatic_retry=False,
            transductive_not_offline_or_heldout=True,
            paper_results_changed=False,
            native_source_commit=runtime.policy.commit if runtime else None,
            verified_native_source_files=dict(runtime.policy.source_hashes) if runtime else {},
        )
        _save(root / "terminal.json", terminal)
        native_files = {
            str(p.relative_to(root)): _sha(p) for p in (root / "native").rglob("*") if p.is_file()
        }
        if (root / "native-result.json").exists():
            native_files["native-result.json"] = _sha(root / "native-result.json")
        _save(root / "native-files.json", native_files)
        files = {
            name: _sha(root / name)
            for name in (
                "plan.json",
                "terminal.json",
                "provider-calls.json",
                "events.jsonl",
                "final-playbook.txt",
                "native-files.json",
            )
        }
        manifest = dict(
            protocol=PROTOCOL, release_id=RELEASE, files=files, offline_reuse_allowed=False
        )
        _save(root / "manifest.json", {**manifest, "artifact_id": content_hash(manifest)})
        return ACEOnlineResult.load(root)


__all__ = ["ACEOnlineControls", "ACEOnlineRecipe", "ACEOnlineResult", "audit_online_stream"]
