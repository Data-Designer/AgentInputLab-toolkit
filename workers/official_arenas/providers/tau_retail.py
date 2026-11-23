"""Isolated Retail continuation; no banking retrieval or retry-to-success.

This module intentionally leaves the source-pinned Banking provider unchanged.
It shares only the measured agent/budget helpers. A campaign must bind the
user simulator and judge to the evaluated backbone and approve its task/method
matrix before invoking this provider. This is backbone-conditioned evaluation,
not a shared independent judge or an official cross-model leaderboard score.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..protocol import (
    OfficialArenaError, ProviderArmResult, content_hash, intervention_usage,
    materialized_state, require_materialized_budget, resolve_shared_budget_config,
    validate_dependency_hashes, validate_shared_budget, visible_shared_budget,
)
from .tau3 import (
    Tau3Provider, _downstream_agent_usage, _install_local_tau_cost_accounting,
    _intervened_tau_agent_class, _is_generated_tool_argument_validation,
    _is_generated_user_message_validation, _orchestrator_step_cap,
    _remaining_agent_budget,
)
from ._live_history import validate_history_transform

SOURCE_REVISION = "cf71a8070269883e38a365ffa85f78f46844c1f4"
ENVIRONMENT_MODEL_POLICY = "evaluated-backbone-user-and-judge-v1"
RETAIL_SOURCE_SHA256 = {
    "data/tau2/domains/retail/db.json": "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765",
    "data/tau2/domains/retail/policy.md": "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467",
    "data/tau2/domains/retail/split_tasks.json": "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20",
    "data/tau2/domains/retail/tasks.json": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
    "src/tau2/domains/retail/environment.py": "5b819fb895a4ef49df71404c3d9678ab09e8578eec841b8ec9229c8f51e32405",
    "src/tau2/domains/retail/tools.py": "34f4fe9702d36f0dd4697700fbcd63d0cf5386e6eb8e7499aaa336243bad8750",
    "src/tau2/domains/retail/data_model.py": "f5ad09590953855cc49d2ab2ad5c0594be408cae74d1500fb278e5a6bf2c50d6",
    "src/tau2/runner/build.py": "2c4b61b83d4108bb9a2b46724a7b9cc0823548cfd866ca0f415bc536269d964b",
    "src/tau2/evaluator/evaluator.py": "204d95a82812402f2fee989250392d3c66de702a8bcc20258c7e2a8153a1ea8d",
    "src/tau2/evaluator/evaluator_nl_assertions.py": "cb6b3758cdceb02a726732b4ee1af618affcc08f8ab84b44f060146b8f3ad90a",
    "src/tau2/data_model/simulation.py": "518a4eda9af783f25243e1bd2c714f4617cf86188878a0bb584fe73c9528a35a",
}


class RetailJudgeError(OfficialArenaError):
    """Judge generation/coverage is unresolved, not an agent success or zero."""


class RetailAgentOutputError(OfficialArenaError):
    """A paid agent response has no executable message; never retry implicitly."""


def _require_retail_agent_message(response: Any) -> None:
    # tau2 constructs AssistantMessage without immediately calling validate().
    # Reject an empty response before it enters history or reaches the user
    # simulator. Reasoning text is not a substitute for a final answer.
    content = getattr(response, "content", None)
    if ((isinstance(content, str) and content.strip())
            or getattr(response, "tool_calls", None)):
        return
    raise RetailAgentOutputError(
        "Retail agent returned neither final text nor tool calls; "
        "the consumed attempt is retained, without user/judge continuation or retry"
    )


def audit_retail_source(checkout: Path) -> dict[str, Any]:
    """Read source and aggregate inventory only; return no hidden DB/gold text."""
    if checkout.is_symlink() or not checkout.is_dir():
        raise OfficialArenaError("Retail checkout is absent or symlinked")
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != SOURCE_REVISION:
        raise OfficialArenaError("Retail upstream revision differs")
    for relative, expected in RETAIL_SOURCE_SHA256.items():
        path = checkout / relative
        if path.is_symlink() or not path.is_file():
            raise OfficialArenaError(f"Retail pinned file is absent: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise OfficialArenaError(f"Retail pinned file differs: {relative}")
    root = checkout / "data/tau2/domains/retail"
    tasks = json.loads((root / "tasks.json").read_bytes())
    splits = json.loads((root / "split_tasks.json").read_bytes())
    ids = [str(task["id"]) for task in tasks]
    if len(ids) != 114 or len(set(ids)) != 114:
        raise OfficialArenaError("Retail task inventory differs")
    if (len(splits["train"]) != 74 or len(splits["test"]) != 40
            or set(splits["train"]) & set(splits["test"])
            or set(splits["train"]) | set(splits["test"]) != set(ids)
            or set(splits["base"]) != set(ids)):
        raise OfficialArenaError("Retail official split differs")
    rewards = Counter(tuple(task["evaluation_criteria"]["reward_basis"]) for task in tasks)
    if rewards != {("DB", "NL_ASSERTION"): 112, ("DB",): 2}:
        raise OfficialArenaError("Retail reward basis differs")
    return {"source_revision": head, "source_sha256": dict(RETAIL_SOURCE_SHA256),
            "task_counts": {"base": 114, "train": 74, "test": 40},
            "task_ids_by_split": splits, "reward_basis_counts": {"DB+NL_ASSERTION": 112, "DB": 2}}


def capture_first_user_once(*, build_orchestrator: Any, run_config: Any,
                            task: Any, seed: int, user_message_type: type) -> tuple[Any, Any]:
    """Exactly one fixed-seed user generation, never an evaluated-agent turn."""
    orchestrator = build_orchestrator(
        run_config, task, seed=seed, simulation_id=f"ailab-retail-capture-{task.id}-{seed}",
    )
    try:
        orchestrator.initialize()
        # Pinned Retail has no user tools. A single native USER step is enough;
        # never step a second time to manufacture a successful checkpoint.
        if str(getattr(orchestrator.to_role, "value", orchestrator.to_role)) != "user":
            raise OfficialArenaError("Retail capture would execute the evaluated agent")
        # Native get_user_tools() raises when the domain has no user toolkit.
        # Inspect its absence without treating that expected condition as failure.
        if getattr(orchestrator.environment, "user_tools", None) is not None:
            raise OfficialArenaError("Retail capture unexpectedly exposes user tools")
        orchestrator.step()
        message = orchestrator.message
        if (not isinstance(message, user_message_type) or not message.content
                or not message.content.strip() or message.is_tool_call()):
            raise OfficialArenaError("Retail first user generation is not a visible request")
        return orchestrator, message
    except BaseException:
        orchestrator._cleanup()
        raise


def _check_judge_response(response: Any, assertions: list[str]) -> None:
    try:
        parsed = json.loads(response.content)
        rows = parsed["results"]
        valid = (isinstance(rows, list) and len(rows) == len(assertions)
                 and all(isinstance(row, Mapping)
                         and type(row.get("metExpectation")) is bool
                         and isinstance(row.get("reasoning"), str) for row in rows)
                 and Counter(row.get("expectedOutcome") for row in rows) == Counter(assertions))
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        # Upstream all([]) otherwise rewards an empty judge result as success.
        raise RetailJudgeError("Retail judge did not cover the exact official assertions")


@contextmanager
def bound_retail_judge(config: Mapping[str, Any], assertions: list[str]):
    """Scoped judge, explicitly locked to the current evaluated backbone."""
    from tau2.evaluator import evaluator_nl_assertions as module

    original = (module.DEFAULT_LLM_NL_ASSERTIONS, module.DEFAULT_LLM_NL_ASSERTIONS_ARGS, module.generate)
    counters = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def measured_generate(*args: Any, **kwargs: Any) -> Any:
        counters["model_calls"] += 1
        try:
            result = original[2](*args, **kwargs)
        except Exception as error:
            raise RetailJudgeError("Retail backbone-bound judge generation failed") from error
        usage = getattr(result, "usage", None)
        if isinstance(usage, Mapping):
            counters["input_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            counters["output_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        _check_judge_response(result, assertions)
        return result

    module.DEFAULT_LLM_NL_ASSERTIONS = config["llm_judge"]
    module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(config["llm_args_judge"])
    module.generate = measured_generate
    try:
        yield counters
    finally:
        module.DEFAULT_LLM_NL_ASSERTIONS, module.DEFAULT_LLM_NL_ASSERTIONS_ARGS, module.generate = original


def _create_only(path: Path, value: Mapping[str, Any]) -> None:
    # Reserve before invoking a model and fsync before continuing. Interrupted
    # or partial records remain evidence; they are never an excuse to retry.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _saved_capture(path: Path, *, binding: str, task_id: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OfficialArenaError("Retail capture receipt is absent or symlinked")
    try:
        saved = json.loads(path.read_bytes())
        valid = (isinstance(saved, Mapping)
                 and saved.get("protocol") == "retail-first-capture.v1"
                 and saved.get("config_binding") == binding
                 and saved.get("task_id") == task_id
                 and type(saved.get("seed")) is int and saved["seed"] == 17
                 and type(saved.get("attempt")) is int and saved["attempt"] == 1
                 and saved.get("terminal") in {"captured", "capture_failed"}
                 and saved.get("receipt_sha256") == content_hash(
                     {key: value for key, value in saved.items() if key != "receipt_sha256"}))
        if saved.get("terminal") == "captured":
            record = saved.get("record", {})
            valid = valid and (isinstance(record, Mapping) and record.get("checkpoint_id") == task_id
                              and record.get("protected", {}).get("task_id") == task_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise OfficialArenaError("Retail existing capture receipt differs")
    return saved


def _retail_agent_class(generation_failures: list[dict[str, Any]]) -> type:
    """Retain errors that the shared measured agent turns into hard stops."""
    from tau2.utils import llm_utils
    original = llm_utils.generate

    def observed_generate(*args: Any, **kwargs: Any):
        try:
            return original(*args, **kwargs)
        except Exception as error:
            generation_failures.append({"exception_type": type(error).__name__,
                                        "exception_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                                        "call_name": kwargs.get("call_name"), "retry_permitted": False})
            raise

    # The shared class closes over generate when built. Restore the module
    # immediately; this observer applies only to the evaluated agent, not to
    # the user simulator or judge, and no previous provider is modified.
    llm_utils.generate = observed_generate
    try:
        base = _intervened_tau_agent_class()
    finally:
        llm_utils.generate = original

    class ValidatedRetailAgent(base):
        def _generate_next_message(self, message: Any, state: Any):
            # The shared implementation has already charged the real response
            # usage. Do not validate inside observed_generate, before accounting.
            response = super()._generate_next_message(message, state)
            _require_retail_agent_message(response)
            return response

    return ValidatedRetailAgent


def _paper_terminal_success(raw_reward: float, termination: str, hard_stops: int) -> float:
    """Separate budget/timeout failure accounting from native raw reward."""
    if hard_stops or termination in {"timeout", "max_steps", "too_many_errors", "agent_error", "user_error",
                                     "infrastructure_error", "context_window_exceeded", "unexpected_error"}:
        return 0.0
    return raw_reward


def _environment_user_usage(messages: list[Any], initial_length: int, user_type: type) -> dict[str, Any]:
    rows = [message for message in messages[initial_length:] if isinstance(message, user_type)]
    known = [message.usage for message in rows if isinstance(getattr(message, "usage", None), Mapping)]
    return {"model_calls": len(rows), "input_tokens": sum(int(row.get("prompt_tokens", row.get("input_tokens", 0)) or 0) for row in known),
            "output_tokens": sum(int(row.get("completion_tokens", row.get("output_tokens", 0)) or 0) for row in known),
            "token_usage_incomplete": len(known) != len(rows)}


class TauRetailProvider:
    benchmark_id = "tau3.retail"
    source_revision = SOURCE_REVISION
    scorer_id = "tau3.retail.official_v1.0.1.backbone_conditioned_judge"
    scorer_revision = content_hash({"native": RETAIL_SOURCE_SHA256["src/tau2/evaluator/evaluator.py"],
                                    "nl": RETAIL_SOURCE_SHA256["src/tau2/evaluator/evaluator_nl_assertions.py"],
                                    "judge_coverage_guard": "exact-assertion-multiset-bool-v1",
                                    "environment_model_policy": ENVIRONMENT_MODEL_POLICY})
    _FIELDS = frozenset({"task_set_name", "task_split_name", "limit", "capture_seed", "llm_agent",
                         "llm_user", "llm_judge", "llm_args_agent", "llm_args_user", "llm_args_judge",
                         "max_steps", "max_errors", "timeout", "locked_dependency_hashes", "budget_override"})

    def __init__(self, *, history_transform: Any = None) -> None:
        validate_history_transform(history_transform)
        self.history_transform = history_transform

    def _validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(config, Mapping):
            raise OfficialArenaError("Retail provider config must be an object")
        parsed = resolve_shared_budget_config(config)
        if set(parsed) - {"capture_task_ids"} != self._FIELDS:
            raise OfficialArenaError("Retail config fields differ; banking retrieval is not applicable")
        if "capture_task_ids" in parsed:
            ids = parsed["capture_task_ids"]
            if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x.isdigit() for x in ids)
                    or len(set(ids)) != len(ids) or type(parsed["limit"]) is not int or len(ids) > parsed["limit"]):
                raise OfficialArenaError("Retail capture_task_ids must be unique official IDs within limit")
        if parsed["task_set_name"] != "retail" or parsed["task_split_name"] not in {"train", "test"}:
            raise OfficialArenaError("Retail requires an explicit official train or test split")
        limit = 40 if parsed["task_split_name"] == "test" else 74
        for name, cap in (("limit", limit), ("max_steps", 10000), ("max_errors", 10000)):
            raw = parsed[name]
            if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= cap:
                raise OfficialArenaError(f"Retail {name} is invalid")
        if parsed["capture_seed"] != 17 or type(parsed["capture_seed"]) is not int:
            raise OfficialArenaError("Retail capture seed must be frozen at 17")
        timeout = parsed["timeout"]
        if (isinstance(timeout, bool) or not isinstance(timeout, (float, int))
                or not math.isfinite(timeout) or timeout <= 0):
            raise OfficialArenaError("Retail episode timeout must be explicit and finite")
        budget = validate_shared_budget(parsed["budget_override"])
        locks = validate_dependency_hashes(parsed["locked_dependency_hashes"], label="Retail")
        if not {"agent_model", "user_simulator", "nl_assertion_judge", "official_runtime", "scorer"} <= set(locks):
            raise OfficialArenaError("Retail model/user/judge/runtime/scorer locks are incomplete")
        for role in ("agent", "user", "judge"):
            model, args = parsed[f"llm_{role}"], parsed[f"llm_args_{role}"]
            if not isinstance(model, str) or not model.strip() or not isinstance(args, Mapping):
                raise OfficialArenaError(f"Retail {role} model binding is invalid")
            if (urlparse(args.get("api_base", "")).hostname not in {"127.0.0.1", "localhost", "::1"}
                    or type(args.get("temperature")) not in {int, float} or args.get("temperature") != 0
                    or type(args.get("seed")) is not int or args.get("seed") != 17
                    or type(args.get("num_retries")) is not int or args.get("num_retries") != 0):
                raise OfficialArenaError(f"Retail {role} requires local, fixed decoding and no retries")
            cap = args.get("max_tokens")
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise OfficialArenaError(f"Retail {role} output cap is absent")
            request_timeout = args.get("timeout")
            if (type(request_timeout) not in {int, float} or not math.isfinite(request_timeout)
                    or request_timeout <= 0):
                raise OfficialArenaError(f"Retail {role} request timeout must be explicit and finite")
        if parsed["llm_args_agent"]["max_tokens"] > budget["max_output_tokens_per_call"]:
            raise OfficialArenaError("Retail per-call agent cap exceeds its declared shared budget")
        if parsed["llm_args_judge"].get("response_format") != {"type": "json_object"}:
            raise OfficialArenaError("Retail judge requires an explicit JSON response format")
        for role in ("user", "judge"):
            if parsed[f"llm_{role}"] != parsed["llm_agent"]:
                raise OfficialArenaError("Retail user and judge must follow the evaluated backbone")
            if parsed[f"llm_args_{role}"]["api_base"].rstrip("/") != parsed["llm_args_agent"]["api_base"].rstrip("/"):
                raise OfficialArenaError("Retail user and judge must use the evaluated model endpoint")
        return parsed

    def doctor(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            self._validate_config(config)
            from tau2.domains.retail import environment
            from tau2.domains.retail.utils import RETAIL_DB_PATH
            from tau2.utils import llm_utils
            checkout = Path(environment.__file__).resolve().parents[4]
            audit_retail_source(checkout)
            if Path(RETAIL_DB_PATH).resolve() != checkout / "data/tau2/domains/retail/db.json":
                raise OfficialArenaError("Retail effective DB is not its pinned data source")
            if getattr(llm_utils.completion, "_agentinputlab_tool_argument_normalizer", False):
                raise OfficialArenaError("Retail requires a fresh process without banking decoder repairs")
        except Exception as error:
            return (f"{type(error).__name__}: {error}",)
        return ()

    def _run_config(self, config: Mapping[str, Any], *, budget: Mapping[str, Any] | None = None):
        from tau2.data_model.simulation import TextRunConfig
        args = dict(config["llm_args_agent"])
        steps = config["max_steps"]
        if budget is not None:
            steps = min(steps, budget["max_model_calls"])
            args["max_tokens"] = min(args["max_tokens"], budget["max_output_tokens"])
        return TextRunConfig(domain="retail", agent="llm_agent", user="user_simulator",
                             llm_agent=config["llm_agent"], llm_user=config["llm_user"],
                             llm_args_agent=args, llm_args_user=dict(config["llm_args_user"]),
                             max_steps=steps, max_errors=config["max_errors"], timeout=config["timeout"],
                             retrieval_config=None, retrieval_config_kwargs={})

    def capture(self, config: Mapping[str, Any], *, output_root: Path) -> tuple[Mapping[str, Any], ...]:
        batch = self.capture_batch(config, output_root=output_root)
        unresolved = [row["task_id"] for row in batch["dispositions"] if row["status"] != "captured"]
        if unresolved:
            raise OfficialArenaError(f"Retail first-capture failures retained; no automatic retry: {unresolved}")
        return tuple(batch["records"])

    def capture_batch(self, config: Mapping[str, Any], *, output_root: Path) -> Mapping[str, Any]:
        """Keep the complete expected inventory even when some captures fail.

        An existing start without a receipt is unresolved, not proof of a dead
        process. The scoped Retail controller can export a partial inventory
        after every task has a terminal disposition, without retrying failures.
        """
        parsed = self._validate_config(config)
        blockers = self.doctor(parsed)
        if blockers:
            raise OfficialArenaError("; ".join(blockers))
        _install_local_tau_cost_accounting(parsed)
        from tau2.data_model.message import UserMessage
        from tau2.runner.build import build_orchestrator
        from tau2.runner.helpers import get_tasks
        selection = {"task_ids": parsed["capture_task_ids"]} if "capture_task_ids" in parsed else {}
        tasks = get_tasks("retail", task_split_name=parsed["task_split_name"], num_tasks=parsed["limit"], **selection)
        output_root.mkdir(parents=True, exist_ok=True)
        rows, dispositions = [], []
        binding = content_hash(parsed)
        for task in tasks:
            # Official IDs are numerical; reject unexpected path components.
            if not str(task.id).isdigit():
                raise OfficialArenaError("Retail task ID is not a safe official identity")
            target = output_root / f"first-capture-{task.id}.json"
            if target.exists() or target.is_symlink():
                saved = _saved_capture(target, binding=binding, task_id=str(task.id))
            else:
                saved = {"protocol": "retail-first-capture.v1", "config_binding": binding,
                         "task_id": str(task.id), "seed": 17, "attempt": 1}
                try:
                    _create_only(output_root / f"first-capture-{task.id}-start.json", saved)
                except FileExistsError:
                    # This can be live or interrupted. Neither elapsed time nor
                    # an absent final file proves termination; never retry it.
                    dispositions.append({"task_id": str(task.id), "status": "reserved_unresolved",
                                         "retry_permitted": False})
                    continue
                orchestrator = None
                try:
                    orchestrator, message = capture_first_user_once(
                        build_orchestrator=build_orchestrator, run_config=self._run_config(parsed),
                        task=task, seed=17, user_message_type=UserMessage)
                    tools = orchestrator.environment.get_tools()
                    schemas = [tool.openai_schema for tool in tools]
                    policy = orchestrator.environment.get_policy()
                    history = [item.model_copy(update={"timestamp": None, "turn_idx": None}).model_dump(mode="json")
                               for item in orchestrator.get_messages()]
                    state = {"query": message.content, "task_id": str(task.id), "instruction": policy,
                             "context": [{"text": policy, "block_id": "official_policy", "source": "tau3.retail.public_policy",
                                          "score": None, "metadata": {"domain": "retail"}},
                                         {"text": json.dumps(schemas, sort_keys=True, ensure_ascii=False),
                                          "block_id": "official_tool_catalog", "source": "tau3.retail.tool_schemas",
                                          "score": None, "metadata": {"domain": "retail", "tool_defs_hash": content_hash(schemas)}}],
                             "messages": [{"role": "user", "content": message.content, "name": None}],
                             "tool_names": [tool.name for tool in tools],
                             "metadata": {"benchmark_id": self.benchmark_id, "checkpoint_phase": "after_initial_user_message",
                                          "clarification_roundtrip_available": True, "state_restore_available": True,
                                          "capture_attempt": 1, "capture_seed": 17,
                                          "acquire_structural_na": "no_native_KB_or_approved_public_retriever"},
                             "budget": visible_shared_budget(parsed, {})}
                    record = {"checkpoint_id": str(task.id), "group_key": str(task.id), "visible_state": state,
                              "eligible_families": ["reformulate", "decompose", "clarify", "curate"],
                              "protected": {"task_id": str(task.id), "initial_message_history": history,
                                            "capture_attempt": 1, "capture_seed": 17,
                                            "locked_dependency_hashes": dict(parsed["locked_dependency_hashes"])}}
                    saved.update({"terminal": "captured", "record": record})
                    captured_usage = getattr(message, "usage", None)
                    saved["environment_capture_usage"] = {"model_calls": 1,
                        "tokens": dict(captured_usage) if isinstance(captured_usage, Mapping) else None,
                        "token_usage_incomplete": not isinstance(captured_usage, Mapping), "outside_agent_budget": True}
                except Exception as error:
                    saved.update({"terminal": "capture_failed", "exception_type": type(error).__name__,
                                  "exception_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                                  "retry_permitted": False})
                finally:
                    if orchestrator is not None:
                        orchestrator._cleanup()
                saved["receipt_sha256"] = content_hash(saved)
                _create_only(target, saved)
            if saved["terminal"] == "captured":
                rows.append(saved["record"])
            dispositions.append({"task_id": str(task.id), "status": saved["terminal"],
                                 "receipt_sha256": saved["receipt_sha256"], "retry_permitted": False})
        if not tasks:
            raise OfficialArenaError("Retail capture resolved no official tasks")
        return {"protocol": "retail-capture-batch.v1", "config_binding": binding,
                "expected_task_ids": [str(task.id) for task in tasks], "records": rows,
                "dispositions": dispositions,
                "all_tasks_terminal": all(row["status"] != "reserved_unresolved" for row in dispositions)}

    def run_arm(self, config: Mapping[str, Any], *, replay_record: Mapping[str, Any],
                arm_payload: Mapping[str, Any], seed: int, role: str, work_dir: Path) -> ProviderArmResult:
        if self.history_transform is not None and (work_dir / "live-history.json").exists():
            raise OfficialArenaError("native history output must be a fresh episode")
        parsed = self._validate_config(config)
        if type(seed) is not int or seed != 17 or role not in {"baseline", "treatment"}:
            raise OfficialArenaError("Retail arm seed or role differs")
        blockers = self.doctor(parsed)
        if blockers:
            raise OfficialArenaError("; ".join(blockers))
        local_models = _install_local_tau_cost_accounting(parsed)
        _install_local_tau_cost_accounting({"llm_agent": parsed["llm_judge"],
                                           "llm_args_agent": parsed["llm_args_judge"]})
        state = materialized_state(arm_payload)
        budget = _remaining_agent_budget(require_materialized_budget(state, parsed), intervention_usage(arm_payload))
        protected = replay_record["protected"]
        if (not isinstance(protected, Mapping)
                or type(protected.get("capture_attempt")) is not int or protected["capture_attempt"] != 1
                or type(protected.get("capture_seed")) is not int or protected["capture_seed"] != 17
                or protected.get("task_id") != replay_record["checkpoint_id"]
                or not str(protected.get("task_id", "")).isdigit()
                or protected.get("locked_dependency_hashes") != parsed["locked_dependency_hashes"]):
            raise OfficialArenaError("Retail replay is not the fixed first capture")
        # Lock this physical arm before spending model budget. Resuming a saved
        # failure may inspect it but must not perform inference in this directory.
        _create_only(work_dir / "attempt-start.json", {"benchmark_id": self.benchmark_id,
                     "checkpoint_id": replay_record["checkpoint_id"], "role": role, "seed": seed,
                     "config_sha256": content_hash(parsed), "arm_payload_sha256": content_hash(arm_payload)})
        if (budget["max_model_calls"] < 1 or budget["max_output_tokens"] < 1
                or (budget.get("max_context_tokens") is not None and budget["max_context_tokens"] < 1)):
            return self._terminal_zero(replay_record, role, seed, work_dir, "shared_budget_exhausted_before_agent", [], {}, budget)
        from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
        from tau2.data_model.tasks import InitialState
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner.build import _build_env_kwargs, build_environment, build_user
        from tau2.runner.helpers import get_tasks
        from tau2.runner.simulation import run_simulation
        tasks = get_tasks("retail", task_split_name=parsed["task_split_name"], task_ids=[protected["task_id"]])
        if len(tasks) != 1:
            raise OfficialArenaError("Retail protected task identity is unavailable")
        task = tasks[0]
        message_types = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
        history = [message_types[row["role"]].model_validate(row) for row in protected["initial_message_history"]]
        if not history or not isinstance(history[-1], UserMessage) or history[-1].is_tool_call():
            raise OfficialArenaError("Retail protected capture history is invalid")
        initial = task.initial_state
        task = task.model_copy(update={"initial_state": InitialState(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=copy.deepcopy(history))}, deep=True)
        run_config = self._run_config(parsed, budget=budget)
        kwargs = _build_env_kwargs(run_config, task)
        if kwargs:
            raise OfficialArenaError("Retail received non-native environment kwargs")
        environment = build_environment("retail", env_kwargs={})
        generation_failures: list[dict[str, Any]] = []
        agent = _retail_agent_class(generation_failures)(
            tools=environment.get_tools(), domain_policy=environment.get_policy(),
            llm=parsed["llm_agent"], llm_args=dict(run_config.llm_args_agent), materialized=state, budget=budget,
            history_transform=self.history_transform)
        user = build_user("user_simulator", environment, task, llm=parsed["llm_user"], llm_args=dict(parsed["llm_args_user"]))
        orchestrator = Orchestrator(domain="retail", agent=agent, user=user, environment=environment,
                                    task=task, seed=17, max_steps=_orchestrator_step_cap(remaining_budget=budget, materialized=state),
                                    max_errors=parsed["max_errors"], timeout=parsed["timeout"],
                                    simulation_id=f"ailab-retail-{role}-{task.id}-17")
        assertions = list(task.evaluation_criteria.nl_assertions or [])
        try:
            with bound_retail_judge(parsed, assertions) as judge_usage:
                simulation = run_simulation(orchestrator, env_kwargs={})
        except RetailAgentOutputError as error:
            _create_only(work_dir / "generation-failure.json", {
                "benchmark_id": self.benchmark_id, "checkpoint_id": replay_record["checkpoint_id"],
                "failure_code": "generated_agent_message_empty", "error": str(error),
                "usage": self._agent_usage(orchestrator.get_messages(), len(history), AssistantMessage, agent),
                "automatic_retries": 0, "native_reward": None,
                "final_text_and_tool_calls_absent": True,
            })
            raise
        except (json.JSONDecodeError, TimeoutError, ValueError) as error:
            recognized = (isinstance(error, (json.JSONDecodeError, TimeoutError))
                          or _is_generated_tool_argument_validation(error)
                          or _is_generated_user_message_validation(error))
            if not recognized:
                raise
            usage = self._agent_usage(orchestrator.get_messages(), len(history), AssistantMessage, agent)
            reason = "episode_timeout" if isinstance(error, TimeoutError) else "malformed_generated_trajectory"
            result = self._terminal_zero(replay_record, role, seed, work_dir, reason,
                                         orchestrator.get_messages(), usage, budget, error=error)
            return Tau3Provider._history_result(result, agent, work_dir)
        finally:
            try:
                if getattr(agent, "_history_session", None) is not None:
                    agent._history_session.write_receipt(work_dir / "live-history.json",
                                                         target_usage=agent._history_target_usage())
            finally:
                orchestrator._cleanup()
        if simulation.reward_info is None:
            raise RetailJudgeError("Retail official evaluator returned no reward")
        messages = simulation.get_messages()
        usage = self._agent_usage(messages, len(history), AssistantMessage, agent)
        if usage["tool_calls"] != agent._budget_tool_calls:
            raise OfficialArenaError("Retail tool-call trace differs from the enforced counter")
        agent_cost, user_cost = float(simulation.agent_cost or 0), float(simulation.user_cost or 0)
        trace = {"provider": "tau_retail", "benchmark_id": self.benchmark_id, "source_revision": self.source_revision,
                 "task_id": str(task.id), "role": role, "seed": 17,
                 "termination_reason": simulation.termination_reason.value,
                 "messages": [message.model_dump(mode="json") for message in messages],
                 "reward_info": simulation.reward_info.model_dump(mode="json"),
                 "environment_model_policy": ENVIRONMENT_MODEL_POLICY,
                 "backbone_conditioned_environment_models": {"agent": parsed["llm_agent"],
                                               "user": parsed["llm_user"], "judge": parsed["llm_judge"],
                                               "user_lock": parsed["locked_dependency_hashes"]["user_simulator"],
                                               "judge_lock": parsed["locked_dependency_hashes"]["nl_assertion_judge"]},
                 "environment_usage": {"judge": judge_usage, "user": _environment_user_usage(messages, len(history), UserMessage),
                                       "user_cost_usd": user_cost,
                                       "outside_agent_budget": True},
                 "cost_accounting": "local_unpriced_not_free" if local_models else "upstream_litellm",
                 "generation_controls": {"hard_stops": agent._budget_hard_stops,
                                         "executed_tool_calls": agent._budget_tool_calls,
                                         "discarded_tool_calls": agent._budget_discarded_tool_calls,
                                         "automatic_retries": 0, "decoder_repair": False,
                                         "retained_generation_failures": generation_failures},
                 "paper_terminal_success": _paper_terminal_success(float(simulation.reward_info.reward),
                     simulation.termination_reason.value, agent._budget_hard_stops),
                 "failure_accounting": "native_raw_reward_preserved; timeout_and_budget_failure_zero_in_paper_metric"}
        _create_only(work_dir / "simulation.json", trace)
        result = ProviderArmResult(metrics={"official_reward": float(simulation.reward_info.reward),
                                          "terminal_success": trace["paper_terminal_success"],
                                          "cost": agent_cost + user_cost, "adapter_hard_stops": float(agent._budget_hard_stops)},
                                 trace=trace, usage={**usage, "cost_usd": agent_cost + user_cost,
                                                     "unpriced_model_calls": usage["model_calls"] if local_models else 0},
                                 runtime_instance_id=content_hash({"provider": self.benchmark_id, "task_id": str(task.id),
                                                                   "role": role, "seed": seed, "work_dir": str(work_dir)}))
        return Tau3Provider._history_result(result, agent, work_dir)

    @staticmethod
    def _agent_usage(messages, initial_length, assistant_type, agent):
        return _downstream_agent_usage(messages, initial_history_length=initial_length,
                                       assistant_message_type=assistant_type,
                                       measured_model_calls=agent._budget_model_calls,
                                       measured_input_tokens=agent._budget_input_tokens,
                                       measured_output_tokens=agent._budget_output_tokens)

    def _terminal_zero(self, replay, role, seed, work_dir, reason, messages, usage, budget, *, error=None):
        trace = {"provider": "tau_retail", "benchmark_id": self.benchmark_id, "source_revision": self.source_revision,
                 "task_id": replay["protected"]["task_id"], "role": role, "seed": seed,
                 "termination_reason": reason, "reward_info": None,
                 "score_provenance": "predeclared_terminal_zero_not_fabricated_official_reward",
                 "messages": [message.model_dump(mode="json") for message in messages],
                 "generation_controls": {"hard_stops": 1, "remaining_agent_budget": dict(budget), "automatic_retries": 0},
                 "exception_type": type(error).__name__ if error else None,
                 "exception_sha256": hashlib.sha256(str(error).encode()).hexdigest() if error else None}
        _create_only(work_dir / "terminal-zero.json", trace)
        measured = {"model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0, **usage}
        return ProviderArmResult(metrics={"official_reward": 0.0, "terminal_success": 0.0, "cost": 0.0, "adapter_hard_stops": 1.0},
                                 trace=trace, usage={**measured, "cost_usd": 0.0, "unpriced_model_calls": measured["model_calls"]},
                                 runtime_instance_id=content_hash({"provider": self.benchmark_id, "checkpoint_id": replay["checkpoint_id"],
                                                                   "role": role, "seed": seed, "work_dir": str(work_dir)}))


def factory() -> TauRetailProvider:
    return TauRetailProvider()
