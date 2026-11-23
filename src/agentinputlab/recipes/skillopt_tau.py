"""SkillOpt on complete native Tau episodes, using frozen development captures."""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import math
import os
import random
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

from agentinputlab.api import _state
from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ContextBlock, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.recipes.ace_native_plan import (
    _save,
    native_process_scope,
    verify_native_checkout,
    verify_source_files,
)
from agentinputlab.recipes.ace_tau import SCHEMA, _native_inputs, _provider, _same_backbone
from agentinputlab.recipes.skillopt_toolsandbox import _launch
from workers.official_arenas.protocol import EXECUTABLE_PAYLOAD_PROTOCOL


def _source(configuration):
    pin = configuration["source_input"]
    source = load_pinned_json_mapping(
        pin["path"], expected_sha256=pin["sha256"], max_bytes=8 * 1024 * 1024
    )
    provider = _provider(source.get("domain"))
    checkout = verify_native_checkout(configuration["checkout"], revision=provider.source_revision)
    if (
        source.get("schema") != SCHEMA
        or source.get("benchmark") != provider.benchmark_id
        or source.get("native_source_revision") != provider.source_revision
    ):
        raise ValidationError("Tau fitting source identity differs")
    verify_source_files(checkout, source["source_files"])
    values = source.get("fitting_rows")
    if not isinstance(values, list) or not values:
        raise ValidationError("Tau fitting requires frozen development captures")
    rows = {r["case_id"]: r for r in values}
    if (
        len(rows) != len(values)
        or set(rows) & set(source["heldout_ids"])
        or {r["role"] for r in values} != {"train", "validation"}
    ):
        raise ValidationError("Tau fitting and heldout groups must be unique and disjoint")
    expected_split = "train" if source["domain"] == "retail" else None
    if source.get("source_official_split") != expected_split or not source.get("partition_source"):
        raise ValidationError("Tau development split provenance differs")
    states = {k: _state(r["state"]) for k, r in rows.items()}
    if len({s.state_hash for s in states.values()}) != len(states):
        raise ValidationError("Tau fitting states must be disjoint")
    configs = {}
    for key, row in rows.items():
        config = provider._validate_config(row["native_configuration"])
        _same_backbone(config, source["target_model"])
        state, capture = states[key], row["protected_checkpoint"]
        if (
            config["task_split_name"] != "train"
            or state.state_hash != row["state_hash"]
            or capture["checkpoint_id"] != key
            or _state(capture["visible_state"]).state_hash != state.state_hash
            or capture["protected"]["locked_dependency_hashes"]
            != config["locked_dependency_hashes"]
        ):
            raise ValidationError("Tau captured state, split or dependencies differ")
        budget = config["budget_override"]
        if configuration["per_call_output_tokens"] != budget["max_output_tokens_per_call"] or any(
            budget[k] != to_jsonable(state.budget)[k]
            for k in (
                "max_context_tokens",
                "max_model_calls",
                "max_output_tokens",
                "max_retrieval_calls",
                "max_tool_calls",
            )
        ):
            raise ValidationError("Tau fitting budget differs from captured budget")
        for role in ("agent", "user", "judge"):
            args = config.get("llm_args_" + role, config["llm_args_agent"])
            if args.get("seed") != configuration["seed"] or args.get("num_retries") != 0:
                raise ValidationError("Tau fitting requires the original seed and no retries")
        for field, locks in (
            ("backend", ("agent_model",)),
            ("environment_backend", ("user_simulator", "nl_assertion_judge")),
        ):
            backend = configuration[field]
            identity = backend["deployment_identity"]
            if backend["model_id"] != source["target_model"]["model_id"] or any(
                config["locked_dependency_hashes"][lock]
                not in (content_hash(identity), identity.get("revision"))
                for lock in locks
            ):
                raise ValidationError("Tau role deployment differs from its captured model")
            original = source["target_model"].get("original_model_json_hashes")
            if original is not None and identity.get("model_json_hashes") != original:
                raise ValidationError("Tau original model asset binding differs")
            # The pinned native judge policy is explicitly local, not an
            # implicit remote judge fallback. Preserve that policy.
            if urlsplit(backend["base_url"]).hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValidationError(
                    "This native Tau fitting profile requires local model endpoints"
                )
        configs[key] = config
    return source, rows, states, provider, checkout, configs


def _native_role():
    frames = inspect.stack(context=0)
    if any(f.filename.endswith("/tau2/evaluator/evaluator_nl_assertions.py") for f in frames):
        return "judge"
    if any(f.filename.endswith("/tau2/user/user_simulator.py") for f in frames):
        return "user"
    if any(
        f.filename.endswith("/providers/tau3.py") and f.function == "_generate_next_message"
        for f in frames
    ):
        return "agent"
    raise ValidationError("Unclassified native Tau model call; no request was forwarded")


@contextlib.contextmanager
def _account_calls(output, configuration, native):
    """Route native roles without changing their request bodies or scoring.

    The native provider requires one logical endpoint for its same-model
    roles. Relocate transport only, allowing the evaluated agent to pass
    through SkillOpt's fit meter while simulator/judge costs remain separate.
    This is scoped to a dedicated worker and native_process_scope's lock.
    """
    import httpx

    original, events = httpx.Client.send, []
    logical = native["llm_args_agent"]["api_base"].rstrip("/") + "/chat/completions"

    def send(client, request, *args, **kwargs):
        if request.method != "POST" or str(request.url) != logical:
            raise ValidationError("Unexpected native Tau endpoint; no request was forwarded")
        role = _native_role()
        raw = request.read()
        body = json.loads(raw)
        controls = native.get("llm_args_" + role, native["llm_args_agent"])
        cap = body.get("max_tokens")
        if (
            body.get("model") != configuration["backend"]["model_id"]
            or body.get("seed") != configuration["seed"]
            or type(cap) is not int
            or cap < 1
            or cap > controls["max_tokens"]
        ):
            raise ValidationError("Tau native model, seed or per-role cap changed")
        backend = configuration["backend" if role == "agent" else "environment_backend"]
        secret = os.environ.get(backend.get("api_key_env", "OPENAI_API_KEY"))
        if not secret:
            raise ValidationError("Tau role api_key_env is not set")
        headers = dict(request.headers)
        headers.pop("host", None)
        headers["authorization"] = "Bearer " + secret
        wire = httpx.Request(
            request.method,
            backend["base_url"].rstrip("/") + "/chat/completions",
            headers=headers,
            content=raw,
            extensions=request.extensions,
        )
        event = dict(role=role, status="started", usage=None, automatic_retry=False)
        events.append(event)
        index = len(events)
        try:
            response = original(client, wire, *args, **kwargs)
            response.read()
            value = response.json()
            event.update(
                status="returned", http_status=response.status_code, usage=value.get("usage")
            )
            return response
        except Exception as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            _save(output / f"native-call-{index:04d}.json", event)

    httpx.Client.send = send
    try:
        yield events
    finally:
        httpx.Client.send = original


def _episode(configuration, case_id, skill, output):
    source, rows, states, provider, checkout, configs = _source(configuration)
    if case_id not in rows or not isinstance(skill, str) or not skill.strip():
        raise ValidationError("Tau episode requires a declared case and nonempty skill")
    with native_process_scope(checkout, package="tau2"):
        _, _, schemas, _, _, pins = _native_inputs(
            checkout, source["domain"], list(rows.values()), provider
        )
        if pins != source["source_files"] or schemas != source["native_function_schemas"]:
            raise ValidationError("Tau native task files or tool schemas differ")
        native = copy.deepcopy(to_jsonable(configs[case_id]))
        for role in ("agent", "user", "judge"):
            name = "llm_args_" + role
            if name not in native:
                continue
            backend = configuration["backend" if role == "agent" else "environment_backend"]
            native[name]["api_base"] = configuration["backend"]["base_url"]
            # A non-secret placeholder only reaches the scoped transport hook.
            native[name]["api_key"] = "agentinputlab-native-role-routing"
            if "extra_body" in backend:
                native[name]["extra_body"] = to_jsonable(backend["extra_body"])
        native = provider._validate_config(native)
        state = states[case_id]
        patched = replace(
            state,
            context=(
                *state.context,
                ContextBlock(
                    text="SKILLOPT CURRENT FITTING SKILL\n" + skill,
                    block_id="skillopt-fit-skill:" + content_hash(skill),
                    source="SkillOpt native fitting candidate",
                ),
            ),
        )
        payload = dict(
            schema_version=1,
            protocol=EXECUTABLE_PAYLOAD_PROTOCOL,
            visible_state_hash=state.state_hash,
            release_id="recipe.skillopt.offline",
            intervention_execution=dict(
                input_state_hash=state.state_hash,
                output_state_hash=patched.state_hash,
                materialized_state=to_jsonable(patched),
                usage=dict(model_calls=0, input_tokens=0, output_tokens=0),
            ),
        )
        with _account_calls(output, configuration, native) as calls:
            result = provider.run_arm(
                native,
                replay_record=rows[case_id]["protected_checkpoint"],
                arm_payload=payload,
                seed=configuration["seed"],
                role="treatment",
                work_dir=output,
            )
    score = result.metrics.get("official_reward")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValidationError("Tau native reward is invalid")
    if (
        result.usage["model_calls"] > state.budget.max_model_calls
        or result.usage["output_tokens"] > state.budget.max_output_tokens
    ):
        raise ValidationError("Tau native agent budget exceeded")
    return dict(
        status="scored",
        metrics=to_jsonable(result.metrics),
        native_usage=to_jsonable(result.usage),
        native_trace=to_jsonable(result.trace),
        native_role_calls=to_jsonable(calls),
        source_state_hash=state.state_hash,
        patched_state_hash=patched.state_hash,
        automatic_retry=False,
    )


def create_tau_environment(*, cfg, configuration):
    from skillopt.envs.base import EnvAdapter

    source, rows, states, provider, checkout, _ = _source(configuration)
    groups = {r: [k for k, v in rows.items() if v["role"] == r] for r in ("train", "validation")}
    byid = {content_hash(k): k for k in rows}

    def visible(k):
        return dict(id=content_hash(k), question=states[k].query, task_type=provider.benchmark_id)

    class TauEnv(EnvAdapter):
        def __init__(self):
            self.analyst_workers = cfg.get("analyst_workers", 1)
            self.failure_only = cfg.get("failure_only", False)
            self.minibatch_size = cfg.get("minibatch_size", 1)
            self.edit_budget = cfg.get("edit_budget", 4)

        def fingerprint(self):
            return dict(
                implementation="agentinputlab.skillopt.native_Tau_environment.v1",
                source_input=configuration["source_input"],
                source_revision=provider.source_revision,
                target_model=source["target_model"],
                partition_source=source["partition_source"],
                groups={
                    r: [dict(case_id=k, state_hash=states[k].state_hash) for k in keys]
                    for r, keys in groups.items()
                },
                native_agent_user_judge_tools_and_scorer_unchanged=True,
                heldout_not_exposed=True,
                automatic_retry=False,
            )

        def _batch(self, role, count, seed):
            if type(count) is not int or not 1 <= count <= len(groups[role]):
                raise ValidationError("Tau batch exceeds declared fitting group")
            return [visible(k) for k in random.Random(seed).sample(groups[role], count)]

        def build_train_env(self, batch_size, seed, **kw):
            return self._batch("train", batch_size, seed)

        def build_eval_env(self, env_num, split, seed, **kw):
            if split not in ("val", "validation", "valid_seen"):
                raise ValidationError("Tau fitting exposes validation only")
            return self._batch("validation", env_num, seed)

        def get_task_types(self):
            return [provider.benchmark_id]

        def rollout(self, env_manager, skill_content, out_dir, **kw):
            _source(configuration)
            if not isinstance(skill_content, str) or not skill_content.strip():
                raise ValidationError("Skill must be nonempty")
            if any(
                kw.get(k)
                for k in (
                    "diagnostic_mode",
                    "diagnostic_instruction",
                    "diagnostic_trace_context_by_id",
                )
            ):
                raise ValidationError("Extra diagnostic episodes not admitted")
            items = list(env_manager)
            if (
                any(not isinstance(i, dict) or i.get("id") not in byid for i in items)
                or len({i["id"] for i in items}) != len(items)
                or any(i != visible(byid[i["id"]]) for i in items)
            ):
                raise ValidationError("Undeclared, duplicated or changed Tau fitting case")
            root = Path(out_dir).absolute()
            if root.resolve().is_relative_to(checkout):
                raise ValidationError("Fit output must be outside native checkout")
            root.mkdir(parents=True, exist_ok=True)
            results = []
            for item in items:
                key = byid[item["id"]]
                folder = root / "predictions" / item["id"]
                folder.mkdir(parents=True, exist_ok=False)
                outcome = _launch(configuration, key, skill_content, folder)
                score = float(outcome["metrics"]["official_reward"])
                messages = outcome["native_trace"].get("messages")
                if not isinstance(messages, list):
                    raise ValidationError("Tau native trace has no visible conversation")
                trace = json.dumps(
                    [m for m in messages if m.get("role") in ("assistant", "user", "tool")],
                    sort_keys=True,
                    ensure_ascii=False,
                )
                result = dict(
                    id=item["id"],
                    source_case_id=key,
                    question=states[key].query,
                    task_type=provider.benchmark_id,
                    hard=int(score >= 1),
                    soft=score,
                    response=trace,
                    agent_ok=True,
                    n_turns=outcome["native_usage"]["model_calls"],
                    fail_reason="" if score >= 1 else "native Tau official_reward<1",
                    native_metrics=outcome["metrics"],
                    native_agent_usage=outcome["native_usage"],
                    native_role_calls=outcome["native_role_calls"],
                    source_state_hash=outcome["source_state_hash"],
                    patched_state_hash=outcome["patched_state_hash"],
                    native_agent_user_judge_tools_and_scorer_unchanged=True,
                    evaluator_only_gold_not_forwarded_to_reflection=True,
                )
                _save(folder / "result.json", result)
                _save(
                    folder / "conversation.json",
                    [
                        dict(type="message", turn=1, content=trace),
                        dict(
                            role="system",
                            content="[NATIVE TAU EVALUATION]\n"
                            + json.dumps(dict(metric="official_reward", score=score)),
                        ),
                    ],
                )
                results.append(result)
            with (root / "results.jsonl").open("x") as stream:
                for result in results:
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            return results

    return TauEnv()
