"""SkillOpt fitting against complete native STATE episodes, not text plans."""

from __future__ import annotations

import contextlib
import json
import random
from dataclasses import replace
from pathlib import Path

from agentinputlab.api import _state
from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ContextBlock, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.recipes.ace_native_plan import (
    _save,
    _sha,
    native_process_scope,
    verify_native_checkout,
    verify_source_files,
)
from agentinputlab.recipes.ace_statebench import _native_inputs, _provider
from workers.official_arenas.protocol import EXECUTABLE_PAYLOAD_PROTOCOL


@contextlib.contextmanager
def _account_native_calls(folder, config):
    """Record costs at the SDK boundary without changing requests or responses.

    Called under native_process_scope's process lock. STATE constructs its
    agent, simulator and judge clients in that fixed order. No generated text
    or protected judge prompt is sent to SkillOpt's reflection by this hook.
    """
    from workers.official_arenas.providers import statebench as module

    original, clients, events = module._openai_compatible_client, [], []

    def create(**kwargs):
        roles = ("agent", "user", "judge")
        if len(clients) >= len(roles):
            raise ValidationError("STATE native client roles changed")
        role = roles[len(clients)]
        if kwargs["model"] != config["agent_model" if role == "agent" else "evaluator_model"]:
            raise ValidationError("STATE native role model changed")
        client = original(**kwargs)
        clients.append(client)
        generate = client._client.responses.create

        def measured(**request):
            event = dict(role=role, status="started", usage=None, automatic_retry=False)
            events.append(event)
            try:
                response = generate(**request)
                usage = getattr(response, "usage", None)
                event.update(
                    status="returned", usage=usage.model_dump() if usage is not None else None
                )
                return response
            except Exception as exc:
                event.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                _save(folder / f"native-call-{len(events):04d}.json", event)

        client._client.responses.create = measured
        return client

    module._openai_compatible_client = create
    try:
        yield events
    finally:
        module._openai_compatible_client = original
        for client in clients:
            client._client.close()


def create_statebench_environment(*, cfg, configuration):
    try:
        from skillopt.envs.base import EnvAdapter
    except ImportError as exc:
        raise ExternalDependencyError(
            "mount the pinned SkillOpt checkout before creating its STATE environment"
        ) from exc
    pin = configuration["source_input"]
    source = load_pinned_json_mapping(
        pin["path"], expected_sha256=pin["sha256"], max_bytes=8 * 1024 * 1024
    )
    provider = _provider()
    checkout = verify_native_checkout(configuration["checkout"], revision=provider.source_revision)
    if (
        source.get("benchmark") != provider.benchmark_id
        or source.get("native_source_revision") != provider.source_revision
    ):
        raise ValidationError("STATE source identity differs")
    native = provider._validate_config(source["native_configuration"])
    if native["split"] != "train":
        raise ValidationError("SkillOpt STATE fitting requires the official training split")
    source_rows = source.get("fitting_rows", [])
    if (
        not isinstance(source_rows, list)
        or not source_rows
        or any(
            not isinstance(r, dict) or not isinstance(r.get("case_id"), str) for r in source_rows
        )
    ):
        raise ValidationError("STATE requires frozen fitting rows")
    rows = {r["case_id"]: r for r in source_rows}
    heldout = set(source.get("heldout_ids", ())) | {
        r["case_id"] for r in source.get("heldout_rows", ())
    }
    if (
        len(rows) != len(source_rows)
        or set(rows) & heldout
        or {r["role"] for r in rows.values()} != {"train", "validation"}
    ):
        raise ValidationError("STATE fitting and heldout groups must be unique and disjoint")
    states = {k: _state(r["state"]) for k, r in rows.items()}
    if len({s.state_hash for s in states.values()}) != len(states):
        raise ValidationError("STATE fitting states must be disjoint")
    cap, seed = configuration["per_call_output_tokens"], configuration["seed"]
    for key, row in rows.items():
        state, original = states[key], row["protected_checkpoint"]
        if (
            row.get("official_split") != "train"
            or row["state_hash"] != state.state_hash
            or original["checkpoint_id"] != key
            or _state(original["visible_state"]).state_hash != state.state_hash
            or original["protected"]["locked_dependency_hashes"]
            != native["locked_dependency_hashes"]
        ):
            raise ValidationError("STATE captured task, state or dependencies differ")
        budget = native["budget_override"]
        if cap != budget["max_output_tokens_per_call"] or any(
            budget[name] != to_jsonable(state.budget)[name]
            for name in (
                "max_context_tokens",
                "max_model_calls",
                "max_output_tokens",
                "max_retrieval_calls",
                "max_tool_calls",
            )
        ):
            raise ValidationError("STATE captured budget differs from fitting configuration")

    def verify():
        load_pinned_json_mapping(
            pin["path"], expected_sha256=pin["sha256"], max_bytes=8 * 1024 * 1024
        )
        verify_native_checkout(checkout, revision=provider.source_revision)
        if "source_files" in source:
            verify_source_files(checkout, source["source_files"])
        else:
            for item in source["native_source_files"].values():
                if _sha(item["path"]) != item["sha256"]:
                    raise ValidationError("STATE original source file changed")

    verify()
    with native_process_scope(checkout, package="state_bench"):
        _, _, actual_sources = _native_inputs(checkout, native, source_rows)
    if "source_files" in source and actual_sources != source["source_files"]:
        raise ValidationError("STATE source task/prompt files differ")
    for role, model_key, lock_key in (
        ("backend", "target_model", "evaluated_model"),
        ("environment_backend", "environment_model", "environment_model"),
    ):
        backend = configuration[role]
        identity = backend["deployment_identity"]
        expected = native["locked_dependency_hashes"][lock_key]
        if backend["model_id"] != source[model_key]["model_id"] or expected not in (
            content_hash(identity),
            identity.get("revision"),
        ):
            raise ValidationError("STATE target/environment deployment binding differs")
        if (
            "original_model_json_hashes" in source[model_key]
            and identity.get("model_json_hashes") != source[model_key]["original_model_json_hashes"]
        ):
            raise ValidationError("STATE original model asset identity differs")
    # Only transport is relocated. Dataset, budget, native judge and method
    # settings remain those in the pinned source configuration.
    native = dict(native, request_attempts=1)
    for field, prefix, key_prefix in (
        ("backend", "agent", "agent"),
        ("environment_backend", "environment", "evaluator"),
    ):
        backend = configuration[field]
        native[key_prefix + "_base_url"] = backend["base_url"]
        native[key_prefix + "_api_key_var"] = backend.get("api_key_env", "OPENAI_API_KEY")
        for option, native_option in (
            ("extra_body", prefix + "_extra_body"),
            ("timeout_seconds", prefix + "_timeout_seconds"),
        ):
            if option in backend:
                native[native_option] = to_jsonable(backend[option])
    native = provider._validate_config(native)
    groups = {
        role: [k for k, r in rows.items() if r["role"] == role] for role in ("train", "validation")
    }
    byid = {content_hash(k): k for k in rows}

    def visible(key):
        return dict(
            id=content_hash(key), question=states[key].query, task_type=provider.benchmark_id
        )

    class StateEnvAdapter(EnvAdapter):
        def __init__(self):
            self.analyst_workers = cfg.get("analyst_workers", 1)
            self.failure_only = cfg.get("failure_only", False)
            self.minibatch_size = cfg.get("minibatch_size", 1)
            self.edit_budget = cfg.get("edit_budget", 4)

        def fingerprint(self):
            return dict(
                implementation="agentinputlab.skillopt.native_STATE_environment.v1",
                source_input=to_jsonable(pin),
                source_files=actual_sources,
                source_revision=provider.source_revision,
                target_model=source["target_model"],
                environment_model=source["environment_model"],
                groups={
                    role: [dict(case_id=k, state_hash=states[k].state_hash) for k in keys]
                    for role, keys in groups.items()
                },
                metric="task_completion_pass",
                original_native_roles_tools_and_scoring=True,
                official_training_partition_only=True,
                heldout_not_exposed=True,
                per_call_agent_output_tokens=cap,
                seed=seed,
                native_STATE_wire_seed_omission_preserved=True,
                simulator_judge_costs_separate=True,
                automatic_retry=False,
            )

        def _batch(self, role, count, batch_seed):
            if type(count) is not int or not 1 <= count <= len(groups[role]):
                raise ValidationError("STATE batch exceeds the declared fitting group")
            return [visible(k) for k in random.Random(batch_seed).sample(groups[role], count)]

        def build_train_env(self, batch_size, seed, **kwargs):
            return self._batch("train", batch_size, seed)

        def build_eval_env(self, env_num, split, seed, **kwargs):
            if split not in ("val", "validation", "valid_seen"):
                raise ValidationError("STATE fit exposes validation only")
            return self._batch("validation", env_num, seed)

        def get_task_types(self):
            return [provider.benchmark_id]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            if not isinstance(skill_content, str) or not skill_content.strip():
                raise ValidationError("STATE fitting skill must be nonempty")
            if any(
                kwargs.get(k)
                for k in (
                    "diagnostic_mode",
                    "diagnostic_instruction",
                    "diagnostic_trace_context_by_id",
                )
            ):
                raise ValidationError("Extra diagnostic STATE episodes are not admitted")
            verify()
            items = list(env_manager)
            if (
                any(not isinstance(i, dict) or i.get("id") not in byid for i in items)
                or len({i["id"] for i in items}) != len(items)
                or any(i != visible(byid[i["id"]]) for i in items)
            ):
                raise ValidationError("STATE fitting case is undeclared, duplicated or changed")
            root = Path(out_dir)
            if root.resolve().is_relative_to(checkout):
                raise ValidationError("STATE fitting output must be outside the native checkout")
            root.mkdir(parents=True, exist_ok=True)
            results = []
            for item in items:
                key = byid[item["id"]]
                state = states[key]
                patched = replace(
                    state,
                    context=(
                        *state.context,
                        ContextBlock(
                            text="SKILLOPT CURRENT FITTING SKILL\n" + skill_content,
                            block_id="skillopt-fit-skill:" + content_hash(skill_content),
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
                folder = root / "predictions" / item["id"]
                folder.mkdir(parents=True, exist_ok=False)
                events = []
                try:
                    with (
                        native_process_scope(checkout, package="state_bench"),
                        _account_native_calls(folder, native) as events,
                    ):
                        outcome = provider.run_arm(
                            native,
                            replay_record=rows[key]["protected_checkpoint"],
                            arm_payload=payload,
                            seed=seed,
                            role="treatment",
                            work_dir=folder,
                        )
                    score = outcome.metrics.get("task_completion_pass")
                    if type(score) not in (float, int) or score not in (0.0, 1.0):
                        raise ValidationError("STATE native completion must be zero or one")
                    if (
                        outcome.usage["model_calls"] > state.budget.max_model_calls
                        or outcome.usage["output_tokens"] > state.budget.max_output_tokens
                    ):
                        raise ValidationError(
                            "STATE native usage exceeded the original agent budget"
                        )
                except Exception as exc:
                    _save(
                        folder / "consumed-execution-error.json",
                        dict(
                            error_type=type(exc).__name__,
                            physical_model_requests=len(events),
                            automatic_retry=False,
                        ),
                    )
                    raise
                trace = json.dumps(
                    to_jsonable(outcome.trace.get("conversation", [])),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                result = dict(
                    id=item["id"],
                    source_case_id=key,
                    question=state.query,
                    task_type=provider.benchmark_id,
                    hard=int(score),
                    soft=float(score),
                    response=trace,
                    agent_ok=True,
                    n_turns=outcome.usage["model_calls"],
                    fail_reason="" if score else "native STATE task_completion_pass=0",
                    native_metrics=to_jsonable(outcome.metrics),
                    native_agent_usage=to_jsonable(outcome.usage),
                    source_state_hash=state.state_hash,
                    patched_state_hash=patched.state_hash,
                    physical_model_requests=len(events),
                    native_role_calls=events,
                    native_roles=["agent", "user", "judge"],
                    original_native_tools_simulator_judge_and_scorer=True,
                )
                _save(folder / "result.json", result)
                _save(
                    folder / "conversation.json",
                    [
                        dict(type="message", turn=1, content=trace),
                        dict(
                            role="system",
                            content="[NATIVE STATE EVALUATION]\n"
                            + json.dumps(dict(metric="task_completion_pass", score=float(score))),
                        ),
                    ],
                )
                results.append(result)
            with (root / "results.jsonl").open("x") as stream:
                for row in results:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            return results

    return StateEnvAdapter()
