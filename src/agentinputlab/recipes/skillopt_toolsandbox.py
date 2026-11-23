"""SkillOpt's full native ToolSandbox episodes in an isolated installed worker."""

from __future__ import annotations

import json
import math
import os
import random
import select
import signal
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

from agentinputlab.api import _state
from agentinputlab.arena_configuration import _implementation_digest
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
from agentinputlab.recipes.ace_toolsandbox import _native_inputs, _provider
from workers.official_arenas.protocol import EXECUTABLE_PAYLOAD_PROTOCOL


def _source(configuration):
    pin = configuration["source_input"]
    source = load_pinned_json_mapping(
        pin["path"], expected_sha256=pin["sha256"], max_bytes=8 * 1024 * 1024
    )
    provider = _provider()
    if (
        source.get("benchmark") != provider.benchmark_id
        or source.get("native_source_revision") != provider.source_revision
    ):
        raise ValidationError("ToolSandbox fitting source identity differs")
    checkout = verify_native_checkout(configuration["checkout"], revision=provider.source_revision)
    if "source_files" in source:
        verify_source_files(checkout, source["source_files"])
    else:
        for pin in source["native_source_files"].values():
            if _sha(pin["path"]) != pin["sha256"]:
                raise ValidationError("Original ToolSandbox fitting source changed")
    config = provider._validate_config(source["native_configuration"])
    values = source.get("fitting_rows", [])
    if not isinstance(values, list) or not values:
        raise ValidationError("ToolSandbox fitting rows must be explicit")
    rows = {r["case_id"]: r for r in values}
    heldout = set(source.get("heldout_ids", ())) | {
        r["case_id"] for r in source.get("heldout_rows", ())
    }
    if (
        len(rows) != len(values)
        or set(rows) & heldout
        or {r["role"] for r in values} != {"train", "validation"}
    ):
        raise ValidationError("ToolSandbox fitting groups must be unique and disjoint")
    states = {k: _state(r["state"]) for k, r in rows.items()}
    if len({s.state_hash for s in states.values()}) != len(states):
        raise ValidationError("ToolSandbox fitting states must be disjoint")
    for key, row in rows.items():
        state = states[key]
        capture = row["protected_checkpoint"]
        protected = capture["protected"]
        if (
            not row.get("caller_declared_fit_partition_NOT_official_split")
            or row["state_hash"] != state.state_hash
            or capture["checkpoint_id"] != key
            or protected["scenario_name"] != key
            or _state(capture["visible_state"]).state_hash != state.state_hash
            or protected["starting_context"].get("interactive_console") is not None
            or content_hash(protected["starting_context"]) != protected["starting_context_hash"]
            or protected["locked_dependency_hashes"] != config["locked_dependency_hashes"]
        ):
            raise ValidationError(
                "ToolSandbox fitting state, protected context or dependencies differ"
            )
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
            raise ValidationError("ToolSandbox fitting budget differs from captured budget")
    for field, model_key, lock in (
        ("backend", "target_model", "evaluated_model"),
        ("environment_backend", "environment_model", "environment_model"),
    ):
        backend = configuration[field]
        identity = backend["deployment_identity"]
        model = source[model_key]
        if backend["model_id"] != model["model_id"] or config["locked_dependency_hashes"][
            lock
        ] not in (content_hash(identity), identity.get("revision")):
            raise ValidationError("ToolSandbox role deployment identity differs")
        if (
            "original_model_json_hashes" in model
            and identity.get("model_json_hashes") != model["original_model_json_hashes"]
        ):
            raise ValidationError("ToolSandbox original model assets differ")
    return source, rows, states, provider, checkout, config


def _episode(configuration, case_id, skill, output):
    source, rows, states, provider, checkout, config = _source(configuration)
    if case_id not in rows or not isinstance(skill, str) or not skill.strip():
        raise ValidationError("ToolSandbox episode needs a declared case and nonempty skill")
    with native_process_scope(checkout, package="tool_sandbox"):
        _, actual = _native_inputs(checkout, config, list(rows.values()), provider)
        if "source_files" in source and actual != source["source_files"]:
            raise ValidationError("ToolSandbox native sources differ")
        config = dict(config)
        for field, prefix, extra in (
            ("backend", "agent", "agent"),
            ("environment_backend", "user", "environment"),
        ):
            backend = configuration[field]
            config[prefix + "_base_url"] = backend["base_url"]
            config[prefix + "_key_env"] = backend.get("api_key_env", "OPENAI_API_KEY")
            for option in ("extra_body", "timeout_seconds"):
                if option in backend:
                    config[extra + "_" + option] = to_jsonable(backend[option])
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
        result = provider.run_arm(
            config,
            replay_record=rows[case_id]["protected_checkpoint"],
            arm_payload=payload,
            seed=configuration["seed"],
            role="treatment",
            work_dir=output,
        )
    score = result.metrics.get("official_similarity")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValidationError("ToolSandbox native similarity is invalid")
    if (
        result.usage["model_calls"] > state.budget.max_model_calls
        or result.usage["output_tokens"] > state.budget.max_output_tokens
    ):
        raise ValidationError("ToolSandbox native agent budget exceeded")
    return dict(
        status="scored",
        metrics=to_jsonable(result.metrics),
        native_usage=to_jsonable(result.usage),
        native_trace=to_jsonable(result.trace),
        source_state_hash=state.state_hash,
        patched_state_hash=patched.state_hash,
        protected_starting_context_hash=rows[case_id]["protected_checkpoint"]["protected"][
            "starting_context_hash"
        ],
        automatic_retry=False,
    )


def _worker(path):
    job = load_pinned_json_mapping(path, expected_sha256=_sha(path), max_bytes=8 * 1024 * 1024)
    output = Path(job["output"])
    try:
        episode = _episode
        if job["configuration"].get("kind") == "tau":
            from agentinputlab.recipes.skillopt_tau import _episode as episode
        result = episode(job["configuration"], job["case_id"], job["skill"], output)
    except Exception as exc:
        _save(
            output / "episode-terminal.json",
            dict(status="failed", error_type=type(exc).__name__, automatic_retry=False),
        )
        raise
    _save(output / "episode-terminal.json", result)
    return 0


def _watch_parent(read_fd):
    """Exit this worker when its launcher's private lifetime pipe closes."""
    # Only the parent owns the write end; no PID lookup or process-group signal
    # is needed, including when the trainer is terminated by its fit timeout.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if select.select([read_fd], [], [], 0)[0]:
        os.close(read_fd)
        raise SystemExit("ToolSandbox trainer exited before worker startup")

    def watch():
        try:
            os.read(read_fd, 1)
        finally:
            os.close(read_fd)
            os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, name="skillopt-parent-lifetime", daemon=True).start()


def _launch(configuration, case_id, skill, output):
    if os.name != "posix":
        raise ExternalDependencyError("Isolated ToolSandbox fitting requires a POSIX worker")
    python = configuration["worker_python"]
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ExternalDependencyError(
            "Install the same toolkit wheel in the configured ToolSandbox worker Python"
        )
    path = output / "episode-job.json"
    _save(path, dict(configuration=configuration, case_id=case_id, skill=skill, output=str(output)))
    command = [
        python,
        "-c",
        "import sys;from pathlib import Path;from agentinputlab.arena_configuration import _implementation_digest\n"
        "if _implementation_digest()!=sys.argv[2]: raise SystemExit('worker toolkit implementation differs; install the same wheel')\n"
        "from agentinputlab.recipes.skillopt_toolsandbox import _worker, _watch_parent\n"
        "_watch_parent(int(sys.argv[3]))\n"
        "raise SystemExit(_worker(Path(sys.argv[1])))",
        str(path),
        _implementation_digest(),
    ]
    env = dict(
        os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    read_fd, write_fd = os.pipe()
    with (
        os.fdopen(read_fd, "rb", buffering=0) as lifetime_reader,
        os.fdopen(write_fd, "wb", buffering=0),
        (output / "worker.log").open("xb") as log,
    ):
        child = subprocess.Popen(
            [*command, str(lifetime_reader.fileno())],
            cwd=output,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(lifetime_reader.fileno(),),
        )
        timed_out = False
        try:
            code = child.wait(timeout=configuration["episode_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _save(
                    output / "worker-exit.json",
                    dict(pid=child.pid, status="termination_pending", automatic_retry=False),
                )
                raise ExternalDependencyError(
                    "Owned ToolSandbox worker has not exited; inspect its log, do not retry"
                ) from None
    _save(
        output / "worker-exit.json",
        dict(returncode=code, execution_timeout=timed_out, automatic_retry=False),
    )
    path = output / "episode-terminal.json"
    if code or not path.is_file():
        raise ExternalDependencyError(
            "ToolSandbox episode failed; inspect worker.log and terminal, do not retry consumed attempts"
        )
    result = load_pinned_json_mapping(path, expected_sha256=_sha(path), max_bytes=32 * 1024 * 1024)
    if result.get("status") != "scored":
        raise ExternalDependencyError("ToolSandbox native scoring did not complete")
    return result


def create_toolsandbox_environment(*, cfg, configuration):
    from skillopt.envs.base import EnvAdapter

    source, rows, states, provider, checkout, _ = _source(configuration)
    groups = {r: [k for k, v in rows.items() if v["role"] == r] for r in ("train", "validation")}
    byid = {content_hash(k): k for k in rows}

    def visible(k):
        return dict(id=content_hash(k), question=states[k].query, task_type=provider.benchmark_id)

    class SandboxEnv(EnvAdapter):
        def __init__(self):
            self.analyst_workers = cfg.get("analyst_workers", 1)
            self.failure_only = cfg.get("failure_only", False)
            self.minibatch_size = cfg.get("minibatch_size", 1)
            self.edit_budget = cfg.get("edit_budget", 4)

        def fingerprint(self):
            return dict(
                implementation="agentinputlab.skillopt.native_ToolSandbox_environment.v1",
                source_input=configuration["source_input"],
                source_revision=provider.source_revision,
                groups={
                    r: [dict(case_id=k, state_hash=states[k].state_hash) for k in keys]
                    for r, keys in groups.items()
                },
                target_model=source["target_model"],
                environment_model=source["environment_model"],
                partition_label=configuration["partition_label"],
                caller_declared_fit_partition_NOT_official_split=True,
                heldout_not_exposed=True,
                native_tools_user_and_scorer_unchanged=True,
                automatic_retry=False,
            )

        def _batch(self, role, count, seed):
            if type(count) is not int or not 1 <= count <= len(groups[role]):
                raise ValidationError("ToolSandbox batch exceeds declared fitting group")
            return [visible(k) for k in random.Random(seed).sample(groups[role], count)]

        def build_train_env(self, batch_size, seed, **kw):
            return self._batch("train", batch_size, seed)

        def build_eval_env(self, env_num, split, seed, **kw):
            if split not in ("val", "validation", "valid_seen"):
                raise ValidationError("ToolSandbox fitting exposes validation only")
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
                raise ValidationError("Undeclared, duplicated or changed ToolSandbox fitting case")
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
                score = float(outcome["metrics"]["official_similarity"])
                native = outcome["native_trace"]
                if "ending_context" in native:
                    messages = native["ending_context"]["_dbs"]["SANDBOX"]
                elif native.get("termination_reason") == "generated_agent_response_invalid":
                    messages = []
                else:
                    raise ValidationError("ToolSandbox native trace has no visible conversation")

                def role(v):
                    return str(v).rsplit(".", 1)[-1].upper()

                trace = json.dumps(
                    [
                        m
                        for m in messages
                        if role(m.get("sender")) == "AGENT" or role(m.get("recipient")) == "AGENT"
                    ],
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
                    fail_reason="" if score >= 1 else "native ToolSandbox official_similarity<1",
                    native_metrics=outcome["metrics"],
                    native_agent_usage=outcome["native_usage"],
                    source_state_hash=outcome["source_state_hash"],
                    patched_state_hash=outcome["patched_state_hash"],
                    native_user_usage=native.get("user_simulator_usage"),
                    protected_starting_context_hash=outcome["protected_starting_context_hash"],
                    native_tools_user_and_scorer_unchanged=True,
                    evaluator_only_messages_not_forwarded_to_reflection=True,
                )
                _save(folder / "result.json", result)
                _save(
                    folder / "conversation.json",
                    [
                        dict(type="message", turn=1, content=trace),
                        dict(
                            role="system",
                            content="[NATIVE TOOLSANDBOX EVALUATION]\n"
                            + json.dumps(dict(metric="official_similarity", score=score)),
                        ),
                    ],
                )
                results.append(result)
            with (root / "results.jsonl").open("x") as stream:
                for row in results:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            return results

    return SandboxEnv()
