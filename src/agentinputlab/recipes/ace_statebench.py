"""Portable STATE-Bench tool-plan feedback for native ACE fitting/online APIs.

This adapter scores a noninteractive plan on original training tasks. It does
not replace the benchmark's closed-loop heldout agent or its native judge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from agentinputlab.api import _state
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import canonical_json, to_jsonable

from .ace import ACEExample
from .ace_native_plan import (
    PLAN_INSTRUCTION,
    NativePlanFeedback,
    _save,
    _sha,
    native_process_scope,
    source_file_pins,
    verify_native_checkout,
)

SCHEMA = "agentinputlab.ace.STATE.native_feedback_input.v1"


def _provider():
    from workers.official_arenas.providers.statebench import StateBenchProvider

    return StateBenchProvider()


def _native_inputs(checkout, configuration, rows):
    from state_bench.domain import get_domain_config
    from state_bench.paths import domain_tasks_dir
    from state_bench.protocol import load_default_protocol, load_split_task_ids
    from state_bench.schemas import TaskDefinition

    protocol = load_default_protocol()
    if protocol.validate_prompt_hashes():
        raise ValidationError("STATE protocol prompt checksums differ")
    domains, tasks, files = (
        {},
        {},
        [
            checkout / "state_bench" / n
            for n in (
                "client.py",
                "domain.py",
                "env_loader.py",
                "orchestrator.py",
                "protocol.py",
                "scoring.py",
            )
        ],
    )
    for row in rows:
        protected = row["protected_checkpoint"]["protected"]
        name, task_id = protected["domain"], protected["task_id"]
        if name not in configuration["domains"] or task_id not in load_split_task_ids(
            name, "train", protocol.split_version
        ):
            raise ValidationError("STATE feedback must use original official training tasks")
        path = domain_tasks_dir(name) / (task_id + ".json")
        task = TaskDefinition.load(path)
        domain = get_domain_config(name)
        state = _state(row["state"])
        if (
            row["case_id"] != name + ":" + task_id
            or state.task_id != task_id
            or state.query != task.opening_message
            or protected["user_id"] != task.user_id
            or protected["protocol_id"] != protocol.protocol_id
            or protected["locked_dependency_hashes"] != configuration["locked_dependency_hashes"]
        ):
            raise ValidationError("STATE fitting checkpoint, task or dependency binding differs")
        instruction = domain.agent_system_prompt.format(now=task.now, user_id=task.user_id)
        names = {t.get("name") or t.get("function", {}).get("name") for t in domain.tool_schemas}
        if state.instruction != instruction or not names or names != set(state.tool_names):
            raise ValidationError("STATE visible policy or native tool catalog differs")
        domains[row["case_id"]], tasks[row["case_id"]] = domain, task
        files.append(path)
        files.extend(p for p in domain.prompts_dir.rglob("*") if p.is_file())
    return domains, tasks, source_file_pins(checkout, files)


def build_ace_statebench_feedback_input(
    *,
    checkout,
    captures: Sequence[Mapping],
    native_configuration: Mapping,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    heldout_ids: Sequence[str],
    target_model: Mapping,
    environment_model: Mapping,
    output_path,
):
    """Freeze existing official captures without a model call or generated labels.

    IDs are native checkpoint IDs (``domain:task_id``). The split is checked
    against the pinned protocol, not inferred from the caller's labels.
    """
    provider = _provider()
    root = verify_native_checkout(checkout, revision=provider.source_revision)
    configuration = provider._validate_config(to_jsonable(native_configuration))
    if configuration["split"] != "train":
        raise ValidationError("STATE ACE fitting requires the official train split")
    ids = [*train_ids, *validation_ids]
    if (
        not train_ids
        or not validation_ids
        or len(ids) != len(set(ids))
        or set(ids) & set(heldout_ids)
        or any(not isinstance(x, str) or not x for x in ids)
    ):
        raise ValidationError("STATE ACE groups must be disjoint nonempty fitting IDs")
    selected = {r["checkpoint_id"]: r for r in captures if r["checkpoint_id"] in ids}
    if len(selected) != len(ids) or len([r for r in captures if r["checkpoint_id"] in ids]) != len(
        ids
    ):
        raise ValidationError("STATE capture IDs are missing or duplicated")
    rows = []
    for role, group in (("train", train_ids), ("validation", validation_ids)):
        for key in group:
            capture = to_jsonable(selected[key])
            state = _state(capture["visible_state"])
            rows.append(
                dict(
                    case_id=key,
                    role=role,
                    official_split="train",
                    state=to_jsonable(state),
                    state_hash=state.state_hash,
                    protected_checkpoint=capture,
                )
            )
    with native_process_scope(root, package="state_bench"):
        _, _, pins = _native_inputs(root, configuration, rows)
    if (
        not isinstance(target_model, Mapping)
        or not isinstance(environment_model, Mapping)
        or target_model.get("model_id") != configuration.get("agent_model")
        or environment_model.get("model_id") != configuration.get("evaluator_model")
    ):
        raise ValidationError("STATE target/judge model fingerprints must match the deployment")
    value = dict(
        schema=SCHEMA,
        benchmark=provider.benchmark_id,
        native_source_revision=provider.source_revision,
        source_files=pins,
        native_configuration=configuration,
        fitting_rows=rows,
        heldout_ids=list(heldout_ids),
        source_official_split="train",
        target_model=to_jsonable(target_model),
        environment_model=to_jsonable(environment_model),
        builder_model_calls=0,
        ToolTalk_not_used=True,
    )
    path = Path(output_path).expanduser().absolute()
    if path.resolve().is_relative_to(root):
        raise ValidationError("STATE feedback input must be outside the clean checkout")
    if path.exists() or path.is_symlink():
        raise ValidationError("STATE feedback input must be a new file")
    _save(path, value)
    return dict(path=str(path), sha256=_sha(path))


class ACEStateBenchFeedback(NativePlanFeedback):
    """Hash-pinned, split-isolated native STATE tools and judge feedback."""

    processor_id = "agentinputlab.ace.STATE.native_tool_plan_feedback.v1"
    metric = "task_completion_pass"

    def __init__(
        self, *, checkout, source_input, output_root, environment_base_url=None, mode="offline"
    ):
        self._initialize(
            source_input=source_input, checkout=checkout, output_root=output_root, mode=mode
        )
        self.provider = _provider()
        if (
            self.source.get("schema") != SCHEMA
            or self.source.get("benchmark") != self.provider.benchmark_id
        ):
            raise ValidationError("STATE ACE feedback input schema or benchmark differs")
        self.checkout = verify_native_checkout(checkout, revision=self.provider.source_revision)
        if self.source["native_source_revision"] != self.provider.source_revision:
            raise ValidationError("STATE ACE feedback source revision differs")
        self._verify_source()
        self.configuration = self.provider._validate_config(self.source["native_configuration"])
        if (
            self.configuration["split"] != "train"
            or self.source["source_official_split"] != "train"
        ):
            raise ValidationError("STATE ACE feedback excludes official test inputs")
        with native_process_scope(self.checkout, package="state_bench"):
            self.domains, self.tasks, actual = _native_inputs(
                self.checkout, self.configuration, list(self.rows.values())
            )
        if actual != self.source["source_files"]:
            raise ValidationError("STATE task, source or prompt file binding differs")
        if (
            self.source["target_model"].get("model_id") != self.configuration["agent_model"]
            or self.source["environment_model"].get("model_id")
            != self.configuration["evaluator_model"]
        ):
            raise ValidationError("STATE feedback target/judge deployment identity differs")
        self.environment_base_url = environment_base_url or self.configuration["evaluator_base_url"]
        url = (
            urlsplit(self.environment_base_url)
            if isinstance(self.environment_base_url, str)
            else None
        )
        if (
            url is None
            or url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValidationError("STATE feedback requires an explicit judge HTTP deployment")
        self.tool_names = {k: set(s.tool_names) for k, s in self.states.items()}
        self._create_output()

    def fingerprint(self):
        self._verify_source()
        return dict(
            processor_id=self.processor_id,
            mode=self.mode,
            source_input=self.source_input,
            source_files=self.source["source_files"],
            source_revision=self.provider.source_revision,
            scorer_id=self.provider.scorer_id,
            scorer_revision=self.provider.scorer_revision,
            target_model=self.source["target_model"],
            environment_model=self.source["environment_model"],
            environment_base_url=self.environment_base_url,
            provider_source_sha256=_sha(
                __import__(self.provider.__module__, fromlist=["__file__"]).__file__
            ),
            task_adapter_source_sha256=_sha(__file__),
            feedback_metric=self.metric,
            source_official_split="train",
            groups={
                role: [
                    dict(case_id=k, state_hash=r["state_hash"])
                    for k, r in self.rows.items()
                    if r["role"] == role
                ]
                for role in ("train", "validation")
            },
            noninteractive_text_plan_NOT_native_closed_loop_agent=True,
            heldout_feedback_NOT_used=True,
            transductive_NOT_independent_heldout=self.mode == "online",
            native_user_simulator_calls=0,
            judge_costs_separate=True,
            automatic_retry=False,
        )

    def _example(self, key, split):
        state = self.states[key]
        visible = dict(
            query=state.query,
            instruction=state.instruction,
            messages=state.messages,
            context=state.context,
            tool_names=state.tool_names,
            native_function_tools=to_jsonable(self.domains[key].tool_schemas),
        )
        return ACEExample(
            example_id=key,
            input=dict(question=PLAN_INSTRUCTION + canonical_json(visible)),
            expected_output=key,
            split=split,
            metadata=dict(
                source_state_hash=state.state_hash,
                task_processor=self.processor_id,
                evaluator_target_is_opaque_case_id=True,
            ),
        )

    def _native_score(self, key, plan, folder):
        with native_process_scope(self.checkout, package="state_bench"):
            from state_bench.env_loader import load_task_environment
            from state_bench.protocol import load_default_protocol
            from state_bench.schemas import StateDiff
            from state_bench.scoring import (
                TaskRequirementsJudge,
                combine_task_completion,
                evaluate_state_requirements,
            )

            import workers.official_arenas.providers.statebench as module

            task, domain = self.tasks[key], self.domains[key]
            data, _ = load_task_environment(domain, task)
            env = domain.environment_class(data.deep_copy(), now=task.now)
            before = env.get_full_snapshot()
            calls = []
            conversation = [dict(role="user", content=self.states[key].query)]
            for action in plan["actions"]:
                result = env.tool_handlers[action["name"]](action["arguments"])
                call = dict(
                    name=action["name"], arguments=action["arguments"], result=to_jsonable(result)
                )
                calls.append(call)
                conversation.append(dict(role="assistant", content="", tool_calls=[call]))
                conversation.append(
                    dict(role="tool", name=action["name"], content=canonical_json(result))
                )
            conversation.append(dict(role="assistant", content=plan["response"]))
            diff = StateDiff.compute(before, env.get_full_snapshot())
            state_result = evaluate_state_requirements(task, diff)
            _save(
                folder / "before-native-judge.json",
                dict(
                    case_id=key,
                    source_state_hash=self.states[key].state_hash,
                    plan=plan,
                    native_tool_calls=calls,
                    conversation=conversation,
                    state_diff=diff.to_dict(),
                    state_requirements_score=state_result.score,
                    evaluated_agent_model_calls=0,
                    noninteractive_plan_execution_NOT_native_agent_rollout=True,
                ),
            )
            cfg = self.configuration
            model = cfg["evaluator_model"]
            client = module._openai_compatible_client(
                model=model,
                base_url=self.environment_base_url,
                api_key_var=cfg["evaluator_api_key_var"],
                single_attempt=cfg.get("request_attempts") == 1,
                extra_body=module.native_extra_body(cfg, "environment_extra_body"),
                **module.native_timeout_options(cfg, "environment_timeout_seconds"),
            )
            module._install_statebench_judge_json_compat(client)
            protocol = load_default_protocol()
            effort = module.normalize_qwen38_reasoning_effort(
                model, cfg["score_reasoning_effort"] or protocol.judge_reasoning_effort
            )
            judge = TaskRequirementsJudge(
                client=client,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=effort,
            )
            try:
                task_result = judge.evaluate(task, conversation, calls, diff)
            finally:
                client._client.close()
            if task_result is None:
                raise ValidationError("original native STATE task judge did not complete")
            score = combine_task_completion(state_result, task_result)
            if score is None:
                raise ValidationError("original native STATE completion is absent")
            value = dict(
                status="native_STATE_tool_plan_scored",
                case_id=key,
                native_metrics=dict(
                    task_completion_pass=float(score),
                    state_requirements_met=float(state_result.score),
                    task_requirements_met=float(task_result.score),
                ),
                evaluator_agent_model_calls=0,
                native_user_simulator_calls=0,
                task_judge_model=model,
                original_judge_controls=True,
            )
            _save(folder / "native-score.json", value)
            return value


def create_statebench_feedback(**configuration):
    """Trusted startup factory for ``python -m workers.ace.driver``."""
    return ACEStateBenchFeedback(**configuration)
