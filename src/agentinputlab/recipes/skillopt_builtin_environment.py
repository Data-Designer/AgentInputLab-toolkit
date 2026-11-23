"""File-configured native SkillOpt environments, without caller Python code.

Only preparation is shared with ACE's retrieval binding. SkillOpt still owns
its original query rollout, native metric and six-stage optimization loop.
"""

from __future__ import annotations

import math

from agentinputlab.configuration import RunConfig, _known, _object
from agentinputlab.contracts import ValidationError


def validate_environment(configuration):
    from agentinputlab.fit_configuration import _path, _reference

    value = _object(configuration, "SkillOpt environment")
    allowed = {
        "kind",
        "source_input",
        "backend",
        "per_call_output_tokens",
        "seed",
        "stop_policy",
        "partition_label",
    }
    if value.get("kind") in ("statebench", "toolsandbox", "tau"):
        allowed |= {"checkout", "environment_backend"}
    if value.get("kind") in ("toolsandbox", "tau"):
        allowed |= {"worker_python", "episode_timeout_seconds"}
    _known(value, allowed, "SkillOpt environment")
    if value.get("kind") not in ("retrieval", "bfcl", "statebench", "toolsandbox", "tau"):
        raise ValidationError(
            "built-in SkillOpt environment kind must be retrieval, bfcl, statebench, toolsandbox or tau"
        )
    _reference(value.get("source_input"), "environment.source_input")
    for name, lower in (("per_call_output_tokens", 1), ("seed", 0)):
        if type(value.get(name)) is not int or value[name] < lower:
            raise ValidationError(f"environment.{name} must be an explicit integer >= {lower}")
    _path(value.get("partition_label"), "environment.partition_label")
    if value.get("stop_policy") not in ("provider", "gpt_oss_final_channel"):
        raise ValidationError("SkillOpt environment stop_policy must be explicit")
    backend = _object(value.get("backend"), "environment.backend")
    if "stop_policy" in backend:
        raise ValidationError("use only environment.stop_policy for SkillOpt native compatibility")
    built = RunConfig(release_id="baseline.no_op", backend=backend)._backend()
    if built is None:
        raise ValidationError("SkillOpt target backend must be configured")
    if value["stop_policy"] == "gpt_oss_final_channel":
        from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend

        built = GPTOSSFinalChannelStopBackend(built)
    if value["kind"] in ("statebench", "toolsandbox", "tau"):
        _path(value.get("checkout"), "environment.checkout")
        if value["stop_policy"] != "provider":
            raise ValidationError(
                "native arenas use their response parser, not a text stop-policy adapter"
            )
        if value["kind"] == "tau":
            environment = _object(value.get("environment_backend"), "environment_backend")
            if RunConfig(release_id="baseline.no_op", backend=environment)._backend() is None:
                raise ValidationError("Tau needs an explicit environment backend")
        else:
            from agentinputlab.arena_configuration import NativeArenaConfig

            NativeArenaConfig(
                value["kind"],
                value["checkout"],
                environment_backend=value.get("environment_backend", {}),
            )
    if value["kind"] in ("toolsandbox", "tau"):
        _path(value.get("worker_python"), "environment.worker_python")
        timeout = value.get("episode_timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValidationError("environment.episode_timeout_seconds must be finite and positive")
    return value, built


def create(*, cfg, configuration):
    from agentinputlab.recipes.ace_retrieval import create_retrieval_feedback
    from agentinputlab.recipes.skillopt_retrieval import create_skillopt_retrieval_adapter

    value, backend = validate_environment(configuration)
    if value["kind"] == "tau":
        from agentinputlab.recipes.skillopt_tau import create_tau_environment

        return create_tau_environment(cfg=cfg, configuration=value)
    if value["kind"] == "toolsandbox":
        from agentinputlab.recipes.skillopt_toolsandbox import create_toolsandbox_environment

        return create_toolsandbox_environment(cfg=cfg, configuration=value)
    if value["kind"] == "statebench":
        from agentinputlab.recipes.skillopt_statebench import create_statebench_environment

        return create_statebench_environment(cfg=cfg, configuration=value)
    if value["kind"] == "bfcl":
        from agentinputlab.recipes.skillopt_bfcl import create_bfcl_environment

        return create_bfcl_environment(cfg=cfg, configuration=value, target_backend=backend)
    source = create_retrieval_feedback(source_input=value["source_input"], output_root=None)
    return create_skillopt_retrieval_adapter(
        adapter=source.adapter,
        retriever=source.retriever,
        target_backend=backend,
        train=source.train,
        validation=source.validation,
        cfg=cfg,
        metric=source.metric,
        top_k=source.top_k,
        allow_json_fence=source.allow_json_fence,
        per_call_output_tokens=value["per_call_output_tokens"],
        seed=value["seed"],
        partition_label=value["partition_label"],
    )
