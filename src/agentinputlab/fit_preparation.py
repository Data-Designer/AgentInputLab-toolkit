"""File-based preparation of native fitting inputs; no model execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path

from agentinputlab.configuration import _known, _object, _unique
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.fit_configuration import FitConfig, _path, _reference, _save, _sha


def _read(path, limit):
    try:
        before = Path(path).lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError("preparation input must be a regular file, not a symlink")
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ValidationError("preparation input changed while opening")
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValidationError("ACE preparation input exceeds its size limit")
        return raw, json.loads(raw, object_pairs_hook=_unique)
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ValidationError("cannot read ACE preparation input as bounded JSON") from None


def _configuration(path):
    _, value = _read(path, 1024 * 1024)
    value = _object(value, "ACE preparation configuration")
    _known(
        value,
        {
            "fit",
            "feedback",
            "environment",
            "captures",
            "capture_bundle",
            "groups",
            "native_configuration",
            "native_configurations",
            "domain",
            "partition_label",
            "target_model",
            "environment_model",
            "timeout_seconds",
        },
        "ACE preparation",
    )
    base = Path(path).absolute().parent
    fit = _object(value.get("fit"), "fit template")
    skillopt = fit.get("release_id") == "recipe.skillopt.offline"
    if skillopt:
        _known(
            fit,
            {
                "release_id",
                "checkout",
                "python_executable",
                "controls",
                "native_config",
                "model_fingerprints",
                "environment_names",
            },
            "SkillOpt fit template",
        )
        if "feedback" in value:
            raise ValidationError("SkillOpt preparation uses environment, not ACE feedback")
        pin = _reference(fit.get("native_config"), "native_config")
        fit["native_config"] = dict(
            pin, path=os.path.abspath(base / Path(pin["path"]).expanduser())
        )
    else:
        _known(
            fit,
            {
                "release_id",
                "checkout",
                "python_executable",
                "timeout_seconds",
                "controls",
                "deployment",
            },
            "ACE fit template",
        )
        if fit.get("release_id") != "recipe.ace.offline" or "environment" in value:
            raise ValidationError("native feedback preparation requires recipe.ace.offline")
    for name in ("checkout", "python_executable"):
        fit[name] = os.path.abspath(base / Path(_path(fit.get(name), name)).expanduser())
    FitConfig(fit)._controls()
    binding_name = "environment" if skillopt else "feedback"
    feedback = _object(value.get(binding_name), binding_name + " preparation")
    if skillopt and feedback.get("kind") not in ("statebench", "toolsandbox", "tau"):
        raise ValidationError("SkillOpt capture preparation supports statebench, toolsandbox or tau")
    if skillopt and feedback.get("kind") in ("toolsandbox", "tau"):
        feedback["worker_python"] = os.path.abspath(
            base / Path(_path(feedback.get("worker_python"), "worker_python")).expanduser()
        )
    timeout = value.get("timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValidationError("preparation timeout_seconds must be finite and positive")
    if feedback.get("kind") in ("bfcl", "retrieval"):
        _known(value, {"fit", "feedback", "timeout_seconds"}, "asset feedback preparation")
        pin = _reference(feedback.get("source_input"), "feedback.source_input")
        pin["path"] = os.path.abspath(base / Path(pin["path"]).expanduser())
        feedback["source_input"] = pin
        FitConfig({"feedback": feedback})._ace_feedback()
        return to_jsonable(dict(value, fit=fit, feedback=feedback))
    # The real fit validator owns the supported factories, endpoint and env-name rules.
    feedback["checkout"] = os.path.abspath(
        base / Path(_path(feedback.get("checkout"), "feedback.checkout")).expanduser()
    )
    if "source_input" in feedback:
        raise ValidationError("prepare-fit builds source_input from the declared captures")
    pending = {**feedback, "source_input": {"path": "pending.json", "sha256": "0" * 64}}
    if skillopt:
        FitConfig.from_dict(dict(fit, environment=pending))
    else:
        FitConfig({"feedback": pending})._ace_feedback()
    groups = _object(value.get("groups"), "groups")
    if set(groups) != {"train", "validation", "heldout"}:
        raise ValidationError("groups must explicitly contain train, validation and heldout IDs")
    seen = set()
    for split, ids in groups.items():
        if (
            not isinstance(ids, list)
            or (split != "heldout" and not ids)
            or any(not isinstance(k, str) or not k.strip() for k in ids)
            or len(set(ids)) != len(ids)
            or seen.intersection(ids)
        ):
            raise ValidationError("preparation groups must be explicit, unique and disjoint")
        seen.update(ids)
    if "capture_bundle" in value:
        from agentinputlab.tau_capture_bundle import load
        if feedback.get("kind") != "tau":
            raise ValidationError("capture_bundle preparation currently applies only to Tau")
        pin = _reference(value["capture_bundle"], "capture_bundle")
        pin["path"] = os.path.abspath(base / Path(pin["path"]).expanduser())
        bundle, inventory, configurations, failures = load(pin)
        fitting = groups["train"] + groups["validation"]
        if set(fitting) - set(configurations) or set(fitting).intersection(failures):
            raise ValidationError("fitting groups require successful captures; no generation on a bundle miss")
        derived = {"captures": inventory,
                   "native_configurations": {key: configurations[key] for key in fitting},
                   "domain": "retail" if bundle["arena"] == "tau_retail" else "banking_knowledge",
                   "target_model": bundle["target_model"]}
        for key, item in derived.items():
            if key in value and value[key] != item:
                raise ValidationError(f"{key} conflicts with the pinned capture bundle")
            value[key] = item
        value["capture_bundle"] = pin
    captures = _reference(value.get("captures"), "captures")
    captures["path"] = os.path.abspath(base / Path(captures["path"]).expanduser())
    for name in ("target_model",):
        if not _object(value.get(name), name):
            raise ValidationError(f"{name} must not be empty")
    kind = feedback["kind"]
    if kind == "tau":
        if "native_configuration" in value or "environment_model" in value:
            raise ValidationError(
                "Tau preparation requires per-case native_configurations and its bound target model"
            )
        _object(value.get("native_configurations"), "native_configurations")
        _path(value.get("domain"), "domain")
    else:
        if "native_configurations" in value or "domain" in value:
            raise ValidationError("STATE/ToolSandbox preparation requires native_configuration")
        _object(value.get("native_configuration"), "native_configuration")
        if not _object(value.get("environment_model"), "environment_model"):
            raise ValidationError("environment_model must not be empty")
    if kind != "statebench":
        _path(value.get("partition_label"), "partition_label")
    elif "partition_label" in value:
        raise ValidationError(
            "STATE uses its official training split, not a caller partition label"
        )
    return to_jsonable(
        dict(value, fit=fit, **{binding_name: feedback}, captures=captures, groups=groups)
    )


def _binding(value):
    return value[
        "environment" if value["fit"]["release_id"] == "recipe.skillopt.offline" else "feedback"
    ]


def _captures(reference):
    raw, rows = _read(reference["path"], 16 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValidationError("capture inventory checksum mismatch")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValidationError("captures must be a nonempty native inventory JSON array")
    return rows


def _build(value, output):
    """Delegate to unchanged native builders, without scoring or optimizing."""
    template = _binding(value)
    kind = template["kind"]
    module = importlib.import_module("agentinputlab.recipes.ace_" + kind)
    if kind in ("bfcl", "retrieval"):
        source = template["source_input"]
    else:
        builder = getattr(module, "build_ace_" + kind + "_feedback_input")
        kwargs = dict(
            checkout=template["checkout"],
            captures=_captures(value["captures"]),
            train_ids=value["groups"]["train"],
            validation_ids=value["groups"]["validation"],
            heldout_ids=value["groups"]["heldout"],
            target_model=value["target_model"],
            output_path=output / "source-input.json",
        )
        for name in (
            "native_configuration",
            "native_configurations",
            "environment_model",
            "domain",
            "partition_label",
        ):
            if name in value:
                kwargs[name] = value[name]
        source = builder(**kwargs)
    binding = dict(template, source_input=source)
    if value["fit"]["release_id"] == "recipe.skillopt.offline":
        fit = dict(value["fit"], environment=binding)
    else:
        processor = getattr(module, "create_" + kind + "_feedback")(
            **{
                **{k: v for k, v in binding.items() if k not in ("kind", "environment_names")},
                "output_root": str(output / "input-verification"),
                "mode": "offline",
            }
        )
        examples = {
            split: [e.as_dict() for e in processor.fit_examples(split)]
            for split in ("train", "validation")
        }
        processor.validate_examples(examples["train"], examples["validation"])
        fit = dict(value["fit"], feedback=binding, examples=examples)
    FitConfig.from_dict(fit)
    _save(output / "fit.json", fit)
    # Ensure the generated file is directly consumable by the same public reader.
    FitConfig.from_file(output / "fit.json")
    return dict(
        status="prepared",
        source_input=source,
        fit_configuration=dict(path=str(output / "fit.json"), sha256=_sha(output / "fit.json")),
        input_configuration_sha256=content_hash(value),
        model_calls=0,
        fit_executed=False,
        scoring_executed=(kind == "retrieval"),
        **(dict(scoring_scope="empty_ranking_preflight_only") if kind == "retrieval" else {}),
        **(
            dict(groups=value["groups"], captures=value["captures"])
            if kind not in ("bfcl", "retrieval")
            else dict(source_groups_from=source)
        ),
    )


def prepare_fit(path, *, output_dir):
    """Build an explicit fit file in its native Python environment once."""
    value = _configuration(path)
    binding = _binding(value)
    if binding["kind"] in ("bfcl", "retrieval"):
        from agentinputlab.artifacts import load_pinned_json_mapping

        pin = binding["source_input"]
        load_pinned_json_mapping(
            pin["path"], expected_sha256=pin["sha256"], max_bytes=8 * 1024 * 1024
        )
    else:
        _captures(value["captures"])
    output = Path(output_dir).expanduser().absolute()
    if any(
        output.resolve().is_relative_to(Path(root).resolve())
        for root in (value["fit"]["checkout"], binding.get("checkout", value["fit"]["checkout"]))
    ):
        raise ValidationError("preparation output must be outside native checkouts")
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        raise ValidationError(
            "preparation output exists; use its prepared fit file or inspect its terminal"
        ) from None
    request = output / "preparation.json"
    _save(request, value)
    _save(output / "started.json", dict(configuration_sha256=content_hash(value), model_calls=0))
    launcher = 'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("agentinputlab.fit_preparation",run_name="__main__")'
    command = [
        value["fit"]["python_executable"],
        "-I",
        "-B",
        "-c",
        launcher,
        str(Path(__file__).resolve().parents[1]),
        "--worker",
        str(request),
    ]
    try:
        with (
            (output / "stdout.log").open("x") as stdout,
            (output / "stderr.log").open("x") as stderr,
        ):
            result = subprocess.run(
                command,
                cwd=output,
                stdout=stdout,
                stderr=stderr,
                timeout=value["timeout_seconds"],
                check=False,
            )
        _, terminal = _read(output / "terminal.json", 1024 * 1024)
        if result.returncode or terminal.get("status") != "prepared":
            raise ExternalDependencyError(
                f"ACE input preparation failed; inspect {output}; no fit was started"
            )
        return terminal
    except (OSError, subprocess.TimeoutExpired, ValidationError):
        if not (output / "terminal.json").exists():
            _save(
                output / "terminal.json",
                dict(status="preparation_failed", model_calls=0, fit_executed=False),
            )
        raise ExternalDependencyError(
            f"ACE input preparation did not complete; inspect {output}; no automatic retry"
        ) from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    output = args.worker.absolute().parent
    try:
        terminal = _build(_configuration(args.worker), output)
    except Exception as exc:
        from agentinputlab.observability.monitor import _safe_error_message

        _save(
            output / "terminal.json",
            dict(
                status="preparation_failed",
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
                error_sha256=hashlib.sha256(str(exc).encode()).hexdigest(),
                model_calls=0,
                fit_executed=False,
            ),
        )
        return 1
    _save(output / "terminal.json", terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
