"""Source-locked ACON history-guideline pipeline with an explicit metered backend.

Use in a dedicated child process. Upstream source imports and random state are
process-global. Generated Jinja is an opaque, untrusted candidate artifact here;
this module never renders or deploys generated templates. It does not attest
input-dataset authenticity or complete downstream guideline acceptance.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import random
import re
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from agentinputlab.backends import GenerationRequest, LLMBackend
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .driver import _load_manifest, _upstream_import_scope, _verify_checkout

_LOCK = threading.Lock()
_SHA = re.compile(r"[0-9a-f]{64}")
_PIPELINE = "experiments/prompt_optimizer/unified_update_history_prompt.py"
_MODULE = "_agentinputlab_acon_guideline_source_v1"


def _save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ACONGuidelineControls:
    per_call_output_tokens: int = 512
    max_model_calls: int = 32
    max_output_tokens: int = 16384
    timeout_seconds: int = 900
    num_prompts: int = 1
    samples_per_prompt: int = 30
    seed: int = 17

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValidationError(f"ACON guideline {name} must be an integer in range")
        if self.per_call_output_tokens > self.max_output_tokens:
            raise ValidationError("ACON per-call cap exceeds total fit budget")


def validate_guideline_checkout(checkout: str | Path) -> tuple[Path, dict[str, str]]:
    root = Path(checkout).expanduser()
    if sys.version_info < (3, 11):
        raise ExternalDependencyError("native ACON guideline training requires Python 3.11+")
    if root.is_symlink() or not root.is_dir():
        raise ExternalDependencyError("ACON guideline source must be a real pinned checkout")
    root = root.resolve()
    core, _ = _load_manifest(Path(__file__).with_name("compatibility_manifest.json"))
    _verify_checkout(root, core)
    extra = json.loads(files("workers.acon").joinpath("guideline_source_manifest.json").read_text())
    if extra["commit"] != core["upstream"]["commit"]:
        raise ExternalDependencyError("ACON guideline and optimizer source revisions differ")
    hashes = dict(extra["files"])
    for relative, expected in hashes.items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
            or _sha(path) != expected
        ):
            raise ExternalDependencyError(f"ACON guideline source differs: {relative}")
    hashes.update(core["source_files"])
    for profile in core["prompt_profiles"].values():
        hashes.update(profile["prompt_files"])
    return root, hashes


class _MeteredLLM:
    def __init__(self, backend: LLMBackend, controls: ACONGuidelineControls, journal: Path):
        self.backend, self.controls, self.journal = backend, controls, journal
        self.attempts = self.input_tokens = self.output_tokens = 0
        self.started = time.monotonic()
        self.usage_unknown = False
        self.system_message = ""

    def generate(self, prompt: str, max_tokens: int = 16384, **kwargs) -> str:
        if kwargs:
            raise ValidationError("unexpected native ACON generation controls")
        c = self.controls
        cap = min(c.per_call_output_tokens, max_tokens)
        if (
            self.usage_unknown
            or self.attempts >= c.max_model_calls
            or self.output_tokens + cap > c.max_output_tokens
            or time.monotonic() - self.started >= c.timeout_seconds
        ):
            raise ValidationError("ACON guideline fit budget exhausted before transport")
        request = GenerationRequest(
            prompt=prompt,
            system=self.system_message,
            temperature=0,
            max_tokens=cap,
            seed=c.seed + self.attempts,
            metadata={"acon_fit_call_index": self.attempts},
        )
        self.attempts += 1
        _save(self.journal / f"{self.attempts:04d}-request.json", request)
        try:
            response = self.backend.generate(request)
        except Exception as exc:
            self.usage_unknown = True
            _save(
                self.journal / f"{self.attempts:04d}-error.json",
                {"error_type": type(exc).__name__, "usage_unknown": True, "automatic_retry": False},
            )
            raise
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        _save(self.journal / f"{self.attempts:04d}-response.json", response)
        if response.usage.input_tokens == response.usage.output_tokens == 0:
            self.usage_unknown = True
            raise ExternalDependencyError("ACON guideline backend did not report token usage")
        if response.usage.output_tokens > cap:
            raise ValidationError("ACON guideline backend exceeded per-call output budget")
        return response.text


@contextlib.contextmanager
def guideline_runtime(
    checkout: str | Path, backend: LLMBackend, controls: ACONGuidelineControls, journal: Path
) -> Iterator[Any]:
    """Bind the native analyzer/updater to one explicitly supplied backend.

    The original LLMManager factory is never called, so this path neither reads
    private_config.yaml nor creates the native retrying/auth-discovery clients.
    Network access is the caller backend's responsibility, not a sandbox claim.
    """
    root, _ = validate_guideline_checkout(checkout)
    journal.mkdir(parents=True, exist_ok=False)
    meter = _MeteredLLM(backend, controls, journal)
    with _LOCK, _upstream_import_scope(root):
        prior_path = list(sys.path)
        prior_random = random.getstate()
        prior_common = {
            k: v for k, v in sys.modules.items() if k == "common" or k.startswith("common.")
        }
        for name in prior_common:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(root / "experiments/prompt_optimizer"))
        old_factory = manager = None
        try:
            module = importlib.import_module("productive_agents.agents.utils")
            manager = module.LLMManager
            old_factory = inspect.getattr_static(manager, "create_llm")
            manager.create_llm = staticmethod(lambda *args, **kwargs: meter)
            spec = importlib.util.spec_from_file_location(_MODULE, root / _PIPELINE)
            if spec is None or spec.loader is None:
                raise ExternalDependencyError("cannot load native ACON guideline pipeline")
            native = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE] = native
            spec.loader.exec_module(native)
            native._agentinputlab_meter = meter
            yield native
        except ModuleNotFoundError as exc:
            raise ExternalDependencyError(
                f"install agentinputlab[acon-fit]; missing {exc.name}"
            ) from exc
        finally:
            if manager is not None and old_factory is not None:
                manager.create_llm = old_factory
            sys.modules.pop(_MODULE, None)
            for name in tuple(sys.modules):
                if name == "common" or name.startswith("common."):
                    sys.modules.pop(name, None)
            sys.modules.update(prior_common)
            sys.path[:] = prior_path
            random.setstate(prior_random)


def validate_training_bundle(root: str | Path, manifest_sha256: str) -> dict[str, Any]:
    """Check caller-frozen train-only native files before any generation."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or not _SHA.fullmatch(manifest_sha256):
        raise ValidationError("invalid ACON training bundle or manifest hash")
    manifest = root / "manifest.json"
    if manifest.is_symlink() or _sha(manifest) != manifest_sha256:
        raise ValidationError("ACON training input manifest differs")
    value = json.loads(manifest.read_text())
    if (
        value.get("schema") != "agentinputlab.acon.training-input.v1"
        or value.get("split") != "train"
    ):
        raise ValidationError("ACON guideline inputs must be explicitly train-only")
    if value.get("benchmark") not in ("appworld", "smolagents", "officebench"):
        raise ValidationError("unsupported native ACON guideline benchmark")
    ids, excluded = value.get("train_ids"), value.get("excluded_evaluation_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        raise ValidationError("ACON training IDs are missing")
    if (
        len(set(ids)) != len(ids)
        or not isinstance(excluded, list)
        or not all(isinstance(x, str) and x for x in excluded)
        or set(ids) & set(excluded)
    ):
        raise ValidationError("ACON training IDs overlap evaluation or are duplicated")
    hashes = value.get("files")
    if not isinstance(hashes, dict) or not hashes:
        raise ValidationError("ACON training file hashes are missing")
    total = 0
    for relative, expected in hashes.items():
        p = PurePosixPath(relative)
        if (
            p.is_absolute()
            or ".." in p.parts
            or not isinstance(expected, str)
            or not _SHA.fullmatch(expected)
        ):
            raise ValidationError("invalid ACON training file identity")
        path = root / p
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
        ):
            raise ValidationError("ACON training file is not contained in bundle")
        total += path.stat().st_size
        if total > 64 * 1024**2 or _sha(path) != expected:
            raise ValidationError("ACON training bundle exceeds limit or file hash differs")
    for name in ("baseline-evaluations.json", "optimized-evaluations.json"):
        if name not in hashes:
            raise ValidationError("ACON native evaluation input is not pinned")
        score = json.loads((root / name).read_text())
        keys = (
            set(score.get("successful_tasks", [])) | set(score.get("failed_tasks", []))
            if value["benchmark"] == "officebench"
            else set(score.get("individual", {}))
        )
        if keys != set(ids):
            raise ValidationError(
                "ACON evaluation input must cover exactly the frozen training IDs"
            )
        if value["benchmark"] == "officebench":
            if set(score.get("successful_tasks", [])) & set(score.get("failed_tasks", [])):
                raise ValidationError("ACON training task cannot be both successful and failed")
        else:
            for row in score["individual"].values():
                if not isinstance(row, dict):
                    raise ValidationError("ACON native training evaluation row is invalid")
                if value["benchmark"] == "smolagents":
                    f1 = row.get("f1")
                    if (
                        isinstance(f1, bool)
                        or not isinstance(f1, (int, float))
                        or not math.isfinite(f1)
                        or not 0 <= f1 <= 1
                    ):
                        raise ValidationError("ACON smolagents-format training F1 is invalid")
                elif not isinstance(row.get("success"), bool):
                    raise ValidationError("ACON appworld training outcome is missing")
    for directory in ("baseline", "optimized"):
        folder = root / directory
        if folder.is_symlink() or not folder.is_dir():
            raise ValidationError("ACON native history directory is missing")
        for path in folder.rglob("*"):
            if path.is_symlink() or (
                path.is_file() and path.relative_to(root).as_posix() not in hashes
            ):
                raise ValidationError("ACON history directory contains an unpinned file")
        native_cases = {}
        if value["benchmark"] == "smolagents":
            samples = folder / "samples"
            if not samples.is_dir() or samples.is_symlink():
                raise ValidationError("native ACON smolagents samples directory is missing")
            native_cases = {p.name: p for p in samples.iterdir() if p.is_dir()}
        else:
            for parent in folder.iterdir():
                if not parent.is_dir():
                    continue
                name = parent.name.lstrip("task_")  # Same identity mapping as pinned source.
                children = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
                if value["benchmark"] == "officebench":
                    native_cases.update(
                        {name + "_" + p.name: p for p in children}
                        if children
                        else {name + "_0": parent}
                    )
                else:
                    native_cases[name] = parent
        if set(native_cases) != set(ids):
            raise ValidationError(
                "native ACON task-directory mapping differs from frozen training IDs"
            )
        if any(not (p / "llm_history.json").is_file() for p in native_cases.values()):
            raise ValidationError("native ACON training case history is missing")
        for case in native_cases.values():
            try:
                history = json.loads((case / "llm_history.json").read_text())
            except (ValueError, UnicodeError) as exc:
                raise ValidationError("native ACON training case history is malformed") from exc
            sessions = (
                history
                if isinstance(history, list)
                and history
                and all(isinstance(x, list) and x and isinstance(x[0], dict) for x in history)
                else [history]
            )
            if not all(isinstance(s, list) and s for s in sessions):
                raise ValidationError("native ACON training history must contain real messages")
            for session in sessions:
                for message in session:
                    valid = (
                        isinstance(message, dict)
                        and isinstance(message.get("role"), str)
                        and bool(message["role"])
                        and "content" in message
                    ) or (
                        isinstance(message, list)
                        and len(message) >= 2
                        and isinstance(message[0], str)
                        and bool(message[0])
                    )
                    if not valid:
                        raise ValidationError("native ACON training message is malformed")
    return value


class ACONGuidelinePipeline:
    """Run original history analysis and history update; save opaque candidates."""

    def __init__(self, controls: ACONGuidelineControls | None = None):
        self.controls = controls or ACONGuidelineControls()

    def fit(
        self,
        *,
        checkout: str | Path,
        training_root: str | Path,
        training_manifest_sha256: str,
        output_root: str | Path,
        backend: LLMBackend,
        base_profile: str = "smolagents.history.v1",
    ) -> Mapping[str, Any]:
        inputs = validate_training_bundle(training_root, training_manifest_sha256)
        source, hashes = validate_guideline_checkout(checkout)
        core, _ = _load_manifest(Path(__file__).with_name("compatibility_manifest.json"))
        profile = core["prompt_profiles"].get(base_profile)
        if not profile or profile["mode"] != "history":
            raise ValidationError("ACON guideline fit requires a pinned history prompt profile")
        prompt_dir = source / profile["prompt_dir"]
        for path in prompt_dir.glob("system*"):
            if path.suffix in (".jinja", ".txt", ".md", ".json") and (
                path.is_symlink() or path.relative_to(source).as_posix() not in hashes
            ):
                raise ValidationError("ACON system prompt copy includes an unpinned file")
        if Path(output_root).is_symlink():
            raise ValidationError("ACON output root must not be a symlink")
        root, data = Path(output_root).resolve(), Path(training_root).resolve()
        if root.is_relative_to(data) or data.is_relative_to(root):
            raise ValidationError("ACON output must be separate from frozen training inputs")
        root.mkdir(parents=True, exist_ok=False)
        c = self.controls
        config = dict(
            benchmark=inputs["benchmark"],
            task_split="train",
            baseline_model_output_dir=data / "baseline",
            baseline_eval_output_file=data / "baseline-evaluations.json",
            optimized_model_output_dir=data / "optimized",
            optimized_eval_output_file=data / "optimized-evaluations.json",
            analysis_model=backend.model_id,
            update_model=backend.model_id,
            analysis_prompt_template=source
            / "experiments/prompt_optimizer/prompts/history_regression_prompt.jinja",
            optimizer_template=source
            / "experiments/prompt_optimizer/prompts/history_prompt_optimizer_prompt_by_samples.jinja",
            base_prompt_template=source
            / profile["prompt_dir"]
            / (profile["user_template"] + ".jinja"),
            num_prompts=c.num_prompts,
            samples_per_prompt=c.samples_per_prompt,
            max_tasks=len(inputs["train_ids"]),
            include_performance_regressions=False,
            skip_existing_analysis=False,
            dedupe=False,
            llm_backend="openai",
            seed=c.seed,
            output_dir=root,
        )
        _save(
            root / "fit-start.json",
            {
                "controls": to_jsonable(c),
                "source_hashes": hashes,
                "training_manifest_sha256": training_manifest_sha256,
                "base_profile": base_profile,
                "backend": backend.fingerprint(),
                "profile_kind": "controlled_adapter",
                "automatic_retry": False,
            },
        )
        meter = None
        try:
            with guideline_runtime(source, backend, c, root / "provider-attempts") as native:
                meter = native._agentinputlab_meter
                aggregate = native.run_history_analysis(config)
                if aggregate is not None:
                    native.run_history_update(config, aggregate)
                paths = sorted(
                    (root / "optimized_prompts").glob("improved_history_prompt_samples_*.jinja")
                )
                result = dict(
                    status="candidates_generated" if paths else "no_regression_candidates",
                    candidates={p.relative_to(root).as_posix(): _sha(p) for p in paths},
                    model_calls=meter.attempts,
                    input_tokens=meter.input_tokens,
                    output_tokens=meter.output_tokens,
                    generated_templates_executed=False,
                    deployment_acceptance_complete=False,
                    training_manifest_sha256=training_manifest_sha256,
                    source_hashes=hashes,
                    artifact_identity=content_hash(
                        {
                            "candidates": {p.name: _sha(p) for p in paths},
                            "source": hashes,
                            "training": training_manifest_sha256,
                            "backend": backend.fingerprint(),
                            "controls": to_jsonable(c),
                            "base_profile": base_profile,
                        }
                    ),
                )
            _save(root / "fit-terminal.json", result)
            return result
        except Exception as exc:
            _save(
                root / "fit-terminal.json",
                {
                    "status": "failed_fit" if meter is not None else "entry_failure",
                    "error_type": type(exc).__name__,
                    "model_calls": meter.attempts if meter is not None else 0,
                    "known_input_tokens": meter.input_tokens if meter is not None else 0,
                    "known_output_tokens": meter.output_tokens if meter is not None else 0,
                    "usage_unknown": meter.usage_unknown if meter is not None else False,
                    "automatic_retry": False,
                    "generated_templates_executed": False,
                },
            )
            raise
