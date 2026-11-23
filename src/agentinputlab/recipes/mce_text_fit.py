"""Fit the declared MCE text-only profile with pinned upstream orchestration.

Run this API in a dedicated process: upstream imports and Python's random state
are process-global. The four controlled seams produce text, never executable
interfaces. This is not the separately attested generated-code/sandbox profile.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from agentinputlab.backends import GenerationRequest, LLMBackend
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.metrics.qa import english_qa_exact_match, english_qa_f1

from .mce import MCE_UPSTREAM_COMMIT, MCEExample, validate_mce_split_isolation
from .mce_text_only import MCETextArtifact

_MAX_TEXT = 48000
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def validate_mce_text_checkout(checkout: str | Path) -> tuple[Path, dict[str, str]]:
    root = Path(checkout).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ExternalDependencyError("MCE source must be a real pinned checkout")
    root = root.resolve()
    manifest = json.loads(
        files("workers.mce").joinpath("compatibility_manifest.json").read_text(encoding="utf-8")
    )
    try:
        revision = subprocess.check_output(
            ("git", "-C", str(root), "rev-parse", "HEAD"), text=True, timeout=15
        ).strip()
        dirty = subprocess.check_output(
            ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"),
            text=True,
            timeout=15,
        ).strip()
        extra = subprocess.check_output(
            (
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "mce",
                "env",
            ),
            text=True,
            timeout=15,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExternalDependencyError("cannot verify MCE source revision") from exc
    if revision != MCE_UPSTREAM_COMMIT or dirty:
        raise ExternalDependencyError("MCE source revision differs or tracked files changed")
    if any(path.endswith((".py", ".pyc", ".so", ".pyd")) for path in extra):
        raise ExternalDependencyError("MCE source contains untracked importable files")
    hashes = dict(manifest["source"]["import_closure"])
    hashes[manifest["license"]["path"]] = manifest["license"]["sha256"]
    for relative, expected in hashes.items():
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ExternalDependencyError(f"MCE pinned source differs: {relative}")
    return root, hashes


def _excluded_sdk(*args, **kwargs):
    raise ExternalDependencyError("this MCE text profile does not execute the excluded SDK")


@contextmanager
def _excluded_sdk_imports():
    """Import sentinels only: any accidental SDK invocation raises immediately."""
    names = {
        "langchain_openai": ("OpenAIEmbeddings",),
        "claude_agent_sdk": ("query", "ClaudeAgentOptions", "ClaudeSDKClient"),
    }
    previous = {name: sys.modules.get(name) for name in names}
    for name, members in names.items():
        module = types.ModuleType(name)
        for member in members:
            setattr(module, member, _excluded_sdk)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _bounded_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValidationError("MCE text artifact is empty or contains NUL")
    if len(value) > _MAX_TEXT:
        raise ValidationError("MCE text artifact exceeds the character limit")
    return value.strip()


def _context_files(root: Path) -> dict[str, str]:
    folder = root / "context"
    if not folder.exists():
        return {}
    if folder.is_symlink() or not folder.is_dir():
        raise ValidationError("MCE context must be a real directory")
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            raise ValidationError("MCE text profile rejected a generated non-text file")
        if path.stat().st_size > 4 * _MAX_TEXT:
            raise ValidationError("MCE context file exceeds the byte limit")
        result["context/" + path.name] = _bounded_text(path.read_text(encoding="utf-8"))
    if sum(len(v) for v in result.values()) > _MAX_TEXT:
        raise ValidationError("MCE context bundle exceeds the character limit")
    return result


@dataclass(frozen=True, slots=True)
class MCETextFitControls:
    iterations: int = 1
    train_batch_size: int = 2
    per_call_output_tokens: int = 512
    max_model_calls: int = 32
    max_output_tokens: int = 16384
    timeout_seconds: int = 900
    seed: int = 17

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValidationError(f"MCE text {name} must be an integer in range")


class _TextRuntime:
    def __init__(self, backend: LLMBackend, controls: MCETextFitControls):
        self.backend, self.controls = backend, controls
        self.attempts = self.input_tokens = self.output_tokens = 0
        self.started = time.monotonic()

    def generate(self, system: str, prompt: str) -> str:
        c = self.controls
        if (
            self.attempts >= c.max_model_calls
            or self.output_tokens + c.per_call_output_tokens > c.max_output_tokens
            or time.monotonic() - self.started >= c.timeout_seconds
        ):
            raise ValidationError("MCE offline fit budget exhausted before dispatch")
        index = self.attempts
        self.attempts += 1
        response = self.backend.generate(
            GenerationRequest(
                system=system,
                prompt=prompt,
                temperature=0,
                max_tokens=c.per_call_output_tokens,
                seed=c.seed + index,
                metadata={"mce_logical_call_index": index},
            )
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if response.usage.output_tokens > c.per_call_output_tokens:
            raise ValidationError("MCE provider exceeded the declared per-call output cap")
        return response.text.strip()

    async def run_meta_agent(self, **kwargs):
        folder, workspace = Path(kwargs["iter_dir"]), Path(kwargs["workspace_base"])
        history = workspace / "meta_agent/evaluations.json"
        prompt = (
            "Create a reusable context-engineering skill for the task below. Analyze only the "
            "provided discovery examples and prior score summary. Output Markdown beginning with "
            "'## Skill Overview'. Do not output executable Python or shell commands. The skill "
            "must tell a later context engineer how to extract generalizable evidence-grounded "
            "guidance without copying labels verbatim.\n\nTASK:\n"
            + str(kwargs["task_instruction"])
            + "\n\nDISCOVERY EXAMPLES:\n"
            + (workspace / "meta_agent/train.jsonl").read_text(encoding="utf-8")
            + "\n\nPRIOR SCORE SUMMARY:\n"
            + (history.read_text(encoding="utf-8") if history.exists() else "{}")
        )
        skill = _bounded_text(
            self.generate(
                "You are the meta-level skill-evolution agent in a text-only MCE runtime.", prompt
            )
        )
        if "## Skill Overview" not in skill:
            skill = "## Skill Overview\n\n" + skill
        target = folder / ".claude/skills/learning-context/SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_bounded_text(skill) + "\n", encoding="utf-8")
        return {"success": True}

    async def run_base_agent(self, **kwargs):
        folder = Path(kwargs["iter_dir"])
        prompt = (
            "Following the evolved skill, synthesize a concise reusable context artifact from "
            "the discovery rollouts. Output Markdown only. Do not output Python, shell commands, "
            "absolute paths, or instructions to access hidden labels. Focus on transferable "
            "answering heuristics and evidence use rather than memorizing example answers.\n\n"
            "EVOLVED SKILL:\n"
            + (folder / ".claude/skills/learning-context/SKILL.md").read_text(encoding="utf-8")
            + "\n\nDISCOVERY ROLLOUTS:\n"
            + (folder / "data/train.json").read_text(encoding="utf-8")
        )
        text = _bounded_text(
            self.generate(
                "You are the base-level context optimizer in a text-only MCE runtime.", prompt
            )
        )
        target = folder / "context/learned_context.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        return {"success": True}

    def load_interfaces(self, iter_dir, signatures):
        if signatures:
            raise ValidationError("MCE text profile forbids executable interfaces")
        text = "\n\n".join(_context_files(Path(iter_dir)).values())
        return {"text": text} if text else {}

    async def batch_evaluate(self, interfaces, samples, env_name, **kwargs):
        del kwargs
        results = []
        context = str(interfaces.get("text", ""))
        for sample in samples:
            prompt = str(sample.question)
            if context:
                prompt += "\n\nLEARNED MCE CONTEXT:\n" + context
            prompt += "\n\nAnswer using only the visible evidence. Return only <answer>your concise answer</answer>."
            raw = self.generate("You answer evidence-grounded scientific questions.", prompt)
            match = _ANSWER.search(raw)
            prediction = (match.group(1) if match else raw).strip()
            # References are scorer-only, never included in the inference prompt.
            reference = str(sample.ground_truth or "")
            metrics = {
                "f1": english_qa_f1(prediction, reference),
                "exact_match": english_qa_exact_match(prediction, reference),
            }
            results.append(
                {
                    "sample": sample.to_dict(),
                    "evaluation": {
                        "ground_truth": reference,
                        "metrics": metrics,
                        "trajectory": [{"prediction": prediction}],
                    },
                }
            )
        metrics = {
            key: sum(r["evaluation"]["metrics"][key] for r in results) / len(results)
            for key in ("f1", "exact_match")
        }
        return {
            "summary": {
                "metrics": metrics,
                "primary_metric": "f1",
                "primary_metric_value": metrics["f1"],
                "total": len(results),
                "successful": len(results),
                "errors": 0,
                "environment": env_name,
            },
            "results": results,
            "errors": [],
        }


@dataclass(slots=True)
class MCETextOfflineRecipe:
    """Train a text artifact; use a dedicated, credential-isolated child process.

    Disable dotenv loading in that process. Upstream creates an unused native
    OpenAI client: bind it to a closed local endpoint with a non-secret dummy
    key; actual calls use only the explicitly supplied, metered backend.
    The wall check is cooperative; impose a child-process/provider deadline too.
    """

    controls: MCETextFitControls = field(default_factory=MCETextFitControls)

    def fit(
        self,
        *,
        checkout: str | Path,
        output_root: str | Path,
        train: Sequence[MCEExample],
        validation: Sequence[MCEExample],
        backend: LLMBackend,
    ) -> MCETextArtifact:
        if threading.current_thread() is not threading.main_thread():
            raise ValidationError("MCE text fitting requires a dedicated process main thread")
        if os.environ.get("PYTHON_DOTENV_DISABLED") != "1":
            raise ValidationError("MCE text fitting requires PYTHON_DOTENV_DISABLED=1")
        train, validation = tuple(train), tuple(validation)
        audit = validate_mce_split_isolation(train, validation)
        if not train or not validation:
            raise ValidationError("MCE text fitting requires nonempty train and validation splits")
        for item in (*train, *validation):
            if (
                not isinstance(item.input, Mapping)
                or set(item.input) != {"question", "context"}
                or not all(isinstance(v, str) and v.strip() for v in item.input.values())
                or not isinstance(item.expected_output, str)
            ):
                raise ValidationError(
                    "MCE text examples require question/context strings and a text answer"
                )
        source, source_hashes = validate_mce_text_checkout(checkout)
        root = Path(output_root).expanduser()
        root.mkdir(parents=True)
        root = root.resolve()
        workspace = root / "workspace"
        workspace.mkdir()
        paths = {}
        for split, items in (("train", train), ("validation", validation)):
            path = root / (split + ".jsonl")
            with path.open("x", encoding="utf-8") as handle:
                for item in items:
                    handle.write(
                        json.dumps(
                            {
                                "example_id": item.example_id,
                                "question": "QUESTION:\n"
                                + item.input["question"]
                                + "\n\nVISIBLE EVIDENCE:\n"
                                + item.input["context"],
                                "answer": item.expected_output,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            paths[split] = path
        runtime = _TextRuntime(backend, self.controls)
        logger = logging.getLogger("agentinputlab.mce.text_fit")
        random_state = random.getstate()
        from workers.mce.driver import _pinned_upstream_modules

        try:
            random.seed(self.controls.seed)
            with _excluded_sdk_imports(), _pinned_upstream_modules(source) as (main, utils):
                base = importlib.import_module("env.base")
                registry = importlib.import_module("env.registry").EnvironmentRegistry

                class TextEnvironment(base.TaskEnvironment):
                    def get_interface_signatures(self):
                        return []

                    def get_task_instruction(self):
                        return "Answer scientific-paper questions from visible evidence. Learn reusable text context without copying answers."

                    async def aevaluate(self, *args, **kwargs):
                        raise ValidationError("evaluation belongs to the controlled text runtime")

                    def load_samples(self, path, limit=10, random_sample=False, shuffle=False):
                        rows = [
                            json.loads(line)
                            for line in Path(path).read_text(encoding="utf-8").splitlines()
                        ]
                        values = [
                            base.Sample(
                                id=i,
                                question=r["question"],
                                ground_truth=r["answer"],
                                extras={"example_id": r["example_id"]},
                            )
                            for i, r in enumerate(rows)
                        ]
                        if random_sample:
                            random.shuffle(values)
                        if shuffle:
                            random.shuffle(values)
                        return values[:limit] if limit else values

                    def get_primary_metric_name(self):
                        return "f1"

                    def format_result_for_training(self, item):
                        e = item["evaluation"]
                        return {
                            "example_id": item["sample"]["example_id"],
                            "question": item["sample"]["question"],
                            "reference": e["ground_truth"],
                            "prediction": e["trajectory"][0]["prediction"],
                            "metrics": e["metrics"],
                        }

                registry.register("agentinputlab_qasper_text", TextEnvironment)
                for seam in (
                    "run_meta_agent",
                    "run_base_agent",
                    "load_interfaces",
                    "batch_evaluate",
                ):
                    setattr(main, seam, getattr(runtime, seam))
                utils.setup_meta_agent_reference(workspace, str(paths["train"]), len(train), logger)

                async def execute():
                    values = []
                    for index in range(self.controls.iterations + 1):
                        result = await main.run_iteration(
                            workspace_base=workspace,
                            iteration=index,
                            env_name="agentinputlab_qasper_text",
                            val_data_path=str(paths["validation"]),
                            train_data_path=str(paths["train"]),
                            train_limit=len(train),
                            val_limit=len(validation),
                            model=backend.model_id,
                            logger=logger,
                            run_dir=root / "logs",
                            e2b_sandbox_manager=None,
                            train_batch_size=self.controls.train_batch_size,
                            skill_path=None,
                            no_meta_agent=False,
                        )
                        if not isinstance(result, Mapping) or "error" in result:
                            raise ExternalDependencyError(f"MCE run_iteration failed: {result}")
                        values.append(result)
                    return values

                results = asyncio.run(asyncio.wait_for(execute(), self.controls.timeout_seconds))
        finally:
            random.setstate(random_state)
        scores = tuple(float(r["val_primary_metric"]) for r in results)
        selected = max(range(len(scores)), key=scores.__getitem__)
        skill, contexts = "", {}
        if selected:
            folder = Path(results[selected]["sub_iterations"][-1]["folder"])
            skill = _bounded_text(
                (folder / ".claude/skills/learning-context/SKILL.md").read_text(encoding="utf-8")
            )
            contexts = _context_files(folder)
        artifact = MCETextArtifact.build(
            selected_iteration=selected,
            skill_markdown=skill,
            context_files=contexts,
            source_hashes=source_hashes,
            iteration_scores=scores,
            discovery_split_sha256=audit.audit_hash,
            model_fingerprint=backend.fingerprint(),
            training_usage={
                "model_calls": runtime.attempts,
                "input_tokens": runtime.input_tokens,
                "output_tokens": runtime.output_tokens,
            },
        )
        with (root / "fit-receipt.json").open("x", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(
                    {
                        "artifact": artifact,
                        "split_audit": audit,
                        "controls": self.controls,
                        "native_iterations": results,
                        "generated_code_executed": False,
                        "paper_results_changed": False,
                        "seed_mode": "base_seed_plus_logical_call_index_v1",
                    }
                ),
                handle,
                indent=2,
            )
        return artifact


__all__ = ["MCETextFitControls", "MCETextOfflineRecipe", "validate_mce_text_checkout"]
