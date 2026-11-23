"""Content-addressed, model-bound deployment of fitted native ACON guidelines.

Generated templates are accepted as literal text plus source-declared variable
placeholders, never as general Jinja programs. The original HistoryOptimizer
still builds the prompt, calls the explicit metered backend and parses output.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agentinputlab.backends import LLMBackend
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from .driver import _construct_optimizer, _load_manifest, _upstream_import_scope
from .guideline import (
    _LOCK,
    ACONGuidelineControls,
    _MeteredLLM,
    _save,
    validate_guideline_checkout,
    validate_training_bundle,
)

_PROTOCOL = "agentinputlab.acon.fitted-history.literal-placeholders.v1"
_VARIABLES = frozenset({"task", "history", "prev_summary"})
_SIGNALS = frozenset(
    {
        "remediation_recommendations",
        "inefficiency_patterns",
        "inefficiency_evidence",
        "missing_or_distorted_facts",
        "summary_inaccuracies",
        "summary_or_session_contributors",
        "lost_state_variables",
        "lost_or_recreated_state",
        "root_cause_categories",
        "primary_inflation_causes",
    }
)


def validate_training_signals(rows):
    """Do not promote an updater's raw-output fallback as completed training."""
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, dict)
            or "raw_output" in row
            or not any(isinstance(row.get(key), list) and row[key] for key in _SIGNALS)
            for row in rows
        )
    ):
        raise ValidationError(
            "native ACON analysis lacks complete parsed training signals; candidate remains diagnostic"
        )


def validate_fitted_template(text: str, expected_variables=_VARIABLES) -> tuple[str, ...]:
    """Parse without evaluation. Reject executable/indirect Jinja constructs."""
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > 65536:
        raise ValidationError("fitted ACON template must be non-empty bounded UTF-8 text")
    try:
        from jinja2 import Environment, meta, nodes
    except ModuleNotFoundError as exc:
        raise ExternalDependencyError(
            "install agentinputlab[acon-fit] for fitted templates"
        ) from exc
    try:
        tree = Environment().parse(text)
    except Exception as exc:
        raise ValidationError("fitted ACON template syntax is invalid") from exc
    allowed = {nodes.Output, nodes.TemplateData, nodes.Name}
    if any(type(node) not in allowed for node in tree.find_all(nodes.Node)):
        raise ValidationError(
            "fitted ACON templates allow literal text and variable placeholders only"
        )
    if any(node.name not in _VARIABLES or node.ctx != "load" for node in tree.find_all(nodes.Name)):
        raise ValidationError("fitted ACON may reference only explicit source variable names")
    variables = meta.find_undeclared_variables(tree)
    if variables != set(expected_variables) or not variables <= _VARIABLES:
        raise ValidationError("fitted ACON placeholders differ from the pinned source template")
    return tuple(sorted(variables))


def _pinned_file(root: Path, relative: str, expected: str, limit=2 * 1024**2) -> bytes:
    p = PurePosixPath(relative)
    path = root / p
    if (
        p.is_absolute()
        or ".." in p.parts
        or root.is_symlink()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValidationError("invalid fitted ACON provenance file")
    if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > limit:
        raise ValidationError("fitted ACON provenance path or size differs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValidationError("fitted ACON provenance checksum differs")
    return raw


@dataclass(frozen=True, slots=True)
class ACONGuidelineArtifact:
    artifact_id: str
    template: str
    template_sha256: str
    base_profile: str
    source_hashes: Mapping[str, str]
    backend_fingerprint: Mapping[str, Any]
    training_manifest_sha256: str
    train_ids: tuple[str, ...]
    excluded_evaluation_ids: tuple[str, ...]
    fit_artifact_identity: str
    provenance: Mapping[str, str]
    protocol: str = _PROTOCOL

    def __post_init__(self):
        if (
            self.protocol != _PROTOCOL
            or hashlib.sha256(self.template.encode()).hexdigest() != self.template_sha256
        ):
            raise ValidationError("fitted ACON protocol or template identity differs")
        validate_fitted_template(self.template)
        for name in ("source_hashes", "backend_fingerprint", "provenance"):
            object.__setattr__(self, name, freeze_mapping(getattr(self, name)))
        required = {"fit_start_sha256", "fit_terminal_sha256", "analysis_sha256", "candidate_path"}
        if set(self.provenance) != required:
            raise ValidationError(
                "fitted ACON artifact must retain its complete fit/analysis provenance"
            )
        for name in ("train_ids", "excluded_evaluation_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(
                values
            ):
                raise ValidationError("fitted ACON split IDs must be unique non-empty strings")
            object.__setattr__(self, name, values)
        if not self.train_ids or set(self.train_ids) & set(self.excluded_evaluation_ids):
            raise ValidationError("fitted ACON train/evaluation IDs overlap or train is empty")
        if self.artifact_id != content_hash(self.identity_payload()):
            raise ValidationError("fitted ACON artifact payload differs from its identity")

    def identity_payload(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "artifact_id"}

    def as_dict(self):
        return to_jsonable({"artifact_id": self.artifact_id, **self.identity_payload()})

    def save(self, path: str | Path):
        _save(Path(path), self.as_dict())

    @classmethod
    def load(cls, path: str | Path, *, sha256: str):
        path = Path(path)
        raw = _pinned_file(path.parent, path.name, sha256)
        return cls(**json.loads(raw))

    @classmethod
    def from_fit(
        cls,
        *,
        checkout,
        fit_root,
        training_root,
        backend: LLMBackend,
        fit_start_sha256: str,
        fit_terminal_sha256: str,
        analysis_sha256: str,
        candidate_path: str,
    ):
        """Freeze one explicitly selected candidate, never select by heldout scores."""
        root = Path(fit_root)
        start = json.loads(_pinned_file(root, "fit-start.json", fit_start_sha256))
        terminal = json.loads(_pinned_file(root, "fit-terminal.json", fit_terminal_sha256))
        analysis = json.loads(
            _pinned_file(root, "aggregated_history_regressions.json", analysis_sha256)
        )
        validate_training_signals(analysis)
        source, hashes = validate_guideline_checkout(checkout)
        if start["source_hashes"] != hashes or terminal["source_hashes"] != hashes:
            raise ValidationError("fitted ACON upstream source identity differs")
        if content_hash(start["backend"]) != content_hash(backend.fingerprint()):
            raise ValidationError("fitted ACON artifact cannot change its model/backend binding")
        if (
            terminal["status"] != "candidates_generated"
            or candidate_path not in terminal["candidates"]
        ):
            raise ValidationError("fitted ACON candidate was not produced by the pinned fit")
        if start["training_manifest_sha256"] != terminal["training_manifest_sha256"]:
            raise ValidationError("fitted ACON training provenance differs")
        inputs = validate_training_bundle(training_root, start["training_manifest_sha256"])
        core, _ = _load_manifest(Path(__file__).with_name("compatibility_manifest.json"))
        profile = core["prompt_profiles"].get(start["base_profile"])
        if not profile or profile["mode"] != "history":
            raise ValidationError("fitted ACON base prompt is not a pinned history profile")
        from jinja2 import Environment, meta

        base = source / profile["prompt_dir"] / (profile["user_template"] + ".jinja")
        expected_vars = meta.find_undeclared_variables(Environment().parse(base.read_text()))
        candidates = terminal["candidates"]
        if len({PurePosixPath(p).name for p in candidates}) != len(candidates):
            raise ValidationError("fitted ACON candidate names collide")
        identity = content_hash(
            {
                "candidates": {PurePosixPath(k).name: v for k, v in candidates.items()},
                "source": hashes,
                "training": start["training_manifest_sha256"],
                "backend": start["backend"],
                "controls": start["controls"],
                "base_profile": start["base_profile"],
            }
        )
        if identity != terminal["artifact_identity"]:
            raise ValidationError("native ACON fit artifact identity differs")
        raw = _pinned_file(root, candidate_path, candidates[candidate_path], limit=65536)
        text = raw.decode("utf-8")
        validate_fitted_template(text, expected_vars)
        values = dict(
            template=text,
            template_sha256=candidates[candidate_path],
            base_profile=start["base_profile"],
            source_hashes=hashes,
            backend_fingerprint=start["backend"],
            training_manifest_sha256=start["training_manifest_sha256"],
            train_ids=tuple(inputs["train_ids"]),
            excluded_evaluation_ids=tuple(inputs["excluded_evaluation_ids"]),
            fit_artifact_identity=identity,
            provenance={
                "fit_start_sha256": fit_start_sha256,
                "fit_terminal_sha256": fit_terminal_sha256,
                "analysis_sha256": analysis_sha256,
                "candidate_path": candidate_path,
            },
            protocol=_PROTOCOL,
        )
        return cls(artifact_id=content_hash(values), **values)

    def compress_history(
        self,
        *,
        checkout,
        backend: LLMBackend,
        task: str,
        history: str,
        output_root,
        previous_summary: str | None = None,
        evaluation_id: str | None = None,
        controls: ACONGuidelineControls | None = None,
    ):
        """Run pinned native compression with one frozen artifact and metered usage.

        Use a dedicated child process for the upstream's process-global imports.
        A supplied evaluation ID must have been explicitly excluded before fit.
        This is caller-bound ID separation, not independent dataset/group attestation.
        Omit the ID for general inference; no heldout-evaluation claim is then made.
        """
        if content_hash(backend.fingerprint()) != content_hash(self.backend_fingerprint):
            raise ValidationError("fitted ACON backend/model differs before inference")
        if evaluation_id is not None and evaluation_id not in self.excluded_evaluation_ids:
            raise ValidationError("fitted ACON evaluation ID was not excluded before training")
        if (
            not isinstance(task, str)
            or not task.strip()
            or not isinstance(history, str)
            or not history.strip()
        ):
            raise ValidationError("fitted ACON requires non-empty task and real history text")
        if previous_summary is not None and not isinstance(previous_summary, str):
            raise ValidationError("fitted ACON previous summary must be text")
        if len((task + history + (previous_summary or "")).encode()) > 4 * 1024**2:
            raise ValidationError("fitted ACON input exceeds limit")
        source, hashes = validate_guideline_checkout(checkout)
        if dict(self.source_hashes) != hashes:
            raise ValidationError("fitted ACON source differs before inference")
        core, _ = _load_manifest(Path(__file__).with_name("compatibility_manifest.json"))
        profile = core["prompt_profiles"].get(self.base_profile)
        if not profile or profile["mode"] != "history":
            raise ValidationError("fitted ACON prompt profile differs")
        validate_fitted_template(self.template)
        controls = controls or ACONGuidelineControls(
            max_model_calls=1, max_output_tokens=512, timeout_seconds=180
        )
        root = Path(output_root)
        if root.is_symlink():
            raise ValidationError("fitted ACON output root must not be a symlink")
        root.mkdir(parents=True, exist_ok=False)
        _save(
            root / "runtime-start.json",
            {
                "artifact_id": self.artifact_id,
                "controls": controls,
                "evaluation_id": evaluation_id,
                "input_hash": content_hash(
                    {"task": task, "history": history, "previous_summary": previous_summary}
                ),
                "automatic_retry": False,
            },
        )
        meter = None
        try:
            prompts = root / "prompts"
            prompts.mkdir()
            system = (
                source / profile["prompt_dir"] / (profile["system_template"] + ".jinja")
            ).read_text()
            for name, text in (
                ("source_system.jinja", system),
                ("fitted_history.jinja", self.template),
            ):
                with (prompts / name).open("x", encoding="utf-8") as f:
                    f.write(text)
            journal = root / "provider-attempts"
            journal.mkdir()
            meter = _NativeHistoryMeter(backend, controls, journal)
            with _LOCK, _upstream_import_scope(source):
                native = importlib.import_module(
                    "productive_agents.ctxopt.history_optimizer"
                ).HistoryOptimizer
                optimizer = _construct_optimizer(
                    native,
                    {
                        "history_prompt_dir": str(prompts.resolve()),
                        "prompts": {
                            "prompt_system": "source_system",
                            "prompt_history_user": "fitted_history",
                        },
                        "history_summarization_threshold": -1,
                        "model": backend.model_id,
                        "temperature": 0.0,
                        "use_llmlingua": False,
                    },
                    meter,
                    "HistoryOptimizer",
                )
                if optimizer.tokenizer is None:
                    raise ValidationError("exact native tokenizer unavailable")
                summary = optimizer.process(
                    task=task,
                    history=history,
                    prev_history_summary=previous_summary,
                    raw_history=[],
                    opt_args=None,
                )
                if not isinstance(summary, str) or not summary.strip():
                    raise ValidationError("fitted native ACON returned an empty summary")
                result = {
                    "status": "native_summary_terminal",
                    "summary": summary,
                    "artifact_id": self.artifact_id,
                    "model_calls": meter.attempts,
                    "input_tokens": meter.input_tokens,
                    "output_tokens": meter.output_tokens,
                    "native_class": native.__module__ + "." + native.__name__,
                    "native_interactions": optimizer.history,
                    "native_system_sha256": hashlib.sha256(
                        meter.system_message.encode()
                    ).hexdigest(),
                    "evaluation_id_verified_against_prefit_exclusions": evaluation_id is not None,
                    "automatic_retry": False,
                    "profile_kind": "controlled_adapter",
                }
            _save(root / "runtime-terminal.json", result)
            return result
        except Exception as exc:
            _save(
                root / "runtime-terminal.json",
                {
                    "status": "failed_runtime" if meter else "entry_failure",
                    "error_type": type(exc).__name__,
                    "model_calls": meter.attempts if meter else 0,
                    "known_input_tokens": meter.input_tokens if meter else 0,
                    "known_output_tokens": meter.output_tokens if meter else 0,
                    "usage_unknown": meter.usage_unknown if meter else False,
                    "automatic_retry": False,
                },
            )
            raise


class _NativeHistoryMeter(_MeteredLLM):
    def generate(self, prompt, temperature=0.0):
        if temperature != 0.0:
            raise ValidationError("fitted native ACON temperature must remain zero")
        return super().generate(prompt)
