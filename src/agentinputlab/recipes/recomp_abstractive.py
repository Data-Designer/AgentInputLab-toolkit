"""Pinned native RECOMP seq2seq training and a closed, safe candidate artifact.

The upstream Trainer owns tokenization, loss, optimizer and training/evaluation.
Observers only record events. Run each fit in a dedicated process.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import runpy
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .recomp_extractive import _LOCK, _regular, _save, _scope, _sha

RELEASE = "training.recomp.abstractive"
PROTOCOL = "agentinputlab.recomp.pinned_seq2seq_fit.v1"
SCRIPT = "train_hf_summarization_model.py"
SCRIPT_SHA256 = "0d89514a5f73f06f0f420b4274960871d8ae403877fe1e2865b7a3cb73e91c5d"
_NLTK_REQUIRED = frozenset(
    {
        "tokenizers/punkt/english.pickle",
        "tokenizers/punkt/PY3/english.pickle",
        "tokenizers/punkt_tab/english/abbrev_types.txt",
        "tokenizers/punkt_tab/english/collocations.tab",
        "tokenizers/punkt_tab/english/ortho_context.tab",
        "tokenizers/punkt_tab/english/sent_starters.txt",
    }
)
_DEPENDENCIES = {
    "transformers": "4.24.0",
    "tokenizers": "0.13.2",
    "numpy": "1.26.4",
    "datasets": "2.14.7",
    "pyarrow": "14.0.2",
    "fsspec": "2023.10.0",
    "huggingface-hub": "0.19.4",
    "nltk": "3.9.2",
    "rouge": "1.0.1",
    "safetensors": "0.5.3",
    "sentencepiece": "0.2.1",
}


@dataclass(frozen=True, slots=True)
class RECOMPAbstractiveFitControls:
    epochs: float = 3.0
    batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    max_source_length: int = 1024
    max_target_length: int = 512
    max_steps: int | None = None
    max_optimizer_steps: int = 10000
    max_data_bytes: int = 64 * 1024 * 1024
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self):
        for name in (
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "max_source_length",
            "max_target_length",
            "max_optimizer_steps",
            "max_data_bytes",
            "seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValidationError(f"RECOMP seq2seq {name} must be an integer in range")
        for name in ("epochs", "learning_rate"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValidationError(f"RECOMP seq2seq {name} must be finite and positive")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValidationError("RECOMP max_steps must be a positive explicit bound")
        if self.max_target_length > 512 or self.device not in {"cpu", "cuda"}:
            raise ValidationError("RECOMP target cap must be <=512 and device cpu or cuda")


def _records(path):
    text = path.read_text(encoding="utf-8")
    try:
        rows = (
            json.loads(text)
            if text.lstrip().startswith("[")
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("RECOMP seq2seq data requires native JSON/JSONLines records") from exc
    if not isinstance(rows, list) or not rows:
        raise ValidationError("RECOMP seq2seq data requires non-empty records")
    return rows


def audit_recomp_seq2seq_data(train_file, dev_file, controls):
    """Keep original summaries (including blanks); never infer labels from tests."""
    evidence, queries = {}, {}
    paths = {
        name: _regular(path, name) for name, path in (("train", train_file), ("dev", dev_file))
    }
    if paths["train"] == paths["dev"]:
        raise ValidationError("RECOMP train and dev files must differ")
    for split, path in paths.items():
        if path.suffix != ".json" or path.stat().st_size > controls.max_data_bytes:
            raise ValidationError(
                "RECOMP native .json data exceeds its declared byte cap or suffix"
            )
        rows = _records(path)
        unique = set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(key), str) or "\x00" in row[key]
                    for key in ("question", "passages", "summary")
                )
                or not row["question"].strip()
            ):
                raise ValidationError(
                    "RECOMP native records require question/passages/summary text"
                )
            unique.add(" ".join(row["question"].casefold().split()))
        queries[split] = unique
        evidence[split] = dict(
            path=str(path),
            sha256=_sha(path),
            rows=len(rows),
            blank_summary_rows=sum(not row["summary"].strip() for row in rows),
            query_set_sha256=content_hash(sorted(unique)),
        )
    if queries["train"] & queries["dev"]:
        raise ValidationError("RECOMP train/dev query overlap is not permitted")
    batches = math.ceil(evidence["train"]["rows"] / controls.batch_size)
    # Exact HF 4.24 single-process Trainer scheduling rule, not data truncation.
    updates = max(batches // controls.gradient_accumulation_steps, 1)
    steps = controls.max_steps or math.ceil(controls.epochs * updates)
    if steps > controls.max_optimizer_steps:
        raise ValidationError("RECOMP native optimizer plan exceeds its explicit step cap")
    return dict(
        files=evidence,
        planned_optimizer_steps=steps,
        query_overlap=False,
        native_blank_summaries_preserved=True,
        source_rows_not_rewritten=True,
    )


def validate_recomp_nltk_assets(root, manifest, expected_sha256):
    supplied = Path(root).expanduser()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValidationError("RECOMP NLTK root requires real user-mounted resources")
    folder = supplied.resolve()
    metadata = load_pinned_json_mapping(manifest, expected_sha256=expected_sha256)
    files = metadata.get("files")
    if (
        metadata.get("schema") != "agentinputlab.recomp.seq2seq.nltk.v1"
        or not isinstance(files, dict)
        or not set(files) >= _NLTK_REQUIRED
    ):
        raise ValidationError("RECOMP NLTK manifest omits genuine English resources")
    for relative, digest in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or str(path) != relative or ".." in path.parts or not path.parts:
            raise ValidationError("RECOMP NLTK resource paths must be canonical and relative")
        target = _regular(folder / relative, "NLTK resource")
        if not target.is_relative_to(folder) or _sha(target) != digest:
            raise ValidationError("RECOMP NLTK resource hash or containment differs")
    return folder


@contextmanager
def _native_observers(observed):
    import torch
    import transformers
    from transformers import Seq2SeqTrainer

    originals = {kind: kind.step for kind in {torch.optim.AdamW, transformers.AdamW}}
    original_init, original_loss = Seq2SeqTrainer.__init__, Seq2SeqTrainer.compute_loss
    owned_loss = "compute_loss" in Seq2SeqTrainer.__dict__
    original_generate = transformers.PreTrainedModel.generate

    def init(trainer, *args, **kwargs):
        original_init(trainer, *args, **kwargs)
        observed["trainer"] = trainer

    def loss(trainer, model, inputs, *args, **kwargs):
        labels = inputs.get("labels")
        label_tokens = int((labels != -100).sum().item()) if labels is not None else 0
        value = original_loss(trainer, model, inputs, *args, **kwargs)
        tensor = value[0] if isinstance(value, tuple) else value
        observed["loss_events"].append(
            dict(
                loss=float(tensor.detach().cpu()),
                gradient_enabled=torch.is_grad_enabled(),
                label_tokens=label_tokens,
                padded_input_tokens=int(inputs["input_ids"].numel()),
            )
        )
        return value

    def generate(model, *args, **kwargs):
        observed["model_generate_calls"] += 1
        return original_generate(model, *args, **kwargs)

    def step(original):
        def wrapped(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            observed["optimizer_steps"] += 1
            return result

        return wrapped

    Seq2SeqTrainer.__init__, Seq2SeqTrainer.compute_loss = init, loss
    transformers.PreTrainedModel.generate = generate
    for kind, original in originals.items():
        kind.step = step(original)
    try:
        yield
    finally:
        Seq2SeqTrainer.__init__ = original_init
        if owned_loss:
            Seq2SeqTrainer.compute_loss = original_loss
        else:
            del Seq2SeqTrainer.compute_loss
        transformers.PreTrainedModel.generate = original_generate
        for kind, original in originals.items():
            kind.step = original


@dataclass(frozen=True, slots=True)
class RECOMPTrainedSeq2SeqArtifact:
    root: Path
    artifact_id: str
    checkpoint_revision: str
    tokenizer_revision: str
    manifest_sha256: str

    @property
    def checkpoint(self):
        return self.root / "checkpoint"

    @property
    def checkpoint_manifest(self):
        return self.root / "checkpoint-manifest.json"

    @classmethod
    def load(cls, root):
        from workers.recomp.driver import verify_checkpoint

        supplied = Path(root).expanduser()
        if supplied.is_symlink() or not supplied.is_dir():
            raise ValidationError("RECOMP trained seq2seq artifact root is invalid")
        folder = supplied.resolve()
        metadata_path = _regular(folder / "artifact.json", "artifact")
        terminal_path = _regular(folder / "fit-terminal.json", "fit terminal")
        plan_path = _regular(folder / "fit-plan.json", "fit plan")
        metadata, terminal = (
            json.loads(metadata_path.read_text()),
            json.loads(terminal_path.read_text()),
        )
        if not isinstance(metadata, dict) or not isinstance(terminal, dict):
            raise ValidationError("RECOMP seq2seq artifact metadata is malformed")
        identity = metadata.pop("artifact_id", None)
        if (
            identity != content_hash(metadata)
            or metadata.get("release_id") != RELEASE
            or metadata.get("protocol") != PROTOCOL
            or metadata.get("fit_terminal_sha256") != _sha(terminal_path)
            or metadata.get("fit_plan_sha256") != _sha(plan_path)
            or terminal.get("status") != "trained_candidate"
            or terminal.get("optimizer_steps", 0) < 1
            or terminal.get("changed_parameter_tensors", 0) < 1
        ):
            raise ValidationError("RECOMP seq2seq artifact identity or terminal is invalid")
        checkpoint = verify_checkpoint(folder / "checkpoint", folder / "checkpoint-manifest.json")
        if checkpoint.mode != "abstractive" or checkpoint.manifest_sha256 != metadata.get(
            "checkpoint_manifest_sha256"
        ):
            raise ValidationError("RECOMP trained seq2seq checkpoint closure differs")
        return cls(
            folder,
            identity,
            checkpoint.checkpoint_revision,
            checkpoint.tokenizer_revision,
            checkpoint.manifest_sha256,
        )

    def inference_parameters(self):
        from workers.recomp.driver import ABSTRACTIVE_INPUT_TEMPLATE_SHA256

        current = self.load(self.root)
        return dict(
            checkpoint_revision=current.checkpoint_revision,
            tokenizer_revision=current.tokenizer_revision,
            asset_manifest_hash=current.manifest_sha256,
            input_template_hash=ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
            asset_license_acknowledged=True,
        )


class RECOMPAbstractiveTraining:
    release_id = RELEASE

    def __init__(self, controls=None):
        self.controls = (
            RECOMPAbstractiveFitControls(**controls)
            if isinstance(controls, Mapping)
            else controls or RECOMPAbstractiveFitControls()
        )
        if not isinstance(self.controls, RECOMPAbstractiveFitControls):
            raise ValidationError("RECOMP seq2seq fit requires typed controls")

    def fit(
        self,
        *,
        checkout,
        initial_checkpoint,
        initial_checkpoint_manifest,
        train_file,
        dev_file,
        nltk_root,
        nltk_manifest,
        nltk_manifest_sha256,
        output_root,
        source_and_asset_acknowledged=False,
    ):
        from workers.recomp.driver import verify_checkout, verify_checkpoint

        if source_and_asset_acknowledged is not True:
            raise ValidationError(
                "RECOMP fit needs explicit acknowledgement of local source/model/data assets"
            )
        source = Path(checkout).expanduser()
        if source.is_symlink():
            raise ValidationError("RECOMP training checkout must not be a symlink")
        checked = verify_checkout(source)
        script = _regular(checked.root / SCRIPT, "seq2seq training script")
        if _sha(script) != SCRIPT_SHA256:
            raise ExternalDependencyError("RECOMP pinned seq2seq source differs")
        initial = verify_checkpoint(Path(initial_checkpoint), Path(initial_checkpoint_manifest))
        if initial.mode != "abstractive":
            raise ValidationError("RECOMP seq2seq training requires an abstractive initialization")
        c = self.controls
        audit = audit_recomp_seq2seq_data(train_file, dev_file, c)
        resources = validate_recomp_nltk_assets(nltk_root, nltk_manifest, nltk_manifest_sha256)
        versions = {name: importlib.metadata.version(name) for name in ("torch", *_DEPENDENCIES)}
        if versions["torch"] not in {"1.13.0", "1.13.0+cpu", "1.13.0+cu117"} or any(
            versions[n] != v for n, v in _DEPENDENCIES.items()
        ):
            raise ExternalDependencyError(
                "RECOMP seq2seq fit requires documented isolated dependency versions"
            )
        if os.environ.get("WORLD_SIZE", "1") != "1":
            raise ValidationError("RECOMP bounded recipe requires a single-process fit")
        import torch

        if c.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise ExternalDependencyError(
                "RECOMP CUDA fit requires one operator-selected visible device"
            )
        root = Path(output_root).expanduser()
        if root.exists() or root.is_symlink():
            raise ValidationError(
                "RECOMP fit output must be create-only; do not repeat consumed attempts"
            )
        root = root.absolute()
        root.mkdir(parents=True)
        command = [
            str(script),
            "--model_name_or_path",
            str(initial.root),
            "--train_file",
            audit["files"]["train"]["path"],
            "--validation_file",
            audit["files"]["dev"]["path"],
            "--text_column",
            "passages",
            "--summary_column",
            "summary",
            "--output_dir",
            str(root / "native"),
            "--cache_dir",
            str(root / "dataset-cache"),
            "--do_train",
            "--do_eval",
            "--predict_with_generate",
            "False",
            "--num_train_epochs",
            str(c.epochs),
            "--per_device_train_batch_size",
            str(c.batch_size),
            "--per_device_eval_batch_size",
            str(c.eval_batch_size),
            "--gradient_accumulation_steps",
            str(c.gradient_accumulation_steps),
            "--learning_rate",
            str(c.learning_rate),
            "--max_source_length",
            str(c.max_source_length),
            "--max_target_length",
            str(c.max_target_length),
            "--seed",
            str(c.seed),
            "--report_to",
            "none",
            "--save_strategy",
            "no",
            "--evaluation_strategy",
            "no",
            "--logging_steps",
            "1",
            "--dataloader_num_workers",
            "0",
            "--no_cuda",
            str(c.device == "cpu"),
        ]
        if c.max_steps is not None:
            command.extend(("--max_steps", str(c.max_steps)))
        plan = dict(
            protocol=PROTOCOL,
            release_id=RELEASE,
            controls=asdict(c),
            split_audit=audit,
            source_commit=checked.manifest["upstream"]["commit"],
            source_script_sha256=_sha(script),
            compatibility_manifest_sha256=checked.manifest_sha256,
            initial_manifest_sha256=initial.manifest_sha256,
            initial_tree_sha256=initial.tree_sha256,
            nltk_manifest_sha256=nltk_manifest_sha256,
            dependency_versions=versions,
            native_argv=command,
            initialization_is_user_selected_not_claimed_original_paper_base=True,
            automatic_retry=False,
        )
        _save(root / "fit-plan.json", plan)
        observed = dict(optimizer_steps=0, loss_events=[], model_generate_calls=0)
        started, cpu = time.monotonic(), time.process_time()
        try:
            with _LOCK, _scope(root, c.seed):
                import nltk
                from safetensors.torch import load_file, save_file

                previous_paths = nltk.data.path[:]
                overrides = dict(
                    NLTK_DATA=str(resources),
                    HF_DATASETS_OFFLINE="1",
                    HF_HUB_DISABLE_TELEMETRY="1",
                    WANDB_DISABLED="true",
                )
                previous_env = {name: os.environ.get(name) for name in overrides}
                nltk.data.path.insert(0, str(resources))
                os.environ.update(overrides)
                try:
                    sys.argv[:] = command
                    with (
                        _native_observers(observed),
                        (root / "native-training.log").open("x") as log,
                        redirect_stdout(log),
                        redirect_stderr(log),
                    ):
                        runpy.run_path(str(script), run_name="__main__")
                finally:
                    nltk.data.path[:] = previous_paths
                    for name, value in previous_env.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
                trainer = observed.pop("trainer")
                if (
                    observed["optimizer_steps"] != audit["planned_optimizer_steps"]
                    or trainer.state.global_step != observed["optimizer_steps"]
                ):
                    raise ValidationError(
                        "RECOMP actual native optimizer steps differ from its bounded plan"
                    )
                if not observed["loss_events"] or not all(
                    math.isfinite(event["loss"]) for event in observed["loss_events"]
                ):
                    raise ValidationError("RECOMP native loss is absent or non-finite")
                if observed["model_generate_calls"]:
                    raise ValidationError(
                        "RECOMP teacher-forced fit unexpectedly generated answers"
                    )
                history = trainer.state.log_history
                initial_tensors = load_file(str(initial.root / "model.safetensors"), device="cpu")
                trained = {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in trainer.model.state_dict().items()
                }
                if set(trained) != set(initial_tensors):
                    raise ValidationError("RECOMP trained model changed its parameter namespace")
                changed = sum(
                    not torch.equal(value, initial_tensors[name]) for name, value in trained.items()
                )
                if not changed:
                    raise ValidationError("RECOMP native fit produced no changed model tensor")
                destination = root / "checkpoint"
                destination.mkdir()
                trainer.model.config.save_pretrained(destination)
                trainer.tokenizer.save_pretrained(destination)
                save_file(
                    trained, str(destination / "model.safetensors"), metadata={"format": "pt"}
                )
                files = {p.name: _sha(p) for p in destination.iterdir() if p.is_file()}
                _save(
                    root / "checkpoint-manifest.json",
                    dict(
                        schema_version="agentinputlab.recomp.checkpoint.v1",
                        mode="abstractive",
                        checkpoint_revision="native-fit-" + content_hash(plan),
                        tokenizer_revision=initial.tokenizer_revision,
                        license="NOASSERTION",
                        files=files,
                    ),
                )
                exported = verify_checkpoint(destination, root / "checkpoint-manifest.json")
                reloaded = load_file(str(destination / "model.safetensors"), device="cpu")
                if not all(torch.equal(reloaded[name], value) for name, value in trained.items()):
                    raise ValidationError(
                        "RECOMP safe export differs from live trained seq2seq model"
                    )
            _save(
                root / "fit-terminal.json",
                dict(
                    status="trained_candidate",
                    **observed,
                    trainer_log_history=history,
                    changed_parameter_tensors=changed,
                    all_exported_tensor_values_equal_live_trained_model=True,
                    native_teacher_forced_pre_and_post_eval=True,
                    generated_quality_metrics=False,
                    wall_seconds=time.monotonic() - started,
                    cpu_seconds=time.process_time() - cpu,
                    training_compute_is_separate_not_free=True,
                    gpu_compute_not_fully_metered=True,
                    full_published_score_parity=False,
                    automatic_retry=False,
                ),
            )
            metadata = dict(
                protocol=PROTOCOL,
                release_id=RELEASE,
                checkpoint_manifest_sha256=exported.manifest_sha256,
                fit_terminal_sha256=_sha(root / "fit-terminal.json"),
                fit_plan_sha256=_sha(root / "fit-plan.json"),
                source_revision=plan["source_commit"],
                heldout_quality_acceptance=False,
                paper_results_changed=False,
            )
            _save(root / "artifact.json", {**metadata, "artifact_id": content_hash(metadata)})
            return RECOMPTrainedSeq2SeqArtifact.load(root)
        except Exception as exc:
            observed.pop("trainer", None)
            if not (root / "fit-terminal.json").exists():
                _save(
                    root / "fit-terminal.json",
                    dict(
                        status="failed_terminal",
                        **observed,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        wall_seconds=time.monotonic() - started,
                        cpu_seconds=time.process_time() - cpu,
                        automatic_retry=False,
                    ),
                )
            raise


__all__ = [
    "RECOMPAbstractiveFitControls",
    "RECOMPAbstractiveTraining",
    "RECOMPTrainedSeq2SeqArtifact",
    "audit_recomp_seq2seq_data",
    "validate_recomp_nltk_assets",
]
