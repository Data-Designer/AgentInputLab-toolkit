"""Pinned Adaptive-RAG silver-label reconstruction and native router training.

The author script owns tokenization, seq2seq loss, AdamW/Accelerate updates and
first-decoder-step A/B/C validation. Run each fit in a dedicated process. No
alternate classifier loss, generated labels, automatic fit retry or test tuning.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import runpy
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .recomp_abstractive import validate_recomp_nltk_assets
from .recomp_extractive import _LOCK, _save, _scope

RELEASE = "acquire.adaptive_rag.rebuilt_offline"
PROTOCOL = "agentinputlab.adaptive-rag.native_router_fit.v1"
SCRIPT = "classifier/run_classifier.py"
_DEPENDENCIES = {
    "transformers": "4.28.1",
    "tokenizers": "0.13.3",
    "accelerate": "0.15.0",
    "numpy": "1.26.4",
    "datasets": "2.14.7",
    "pyarrow": "14.0.2",
    "fsspec": "2023.10.0",
    "huggingface-hub": "0.19.4",
    "nltk": "3.9.2",
    "safetensors": "0.5.3",
    "sentencepiece": "0.2.1",
    "jsonlines": "4.0.0",
}


def _sha(path):
    from workers.adaptive_rag.driver import _file_sha256

    return _file_sha256(path)


def _regular(path, label):
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValidationError(f"Adaptive-RAG {label} requires a regular non-symlink file")
    return supplied.resolve()


def _new_output_root(output_root, *protected):
    root = Path(output_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise ValidationError("Adaptive-RAG output must be create-only; no automatic retry")
    if any(root.resolve().is_relative_to(Path(p).resolve()) for p in protected):
        raise ValidationError("Adaptive-RAG output must not mutate source or initial assets")
    root.mkdir(parents=True)
    return root


def validate_adaptive_training_checkout(checkout):
    from workers.adaptive_rag.driver import (
        DEFAULT_MANIFEST,
        CompatibilityPolicy,
        _git,
        _safe_file,
    )

    root = Path(checkout).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("Adaptive-RAG requires a regular native source checkout")
    root = root.resolve()
    policy = CompatibilityPolicy.from_mapping(json.loads(DEFAULT_MANIFEST.read_text()))
    if (
        _git(root, "rev-parse", "HEAD") != policy.commit
        or _git(root, "rev-parse", "HEAD^{tree}") != policy.tree
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise ExternalDependencyError("Adaptive-RAG training source must be exact and clean")
    sources = {**policy.source_hashes, policy.license_path: policy.license_sha256}
    # Preprocessing is also executed, not just copied as a provenance label.
    relative = "classifier/preprocess/preprocess_utils.py"
    import hashlib

    raw = _safe_file(root, relative).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != _git(root, "rev-parse", policy.commit + ":" + relative):
        raise ExternalDependencyError("Adaptive-RAG native label source differs")
    sources[relative] = hashlib.sha256(raw).hexdigest()
    for relative, digest in sources.items():
        if _sha(_safe_file(root, relative)) != digest:
            raise ExternalDependencyError("Adaptive-RAG pinned source hash differs")
    return root, dict(commit=policy.commit, tree=policy.tree, source_files=sources)


def rebuild_adaptive_silver(
    *,
    checkout,
    original_file,
    zero_file,
    single_file,
    multi_file,
    dataset_name,
    teacher_revision,
    output_root,
    max_data_bytes=32 * 1024 * 1024,
    source_and_asset_acknowledged=False,
):
    """Call native label_complexity on caller-mounted original/prediction files.

    The mappings must be native correct-prediction ID sets from the declared
    teacher. This function does not run a teacher or assert teacher/reader parity.
    Native C then B then A overwrites yield A > B > C; unanswerable IDs excluded.
    """
    if source_and_asset_acknowledged is not True:
        raise ValidationError("Adaptive-RAG label rebuild requires acknowledgement")
    for value in (dataset_name, teacher_revision):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValidationError("Adaptive-RAG dataset and teacher revision must be explicit")
    if type(max_data_bytes) is not int or max_data_bytes < 1:
        raise ValidationError("Adaptive-RAG data byte cap must be positive")
    source, binding = validate_adaptive_training_checkout(checkout)
    paths = {
        name: _regular(path, name)
        for name, path in (
            ("original", original_file),
            ("zero", zero_file),
            ("single", single_file),
            ("multi", multi_file),
        )
    }
    if any(p.stat().st_size > max_data_bytes for p in paths.values()):
        raise ValidationError("Adaptive-RAG label inputs exceed the declared byte cap")
    before = {name: _sha(path) for name, path in paths.items()}
    root = _new_output_root(output_root, source)
    spec = importlib.util.spec_from_file_location(
        "_agentinputlab_pinned_adaptive_labels",
        source / "classifier/preprocess/preprocess_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.label_complexity(
        *(str(paths[n]) for n in ("original", "zero", "single", "multi")), dataset_name
    )
    if before != {name: _sha(path) for name, path in paths.items()}:
        raise ValidationError("Adaptive-RAG label source inputs changed during reconstruction")
    _save(root / "silver.json", rows)
    _save(
        root / "provenance.json",
        dict(
            source=binding,
            native_function="label_complexity",
            dataset=dataset_name,
            teacher_revision=teacher_revision,
            input_files={name: dict(path=str(p), sha256=before[name]) for name, p in paths.items()},
            silver_sha256=_sha(root / "silver.json"),
            rows=len(rows),
            class_counts=dict(Counter(r["answer"] for r in rows)),
            priority=["A", "B", "C"],
            unanswerable_ids_excluded_by_native_source=True,
            new_teacher_calls=0,
            original_reader_matched_teacher_claimed=False,
        ),
    )
    return root / "silver.json"


@dataclass(frozen=True, slots=True)
class AdaptiveRAGFitControls:
    epochs: int = 2
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_seq_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    max_steps: int | None = None
    max_optimizer_steps: int = 10000
    max_data_bytes: int = 64 * 1024 * 1024
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self):
        for name in (
            "epochs",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
            "doc_stride",
            "max_answer_length",
            "max_optimizer_steps",
            "max_data_bytes",
            "seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValidationError(f"Adaptive-RAG {name} must be an integer in range")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise ValidationError(f"Adaptive-RAG {name} is invalid")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValidationError("Adaptive-RAG max_steps must be a positive explicit bound")
        if (
            self.doc_stride >= self.max_seq_length
            or self.max_answer_length > 512
            or self.device not in {"cpu", "cuda"}
        ):
            raise ValidationError("Adaptive-RAG sequence, output cap or device is invalid")


def audit_adaptive_training_data(train_file, dev_file, controls):
    paths = {n: _regular(p, n) for n, p in (("train", train_file), ("dev", dev_file))}
    if paths["train"] == paths["dev"]:
        raise ValidationError("Adaptive-RAG train/dev files must differ")
    evidence, questions, ids = {}, {}, {}
    for split, path in paths.items():
        if path.suffix != ".json" or path.stat().st_size > controls.max_data_bytes:
            raise ValidationError("Adaptive-RAG native JSON data exceeds its declared byte cap")
        try:
            rows = json.loads(path.read_text())
        except (ValueError, UnicodeError) as exc:
            raise ValidationError("Adaptive-RAG data must be native JSON record arrays") from exc
        if not isinstance(rows, list) or not rows:
            raise ValidationError("Adaptive-RAG requires nonempty native records")
        for row in rows:
            if (
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(k), str) or not row[k].strip() or "\x00" in row[k]
                    for k in ("id", "question", "dataset_name", "answer")
                )
                or row["answer"] not in {"A", "B", "C"}
            ):
                raise ValidationError(
                    "Adaptive-RAG native records require id/question/dataset/A-B-C"
                )
        questions[split] = {" ".join(r["question"].casefold().split()) for r in rows}
        ids[split] = {r["id"] for r in rows}
        if len(ids[split]) != len(rows):
            raise ValidationError("Adaptive-RAG native IDs must be unique within each split")
        evidence[split] = dict(
            path=str(path),
            sha256=_sha(path),
            rows=len(rows),
            class_counts=dict(Counter(r["answer"] for r in rows)),
        )
    if questions["train"] & questions["dev"] or ids["train"] & ids["dev"]:
        raise ValidationError("Adaptive-RAG train/dev query or ID overlap is not permitted")
    steps = controls.max_steps or controls.epochs * math.ceil(
        math.ceil(evidence["train"]["rows"] / controls.batch_size)
        / controls.gradient_accumulation_steps
    )
    if steps > controls.max_optimizer_steps:
        raise ValidationError("Adaptive-RAG optimizer plan exceeds its declared step cap")
    return dict(
        files=evidence,
        minimum_planned_optimizer_steps=steps,
        query_and_id_overlap=False,
        overflow_aware_step_cap_checked_before_native_training=True,
    )


def validate_adaptive_router_checkpoint(root, manifest, expected_sha256):
    metadata = load_pinned_json_mapping(manifest, expected_sha256=expected_sha256)
    folder = Path(root).expanduser()
    if folder.is_symlink() or not folder.is_dir():
        raise ValidationError("Adaptive-RAG router checkpoint requires a regular directory")
    folder = folder.resolve()
    files = metadata.get("files")
    if (
        metadata.get("schema") != "agentinputlab.adaptive-rag.router_checkpoint.v1"
        or not isinstance(files, dict)
        or not {"config.json", "model.safetensors", "tokenizer.json"} <= files.keys()
    ):
        raise ValidationError(
            "Adaptive-RAG requires an explicit safe T5/tokenizer checkpoint closure"
        )
    if {p.name for p in folder.iterdir()} != set(files):
        raise ValidationError("Adaptive-RAG checkpoint closure contains missing or extra files")
    for name, expected in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {".", ".."}
            or name.endswith((".bin", ".pkl", ".pickle", ".py"))
        ):
            raise ValidationError("Adaptive-RAG checkpoint paths must be flat and safe")
        if _sha(_regular(folder / name, "checkpoint")) != expected:
            raise ValidationError("Adaptive-RAG checkpoint file hash differs")
    config = json.loads((folder / "config.json").read_text())
    tokenizer_config = folder / "tokenizer_config.json"
    custom_tokenizer = tokenizer_config.exists() and json.loads(tokenizer_config.read_text()).get(
        "auto_map"
    )
    if config.get("model_type") != "t5" or config.get("auto_map") or custom_tokenizer:
        raise ValidationError("Adaptive-RAG native initialization requires local standard T5")
    return folder, metadata


def _t5_tied_aliases(tie_word_embeddings):
    aliases = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }
    if tie_word_embeddings:
        aliases["lm_head.weight"] = "shared.weight"
    return aliases


def _align_t5_initial_keys(initial, expected_keys, *, tie_word_embeddings):
    """Expand only standard T5 tied aliases omitted by safe serialization."""
    expected = set(expected_keys)
    missing = expected - set(initial)
    allowed = _t5_tied_aliases(tie_word_embeddings)
    if set(initial) - expected or missing - set(allowed) or "shared.weight" not in initial:
        raise ValidationError("Adaptive-RAG trained namespace differs from initialization")
    expanded = dict(initial)
    for alias in missing:
        expanded[alias] = initial[allowed[alias]]
    return expanded, {alias: allowed[alias] for alias in sorted(missing)}


@contextmanager
def _observe_native(observed, controls):
    import torch
    import transformers
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, RandomSampler

    prepare, step = Accelerator.prepare, torch.optim.AdamW.step
    generate = transformers.PreTrainedModel.generate
    restores = []

    def preparing(accelerator, *objects, **kwargs):
        for obj in objects:
            if isinstance(obj, torch.nn.Module):
                if "model" in observed:
                    raise ValidationError("Adaptive-RAG native fit prepared more than one model")
                observed["model"] = obj
                forward = obj.forward

                @wraps(forward)
                def forwarding(*args, _original_forward=forward, **kw):
                    result = _original_forward(*args, **kw)
                    if result.loss is not None:
                        observed["loss_events"].append(
                            dict(
                                loss=float(result.loss.detach().cpu()),
                                gradient_enabled=torch.is_grad_enabled(),
                                label_tokens=int((kw["labels"] != -100).sum().item()),
                            )
                        )
                    return result

                obj.forward = forwarding
                restores.append((obj, forward))
            if isinstance(obj, DataLoader) and isinstance(obj.sampler, RandomSampler):
                updates = math.ceil(len(obj) / controls.gradient_accumulation_steps)
                planned = controls.max_steps or controls.epochs * updates
                if planned > controls.max_optimizer_steps:
                    raise ValidationError("Adaptive-RAG overflow-aware optimizer plan exceeds cap")
                observed["planned_optimizer_steps"] = planned
                observed["native_training_batches"] = len(obj)
        return prepare(accelerator, *objects, **kwargs)

    def stepping(optimizer, *args, **kwargs):
        result = step(optimizer, *args, **kwargs)
        observed["optimizer_steps"] += 1
        return result

    def generating(model, *args, **kwargs):
        result = generate(model, *args, **kwargs)
        tokens = result.sequences[:, 1:]
        observed["validation_generation_events"].append(
            dict(
                examples=int(tokens.shape[0]),
                generated_sequence_slots=int(tokens.numel()),
                returned_nonpadding_tokens=int((tokens != model.config.pad_token_id).sum().item()),
                max_length=kwargs.get("max_length"),
            )
        )
        return result

    Accelerator.prepare, torch.optim.AdamW.step = preparing, stepping
    transformers.PreTrainedModel.generate = generating
    try:
        yield
    finally:
        for model, forward in restores:
            model.forward = forward
        Accelerator.prepare, torch.optim.AdamW.step = prepare, step
        transformers.PreTrainedModel.generate = generate


@dataclass(frozen=True, slots=True)
class AdaptiveRAGTrainedRouterArtifact:
    root: Path
    artifact_id: str
    checkpoint_manifest_sha256: str

    @property
    def checkpoint(self):
        return self.root / "checkpoint"

    @classmethod
    def load(cls, root):
        folder = Path(root).expanduser()
        if folder.is_symlink() or not folder.is_dir():
            raise ValidationError("Adaptive-RAG fit artifact directory is invalid")
        folder = folder.resolve()
        metadata = json.loads(_regular(folder / "artifact.json", "artifact").read_text())
        identity = metadata.pop("artifact_id", None)
        if (
            identity != content_hash(metadata)
            or metadata.get("protocol") != PROTOCOL
            or metadata.get("release_id") != RELEASE
        ):
            raise ValidationError("Adaptive-RAG fit artifact identity differs")
        for name in ("fit-plan.json", "fit-terminal.json"):
            if _sha(_regular(folder / name, name)) != metadata.get(name + "_sha256"):
                raise ValidationError("Adaptive-RAG fit plan/terminal hash differs")
        terminal = json.loads((folder / "fit-terminal.json").read_text())
        if (
            terminal.get("status") != "trained_candidate"
            or terminal.get("optimizer_steps", 0) < 1
            or terminal.get("changed_parameter_tensors", 0) < 1
            or not terminal.get("validation_generation_events")
        ):
            raise ValidationError("Adaptive-RAG native fit/validation is incomplete")
        native_files = metadata.get("native_files")
        if (
            not isinstance(native_files, dict)
            or not {
                "pytorch_model.bin",
                "dict_id_pred_results.json",
                "final_eval_results.json",
                "final_eval_results_perClass.json",
            }
            <= native_files.keys()
        ):
            raise ValidationError("Adaptive-RAG native output closure is incomplete")
        if (folder / "native").is_symlink() or any(
            p.is_symlink() for p in (folder / "native").rglob("*")
        ):
            raise ValidationError("Adaptive-RAG native output closure must not contain symlinks")
        actual_files = {
            p.relative_to(folder / "native").as_posix()
            for p in (folder / "native").rglob("*")
            if p.is_file()
        }
        if actual_files != set(native_files):
            raise ValidationError(
                "Adaptive-RAG native output closure contains extra or missing files"
            )
        for relative, expected in native_files.items():
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or str(p) != relative:
                raise ValidationError("Adaptive-RAG native closure path is invalid")
            if _sha(_regular(folder / "native" / p, "native output")) != expected:
                raise ValidationError("Adaptive-RAG native output closure differs")
        digest = metadata["checkpoint_manifest_sha256"]
        validate_adaptive_router_checkpoint(
            folder / "checkpoint", folder / "checkpoint-manifest.json", digest
        )
        return cls(folder, identity, digest)

    def router_bindings(self):
        """Bind the new snapshot to the existing controlled router deployment.

        Generator/retriever/corpus/index identities remain independently required.
        This does not promote controlled retrieval arms to upstream score parity.
        """
        from workers.adaptive_rag.driver import _path_closure_sha256

        current = self.load(self.root)
        return dict(
            router_revision="native-fit-" + current.artifact_id,
            router_snapshot_sha256=_path_closure_sha256(current.checkpoint),
            router_weights_sha256=_sha(current.checkpoint / "model.safetensors"),
            router_tokenizer_sha256=_sha(current.checkpoint / "tokenizer.json"),
        )

    def router_configuration(self):
        """Keep native fit/validation input truncation explicit at deployment."""
        current = self.load(self.root)
        plan = json.loads((current.root / "fit-plan.json").read_text())
        return dict(max_input_tokens=plan["controls"]["max_seq_length"])


class AdaptiveRAGOfflineTraining:
    release_id = RELEASE

    def __init__(self, controls=None):
        self.controls = (
            AdaptiveRAGFitControls(**controls)
            if isinstance(controls, Mapping)
            else controls or AdaptiveRAGFitControls()
        )
        if not isinstance(self.controls, AdaptiveRAGFitControls):
            raise ValidationError("Adaptive-RAG fit requires typed controls")

    def fit(
        self,
        *,
        checkout,
        initial_checkpoint,
        initial_checkpoint_manifest,
        initial_manifest_sha256,
        train_file,
        dev_file,
        data_provenance,
        data_provenance_sha256,
        nltk_root,
        nltk_manifest,
        nltk_manifest_sha256,
        output_root,
        source_and_asset_acknowledged=False,
    ):
        if source_and_asset_acknowledged is not True:
            raise ValidationError(
                "Adaptive-RAG fit requires explicit source/model/data acknowledgement"
            )
        source, binding = validate_adaptive_training_checkout(checkout)
        initial, _ = validate_adaptive_router_checkpoint(
            initial_checkpoint, initial_checkpoint_manifest, initial_manifest_sha256
        )
        c = self.controls
        audit = audit_adaptive_training_data(train_file, dev_file, c)
        provenance = load_pinned_json_mapping(
            data_provenance, expected_sha256=data_provenance_sha256
        )
        if (
            provenance.get("schema") != "agentinputlab.adaptive-rag.training_data.v1"
            or not isinstance(provenance.get("teacher_revision"), str)
            or not provenance["teacher_revision"].strip()
            or any(
                provenance.get("files", {}).get(split, {}).get("sha256")
                != audit["files"][split]["sha256"]
                for split in ("train", "dev")
            )
        ):
            raise ValidationError(
                "Adaptive-RAG training provenance must bind both native data files and teacher"
            )
        resources = validate_recomp_nltk_assets(nltk_root, nltk_manifest, nltk_manifest_sha256)
        versions = {name: importlib.metadata.version(name) for name in ("torch", *_DEPENDENCIES)}
        if versions["torch"] not in {"1.13.0", "1.13.0+cpu", "1.13.0+cu117"} or any(
            versions[k] != v for k, v in _DEPENDENCIES.items()
        ):
            raise ExternalDependencyError(
                "Adaptive-RAG fit requires documented isolated dependency versions"
            )
        if os.environ.get("WORLD_SIZE", "1") != "1":
            raise ValidationError("Adaptive-RAG bounded fit requires one process")
        import torch

        if (c.device == "cpu" and torch.cuda.is_available()) or (
            c.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1)
        ):
            raise ValidationError("Select no visible CUDA devices for CPU or exactly one for CUDA")
        from safetensors.torch import load_file

        initial_tensors = load_file(str(initial / "model.safetensors"), device="cpu")
        tied_words = json.loads((initial / "config.json").read_text()).get(
            "tie_word_embeddings", True
        )
        for alias, canonical in _t5_tied_aliases(tied_words).items():
            if alias in initial_tensors and (
                canonical not in initial_tensors
                or not torch.equal(initial_tensors[alias], initial_tensors[canonical])
            ):
                raise ValidationError("Adaptive-RAG initial tied T5 weights disagree")
        root = _new_output_root(output_root, source, initial)
        (root / "native").mkdir()
        command = [
            str(source / SCRIPT),
            "--do_train",
            "--do_eval",
            "--model_name_or_path",
            str(initial),
            "--train_file",
            audit["files"]["train"]["path"],
            "--validation_file",
            audit["files"]["dev"]["path"],
            "--question_column",
            "question",
            "--answer_column",
            "answer",
            "--output_dir",
            str(root / "native"),
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
            "--weight_decay",
            str(c.weight_decay),
            "--max_seq_length",
            str(c.max_seq_length),
            "--doc_stride",
            str(c.doc_stride),
            "--max_answer_length",
            str(c.max_answer_length),
            "--val_max_answer_length",
            str(c.max_answer_length),
            "--seed",
            str(c.seed),
        ]
        if c.max_steps is not None:
            command.extend(("--max_train_steps", str(c.max_steps)))
        plan = dict(
            protocol=PROTOCOL,
            release_id=RELEASE,
            controls=asdict(c),
            source=binding,
            split_audit=audit,
            data_provenance=provenance,
            data_provenance_sha256=data_provenance_sha256,
            initial_manifest_sha256=initial_manifest_sha256,
            nltk_manifest_sha256=nltk_manifest_sha256,
            dependency_versions=versions,
            native_argv=command,
            dependency_lock_adaptation_not_original_git_transformers=True,
            original_paper_initialization_or_teacher_parity_claimed=False,
            automatic_retry=False,
        )
        _save(root / "fit-plan.json", plan)
        observed = dict(optimizer_steps=0, loss_events=[], validation_generation_events=[])
        started, cpu = time.monotonic(), time.process_time()
        try:
            with _LOCK, _scope(root, c.seed):
                import nltk
                from safetensors.torch import load_file, save_file
                from transformers import AutoTokenizer

                paths, modules = sys.path[:], sys.modules.pop("utils", None)
                nltk_paths = nltk.data.path[:]
                environment = {
                    k: os.environ.get(k)
                    for k in (
                        "TRANSFORMERS_CACHE",
                        "HF_DATASETS_CACHE",
                        "NLTK_DATA",
                        "HF_DATASETS_OFFLINE",
                    )
                }
                sys.path.insert(0, str(source / "classifier"))
                nltk.data.path.insert(0, str(resources))
                os.environ.update(
                    NLTK_DATA=str(resources),
                    HF_DATASETS_OFFLINE="1",
                    HF_DATASETS_CACHE=str(root / "dataset-cache"),
                )
                sys.argv[:] = command
                try:
                    with (
                        _observe_native(observed, c),
                        (root / "native-training.log").open("x") as log,
                        redirect_stdout(log),
                        redirect_stderr(log),
                    ):
                        runpy.run_path(str(source / SCRIPT), run_name="__main__")
                finally:
                    sys.path[:] = paths
                    sys.modules.pop("utils", None)
                    if modules is not None:
                        sys.modules["utils"] = modules
                    nltk.data.path[:] = nltk_paths
                    for name, value in environment.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
                model = observed.pop("model")
                if (
                    observed["optimizer_steps"] != observed.get("planned_optimizer_steps")
                    or not observed["loss_events"]
                    or not all(math.isfinite(e["loss"]) for e in observed["loss_events"])
                ):
                    raise ValidationError("Adaptive-RAG native optimizer steps or losses differ")
                validation = json.loads((root / "native/dict_id_pred_results.json").read_text())
                dev = json.loads(Path(audit["files"]["dev"]["path"]).read_text())
                if set(validation) != {r["id"] for r in dev} or any(
                    validation[r["id"]]["answer"] != r["answer"]
                    or validation[r["id"]]["prediction"] not in {"A", "B", "C"}
                    for r in dev
                ):
                    raise ValidationError("Adaptive-RAG native validation records differ")
                accuracy = (
                    100
                    * sum(validation[r["id"]]["prediction"] == r["answer"] for r in dev)
                    / len(dev)
                )
                native_accuracy = json.loads((root / "native/final_eval_results.json").read_text())[
                    "final_acc_score"
                ]
                if not math.isclose(accuracy, native_accuracy, rel_tol=1e-12):
                    raise ValidationError("Adaptive-RAG native validation accuracy differs")
                live = model.state_dict()
                before, aliases = _align_t5_initial_keys(
                    initial_tensors, live, tie_word_embeddings=tied_words
                )
                for alias, canonical in aliases.items():
                    if live[alias].data_ptr() != live[canonical].data_ptr():
                        raise ValidationError(
                            "Adaptive-RAG safe omitted key is not a native tied weight"
                        )
                trained = {n: t.detach().cpu().contiguous().clone() for n, t in live.items()}
                changed = sum(not torch.equal(t, before[n]) for n, t in trained.items())
                if not changed:
                    raise ValidationError("Adaptive-RAG native fit changed no model tensors")
                destination = root / "checkpoint"
                destination.mkdir()
                model.config.save_pretrained(destination)
                AutoTokenizer.from_pretrained(
                    str(initial), local_files_only=True, trust_remote_code=False
                ).save_pretrained(destination)
                save_file(
                    trained, str(destination / "model.safetensors"), metadata={"format": "pt"}
                )
                files = {p.name: _sha(p) for p in destination.iterdir()}
                _save(
                    root / "checkpoint-manifest.json",
                    dict(schema="agentinputlab.adaptive-rag.router_checkpoint.v1", files=files),
                )
                digest = _sha(root / "checkpoint-manifest.json")
                validate_adaptive_router_checkpoint(
                    destination, root / "checkpoint-manifest.json", digest
                )
                exported = load_file(str(destination / "model.safetensors"), device="cpu")
                if not all(torch.equal(t, exported[n]) for n, t in trained.items()):
                    raise ValidationError(
                        "Adaptive-RAG safe export differs from live trained states"
                    )
            _save(
                root / "fit-terminal.json",
                dict(
                    status="trained_candidate",
                    **observed,
                    changed_parameter_tensors=changed,
                    initial_safe_tensor_count=len(initial_tensors),
                    exported_tensor_count=len(trained),
                    initial_native_tied_aliases_expanded=aliases,
                    validation_rows=len(dev),
                    native_validation_accuracy=accuracy,
                    wall_seconds=time.monotonic() - started,
                    cpu_seconds=time.process_time() - cpu,
                    training_and_validation_compute_separate_not_free=True,
                    GPU_compute_not_fully_metered=True,
                    native_generation_costs_included=True,
                    original_published_score_parity=False,
                    automatic_retry=False,
                ),
            )
            metadata = dict(
                protocol=PROTOCOL,
                release_id=RELEASE,
                checkpoint_manifest_sha256=digest,
                **{n + "_sha256": _sha(root / n) for n in ("fit-plan.json", "fit-terminal.json")},
                native_files={
                    p.relative_to(root / "native").as_posix(): _sha(p)
                    for p in (root / "native").rglob("*")
                    if p.is_file()
                },
                full_quality_acceptance=False,
                paper_results_changed=False,
            )
            _save(root / "artifact.json", dict(**metadata, artifact_id=content_hash(metadata)))
            return AdaptiveRAGTrainedRouterArtifact.load(root)
        except Exception as exc:
            observed.pop("model", None)
            if not (root / "fit-terminal.json").exists():
                _save(
                    root / "fit-terminal.json",
                    dict(
                        status="failed_terminal",
                        **observed,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        automatic_retry=False,
                        wall_seconds=time.monotonic() - started,
                        cpu_seconds=time.process_time() - cpu,
                    ),
                )
            raise


__all__ = [
    "AdaptiveRAGFitControls",
    "AdaptiveRAGOfflineTraining",
    "AdaptiveRAGTrainedRouterArtifact",
    "audit_adaptive_training_data",
    "rebuild_adaptive_silver",
    "validate_adaptive_router_checkpoint",
]
