"""Native LLMLingua-2 word alignment, filtering and token-classifier training.

Only mounted teacher-compressed records are accepted: this recipe does not
silently buy GPT-4 calls, invent retained-word labels, or increase inference
budgets. Native label/fit scripts run in a dedicated, offline process. Original
Adam, loss, padding labels, clipping order and random 80/20 split are retained.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import random
import runpy
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .recomp_extractive import _LOCK, _save, _scope

RELEASE = "training.llmlingua2.distillation"
PROTOCOL = "agentinputlab.llmlingua2.native_token_fit.v1"
COMMIT = "e0e9d99beb94098bbd924aa53c2c112eac41c758"
BASE = "experiments/llmlingua2/"
TRAIN_SCRIPT = BASE + "model_training/train_roberta.py"
SOURCE_BLOBS = {
    TRAIN_SCRIPT: "a5ae084d4882a9c50de76b2db3428a17bf8de39c",
    BASE + "model_training/utils.py": "87877f6b4d59b5e7fb926d1da512b8e299f75875",
    BASE + "data_collection/label_word.py": "25e82a6171e29c1a2e93bb0f9e3a4e6ac5467ecd",
    BASE + "data_collection/filter.py": "5b314177a87d4bd0f4eeaac72430b0c07940ed76",
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path, label):
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValidationError(f"LLMLingua-2 {label} requires a regular non-symlink file")
    return supplied.resolve()


def _new_root(path, *protected):
    root = Path(path).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise ValidationError("LLMLingua-2 output must be create-only; no automatic retry")
    if any(root.resolve().is_relative_to(Path(p).resolve()) for p in protected):
        raise ValidationError("LLMLingua-2 output must not mutate mounted source or assets")
    root.mkdir(parents=True)
    return root


def validate_llmlingua_training_checkout(checkout):
    from workers.adaptive_rag.driver import _git, _safe_file

    supplied = Path(checkout).expanduser()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValidationError("LLMLingua-2 requires a regular pinned source checkout")
    root = supplied.resolve()
    if _git(root, "rev-parse", "HEAD") != COMMIT or _git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise ExternalDependencyError("LLMLingua-2 training source must be exact and clean")
    evidence = {}
    for name, expected in SOURCE_BLOBS.items():
        raw = _safe_file(root, name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected or _git(root, "rev-parse", COMMIT + ":" + name) != expected:
            raise ExternalDependencyError("LLMLingua-2 native script differs from pinned source")
        evidence[name] = hashlib.sha256(raw).hexdigest()
    return root, dict(commit=COMMIT, source_files=evidence)


def _closed_files(folder, manifest, digest, schema):
    data = load_pinned_json_mapping(manifest, expected_sha256=digest)
    root = Path(folder).expanduser()
    if root.is_symlink() or not root.is_dir() or data.get("schema") != schema:
        raise ValidationError("LLMLingua-2 mounted asset/manifest is invalid")
    root = root.resolve()
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValidationError("LLMLingua-2 requires a complete mounted file closure")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValidationError("LLMLingua-2 asset symlinks are not permitted")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != set(files):
        raise ValidationError("LLMLingua-2 asset closure has extra or missing files")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValidationError("LLMLingua-2 asset relative path is unsafe")
        if _sha(_regular(root / path, "asset")) != expected:
            raise ValidationError("LLMLingua-2 mounted asset hash differs")
    return root, data


def validate_llmlingua_initial_checkpoint(root, manifest, digest):
    folder, data = _closed_files(
        root, manifest, digest, "agentinputlab.llmlingua2.token_checkpoint.v1"
    )
    if not {"model.safetensors", "config.json", "tokenizer.json"} <= data["files"].keys():
        raise ValidationError("LLMLingua-2 requires a safe model and tokenizer closure")
    if any(name.endswith((".bin", ".pkl", ".pickle", ".py")) for name in data["files"]):
        raise ValidationError("LLMLingua-2 unsafe model payload is not accepted")
    config = json.loads((folder / "config.json").read_text())
    tokenizer = json.loads((folder / "tokenizer_config.json").read_text())
    if not isinstance(config, dict) or not isinstance(tokenizer, dict):
        raise ValidationError("LLMLingua-2 model/tokenizer configuration must be objects")
    kind = config.get("model_type")
    marker = {"bert": "bert-base-multilingual-cased", "xlm-roberta": "xlm-roberta-large"}
    # Standard HF configs omit the default two-label mapping on serialization.
    # A local strict from_pretrained below still rejects a mismatched tensor head.
    labels = config.get("id2label")
    label_count = len(labels) if isinstance(labels, dict) else config.get("num_labels", 2)
    if (
        kind not in marker
        or config.get("auto_map")
        or tokenizer.get("auto_map")
        or (labels is not None and not isinstance(labels, dict))
        or type(label_count) is not int
        or label_count != 2
    ):
        raise ValidationError("LLMLingua-2 native path requires standard two-label BERT/XLM-R")
    return folder, data


def _validate_nlp(root, manifest, digest):
    folder, data = _closed_files(root, manifest, digest, "agentinputlab.llmlingua2.spacy_model.v1")
    meta = json.loads((folder / "meta.json").read_text())
    if meta.get("lang") != "en" or meta.get("name") != "core_web_sm":
        raise ValidationError("LLMLingua-2 requires the declared English small spaCy model")
    if any(name.endswith(".py") for name in data["files"]):
        raise ValidationError("LLMLingua-2 NLP asset must not contain executable Python")
    return folder, data


@contextmanager
def _nlp_mount(path):
    import spacy

    original = spacy.load

    def load(name, *args, **kwargs):
        if name != "en_core_web_sm" or args or kwargs:
            raise ValidationError("LLMLingua-2 unexpected NLP load")
        return original(str(path))

    spacy.load = load
    try:
        yield
    finally:
        spacy.load = original


def label_llmlingua_compressed_records(
    *,
    checkout,
    records_file,
    data_provenance,
    data_provenance_sha256,
    nlp_root,
    nlp_manifest,
    nlp_manifest_sha256,
    output_root,
    window_size=400,
    filter_native=True,
    max_data_bytes=32 * 1024 * 1024,
    source_and_asset_acknowledged=False,
):
    """Run native alignment/filter over caller-mounted real teacher outputs.

    JSON only at the public boundary. The upstream PT intermediary is created
    locally from those checked records and loaded with weights_only=True. The
    native alignment loop and filtering percentiles are not replaced.
    """
    if source_and_asset_acknowledged is not True:
        raise ValidationError("LLMLingua-2 requires explicit source/data acknowledgement")
    if type(window_size) is not int or window_size < 1 or type(filter_native) is not bool:
        raise ValidationError("LLMLingua-2 alignment controls are invalid")
    if type(max_data_bytes) is not int or max_data_bytes < 1:
        raise ValidationError("LLMLingua-2 data byte cap must be positive")
    source, binding = validate_llmlingua_training_checkout(checkout)
    path = _regular(records_file, "teacher records")
    if path.suffix != ".json" or path.stat().st_size > max_data_bytes:
        raise ValidationError("LLMLingua-2 teacher JSON exceeds byte cap")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValidationError("LLMLingua-2 requires nonempty real compression records")
    seen = set()
    for row in records:
        if (
            not isinstance(row, dict)
            or not all(
                isinstance(row.get(k), list)
                and row[k]
                and all(isinstance(t, str) and t.strip() and "\0" not in t for t in row[k])
                for k in ("prompt_list", "compressed_prompt_list")
            )
            or len(row["prompt_list"]) != len(row["compressed_prompt_list"])
        ):
            raise ValidationError("LLMLingua-2 requires paired nonempty original/teacher chunks")
        for text in row["prompt_list"]:
            identity = " ".join(text.casefold().split())
            if identity in seen:
                raise ValidationError("LLMLingua-2 duplicated original chunks are not permitted")
            seen.add(identity)
    provenance = load_pinned_json_mapping(data_provenance, expected_sha256=data_provenance_sha256)
    if (
        provenance.get("records_sha256") != _sha(path)
        or not isinstance(provenance.get("teacher_revision"), str)
        or not provenance["teacher_revision"].strip()
    ):
        raise ValidationError("LLMLingua-2 data provenance must bind real teacher outputs")
    nlp, _ = _validate_nlp(nlp_root, nlp_manifest, nlp_manifest_sha256)
    root = _new_root(output_root, source, nlp, path.parent)
    import torch

    original_load, original_save = torch.load, torch.save
    previous_datasets = sys.modules.get("datasets")
    mounted = ModuleType("datasets")

    def load_dataset(name, *, split):
        if name != str(path) or split != "train":
            raise ValidationError("LLMLingua-2 unexpected native dataset access")
        return records

    mounted.load_dataset = load_dataset

    def save_plain(value, target, *args, **kwargs):
        # defaultdict -> ordinary dict changes serialization only, not values.
        return original_save(
            dict(value) if isinstance(value, Mapping) else value, target, *args, **kwargs
        )

    def load_safe(target, *args, **kwargs):
        if args or kwargs or Path(target).resolve() != root / "alignment.pt":
            raise ValidationError("LLMLingua-2 unexpected native PT input")
        return original_load(target, map_location="cpu", weights_only=True)

    plan = dict(
        source=binding,
        records_sha256=_sha(path),
        data_provenance=provenance,
        source_chunks=len(seen),
        window_size=window_size,
        native_filter=filter_native,
        new_teacher_calls=0,
        nlp_manifest_sha256=nlp_manifest_sha256,
    )
    _save(root / "label-plan.json", plan)
    try:
        with _LOCK, _scope(root, 17), _nlp_mount(nlp), (root / "native.log").open("x") as log:
            sys.modules["datasets"] = mounted
            torch.load, torch.save = load_safe, save_plain
            with redirect_stdout(log), redirect_stderr(log):
                sys.argv = [
                    str(source / (BASE + "data_collection/label_word.py")),
                    "--load_prompt_from",
                    str(path),
                    "--save_path",
                    str(root / "alignment.json"),
                    "--window_size",
                    str(window_size),
                ]
                runpy.run_path(sys.argv[0], run_name="__main__")
                if filter_native:
                    sys.argv = [
                        str(source / (BASE + "data_collection/filter.py")),
                        "--load_path",
                        str(root / "alignment.pt"),
                        "--save_path",
                        str(root / "filtered.pt"),
                    ]
                    runpy.run_path(sys.argv[0], run_name="__main__")
            selected = original_load(
                root / ("filtered.pt" if filter_native else "alignment.pt"),
                map_location="cpu",
                weights_only=True,
            )
            _save(root / "training-data.json", selected)
    except Exception as exc:
        _save(
            root / "label-failed.json",
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                new_teacher_calls=0,
                automatic_retry=False,
            ),
        )
        raise
    finally:
        torch.load, torch.save = original_load, original_save
        if previous_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = previous_datasets
    _save(
        root / "label-terminal.json",
        dict(
            rows=len(selected["origin"]),
            new_teacher_calls=0,
            native_alignment=True,
            native_filter=filter_native,
            output_sha256=_sha(root / "training-data.json"),
            plan_sha256=_sha(root / "label-plan.json"),
        ),
    )
    return root / "training-data.json"


@dataclass(frozen=True, slots=True)
class LLMLingua2FitControls:
    epochs: int = 10
    batch_size: int = 10
    learning_rate: float = 1e-5
    seed: int = 17
    max_optimizer_steps: int = 10000
    max_data_bytes: int = 32 * 1024 * 1024
    device: str = "cpu"

    def __post_init__(self):
        for name in ("epochs", "batch_size", "seed", "max_optimizer_steps", "max_data_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValidationError(f"LLMLingua-2 {name} must be an integer in range")
        if (
            type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.device not in {"cpu", "cuda"}
        ):
            raise ValidationError("LLMLingua-2 learning rate or device is invalid")


def audit_llmlingua_training_data(path, controls):
    path = _regular(path, "training JSON")
    if path.suffix != ".json" or path.stat().st_size > controls.max_data_bytes:
        raise ValidationError("LLMLingua-2 native JSON exceeds byte cap")
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("origin"), list)
        or not isinstance(data.get("labels"), list)
        or len(data["origin"]) != len(data["labels"])
        or len(data["origin"]) < 5
    ):
        raise ValidationError("LLMLingua-2 requires at least five paired original/label rows")
    keys = []
    for text, labels in zip(data["origin"], data["labels"], strict=True):
        if (
            not isinstance(text, str)
            or not text.strip()
            or "\0" in text
            or not isinstance(labels, list)
            or not labels
            or any(type(value) is not bool for value in labels)
        ):
            raise ValidationError("LLMLingua-2 labels must be native retained-word booleans")
        keys.append(" ".join(text.casefold().split()))
    if len(set(keys)) != len(keys):
        raise ValidationError("LLMLingua-2 split cannot include duplicated original text")
    order = list(range(len(keys)))
    random.Random(controls.seed).shuffle(order)
    cut = int(len(keys) * 0.8)
    planned = math.ceil(cut / controls.batch_size) * controls.epochs
    if planned > controls.max_optimizer_steps:
        raise ValidationError("LLMLingua-2 native optimizer plan exceeds step cap; not truncated")
    return data, dict(
        path=str(path),
        sha256=_sha(path),
        rows=len(keys),
        native_train_indices=order[:cut],
        native_validation_indices=order[cut:],
        planned_optimizer_steps=planned,
        split_policy="native_random_shuffle_80_20",
        source_query_overlap=False,
        token_label_alignment_checked_by_native_dataset=True,
    )


def _tensor_hash(tensor, torch):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class LLMLingua2TrainedArtifact:
    root: Path
    artifact_id: str

    @property
    def checkpoint(self):
        return self.root / "checkpoint"

    def intervention(self, *, config=None, device_map="cpu"):
        """Deploy this fitted candidate without impersonating a released model."""
        from .llmlingua2_inference import LLMLingua2TrainedCompressor

        return LLMLingua2TrainedCompressor(self, config=config, device_map=device_map)

    @classmethod
    def load(cls, root):
        supplied = Path(root).expanduser()
        if supplied.is_symlink() or not supplied.is_dir():
            raise ValidationError("LLMLingua-2 trained artifact directory is invalid")
        folder = supplied.resolve()
        metadata = json.loads(_regular(folder / "artifact.json", "artifact").read_text())
        identity = metadata.pop("artifact_id", None)
        if (
            identity != content_hash(metadata)
            or metadata.get("protocol") != PROTOCOL
            or metadata.get("release_id") != RELEASE
        ):
            raise ValidationError("LLMLingua-2 trained artifact identity differs")
        for name in ("fit-plan.json", "fit-terminal.json"):
            if _sha(_regular(folder / name, name)) != metadata.get(name + "_sha256"):
                raise ValidationError("LLMLingua-2 trained evidence hash differs")
        terminal = json.loads((folder / "fit-terminal.json").read_text())
        if (
            terminal.get("status") != "trained_candidate"
            or terminal.get("optimizer_steps", 0) < 1
            or terminal.get("changed_parameter_tensors", 0) < 1
        ):
            raise ValidationError("LLMLingua-2 native training did not complete")
        validate_llmlingua_initial_checkpoint(
            folder / "checkpoint",
            folder / "checkpoint-manifest.json",
            metadata["checkpoint_manifest_sha256"],
        )
        native_files = metadata.get("native_files")
        native_name = metadata.get("native_directory")
        if (
            not isinstance(native_name, str)
            or Path(native_name).name != native_name
            or native_name
            not in {"native-bert-base-multilingual-cased", "native-xlm-roberta-large"}
            or not isinstance(native_files, dict)
        ):
            raise ValidationError("LLMLingua-2 native output evidence is incomplete")
        native = folder / native_name
        if (
            native.is_symlink()
            or not native.is_dir()
            or any(p.is_symlink() for p in native.rglob("*"))
        ):
            raise ValidationError("LLMLingua-2 native output closure is invalid")
        if {p.relative_to(native).as_posix() for p in native.rglob("*") if p.is_file()} != set(
            native_files
        ):
            raise ValidationError("LLMLingua-2 native output files differ")
        for name, digest in native_files.items():
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or Path(name).as_posix() != name
            ):
                raise ValidationError("LLMLingua-2 native path is invalid")
            if _sha(_regular(native / name, "native output")) != digest:
                raise ValidationError("LLMLingua-2 native output hash differs")
        return cls(folder, identity)


class LLMLingua2DistillationTraining:
    release_id = RELEASE

    def __init__(self, controls=None):
        self.controls = (
            LLMLingua2FitControls(**controls)
            if isinstance(controls, Mapping)
            else (controls or LLMLingua2FitControls())
        )
        if not isinstance(self.controls, LLMLingua2FitControls):
            raise ValidationError("LLMLingua-2 fit requires typed controls")

    def fit(
        self,
        *,
        checkout,
        initial_checkpoint,
        initial_checkpoint_manifest,
        initial_manifest_sha256,
        training_file,
        data_provenance,
        data_provenance_sha256,
        nlp_root,
        nlp_manifest,
        nlp_manifest_sha256,
        output_root,
        source_and_asset_acknowledged=False,
    ):
        if source_and_asset_acknowledged is not True:
            raise ValidationError(
                "LLMLingua-2 fit requires explicit source/model/data acknowledgement"
            )
        source, binding = validate_llmlingua_training_checkout(checkout)
        initial, _ = validate_llmlingua_initial_checkpoint(
            initial_checkpoint, initial_checkpoint_manifest, initial_manifest_sha256
        )
        model_kind = json.loads((initial / "config.json").read_text())["model_type"]
        native_marker = {
            "bert": "bert-base-multilingual-cased",
            "xlm-roberta": "xlm-roberta-large",
        }[model_kind]
        if native_marker not in str(initial):
            raise ValidationError(
                "LLMLingua-2 initial path must retain native model-family substring"
            )
        nlp, _ = _validate_nlp(nlp_root, nlp_manifest, nlp_manifest_sha256)
        c = self.controls
        data, audit = audit_llmlingua_training_data(training_file, c)
        provenance = load_pinned_json_mapping(
            data_provenance, expected_sha256=data_provenance_sha256
        )
        if (
            provenance.get("training_sha256") != audit["sha256"]
            or not isinstance(provenance.get("teacher_revision"), str)
            or not provenance["teacher_revision"].strip()
        ):
            raise ValidationError("LLMLingua-2 provenance must bind real teacher labels")
        import torch
        import transformers

        if (
            (c.device == "cpu" and torch.cuda.is_available())
            or (
                c.device == "cuda"
                and (not torch.cuda.is_available() or torch.cuda.device_count() != 1)
            )
            or os.environ.get("WORLD_SIZE", "1") != "1"
        ):
            raise ValidationError(
                "LLMLingua-2 fit requires one process and declared CUDA visibility"
            )
        root = _new_root(output_root, source, initial, nlp, Path(audit["path"]).parent)
        # Keep native model-family substring branch without rewriting source.
        checkpoint_name = (
            "bert-base-multilingual-cased"
            if "bert-base-multilingual-cased" in str(initial)
            else "xlm-roberta-large"
        )
        checkpoint = root / "checkpoint"
        native = root / ("native-" + checkpoint_name)
        native.mkdir()
        command = [
            str(source / TRAIN_SCRIPT),
            "--model_name",
            str(initial),
            "--data_path",
            audit["path"],
            "--save_path",
            str(native),
            "--num_epoch",
            str(c.epochs),
            "--batch_size",
            str(c.batch_size),
            "--lr",
            str(c.learning_rate),
        ]
        plan = dict(
            protocol=PROTOCOL,
            release_id=RELEASE,
            source=binding,
            controls=asdict(c),
            data_audit=audit,
            data_provenance=provenance,
            initial_manifest_sha256=initial_manifest_sha256,
            nlp_manifest_sha256=nlp_manifest_sha256,
            native_argv=command,
            dependencies={
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "spacy",
                    "numpy",
                    "scikit-learn",
                    "tensorboard",
                    "safetensors",
                )
            },
            label_type_argument_unused_by_native=True,
            native_pad_labels_and_clip_order_preserved=True,
            new_teacher_calls=0,
            automatic_retry=False,
            paper_result_parity_claimed=False,
        )
        _save(root / "fit-plan.json", plan)
        observed = dict(optimizer_steps=0, loss_events=[])
        original_load, original_step = torch.load, torch.optim.Adam.step
        original_model_loader = transformers.AutoModelForTokenClassification.from_pretrained
        original_tokenizer_loader = transformers.AutoTokenizer.from_pretrained
        original_model_descriptor = inspect.getattr_static(
            transformers.AutoModelForTokenClassification, "from_pretrained"
        )
        original_tokenizer_descriptor = inspect.getattr_static(
            transformers.AutoTokenizer, "from_pretrained"
        )
        previous_utils = sys.modules.get("utils")
        spec = importlib.util.spec_from_file_location(
            "utils", source / (BASE + "model_training/utils.py")
        )
        utils = importlib.util.module_from_spec(spec)

        def load_data(path, *args, **kwargs):
            if args or kwargs or str(path) != audit["path"]:
                raise ValidationError("LLMLingua-2 unexpected native serialized input")
            return data

        def load_model(name, *args, **kwargs):
            if (
                name != str(initial)
                or args
                or kwargs != dict(num_labels=2, ignore_mismatched_sizes=True)
            ):
                raise ValidationError("LLMLingua-2 unexpected native model load")
            # No random replacement head or remote code: initial two-label shape
            # must already match. The observed model/loss/train loop is native.
            model, loading = original_model_loader(
                name,
                num_labels=2,
                ignore_mismatched_sizes=False,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                output_loading_info=True,
            )
            if any(
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValidationError(
                    "LLMLingua-2 initialization has missing or incompatible weights"
                )
            observed["model"] = model
            observed["initial_tensors"] = {
                k: _tensor_hash(v, torch) for k, v in model.state_dict().items()
            }
            forward = model.forward

            def forwarding(*args, **kwargs):
                result = forward(*args, **kwargs)
                if result.loss is not None:
                    value = float(result.loss.detach().cpu())
                    if not math.isfinite(value):
                        raise ValidationError("LLMLingua-2 native loss is not finite")
                    observed["loss_events"].append(
                        dict(
                            loss=value,
                            training=bool(model.training),
                            examples=int(kwargs["labels"].shape[0]),
                            native_label_slots=int(kwargs["labels"].numel()),
                        )
                    )
                return result

            model.forward = forwarding
            return model

        def load_tokenizer(name, *args, **kwargs):
            if name != str(initial) or args or kwargs:
                raise ValidationError("LLMLingua-2 unexpected native tokenizer load")
            return original_tokenizer_loader(name, local_files_only=True, trust_remote_code=False)

        def step(optimizer, *args, **kwargs):
            if observed["optimizer_steps"] >= audit["planned_optimizer_steps"]:
                raise ValidationError("LLMLingua-2 native optimizer exceeded declared plan")
            result = original_step(optimizer, *args, **kwargs)
            observed["optimizer_steps"] += 1
            return result

        started, cpu = time.monotonic(), time.process_time()
        try:
            with (
                _LOCK,
                _scope(root, c.seed),
                _nlp_mount(nlp),
                (root / "native.log").open("x") as log,
            ):
                spec.loader.exec_module(utils)
                sys.modules["utils"] = utils
                torch.load, torch.optim.Adam.step = load_data, step
                transformers.AutoModelForTokenClassification.from_pretrained = load_model
                transformers.AutoTokenizer.from_pretrained = load_tokenizer
                sys.argv = command
                with redirect_stdout(log), redirect_stderr(log):
                    namespace = runpy.run_path(str(source / TRAIN_SCRIPT), run_name="__main__")
                namespace["writer"].close()
                if observed["optimizer_steps"] != audit["planned_optimizer_steps"]:
                    raise ValidationError("LLMLingua-2 native optimizer plan was not completed")
                expected_train = [data["origin"][i] for i in audit["native_train_indices"]]
                expected_val = [data["origin"][i] for i in audit["native_validation_indices"]]
                if (
                    namespace["train_text"] != expected_train
                    or namespace["val_text"] != expected_val
                ):
                    raise ValidationError(
                        "LLMLingua-2 actual native split differs from declared seeded split"
                    )
                model = observed["model"]
                actual = {k: _tensor_hash(v, torch) for k, v in model.state_dict().items()}
                if set(actual) != set(observed["initial_tensors"]):
                    raise ValidationError(
                        "LLMLingua-2 parameter namespace changed during native fit"
                    )
                changed = sum(actual[k] != observed["initial_tensors"][k] for k in actual)
                if not changed:
                    raise ValidationError("LLMLingua-2 native fit changed no parameter tensors")
                model.save_pretrained(checkpoint, safe_serialization=True)
                namespace["tokenizer"].save_pretrained(checkpoint)
                # The native source selects best only if validation accuracy>0;
                # final export remains explicitly a candidate, not invented best.
                terminal = dict(
                    status="trained_candidate",
                    optimizer_steps=observed["optimizer_steps"],
                    changed_parameter_tensors=changed,
                    loss_events=observed["loss_events"],
                    native_validation_accuracy=float(namespace["acc"]),
                    native_best_accuracy=float(namespace["best_acc"]),
                    native_best_checkpoint_selected=(native / "state_dict.pth").exists(),
                    checkpoint_selection="explicit_final_candidate_not_substituted_native_best",
                    new_teacher_calls=0,
                    wall_seconds=time.monotonic() - started,
                    cpu_seconds=time.process_time() - cpu,
                    peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated())
                    if c.device == "cuda"
                    else 0,
                )
                _save(
                    root / "checkpoint-manifest.json",
                    dict(
                        schema="agentinputlab.llmlingua2.token_checkpoint.v1",
                        files={p.name: _sha(p) for p in checkpoint.iterdir() if p.is_file()},
                    ),
                )
                _save(root / "fit-terminal.json", terminal)
                metadata = dict(
                    protocol=PROTOCOL,
                    release_id=RELEASE,
                    **{
                        name + "_sha256": _sha(root / name)
                        for name in ("fit-plan.json", "fit-terminal.json")
                    },
                    checkpoint_manifest_sha256=_sha(root / "checkpoint-manifest.json"),
                    native_directory=native.name,
                    native_files={
                        p.relative_to(native).as_posix(): _sha(p)
                        for p in native.rglob("*")
                        if p.is_file()
                    },
                )
                _save(root / "artifact.json", dict(metadata, artifact_id=content_hash(metadata)))
        except Exception as exc:
            _save(
                root / "fit-failed.json",
                dict(
                    error_type=type(exc).__name__,
                    error=str(exc),
                    consumed_optimizer_steps=observed["optimizer_steps"],
                    automatic_retry=False,
                    wall_seconds=time.monotonic() - started,
                    loss_events=observed["loss_events"],
                ),
            )
            raise
        finally:
            torch.load, torch.optim.Adam.step = original_load, original_step
            transformers.AutoModelForTokenClassification.from_pretrained = original_model_descriptor
            transformers.AutoTokenizer.from_pretrained = original_tokenizer_descriptor
            if previous_utils is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = previous_utils
        return LLMLingua2TrainedArtifact.load(root)
