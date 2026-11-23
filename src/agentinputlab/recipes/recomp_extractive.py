"""Actual pinned RECOMP encoder training, safe candidate export and provenance.

Run fit in a dedicated process. The upstream script owns dataset construction,
MultipleNegativesRankingLoss, optimizer, scheduler, batching and training loop.
The adapter supplies declared arguments and seeds, observes loss/steps without
changing them, and exports the resulting live encoder to a safe checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import runpy
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

RECOMP_EXTRACTIVE_TRAINING_RELEASE = "training.recomp.extractive"
RECOMP_TRAINING_SCRIPT = "train_extractive_compressor.py"
RECOMP_TRAINING_SCRIPT_SHA256 = "795239fa4a2bb74c2b9225e671ec252c0755e06b5f664ff45e43e1c707297b94"
RECOMP_TRAINING_PROTOCOL = "agentinputlab.recomp.pinned_extractive_fit.v1"
_LOCK = threading.RLock()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _regular(path, label):
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValidationError(f"RECOMP {label} requires a regular non-symlink file")
    return supplied.resolve()


@dataclass(frozen=True, slots=True)
class RECOMPExtractiveFitControls:
    epochs: int = 10
    batch_size: int = 64
    max_seq_length: int = 100
    warmup_steps: int = 1000
    learning_rate: float = 2e-5
    seed: int = 17
    max_training_steps: int = 10000
    max_data_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in (
            "epochs",
            "batch_size",
            "max_seq_length",
            "warmup_steps",
            "seed",
            "max_training_steps",
            "max_data_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in {"warmup_steps", "seed"} else 1):
                raise ValidationError(f"RECOMP fit {name} must be an integer in range")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValidationError("RECOMP fit learning rate must be finite and positive")


def audit_recomp_training_data(train_file, dev_file, controls):
    """Validate native records and a bounded optimizer plan; never fabricate labels."""
    paths = {
        name: _regular(path, name) for name, path in (("train", train_file), ("dev", dev_file))
    }
    if paths["train"] == paths["dev"]:
        raise ValidationError("RECOMP train and dev files must differ")
    queries, evidence = {}, {}
    for split, path in paths.items():
        if path.stat().st_size > controls.max_data_bytes:
            raise ValidationError("RECOMP data file exceeds the caller's declared byte cap")
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ValidationError("RECOMP training data must be a JSON record array") from exc
        if not isinstance(rows, list) or not rows:
            raise ValidationError("RECOMP training data requires a non-empty record array")
        query_set, triplets = set(), 0
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("dpr_instance"), dict):
                raise ValidationError("RECOMP native training record requires dpr_instance")
            native = row["dpr_instance"]
            query, documents = native.get("query"), row.get("retrieved_docs")
            if not isinstance(query, str) or not query.strip() or "\x00" in query:
                raise ValidationError("RECOMP training query must be non-empty text")
            if not isinstance(documents, list) or not documents:
                raise ValidationError("RECOMP native training record requires documents")
            for document in documents:
                if (
                    not isinstance(document, dict)
                    or not isinstance(document.get("text"), str)
                    or "\x00" in document["text"]
                    or type(document.get("em")) not in (int, float)
                    or not math.isfinite(document["em"])
                ):
                    raise ValidationError(
                        "RECOMP training documents require native text and finite EM labels"
                    )
            for field in ("positive_ctxs", "negative_ctxs"):
                values = native.get(field)
                if not isinstance(values, list) or any(
                    type(v) is not int or not 0 <= v < len(documents) for v in values
                ):
                    raise ValidationError("RECOMP native training indices are invalid")
            # Exact active condition in the pinned source, not its commented
            # num_negatives option or a replacement relevance classifier.
            triplets += sum(
                documents[n]["em"] < documents[p]["em"]
                for p in native["positive_ctxs"]
                for n in native["negative_ctxs"]
            )
            query_set.add(" ".join(query.casefold().split()))
        queries[split] = query_set
        evidence[split] = {
            "path": str(path),
            "sha256": _sha(path),
            "rows": len(rows),
            "native_triplets": triplets,
            "query_set_sha256": content_hash(sorted(query_set)),
        }
    if queries["train"] & queries["dev"]:
        raise ValidationError("RECOMP train/dev query overlap is not permitted")
    count = evidence["train"]["native_triplets"]
    if not count:
        raise ValidationError("RECOMP training records produce zero native EM-ranked triplets")
    steps = math.ceil(count / controls.batch_size) * controls.epochs
    if steps > controls.max_training_steps:
        raise ValidationError(
            "RECOMP native optimizer plan exceeds max_training_steps; input not truncated"
        )
    return {
        "files": evidence,
        "planned_optimizer_steps": steps,
        "native_triplet_predicate": "negative.em < positive.em",
        "query_overlap": False,
        "dev_loaded_by_native_script_but_not_used_for_optimization_or_validation": True,
    }


@contextmanager
def _scope(root, seed):
    previous_cwd, previous_argv, previous_bytecode = (
        Path.cwd(),
        sys.argv[:],
        sys.dont_write_bytecode,
    )
    overrides = dict(
        HOME=str(root / "home"),
        HF_HOME=str(root / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    previous_env = {name: os.environ.get(name) for name in overrides}
    for name in ("home", "hf-cache"):
        (root / name).mkdir()
    os.environ.update(overrides)
    sys.dont_write_bytecode = True
    os.chdir(root)
    try:
        # Dedicated-process contract: native imports/RNG are process-global.
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        os.chdir(previous_cwd)
        sys.argv[:] = previous_argv
        sys.dont_write_bytecode = previous_bytecode
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True, slots=True)
class RECOMPTrainedEncoderArtifact:
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
            raise ValidationError("RECOMP trained artifact root is invalid")
        folder = supplied.resolve()
        metadata_path = _regular(folder / "artifact.json", "artifact")
        terminal_path = _regular(folder / "fit-terminal.json", "fit terminal")
        metadata, terminal = (
            json.loads(metadata_path.read_text()),
            json.loads(terminal_path.read_text()),
        )
        if not isinstance(metadata, dict) or not isinstance(terminal, dict):
            raise ValidationError("RECOMP trained artifact metadata is malformed")
        identity = metadata.pop("artifact_id", None)
        plan_path = _regular(folder / "fit-plan.json", "fit plan")
        if (
            identity != content_hash(metadata)
            or metadata["release_id"] != RECOMP_EXTRACTIVE_TRAINING_RELEASE
            or metadata.get("protocol") != RECOMP_TRAINING_PROTOCOL
            or metadata["fit_terminal_sha256"] != _sha(terminal_path)
            or metadata.get("fit_plan_sha256") != _sha(plan_path)
            or terminal["status"] != "trained_candidate"
            or terminal["optimizer_steps"] < 1
            or terminal["changed_parameter_tensors"] < 1
        ):
            raise ValidationError("RECOMP trained artifact identity or terminal status is invalid")
        checkpoint = verify_checkpoint(folder / "checkpoint", folder / "checkpoint-manifest.json")
        if (
            checkpoint.mode != "extractive"
            or checkpoint.manifest_sha256 != metadata["checkpoint_manifest_sha256"]
        ):
            raise ValidationError("RECOMP trained checkpoint closure differs")
        return cls(
            folder,
            identity,
            checkpoint.checkpoint_revision,
            checkpoint.tokenizer_revision,
            checkpoint.manifest_sha256,
        )

    def inference_parameters(self, *, sentence_splitter_revision):
        # Recheck before handing a current path to an external inference worker.
        current = self.load(self.root)
        return dict(
            checkpoint_revision=current.checkpoint_revision,
            tokenizer_revision=current.tokenizer_revision,
            asset_manifest_hash=current.manifest_sha256,
            sentence_splitter_revision=sentence_splitter_revision,
            asset_license_acknowledged=True,
        )


class RECOMPExtractiveTraining:
    release_id = RECOMP_EXTRACTIVE_TRAINING_RELEASE

    def __init__(self, controls=None):
        self.controls = (
            RECOMPExtractiveFitControls(**controls)
            if isinstance(controls, Mapping)
            else controls or RECOMPExtractiveFitControls()
        )
        if not isinstance(self.controls, RECOMPExtractiveFitControls):
            raise ValidationError("RECOMP fit requires typed controls")

    def fit(
        self,
        *,
        checkout,
        initial_checkpoint,
        initial_checkpoint_manifest,
        train_file,
        dev_file,
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
        script = _regular(checked.root / RECOMP_TRAINING_SCRIPT, "training script")
        if _sha(script) != RECOMP_TRAINING_SCRIPT_SHA256:
            raise ExternalDependencyError("RECOMP pinned native training source differs")
        initial = verify_checkpoint(Path(initial_checkpoint), Path(initial_checkpoint_manifest))
        if initial.mode != "extractive":
            raise ValidationError("RECOMP encoder training requires an extractive initialization")
        c = self.controls
        audit = audit_recomp_training_data(train_file, dev_file, c)
        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "sentence-transformers",
                "numpy",
                "pandas",
                "safetensors",
            )
        }
        if versions["torch"] not in {"1.13.0", "1.13.0+cpu"} or any(
            versions[n] != v
            for n, v in {
                "transformers": "4.24.0",
                "tokenizers": "0.13.2",
                "sentence-transformers": "2.2.2",
                "numpy": "1.26.4",
                "pandas": "1.5.3",
                "safetensors": "0.5.3",
            }.items()
        ):
            raise ExternalDependencyError(
                "RECOMP fit requires the documented isolated legacy dependency versions"
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
            "--model_name",
            str(initial.root),
            "--train_data_path",
            audit["files"]["train"]["path"],
            "--dev_data_path",
            audit["files"]["dev"]["path"],
            "--train_batch_size",
            str(c.batch_size),
            "--max_seq_length",
            str(c.max_seq_length),
            "--epochs",
            str(c.epochs),
            "--warmup_steps",
            str(c.warmup_steps),
            "--lr",
            str(c.learning_rate),
            "--pooling",
            "mean",
        ]
        plan = dict(
            protocol=RECOMP_TRAINING_PROTOCOL,
            release_id=self.release_id,
            controls=asdict(c),
            split_audit=audit,
            source_commit=checked.manifest["upstream"]["commit"],
            source_script_sha256=_sha(script),
            compatibility_manifest_sha256=checked.manifest_sha256,
            initial_manifest_sha256=initial.manifest_sha256,
            initial_tree_sha256=initial.tree_sha256,
            dependency_versions=versions,
            native_argv=command,
            seed_seam="process_random_numpy_torch_seed_before_unchanged_native_script",
            initialization_is_user_selected_not_claimed_original_paper_base=True,
            automatic_retry=False,
        )
        _save(root / "fit-plan.json", plan)
        observed = {"optimizer_steps": 0, "losses": []}
        started, cpu = time.monotonic(), time.process_time()
        try:
            with _LOCK, _scope(root, c.seed):
                import torch
                import transformers
                from safetensors.torch import load_file, save_file
                from sentence_transformers import losses

                original_loss = losses.MultipleNegativesRankingLoss.forward
                originals = {kind: kind.step for kind in {torch.optim.AdamW, transformers.AdamW}}

                def loss(module, *args, **kwargs):
                    value = original_loss(module, *args, **kwargs)
                    observed["losses"].append(float(value.detach().cpu()))
                    return value

                def observed_step(original):
                    def step(optimizer, *args, **kwargs):
                        result = original(optimizer, *args, **kwargs)
                        observed["optimizer_steps"] += 1
                        return result

                    return step

                losses.MultipleNegativesRankingLoss.forward = loss
                for kind, original in originals.items():
                    kind.step = observed_step(original)
                try:
                    sys.argv[:] = command
                    with (
                        (root / "native-training.log").open("x") as log,
                        redirect_stdout(log),
                        redirect_stderr(log),
                    ):
                        native = runpy.run_path(str(script), run_name="__main__")
                finally:
                    losses.MultipleNegativesRankingLoss.forward = original_loss
                    for kind, original in originals.items():
                        kind.step = original
                if observed["optimizer_steps"] != audit["planned_optimizer_steps"]:
                    raise ValidationError(
                        "RECOMP actual native optimizer steps differ from the bounded plan"
                    )
                if not observed["losses"] or not all(math.isfinite(v) for v in observed["losses"]):
                    raise ValidationError("RECOMP native loss is absent or non-finite")
                encoder = native["model"][0].auto_model
                initial_tensors = load_file(str(initial.root / "model.safetensors"), device="cpu")
                trained = {
                    name: value.detach().cpu().contiguous()
                    for name, value in encoder.state_dict().items()
                }
                if set(trained) != set(initial_tensors):
                    raise ValidationError("RECOMP trained encoder changed its parameter namespace")
                changed = sum(
                    not torch.equal(value, initial_tensors[name]) for name, value in trained.items()
                )
                if not changed:
                    raise ValidationError("RECOMP native fit produced no changed encoder tensor")
                destination = root / "checkpoint"
                destination.mkdir()
                encoder.config.save_pretrained(destination)
                native["model"][0].tokenizer.save_pretrained(destination)
                save_file(
                    trained, str(destination / "model.safetensors"), metadata={"format": "pt"}
                )
                checkpoint_files = {p.name: _sha(p) for p in destination.iterdir() if p.is_file()}
                revision = "native-fit-" + content_hash(plan)
                _save(
                    root / "checkpoint-manifest.json",
                    dict(
                        schema_version="agentinputlab.recomp.checkpoint.v1",
                        mode="extractive",
                        checkpoint_revision=revision,
                        tokenizer_revision=initial.tokenizer_revision,
                        license="NOASSERTION",
                        files=checkpoint_files,
                    ),
                )
                exported = verify_checkpoint(destination, root / "checkpoint-manifest.json")
                reloaded = load_file(str(destination / "model.safetensors"), device="cpu")
                if not all(torch.equal(reloaded[name], value) for name, value in trained.items()):
                    raise ValidationError(
                        "RECOMP safe export differs from the live trained encoder"
                    )
            _save(
                root / "fit-terminal.json",
                dict(
                    status="trained_candidate",
                    **observed,
                    changed_parameter_tensors=changed,
                    all_exported_tensor_values_equal_live_trained_model=True,
                    wall_seconds=time.monotonic() - started,
                    cpu_seconds=time.process_time() - cpu,
                    generated_LLM_responses=0,
                    training_compute_is_separate_not_free=True,
                    native_dev_validation_performed=False,
                    full_published_score_parity=False,
                    automatic_retry=False,
                ),
            )
            metadata = dict(
                protocol=RECOMP_TRAINING_PROTOCOL,
                release_id=self.release_id,
                checkpoint_manifest_sha256=exported.manifest_sha256,
                fit_terminal_sha256=_sha(root / "fit-terminal.json"),
                fit_plan_sha256=_sha(root / "fit-plan.json"),
                source_revision=plan["source_commit"],
                pretrained_model_identity_and_full_training_data_are_caller_controlled=True,
                heldout_quality_acceptance=False,
                paper_results_changed=False,
            )
            _save(root / "artifact.json", {**metadata, "artifact_id": content_hash(metadata)})
            return RECOMPTrainedEncoderArtifact.load(root)
        except Exception as exc:
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
    "RECOMPExtractiveFitControls",
    "RECOMPExtractiveTraining",
    "RECOMPTrainedEncoderArtifact",
    "audit_recomp_training_data",
]
