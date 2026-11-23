"""Local-only trained InfoCQR T5 runtime with genuine source collation/decode."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import stat
import threading
from dataclasses import replace
from functools import wraps
from pathlib import Path, PurePosixPath

from agentinputlab.algorithms.reformulate.infocqr.cache_replay import INFOCQR_UPSTREAM_COMMIT
from agentinputlab.algorithms.reformulate.infocqr.student import (
    INFOCQR_STUDENT_MODE,
    INFOCQR_STUDENT_SOURCE_HASHES,
    InfoCQRStudentRequest,
)
from agentinputlab.artifacts import load_pinned_json_mapping, sha256_file
from agentinputlab.contracts import ArtifactError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

STUDENT_ARTIFACT_SCHEMA = "agentinputlab.infocqr.student-artifact.v1"
_TORCH_RNG_LOCK = threading.RLock()


def _locked(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return invoke


class InfoCQRStudentRuntime:
    """Bind caller-mounted, trained student weights; never download substitute weights.

    The native notebook source serializer and T5DataCollator execute unchanged.
    Only inference-only empty labels are supplied to the collator, then removed
    with decoder_input_ids before the native model.generate(max_length=64) call.
    A fixed auxiliary student is reusable with different downstream backbones.
    """

    worker_id = "agentinputlab.infocqr.trained-student.v1"

    def __init__(self, *, checkout: str | Path, checkpoint: str | Path,
                 manifest: str | Path, expected_manifest_sha256: str,
                 source_license_acknowledged: bool = False, device: str = "cpu"):
        if source_license_acknowledged is not True:
            raise ValidationError("InfoCQR external source acknowledgement required")
        if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValidationError("InfoCQR student device must be cpu or explicit cuda:N")
        self.device = device
        self._lock = threading.RLock()
        self.checkout, self.checkpoint = Path(checkout).expanduser(), Path(checkpoint).expanduser()
        self.artifact = load_pinned_json_mapping(manifest, expected_sha256=expected_manifest_sha256)
        self.manifest_sha256 = expected_manifest_sha256
        self._validate_artifact()
        self.artifact = freeze_mapping(self.artifact)
        self._source, self._file_stats = {}, {}
        for relative, expected in INFOCQR_STUDENT_SOURCE_HASHES.items():
            path = self.checkout/relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2*1024*1024:
                raise ArtifactError("InfoCQR student pinned source is unavailable")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ArtifactError(f"InfoCQR student native source hash differs: {relative}")
            self._source[relative] = raw.decode()
        self._verify_checkpoint()
        self._tokenizer = self._model = self._collator = None
        self._pending = {}
        self.last_attempt = None

    def _validate_artifact(self):
        artifact = self.artifact
        if (set(artifact) != {"schema_version", "upstream_revision", "dataset_id", "dataset_revision",
                             "provenance_kind", "training_receipt_sha256", "training_source_sha256", "files"}
                or artifact["schema_version"] != STUDENT_ARTIFACT_SCHEMA
                or artifact["upstream_revision"] != INFOCQR_UPSTREAM_COMMIT
                or artifact["dataset_id"] != "qrecc"
                or not isinstance(artifact["dataset_revision"], str) or not artifact["dataset_revision"].strip()
                or artifact["provenance_kind"] not in {"author_student", "source_recipe_finetuned"}
                or artifact["training_source_sha256"] != INFOCQR_STUDENT_SOURCE_HASHES["T5QR/T5train.py"]):
            raise ArtifactError("InfoCQR student artifact/provenance schema differs")
        value = artifact["training_receipt_sha256"]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ArtifactError("InfoCQR student requires a training provenance receipt, not generic t5-base")
        files = artifact["files"]
        if not isinstance(files, dict) or not files or len(files) > 256:
            raise ArtifactError("InfoCQR student requires a bounded closed checkpoint file manifest")
        for name, digest in files.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or path.as_posix() != name or ".." in path.parts
                    or name.startswith(".") or path.suffix in {".bin", ".pt", ".pth", ".pkl"}
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ArtifactError("InfoCQR student checkpoint paths/hash/weight format differ")
        if "config.json" not in files or not any(n.endswith(".safetensors") for n in files):
            raise ArtifactError("InfoCQR student requires a T5 config and safe trained weight files")

    @staticmethod
    def _signature(path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError("InfoCQR checkpoint contains a link or special file")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _verify_checkpoint(self):
        if self.checkpoint.is_symlink() or not self.checkpoint.is_dir():
            raise ArtifactError("trained InfoCQR student checkpoint not mounted; no t5-base fallback")
        actual = set()
        for root, directories, files in os.walk(self.checkpoint, followlinks=False):
            for name in directories:
                if not stat.S_ISDIR((Path(root)/name).lstat().st_mode):
                    raise ArtifactError("InfoCQR student checkpoint directory contains a symlink")
            for name in files:
                path = Path(root)/name
                relative = path.relative_to(self.checkpoint).as_posix()
                before = self._signature(path)
                if relative not in self.artifact["files"]:
                    raise ArtifactError("unlisted InfoCQR checkpoint file")
                if sha256_file(path) != self.artifact["files"][relative] or self._signature(path) != before:
                    raise ArtifactError("InfoCQR checkpoint file checksum/identity differs")
                actual.add(relative)
                self._file_stats[relative] = before
        if actual != set(self.artifact["files"]):
            raise ArtifactError("InfoCQR checkpoint manifest file missing")
        config = load_pinned_json_mapping(self.checkpoint/"config.json",
            expected_sha256=self.artifact["files"]["config.json"])
        if config.get("model_type") != "t5" or not config.get("is_encoder_decoder"):
            raise ArtifactError("InfoCQR student checkpoint must be a trained encoder-decoder T5")

    def fingerprint(self):
        if self.checkpoint.is_symlink() or not self.checkpoint.is_dir():
            raise ArtifactError("InfoCQR checkpoint root changed or became a symlink")
        present = set()
        for root, directories, files in os.walk(self.checkpoint, followlinks=False):
            for name in directories:
                if not stat.S_ISDIR((Path(root)/name).lstat().st_mode):
                    raise ArtifactError("InfoCQR checkpoint directory changed or became a symlink")
            present.update((Path(root)/name).relative_to(self.checkpoint).as_posix() for name in files)
        if present != set(self._file_stats):
            raise ArtifactError("InfoCQR checkpoint closed file set changed")
        for relative, expected in self._file_stats.items():
            if self._signature(self.checkpoint/relative) != expected:
                raise ArtifactError("InfoCQR student checkpoint changed since verification")
        versions = {}
        for name in ("torch", "transformers", "sentencepiece", "safetensors"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not_installed"
        return {"runtime_id": self.worker_id, "mode": INFOCQR_STUDENT_MODE,
                "upstream_revision": INFOCQR_UPSTREAM_COMMIT,
                "source_sha256": dict(INFOCQR_STUDENT_SOURCE_HASHES),
                "checkpoint_manifest_sha256": self.manifest_sha256,
                "dataset_revision": self.artifact["dataset_revision"],
                "provenance_kind": self.artifact["provenance_kind"],
                "training_receipt_sha256": self.artifact["training_receipt_sha256"],
                "dependencies": versions, "device": self.device,
                "max_source_length": 384, "generation_max_length": 64,
                "native_environment_parity_claimed": False}

    def _serializer(self):
        notebook = json.loads(self._source["T5QR/data_preprocessing.ipynb"])
        found = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.parse("".join(cell.get("source", ()))).body:
                    if isinstance(node, ast.FunctionDef) and node.name == "convert_to_source_target_pairs":
                        found.append(node)
        if len(found) != 1:
            raise ArtifactError("InfoCQR native source serializer differs")
        namespace = {}
        exec(compile(ast.Module(body=found, type_ignores=[]), "pinned-infocqr-source-serializer", "exec"), namespace)
        return namespace["convert_to_source_target_pairs"]

    def _load(self):
        if self._model is not None:
            return
        try:
            transformers = importlib.import_module("transformers")
            bart = importlib.import_module("transformers.models.bart.modeling_bart")
            importlib.import_module("torch")
        except ImportError as exc:
            raise ExternalDependencyError("install agentinputlab[infocqr-student] in the student runtime") from exc
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.checkpoint),
            local_files_only=True, trust_remote_code=False)
        tokenizer.add_tokens(["<Que>", "<Ans>"])
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(str(self.checkpoint),
            local_files_only=True, trust_remote_code=False, use_safetensors=True).to(self.device)
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            raise ArtifactError("trained InfoCQR weights lack the native Que/Ans token embeddings; no resize fallback")
        model.eval()
        tree = ast.parse(self._source["T5QR/utils/dataCollator.py"])
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "T5DataCollator"]
        if len(classes) != 1:
            raise ArtifactError("InfoCQR native collator differs")
        namespace = {"shift_tokens_right": bart.shift_tokens_right}
        exec(compile(ast.Module(body=classes, type_ignores=[]), "pinned-infocqr-collator", "exec"), namespace)
        collator = namespace["T5DataCollator"](tokenizer, model.config.decoder_start_token_id, 384, 64)
        self._tokenizer, self._model, self._collator = tokenizer, model, collator

    @_locked
    def run(self, request: InfoCQRStudentRequest):
        if not isinstance(request, InfoCQRStudentRequest):
            raise ValidationError("InfoCQR student runtime requires its dedicated request")
        identity = self.fingerprint()
        if (request.checkpoint_manifest_sha256 != self.manifest_sha256
                or request.dataset_revision != self.artifact["dataset_revision"]):
            raise ArtifactError("InfoCQR request uses another checkpoint or dataset")
        if request.phase == "discard":
            self._pending.pop(replace(request, phase="generate").request_id, None)
            return {"status": "discarded", "request_id": request.request_id, "identity": identity,
                    "usage": {"model_calls": 0, "retrieval_calls": 0, "input_tokens": 0, "output_tokens": 0}}
        if request.phase == "prepare":
            key = replace(request, phase="generate").request_id
            if key in self._pending:
                raise ValidationError("an identical InfoCQR student preparation is already pending")
            self._load()
            row = {"Conversation_source": request.conversation_source,
                   "Conversation_no": request.conversation_no, "Turn_no": request.turn_no,
                   "NewContext": list(request.new_context), "Question": request.query,
                   "_unused_target": ""}
            serialized = self._serializer()([row], "_unused_target")
            source = serialized["source"][0]
            inputs = self._collator([{"source": source, "target": ""}])
            del inputs["decoder_input_ids"]
            del inputs["labels"]
            inputs = inputs.to(self.device)
            input_tokens = int(inputs["attention_mask"].sum())
            if len(self._pending) >= 64 and key not in self._pending:
                raise ValidationError("InfoCQR student has too many pending zero-generation preparations")
            self._pending[key] = (inputs, input_tokens, content_hash(source))
            return {"status": "ready", "request_id": request.request_id, "identity": identity,
                    "input_tokens": input_tokens, "source_text_hash": content_hash(source),
                    "usage": {"model_calls": 0, "retrieval_calls": 0,
                              "input_tokens": 0, "output_tokens": 0}}
        if self._model is None or request.request_id not in self._pending:
            raise ValidationError("InfoCQR student requires an unconsumed execution preflight")
        inputs, input_tokens, source_hash = self._pending.pop(request.request_id)
        torch = importlib.import_module("torch")
        devices = [] if self.device == "cpu" else [int(self.device[5:])]
        self.last_attempt = {"request_id": request.request_id, "input_tokens": input_tokens,
                             "output_tokens": None, "status": "started"}
        try:
            with _TORCH_RNG_LOCK, torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.random.default_generator.manual_seed(request.seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(request.seed)
                outputs = self._model.generate(**inputs, max_length=64)
            if len(outputs.shape) != 2 or int(outputs.shape[0]) != 1 or not 1 <= int(outputs.shape[1]) <= 64:
                raise ArtifactError("InfoCQR T5 generation shape exceeds the native single-output boundary")
            output_tokens = int(outputs.shape[1]) - 1
            self.last_attempt.update(status="finished", output_tokens=output_tokens)
        except Exception:
            self.last_attempt["status"] = "failed_output_usage_unknown"
            raise
        try:
            text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if not isinstance(text, str):
                raise TypeError("student decoder did not return text")
        except Exception:
            self.last_attempt["status"] = "decode_failed_known_usage"
            return {"request_id": request.request_id, "identity": identity,
                    "status": "decode_failed", "source_text_hash": source_hash,
                    "usage": {"model_calls": 1, "retrieval_calls": 0,
                              "input_tokens": input_tokens, "output_tokens": output_tokens}}
        return {"request_id": request.request_id, "identity": identity,
                "status": "ok" if text.strip() else "empty_output", "rewrite": text,
                "source_text_hash": source_hash,
                "usage": {"model_calls": 1, "retrieval_calls": 0,
                          "input_tokens": input_tokens, "output_tokens": output_tokens}}


__all__ = ["STUDENT_ARTIFACT_SCHEMA", "InfoCQRStudentRuntime"]
