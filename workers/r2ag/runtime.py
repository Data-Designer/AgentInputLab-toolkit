"""Production, local-files-only runtime for the pinned yefd/RRAG source tree."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from .driver import (
    ConditionedReader,
    DriverError,
    GenerationResult,
    PreparedReader,
    TensorStage,
    _verify_artifact_closure,
)


@dataclass(frozen=True, slots=True)
class _PreparedTensors:
    input_ids: Any
    input_embeddings: Any
    attention_mask: Any


@dataclass(frozen=True, slots=True)
class _ConditionedTensors:
    inputs_embeds: Any
    attention_mask: Any


class OfficialR2AGRuntime:
    """Run the upstream RFormer with safely loaded, user-mounted components.

    The runtime intentionally does not use upstream ``from_pretrained`` because
    that function deserializes the combined checkpoint without
    ``weights_only=True``. The RFormer class and its forward path are imported
    directly from the pinned source; component state dicts are loaded strictly.
    """

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint_root: Path,
        base_model_root: Path,
        tokenizer_root: Path,
        expected_runtime_versions: Mapping[str, str],
    ) -> None:
        self.source_root = source_root
        self.checkpoint_root = checkpoint_root
        self.base_model_root = base_model_root
        self.tokenizer_root = tokenizer_root
        self.runtime_versions = _verify_runtime_versions(expected_runtime_versions)
        _preflight_reader_configuration(
            self.base_model_root,
            transformers_version=self.runtime_versions["transformers"],
        )
        self._config: Mapping[str, Any] | None = None
        self._rformer_state: Mapping[str, Any] | None = None
        self._projection_state: Mapping[str, Any] | None = None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._base_model: Any = None
        self._rformer: Any = None
        self._projection: Any = None
        self._device: Any = None

    def verify_artifacts(self, artifacts: Mapping[str, Any]) -> Mapping[str, Any]:
        torch = _import_torch()
        config_path = _regular_file(self.checkpoint_root / "config.json")
        checkpoint_path = _regular_file(self.checkpoint_root / "RRAGLlama_pytorch_model.bin")
        config = _json_object(config_path)
        num_k = _positive_int(config, "num_k")
        if num_k != artifacts.get("checkpoint_num_documents"):
            raise ValueError("checkpoint config.num_k does not match artifact identity")
        if _positive_int(config, "input_dim") != 3:
            raise ValueError("R²AG checkpoint input_dim must be three")
        if _sha256_file(config_path) != artifacts.get("checkpoint_config_sha256"):
            raise ValueError("checkpoint config hash mismatch")
        try:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError("R²AG requires torch.load(weights_only=True)") from exc
        if not isinstance(state, Mapping) or set(state) != {"r_former", "llama_proj"}:
            raise ValueError("R²AG checkpoint must contain only r_former and llama_proj")
        rformer_state = state["r_former"]
        projection_state = state["llama_proj"]
        if not isinstance(rformer_state, Mapping) or not isinstance(projection_state, Mapping):
            raise ValueError("R²AG component state dicts are invalid")
        rformer_hash = _state_dict_sha256(rformer_state, torch)
        projection_hash = _state_dict_sha256(projection_state, torch)
        if rformer_hash != artifacts.get("rformer_weights_sha256"):
            raise ValueError("RFormer state hash mismatch")
        if projection_hash != artifacts.get("projection_weights_sha256"):
            raise ValueError("projection state hash mismatch")
        _verify_named_manifest(
            self.base_model_root,
            repository=artifacts.get("base_model_repository"),
            revision=artifacts.get("base_model_revision"),
            manifest_sha256=artifacts.get("base_model_manifest_sha256"),
            label="base model",
        )
        _verify_named_manifest(
            self.tokenizer_root,
            repository=artifacts.get("tokenizer_repository"),
            revision=artifacts.get("tokenizer_revision"),
            manifest_sha256=artifacts.get("tokenizer_manifest_sha256"),
            label="tokenizer",
        )
        self._torch = torch
        self._config = config
        self._rformer_state = rformer_state
        self._projection_state = projection_state
        return {
            "checkpoint_num_documents": num_k,
            "checkpoint_config_sha256": _sha256_file(config_path),
            "rformer_weights_sha256": rformer_hash,
            "projection_weights_sha256": projection_hash,
        }

    def compute_features(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        retriever_identity: Mapping[str, Any],
    ) -> tuple[Mapping[str, float], ...]:
        """Execute the pinned retrieval formulas with a local sentence encoder."""

        embedder_root = _regular_directory(self.checkpoint_root / "feature-embedder")
        expected = {
            "repository": retriever_identity.get("repository"),
            "revision": retriever_identity.get("revision"),
        }
        manifest_hash = retriever_identity.get("manifest_sha256")
        _verify_named_manifest(
            embedder_root,
            repository=expected["repository"],
            revision=expected["revision"],
            manifest_sha256=manifest_hash,
            label="feature embedder",
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for worker features") from exc
        encoder = SentenceTransformer(
            str(embedder_root),
            trust_remote_code=False,
            local_files_only=True,
        )
        texts = [query]
        texts.extend(
            f"Title: {candidate['title']}\n{candidate['text']}" for candidate in candidates
        )
        encoded = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = tuple(tuple(float(value) for value in row) for row in encoded)
        return _retrieval_features(vectors[0], vectors[1:])

    def prepare_reader(self, reader_input: str, num_k: int) -> PreparedReader:
        self._ensure_loaded()
        torch = self._torch
        encoded = self._tokenizer(
            reader_input,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        unknown_id = _nonnegative_int(self._config, "unk_token_id")
        positions_tensor = torch.nonzero(input_ids == unknown_id, as_tuple=False)
        if positions_tensor.shape[0] != num_k:
            raise ValueError("tokenizer did not produce exactly one marker per candidate")
        marker_positions = tuple(int(row[1].item()) for row in positions_tensor)
        with torch.inference_mode():
            input_embeddings = self._base_model.get_input_embeddings()(input_ids)
        return PreparedReader(
            _PreparedTensors(input_ids, input_embeddings, attention_mask),
            marker_positions,
            int(input_ids.numel()),
        )

    def run_rformer(
        self,
        features: Sequence[Mapping[str, float]],
    ) -> TensorStage:
        self._ensure_loaded()
        torch = self._torch
        values = [[row["rerank"], row["neighbor"], row["precedent"]] for row in features]
        tensor = torch.tensor(values, dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            output, loss = self._rformer(tensor)
        if loss is not None:
            raise ValueError("inference-only RFormer unexpectedly produced training loss")
        return TensorStage(output, _tensor_sha256(output, torch))

    def project(self, rformer_output: Any) -> TensorStage:
        self._ensure_loaded()
        torch = self._torch
        with torch.inference_mode():
            output = self._projection(rformer_output)
        return TensorStage(output, _tensor_sha256(output, torch))

    def replace_markers(
        self,
        prepared: PreparedReader,
        projected: Any,
    ) -> ConditionedReader:
        self._ensure_loaded()
        torch = self._torch
        value = prepared.value
        if not isinstance(value, _PreparedTensors):
            raise TypeError("prepared reader tensor bundle is invalid")
        flattened = projected.reshape(-1, projected.shape[-1])
        if flattened.shape[0] != len(prepared.marker_positions):
            raise ValueError("projected retrieval vectors do not match marker count")
        updated = value.input_embeddings.clone()
        before = value.input_embeddings.clone()
        for index, position in enumerate(prepared.marker_positions):
            updated[0, position] = flattened[index].to(updated.dtype)
        marker_set = set(prepared.marker_positions)
        untouched = tuple(index for index in range(updated.shape[1]) if index not in marker_set)
        only_markers = True
        if untouched:
            selector = torch.tensor(untouched, dtype=torch.long, device=updated.device)
            only_markers = bool(torch.equal(updated[:, selector], before[:, selector]))
        return ConditionedReader(
            _ConditionedTensors(updated.contiguous(), value.attention_mask),
            prepared.marker_positions,
            only_markers,
        )

    def generate(
        self,
        conditioned: Any,
        *,
        max_new_tokens: int,
        seed: int,
    ) -> GenerationResult:
        self._ensure_loaded()
        torch = self._torch
        if not isinstance(conditioned, _ConditionedTensors):
            raise TypeError("conditioned reader tensor bundle is invalid")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            outputs = self._base_model.generate(
                inputs_embeds=conditioned.inputs_embeds,
                attention_mask=conditioned.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                length_penalty=1.0,
            )
        return _decode_complete_generation(
            outputs,
            tokenizer=self._tokenizer,
            max_new_tokens=max_new_tokens,
        )

    def _ensure_loaded(self) -> None:
        if self._base_model is not None:
            return
        if (
            self._torch is None
            or self._config is None
            or self._rformer_state is None
            or self._projection_state is None
        ):
            raise RuntimeError("verify_artifacts must run before model construction")
        torch = self._torch
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for R²AG execution") from exc
        module = _load_source_module(
            self.source_root / "RRAG/models/modeling_rrag.py",
            "agentinputlab_pinned_rrag_modeling",
        )
        rformer_type = getattr(module, "RFormer", None)
        if not isinstance(rformer_type, type):
            raise RuntimeError("pinned source does not expose RFormer")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_root),
            local_files_only=True,
            trust_remote_code=False,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_root),
            local_files_only=True,
            trust_remote_code=False,
        ).to(device)
        base_model.eval()
        d_model = _positive_int(self._config, "d_model")
        num_k = _positive_int(self._config, "num_k")
        rformer = rformer_type(
            input_dim=3,
            num_k=num_k,
            d_model=d_model,
            n_head=4,
            num_layers=1,
            dropout=0.1,
        ).to(device)
        hidden_size = _positive_int(self._config, "hidden_size")
        actual_hidden = int(getattr(base_model.config, "hidden_size", hidden_size))
        if actual_hidden != hidden_size:
            raise ValueError("base model hidden size does not match projection config")
        projection = torch.nn.Linear(d_model, hidden_size, device=device)
        rformer.load_state_dict(self._rformer_state, strict=True)
        projection.load_state_dict(self._projection_state, strict=True)
        rformer.eval()
        projection.eval()
        for module_value in (rformer, projection, base_model):
            for parameter in module_value.parameters():
                parameter.requires_grad_(False)
        self._tokenizer = tokenizer
        self._base_model = base_model
        self._rformer = rformer
        self._projection = projection
        self._device = device


def _retrieval_features(
    query: Sequence[float],
    documents: Sequence[Sequence[float]],
) -> tuple[Mapping[str, float], ...]:
    if len(documents) < 2:
        raise ValueError("R²AG retrieval features require at least two documents")
    rerank = tuple(_cosine(query, document) for document in documents)
    neighbor = []
    for index, document in enumerate(documents):
        if index == 0:
            value = _cosine(document, documents[1])
        elif index == len(documents) - 1:
            value = _cosine(documents[-2], document)
        else:
            value = (
                _cosine(document, documents[index - 1]) + _cosine(document, documents[index + 1])
            ) / 2
        neighbor.append(value)
    precedent = [1.0]
    for index in range(1, len(documents)):
        scores = rerank[:index]
        maximum = max(scores)
        weights_raw = tuple(math.exp(score - maximum) for score in scores)
        denominator = sum(weights_raw)
        weights = tuple(value / denominator for value in weights_raw)
        weighted = tuple(
            sum(weights[row] * documents[row][column] for row in range(index))
            for column in range(len(documents[0]))
        )
        precedent.append(_cosine(documents[index], weighted))
    return tuple(
        {
            "rerank": rerank[index],
            "neighbor": neighbor[index],
            "precedent": precedent[index],
        }
        for index in range(len(documents))
    )


def _preflight_reader_configuration(root: Path, *, transformers_version: str) -> None:
    """Reject incompatible reader configs without opening any model weights.

    This negative capability check does not certify a snapshot or checkpoint.
    Successful execution still requires all existing complete closure checks.
    Never unwrap a conditional-generation config or substitute another reader.
    """

    root = _regular_directory(root)
    raw = _json_object(_regular_file(root / "config.json"))
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required for R²AG reader preflight") from exc
    declared = raw.get("model_type", "<unspecified>")
    try:
        config = AutoConfig.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
    except ValueError as exc:
        raise DriverError(
            "READER_CONFIG_INCOMPATIBLE",
            f"Pinned R²AG transformers=={transformers_version} cannot load reader "
            f"configuration model_type={declared!r}; no model weights were loaded. "
            "Use a reader supported by this pinned profile; a modern-reader "
            "adaptation must be configured and validated separately.",
        ) from exc
    if type(config) not in AutoModelForCausalLM._model_mapping:
        raise DriverError(
            "READER_CONFIG_INCOMPATIBLE",
            f"Pinned R²AG transformers=={transformers_version} has no causal-LM "
            f"reader for configuration model_type={declared!r}; no model weights "
            "were loaded. Conditional-generation wrappers are not silently unwrapped.",
        )


def _verify_runtime_versions(expected: Mapping[str, str]) -> Mapping[str, str]:
    required = {
        "torch": "2.1.2",
        "transformers": "4.40.2",
        "sentence-transformers": "2.6.1",
    }
    if dict(expected) != required:
        raise RuntimeError("R²AG expected runtime versions do not match the pinned profile")
    installed: dict[str, str] = {}
    for distribution_name, expected_version in required.items():
        try:
            actual_version = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"R²AG runtime distribution {distribution_name!r} is unavailable"
            ) from exc
        if actual_version != expected_version:
            raise RuntimeError(
                f"R²AG runtime distribution {distribution_name!r} "
                f"must be exactly {expected_version}, found {actual_version}"
            )
        installed[distribution_name] = actual_version
    return installed


def _decode_complete_generation(
    outputs: Any,
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> GenerationResult:
    """Validate and decode the complete upstream ``inputs_embeds`` output.

    The pinned upstream runner passes only ``inputs_embeds`` to ``generate`` and
    decodes the returned sequence directly.  There are no prompt token IDs that
    can be safely sliced from this tensor, so this compatibility path preserves
    that behavior and validates the complete generated sequence instead.
    """

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("reader max_new_tokens must be an integer")
    if max_new_tokens < 1:
        raise ValueError("reader max_new_tokens must be positive")
    shape = getattr(outputs, "shape", None)
    rank = getattr(outputs, "ndim", None)
    if rank != 2 or shape is None or len(shape) != 2:
        raise ValueError("reader output must be a rank-two token tensor")
    batch_size = int(shape[0])
    output_tokens = int(shape[1])
    if batch_size != 1:
        raise ValueError("reader output batch size must be exactly one")
    if output_tokens < 1 or output_tokens > max_new_tokens:
        raise ValueError("reader output length must be within max_new_tokens")
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if (
        isinstance(decoded, (str, bytes))
        or not isinstance(decoded, Sequence)
        or len(decoded) != 1
        or not isinstance(decoded[0], str)
        or not decoded[0].strip()
    ):
        raise ValueError("reader returned no non-empty decoded sequence")
    return GenerationResult(decoded[0].strip(), output_tokens)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0 or len(left) != len(right):
        raise ValueError("invalid embedding vector")
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    if not math.isfinite(value):
        raise ValueError("non-finite cosine score")
    return max(-1.0, min(1.0, value))


def _state_dict_sha256(value: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name]
        if not torch.is_tensor(tensor):
            raise ValueError("state dict contains a non-tensor value")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: Any, torch: Any) -> str:
    return _state_dict_sha256({"tensor": value}, torch)


def _verify_named_manifest(
    root: Path,
    *,
    repository: Any,
    revision: Any,
    manifest_sha256: Any,
    label: str,
) -> None:
    _verify_artifact_closure(
        root,
        repository=repository,
        revision=revision,
        manifest_sha256=manifest_sha256,
        label=label,
    )


def _load_source_module(path: Path, name: str) -> ModuleType:
    path = _regular_file(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import pinned R²AG source")
    module = importlib.util.module_from_spec(spec)
    # Execute the checked source itself. SourceFileLoader can both trust a
    # timestamp-valid stale .pyc and create an untracked __pycache__, making
    # the next strict clean-checkout verification reject our own import.
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for R²AG checkpoint verification") from exc
    return torch


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing: {path.name}")
    return path


def _regular_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"required regular directory is missing: {path.name}")
    return path


def _json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise ValueError(f"{key} must be a positive integer")
    return result


def _nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return result


__all__ = ["OfficialR2AGRuntime"]
