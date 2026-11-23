"""R²AG hidden-state conditioning adapted to the shared agent arenas.

The upstream-compatible worker remains the strict paper-shaped composite path.
This module defines a separate common-support intervention.  It preserves the
three retrieval features, R²-Former, learned projection, and marker-only input
embedding replacement.  The custom reader's answer is exposed to the
downstream agent only as a non-authoritative generated anchor beside the real
retrieved documents.  That final placement is an explicit agent-arena
adaptation and is never reported as upstream checkpoint or paper-result parity.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CapabilityError,
    ContextBlock,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .bridge import R2AG_UPSTREAM_COMMIT, compute_r2ag_retrieval_features

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_NAME = "artifact-manifest.json"
_MANIFEST_SCHEMA = "agentinputlab.r2ag-common-support-artifact.v1"
_MARKER = "<R>"
_QWEN38_FP8_BACKBONE = "Qwen3.8-27B-FP8"


def _qwen38_fp8_min_new_tokens(*, backbone: str, max_new_tokens: int) -> int | None:
    """Prevent an EOS-only anchor for the exact Qwen3.8 FP8 reader.

    ``generate(inputs_embeds=...)`` can otherwise terminate immediately with
    EOS for a subset of prompts.  R2AG requires a non-empty, explicitly
    non-authoritative anchor, so require one generated token for this exact
    backbone while leaving every other reader unchanged.
    """

    if backbone != _QWEN38_FP8_BACKBONE:
        return None
    return min(1, max_new_tokens)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r2ag_reader_backbone_identity(model_root: Path) -> str:
    """Return the portable local-snapshot identity recorded in the artifact."""

    identity = model_root.expanduser().resolve().name
    if not identity:
        raise ValidationError("R²AG base-model snapshot identity is empty")
    return identity


def _repair_qwen38_fp8_model_config(model_config: Any, *, backbone: str) -> int:
    """Remove stale dense-layer exclusions from the official Qwen3.8 FP8 config.

    The checkpoint contains FP8 weights and inverse scales for every language
    MLP gate projection, but its generated ``modules_to_not_convert`` list also
    names those gates.  Transformers would therefore build a dense Linear for
    an FP8 tensor and fail at the first forward pass.  Keep this repair scoped
    to the exact local backbone used by the campaign.
    """

    if backbone != _QWEN38_FP8_BACKBONE:
        return 0
    quantization_config = getattr(model_config, "quantization_config", None)
    if not isinstance(quantization_config, Mapping):
        return 0
    normalized = dict(quantization_config)
    if normalized.get("quant_method") != "fp8":
        return 0
    exclusions = list(normalized.get("modules_to_not_convert", ()))
    retained = [name for name in exclusions if not str(name).endswith(".mlp.gate")]
    repaired = len(exclusions) - len(retained)
    if repaired:
        normalized["modules_to_not_convert"] = retained
        model_config.quantization_config = normalized
    return repaired


@dataclass(frozen=True, slots=True)
class R2AGReaderResult:
    anchor: str
    input_tokens: int
    output_tokens: int
    marker_positions: tuple[int, ...]
    rformer_output_sha256: str
    projection_output_sha256: str

    def __post_init__(self) -> None:
        if not self.anchor.strip():
            raise ValidationError("R²AG common-support reader returned an empty anchor")
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"R²AG reader {name} must be positive")
        positions = tuple(self.marker_positions)
        if not positions or positions != tuple(sorted(set(positions))):
            raise ValidationError("R²AG marker positions must be ordered and unique")
        if any(
            _SHA256.fullmatch(value) is None
            for value in (
                self.rformer_output_sha256,
                self.projection_output_sha256,
            )
        ):
            raise ValidationError("R²AG tensor identities must be lowercase SHA-256")
        object.__setattr__(self, "marker_positions", positions)


@runtime_checkable
class R2AGCommonSupportReader(Protocol):
    def run(
        self,
        *,
        query: str,
        documents: Sequence[RetrievalResult],
        features: Sequence[Mapping[str, float]],
        max_new_tokens: int,
        seed: int,
    ) -> R2AGReaderResult:
        """Run R²-Former, projection, marker replacement, and generation."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return the closed source/model/training identity."""


@runtime_checkable
class R2AGFeatureEncoder(Protocol):
    def encode(
        self,
        *,
        query: str,
        documents: Sequence[RetrievalResult],
    ) -> tuple[Sequence[float], tuple[Sequence[float], ...]]:
        """Encode one query and ordered documents with role-correct prefixes."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return the checksum-closed encoder identity."""


@dataclass(frozen=True, slots=True)
class R2AGRuntimeRequest:
    query: str
    documents: tuple[RetrievalResult, ...]
    max_new_tokens: int
    seed: int

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.documents:
            raise ValidationError("R²AG runtime request requires query and documents")
        if self.max_new_tokens < 1 or self.seed < 0:
            raise ValidationError("R²AG runtime request controls are invalid")
        object.__setattr__(self, "documents", tuple(self.documents))

    def external_reservation(self) -> Mapping[str, int]:
        return {"model_calls": 1, "embedding_calls": 1}


@runtime_checkable
class R2AGCommonSupportRuntime(Protocol):
    def run(self, request: R2AGRuntimeRequest) -> Mapping[str, Any]:
        """Encode features and execute the local hidden-state-conditioned reader."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return both reader and feature-encoder identities."""


@dataclass(slots=True)
class LocalSentenceTransformerR2AGFeatureEncoder:  # pragma: no cover - L20 GPU acceptance
    """Role-correct local E5 encoder used only for R²AG retrieval features."""

    model_root: Path
    model_load_set_sha256: str
    device: str = "cuda"
    batch_size: int = 16
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    _model: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        root = self.model_root.expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise ValidationError("R²AG feature model_root must be a real directory")
        if _SHA256.fullmatch(self.model_load_set_sha256) is None:
            raise ValidationError("R²AG feature model load-set identity is invalid")
        if r2ag_model_tree_sha256(root) != self.model_load_set_sha256:
            raise ValidationError("R²AG feature model load-set checksum mismatch")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValidationError("R²AG feature batch_size must be an integer")
        if self.batch_size < 1:
            raise ValidationError("R²AG feature batch_size must be positive")
        if not self.query_prefix or not self.document_prefix:
            raise ValidationError("R²AG feature prefixes must be non-empty")
        object.__setattr__(self, "model_root", root)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ExternalDependencyError(
                "R²AG feature encoding requires sentence-transformers"
            ) from exc
        try:
            model = SentenceTransformer(
                str(self.model_root),
                device=self.device,
                local_files_only=True,
            )
        except TypeError:
            model = SentenceTransformer(str(self.model_root), device=self.device)
        self._model = model
        return model

    def encode(
        self,
        *,
        query: str,
        documents: Sequence[RetrievalResult],
    ) -> tuple[Sequence[float], tuple[Sequence[float], ...]]:
        if not query.strip() or not documents:
            raise ValidationError("R²AG feature encoding requires query and documents")
        texts = [
            f"{self.query_prefix}{query.strip()}",
            *(
                f"{self.document_prefix}Title: {item.document.title}\n{item.document.text}"
                for item in documents
            ),
        ]
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        rows = tuple(tuple(float(value) for value in row) for row in vectors)
        if len(rows) != len(texts):
            raise AlgorithmOutputError("R²AG feature encoder returned an invalid vector count")
        if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise AlgorithmOutputError("R²AG feature encoder returned ragged vectors")
        return rows[0], rows[1:]

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "runtime": "sentence_transformers_role_correct.v1",
                "model_root": str(self.model_root),
                "model_load_set_sha256": self.model_load_set_sha256,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix,
                "normalize_embeddings": True,
            }
        )


@dataclass(slots=True)
class LocalR2AGCommonSupportRuntime:
    """Monitored context runtime bundling the reader and role-correct encoder."""

    reader: R2AGCommonSupportReader
    feature_encoder: R2AGFeatureEncoder
    worker_id: str = "r2ag-common-support-local"
    allow_short_candidates: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_short_candidates, bool):
            raise ValidationError("R²AG allow_short_candidates must be boolean")
        if not isinstance(self.reader, R2AGCommonSupportReader):
            raise CapabilityError("R²AG runtime reader does not implement its protocol")
        if not isinstance(self.feature_encoder, R2AGFeatureEncoder):
            raise CapabilityError("R²AG runtime feature encoder does not implement its protocol")

    def run(self, request: R2AGRuntimeRequest) -> Mapping[str, Any]:
        if not isinstance(request, R2AGRuntimeRequest):
            raise TypeError("R²AG runtime requires R2AGRuntimeRequest")
        query_vector, document_vectors = self.feature_encoder.encode(
            query=request.query,
            documents=request.documents,
        )
        if self.allow_short_candidates:
            from .candidate_count import short_pool_features

            features = short_pool_features(query_vector, document_vectors)
        else:
            features = compute_r2ag_retrieval_features(query_vector, document_vectors)
        reader = self.reader.run(
            query=request.query,
            documents=request.documents,
            features=features,
            max_new_tokens=request.max_new_tokens,
            seed=request.seed,
        )
        return {
            "status": "ok",
            "features": [dict(row) for row in features],
            "reader": {
                "anchor": reader.anchor,
                "input_tokens": reader.input_tokens,
                "output_tokens": reader.output_tokens,
                "marker_positions": list(reader.marker_positions),
                "rformer_output_sha256": reader.rformer_output_sha256,
                "projection_output_sha256": reader.projection_output_sha256,
            },
            "usage": {
                "model_calls": 1,
                "retrieval_calls": 0,
                "tool_calls": 0,
                "embedding_calls": 1,
                "input_tokens": reader.input_tokens,
                "output_tokens": reader.output_tokens,
            },
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "runtime": "r2ag_common_support_context_runtime.v1",
                "reader": dict(self.reader.fingerprint()),
                "feature_encoder": dict(self.feature_encoder.fingerprint()),
                **({"feature_boundary_adapter": "absent_neighbor_zero.v1"}
                   if self.allow_short_candidates else {}),
            }
        )


@dataclass(slots=True)
class LocalR2AGCommonSupportReader:  # pragma: no cover - L20 GPU acceptance
    """Lazy local Qwen reader backed by a checksum-closed retrained artifact."""

    artifact_root: Path
    manifest_sha256: str
    source_root: Path
    base_model_root: Path
    base_model_load_set_sha256: str
    device: str = "cuda"
    allow_short_candidates: bool = False
    _manifest: Mapping[str, Any] = field(init=False, repr=False)
    _runtime: tuple[Any, Any, Any, Any, Any] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.allow_short_candidates, bool):
            raise ValidationError("R²AG allow_short_candidates must be boolean")
        artifact = self.artifact_root.expanduser().resolve()
        source = self.source_root.expanduser().resolve()
        base_model = self.base_model_root.expanduser().resolve()
        for label, path in (
            ("artifact_root", artifact),
            ("source_root", source),
            ("base_model_root", base_model),
        ):
            if not path.is_dir() or path.is_symlink():
                raise ValidationError(f"R²AG {label} must be a real directory")
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise ValidationError("R²AG manifest_sha256 must be lowercase SHA-256")
        if _SHA256.fullmatch(self.base_model_load_set_sha256) is None:
            raise ValidationError("R²AG base model load-set identity is invalid")
        manifest_path = artifact / _MANIFEST_NAME
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or _sha256_file(manifest_path) != self.manifest_sha256
        ):
            raise ValidationError("R²AG artifact manifest checksum mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("R²AG artifact manifest is invalid") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != _MANIFEST_SCHEMA:
            raise ValidationError("R²AG artifact manifest schema mismatch")
        expected = {
            "upstream_commit": R2AG_UPSTREAM_COMMIT,
            "marker_token": _MARKER,
            "feature_names": ["rerank", "neighbor", "precedent"],
            "reader_backbone": r2ag_reader_backbone_identity(base_model),
            "base_model_load_set_sha256": self.base_model_load_set_sha256,
            "upstream_checkpoint_parity_claimed": False,
            "paper_result_parity_claimed": False,
            "generated_anchor_is_evidence": False,
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValidationError(f"R²AG artifact manifest drifted at {name}")
        for name in (
            "num_documents",
            "marker_token_id",
            "rformer_hidden_size",
            "rformer_heads",
            "rformer_layers",
            "reader_hidden_size",
            "max_sequence_length",
            "max_document_chars",
        ):
            value = manifest.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"R²AG artifact manifest has invalid {name}")
        model_source = source / "RRAG/models/modeling_rrag.py"
        if (
            not model_source.is_file()
            or model_source.is_symlink()
            or _sha256_file(model_source) != manifest.get("modeling_source_sha256")
        ):
            raise ValidationError("R²AG pinned modeling source checksum mismatch")
        files = manifest.get("artifact_files")
        if not isinstance(files, Mapping) or not files:
            raise ValidationError("R²AG artifact closure is empty")
        observed: set[str] = set()
        for raw_name, raw_digest in files.items():
            if (
                not isinstance(raw_name, str)
                or not raw_name
                or Path(raw_name).is_absolute()
                or ".." in Path(raw_name).parts
                or not isinstance(raw_digest, str)
                or _SHA256.fullmatch(raw_digest) is None
            ):
                raise ValidationError("R²AG artifact closure entry is invalid")
            target = artifact / raw_name
            if not target.is_file() or target.is_symlink() or _sha256_file(target) != raw_digest:
                raise ValidationError(f"R²AG artifact closure mismatch: {raw_name}")
            observed.add(raw_name)
        actual = {
            str(path.relative_to(artifact))
            for path in artifact.rglob("*")
            if path.is_file() and path.name != _MANIFEST_NAME
        }
        if observed != actual or "checkpoint.pt" not in observed:
            raise ValidationError("R²AG artifact closure has missing or extra files")
        tokenizer_files = {name for name in observed if name.startswith("tokenizer/")}
        if not tokenizer_files:
            raise ValidationError("R²AG artifact has no closed tokenizer snapshot")
        object.__setattr__(self, "artifact_root", artifact)
        object.__setattr__(self, "source_root", source)
        object.__setattr__(self, "base_model_root", base_model)
        object.__setattr__(self, "_manifest", freeze_mapping(dict(manifest)))

    def _load(self) -> tuple[Any, Any, Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ExternalDependencyError(
                "R²AG common-support runtime requires torch and transformers"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ExternalDependencyError("R²AG requested unavailable CUDA")
        module = _load_source_module(
            self.source_root / "RRAG/models/modeling_rrag.py",
            "agentinputlab_common_support_rrag_modeling",
        )
        rformer_type = getattr(module, "RFormer", None)
        if not isinstance(rformer_type, type):
            raise ExternalDependencyError("R²AG pinned source does not expose RFormer")
        tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_root / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )
        marker_id = tokenizer.convert_tokens_to_ids(_MARKER)
        if marker_id != self._manifest.get("marker_token_id"):
            raise ValidationError("R²AG marker token ID differs from the training artifact")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model_config = AutoConfig.from_pretrained(
            self.base_model_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        _repair_qwen38_fp8_model_config(
            model_config,
            backbone=str(self._manifest["reader_backbone"]),
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_root,
            config=model_config,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        ).to(self.device)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        num_k = int(self._manifest["num_documents"])
        d_model = int(self._manifest["rformer_hidden_size"])
        rformer = rformer_type(
            input_dim=3,
            num_k=num_k,
            d_model=d_model,
            n_head=int(self._manifest["rformer_heads"]),
            num_layers=int(self._manifest["rformer_layers"]),
            dropout=float(self._manifest["rformer_dropout"]),
        ).to(self.device)
        projection = torch.nn.Linear(
            d_model,
            int(self._manifest["reader_hidden_size"]),
            device=self.device,
        )
        try:
            checkpoint = torch.load(
                self.artifact_root / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise ExternalDependencyError("R²AG requires safe weights-only loading") from exc
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"r_former", "llama_proj"}:
            raise ValidationError("R²AG checkpoint component schema mismatch")
        rformer.load_state_dict(checkpoint["r_former"], strict=True)
        projection.load_state_dict(checkpoint["llama_proj"], strict=True)
        if self.allow_short_candidates:
            from .candidate_count import install_short_pool_compat

            install_short_pool_compat(rformer, self.source_root / "RRAG/models/modeling_rrag.py")
        rformer.eval()
        projection.eval()
        for component in (rformer, projection):
            for parameter in component.parameters():
                parameter.requires_grad_(False)
        self._runtime = (torch, tokenizer, model, rformer, projection)
        return self._runtime

    def run(
        self,
        *,
        query: str,
        documents: Sequence[RetrievalResult],
        features: Sequence[Mapping[str, float]],
        max_new_tokens: int,
        seed: int,
    ) -> R2AGReaderResult:
        if not query.strip():
            raise ValidationError("R²AG reader requires a non-empty query")
        trained_count = int(self._manifest["num_documents"])
        count_matches = (1 <= len(documents) <= trained_count if self.allow_short_candidates
                         else len(documents) == trained_count)
        if not count_matches:
            raise ValidationError("R²AG document count differs from its trained checkpoint")
        if len(features) != len(documents):
            raise ValidationError("R²AG feature count differs from its documents")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValidationError("R²AG max_new_tokens must be positive")
        torch, tokenizer, model, rformer, projection = self._load()
        prompt = r2ag_reader_prompt(
            query,
            documents,
            max_document_chars=int(self._manifest["max_document_chars"]),
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        ).to(self.device)
        input_tokens = int(encoded.attention_mask.sum().detach().cpu())
        if input_tokens > int(self._manifest["max_sequence_length"]):
            raise BudgetExceededError("R²AG reader input exceeds its trained sequence length")
        marker_id = int(self._manifest["marker_token_id"])
        marker_rows = torch.nonzero(encoded.input_ids == marker_id, as_tuple=False)
        marker_positions = tuple(int(row[1].item()) for row in marker_rows)
        if len(marker_positions) != len(documents):
            raise AlgorithmOutputError("R²AG tokenizer did not preserve one marker per document")
        values = [
            [float(row[name]) for name in ("rerank", "neighbor", "precedent")] for row in features
        ]
        feature_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            rformer_output, loss = rformer(feature_tensor)
            if loss is not None:
                raise AlgorithmOutputError("R²AG inference RFormer returned training loss")
            projected = projection(rformer_output)
            input_embeddings = model.get_input_embeddings()(encoded.input_ids)
            updated = input_embeddings.clone()
            flattened = projected.reshape(-1, projected.shape[-1])
            for index, position in enumerate(marker_positions):
                updated[0, position] = flattened[index].to(updated.dtype)
            before = input_embeddings.clone()
            untouched = [
                index for index in range(updated.shape[1]) if index not in marker_positions
            ]
            if untouched:
                selector = torch.tensor(untouched, dtype=torch.long, device=self.device)
                if not torch.equal(updated[:, selector], before[:, selector]):
                    raise AlgorithmOutputError("R²AG modified a non-marker embedding")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            generation_compatibility: dict[str, int] = {}
            min_new_tokens = _qwen38_fp8_min_new_tokens(
                backbone=str(self._manifest["reader_backbone"]),
                max_new_tokens=max_new_tokens,
            )
            if min_new_tokens is not None:
                generation_compatibility["min_new_tokens"] = min_new_tokens
            output = model.generate(
                inputs_embeds=updated.contiguous(),
                attention_mask=encoded.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                length_penalty=1.0,
                **generation_compatibility,
            )
        if output.ndim != 2 or output.shape[0] != 1:
            raise AlgorithmOutputError("R²AG reader output must be one token sequence")
        output_tokens = int(output.shape[1])
        if not 1 <= output_tokens <= max_new_tokens:
            raise AlgorithmOutputError("R²AG reader output exceeds its generation budget")
        anchor = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not anchor:
            raise AlgorithmOutputError("R²AG reader returned an empty generated anchor")
        return R2AGReaderResult(
            anchor=anchor,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            marker_positions=marker_positions,
            rformer_output_sha256=_tensor_sha256(rformer_output, torch),
            projection_output_sha256=_tensor_sha256(projected, torch),
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "runtime": "r2ag_marker_only_hidden_state_conditioning.v1",
                "upstream_commit": R2AG_UPSTREAM_COMMIT,
                "manifest_sha256": self.manifest_sha256,
                "artifact_closure_sha256": self._manifest.get("artifact_closure_sha256"),
                "base_model_load_set_sha256": self.base_model_load_set_sha256,
                "training_data_sha256": self._manifest.get("training_data_sha256"),
                "reader_backbone": self._manifest.get("reader_backbone"),
                "fp8_model_config_compatibility": (
                    "qwen3.8_fp8_mlp_gate_exclusion.v1"
                    if self._manifest.get("reader_backbone") == _QWEN38_FP8_BACKBONE
                    else None
                ),
                "fp8_generation_compatibility": (
                    "qwen3.8_fp8_min_new_tokens_1.v1"
                    if self._manifest.get("reader_backbone") == _QWEN38_FP8_BACKBONE
                    else None
                ),
                "num_documents": self._manifest.get("num_documents"),
                "marker_token_id": self._manifest.get("marker_token_id"),
                **({"candidate_count_adapter": "r2ag.real_candidate_prefix.absent_neighbor_zero.v1"}
                   if self.allow_short_candidates else {}),
                "upstream_checkpoint_parity_claimed": False,
                "paper_result_parity_claimed": False,
            }
        )


@dataclass(frozen=True, slots=True)
class R2AGCommonSupportConfig:
    retrieval_top_k: int = 6
    max_new_tokens: int = 96
    allow_short_candidates: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_short_candidates, bool):
            raise ValidationError("R²AG allow_short_candidates must be boolean")
        for name, value in (
            ("retrieval_top_k", self.retrieval_top_k),
            ("max_new_tokens", self.max_new_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"R²AG {name} must be positive")
        if self.retrieval_top_k < 2 and not self.allow_short_candidates:
            raise ValidationError("R²AG requires at least two documents")


@dataclass(slots=True)
class R2AGCommonSupport:
    """Expose a retrieval-conditioned anchor and its real documents to an agent."""

    config: R2AGCommonSupportConfig = R2AGCommonSupportConfig()
    method_id: str = "r2ag"
    release_id: str = "acquire.r2ag.common_support"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("R²AG requires a non-empty query")
        runtime = _runtime(context.external_runtime)
        retriever = _retriever(context.retriever)
        if state.budget.max_model_calls < 1:
            raise BudgetExceededError("R²AG has no reader model-call budget")
        if state.budget.max_retrieval_calls < 1:
            raise BudgetExceededError("R²AG has no retrieval budget")
        if state.budget.max_output_tokens < 1:
            raise BudgetExceededError("R²AG has no output-token budget")
        results = _validated_results(
            retriever.retrieve(state.query, top_k=self.config.retrieval_top_k),
            top_k=self.config.retrieval_top_k,
        )
        if not results:
            raise AlgorithmOutputError("R²AG requires at least one real retrieved document")
        if len(results) != self.config.retrieval_top_k and not self.config.allow_short_candidates:
            raise AlgorithmOutputError("R²AG requires its exact checkpoint document count")
        runtime_response = runtime.run(
            R2AGRuntimeRequest(
                query=state.query,
                documents=results,
                max_new_tokens=min(self.config.max_new_tokens, state.budget.max_output_tokens),
                seed=context.seed,
            )
        )
        features, reader_result = _validated_runtime_response(runtime_response, len(results))
        if (
            state.budget.max_context_tokens is not None
            and reader_result.input_tokens > state.budget.max_context_tokens
        ):
            raise BudgetExceededError("R²AG reader exceeded max_context_tokens")
        real_blocks = tuple(
            _real_context(result, feature)
            for result, feature in zip(results, features, strict=True)
        )
        anchor_hash = content_hash(
            {
                "anchor": reader_result.anchor,
                "rformer": reader_result.rformer_output_sha256,
                "projection": reader_result.projection_output_sha256,
            }
        )
        anchor_block = ContextBlock(
            block_id=f"r2ag-generated-anchor:{anchor_hash[:16]}",
            text=(
                "R²AG retrieval-conditioned hypothesis (non-authoritative; verify against the "
                f"retrieved documents): {reader_result.anchor}"
            ),
            source="r2ag_common_support_generated_anchor",
            metadata={
                "evidence_kind": "model_generated_non_authoritative_anchor",
                "generated_anchor_is_evidence": False,
                "anchor_sha256": anchor_hash,
                "rformer_output_sha256": reader_result.rformer_output_sha256,
                "projection_output_sha256": reader_result.projection_output_sha256,
            },
        )
        runtime_identity = dict(runtime.fingerprint())
        reader_identity = runtime_identity.get("reader", {})
        feature_encoder_identity = runtime_identity.get("feature_encoder", {})
        if not isinstance(reader_identity, Mapping) or not isinstance(
            feature_encoder_identity, Mapping
        ):
            raise AlgorithmOutputError("R²AG runtime fingerprint lacks component identities")
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "r2ag_common_support_configuration",
                    {
                        "upstream_commit": R2AG_UPSTREAM_COMMIT,
                        **({"candidate_count_adapter": "r2ag.real_candidate_prefix.absent_neighbor_zero.v1",
                            "real_candidate_count": len(results), "retrieval_cap": self.config.retrieval_top_k}
                           if self.config.allow_short_candidates else {}),
                        "reader": reader_identity,
                        "retriever": dict(retriever.fingerprint()),
                        "feature_encoder": feature_encoder_identity,
                        "external_runtime": runtime_identity,
                        "candidate_document_ids": tuple(item.document.doc_id for item in results),
                        "upstream_checkpoint_parity_claimed": False,
                        "paper_result_parity_claimed": False,
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "r2ag_retrieval_feature_tensor",
                    {
                        "feature_names": ("rerank", "neighbor", "precedent"),
                        "features": tuple(dict(row) for row in features),
                        "feature_tensor_sha256": content_hash(features),
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "r2ag_marker_conditioned_anchor",
                    {
                        "marker_positions": reader_result.marker_positions,
                        "rformer_output_sha256": reader_result.rformer_output_sha256,
                        "projection_output_sha256": reader_result.projection_output_sha256,
                        "anchor_sha256": anchor_hash,
                        "generated_anchor_visible_to_downstream": True,
                        "generated_anchor_is_evidence": False,
                    },
                ),
            ),
            counters={
                "model_calls": 1,
                "retrieval_calls": 1,
                "embedding_calls": 1,
                "input_tokens": reader_result.input_tokens,
                "output_tokens": reader_result.output_tokens,
                "conditioned_documents": len(results),
                "retrieval_feature_values": len(results) * 3,
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    (*real_blocks, anchor_block),
                    target="retrieved_evidence_and_generated_anchor",
                    rationale=(
                        "preserve R²AG hidden-state conditioning as a non-authoritative "
                        "agent input anchor beside real retrieved documents"
                    ),
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"controlled_marker_conditioned_reader:yefd/RRAG@{R2AG_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "common_support_adaptation": (
                    "custom_reader_answer_as_non_authoritative_downstream_context"
                ),
                "generated_anchor_visible_to_downstream": True,
                "generated_anchor_is_evidence": False,
                "real_documents_are_evidence": True,
                "marker_only_embedding_replacement": True,
                "upstream_checkpoint_parity_claimed": False,
                "paper_result_parity_claimed": False,
                "reader_manifest_sha256": reader_identity.get("manifest_sha256"),
                "reader_backbone": reader_identity.get("reader_backbone"),
                "feature_encoder_load_set_sha256": feature_encoder_identity.get(
                    "model_load_set_sha256"
                ),
            },
        )


def _retriever(value: object) -> Retriever:
    if value is None or not hasattr(value, "retrieve") or not hasattr(value, "fingerprint"):
        raise ExternalDependencyError("R²AG common-support requires a Retriever")
    return value  # type: ignore[return-value]


def _runtime(value: object) -> R2AGCommonSupportRuntime:
    if not isinstance(value, R2AGCommonSupportRuntime):
        raise ExternalDependencyError(
            "R²AG common-support requires InterventionContext.external_runtime"
        )
    return value


def _validated_runtime_response(
    value: Mapping[str, Any],
    expected_documents: int,
) -> tuple[tuple[Mapping[str, float], ...], R2AGReaderResult]:
    if not isinstance(value, Mapping) or value.get("status") != "ok":
        raise AlgorithmOutputError("R²AG external runtime returned a failed response")
    raw_features = value.get("features")
    raw_reader = value.get("reader")
    if (
        isinstance(raw_features, (str, bytes))
        or not isinstance(raw_features, Sequence)
        or len(raw_features) != expected_documents
        or not isinstance(raw_reader, Mapping)
    ):
        raise AlgorithmOutputError("R²AG external runtime response schema is invalid")
    features: list[Mapping[str, float]] = []
    for row in raw_features:
        if not isinstance(row, Mapping) or set(row) != {"rerank", "neighbor", "precedent"}:
            raise AlgorithmOutputError("R²AG external runtime feature schema is invalid")
        try:
            parsed = {name: float(row[name]) for name in ("rerank", "neighbor", "precedent")}
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlgorithmOutputError("R²AG external runtime features are non-numeric") from exc
        if not all(math.isfinite(item) for item in parsed.values()):
            raise AlgorithmOutputError("R²AG external runtime features are non-finite")
        features.append(freeze_mapping(parsed))
    try:
        result = R2AGReaderResult(
            anchor=raw_reader["anchor"],
            input_tokens=raw_reader["input_tokens"],
            output_tokens=raw_reader["output_tokens"],
            marker_positions=tuple(raw_reader["marker_positions"]),
            rformer_output_sha256=raw_reader["rformer_output_sha256"],
            projection_output_sha256=raw_reader["projection_output_sha256"],
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise AlgorithmOutputError("R²AG external runtime reader receipt is invalid") from exc
    usage = value.get("usage")
    if (
        not isinstance(usage, Mapping)
        or usage.get("model_calls") != 1
        or usage.get("embedding_calls") != 1
        or usage.get("input_tokens") != result.input_tokens
        or usage.get("output_tokens") != result.output_tokens
    ):
        raise AlgorithmOutputError("R²AG external runtime usage receipt is invalid")
    return tuple(features), result


def _validated_results(
    values: Sequence[RetrievalResult], *, top_k: int
) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(values)
    except TypeError as exc:
        raise AlgorithmOutputError("R²AG retriever returned a non-sequence") from exc
    if not all(isinstance(item, RetrievalResult) for item in results) or len(results) > top_k:
        raise AlgorithmOutputError("R²AG retriever returned invalid results")
    ids = tuple(item.document.doc_id for item in results)
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise AlgorithmOutputError("R²AG retrieval IDs must be non-empty and unique")
    if tuple(item.rank for item in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("R²AG retrieval ranks must be contiguous and one-based")
    return results


def _real_context(
    result: RetrievalResult,
    feature: Mapping[str, float],
) -> ContextBlock:
    document = result.document
    if not document.corpus_revision:
        raise AlgorithmOutputError("R²AG evidence requires a corpus revision")
    reserved = {
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "rank",
        "r2ag_features",
        "title",
    }
    if reserved & set(document.metadata):
        raise AlgorithmOutputError("R²AG document metadata uses reserved provenance keys")
    return ContextBlock(
        block_id=document.doc_id,
        text=document.text,
        source=document.corpus_revision,
        score=result.score,
        metadata={
            **dict(document.metadata),
            "document_id": document.doc_id,
            "title": document.title,
            "corpus_revision": document.corpus_revision,
            "rank": result.rank,
            "r2ag_features": dict(feature),
            "evidence_kind": "real_corpus_document",
        },
    )


def r2ag_reader_prompt(
    query: str,
    documents: Sequence[RetrievalResult],
    *,
    max_document_chars: int,
) -> str:
    rows = [
        "Use the retrieval signals and documents to propose a concise answer. The answer is a "
        "hypothesis and must not invent facts absent from the documents.",
        f"Question: {query.strip()}",
        "Documents:",
    ]
    for index, item in enumerate(documents, start=1):
        title = item.document.title.strip() or item.document.doc_id
        text = item.document.text.strip()[:max_document_chars]
        rows.append(f"[{index}] relevance: {_MARKER} Title: {title}\n{text}")
    rows.append("Answer:")
    return "\n".join(rows)


def _tensor_sha256(value: Any, torch: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def r2ag_model_tree_sha256(root: Path) -> str:
    records: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValidationError("R²AG feature model load set may not contain symlinks")
        if path.is_file():
            records.append((str(path.relative_to(root)), _sha256_file(path)))
    if not records:
        raise ValidationError("R²AG feature model load set is empty")
    return content_hash(records)


def _load_source_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExternalDependencyError("unable to import pinned R²AG source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = [
    "LocalR2AGCommonSupportReader",
    "LocalR2AGCommonSupportRuntime",
    "LocalSentenceTransformerR2AGFeatureEncoder",
    "R2AGCommonSupport",
    "R2AGCommonSupportConfig",
    "R2AGCommonSupportReader",
    "R2AGCommonSupportRuntime",
    "R2AGFeatureEncoder",
    "R2AGReaderResult",
    "R2AGRuntimeRequest",
    "r2ag_model_tree_sha256",
    "r2ag_reader_prompt",
]
