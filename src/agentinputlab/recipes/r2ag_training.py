"""R²AG reader-aligned training with the pinned native model and loss.

Native retrieval feature extraction, HotpotQA instruction construction,
RFormer, marker replacement and combined forward loss are imported unchanged.
The explicit training loop supports modern frozen reader snapshots without
executing the upstream runner's hard-coded CUDA assignment or unsafe pickle IO.
"""

from __future__ import annotations

import ast
import gc
import importlib.metadata
import importlib.util
import json
import math
import random
import subprocess
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from agentinputlab.algorithms.acquire.r2ag.bridge import R2AG_UPSTREAM_COMMIT
from agentinputlab.algorithms.acquire.r2ag.common_support import _repair_qwen38_fp8_model_config
from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .llmlingua2_training import _closed_files
from .qrag import QRAGFitControls, _file, _sha, audit_qrag_training_data
from .recomp_extractive import _LOCK, _save, _scope

RELEASE = "acquire.r2ag.official_retrained"
PROTOCOL = "agentinputlab.r2ag.native_reader_fit.v1"
FEATURE_PROTOCOL = "agentinputlab.r2ag.native_features.v1"
MARKER = "<R>"
SOURCE_FILES = {
    "RRAG/models/modeling_rrag.py": "6b63388bfc97ad093aee9464c73098b1a198e7dd4e8b439404dc1be4309b4be9",
    "runner.py": "6e7eb0d721acc24a9419a7632a2aa97192248ce8b991dbc202eaaad982b5fd3e",
    "RRAG/utils/trainer.py": "e061c49bab35c4b4236c798dd89ae66243b68156186f79d4947a89e3e1f595fe",
    "RRAG/dataset/load_hotpotqa.py": "6851a85a42dc18a958c4ff7a16fac8c5bcccae7b00092bc65180fd9e302027fe",
    "retrieval/retrieval_utils.py": "6097ba8196f0847d19f6d1ae6cf857899ce85e0035f066d09a2a2d9ab87a6aa3",
    "RRAG/prompts/qa_similarity.prompt": "f850c6763671e7ec5567d13223ed9964077ab150fdcd5180d649af56fa0bb55f",
    "scripts/train/train_hotpotqa.sh": "6968b2df00cc0bccb6220dc1f08279d86c7d150df54e23093f5230c6792c427a",
    "LICENSE.txt": "0c8b7c4d48fc1a3c3ef53aead40f367e52bcb865b03e148dfb7bedd7c5f61044",
}


def validate_r2ag_training_source(checkout):
    p = Path(checkout).expanduser()
    if p.is_symlink() or not p.is_dir():
        raise ValidationError("R²AG training requires a regular mounted source directory")
    p = p.resolve()
    if (p / ".git").exists():

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(p), *args], text=True, timeout=30
            ).strip()

        if git("rev-parse", "HEAD") != R2AG_UPSTREAM_COMMIT or git("diff", "HEAD", "--name-only"):
            raise ValidationError("R²AG native training source commit or tracked files differ")
        # Imports may leave interpreter caches; other untracked source is rejected.
        unknown = git("ls-files", "--others", "--exclude-standard").splitlines()
        if any("__pycache__" not in Path(n).parts and not n.endswith(".pyc") for n in unknown):
            raise ValidationError("R²AG source contains untracked non-cache files")
    for name, expected in SOURCE_FILES.items():
        f = _file(p / name, "R²AG native source")
        if not f.is_relative_to(p) or _sha(f) != expected:
            raise ValidationError(f"R²AG pinned source differs: {name}")
    return p


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExternalDependencyError("cannot import pinned R²AG source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_formatter(source):
    # Import just the original class, not runner.py's CUDA=1 and seed42 side effects.
    tree = ast.parse((source / "runner.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RRAGRunner")
    scope = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source / "runner.py"), "exec"), scope)
    formatter = scope["RRAGRunner"]
    formatter.set_retrieval_token(MARKER)
    formatter.set_unk_token(MARKER)
    return formatter


@contextmanager
def _observe_embedding(model, counters):
    """Count native direct ``forward`` calls, not only Module.__call__ hooks."""
    original = model.forward

    def forward(*args, **kwargs):
        counters["embedding_forward_attempts"] += 1
        features = args[0] if args else kwargs.get("input", {})
        mask = features.get("attention_mask") if isinstance(features, Mapping) else None
        if mask is not None:
            counters["input_tokens"] += int(mask.sum().item())
        elif "input_token_unknown_attempts" in counters:
            counters["input_token_unknown_attempts"] += 1
        response = original(*args, **kwargs)
        counters["embedding_forward_responses"] += 1
        return response

    model.forward = forward
    try:
        yield
    finally:
        model.forward = original


@contextmanager
def _numeric_scope(cuda):
    """Use the native CUDA TF32 setting without leaking it to the caller."""
    import torch

    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    previous = matmul.allow_tf32, cudnn.allow_tf32
    try:
        if cuda:
            matmul.allow_tf32 = cudnn.allow_tf32 = True
        yield
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = previous


def _new_root(value, *protected):
    p = Path(value).expanduser().absolute()
    if p.exists() or p.is_symlink():
        raise ValidationError("R²AG output must be create-only; no automatic attempt retry")
    if any(p.resolve().is_relative_to(Path(q).resolve()) for q in protected):
        raise ValidationError("R²AG output must not modify mounted source, data or model")
    p.mkdir(parents=True)
    return p


def _safe_model(root, manifest, digest):
    p, closure = _closed_files(root, manifest, digest, "agentinputlab.r2ag.model_snapshot.v1")
    if not {"config.json", "tokenizer_config.json"} <= closure["files"].keys() or not any(
        n.endswith(".safetensors") for n in closure["files"]
    ):
        raise ValidationError("R²AG needs a complete safe model/tokenizer snapshot")
    config = json.loads((p / "config.json").read_text())
    quantization = _quantization_config(config)
    unsafe = {n for n in closure["files"] if n.endswith((".bin", ".pkl", ".pickle", ".py"))}
    # The official GPT-OSS snapshot also ships a Metal-only weight file. It is
    # checksum-bound but never a Transformers loader input: the loader below
    # explicitly requires safetensors. Other binary/code payloads stay rejected.
    if config.get("model_type") == "gpt_oss" and quantization.get("quant_method") == "mxfp4":
        unsafe.discard("metal/model.bin")
    if unsafe:
        raise ValidationError("R²AG mounted model contains an unsafe payload")
    for name in ("config.json", "tokenizer_config.json"):
        if json.loads((p / name).read_text()).get("auto_map"):
            raise ValidationError("R²AG model requires unreviewed remote code")
    files = closure["files"]
    if "adapter_config.json" in files:
        raise ValidationError("R²AG does not implicitly load a PEFT adapter")
    if "model.safetensors.index.json" in files:
        index = json.loads((p / "model.safetensors.index.json").read_text())
        shards = index.get("weight_map", {})
        if (
            "model.safetensors" in files
            or not isinstance(shards, dict)
            or not shards
            or any(
                not isinstance(n, str) or n not in files or not n.endswith(".safetensors")
                for n in shards.values()
            )
        ):
            raise ValidationError("R²AG safetensors index is ambiguous or has unpinned shards")
    elif "model.safetensors" not in files:
        raise ValidationError("R²AG needs a pinned single or indexed safetensors layout")
    return p, closure


def _quantization_config(config):
    if not isinstance(config, Mapping):
        raise ValidationError("R²AG model config must be an object")
    quantization = config.get("quantization_config")
    if quantization is None:
        return {}
    if not isinstance(quantization, Mapping):
        raise ValidationError("R²AG quantization config must be an object or null")
    return quantization


def _reader_training_precision_config(config):
    """Reject a known missing backward path before paying for reader execution.

    RFormer training needs gradients with respect to frozen-reader inputs.
    The native finegrained FP8 path currently has no input autograd formula;
    loading it for inference successfully does not establish trainability.
    """
    if _quantization_config(config).get("quant_method") == "fp8":
        raise ExternalDependencyError(
            "R²AG reader-aligned training requires frozen-reader input gradients, but the "
            "native finegrained FP8 kernel has no registered input backward formula. "
            "This training configuration is unsupported; no reader forward was started. "
            "Inference compatibility does not establish training compatibility, and the "
            "recipe will not silently dequantize or substitute a gradient estimator."
        )


def _reader_training_precision(root):
    _reader_training_precision_config(json.loads((Path(root) / "config.json").read_text()))


def _reader_training_snapshot_preflight(root, manifest, digest):
    """Reject a checksum-bound unsupported header without reading large weights.

    This negative-only check does not certify the snapshot. Supported headers
    still go through the complete file-closure and payload checks in _safe_model.
    """
    closure = load_pinned_json_mapping(manifest, expected_sha256=digest)
    p = Path(root).expanduser()
    if p.is_symlink() or not p.is_dir():
        raise ValidationError("R²AG training requires a regular mounted model directory")
    files = closure.get("files")
    if (
        closure.get("schema") != "agentinputlab.r2ag.model_snapshot.v1"
        or not isinstance(files, dict)
        or "config.json" not in files
    ):
        raise ValidationError("R²AG training requires a pinned model config in its snapshot")
    config = load_pinned_json_mapping(p / "config.json", expected_sha256=files["config.json"])
    _reader_training_precision_config(config)


def _quantized_reader_dependencies(root):
    """Resolve native kernel software before a retrieval/reader forward starts."""
    config = json.loads((Path(root) / "config.json").read_text())
    kind = _quantization_config(config).get("quant_method")
    if kind not in {"fp8", "mxfp4"}:
        return dict(quant_method=kind, kernels=[])
    try:
        from kernels import get_loaded_kernels
        from transformers.integrations.hub_kernels import get_kernel

        if kind == "fp8":
            from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel

            load_finegrained_fp8_kernel()
            repository = "kernels-community/finegrained-fp8"
        else:
            repository = "kernels-community/gpt-oss-triton-kernels"
            get_kernel(repository, version=1)
        records = []
        for loaded in get_loaded_kernels():
            infos = getattr(loaded, "repo_infos", None)
            if infos is None:
                info = getattr(loaded, "repo_info", None)
                infos = [] if info is None else [info]
            for info in infos:
                if info.repo_id == repository:
                    records.append(dict(repository=repository, revision=str(info.revision)))
        if not records:
            raise ExternalDependencyError("R²AG native kernel origin was not recorded")
        records = [
            dict(repository=repo, revision=revision)
            for repo, revision in sorted({(r["repository"], r["revision"]) for r in records})
        ]
        return dict(quant_method=kind, kernels=records, preflight_neural_forwards=0)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        raise ExternalDependencyError(
            f"R²AG {kind} kernel preflight failed before model execution; preload the pinned "
            "native kernel into KERNELS_CACHE and keep the checkpoint quantization unchanged"
        ) from exc


def _audit_hotpot(train, dev, byte_cap):
    audit = audit_qrag_training_data(train, dev, QRAGFitControls(max_data_bytes=byte_cap))
    for spec in audit["files"].values():
        rows = json.loads(Path(spec["path"]).read_text())
        if any(len(r["context"]) != 10 for r in rows):
            raise ValidationError(
                "native R²AG HotpotQA preparation requires exactly ten source documents"
            )
    return dict(
        files=audit["files"],
        id_overlap=False,
        query_overlap=False,
        task="hotpotqa_native_10",
        real_support_labels_required=True,
    )


@dataclass(frozen=True, slots=True)
class R2AGFitControls:
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_steps: int | None = None
    max_optimizer_steps: int = 10000
    max_sequence_length: int = 4096
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    seed: int = 17
    device: str = "cuda:0"
    max_data_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in (
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "max_optimizer_steps",
            "max_sequence_length",
            "seed",
            "max_data_bytes",
        ):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name == "seed" else 1):
                raise ValidationError(f"R²AG invalid integer control: {name}")
        if self.max_steps is not None and (
            type(self.max_steps) is not int or not 1 <= self.max_steps <= self.max_optimizer_steps
        ):
            raise ValidationError("R²AG explicit optimizer steps exceed the declared cap")
        for name in ("learning_rate", "warmup_ratio", "max_grad_norm"):
            v = getattr(self, name)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
            ):
                raise ValidationError(f"R²AG invalid numeric control: {name}")
        if self.warmup_ratio >= 1 or self.device not in {"cpu", "cuda:0"}:
            raise ValidationError("R²AG requires a valid warmup ratio and one logical device")


def prepare_r2ag_native_features(
    *,
    checkout,
    encoder_root,
    encoder_manifest,
    encoder_manifest_sha256,
    train_file,
    dev_file,
    output_root,
    device="cpu",
    max_data_bytes=64 * 1024 * 1024,
    source_and_asset_acknowledged=False,
):
    """Run the unchanged upstream extractor on caller-provided native JSON rows."""
    if source_and_asset_acknowledged is not True:
        raise ValidationError(
            "R²AG feature extraction requires mounted-source/data acknowledgement"
        )
    if device not in {"cpu", "cuda:0"}:
        raise ValidationError("R²AG feature extraction requires one logical device")
    source = validate_r2ag_training_source(checkout)
    encoder, _ = _safe_model(encoder_root, encoder_manifest, encoder_manifest_sha256)
    audit = _audit_hotpot(train_file, dev_file, max_data_bytes)
    root = _new_root(output_root, source, encoder, Path(train_file).parent)
    plan = dict(
        protocol=FEATURE_PROTOCOL,
        source_commit=R2AG_UPSTREAM_COMMIT,
        source_files=SOURCE_FILES,
        encoder_manifest_sha256=encoder_manifest_sha256,
        data_audit=audit,
        native_function="retrieval_utils.get_dual_sim_hotpotqa",
        device=device,
        role_prefixes="upstream bare query and Title/document text",
        references_used_only_for_offline_support_labels=True,
        embedding_forward_counter_scope="SentenceTransformer.forward, including direct calls",
        automatic_retry=False,
    )
    _save(root / "feature-plan.json", plan)
    counters = dict(embedding_forward_attempts=0, embedding_forward_responses=0, input_tokens=0)
    start, cpu = time.monotonic(), time.process_time()
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        with _LOCK, _scope(root, 17), torch.no_grad():
            native = _module(
                source / "retrieval/retrieval_utils.py", "agentinputlab_r2ag_native_features"
            )
            model = SentenceTransformer(
                str(encoder), device=device, local_files_only=True, trust_remote_code=False
            )
            model.eval()

            with _observe_embedding(model, counters):
                for split, spec in audit["files"].items():
                    raw = json.loads(Path(spec["path"]).read_text())
                    featured = native.get_dual_sim_hotpotqa(raw, model)
                    if len(featured) != len(raw) or any(len(r["context"]) != 10 for r in featured):
                        raise ValidationError("R²AG native feature extraction changed source count")
                    _save(root / f"{split}-features.json", featured)
        manifest = dict(
            protocol=FEATURE_PROTOCOL,
            source_commit=R2AG_UPSTREAM_COMMIT,
            encoder_manifest_sha256=encoder_manifest_sha256,
            data_audit=audit,
            files={
                n: _sha(root / n)
                for n in ("feature-plan.json", "train-features.json", "dev-features.json")
            },
        )
        _save(root / "feature-manifest.json", manifest)
        _save(
            root / "feature-terminal.json",
            dict(
                status="features_prepared",
                **counters,
                cpu_seconds=time.process_time() - cpu,
                wall_seconds=time.monotonic() - start,
                new_optimizer_steps=0,
                new_generative_model_calls=0,
                automatic_retry=False,
            ),
        )
        return root
    except Exception as exc:
        _save(
            root / "feature-terminal.json",
            dict(
                status="failed_terminal",
                **counters,
                error_type=type(exc).__name__,
                error=str(exc),
                cpu_seconds=time.process_time() - cpu,
                wall_seconds=time.monotonic() - start,
                automatic_retry=False,
            ),
        )
        raise


def validate_r2ag_features(root, expected_manifest_sha256):
    p = Path(root).resolve()
    manifest = _file(p / "feature-manifest.json", "feature manifest")
    if _sha(manifest) != expected_manifest_sha256:
        raise ValidationError("R²AG native feature manifest checksum differs")
    data = json.loads(manifest.read_text())
    if (
        data.get("protocol") != FEATURE_PROTOCOL
        or data.get("source_commit") != R2AG_UPSTREAM_COMMIT
    ):
        raise ValidationError("R²AG feature protocol/source differs")
    for name in ("feature-plan.json", "train-features.json", "dev-features.json"):
        if _sha(_file(p / name, "native feature file")) != data["files"][name]:
            raise ValidationError("R²AG native feature file differs")
    return p, data


@contextmanager
def _reader_loader(native, root, device, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM

    original = native.AutoModelForCausalLM

    def load(path, *, device_map, load_in_8bit):
        if Path(path).resolve() != root or device_map != "auto" or load_in_8bit is not False:
            raise ValidationError("R²AG native reader attempted an undeclared model load")
        cfg = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        _repair_qwen38_fp8_model_config(cfg, backbone=root.name)
        model = AutoModelForCausalLM.from_pretrained(
            root,
            config=cfg,
            dtype=dtype,
            device_map={"": device},
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        expected_quant = getattr(cfg, "quantization_config", {})
        if isinstance(expected_quant, Mapping) and expected_quant.get("quant_method") in {
            "fp8",
            "mxfp4",
        }:
            actual_quant = getattr(model.config, "quantization_config", {})
            if hasattr(actual_quant, "to_dict"):
                actual_quant = actual_quant.to_dict()
            if (
                not getattr(model, "is_quantized", False)
                or not isinstance(actual_quant, Mapping)
                or actual_quant.get("quant_method") != expected_quant["quant_method"]
                or actual_quant.get("dequantize", False)
            ):
                raise ExternalDependencyError(
                    "R²AG reader load changed the checkpoint quantization; dense fallback is disabled"
                )
        return model

    native.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load)
    try:
        yield
    finally:
        native.AutoModelForCausalLM = original


def _parameter_hashes(module):
    import torch

    return {
        n: __import__("hashlib")
        .sha256(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        .hexdigest()
        for n, v in module.named_parameters()
    }


def _collate(rows, tokenizer, formatter, device, max_length):
    import torch

    texts = formatter.format_instruction(
        {"instruction": [r["instruction"] for r in rows], "output": [r["output"] for r in rows]}
    )
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    labels = tokens.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return dict(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        labels=labels,
        embeds=torch.tensor([r["embeds"] for r in rows], dtype=torch.float32, device=device),
        label=torch.tensor([r["label"] for r in rows], dtype=torch.float32, device=device),
    )


def _count_reader_request(observed, kwargs):
    observed["reader_forward_attempts"] += 1
    mask = kwargs.get("attention_mask")
    if mask is None:
        observed["reader_input_token_unknown_attempts"] += 1
    else:
        observed["reader_input_tokens_attempted"] += int(mask.sum().item())


@dataclass(frozen=True, slots=True)
class R2AGTrainedArtifact:
    root: Path
    artifact_id: str

    def runtime(self, **kwargs):
        from .r2ag_inference import R2AGTrainedReader

        return R2AGTrainedReader(self, **kwargs)

    @classmethod
    def load(cls, root):
        p = Path(root).expanduser()
        if p.is_symlink() or not p.is_dir():
            raise ValidationError("R²AG trained artifact requires a regular directory")
        metadata = json.loads(_file(p / "artifact.json", "artifact metadata").read_text())
        identity = metadata.pop("artifact_id")
        if metadata.get("protocol") != PROTOCOL or identity != content_hash(metadata):
            raise ValidationError("R²AG trained artifact identity differs")
        for name, digest in metadata["files"].items():
            file = _file(p / name, "trained artifact file")
            if not file.is_relative_to(p.resolve()) or _sha(file) != digest:
                raise ValidationError("R²AG trained artifact file differs")
        terminal = json.loads((p / "fit-terminal.json").read_text())
        if terminal.get("status") != "trained_candidate" or terminal.get("optimizer_steps", 0) < 1:
            raise ValidationError("R²AG artifact is not a closed trained candidate")
        return cls(p.resolve(), identity)


class R2AGOfflineTraining:
    release_id = RELEASE

    def __init__(self, controls=None):
        self.controls = (
            R2AGFitControls(**controls)
            if isinstance(controls, Mapping)
            else controls or R2AGFitControls()
        )
        if not isinstance(self.controls, R2AGFitControls):
            raise ValidationError("R²AG fit requires typed controls")

    def fit(
        self,
        *,
        checkout,
        base_model_root,
        base_model_manifest,
        base_model_manifest_sha256,
        features_root,
        feature_manifest_sha256,
        output_root,
        source_and_asset_acknowledged=False,
    ):
        if source_and_asset_acknowledged is not True:
            raise ValidationError("R²AG fit requires mounted-source/model/data acknowledgement")
        c = self.controls
        source = validate_r2ag_training_source(checkout)
        _reader_training_snapshot_preflight(
            base_model_root, base_model_manifest, base_model_manifest_sha256
        )
        base, model_closure = _safe_model(
            base_model_root, base_model_manifest, base_model_manifest_sha256
        )
        _reader_training_precision(base)
        features, feature_manifest = validate_r2ag_features(features_root, feature_manifest_sha256)
        try:
            versions = {
                n: importlib.metadata.version(n)
                for n in ("torch", "transformers", "bitsandbytes", "kernels")
            }
        except importlib.metadata.PackageNotFoundError as exc:
            raise ExternalDependencyError(
                "R²AG fit requires pip install 'agentinputlab[r2ag-fit]'"
            ) from exc
        root = _new_root(output_root, source, base, features)
        _save(
            root / "fit-plan.json",
            dict(
                protocol=PROTOCOL,
                release_id=RELEASE,
                source_commit=R2AG_UPSTREAM_COMMIT,
                source_files=SOURCE_FILES,
                controls=asdict(c),
                base_model_manifest_sha256=base_model_manifest_sha256,
                feature_manifest_sha256=feature_manifest_sha256,
                dependency_versions=versions,
                native_model_class="RRAGLlamaForCausalLM",
                native_forward_loss="language_model_loss + RFormer_BCE",
                optimizer="bitsandbytes.PagedAdamW32bit",
                optimizer_beta=[0.9, 0.999],
                optimizer_epsilon=1e-8,
                weight_decay=0.0,
                scheduler="constant_with_warmup",
                labels="all non-padding tokens, as native SFT collator",
                trainer="explicit modern-reader loop; source RFormer/encode_inputs/forward unchanged",
                marker_transport="dedicated <R> vocabulary row; reader frozen after resize",
                model_load_format="safetensors_only",
                ancillary_metal_weights_deserialized=False,
                quantization_fallback_allowed=False,
                bf16_autocast=c.device.startswith("cuda"),
                tf32=c.device.startswith("cuda"),
                offline_training_cost_separate=True,
                automatic_retry=False,
                paper_result_parity=False,
            ),
        )
        observed = dict(
            reader_forward_attempts=0,
            reader_forward_responses=0,
            reader_input_tokens_attempted=0,
            reader_input_token_unknown_attempts=0,
            optimizer_steps=0,
            backward_calls=0,
            training_input_tokens=0,
            microbatches=[],
        )
        start, cpu = time.monotonic(), time.process_time()
        try:
            import torch
            from bitsandbytes.optim import PagedAdamW32bit
            from transformers import AutoConfig, AutoTokenizer, get_constant_schedule_with_warmup

            _save(root / "runtime-dependencies.json", _quantized_reader_dependencies(base))
            with _LOCK, _scope(root, c.seed), _numeric_scope(c.device.startswith("cuda")):
                dtype = torch.bfloat16 if c.device.startswith("cuda") else torch.float32
                native = _module(
                    source / "RRAG/models/modeling_rrag.py", "agentinputlab_r2ag_native_training"
                )
                loader = _module(
                    source / "RRAG/dataset/load_hotpotqa.py", "agentinputlab_r2ag_native_data"
                )
                loader.PROMPTS_ROOT = source / "RRAG/prompts"
                formatter = _native_formatter(source)
                tokenizer = AutoTokenizer.from_pretrained(
                    base, local_files_only=True, trust_remote_code=False
                )
                tokenizer.padding_side, tokenizer.pad_token = "left", tokenizer.eos_token
                added = tokenizer.add_special_tokens({"additional_special_tokens": [MARKER]})
                if added != 1:
                    raise ValidationError("R²AG expected a new dedicated retrieval marker")
                cfg = AutoConfig.from_pretrained(
                    base, local_files_only=True, trust_remote_code=False
                )
                decoder_cfg = getattr(cfg, "text_config", cfg)
                config = native.RRAGLlamaConfig(
                    model_name_or_path=str(base),
                    load_in_8bit=False,
                    input_dim=3,
                    hidden_size=int(decoder_cfg.hidden_size),
                    unk_token=MARKER,
                    unk_token_id=tokenizer.convert_tokens_to_ids(MARKER),
                    num_k=10,
                    d_model=256,
                    freeze_llm=True,
                )
                with _reader_loader(native, base, c.device, dtype):
                    model = native.RRAGLlamaForCausalLM(config)
                model.llama_model.resize_token_embeddings(len(tokenizer))
                for p in model.llama_model.parameters():
                    p.requires_grad_(False)
                model.llama_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": True}
                )
                if model.get_input_embeddings().weight.shape[-1] != config.hidden_size:
                    raise ValidationError(
                        "R²AG trained projection width differs from the actual reader"
                    )
                data = {}
                for split in ("train", "dev"):
                    rows = json.loads((features / f"{split}-features.json").read_text())
                    prepared = loader.pre_hotpotqa(rows)
                    prepared = loader.get_instruction_dataset(
                        prepared, c.max_sequence_length, tokenizer, True, False, MARKER
                    )
                    data[split] = loader.get_embeds(prepared)
                    if not data[split]:
                        raise ValidationError(
                            "R²AG native input-length filtering removed every example"
                        )
                _save(
                    root / "native-data-audit.json",
                    dict(
                        feature_source=feature_manifest,
                        emitted_rows={s: len(v) for s, v in data.items()},
                        emitted_ids={
                            s: [r.get("_id", r.get("id")) for r in v] for s, v in data.items()
                        },
                        native_length_filter_applied=True,
                        tokenizer_marker_id=config.unk_token_id,
                        reader_hidden_size=config.hidden_size,
                        model_snapshot=model_closure,
                    ),
                )
                params = [p for m in (model.r_former, model.llama_proj) for p in m.parameters()]
                initial = {
                    name: _parameter_hashes(getattr(model, name))
                    for name in ("r_former", "llama_proj")
                }
                batches_per_epoch = math.ceil(len(data["train"]) / c.batch_size)
                available_steps = (
                    math.ceil(batches_per_epoch / c.gradient_accumulation_steps) * c.epochs
                )
                steps = c.max_steps or available_steps
                if steps > min(available_steps, c.max_optimizer_steps):
                    raise ValidationError("R²AG optimizer plan exceeds data/epoch or step cap")
                optimizer = PagedAdamW32bit(
                    params, lr=c.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
                )
                scheduler = get_constant_schedule_with_warmup(
                    optimizer, math.ceil(c.warmup_ratio * steps)
                )

                def before(module, args, kwargs):
                    _count_reader_request(observed, kwargs)

                def after(module, args, kwargs, response):
                    observed["reader_forward_responses"] += 1
                    observed["last_language_loss"] = float(response.loss.detach())

                def retrieval_loss(module, args, response):
                    observed["last_retrieval_loss"] = (
                        float(response[1].detach()) if response[1] is not None else None
                    )

                hooks = [
                    model.llama_model.register_forward_pre_hook(before, with_kwargs=True),
                    model.llama_model.register_forward_hook(after, with_kwargs=True),
                    model.r_former.register_forward_hook(retrieval_loss),
                ]
                try:
                    optimizer.zero_grad(set_to_none=True)
                    for epoch in range(c.epochs):
                        order = list(range(len(data["train"])))
                        random.Random(c.seed + epoch).shuffle(order)
                        model.train()
                        for offset in range(0, len(order), c.batch_size):
                            group = [
                                data["train"][i] for i in order[offset : offset + c.batch_size]
                            ]
                            batch = _collate(
                                group, tokenizer, formatter, c.device, c.max_sequence_length
                            )
                            if batch["input_ids"].shape[-1] > c.max_sequence_length:
                                raise ValidationError(
                                    "R²AG formatted training prompt exceeds declared length"
                                )
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=c.device.startswith("cuda"),
                            ):
                                response = model(**batch, use_cache=False)
                            if not torch.isfinite(response.loss):
                                raise ValidationError("R²AG native combined loss is non-finite")
                            (response.loss / c.gradient_accumulation_steps).backward()
                            observed["backward_calls"] += 1
                            observed["training_input_tokens"] += int(batch["attention_mask"].sum())
                            observed["microbatches"].append(
                                dict(
                                    epoch=epoch,
                                    ids=[r.get("_id", r.get("id")) for r in group],
                                    combined_loss=float(response.loss.detach()),
                                    language_loss=observed["last_language_loss"],
                                    retrieval_loss=observed["last_retrieval_loss"],
                                    reader_forward=True,
                                )
                            )
                            index = offset // c.batch_size + 1
                            if (
                                index % c.gradient_accumulation_steps == 0
                                or index == batches_per_epoch
                            ):
                                if any(p.grad is not None for p in model.llama_model.parameters()):
                                    raise ValidationError(
                                        "R²AG frozen reader accumulated parameter gradients"
                                    )
                                torch.nn.utils.clip_grad_norm_(
                                    params, c.max_grad_norm, error_if_nonfinite=True
                                )
                                optimizer.step()
                                observed["optimizer_steps"] += 1
                                observed.setdefault("optimizer_events", []).append(
                                    dict(
                                        step=observed["optimizer_steps"],
                                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                                    )
                                )
                                scheduler.step()
                                optimizer.zero_grad(set_to_none=True)
                            if observed["optimizer_steps"] == steps:
                                break
                        if observed["optimizer_steps"] == steps:
                            break
                    model.eval()
                    validation = []
                    with torch.no_grad():
                        for row in data["dev"]:
                            batch = _collate(
                                [row], tokenizer, formatter, c.device, c.max_sequence_length
                            )
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=c.device.startswith("cuda"),
                            ):
                                value = model(**batch, use_cache=False).loss
                            if not torch.isfinite(value):
                                raise ValidationError("R²AG native validation loss is non-finite")
                            validation.append(float(value))
                finally:
                    for hook in hooks:
                        hook.remove()
                final = {name: _parameter_hashes(getattr(model, name)) for name in initial}
                changed = {
                    name: sum(h != final[name][key] for key, h in initial[name].items())
                    for name in initial
                }
                if not all(changed.values()) or observed["optimizer_steps"] != steps:
                    raise ValidationError(
                        "R²AG fit did not change both trained components as planned"
                    )
                model.save_model(str(root / "native"))
                tokenizer.save_pretrained(root / "native/tokenizer")
                marker_id = config.unk_token_id
                torch.save(
                    dict(
                        input_row=model.get_input_embeddings().weight[marker_id].detach().cpu(),
                        output_row=model.get_output_embeddings().weight[marker_id].detach().cpu(),
                        marker_id=marker_id,
                    ),
                    root / "native/marker-rows.pt",
                )
                _save(
                    root / "parameter-proof.json",
                    dict(
                        initial=initial,
                        final=final,
                        changed=changed,
                        frozen_reader_parameter_gradients=False,
                        new_neural_forwards=0,
                    ),
                )
                terminal = dict(
                    status="trained_candidate",
                    **observed,
                    changed_parameter_tensors=changed,
                    validation_losses=validation,
                    cpu_seconds=time.process_time() - cpu,
                    wall_seconds=time.monotonic() - start,
                    new_generative_model_calls=0,
                    gradient_checkpointing=True,
                    reader_forward_counter_scope="outer reader forward only; checkpoint layer recomputations included in CPU/wall/peak cost",
                    training_input_token_counter_scope="completed backward microbatches only",
                    reader_input_token_counter_scope="all attempted outer reader forwards, including failures and validation; excludes checkpoint recomputation",
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
                    if c.device.startswith("cuda")
                    else 0,
                    native_training_outputs_not_benchmark_scores=True,
                    automatic_retry=False,
                )
                _save(root / "fit-terminal.json", terminal)
                files = {
                    p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*") if p.is_file()
                }
                metadata = dict(
                    protocol=PROTOCOL,
                    release_id=RELEASE,
                    source_commit=R2AG_UPSTREAM_COMMIT,
                    files=files,
                    base_model_manifest_sha256=base_model_manifest_sha256,
                    feature_manifest_sha256=feature_manifest_sha256,
                    paper_result_parity=False,
                )
                artifact_id = content_hash(metadata)
                _save(root / "artifact.json", dict(metadata, artifact_id=artifact_id))
                del model
                gc.collect()
                if c.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            return R2AGTrainedArtifact(root.resolve(), artifact_id)
        except Exception as exc:
            if not (root / "fit-terminal.json").exists():
                _save(
                    root / "fit-terminal.json",
                    dict(
                        status="failed_terminal",
                        **observed,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        cpu_seconds=time.process_time() - cpu,
                        wall_seconds=time.monotonic() - start,
                        automatic_retry=False,
                    ),
                )
            raise


__all__ = [
    "R2AGFitControls",
    "R2AGOfflineTraining",
    "R2AGTrainedArtifact",
    "prepare_r2ag_native_features",
]
