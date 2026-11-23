"""Typed deployment of the actual native-fitted LLMLingua-2 candidate.

The official released BERT/XLM-R identities keep their original strict guards.
This path runs the same native compression dispatcher under an explicit trained
artifact identity and controlled profile, without re-training or a fallback.
"""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Mapping
from dataclasses import replace

from agentinputlab.algorithms.curate.llmlingua.adapters import (
    LLMLingua2Config,
    LLMLingua2OfficialAdapter,
)
from agentinputlab.contracts import ExternalDependencyError, TraceEvent, TracePhase, ValidationError

from .llmlingua2_training import RELEASE, LLMLingua2TrainedArtifact, _sha


def _file_identity(root):
    if (
        root.is_symlink()
        or (root / "checkpoint").is_symlink()
        or not (root / "checkpoint").is_dir()
    ):
        raise ValidationError("LLMLingua-2 fitted deployment directories changed")
    paths = [
        root / n
        for n in ("artifact.json", "fit-plan.json", "fit-terminal.json", "checkpoint-manifest.json")
    ]
    paths.extend(p for p in (root / "checkpoint").rglob("*") if p.is_file() or p.is_symlink())
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValidationError("LLMLingua-2 fitted deployment files are missing or symlinked")
    return tuple(
        sorted(
            (p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in paths
        )
    )


class LLMLingua2TrainedCompressor(LLMLingua2OfficialAdapter):
    """Native inference with closed fitted weights and a distinct profile.

    Reuse one object for immutable assets. A changed file stamp is rejected,
    not silently reloaded as a different treatment. Full hashes are checked on
    construction; subsequent immutable-file checks avoid repeatedly reading
    multi-gigabyte native evidence. No caller-supplied compressor factory.
    """

    __slots__ = ("_artifact_files", "_artifact_id", "_artifact_root", "_checkpoint_manifest_hash")

    def __init__(self, artifact, *, config=None, device_map="cpu"):
        if not isinstance(artifact, LLMLingua2TrainedArtifact):
            raise ValidationError("LLMLingua-2 deployment requires a typed trained artifact")
        verified = LLMLingua2TrainedArtifact.load(artifact.root)
        if verified.artifact_id != artifact.artifact_id:
            raise ValidationError("LLMLingua-2 deployment artifact identity differs")
        if not isinstance(device_map, str) or not device_map.strip() or "\0" in device_map:
            raise ValidationError("LLMLingua-2 deployment device must be explicit")
        settings = (
            LLMLingua2Config(**config)
            if isinstance(config, Mapping)
            else (config or LLMLingua2Config())
        )
        if not isinstance(settings, LLMLingua2Config):
            raise ValidationError("LLMLingua-2 deployment requires typed compression controls")
        self._artifact_root, self._artifact_id = verified.root, verified.artifact_id
        self._artifact_files = _file_identity(verified.root)
        self._checkpoint_manifest_hash = _sha(verified.root / "checkpoint-manifest.json")
        super().__init__(
            model_name=str(verified.checkpoint),
            model_revision="native-fit-" + verified.artifact_id,
            device_map=device_map,
            model_config={"local_files_only": True, "torch_dtype": "float32"},
            profile_kind="controlled_adapter",
            config=settings,
            release_id=RELEASE,
        )

    def _verify_binding(self):
        if (
            self.model_name != str(self._artifact_root / "checkpoint")
            or self.model_revision != "native-fit-" + self._artifact_id
            or self.release_id != RELEASE
            or self.profile_kind != "controlled_adapter"
            or self.compressor_factory is not None
            or dict(self.model_config) != {"local_files_only": True, "torch_dtype": "float32"}
            or _file_identity(self._artifact_root) != self._artifact_files
        ):
            raise ValidationError("LLMLingua-2 trained deployment binding changed; no substitution")

    def _load(self, *, use_llmlingua2):
        self._verify_binding()
        if use_llmlingua2 is not True:
            raise ValidationError("LLMLingua-2 trained deployment requires token classification")
        native = super()._load(use_llmlingua2=True)
        if importlib.metadata.version("llmlingua") != "0.2.2":
            raise ExternalDependencyError("trained compressor requires reviewed LLMLingua 0.2.2")
        family = {"bert": "bert-base-multilingual-cased", "xlm-roberta": "xlm-roberta-large"}.get(
            native.model.config.model_type
        )
        if family is None or native.model_name not in {self.model_name, family}:
            raise ValidationError("LLMLingua-2 native tokenizer family or source binding differs")
        # In 0.2.2 this post-load field routes ONLY WordPiece/SentencePiece
        # boundary handling in compress_prompt_llmlingua2. Loading already used
        # the real hash-bound fitted path. Do not alias weights or substitute a
        # released checkpoint just to satisfy the upstream substring check.
        native.model_name = family
        return native

    def propose(self, state, context):
        self._verify_binding()
        started, cpu = time.monotonic(), time.process_time()
        patch = super().propose(state, context)
        elapsed, cpu_used = time.monotonic() - started, time.process_time() - cpu
        binding = dict(
            trained_artifact_id=self._artifact_id,
            checkpoint_manifest_sha256=self._checkpoint_manifest_hash,
            native_candidate_not_released_checkpoint=True,
            native_training_repeated=False,
            native_dispatcher="llmlingua.PromptCompressor.compress_prompt",
            cpu_seconds=cpu_used,
            wall_seconds=elapsed,
            generative_model_calls=0,
            query_consumed=False,
            native_word_boundary_family_from_actual_model_config=True,
            original_paper_or_from_scratch_training_parity_claimed=False,
        )
        trace = replace(
            patch.trace,
            events=(
                TraceEvent(TracePhase.PREPARE, "native_fitted_compressor", binding),
                *patch.trace.events,
            ),
            counters={**patch.trace.counters, "cpu_seconds": cpu_used},
        )
        return replace(
            patch,
            trace=trace,
            implementation_identity="native_llmlingua2_dispatcher_with_fitted_candidate",
            metadata={**patch.metadata, **binding},
        )
