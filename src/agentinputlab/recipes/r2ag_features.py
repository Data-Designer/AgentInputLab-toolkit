"""Label-free native R²AG retrieval features for the trained-reader API."""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .qrag_inference import _stamps
from .r2ag_inference import R2AGReaderDocument
from .r2ag_training import (
    R2AG_UPSTREAM_COMMIT,
    SOURCE_FILES,
    _module,
    _new_root,
    _observe_embedding,
    _safe_model,
    validate_r2ag_training_source,
)
from .recomp_extractive import _LOCK, _save, _scope


@dataclass(frozen=True, slots=True)
class R2AGRetrievalDocument:
    document_id: str
    title: str
    text: str

    def __post_init__(self):
        for name in ("document_id", "title", "text"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or "\x00" in value
                or (name != "title" and not value.strip())
            ):
                raise ValidationError(f"R²AG retrieval document {name} is invalid")


def _validate_inputs(query, documents):
    if not isinstance(query, str) or not query.strip() or "\x00" in query:
        raise ValidationError("R²AG feature query requires non-empty text")
    if (
        not isinstance(documents, Sequence)
        or not 10 <= len(documents) <= 100
        or any(not isinstance(d, R2AGRetrievalDocument) for d in documents)
        or len({d.document_id for d in documents}) != len(documents)
    ):
        raise ValidationError("R²AG native features require 10-100 unique typed documents")
    if sum(len(d.text.encode()) + len(d.title.encode()) for d in documents) > 4 * 1024 * 1024:
        raise ValidationError(
            "R²AG candidate text exceeds 4 MiB; explicitly retrieve fewer documents"
        )


def _native_documents(query, documents):
    # The unchanged native extractor reads only question and the first two
    # context fields. A third field transports IDs through its native sort.
    # It reads but never uses supporting_facts in retrieval computations.
    return dict(
        question=query,
        context=[[d.title, [d.text], {"document_id": d.document_id}] for d in documents],
        supporting_facts=[],
    )


def _restore_documents(rows, documents):
    original = {d.document_id: d for d in documents}
    if not isinstance(rows, list) or len(rows) != len(documents):
        raise ValidationError("R²AG native extraction changed candidate count")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 4 or not isinstance(row[2], dict):
            raise ValidationError("R²AG native feature/document layout differs")
        identifier = row[2].get("document_id")
        if identifier not in original or identifier in seen:
            raise ValidationError("R²AG native extraction changed candidate identity")
        source = original[identifier]
        if row[:3] != [source.title, [source.text], {"document_id": source.document_id}]:
            raise ValidationError("R²AG native extraction changed document text")
        seen.add(identifier)
        result.append(R2AGReaderDocument(identifier, source.title, source.text, row[3]))
    if any(a.features[0] < b.features[0] for a, b in pairwise(result)):
        raise ValidationError("R²AG native extraction did not retain native descending rank")
    return tuple(result)


class R2AGNativeFeatureEncoder:
    """Run the pinned native feature formula without test references or labels.

    The caller chooses the visible candidate pool explicitly. This API does
    not retrieve a hidden corpus, change prefixes or substitute embeddings.
    All candidates are encoded once and saved; the matching reader consumes
    the first ten native-ranked records.
    """

    def __init__(
        self,
        *,
        checkout,
        encoder_root,
        encoder_manifest,
        encoder_manifest_sha256,
        device="cpu",
        source_and_asset_acknowledged=False,
    ):
        if source_and_asset_acknowledged is not True:
            raise ValidationError(
                "R²AG feature encoding requires mounted-source/asset acknowledgement"
            )
        if device not in {"cpu", "cuda:0"}:
            raise ValidationError("R²AG feature encoding requires one logical device")
        self.source = validate_r2ag_training_source(checkout)
        self.encoder, _ = _safe_model(encoder_root, encoder_manifest, encoder_manifest_sha256)
        self.device, self._loaded = device, None
        self._paths = [Path(encoder_manifest)]
        self._paths += [self.source / n for n in SOURCE_FILES]
        self._paths += sorted(p for p in self.encoder.rglob("*") if p.is_file())
        self._identity = _stamps(self._paths)
        self._binding = dict(
            protocol="agentinputlab.r2ag.native_query_features.v1",
            source_commit=R2AG_UPSTREAM_COMMIT,
            encoder_manifest_sha256=encoder_manifest_sha256,
            source_function="retrieval_utils.get_dual_sim_hotpotqa",
            role_prefixes="upstream bare query and Title/document text",
            label_free=True,
            automatic_retrieval=False,
            automatic_retry=False,
            feature_cost_separate=True,
        )
        self.deployment_id = content_hash(self._binding)
        self._parameters = self._parameter_binding()

    def _parameter_binding(self):
        return (
            self.source,
            self.encoder,
            self.device,
            self.deployment_id,
            content_hash(self._binding),
        )

    def _verify(self):
        if self._parameter_binding() != self._parameters or _stamps(self._paths) != self._identity:
            raise ValidationError("R²AG feature deployment files or parameters changed")

    def fingerprint(self):
        self._verify()
        return freeze_mapping(dict(self._binding, deployment_id=self.deployment_id))

    def _load(self):
        self._verify()
        if self._loaded is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ExternalDependencyError(
                    "R²AG native features require pip install 'agentinputlab[r2ag-fit]'"
                ) from exc
            native = _module(
                self.source / "retrieval/retrieval_utils.py", "agentinputlab_r2ag_query_features"
            )
            encoder = SentenceTransformer(
                str(self.encoder),
                device=self.device,
                local_files_only=True,
                trust_remote_code=False,
            )
            encoder.eval()
            self._loaded = torch, native, encoder
        return self._loaded

    def encode(self, *, query, documents, output_root):
        self._verify()
        _validate_inputs(query, documents)
        root = _new_root(output_root, self.source, self.encoder)
        _save(
            root / "claim.json",
            dict(
                query=query,
                documents=[asdict(d) for d in documents],
                deployment=dict(self.fingerprint()),
                reference_answers_supplied=False,
                support_labels_supplied=False,
                automatic_retry=False,
            ),
        )
        observed = dict(
            embedding_forward_attempts=0,
            embedding_forward_responses=0,
            input_tokens=0,
            input_token_unknown_attempts=0,
        )
        start, cpu = time.monotonic(), time.process_time()
        terminal = dict(status="interrupted_unresolved")
        try:
            torch, native, encoder = self._load()
            with _LOCK, _scope(root, 17), torch.no_grad(), _observe_embedding(encoder, observed):
                rows = native.get_dual_sim_hotpotqa([_native_documents(query, documents)], encoder)
            if not isinstance(rows, list) or len(rows) != 1 or rows[0]["question"] != query:
                raise ValidationError("R²AG native extraction changed the query")
            result = _restore_documents(rows[0]["context"], documents)
            _save(root / "features.json", [asdict(d) for d in result])
            terminal = dict(status="features_prepared", candidates=len(result), reader_documents=10)
        except Exception as exc:
            terminal = dict(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            _save(
                root / "terminal.json",
                dict(
                    **terminal,
                    **observed,
                    cpu_seconds=time.process_time() - cpu,
                    wall_seconds=time.monotonic() - start,
                    new_generator_calls=0,
                    new_optimizer_steps=0,
                    automatic_retry=False,
                ),
            )
        return result

    def close(self):
        if self._loaded is not None:
            torch = self._loaded[0]
            self._loaded = None
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
