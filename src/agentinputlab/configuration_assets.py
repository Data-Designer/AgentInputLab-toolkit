"""Allowlisted local artifact bindings for editable run configurations.

These are data loaders, not arbitrary Python factories. Inspection performs no
file I/O; execution verifies exact bytes with the existing native loaders.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import ValidationError

_FIELDS = {
    "recipe.ace.eval_only": "artifact",
    "recipe.skillopt.eval_only": "artifact",
    "recipe.mce.text_artifact_eval": "artifact",
    "rewrite.query2doc.official_cached": "cache",
    "acquire.adaptive_rag.common_support": "router",
    "acquire.adaptive_rag.selector_replay": "records",
    "plan.plan_and_budget.native_cached_soft_budget": "records",
    "plan.plan_and_budget.full": "records",
    "plan.plan_and_budget.hard_cap_extension": "records",
    "rewrite.infocqr.cached_rw_zsl": "cache",
    "rewrite.infocqr.cached_rw_fsl": "cache",
    "rewrite.infocqr.cached_edit": "cache",
}


def artifact_parameters(release_id: str, references: Mapping[str, Any], *,
                        load: bool) -> dict[str, Any]:
    result = {}
    for field, reference in references.items():
        if _FIELDS.get(release_id) != field:
            raise ValidationError(f"artifact field {field!r} is not supported for {release_id}")
        if release_id == "rewrite.query2doc.official_cached":
            from agentinputlab.algorithms.reformulate.query2doc.official_cached import (
                OFFICIAL_TREC_ARTIFACTS,
                Query2DocCache,
            )

            if (not isinstance(reference, Mapping)
                    or set(reference) != {"path", "split", "accept_license_review"}):
                raise ValidationError("Query2doc cache requires path, split and accept_license_review")
            path, split = reference["path"], reference["split"]
            if not isinstance(path, str) or not path.strip() or "\x00" in path:
                raise ValidationError("Query2doc cache.path must be a non-empty local path")
            if not isinstance(split, str) or split not in OFFICIAL_TREC_ARTIFACTS:
                raise ValidationError("Query2doc cache.split must be trec_dl2019 or trec_dl2020")
            if reference["accept_license_review"] is not True:
                raise ValidationError("review Query2doc and underlying dataset terms before accepting")
            result[field] = (Query2DocCache.from_official_jsonl(
                path, split=split, accept_license_review=True) if load else None)
            continue
        required = {"path", "sha256"}
        if field == "cache":
            required |= {"artifact_id", "dataset_revision"}
        optional = {"device"} if field == "router" else set()
        if (not isinstance(reference, Mapping) or not required.issubset(reference)
                or set(reference) - required - optional):
            raise ValidationError(f"artifacts.{field} requires exactly {', '.join(sorted(required))}")
        if field == "router":
            device = reference.get("device", "cpu")
            if not isinstance(device, str) or re.fullmatch(r"(?:cpu|cuda|mps)(?::[0-9]+)?", device) is None:
                raise ValidationError("artifacts.router.device must be cpu, cuda or mps with an optional device index")
        if field == "cache":
            for identity in ("artifact_id", "dataset_revision"):
                if not isinstance(reference[identity], str) or not reference[identity].strip():
                    raise ValidationError(f"artifacts.cache.{identity} must be non-empty text")
        path, digest = reference["path"], reference["sha256"]
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise ValidationError(f"artifacts.{field}.path must be a non-empty local path")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest.lower())):
            raise ValidationError(f"artifacts.{field}.sha256 must be a SHA-256 hex digest")
        if not load:
            # Only used for signature binding; never passed to a constructor.
            result[field] = None
        elif field == "router":
            from agentinputlab.algorithms.acquire.adaptive_rag_common_support import (
                T5AdaptiveRAGRouter,
            )

            result[field] = T5AdaptiveRAGRouter(
                artifact_root=Path(path), manifest_sha256=digest, device=device,
            )
        elif field == "cache":
            from agentinputlab.algorithms.reformulate.infocqr import InfoCQRCache

            result[field] = InfoCQRCache.from_jsonl(
                path, expected_sha256=digest, artifact_id=reference["artifact_id"],
                dataset_revision=reference["dataset_revision"],
            )
        elif release_id == "acquire.adaptive_rag.selector_replay":
            from collections.abc import Sequence

            from agentinputlab.algorithms.acquire.adaptive_rag.replay import AdaptiveRAGRecord
            from agentinputlab.artifacts import load_pinned_json_mapping
            from agentinputlab.configuration import _typed_parameter

            payload = load_pinned_json_mapping(path, expected_sha256=digest)
            records = payload.get("records")
            if not isinstance(records, list) or not records or len(records) > 10000:
                raise ValidationError("Adaptive-RAG artifact requires a nonempty bounded records array")
            typed = _typed_parameter(records, Sequence[AdaptiveRAGRecord], "artifacts.records")
            if any(record.dependency_hash != payload.get("dependency_hash") for record in typed):
                raise ValidationError("Adaptive-RAG artifact mixes dependency bundles")
            result[field] = typed
        elif release_id == "recipe.ace.eval_only":
            from agentinputlab.recipes.ace import ACEContextArtifact

            result[field] = ACEContextArtifact.from_json(path, expected_sha256=digest)
        elif release_id == "recipe.skillopt.eval_only":
            from agentinputlab.recipes.skillopt import SkillOptArtifact

            result[field] = SkillOptArtifact.from_json(path, expected_sha256=digest)
        elif release_id == "recipe.mce.text_artifact_eval":
            from agentinputlab.recipes.mce_text_only import MCETextArtifact

            result[field] = MCETextArtifact.from_json(path, expected_sha256=digest)
        else:
            from agentinputlab.algorithms.decompose.plan_and_budget.cached_soft_budget import (
                PlanAndBudgetCache,
            )

            result[field] = PlanAndBudgetCache.from_native_json(path, expected_sha256=digest)
    return result
