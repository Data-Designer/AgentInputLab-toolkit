"""Leakage-safe Adaptive-RAG replay and controlled router-parity reimplementation."""

from .replay import (
    ADAPTIVE_RAG_UPSTREAM_COMMIT,
    AdaptiveRAGEvaluationLabelAccess,
    AdaptiveRAGPredictionSource,
    AdaptiveRAGPredictionType,
    AdaptiveRAGRecord,
    AdaptiveRAGRouterPrediction,
    AdaptiveRAGSelectorReplay,
    AdaptiveRAGStrategy,
)
from .runtime import (
    ADAPTIVE_RAG_COMPATIBILITY_MANIFEST_SHA256,
    ADAPTIVE_RAG_SOURCE_CLOSURE_SHA256,
    ADAPTIVE_RAG_UPSTREAM_TREE,
    AdaptiveRAGArtifactIdentity,
    AdaptiveRAGControlledWorker,
    AdaptiveRAGControls,
    AdaptiveRAGDocument,
    AdaptiveRAGInitialRetrieval,
    AdaptiveRAGIRCoTStep,
)

__all__ = [
    "ADAPTIVE_RAG_COMPATIBILITY_MANIFEST_SHA256",
    "ADAPTIVE_RAG_SOURCE_CLOSURE_SHA256",
    "ADAPTIVE_RAG_UPSTREAM_COMMIT",
    "ADAPTIVE_RAG_UPSTREAM_TREE",
    "AdaptiveRAGArtifactIdentity",
    "AdaptiveRAGControlledWorker",
    "AdaptiveRAGControls",
    "AdaptiveRAGDocument",
    "AdaptiveRAGEvaluationLabelAccess",
    "AdaptiveRAGIRCoTStep",
    "AdaptiveRAGInitialRetrieval",
    "AdaptiveRAGPredictionSource",
    "AdaptiveRAGPredictionType",
    "AdaptiveRAGRecord",
    "AdaptiveRAGRouterPrediction",
    "AdaptiveRAGSelectorReplay",
    "AdaptiveRAGStrategy",
]
