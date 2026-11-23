"""Compatibility imports for evidence algorithms.

New code should import from :mod:`agentinputlab.algorithms.acquire`.
"""

from agentinputlab.algorithms.acquire import (
    DenseEmbedder,
    HyDECleanRoom,
    HyDEConfig,
    IRCoTControlledWorker,
    IRCoTControls,
    VectorSearcher,
)

__all__ = [
    "DenseEmbedder",
    "HyDECleanRoom",
    "HyDEConfig",
    "IRCoTControlledWorker",
    "IRCoTControls",
    "VectorSearcher",
]
