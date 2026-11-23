"""Compatibility exports for canonical query-reformulation algorithms.

New code should import from :mod:`agentinputlab.algorithms.reformulate`.
"""

from agentinputlab.algorithms.reformulate import (
    OfficialQuery2DocRow,
    Query2DocCache,
    Query2DocOfficialCached,
    QueryGymLiveBridge,
    QueryGymMode,
    QueryGymRawResult,
    QueryGymRequest,
    QueryGymRuntime,
    compose_sparse_expansion,
    select_demonstration_ids,
)

__all__ = [
    "OfficialQuery2DocRow",
    "Query2DocCache",
    "Query2DocOfficialCached",
    "QueryGymLiveBridge",
    "QueryGymMode",
    "QueryGymRawResult",
    "QueryGymRequest",
    "QueryGymRuntime",
    "compose_sparse_expansion",
    "select_demonstration_ids",
]
