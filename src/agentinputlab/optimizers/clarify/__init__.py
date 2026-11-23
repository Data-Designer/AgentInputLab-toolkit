"""Compatibility imports for clarification algorithms.

New code should import from :mod:`agentinputlab.algorithms.clarify`.
"""

from agentinputlab.algorithms.clarify import (
    UDCSDecision,
    UDCSPromptNativePolicy,
    parse_udcs_response,
)

__all__ = ["UDCSDecision", "UDCSPromptNativePolicy", "parse_udcs_response"]
