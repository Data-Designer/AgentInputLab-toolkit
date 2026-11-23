"""Canonical runtime algorithm adapters.

Adapters in this package either implement a documented clean-room formula or
delegate to a pinned upstream runtime.  They never treat an unavailable
dependency as an algorithmic ``NO_OP`` decision.
"""
