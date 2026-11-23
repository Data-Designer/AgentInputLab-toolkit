"""Public exception hierarchy.

The library uses typed errors instead of silent fallbacks.  A benchmark run must
be able to distinguish an unavailable upstream asset from an algorithm deciding
to return ``NO_OP``.
"""


class AgentInputLabError(Exception):
    """Base class for all expected library errors."""


class ValidationError(AgentInputLabError, ValueError):
    """A state, patch, manifest, or configuration violates its schema."""


class ArmExecutionError(ValidationError):
    """An expected execution failure invalidates one experimental arm only.

    This marker remains a :class:`ValidationError` for API compatibility, but
    paired-study orchestration can distinguish it from invalid experiment
    configuration without inspecting exception messages or traceback text.
    """


class AlgorithmOutputError(ArmExecutionError):
    """A model, embedder, retriever, or algorithm emitted an invalid result."""


class StateConflictError(AgentInputLabError):
    """A patch was applied to a state other than the one it was proposed for."""


class CapabilityError(AgentInputLabError):
    """A backend or connector lacks a capability required by a profile."""


class AlgorithmUnavailableError(AgentInputLabError):
    """A registered profile cannot run in the current environment."""


class ExternalDependencyError(AlgorithmUnavailableError):
    """An optional upstream checkout, package, model, or service is unavailable."""


class ArtifactError(AgentInputLabError):
    """An artifact is absent, corrupt, unsafe, or has the wrong lineage."""


class RegistryError(AgentInputLabError):
    """The algorithm registry is inconsistent or cannot be resolved."""


class BalanceError(AgentInputLabError):
    """A claimed controlled comparison differs on an undeclared dependency."""


class BudgetExceededError(AgentInputLabError):
    """A profile attempted to exceed a declared execution budget."""


class IntentViolationError(AgentInputLabError):
    """An intervention violated a runner-enforced observable intent contract."""
