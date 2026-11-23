from __future__ import annotations

from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.experiments.secrets import (
    environment_secret_values,
    redact_secret_text,
    require_secret_values_absent,
)


def test_exact_environment_values_are_redacted_without_secret_shape() -> None:
    environment = {
        "DEMO_SECRET": "correcthorsebatterystaple",
        "UNPASSED_SECRET": "must-remain-out-of-scope",
    }
    values = environment_secret_values(environment, ("DEMO_SECRET",))
    rendered = redact_secret_text(
        "failure correcthorsebatterystaple",
        secret_values=values,
    )

    assert rendered == "failure <redacted>"
    assert "must-remain-out-of-scope" not in values

    short_values = environment_secret_values(
        {"DEMO_SECRET": "abc", "WORKER_COUNT": "1"},
        ("DEMO_SECRET", "WORKER_COUNT"),
    )
    assert short_values == ("abc",)
    assert (
        redact_secret_text(
            "failure abc",
            secret_values=short_values,
        )
        == "failure <redacted>"
    )


def test_nested_worker_output_cannot_copy_an_opaque_environment_value() -> None:
    secret = "opaqueValue987654321"

    with pytest.raises(ArtifactError, match="copied from the worker environment"):
        require_secret_values_absent(
            {"output": {"answer": f"prefix-{secret}-suffix"}},
            secret_values=(secret,),
            label="worker response",
        )
