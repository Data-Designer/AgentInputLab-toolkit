from __future__ import annotations

from unittest import TestCase

from agentinputlab.contracts import AlgorithmOutputError, ExternalDependencyError
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest

_CASE = TestCase()


def _request() -> MethodWorkerRequest:
    return MethodWorkerRequest(
        method_id="fixture.method",
        release_id="fixture.method.v1",
        upstream_commit="a" * 40,
        state_hash="b" * 64,
        payload={},
    )


def test_invalid_backend_output_is_an_arm_local_algorithm_failure() -> None:
    with _CASE.assertRaisesRegex(AlgorithmOutputError, "BACKEND_OUTPUT_INVALID"):
        _request().raise_for_external_status(
            {"status": "error", "failure_code": "BACKEND_OUTPUT_INVALID"}
        )


def test_invalid_json_is_an_arm_local_algorithm_failure() -> None:
    with _CASE.assertRaisesRegex(AlgorithmOutputError, "INVALID_JSON"):
        _request().raise_for_external_status({"status": "error", "failure_code": "INVALID_JSON"})


def test_out_of_library_pattern_is_an_arm_local_algorithm_failure() -> None:
    with _CASE.assertRaisesRegex(AlgorithmOutputError, "PATTERN_SELECTION_OUTSIDE_LIBRARY"):
        _request().raise_for_external_status(
            {"status": "error", "failure_code": "PATTERN_SELECTION_OUTSIDE_LIBRARY"}
        )


def test_worker_infrastructure_failure_remains_external() -> None:
    with _CASE.assertRaisesRegex(ExternalDependencyError, "BACKEND_UNAVAILABLE"):
        _request().raise_for_external_status(
            {"status": "error", "failure_code": "BACKEND_UNAVAILABLE"}
        )
