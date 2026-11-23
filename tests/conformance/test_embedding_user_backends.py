"""Conformance tests for embedding and replay-user backend seams."""

from __future__ import annotations

import math
import unittest

from agentinputlab.backends.embedding import (
    CallableEmbeddingBackend,
    DeterministicHashEmbeddingBackend,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ReplayEmbeddingBackend,
)
from agentinputlab.backends.user import (
    CallableUserResponder,
    ReplayTurn,
    ReplayUserResponder,
    UserRequest,
    UserResponse,
)
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash


class EmbeddingBackendConformanceTests(unittest.TestCase):
    def test_embedding_contracts_detach_metadata_and_keep_hashes_stable(self) -> None:
        metadata = {"audit": {"labels": ["initial"]}}
        request = EmbeddingRequest(texts=("alpha",), metadata=metadata)
        response = EmbeddingResponse(vectors=((1.0, 2.0),), metadata=metadata)
        request_hash = request.request_hash
        response_hash = content_hash(response)

        metadata["audit"]["labels"].append("late")
        self.assertEqual(request.request_hash, request_hash)
        self.assertEqual(content_hash(response), response_hash)
        self.assertEqual(request.metadata["audit"]["labels"], ("initial",))
        self.assertEqual(response.metadata["audit"]["labels"], ("initial",))

    def test_embedding_usage_rejects_invalid_numeric_values(self) -> None:
        invalid = (
            {"input_count": True},
            {"input_count": 1.5},
            {"input_count": -1},
            {"input_tokens": True},
            {"input_tokens": 1.5},
            {"input_tokens": -1},
            {"cost_usd": True},
            {"cost_usd": float("nan")},
            {"cost_usd": float("inf")},
            {"cost_usd": 10**1000},
            {"cost_usd": -0.01},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                EmbeddingUsage(**values)

    def test_hash_embedding_is_deterministic_and_normalizable(self) -> None:
        backend = DeterministicHashEmbeddingBackend(dimensions=12)
        request = EmbeddingRequest(texts=("alpha", "beta"), normalize=True)
        first = backend.embed(request)
        second = backend.embed(request)
        self.assertEqual(first, second)
        self.assertEqual(len(first.vectors), 2)
        self.assertTrue(
            all(
                math.isclose(
                    math.sqrt(sum(value * value for value in vector)),
                    1.0,
                    rel_tol=1e-12,
                )
                for vector in first.vectors
            )
        )
        self.assertEqual(backend.fingerprint(), backend.fingerprint())

    def test_callable_backend_is_lazy_and_validates_count(self) -> None:
        calls: list[EmbeddingRequest] = []

        def client(request: EmbeddingRequest) -> list[list[float]]:
            calls.append(request)
            return [[1.0, 2.0] for _ in request.texts]

        backend = CallableEmbeddingBackend(
            client=client,
            model_id="fixture",
            revision="abc123",
            dimensions=2,
            identity={"endpoint": "offline-fixture"},
        )
        first_fingerprint = backend.fingerprint()
        self.assertEqual(calls, [])
        self.assertEqual(first_fingerprint, backend.fingerprint())
        response = backend.embed(EmbeddingRequest(texts=("one", "two")))
        self.assertEqual(len(response.vectors), 2)
        self.assertEqual(len(calls), 1)

        broken = CallableEmbeddingBackend(
            client=lambda request: [[1.0, 2.0]],
            model_id="broken",
            revision="fixture-v1",
            dimensions=2,
            identity={"implementation": "tests.broken_embedding.v1"},
        )
        with self.assertRaisesRegex(ValidationError, "count does not match"):
            broken.embed(EmbeddingRequest(texts=("one", "two")))

    def test_replay_embedding_is_order_stable_and_fails_closed(self) -> None:
        left = ReplayEmbeddingBackend(vectors={"b": [0.0, 1.0], "a": [1.0, 0.0]})
        right = ReplayEmbeddingBackend(vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
        self.assertEqual(left.fingerprint(), right.fingerprint())
        response = left.embed(EmbeddingRequest(texts=("b", "a")))
        self.assertEqual(response.vectors, ((0.0, 1.0), (1.0, 0.0)))
        with self.assertRaisesRegex(ValidationError, "missing"):
            left.embed(EmbeddingRequest(texts=("unknown",)))


class UserBackendConformanceTests(unittest.TestCase):
    def test_replay_user_is_prompt_bound_and_repeatable(self) -> None:
        request = UserRequest(
            request_id="request-1",
            case_id="case-1",
            turn_index=0,
            prompt="Which account?",
            state_hash="state-a",
        )
        responder = ReplayUserResponder(
            turns=[
                ReplayTurn(
                    request_id="request-1",
                    case_id="case-1",
                    turn_index=0,
                    prompt_hash=request.prompt_hash,
                    response="The savings account.",
                )
            ]
        )
        first = responder.respond(request)
        second = responder.respond(request)
        self.assertEqual(first, second)
        self.assertEqual(first.text, "The savings account.")
        self.assertEqual(responder.fingerprint(), responder.fingerprint())

        changed = UserRequest(
            request_id="request-1",
            case_id="case-1",
            turn_index=0,
            prompt="Which card?",
            state_hash="state-a",
        )
        with self.assertRaisesRegex(ValidationError, "prompt hash"):
            responder.respond(changed)

    def test_replay_user_fingerprint_does_not_depend_on_input_order(self) -> None:
        first = ReplayTurn(case_id="b", turn_index=0, response="B")
        second = ReplayTurn(case_id="a", turn_index=0, response="A")
        left = ReplayUserResponder(turns=[first, second])
        right = ReplayUserResponder(turns=[second, first])
        self.assertEqual(left.fingerprint(), right.fingerprint())

    def test_replay_user_fingerprint_is_stable_after_caller_metadata_mutation(self) -> None:
        metadata = {"fixture": {"tags": ["stable"]}}
        responder = ReplayUserResponder(
            turns=[ReplayTurn(case_id="a", turn_index=0, response="A", metadata=metadata)]
        )
        fingerprint = responder.fingerprint()
        metadata["fixture"]["tags"].append("late")
        self.assertEqual(responder.fingerprint(), fingerprint)

    def test_callable_user_responder_is_lazy(self) -> None:
        calls: list[UserRequest] = []

        def client(request: UserRequest) -> str:
            calls.append(request)
            return "confirmed"

        responder = CallableUserResponder(
            client=client,
            responder_id="fixture",
            revision="fixture-v1",
            identity={"mode": "offline"},
        )
        self.assertEqual(responder.fingerprint(), responder.fingerprint())
        self.assertEqual(calls, [])
        response = responder.respond(
            UserRequest(
                request_id="r",
                case_id="c",
                turn_index=1,
                prompt="Confirm?",
            )
        )
        self.assertEqual(response.text, "confirmed")
        self.assertEqual(len(calls), 1)

        mismatched = CallableUserResponder(
            client=lambda request: UserResponse(request_id="other", text="wrong"),
            revision="fixture-v1",
            identity={"implementation": "tests.mismatched_user.v1"},
        )
        with self.assertRaisesRegex(ValidationError, "does not match"):
            mismatched.respond(
                UserRequest(
                    request_id="expected",
                    case_id="c",
                    turn_index=0,
                    prompt="Confirm?",
                )
            )


if __name__ == "__main__":
    unittest.main()
