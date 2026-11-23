from __future__ import annotations

import threading
import tempfile
from pathlib import Path
import pytest

from agentinputlab.algorithms.acquire.r2ag import (
    R2AGRuntimeRequest,
    UnixSocketR2AGCommonSupportRuntime,
    serve_unix_socket_r2ag_runtime,
)
from agentinputlab.backends.retrieval import Document, RetrievalResult


class _FixtureRuntime:
    def run(self, request: R2AGRuntimeRequest):
        return {
            "status": "ok",
            "query": request.query,
            "doc_ids": [item.document.doc_id for item in request.documents],
            "seed": request.seed,
        }

    def fingerprint(self):
        return {"runtime": "fixture-r2ag", "reader": {"sha256": "a" * 64}}


@pytest.fixture
def short_socket_directory():
    # macOS's default temporary root can exceed AF_UNIX's path limit.
    with tempfile.TemporaryDirectory(prefix="ail-", dir="/tmp") as directory:
        yield Path(directory)


def test_unix_socket_runtime_round_trip(short_socket_directory) -> None:
    tmp_path = short_socket_directory
    socket_path = tmp_path / "r2ag.sock"
    ready = threading.Event()
    thread = threading.Thread(
        target=serve_unix_socket_r2ag_runtime,
        args=(socket_path, _FixtureRuntime()),
        kwargs={"max_requests": 2, "ready_callback": ready.set},
        daemon=True,
    )
    thread.start()
    # bind() creates the path before listen(); path existence is not readiness.
    assert ready.wait(timeout=2), "fixture server did not finish listen()"
    client = UnixSocketR2AGCommonSupportRuntime(socket_path, timeout_seconds=2)
    request = R2AGRuntimeRequest(
        query="Who wrote Hamlet?",
        documents=(
            RetrievalResult(
                document=Document(
                    doc_id="doc-1",
                    text="Hamlet was written by William Shakespeare.",
                    title="Hamlet",
                    metadata={"source": "fixture"},
                ),
                score=1.5,
                rank=1,
                features={"bm25": 1.5},
            ),
        ),
        max_new_tokens=8,
        seed=17,
    )
    assert client.fingerprint() == {
        "runtime": "fixture-r2ag",
        "reader": {"sha256": "a" * 64},
    }
    assert client.run(request) == {
        "status": "ok",
        "query": "Who wrote Hamlet?",
        "doc_ids": ("doc-1",),
        "seed": 17,
    }
    thread.join(timeout=2)
    assert not thread.is_alive()
