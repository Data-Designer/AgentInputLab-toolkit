"""User-facing boundary tests; all calls are isolated local protocol fixtures."""

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from agentinputlab.api import ToolkitService, make_server

TOKEN = "consumer-boundary-test-not-a-secret"


@pytest.mark.parametrize("body", [
    b'{"release_id":"baseline.no_op","state":{"query":"first","query":"second"}}',
    b'{"release_id":"baseline.no_op","state":{"query":"x"},"seed":17,"seed":18}',
    b'{"release_id":"baseline.no_op","state":{"query":"x","metadata":{"x":1e999}}}',
])
def test_ambiguous_or_nonfinite_json_never_dispatches(body):
    calls = []
    service = ToolkitService()
    original = service.execute

    def observed(value):
        calls.append(value)
        return original(value)

    service.execute = observed
    server = make_server(service, token=TOKEN, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/interventions", data=body,
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with pytest.raises(urllib.error.HTTPError) as caught:
            opener.open(request, timeout=5)
        with caught.value as response:
            assert response.status == 400
            assert json.load(response)["error"]["code"] == "invalid_request"
        assert calls == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("token", ["x" * 24 + "\n", "x" * 24 + "\t", " " * 24, "x" * 24 + "\x7f"])
def test_unusable_authorization_token_is_rejected_before_binding(token):
    with pytest.raises(ValueError, match="token"):
        make_server(ToolkitService(), token=token, port=0)


@pytest.mark.parametrize("extra_headers,length_delta", [
    (["Content-Length: 0"], 0),
    (["Transfer-Encoding:"], 0),
    ([], 5),
])
def test_ambiguous_or_truncated_http_body_never_dispatches(extra_headers, length_delta):
    calls = []
    service = ToolkitService()
    service.execute = lambda value: calls.append(value) or {}
    server = make_server(service, token=TOKEN, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = b'{"release_id":"baseline.no_op","state":{"query":"x"}}'
    try:
        headers = [
            "POST /v1/interventions HTTP/1.1", "Host: 127.0.0.1",
            f"Authorization: Bearer {TOKEN}", "Content-Type: application/json",
            f"Content-Length: {len(body) + length_delta}", *extra_headers,
        ]
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as sock:
            sock.sendall("\r\n".join(headers).encode() + b"\r\n\r\n" + body)
            sock.shutdown(socket.SHUT_WR)
            with sock.makefile("rb") as reply:
                status = reply.readline()
                assert b" 400 " in status
        assert calls == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
