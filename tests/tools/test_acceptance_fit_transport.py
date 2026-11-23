from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "fit_transport_test", tools / "acceptance_fit_transport.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "visible"}]
        return [1, 2, 3]


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return json.dumps(
            {"model": "fixture", "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
        ).encode()


class Opener:
    def __init__(self):
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request)
        return Response()


def payload():
    return {
        "model": "fixture",
        "messages": [{"role": "user", "content": "visible"}],
        "max_tokens": 512,
    }


def test_physical_retries_share_one_attempt_cap_and_charge_actual_usage(module, tmp_path):
    relay = module.BoundedFitRelay(
        model={"id": "fixture", "base_url": "http://127.0.0.1:9876/v1"},
        root=tmp_path / "calls",
        tokenizer=Tokenizer(),
        max_attempts=1,
    )
    opener = Opener()
    relay.opener = opener
    with relay:
        assert relay.forward(payload())[0] == 200
        assert relay.forward(payload())[0] == 400
        assert len(opener.calls) == 1
        assert not opener.calls[0].has_header("Authorization")
        assert json.loads(opener.calls[0].data)["seed"] == 17
        assert relay.summary()["charged_tokens"] == 6
        assert relay.summary()["physical_provider_attempts"] == 1
        assert relay.summary()["rejected_before_model"] == 1


def test_aggregate_reservation_stops_before_network(module, tmp_path):
    relay = module.BoundedFitRelay(
        model={"id": "fixture", "base_url": "http://127.0.0.1:9876/v1"},
        root=tmp_path / "calls",
        tokenizer=Tokenizer(),
        max_tokens=514,
    )
    relay.opener = Opener()
    with relay:
        assert relay.forward(payload())[0] == 400
        assert relay.opener.calls == []


def test_native_completion_cap_is_preserved_and_uses_same_budget(module, tmp_path):
    relay = module.BoundedFitRelay(
        model={"id": "fixture", "base_url": "http://127.0.0.1:9876/v1"},
        root=tmp_path / "calls",
        tokenizer=Tokenizer(),
        max_attempts=1,
    )
    relay.opener = Opener()
    request = payload()
    request["max_completion_tokens"] = request.pop("max_tokens")
    with relay:
        assert relay.forward(request)[0] == 200
        wire = json.loads(relay.opener.calls[0].data)
        assert wire["max_completion_tokens"] == 512 and "max_tokens" not in wire
        assert relay.summary()["physical_provider_attempts"] == 1
        assert relay.summary()["charged_tokens"] == 6
        assert relay.forward(request)[0] == 400


@pytest.mark.parametrize("caps", [{}, {"max_tokens": 512, "max_completion_tokens": 512},
    {"max_completion_tokens": 513}, {"max_completion_tokens": True},
    {"max_completion_tokens": 0}, {"max_completion_tokens": "512"}])
def test_missing_ambiguous_or_invalid_cap_never_reaches_provider(module, tmp_path, caps):
    relay = module.BoundedFitRelay(
        model={"id": "fixture", "base_url": "http://127.0.0.1:9876/v1"},
        root=tmp_path / "calls", tokenizer=Tokenizer(),
    )
    relay.opener = Opener()
    request = payload()
    request.pop("max_tokens")
    request.update(caps)
    with relay:
        with pytest.raises(ValueError):
            relay.forward(request)
        assert relay.opener.calls == []
        assert relay.summary()["physical_provider_attempts"] == 0


def test_server_error_retains_unknown_consumption_reservation(module, tmp_path):
    relay = module.BoundedFitRelay(
        model={"id": "fixture", "base_url": "http://127.0.0.1:9876/v1"},
        root=tmp_path / "calls",
        tokenizer=Tokenizer(),
    )

    class Failing:
        def open(self, request, timeout):
            raise TimeoutError("fixture only")

    relay.opener = Failing()
    with relay:
        assert relay.forward(payload())[0] == 502
        assert relay.summary()["unknown_usage_attempts"] == 1
        assert relay.summary()["charged_tokens"] == 515
        row = json.loads((tmp_path / "calls/0001/response.json").read_text())
        assert row["unknown_usage_reservation_retained"] is True


def test_untrusted_endpoint_and_budget_override_are_rejected(module, tmp_path):
    for url in ("https://remote.invalid/v1", "http://key@127.0.0.1/v1"):
        with pytest.raises(ValueError, match="local unauthenticated"):
            module.BoundedFitRelay(
                model={"id": "fixture", "base_url": url}, root=tmp_path, tokenizer=Tokenizer()
            )
    with pytest.raises(ValueError, match="cannot override"):
        module.BoundedFitRelay(
            model={
                "id": "fixture",
                "base_url": "http://127.0.0.1:9876/v1",
                "extra_body": {"max_tokens": 1024},
            },
            root=tmp_path,
            tokenizer=Tokenizer(),
        )


def test_wrong_actual_serving_model_stops_future_attempts_but_retains_cost(module, tmp_path):
    relay = module.BoundedFitRelay(
        model={"id": "expected", "base_url": "http://127.0.0.1:9876/v1"},
        root=tmp_path / "calls",
        tokenizer=Tokenizer(),
    )
    opener = Opener()  # Returns the deliberately different fixture model.
    relay.opener = opener
    request = {**payload(), "model": "expected"}
    with relay:
        assert relay.forward(request)[0] == 502
        assert relay.forward(request)[0] == 400
        assert len(opener.calls) == 1
        assert relay.summary()["provider_identity_mismatch"] is True
        assert relay.summary()["charged_tokens"] == 6
