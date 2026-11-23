import pytest
from run_decomp_live_acceptance import NativeWire


def test_native_transport_preserves_status_and_unknown_consumption_without_secret_body(monkeypatch):
    class Failed(Exception):
        code = 500
    attempts=[]
    def fail(*args):
        attempts.append(args)
        raise Failed("opaque SECRET provider body")
    monkeypatch.setattr("run_decomp_live_acceptance.local_transport",fail)
    wire=NativeWire()
    with pytest.raises(Failed):
        wire.transport("http://127.0.0.1/v1/completions",{"Authorization":"secret"},{"model":"fixed"},10)
    assert len(attempts)==len(wire.calls)==1
    assert wire.calls[0]["http_status"]==500 and wire.calls[0]["usage_unknown"]
    assert "SECRET" not in str(wire.calls) and "Authorization" not in str(wire.calls)
