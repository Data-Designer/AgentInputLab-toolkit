"""Constructor wiring tests; no model generations or native-quality claims."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from workers.official_arenas.protocol import OfficialArenaError, native_timeout_options
from workers.official_arenas.providers import statebench, tooltalk


@pytest.mark.parametrize("value", [True, False, 0, -1, "30", None, float("inf"), float("nan")])
def test_timeout_must_be_explicit_finite_positive_number(value):
    with pytest.raises(OfficialArenaError, match="finite positive"):
        native_timeout_options({"agent_timeout_seconds": value}, "agent_timeout_seconds")


def test_absent_timeout_does_not_replace_sdk_defaults():
    assert native_timeout_options({}, "agent_timeout_seconds") == {}
    assert native_timeout_options({"agent_timeout_seconds": 0.25}, "agent_timeout_seconds") == {"timeout": 0.25}


@pytest.mark.parametrize("value", [None, 31.5])
def test_tooltalk_predictor_receives_timeout_without_enabling_retries(monkeypatch, value):
    clients = []
    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: clients.append(kwargs) or SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setattr(tooltalk, "_tool_schemas", lambda: [])
    options = {} if value is None else {"timeout_seconds": value}
    tooltalk._Predictor(model="fixture", base_url="http://localhost:1/v1", api_key="fixture",
        materialized={}, budget={"max_model_calls": 8, "max_tool_calls": 8, "max_output_tokens": 4096},
        optimizer_usage={}, seed=17, max_output_tokens_per_call=512, **options)
    assert clients[0]["timeout"] == (600.0 if value is None else value)
    assert clients[0]["max_retries"] == 0


@pytest.mark.parametrize("value", [None, 73.5])
def test_statebench_client_receives_timeout_without_changing_native_logic(monkeypatch, value):
    clients = []
    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: clients.append(kwargs) or SimpleNamespace()
    native = ModuleType("state_bench.client")
    native.LLMClient = type("LLMClient", (), {"complete_with_tools": lambda *args, **kwargs: None})
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setitem(sys.modules, "state_bench.client", native)
    monkeypatch.setenv("LOCAL_TEST_KEY", "fixture")
    options = {} if value is None else {"timeout": value}
    client = statebench._openai_compatible_client(model="fixture", base_url="http://localhost:1/v1",
                                                 api_key_var="LOCAL_TEST_KEY", **options)
    assert isinstance(client, native.LLMClient)
    assert clients[0]["max_retries"] == 0
    if value is None:
        assert "timeout" not in clients[0]
    else:
        assert clients[0]["timeout"] == value
