"""An explicit provider capability enables existing schema-constrained helpers."""
import pytest

from agentinputlab import RunConfig
from agentinputlab.backends.llm import ModelCapability
from agentinputlab.contracts import ValidationError

BASE={"model_id":"caller-model", "base_url":"http://127.0.0.1:1/v1",
      "deployment_identity":{"revision":"fixture"}}


def test_default_capabilities_do_not_change():
    backend=RunConfig("baseline.no_op",backend=BASE)._backend()
    assert ModelCapability.JSON_OBJECT in backend.capabilities
    assert ModelCapability.JSON_SCHEMA not in backend.capabilities


def test_explicit_capabilities_are_validated_without_contacting_provider():
    backend=RunConfig("baseline.no_op",backend={**BASE,
        "capabilities":["chat","completion","json_schema","seed"]})._backend()
    assert backend.capabilities==frozenset({ModelCapability.CHAT,ModelCapability.COMPLETION,
        ModelCapability.JSON_SCHEMA,ModelCapability.SEED})


@pytest.mark.parametrize("value", [[],"json_schema",["unknown"],[True],["chat","chat"]])
def test_invalid_capabilities_fail_before_inference(value):
    with pytest.raises(ValidationError,match="capabilities"):
        RunConfig("baseline.no_op",backend={**BASE,"capabilities":value}).validate()
