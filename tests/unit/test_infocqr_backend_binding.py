import json
import pytest
from workers.infocqr.driver import BackendConfig, DriverError


def configuration(fingerprint):
    return json.dumps({'base_url':'http://127.0.0.1:9000/v1','api_key_env':'LOCAL_KEY',
        'model':'real-configured-model','fingerprint':fingerprint,'extra_body':{},
        'timeout_seconds':30,'max_input_tokens_per_call':28000,'max_output_tokens':512,
        'temperature':0,'input_cost_per_million':0,'output_cost_per_million':0})


@pytest.mark.parametrize('fingerprint',[
    {'backend':'openai_compatible','model_id':'real-configured-model'},
    {'model':'real-configured-model'},
    {'backend':{'model_id':'real-configured-model'}},
])
def test_accepts_actual_native_and_flat_toolkit_backend_fingerprints(fingerprint):
    assert BackendConfig.from_json(configuration(fingerprint)).model=='real-configured-model'


def test_mismatched_model_is_still_rejected():
    with pytest.raises(DriverError,match='exact model'):
        BackendConfig.from_json(configuration({'model_id':'another-model'}))
