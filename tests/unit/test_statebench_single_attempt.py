"""Instance-only native retry binding; synthetic failures are not model results."""
from types import SimpleNamespace

import pytest

from workers.official_arenas.protocol import OfficialArenaError
from workers.official_arenas.providers import statebench


def retrying_method(original):
    def wrap(stop):
        def call(self):
            for attempt in range(1, 6):
                try:
                    return original(self)
                except RuntimeError:
                    if stop(SimpleNamespace(attempt_number=attempt)):
                        raise
        call.retry_with = lambda **settings: wrap(settings['stop'])
        return call
    return wrap(lambda state: state.attempt_number >= 5)


class Client:
    def __init__(self):
        self.calls = 0

    @retrying_method
    def complete_chat(self):
        self.calls += 1
        raise RuntimeError('synthetic transport failure')

    complete_with_tools = complete_chat
    complete_json_response = complete_chat

    @retrying_method
    def complete_json(self):
        return self.complete_json_response()


@pytest.mark.parametrize('method', ['complete_chat', 'complete_with_tools', 'complete_json_response', 'complete_json'])
def test_single_attempt_including_nested_json_never_retries_or_changes_other_instances(method):
    configured, legacy = Client(), Client()
    statebench._bind_single_attempt(configured)
    with pytest.raises(RuntimeError, match='synthetic transport'):
        getattr(configured, method)()
    assert configured.calls == 1
    with pytest.raises(RuntimeError):
        legacy.complete_chat()
    assert legacy.calls == 5


def test_incompatible_native_retry_api_is_not_silently_accepted():
    client = Client()
    client.complete_json = lambda: None
    with pytest.raises(OfficialArenaError, match='retry binding'):
        statebench._bind_single_attempt(client)
    assert client.calls == 0
