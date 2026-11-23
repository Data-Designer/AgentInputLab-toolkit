from pathlib import Path
from types import SimpleNamespace

import pytest
from acceptance_method_bindings import REFORMER_RELEASES, bind_external_rewriter
from run_reformer_profiles_acceptance import CompletionTape, NativeWorker

from workers.reformer.driver import BackendConfig, CompletionResult, DriverError


class Backend:
    def __init__(self, error=False):
        self.calls, self.error = [], error

    def fingerprint(self):
        return {"model": "fixture"}

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise DriverError("BACKEND_OUTPUT_INVALID", "empty final", consumed=True,
                provider_usage={"input_tokens": 4, "output_tokens": 512},
                diagnostics={"finish_reason": "length"})
        return CompletionResult("rewrite", 4, 3, "stop")


REQUEST = {"messages": [{"role": "user", "content": "query"}], "seed": 17,
           "temperature": 0.0, "max_output_tokens": 512}


def test_native_completion_is_physically_called_once_then_exactly_replayed():
    backend = Backend()
    source = CompletionTape(backend)
    assert source.complete(**REQUEST).text == "rewrite"
    replay = CompletionTape(backend, saved=source.calls)
    assert replay.complete(**REQUEST) == CompletionResult("rewrite", 4, 3, "stop")
    replay.require_complete_replay()
    assert replay.calls == [] and replay.index == 1 and len(backend.calls) == 1


@pytest.mark.parametrize("change", [
    {"seed": 18}, {"max_output_tokens": 1024}, {"temperature": 0.5},
    {"messages": [{"role": "user", "content": "another query"}]},
])
def test_replay_payload_mismatch_never_uses_network(change):
    backend = Backend()
    source = CompletionTape(backend)
    source.complete(**REQUEST)
    replay = CompletionTape(backend, saved=source.calls)
    with pytest.raises(DriverError, match="request changed"):
        replay.complete(**{**REQUEST, **change})
    with pytest.raises(RuntimeError, match="exactly"):
        replay.require_complete_replay()
    assert len(backend.calls) == 1


def test_replay_failure_preserves_consumption_and_does_not_retry():
    backend = Backend(error=True)
    source = CompletionTape(backend)
    with pytest.raises(DriverError):
        source.complete(**REQUEST)
    replay = CompletionTape(backend, saved=source.calls)
    with pytest.raises(DriverError) as info:
        replay.complete(**REQUEST)
    assert info.value.provider_usage == {"input_tokens": 4, "output_tokens": 512}
    assert info.value.diagnostics == {"finish_reason": "length"}
    assert info.value.consumed and len(backend.calls) == 1
    replay.require_complete_replay()
    with pytest.raises(DriverError, match="exhausted"):
        replay.complete(**REQUEST)
    assert len(backend.calls) == 1


def test_unconsumed_replay_is_not_complete():
    source = CompletionTape(Backend())
    source.complete(**REQUEST)
    replay = CompletionTape(Backend(), saved=source.calls)
    with pytest.raises(RuntimeError, match="exactly"):
        replay.require_complete_replay()


@pytest.mark.parametrize("release", REFORMER_RELEASES)
def test_all_four_profiles_bind_the_reformer_worker_and_exact_manual_controls(release):
    manual = release in REFORMER_RELEASES[2:]
    method, worker = bind_external_rewriter(release, upstream=Path("/unused"),
        backend=Backend(), model={"id": "fixture", "base_url": "http://localhost:1234/v1"},
        dataset_revision="test", benchmark="fixture", manual_pattern_id="fixed" if manual else "")
    assert worker.worker_id == "toolkit-source-mounted-reformer"
    assert method.release_id == release
    assert method.pattern_mode == ("manual" if manual else "automatic")
    assert method.manual_pattern_id == ("fixed" if manual else "")
    assert method.controls.max_model_calls == (1 if manual else 2)
    assert method.controls.max_tokens == (1 if manual else 2) * (28000 + 512)


@pytest.mark.parametrize("release, pattern", [
    ("rewrite.unknown.profile", ""), (REFORMER_RELEASES[2], ""),
    (REFORMER_RELEASES[0], "fixed"), ("rewrite.infocqr.common_support_rw_zsl", "fixed"),
])
def test_unknown_or_mismatched_manual_profiles_fail_before_calls(release, pattern):
    backend = Backend()
    with pytest.raises(ValueError):
        bind_external_rewriter(release, upstream=Path("/unused"), backend=backend,
            model={}, dataset_revision="test", benchmark="fixture", manual_pattern_id=pattern)
    assert backend.calls == []


def test_runtime_copy_executes_each_new_profile_instead_of_forging_response_identity():
    calls = []

    class Runtime:
        def run(self, payload):
            calls.append(payload)
            return {"status": "ok", "request_id": payload["request_id"],
                    "native_result": self.backend.complete(**REQUEST).text}

    template = Runtime()
    template.backend = "unchanged"
    tape = CompletionTape(Backend())
    worker = NativeWorker(template, SimpleNamespace(), tape, {"source": "pinned"})
    assert worker.call({"request_id": "r-only"})["native_result"] == "rewrite"
    replay = NativeWorker(template, SimpleNamespace(), CompletionTape(tape.backend, saved=tape.calls), {})
    assert replay.call({"request_id": "concat"})["request_id"] == "concat"
    assert calls == [{"request_id": "r-only"}, {"request_id": "concat"}]
    assert template.backend == "unchanged"
    assert len(tape.backend.calls) == 1


def test_documented_backend_config_has_all_strict_startup_fields():
    readme = Path(__file__).resolve().parents[2] / "workers/reformer/README.md"
    raw = readme.read_text().split("```json\n", 1)[1].split("```", 1)[0]
    config = BackendConfig.from_json(raw)
    assert config.max_input_tokens_per_call == 28000
    assert config.max_output_tokens == 500
