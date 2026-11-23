import pytest

from workers.smartsearch.driver import DriverError, _completion_stop, _sha256_file


@pytest.mark.parametrize(
    "text,choice,expected",
    [
        (
            "<search>query",
            {"finish_reason": "stop", "stop_reason": "</search>"},
            ("<search>query", "</search>"),
        ),
        (
            "<answer>answer</answer>",
            {"finish_reason": "stop", "stop_reason": "</answer>"},
            ("<answer>answer", "</answer>"),
        ),
        ("<search>query</search>", {"finish_reason": "stop"}, ("<search>query", "</search>")),
        ("plain text", {"finish_reason": "stop", "stop_reason": 151645}, ("plain text", "")),
        (
            "<answer>incomplete",
            {"finish_reason": "stop", "stop_reason": 151645},
            ("<answer>incomplete", ""),
        ),
        ("<answer>incomplete", {"finish_reason": "length"}, ("<answer>incomplete", "")),
    ],
)
def test_completion_stop_preserves_exact_provider_termination(text, choice, expected):
    assert _completion_stop(text, choice) == expected


def test_conflicting_stops_fail_without_inventing_an_answer():
    with pytest.raises(DriverError, match="conflicting"):
        _completion_stop(
            "<search>q</search>", {"finish_reason": "stop", "stop_reason": "</answer>"}
        )


def test_snapshot_hash_is_streamed(tmp_path):
    import hashlib

    value = b"large-weight-fixture" * 100000
    path = tmp_path / "model.safetensors"
    path.write_bytes(value)
    assert _sha256_file(path) == hashlib.sha256(value).hexdigest()
