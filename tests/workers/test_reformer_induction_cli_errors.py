"""CLI failure contract only; model acceptance uses source-bound saved responses."""
import json

import pytest

from agentinputlab.contracts import ValidationError
from workers.reformer import induction
from workers.reformer.driver import DriverError


@pytest.mark.parametrize('error,code', [
    (DriverError('PATTERN_ARTIFACT_INVALID', 'private provider text'), 'PATTERN_ARTIFACT_INVALID'),
    (DriverError('BACKEND_OUTPUT_INVALID', 'private provider text', consumed=True), 'BACKEND_OUTPUT_INVALID'),
    (DriverError('BACKEND_CALL_FAILED', 'private provider text', consumed=True), 'BACKEND_CALL_FAILED'),
    (DriverError('private provider text', 'private provider text'), 'FIT_EXECUTION_FAILED'),
    (ValidationError('private provider text'), 'INVALID_FIT_CONFIGURATION'),
    (FileExistsError('private provider text'), 'OUTPUT_ALREADY_EXISTS'),
    (FileNotFoundError('private provider text'), 'FIT_FILE_UNAVAILABLE'),
    (PermissionError('private provider text'), 'FIT_FILE_UNAVAILABLE'),
])
def test_expected_failure_is_stderr_json_without_retry_or_exception_chain(monkeypatch, capsys, error, code):
    calls = []

    def fail(argv):
        calls.append(argv)
        try:
            raise RuntimeError('private nested credential')
        except RuntimeError as cause:
            raise error from cause

    monkeypatch.setattr(induction, '_main', fail)
    assert induction.main(['--supplied']) == 2
    assert calls == [['--supplied']]
    output = capsys.readouterr()
    assert output.out == ''
    assert 'private' not in output.err and 'Traceback' not in output.err
    result = json.loads(output.err)
    assert result['error_code'] == code and result['status'] == 'failed'
    assert result['automatic_retry'] is False
    assert 'usage' not in result  # No fabricated zero usage for unknown failures.


def test_unexpected_bug_is_not_silently_reclassified(monkeypatch):
    def fail(argv):
        raise AssertionError('implementation bug')
    monkeypatch.setattr(induction, '_main', fail)
    with pytest.raises(AssertionError, match='implementation bug'):
        induction.main([])


def test_success_output_and_exit_code_unchanged(monkeypatch, capsys):
    def success(argv):
        print('{"status":"candidate_generated"}')
        return 0
    monkeypatch.setattr(induction, '_main', success)
    assert induction.main([]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)['status'] == 'candidate_generated'
    assert not captured.err


def test_actual_missing_pair_file_is_clean_before_backend_or_fit(tmp_path, capsys):
    assert induction.main([
        '--checkout', str(tmp_path / 'checkout'), '--pairs', str(tmp_path / 'absent'),
        '--backend-config', str(tmp_path / 'backend'), '--output', str(tmp_path / 'out'),
    ]) == 2
    assert not (tmp_path / 'out').exists()
    captured = capsys.readouterr()
    assert captured.out == ''
    assert json.loads(captured.err)['error_code'] == 'FIT_FILE_UNAVAILABLE'
