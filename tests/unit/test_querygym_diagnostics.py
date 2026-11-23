"""Isolated upstream diagnostics must leave machine-readable stdout intact."""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from agentinputlab.algorithms.reformulate.query2doc.querygym_runtime import (
    _diagnostic_print,
    _load_source_module,
)


def test_upstream_fallback_diagnostic_preserves_json_and_return(tmp_path, capsys):
    source = tmp_path / "native.py"
    source.write_text(
        'print("native loaded")\n'
        'def fallback(query):\n'
        '    print("empty response", file=None)\n'
        '    return {"reformulated": query, "fallback": True}\n'
    )
    name = "_test_querygym_diagnostics"
    stdout = sys.stdout
    try:
        module = _load_source_module(name, source)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(module.fallback, ["a", "b"]))
        assert sys.stdout is stdout
        print(json.dumps(results))
        captured = capsys.readouterr()
        assert json.loads(captured.out) == [
            {"reformulated": "a", "fallback": True},
            {"reformulated": "b", "fallback": True},
        ]
        assert captured.err.count("empty response") == 2
        assert "native loaded" in captured.err
    finally:
        sys.modules.pop(name, None)


def test_diagnostics_preserve_explicit_stream_and_print_options(capsys):
    destination = io.StringIO()
    _diagnostic_print("a", "b", sep="/", end="!", file=destination, flush=True)
    assert destination.getvalue() == "a/b!"
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
