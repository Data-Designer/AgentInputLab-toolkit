"""Pinned R2AG imports must not dirty or trust bytecode in the source tree."""
import os
import py_compile

from workers.r2ag.runtime import _load_source_module


def test_native_source_import_does_not_generate_untracked_bytecode(tmp_path):
    source = tmp_path / "native.py"
    source.write_text("VALUE = 1\n")
    first = _load_source_module(source, "r2ag_fixture_first")
    second = _load_source_module(source, "r2ag_fixture_second")
    assert first.VALUE == second.VALUE == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["native.py"]


def test_native_source_import_uses_verified_source_not_timestamp_valid_stale_cache(tmp_path):
    source = tmp_path / "native.py"
    source.write_text("VALUE = 1\n")
    original = source.stat()
    py_compile.compile(str(source), doraise=True)
    source.write_text("VALUE = 2\n")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    cached = {p: p.read_bytes() for p in (tmp_path / "__pycache__").iterdir()}
    assert _load_source_module(source, "r2ag_fixture_stale").VALUE == 2
    assert {p: p.read_bytes() for p in cached} == cached
