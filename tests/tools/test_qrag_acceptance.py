from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def test_native_selector_compiler_cache_is_scoped_to_its_own_output(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        monkeypatch.setenv(key, "unrelated-shared-cache")
    spec = importlib.util.spec_from_file_location(
        "qrag_operator_test", tools / "run_qrag_acceptance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.configure_runtime_caches(tmp_path)
    assert len(values) == 3
    assert all(Path(value).is_relative_to(tmp_path) for value in values.values())
    assert all(os.environ[key] == value for key, value in values.items())
