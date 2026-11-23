from importlib import resources

from workers.official_arenas.retail_capture import _source_pins
from workers.official_arenas.capture_records import _records


def test_retail_capture_pins_only_shipped_modules():
    pins = _source_pins()
    assert len(pins) == 7
    assert "workers/official_arenas/capture_records.py" in pins
    assert not any(name.startswith("scripts/") for name in pins)
    assert all(len(value) == 64 for value in pins.values())
    assert callable(_records)
    assert resources.files("workers.llmcompiler").joinpath("compatibility_manifest.json").is_file()
