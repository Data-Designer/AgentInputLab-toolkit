import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("distribution_scan", ROOT / "tools/check_distribution.py")
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


def test_distribution_rejects_research_and_private_files():
    assert scanner.check_member("release/results/run.json", b"{}")
    assert scanner.check_member("release/weights/model.safetensors", b"weights")
    assert scanner.check_member("../escape.py", b"code")
    assert scanner.check_member("release/key.txt", b"-----BEGIN " + b"PRIVATE KEY-----")


def test_distribution_keeps_code_and_synthetic_fixtures():
    assert not scanner.check_member("src/agentinputlab/runner.py", b"# reusable code")
    assert not scanner.check_member("src/agentinputlab/artifacts/fixtures/example.json", b'{"synthetic":true}')
    assert not scanner.check_member("tools/check_distribution.py", (ROOT / "tools/check_distribution.py").read_bytes())
