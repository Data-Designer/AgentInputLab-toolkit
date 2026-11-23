"""Fail closed on research files and unsafe paths in wheels/sdists/npm packs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

FORBIDDEN_PARTS = {"results", "datasets", "checkpoints", "weights", ".runtime", "backups",
                   "paper_campaigns", "paper_main_arenas", "node_modules", ".git"}
FORBIDDEN_SUFFIXES = {".csv", ".jsonl", ".log", ".pdf", ".pt", ".pth", ".safetensors", ".pem", ".key"}
PRIVATE_PATHS = (b"/Users/" + b"zhaochuang/", b"/nfs/" + b"usrhome2/", b"/Data2/" + b"bighome/")


def check_member(name: str, content: bytes) -> list[str]:
    p = PurePosixPath(name)
    errors = []
    if p.is_absolute() or ".." in p.parts:
        errors.append("unsafe archive path")
    if FORBIDDEN_PARTS.intersection(p.parts) or any(x.startswith("paper_kdd") for x in p.parts):
        errors.append("research/private directory")
    if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name.startswith(".env"):
        errors.append("forbidden artifact type")
    if any(marker in content for marker in PRIVATE_PATHS):
        errors.append("private execution path")
    if b"-----BEGIN " + b"PRIVATE KEY-----" in content or b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----" in content:
        errors.append("private key")
    return [f"{name}: {error}" for error in errors]


def scan(path: Path) -> dict:
    errors, count = [], 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    count += 1
                    errors.extend(check_member(item.filename, archive.read(item)))
    else:
        with tarfile.open(path) as archive:
            for item in archive:
                if item.issym() or item.islnk():
                    errors.append(f"{item.name}: archive links are not permitted")
                elif item.isfile():
                    count += 1
                    errors.extend(check_member(item.name, archive.extractfile(item).read()))
    return {"artifact": path.name, "files": count, "passed": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    reports = [scan(p) for p in parser.parse_args().artifacts]
    print(json.dumps(reports, indent=2))
    raise SystemExit(any(not r["passed"] for r in reports))


if __name__ == "__main__":
    main()
