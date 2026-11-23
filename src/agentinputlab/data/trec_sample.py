"""Bounded, source-verifiable TREC passage samples, never a full-corpus download."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

from agentinputlab.contracts import ArtifactError, ValidationError

from .downloader import DownloadPolicy, DownloadTransport, _validate_peer, validate_download_url
from .http import PinnedHTTPSDownloadTransport
from .storage import require_storage_capacity

_HOSTS = (
    "trec.nist.gov",
    "msmarco.z22.web.core.windows.net",
    "huggingface.co",
    "datasets-server.huggingface.co",
)
_BUNDLE_LIMIT = 16 * 1024 * 1024
_COMPONENT_LIMIT = 256 * 1024
_REPO = "BeIR/msmarco"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes) -> Any:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ArtifactError("TREC source returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactError("TREC source JSON must be an object")
    return value


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactError("TREC cache component must not be a symlink")
    if path.exists():
        if path.read_bytes() != raw:
            raise ArtifactError("TREC cache differs; original bytes were not overwritten")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def _dump(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def prepare_trec_dl_sample(
    root: str | Path,
    *,
    year: int,
    query_count: int = 2,
    query_offset: int = 0,
    background_count: int = 32,
    offline: bool = False,
    accept_license_review: bool = False,
    max_download_bytes: int = _BUNDLE_LIMIT,
    expected_corpus_revision: str | None = None,
    source_sample_root: str | Path | None = None,
    source_manifest_sha256: str | None = None,
    transport: DownloadTransport | None = None,
) -> dict[str, Any]:
    """Prepare or reuse a small official-source compatibility sample.

    Queries are selected lexicographically before any model output. ``query_offset``
    permits independent fitting groups without modifying a saved test sample.
    Candidate passages are explicitly gold-enriched: these are compatibility
    inputs, not full TREC performance estimates. All source bytes/revisions and
    original grades are retained; binary metrics continue to use grade >= 2.

    Missing components use a DNS-pinned HTTPS transport. ``offline=True`` never
    performs network IO. A supplied source sample must carry its pinned manifest;
    its verified query/qrel/row resources may be reused, never modified. Partial
    preparation resumes from verified components, without redownloading them.
    """
    if type(year) is not int or year not in (2019, 2020):
        raise ValidationError("TREC sample year must be 2019 or 2020")
    for name, value, low, high in (
        ("query_count", query_count, 1, 8),
        ("query_offset", query_offset, 0, 10000),
        ("background_count", background_count, 1, 32),
        ("max_download_bytes", max_download_bytes, 1, _BUNDLE_LIMIT),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValidationError(f"TREC {name} must be between {low} and {high}")
    if type(offline) is not bool or accept_license_review is not True:
        raise ValidationError("review TREC/MS MARCO data terms and set accept_license_review=True")
    if expected_corpus_revision is not None and (
        not isinstance(expected_corpus_revision, str)
        or len(expected_corpus_revision) != 40
        or any(c not in "0123456789abcdef" for c in expected_corpus_revision)
    ):
        raise ValidationError("expected_corpus_revision must be a full lowercase Hub commit SHA")
    if (source_sample_root is None) != (source_manifest_sha256 is None):
        raise ValidationError("a reusable source sample requires its manifest SHA-256")
    from agentinputlab.benchmarks.trec_dl_sample import TRECDLSampleAdapter

    reused: dict[str, bytes] = {}
    reused_revisions: dict[str, str | None] = {}
    if source_sample_root is not None:
        old = TRECDLSampleAdapter(
            source_sample_root, manifest_sha256=source_manifest_sha256, license_acknowledged=True
        )
        for item in old.manifest["resources"]:
            path = old.root / item["path"]
            if path.is_symlink() or path.stat().st_size != item["size_bytes"]:
                raise ArtifactError("reusable TREC source component changed")
            raw = path.read_bytes()
            if _sha(raw) != item["sha256"]:
                raise ArtifactError("reusable TREC source checksum changed")
            reused[item["url"]] = raw
            reused_revisions[item["url"]] = (
                item["source_revision"] if item["role"] == "corpus_rows" else None
            )
    destination = Path(root).expanduser()
    if destination.is_symlink():
        raise ArtifactError("TREC sample destination must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    for name in ("resources", "requests"):
        if (destination / name).is_symlink():
            raise ArtifactError("TREC cache directories must not be symlinks")
    lock = destination / ".prepare-lock"
    deadline = time.monotonic() + 120
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if lock.is_symlink() or not lock.is_dir():
                raise ArtifactError("unsafe TREC preparation lock") from None
            if time.monotonic() >= deadline:
                raise ArtifactError(
                    "another TREC preparation holds the lock; cache was not modified"
                ) from None
            time.sleep(0.02)
    lock_identity = lock.stat().st_ino
    try:
        controls = dict(
            year=year,
            query_count=query_count,
            query_offset=query_offset,
            background_count=background_count,
            expected_corpus_revision=expected_corpus_revision,
        )
        plan = destination / "preparation.json"
        if not plan.exists() and any(p != lock for p in destination.iterdir()):
            raise ArtifactError("destination is not an empty toolkit sample cache")
        _write(plan, _dump(controls))
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            if manifest_path.is_symlink():
                raise ArtifactError("unsafe TREC sample manifest")
            digest = _sha(manifest_path.read_bytes())
            adapter = TRECDLSampleAdapter(
                destination, manifest_sha256=digest, license_acknowledged=True
            )
            return dict(
                root=str(destination.resolve()),
                manifest_sha256=digest,
                counts=dict(adapter.manifest["counts"]),
                acquired_from="verified_cache",
                corpus_scope="fixed_gold_enriched_compatibility_subset",
                full_dataset_score=False,
                downloaded_bytes=0,
            )
        require_storage_capacity([(destination, max_download_bytes + _BUNDLE_LIMIT)])
        client = transport if transport is not None else PinnedHTTPSDownloadTransport()
        policy = DownloadPolicy(
            allow_network=not offline,
            allowed_hosts=_HOSTS,
            max_bytes=max_download_bytes,
            timeout_seconds=60,
            max_redirects=3,
        )
        downloaded = [0]

        def get(
            url: str, *, revision: str | None = None, discovery: bool = False
        ) -> tuple[bytes, str]:
            key = _sha(url.encode())
            if discovery:
                if offline:
                    candidates = sorted((destination / "requests").glob(key + "-*.json"))
                    if not candidates:
                        raise ArtifactError("TREC corpus identity is not cached for offline use")
                    key = candidates[-1].stem
                else:
                    key += "-" + str(time.time_ns())
            meta = destination / "requests" / (key + ".json")
            name = "resources/" + key + ".raw"
            path = destination / name
            if meta.exists():
                if meta.is_symlink() or path.is_symlink():
                    raise ArtifactError("unsafe TREC source cache")
                record = _json(meta.read_bytes())
                raw = path.read_bytes()
                expected = dict(url=url, path=name, sha256=_sha(raw), size_bytes=len(raw))
                if revision is not None:
                    expected["corpus_revision"] = revision
                if record != expected:
                    raise ArtifactError("TREC source cache integrity differs")
                return raw, name
            if url in reused and reused_revisions[url] == revision:
                raw = reused[url]
            else:
                if offline:
                    raise ArtifactError(
                        "TREC source component is not cached; offline mode forbids download"
                    )
                target = validate_download_url(url, policy)
                remaining = max_download_bytes - downloaded[0]
                if remaining < 1:
                    raise ArtifactError("TREC source download allowance exhausted")
                for hop in range(policy.max_redirects + 1):
                    response = client.request(target, timeout_seconds=60, max_bytes=remaining)
                    _validate_peer(response.peer_address, target)
                    headers = {k.lower(): v for k, v in response.headers.items()}
                    if headers.get("content-encoding", "identity").lower() != "identity":
                        raise ArtifactError("TREC source transport must return identity encoding")
                    try:
                        declared = int(headers.get("content-length", len(response.body)))
                    except (ValueError, TypeError) as exc:
                        raise ArtifactError("TREC source Content-Length is invalid") from exc
                    if declared < 0 or declared > remaining:
                        raise ArtifactError("TREC source declared size exceeds download allowance")
                    if 300 <= response.status_code < 400:
                        location = next(
                            (v for k, v in response.headers.items() if k.lower() == "location"),
                            None,
                        )
                        if not location or hop == policy.max_redirects:
                            raise ArtifactError("TREC source redirect is absent or exceeds limit")
                        target = validate_download_url(
                            urllib.parse.urljoin(target.url, location), policy
                        )
                        continue
                    if response.status_code != 200:
                        raise ArtifactError(f"TREC source returned HTTP {response.status_code}")
                    break
                raw = response.body
                if not isinstance(raw, bytes) or len(raw) > remaining:
                    raise ArtifactError("TREC source exceeds bounded download allowance")
                downloaded[0] += len(raw)
            if len(raw) > _BUNDLE_LIMIT:
                raise ArtifactError("TREC source component exceeds sample bound")
            _write(path, raw)
            record = dict(url=url, path=name, sha256=_sha(raw), size_bytes=len(raw))
            if revision is not None:
                record["corpus_revision"] = revision
            _write(meta, _dump(record))
            return raw, name

        resources = []

        def resource(url: str, role: str, revision: str | None = None) -> bytes:
            raw, name = get(url, revision=revision)
            if len(raw) > _COMPONENT_LIMIT:
                raise ArtifactError("TREC native source component exceeds 256 KiB")
            resources.append(
                dict(
                    path=name,
                    role=role,
                    url=url,
                    sha256=_sha(raw),
                    size_bytes=len(raw),
                    source_revision=revision or _sha(raw),
                )
            )
            return raw

        metadata = _json(get("https://huggingface.co/api/datasets/BeIR/msmarco", discovery=True)[0])
        revision = metadata.get("sha")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or (expected_corpus_revision is not None and revision != expected_corpus_revision)
        ):
            raise ArtifactError("TREC corpus Hub revision differs from the explicit pin")
        splits = _json(
            get("https://datasets-server.huggingface.co/splits?dataset=BeIR%2Fmsmarco")[0]
        )
        if not any(
            s.get("config") == "corpus" and s.get("split") == "corpus"
            for s in splits.get("splits", [])
        ):
            raise ArtifactError("Dataset Viewer did not expose the native corpus split")
        qrel_raw = resource(f"https://trec.nist.gov/data/deep/{year}qrels-pass.txt", "qrels")
        query_raw = resource(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
            f"msmarco-test{year}-queries.tsv.gz",
            "queries_gzip",
        )
        with gzip.GzipFile(fileobj=io.BytesIO(query_raw)) as stream:
            plain = stream.read(_COMPONENT_LIMIT + 1)
        if len(plain) > _COMPONENT_LIMIT:
            raise ArtifactError("decoded TREC query source exceeds bound")
        queries = {}
        for line in plain.decode().splitlines():
            qid, text = line.split("\t", 1)
            if qid in queries:
                raise ArtifactError("duplicate official TREC query")
            queries[qid] = text
        qrels: dict[str, dict[str, int]] = {}
        for line in qrel_raw.decode().splitlines():
            qid, unused, did, label = line.split()
            grade = int(label)
            if (
                unused not in ("0", "Q0")
                or grade not in range(4)
                or did in qrels.setdefault(qid, {})
            ):
                raise ArtifactError("official TREC qrel duplicate or grade differs")
            qrels[qid][did] = grade
        chosen = sorted(set(qrels) & set(queries))[query_offset : query_offset + query_count]
        if len(chosen) != query_count:
            raise ArtifactError("insufficient independent official TREC queries")
        wanted = set()
        for qid in chosen:
            if not any(g >= 2 for g in qrels[qid].values()):
                raise ArtifactError("selected TREC query has no relevant official judgement")
            for grade, count in ((0, 2), (1, 1), (2, 2), (3, 1)):
                wanted.update(sorted(d for d, g in qrels[qid].items() if g == grade)[:count])
        corpus: dict[str, dict] = {}

        def rows(offset: int, length: int) -> None:
            url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
                dict(dataset=_REPO, config="corpus", split="corpus", offset=offset, length=length)
            )
            raw = resource(url, "corpus_rows", revision)
            value = _json(raw)
            if len(value.get("rows", [])) != length:
                raise ArtifactError("Dataset Viewer returned a partial source row range")
            for index, entry in enumerate(value["rows"]):
                row = entry["row"]
                did = str(row["_id"])
                if entry.get("truncated_cells") != [] or did != str(offset + index):
                    raise ArtifactError("Dataset Viewer returned truncated/different passage IDs")
                corpus[did] = dict(id=did, title=row.get("title", ""), text=row["text"], forum="")

        rows(0, background_count)
        for did in sorted(wanted - corpus.keys()):
            if not did.isdecimal():
                raise ArtifactError("TREC passage ID is not a native numeric row ID")
            rows(int(did), 1)
        normalized = {
            "inputs.jsonl": [
                dict(
                    id=qid,
                    query=queries[qid],
                    instruction="Retrieve passages.",
                    visible_metadata=dict(
                        benchmark_cell=f"trec_dl.{year}", source_split="test", forum=""
                    ),
                )
                for qid in chosen
            ],
            "gold.jsonl": [
                dict(id=qid, relevance={d: g for d, g in qrels[qid].items() if d in corpus})
                for qid in chosen
            ],
            "corpus.jsonl": list(corpus.values()),
        }
        files = {}
        for name, records in normalized.items():
            raw = b"".join(_dump(r) + b"\n" for r in records)
            _write(destination / name, raw)
            files[name] = dict(sha256=_sha(raw), size_bytes=len(raw))
        manifest = dict(
            schema_version="agentinputlab.trec_dl.sample.v1",
            benchmark_id=f"trec_dl.{year}",
            split="test",
            corpus_scope="fixed_gold_enriched_compatibility_subset",
            full_dataset_score=False,
            full_benchmark_corpus_download=False,
            binary_relevance_threshold=2,
            corpus_repository=_REPO,
            corpus_revision=revision,
            license_ids={"microsoft/msmarco": "CC-BY-4.0", "BeIR/msmarco": "CC-BY-SA-4.0"},
            files=files,
            resources=resources,
            counts=dict(inputs=len(chosen), gold=len(chosen), corpus=len(corpus)),
            query_selection=f"lexicographic_judged_query_ids[{query_offset}:{query_offset + query_count}]",
            candidate_selection="per_query_first2grade0_first1grade1_first2grade2_first1grade3_plus_background",
        )
        raw = _dump(manifest)
        if (
            len(raw)
            + sum(x["size_bytes"] for x in resources)
            + sum(x["size_bytes"] for x in files.values())
            > _BUNDLE_LIMIT
        ):
            raise ArtifactError("TREC normalized sample exceeds native 16 MiB bundle bound")
        _write(manifest_path, raw)
        digest = _sha(raw)
        adapter = TRECDLSampleAdapter(
            destination, manifest_sha256=digest, license_acknowledged=True
        )
        return dict(
            root=str(destination.resolve()),
            manifest_sha256=digest,
            counts=dict(adapter.manifest["counts"]),
            acquired_from="bounded_source_preparation",
            corpus_scope="fixed_gold_enriched_compatibility_subset",
            full_dataset_score=False,
            downloaded_bytes=downloaded[0],
        )
    finally:
        if lock.is_dir() and not lock.is_symlink() and lock.stat().st_ino == lock_identity:
            lock.rmdir()


__all__ = ["prepare_trec_dl_sample"]
