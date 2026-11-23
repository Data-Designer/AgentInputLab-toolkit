"""Data acquisition, checksum, preprocessing, index, and overlap utilities."""

from .downloader import (
    DownloadPolicy,
    DownloadResponse,
    DownloadTransport,
    StreamingDownloadResponse,
    StreamingDownloadTransport,
    ValidatedDownloadTarget,
    download_artifact,
    download_artifact_streamed,
    validate_download_url,
)
from .http import PinnedHTTPSDownloadTransport
from .overlap import OverlapRecord, detect_id_overlap
from .trec_sample import prepare_trec_dl_sample

__all__ = [
    "DownloadPolicy",
    "DownloadResponse",
    "DownloadTransport",
    "OverlapRecord",
    "PinnedHTTPSDownloadTransport",
    "StreamingDownloadResponse",
    "StreamingDownloadTransport",
    "ValidatedDownloadTarget",
    "detect_id_overlap",
    "download_artifact",
    "download_artifact_streamed",
    "prepare_trec_dl_sample",
    "validate_download_url",
]
