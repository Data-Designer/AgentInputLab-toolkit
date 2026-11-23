"""R²AG retrieval-feature conditioning with a checkpoint worker seam."""

from .bridge import (
    R2AG_FEATURE_NAMES,
    R2AG_PROTOCOL,
    R2AG_UPSTREAM_COMMIT,
    R2AGArtifactIdentity,
    R2AGControlledWorker,
    R2AGControls,
    R2AGFeatureSource,
)
from .common_support import (
    LocalR2AGCommonSupportReader,
    LocalR2AGCommonSupportRuntime,
    LocalSentenceTransformerR2AGFeatureEncoder,
    R2AGCommonSupport,
    R2AGCommonSupportConfig,
    R2AGCommonSupportReader,
    R2AGCommonSupportRuntime,
    R2AGFeatureEncoder,
    R2AGReaderResult,
    R2AGRuntimeRequest,
    r2ag_model_tree_sha256,
    r2ag_reader_prompt,
)
from .socket_runtime import (
    R2AG_RUNTIME_SOCKET_ENV,
    R2AG_RUNTIME_TIMEOUT_ENV,
    UnixSocketR2AGCommonSupportRuntime,
    serve_unix_socket_r2ag_runtime,
)

__all__ = [
    "R2AG_FEATURE_NAMES",
    "R2AG_PROTOCOL",
    "R2AG_RUNTIME_SOCKET_ENV",
    "R2AG_RUNTIME_TIMEOUT_ENV",
    "R2AG_UPSTREAM_COMMIT",
    "LocalR2AGCommonSupportReader",
    "LocalR2AGCommonSupportRuntime",
    "LocalSentenceTransformerR2AGFeatureEncoder",
    "R2AGArtifactIdentity",
    "R2AGCommonSupport",
    "R2AGCommonSupportConfig",
    "R2AGCommonSupportReader",
    "R2AGCommonSupportRuntime",
    "R2AGControlledWorker",
    "R2AGControls",
    "R2AGFeatureEncoder",
    "R2AGFeatureSource",
    "R2AGReaderResult",
    "R2AGRuntimeRequest",
    "UnixSocketR2AGCommonSupportRuntime",
    "r2ag_model_tree_sha256",
    "r2ag_reader_prompt",
    "serve_unix_socket_r2ag_runtime",
]
