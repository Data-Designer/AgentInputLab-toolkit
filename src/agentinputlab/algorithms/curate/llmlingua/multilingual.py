"""Official LLMLingua-2 multilingual BERT, with an exact asset closure.

The token classifier is a fixed auxiliary model, independent of the evaluated
reader. No XLM-R weights or downstream-backbone weights are substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentinputlab.artifacts import sha256_file
from agentinputlab.contracts import ExternalDependencyError, ValidationError

from .adapters import LLMLingua2OfficialAdapter, _OfficialAdapterBase

BERT_MULTILINGUAL_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
BERT_MULTILINGUAL_REVISION = "5f0c82792b7ea14c6484e015b6a072009496b7f2"
BERT_MULTILINGUAL_FILES = (
    ("config.json", "e6c33ec2f099e659e125efaa8b7bb07a6e65b7d9fc36211c6fa3206e1c399085"),
    ("tokenizer_config.json", "214f78f6c5d034a8b405632d301ee3710eb6f075b3f5faf8241a273a13908746"),
    ("tokenizer.json", "bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088"),
    ("vocab.txt", "fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c"),
    ("special_tokens_map.json", "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    ("model.safetensors", "22b9ecde52fec5c97e8c54a293be768727df95a81c6c8dccb03f262a50c58324"),
)


@lru_cache(maxsize=8)
def _verify_snapshot(snapshot: str, identities: tuple[tuple[str, int, int], ...]) -> None:
    """Hash immutable weights once per local path/size/mtime identity."""
    del identities
    for name, expected in BERT_MULTILINGUAL_FILES:
        if sha256_file(Path(snapshot) / name) != expected:
            raise ExternalDependencyError(f"multilingual BERT asset checksum differs: {name}")


@dataclass(slots=True)
class LLMLingua2BertMultilingualOfficialAdapter(LLMLingua2OfficialAdapter):
    """Pinned author checkpoint and unchanged upstream compression dispatcher.

    Pass an existing snapshot as ``model_name`` for offline use. Otherwise the
    built-in loader downloads only the six pinned model/tokenizer files. An
    injected compressor factory is caller-owned code, not accepted-model proof.
    """

    model_name: str = BERT_MULTILINGUAL_MODEL
    model_revision: str = BERT_MULTILINGUAL_REVISION
    release_id: str = "curate.llmlingua2.bert_multilingual_official"

    def __post_init__(self) -> None:
        _OfficialAdapterBase.__post_init__(self)
        if self.model_revision != BERT_MULTILINGUAL_REVISION:
            raise ValidationError("multilingual BERT requires its pinned checkpoint revision")
        if self.release_id != "curate.llmlingua2.bert_multilingual_official":
            raise ValidationError("multilingual BERT release identity cannot be replaced")
        if self.profile_kind not in {"official_native", "factorial_adapter"}:
            raise ValidationError("multilingual BERT requires an official or factorial profile")
        allowed = {"revision", "trust_remote_code", "local_files_only", "torch_dtype"}
        if set(self.model_config) - allowed:
            raise ValidationError("unsupported multilingual BERT loading option")
        if self.model_config.get("revision", self.model_revision) != self.model_revision:
            raise ValidationError("multilingual BERT revision cannot be overridden")
        if self.model_config.get("trust_remote_code") not in {None, False}:
            raise ValidationError("multilingual BERT disables remote code")
        if not isinstance(self.model_config.get("local_files_only", False), bool):
            raise ValidationError("local_files_only must be boolean")
        self._validate_torch_dtype(self.model_config.get("torch_dtype", "auto"))
        if self.model_name != BERT_MULTILINGUAL_MODEL and not Path(self.model_name).is_dir():
            raise ValidationError("model_name must be the official BERT ID or a local snapshot")

    def _load(self, *, use_llmlingua2: bool) -> Any:
        if not use_llmlingua2:
            raise ValidationError("multilingual BERT requires LLMLingua-2 token classification")
        if self.compressor_factory is None:
            try:
                snapshot = Path(self.model_name)
                if self.model_name == BERT_MULTILINGUAL_MODEL:
                    from huggingface_hub import snapshot_download

                    snapshot = Path(snapshot_download(
                        self.model_name,
                        revision=self.model_revision,
                        allow_patterns=[name for name, _ in BERT_MULTILINGUAL_FILES],
                        local_files_only=self.model_config.get("local_files_only", False),
                    ))
                identities = tuple(
                    (name, (snapshot / name).stat().st_size, (snapshot / name).stat().st_mtime_ns)
                    for name, _ in BERT_MULTILINGUAL_FILES
                )
                _verify_snapshot(str(snapshot.resolve()), identities)
            except (ImportError, OSError, ValueError) as exc:
                raise ExternalDependencyError(
                    "multilingual BERT needs its complete pinned safetensors/tokenizer snapshot "
                    "and `pip install agentinputlab[compression]`"
                ) from exc
        return _OfficialAdapterBase._load(self, use_llmlingua2=True)


__all__ = [
    "BERT_MULTILINGUAL_FILES", "BERT_MULTILINGUAL_MODEL", "BERT_MULTILINGUAL_REVISION",
    "LLMLingua2BertMultilingualOfficialAdapter",
]
