from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SNAPSHOT_FORMAT_VERSION = "1"


@dataclass(frozen=True)
class TokenizerManifest:
    backend: str
    name: str
    vocab_size: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ShardManifest:
    shard_id: int
    name: str
    token_path: str
    index_path: str
    token_offset_begin: int
    token_offset_end: int
    num_tokens: int
    num_documents: int
    token_checksum_sha256: str | None = None
    index_checksum_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "name": self.name,
            "token_path": self.token_path,
            "index_path": self.index_path,
            "token_offset_begin": self.token_offset_begin,
            "token_offset_end": self.token_offset_end,
            "num_tokens": self.num_tokens,
            "num_documents": self.num_documents,
            "token_checksum_sha256": self.token_checksum_sha256,
            "index_checksum_sha256": self.index_checksum_sha256,
        }


@dataclass(frozen=True)
class SnapshotManifest:
    format_version: str
    snapshot_id: str
    dataset_name: str
    split: str
    token_dtype: str
    tokenizer: TokenizerManifest
    shards: list[ShardManifest]
    total_tokens: int
    total_documents: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "snapshot_id": self.snapshot_id,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "token_dtype": self.token_dtype,
            "tokenizer": self.tokenizer.to_dict(),
            "shards": [shard.to_dict() for shard in self.shards],
            "total_tokens": self.total_tokens,
            "total_documents": self.total_documents,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotManifest":
        tokenizer = TokenizerManifest(**data["tokenizer"])
        shards = [ShardManifest(**shard) for shard in data["shards"]]
        return cls(
            format_version=str(data["format_version"]),
            snapshot_id=str(data["snapshot_id"]),
            dataset_name=str(data["dataset_name"]),
            split=str(data["split"]),
            token_dtype=str(data["token_dtype"]),
            tokenizer=tokenizer,
            shards=shards,
            total_tokens=int(data["total_tokens"]),
            total_documents=int(data["total_documents"]),
            created_at=str(data["created_at"]),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "SnapshotManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        dataset_name: str,
        split: str,
        token_dtype: str,
        tokenizer: TokenizerManifest,
        shards: list[ShardManifest],
        total_tokens: int,
        total_documents: int,
        metadata: dict[str, Any] | None = None,
    ) -> "SnapshotManifest":
        return cls(
            format_version=SNAPSHOT_FORMAT_VERSION,
            snapshot_id=snapshot_id,
            dataset_name=dataset_name,
            split=split,
            token_dtype=token_dtype,
            tokenizer=tokenizer,
            shards=shards,
            total_tokens=total_tokens,
            total_documents=total_documents,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(metadata or {}),
        )
