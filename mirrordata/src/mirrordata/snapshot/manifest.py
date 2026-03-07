from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenizerManifest:
    backend: str
    name: str
    vocab_size: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }


@dataclass(frozen=True)
class ShardManifest:
    name: str
    token_path: str
    index_path: str
    num_tokens: int
    num_documents: int
    checksum_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "token_path": self.token_path,
            "index_path": self.index_path,
            "num_tokens": self.num_tokens,
            "num_documents": self.num_documents,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True)
class SnapshotManifest:
    format_version: str
    snapshot_id: str
    token_dtype: str
    tokenizer: TokenizerManifest
    shards: list[ShardManifest]
    total_tokens: int
    total_documents: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "snapshot_id": self.snapshot_id,
            "token_dtype": self.token_dtype,
            "tokenizer": self.tokenizer.to_dict(),
            "shards": [shard.to_dict() for shard in self.shards],
            "total_tokens": self.total_tokens,
            "total_documents": self.total_documents,
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
            token_dtype=str(data["token_dtype"]),
            tokenizer=tokenizer,
            shards=shards,
            total_tokens=int(data["total_tokens"]),
            total_documents=int(data["total_documents"]),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "SnapshotManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))
