from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .index import IndexWriter
from .manifest import ShardManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenShardWriter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str,
        shard_id: int,
        max_tokens: int,
        dtype: np.dtype = np.dtype(np.uint32),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.split = split
        self.shard_id = int(shard_id)
        self.max_tokens = int(max_tokens)
        self.dtype = np.dtype(dtype)

        self.name = f"{split}-tokens-{self.shard_id:05d}"
        self.token_path = self.output_dir / f"{self.name}.bin"
        self.index_path = self.output_dir / f"{self.name}.idx"

        self._temp_path = self.output_dir / f"{self.name}.bin.tmp"
        self._handle = open(self._temp_path, "wb")
        self._index_writer = IndexWriter(self.index_path)
        self._num_tokens = 0
        self._num_documents = 0

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_documents(self) -> int:
        return self._num_documents

    def can_fit(self, num_tokens: int) -> bool:
        return self._num_tokens + int(num_tokens) <= self.max_tokens

    def add_document(
        self,
        token_ids: Sequence[int],
        *,
        allow_oversized_empty: bool = False,
    ) -> bool:
        token_array = np.asarray(token_ids, dtype=self.dtype)
        if token_array.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if token_array.size == 0:
            return True
        if not self.can_fit(int(token_array.size)):
            if not (allow_oversized_empty and self._num_documents == 0):
                return False
        if self._num_tokens > 0 and not self.can_fit(int(token_array.size)):
            return False

        start = self._num_tokens
        end = start + int(token_array.size)
        self._index_writer.add_document(start, end)
        self._handle.write(token_array.tobytes())
        self._num_tokens = end
        self._num_documents += 1
        return True

    def finalize(self, *, token_offset_begin: int) -> ShardManifest:
        if self._num_documents == 0:
            raise ValueError("cannot finalize an empty shard")
        self._handle.close()
        self._temp_path.replace(self.token_path)
        self._index_writer.finalize()
        token_checksum = _sha256_file(self.token_path)
        index_checksum = _sha256_file(self.index_path)
        return ShardManifest(
            shard_id=self.shard_id,
            name=self.name,
            token_path=str(Path("shards") / self.token_path.name),
            index_path=str(Path("shards") / self.index_path.name),
            token_offset_begin=int(token_offset_begin),
            token_offset_end=int(token_offset_begin + self._num_tokens),
            num_tokens=self._num_tokens,
            num_documents=self._num_documents,
            token_checksum_sha256=token_checksum,
            index_checksum_sha256=index_checksum,
        )

    def discard(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self._temp_path.unlink(missing_ok=True)


class SnapshotBuilder:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        snapshot_id: str,
        dataset_name: str,
        split: str,
        tokenizer_manifest,
        max_tokens_per_shard: int,
        token_dtype: np.dtype = np.dtype(np.uint32),
        metadata: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.snapshot_id = snapshot_id
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer_manifest = tokenizer_manifest
        self.max_tokens_per_shard = int(max_tokens_per_shard)
        self.token_dtype = np.dtype(token_dtype)
        self.metadata = dict(metadata or {})

        self.shards_dir = self.output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self._current_shard_id = 0
        self._current_writer = TokenShardWriter(
            output_dir=self.shards_dir,
            split=self.split,
            shard_id=self._current_shard_id,
            max_tokens=self.max_tokens_per_shard,
            dtype=self.token_dtype,
        )
        self._shards: list[ShardManifest] = []
        self._total_tokens = 0
        self._total_documents = 0

    def write_document(self, token_ids: Sequence[int]) -> None:
        if len(token_ids) == 0:
            return
        if not self._current_writer.can_fit(len(token_ids)) and self._current_writer.num_documents > 0:
            self._roll_shard()
        if not self._current_writer.add_document(token_ids, allow_oversized_empty=True):
            raise ValueError(
                f"document with {len(token_ids)} tokens could not be written to shard "
                f"with max_tokens_per_shard={self.max_tokens_per_shard}"
            )
        self._total_documents += 1
        self._total_tokens += len(token_ids)

    def _roll_shard(self) -> None:
        if self._current_writer.num_documents > 0:
            self._finalize_current_writer()
        self._current_shard_id += 1
        self._current_writer = TokenShardWriter(
            output_dir=self.shards_dir,
            split=self.split,
            shard_id=self._current_shard_id,
            max_tokens=self.max_tokens_per_shard,
            dtype=self.token_dtype,
        )

    def _finalize_current_writer(self) -> None:
        shard_manifest = self._current_writer.finalize(
            token_offset_begin=sum(shard.num_tokens for shard in self._shards)
        )
        self._shards.append(shard_manifest)

    def finalize(self):
        if self._current_writer.num_documents > 0:
            self._finalize_current_writer()
        else:
            self._current_writer.discard()

        from .manifest import SnapshotManifest

        manifest = SnapshotManifest.create(
            snapshot_id=self.snapshot_id,
            dataset_name=self.dataset_name,
            split=self.split,
            token_dtype=str(self.token_dtype),
            tokenizer=self.tokenizer_manifest,
            shards=self._shards,
            total_tokens=self._total_tokens,
            total_documents=self._total_documents,
            metadata=self.metadata,
        )
        manifest.write_json(self.output_dir / "manifest.json")
        return manifest
