from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Sequence

import numpy as np

from .manifest import SnapshotManifest


class TokenSnapshot:
    def __init__(self, root: str | Path, manifest: SnapshotManifest | None = None) -> None:
        root_path = Path(root)
        self.root = root_path if root_path.is_dir() else root_path.parent
        self.manifest = manifest or SnapshotManifest.read_json(self.root / "manifest.json")
        self._shard_ends = [shard.token_offset_end for shard in self.manifest.shards]
        self._token_arrays = [
            np.memmap(
                self.root / shard.token_path,
                dtype=np.uint32,
                mode="r",
                shape=(shard.num_tokens,),
            )
            for shard in self.manifest.shards
        ]

    @classmethod
    def open(cls, path: str | Path) -> "TokenSnapshot":
        input_path = Path(path)
        if input_path.is_dir():
            return cls(input_path)
        if input_path.name == "manifest.json":
            return cls(input_path.parent, SnapshotManifest.read_json(input_path))
        raise ValueError(f"expected snapshot directory or manifest.json, got {input_path}")

    @property
    def total_tokens(self) -> int:
        return self.manifest.total_tokens

    def _resolve(self, global_offset: int) -> tuple[int, int]:
        if global_offset < 0 or global_offset >= self.total_tokens:
            raise IndexError(global_offset)
        shard_index = bisect_right(self._shard_ends, global_offset)
        shard_manifest = self.manifest.shards[shard_index]
        local_offset = global_offset - shard_manifest.token_offset_begin
        return shard_index, int(local_offset)

    def read_tokens(self, start: int, length: int) -> np.ndarray:
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return np.empty((0,), dtype=np.uint32)
        if start < 0 or start + length > self.total_tokens:
            raise IndexError((start, length))

        remaining = int(length)
        cursor = int(start)
        pieces: list[np.ndarray] = []
        while remaining > 0:
            shard_index, local_offset = self._resolve(cursor)
            shard_manifest = self.manifest.shards[shard_index]
            shard_tokens = self._token_arrays[shard_index]
            available = min(remaining, shard_manifest.num_tokens - local_offset)
            pieces.append(np.asarray(shard_tokens[local_offset : local_offset + available]))
            remaining -= available
            cursor += available
        return pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)

    def read_window(self, start: int, length: int) -> np.ndarray:
        return self.read_tokens(start, length)

    def shard_paths(self) -> Sequence[Path]:
        return [self.root / shard.token_path for shard in self.manifest.shards]
