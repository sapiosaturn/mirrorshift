from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


INDEX_MAGIC = b"MRRDIDX\x00"
INDEX_VERSION = 1
HEADER_FORMAT = "<8sQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass(frozen=True)
class IndexHeader:
    version: int
    num_documents: int
    num_tokens: int

    def to_bytes(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            INDEX_MAGIC,
            int(self.version),
            int(self.num_documents),
            int(self.num_tokens),
            0,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "IndexHeader":
        magic, version, num_documents, num_tokens, _reserved = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        if magic != INDEX_MAGIC:
            raise ValueError(f"invalid index magic {magic!r}")
        if version != INDEX_VERSION:
            raise ValueError(f"unsupported index version {version}")
        return cls(
            version=int(version),
            num_documents=int(num_documents),
            num_tokens=int(num_tokens),
        )


class IndexWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._boundaries: list[tuple[int, int]] = []
        self._num_tokens = 0

    def add_document(self, start: int, end: int) -> None:
        if start < 0 or end < start:
            raise ValueError(f"invalid document boundary ({start}, {end})")
        self._boundaries.append((int(start), int(end)))
        self._num_tokens = max(self._num_tokens, int(end))

    def finalize(self) -> IndexHeader:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = IndexHeader(
            version=INDEX_VERSION,
            num_documents=len(self._boundaries),
            num_tokens=self._num_tokens,
        )
        body = np.array(self._boundaries, dtype=np.uint64).reshape(-1)

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "wb") as handle:
            handle.write(header.to_bytes())
            handle.write(body.tobytes())
        tmp_path.replace(self.path)
        return header


class IndexReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._header: IndexHeader | None = None
        self._boundaries: np.ndarray | None = None

    @property
    def header(self) -> IndexHeader:
        if self._header is None:
            with open(self.path, "rb") as handle:
                self._header = IndexHeader.from_bytes(handle.read(HEADER_SIZE))
        return self._header

    @property
    def boundaries(self) -> np.ndarray:
        if self._boundaries is None:
            data = np.memmap(
                self.path,
                dtype=np.uint64,
                mode="r",
                offset=HEADER_SIZE,
                shape=(self.header.num_documents, 2),
            )
            self._boundaries = data
        return self._boundaries

    def get_document(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.header.num_documents:
            raise IndexError(index)
        start, end = self.boundaries[index]
        return int(start), int(end)

    def iter_documents(self) -> Iterator[tuple[int, int]]:
        for row in self.boundaries:
            yield int(row[0]), int(row[1])
