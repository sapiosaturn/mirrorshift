from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from mirrordata.snapshot import SnapshotManifest
from mirrordata.tokenizers import TokenizerBackend


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentSource(Protocol):
    def estimate_documents(self) -> int | None:
        ...

    def __iter__(self) -> Iterable[Document]:
        ...


class TextTransform(Protocol):
    def __call__(self, document: Document) -> Document | None:
        ...


class TokenTransform(Protocol):
    def __call__(self, token_ids: Sequence[int], *, document: Document) -> Sequence[int] | None:
        ...


class SnapshotWriter(Protocol):
    def write_document(self, token_ids: Sequence[int]) -> None:
        ...

    def finalize(self) -> SnapshotManifest:
        ...


@dataclass(frozen=True)
class ParquetInput:
    paths: tuple[Path, ...]
    text_column: str = "text"
