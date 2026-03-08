from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow.parquet as pq

from .contracts import Document


def expand_paths(paths: Sequence[str | Path]) -> list[Path]:
    expanded = sorted({Path(path).expanduser().resolve() for path in paths})
    if not expanded:
        raise ValueError("no input paths provided")
    return expanded


class ParquetTextSource:
    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        text_column: str = "text",
        batch_size: int = 4096,
        limit_documents: int | None = None,
    ) -> None:
        self.paths = expand_paths(paths)
        self.text_column = text_column
        self.batch_size = int(batch_size)
        self.limit_documents = limit_documents

    def estimate_documents(self) -> int | None:
        total = 0
        for path in self.paths:
            total += pq.ParquetFile(path).metadata.num_rows
        if self.limit_documents is not None:
            return min(total, self.limit_documents)
        return total

    def __iter__(self) -> Iterator[Document]:
        emitted = 0
        for path in self.paths:
            parquet = pq.ParquetFile(path)
            schema_names = parquet.schema_arrow.names
            if self.text_column not in schema_names:
                raise ValueError(f"missing text column '{self.text_column}' in {path}")

            row_offset = 0
            for batch in parquet.iter_batches(columns=[self.text_column], batch_size=self.batch_size):
                texts = batch.column(0).to_pylist()
                for local_index, text in enumerate(texts):
                    if self.limit_documents is not None and emitted >= self.limit_documents:
                        return
                    if text is None:
                        continue
                    yield Document(
                        document_id=f"{path.name}:{row_offset + local_index}",
                        text=str(text),
                        metadata={"source_path": str(path), "row_index": row_offset + local_index},
                    )
                    emitted += 1
                row_offset += len(texts)
