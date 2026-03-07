from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mirrordata.snapshot import SnapshotBuilder, SnapshotManifest
from mirrordata.tokenizers import TiktokenTokenizer

from .contracts import Document
from .sources import ParquetTextSource
from .transforms import append_eos, normalize_text


@dataclass(frozen=True)
class ParquetSnapshotConfig:
    input_paths: tuple[str, ...]
    output_dir: str
    snapshot_id: str
    dataset_name: str
    split: str = "train"
    text_column: str = "text"
    tokenizer_name: str = "p50k_base"
    max_tokens_per_shard: int = 50_000_000
    min_document_tokens: int = 1
    batch_size: int = 4096
    append_eos_token: bool = True
    token_dtype: str = "uint32"
    metadata: dict[str, str] = field(default_factory=dict)


class ParquetSnapshotPreprocessor:
    def __init__(self, config: ParquetSnapshotConfig) -> None:
        self.config = config
        self.tokenizer = TiktokenTokenizer(config.tokenizer_name)

    def _tokenize_document(self, document: Document) -> list[int] | None:
        text = normalize_text(document.text)
        if not text:
            return None
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) < self.config.min_document_tokens:
            return None
        if self.config.append_eos_token:
            token_ids = append_eos(token_ids, self.tokenizer.eos_token_id)
        return token_ids

    def run(self) -> SnapshotManifest:
        source = ParquetTextSource(
            self.config.input_paths,
            text_column=self.config.text_column,
            batch_size=self.config.batch_size,
        )
        builder = SnapshotBuilder(
            output_dir=self.config.output_dir,
            snapshot_id=self.config.snapshot_id,
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            tokenizer_manifest=self.tokenizer.to_manifest(),
            max_tokens_per_shard=self.config.max_tokens_per_shard,
            token_dtype=np.dtype(self.config.token_dtype),
            metadata={
                "input_paths": list(self.config.input_paths),
                "text_column": self.config.text_column,
                **self.config.metadata,
            },
        )

        for document in source:
            token_ids = self._tokenize_document(document)
            if token_ids is None:
                continue
            builder.write_document(token_ids)
        return builder.finalize()


def build_snapshot_from_parquet(config: ParquetSnapshotConfig) -> SnapshotManifest:
    return ParquetSnapshotPreprocessor(config).run()
