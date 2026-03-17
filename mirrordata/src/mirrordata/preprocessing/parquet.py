from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mirrordata.snapshot import SnapshotBuilder, SnapshotManifest
from mirrordata.tokenizers import TiktokenTokenizer

from .contracts import Document
from .sources import ParquetTextSource
from .transforms import append_eos, normalize_text

LOGGER = logging.getLogger("mirrordata.preprocessing.parquet")


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
    max_documents: int | None = None
    log_every_documents: int = 10_000
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
            limit_documents=self.config.max_documents,
        )
        estimated_documents = source.estimate_documents()
        LOGGER.info(
            "Starting parquet snapshot build: snapshot_id=%s dataset=%s split=%s "
            "input_files=%d estimated_documents=%s output_dir=%s tokenizer=%s",
            self.config.snapshot_id,
            self.config.dataset_name,
            self.config.split,
            len(source.paths),
            estimated_documents if estimated_documents is not None else "unknown",
            self.config.output_dir,
            self.config.tokenizer_name,
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

        documents_seen = 0
        documents_written = 0
        documents_skipped = 0
        tokens_written = 0

        for document in source:
            documents_seen += 1
            token_ids = self._tokenize_document(document)
            if token_ids is None:
                documents_skipped += 1
                if (
                    self.config.log_every_documents > 0
                    and documents_seen % self.config.log_every_documents == 0
                ):
                    LOGGER.info(
                        "Preprocessing progress: seen=%d written=%d skipped=%d tokens=%d",
                        documents_seen,
                        documents_written,
                        documents_skipped,
                        tokens_written,
                    )
                continue
            builder.write_document(token_ids)
            documents_written += 1
            tokens_written += len(token_ids)
            if (
                self.config.log_every_documents > 0
                and documents_seen % self.config.log_every_documents == 0
            ):
                LOGGER.info(
                    "Preprocessing progress: seen=%d written=%d skipped=%d tokens=%d",
                    documents_seen,
                    documents_written,
                    documents_skipped,
                    tokens_written,
                )

        manifest = builder.finalize()
        LOGGER.info(
            "Finished parquet snapshot build: snapshot_id=%s documents_seen=%d "
            "documents_written=%d documents_skipped=%d total_tokens=%d shards=%d manifest=%s",
            manifest.snapshot_id,
            documents_seen,
            documents_written,
            documents_skipped,
            manifest.total_tokens,
            len(manifest.shards),
            Path(self.config.output_dir) / "manifest.json",
        )
        return manifest


def build_snapshot_from_parquet(config: ParquetSnapshotConfig) -> SnapshotManifest:
    return ParquetSnapshotPreprocessor(config).run()
