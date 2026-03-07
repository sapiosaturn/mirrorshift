from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mirrordata import (
    ParquetSnapshotConfig,
    ParquetTextSource,
    TiktokenTextDataset,
    TiktokenTokenizer,
    build_snapshot_from_parquet,
)


def _write_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table(
        {
            "text": texts,
            "id": [f"doc-{i}" for i in range(len(texts))],
            "language": ["en"] * len(texts),
        }
    )
    pq.write_table(table, path, row_group_size=2)


def test_tiktoken_text_dataset_round_trip(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text("hello world " * 20)

    dataset = TiktokenTextDataset(str(corpus_path), sequence_length=8)
    x, y = dataset[0]

    assert x.shape == (8,)
    assert y.shape == (8,)
    assert dataset.get_vocab_size() > 0


def test_parquet_text_source_reads_documents(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path, ["first doc", "second doc", "third doc"])

    source = ParquetTextSource([parquet_path], text_column="text", batch_size=2)
    docs = list(source)

    assert source.estimate_documents() == 3
    assert [doc.text for doc in docs] == ["first doc", "second doc", "third doc"]
    assert docs[0].metadata["source_path"].endswith("data.parquet")


def test_build_snapshot_from_parquet_writes_manifest_and_shards(tmp_path: Path) -> None:
    parquet_path = tmp_path / "train.parquet"
    texts = [
        "alpha beta gamma delta epsilon",
        "zeta eta theta iota kappa",
        "lambda mu nu xi omicron",
    ]
    _write_parquet(parquet_path, texts)

    tokenizer = TiktokenTokenizer("p50k_base")
    max_doc_tokens = max(
        len(tokenizer.encode(text)) + (1 if tokenizer.eos_token_id is not None else 0)
        for text in texts
    )
    max_tokens_per_shard = max_doc_tokens + 1

    manifest = build_snapshot_from_parquet(
        ParquetSnapshotConfig(
            input_paths=(str(parquet_path),),
            output_dir=str(tmp_path / "snapshot"),
            snapshot_id="train-snapshot",
            dataset_name="tiny-train",
            split="train",
            max_tokens_per_shard=max_tokens_per_shard,
        )
    )

    assert manifest.snapshot_id == "train-snapshot"
    assert manifest.total_documents == 3
    assert manifest.total_tokens > 0
    assert len(manifest.shards) >= 2
    assert (tmp_path / "snapshot" / "manifest.json").exists()
    for shard in manifest.shards:
        assert (tmp_path / "snapshot" / shard.token_path).exists()
        assert (tmp_path / "snapshot" / shard.index_path).exists()


def test_reference_parquet_smoke() -> None:
    parquet_path = Path("/Users/varunsingh/Documents/mirrorshift/reference/000_00000.parquet")
    if not parquet_path.exists():
        pytest.skip("reference parquet not available")

    source = ParquetTextSource([parquet_path], text_column="text", batch_size=8)
    first = next(iter(source))

    assert len(first.text) > 0
