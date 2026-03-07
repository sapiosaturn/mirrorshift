from pathlib import Path

import numpy as np

from mirrordata import (
    IndexReader,
    IndexWriter,
    SnapshotBuilder,
    SnapshotManifest,
    TokenSnapshot,
    TokenizerManifest,
)


def test_index_writer_reader_round_trip(tmp_path: Path) -> None:
    index_path = tmp_path / "tokens.idx"
    writer = IndexWriter(index_path)
    writer.add_document(0, 3)
    writer.add_document(3, 7)
    header = writer.finalize()

    reader = IndexReader(index_path)

    assert header.num_documents == 2
    assert header.num_tokens == 7
    assert reader.header.num_documents == 2
    assert reader.get_document(0) == (0, 3)
    assert list(reader.iter_documents()) == [(0, 3), (3, 7)]


def test_snapshot_manifest_round_trip(tmp_path: Path) -> None:
    manifest = SnapshotManifest.create(
        snapshot_id="tiny-snapshot",
        dataset_name="tiny",
        split="train",
        token_dtype="uint32",
        tokenizer=TokenizerManifest(backend="tiktoken", name="p50k_base", vocab_size=50281),
        shards=[],
        total_tokens=0,
        total_documents=0,
        metadata={"source": "unit-test"},
    )

    manifest_path = tmp_path / "manifest.json"
    manifest.write_json(manifest_path)

    loaded = SnapshotManifest.read_json(manifest_path)

    assert loaded.snapshot_id == manifest.snapshot_id
    assert loaded.dataset_name == "tiny"
    assert loaded.tokenizer.name == "p50k_base"


def test_snapshot_builder_and_reader_across_shards(tmp_path: Path) -> None:
    builder = SnapshotBuilder(
        output_dir=tmp_path,
        snapshot_id="synthetic",
        dataset_name="synthetic",
        split="train",
        tokenizer_manifest=TokenizerManifest(backend="tiktoken", name="p50k_base", vocab_size=100),
        max_tokens_per_shard=4,
        token_dtype=np.dtype(np.uint32),
    )
    builder.write_document([1, 2, 3])
    builder.write_document([4, 5, 6])
    builder.write_document([7, 8])

    manifest = builder.finalize()
    snapshot = TokenSnapshot.open(tmp_path)

    assert manifest.total_documents == 3
    assert manifest.total_tokens == 8
    assert len(manifest.shards) == 3
    assert snapshot.read_tokens(0, 8).tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert snapshot.read_window(2, 4).tolist() == [3, 4, 5, 6]
