from pathlib import Path

import torch

from mirrordata import SnapshotManifest, TiktokenTextDataset, TiktokenEncoding, TokenizerManifest


def test_tiktoken_text_dataset_round_trip(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text("hello world " * 20)

    dataset = TiktokenTextDataset(str(corpus_path), sequence_length=8)

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert dataset.get_vocab_size() > 0


def test_snapshot_manifest_round_trip(tmp_path: Path) -> None:
    manifest = SnapshotManifest(
        format_version="0.1",
        snapshot_id="tiny-snapshot",
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
    assert loaded.tokenizer.name == "p50k_base"


def test_tiktoken_encoding_exposes_vocab_size() -> None:
    encoding = TiktokenEncoding("p50k_base")

    assert encoding.n_vocab > 0
