from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mirrordata import (
    CausalLMSequenceDataset,
    DeterministicBatchLoader,
    ParquetSnapshotConfig,
    SequencePlan,
    SequencePlanSpec,
    TokenSnapshot,
    build_sequence_plan,
    build_snapshot_from_parquet,
)


def _write_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path, row_group_size=2)


def _build_snapshot_and_plan(tmp_path: Path) -> tuple[TokenSnapshot, SequencePlan]:
    parquet_path = tmp_path / "train.parquet"
    texts = [
        "one two three four five six seven eight nine ten",
        "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen",
        "nineteen twenty twentyone twentytwo twentythree twentyfour",
    ]
    _write_parquet(parquet_path, texts)
    snapshot_manifest = build_snapshot_from_parquet(
        ParquetSnapshotConfig(
            input_paths=(str(parquet_path),),
            output_dir=str(tmp_path / "snapshot"),
            snapshot_id="tiny",
            dataset_name="tiny",
            split="train",
            max_tokens_per_shard=20,
        )
    )
    build_sequence_plan(
        SequencePlanSpec(
            snapshot_path=str(tmp_path / "snapshot"),
            output_dir=str(tmp_path / "plan"),
            sequence_length=4,
            stride=2,
            shuffle=True,
            shuffle_seed=123,
        )
    )
    return TokenSnapshot.open(tmp_path / "snapshot"), SequencePlan.open(tmp_path / "plan")


def test_build_sequence_plan_outputs_stable_order(tmp_path: Path) -> None:
    snapshot, plan = _build_snapshot_and_plan(tmp_path)

    second_manifest = build_sequence_plan(
        SequencePlanSpec(
            snapshot_path=str(tmp_path / "snapshot"),
            output_dir=str(tmp_path / "plan-second"),
            sequence_length=4,
            stride=2,
            shuffle=True,
            shuffle_seed=123,
        )
    )
    second_plan = SequencePlan.open(tmp_path / "plan-second")

    expected = ((snapshot.total_tokens - 5) // 2) + 1
    assert plan.manifest.num_samples == expected
    assert second_manifest.num_samples == expected
    assert second_plan.sample_order.tolist() == plan.sample_order.tolist()


def test_causal_lm_sequence_dataset_returns_shifted_pairs(tmp_path: Path) -> None:
    snapshot, plan = _build_snapshot_and_plan(tmp_path)
    dataset = CausalLMSequenceDataset(str(tmp_path / "snapshot"), str(tmp_path / "plan"))

    x, y = dataset[0]
    start = plan.sample_start(0)
    expected = snapshot.read_window(start, 5)

    assert torch.equal(x, torch.tensor(expected[:-1], dtype=torch.long))
    assert torch.equal(y, torch.tensor(expected[1:], dtype=torch.long))
    assert len(dataset) == plan.manifest.num_samples


def test_deterministic_batch_loader_resume(tmp_path: Path) -> None:
    _snapshot, _plan = _build_snapshot_and_plan(tmp_path)
    dataset = CausalLMSequenceDataset(str(tmp_path / "snapshot"), str(tmp_path / "plan"))

    loader = DeterministicBatchLoader(dataset, global_batch_size=4)
    first_batch = loader.next()
    saved_state = loader.state_dict()
    second_batch = loader.next()

    resumed = DeterministicBatchLoader(dataset, global_batch_size=4)
    resumed.load_state_dict(saved_state)
    resumed_second_batch = resumed.next()

    assert first_batch[0].shape[0] == 4
    assert torch.equal(second_batch[0], resumed_second_batch[0])
    assert torch.equal(second_batch[1], resumed_second_batch[1])


def test_deterministic_batch_loader_world_slicing(tmp_path: Path) -> None:
    _snapshot, _plan = _build_snapshot_and_plan(tmp_path)
    dataset = CausalLMSequenceDataset(str(tmp_path / "snapshot"), str(tmp_path / "plan"))

    loader_rank0 = DeterministicBatchLoader(dataset, global_batch_size=4, world_size=2, rank=0)
    loader_rank1 = DeterministicBatchLoader(dataset, global_batch_size=4, world_size=2, rank=1)

    x0, y0 = loader_rank0.next()
    x1, y1 = loader_rank1.next()

    expected = [dataset[i] for i in range(4)]
    expected_x = torch.stack([sample[0] for sample in expected])
    expected_y = torch.stack([sample[1] for sample in expected])

    combined_x = torch.cat([x0, x1], dim=0)
    combined_y = torch.cat([y0, y1], dim=0)

    assert torch.equal(combined_x, expected_x)
    assert torch.equal(combined_y, expected_y)
