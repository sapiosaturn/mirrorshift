import json
import sys
from pathlib import Path
from textwrap import dedent

import pyarrow as pa
import pyarrow.parquet as pq

from mirrordata import (
    ParquetSnapshotConfig,
    SequencePlanSpec,
    build_sequence_plan,
    build_snapshot_from_parquet,
)
from mirrorshift.train import main


def _write_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path, row_group_size=2)


def _build_snapshot_and_plan(parquet_path: Path, tmp_path: Path) -> tuple[Path, Path]:
    snapshot_dir = tmp_path / "snapshot"
    plan_dir = tmp_path / "plan"
    build_snapshot_from_parquet(
        ParquetSnapshotConfig(
            input_paths=(str(parquet_path),),
            output_dir=str(snapshot_dir),
            snapshot_id="integration-snapshot",
            dataset_name="integration",
            split="train",
            max_tokens_per_shard=256,
            max_documents=4,
        )
    )
    build_sequence_plan(
        SequencePlanSpec(
            snapshot_path=str(snapshot_dir),
            output_dir=str(plan_dir),
            sequence_length=8,
            shuffle=True,
            shuffle_seed=7,
        )
    )
    return snapshot_dir, plan_dir


def test_main_trains_from_prebuilt_mirrordata_artifacts(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "train.parquet"
    _write_parquet(
        parquet_path,
        [
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega alpha beta",
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod",
            "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim",
        ],
    )
    snapshot_dir, plan_dir = _build_snapshot_and_plan(parquet_path, tmp_path)

    config_path = tmp_path / "train.toml"
    run_dir = tmp_path / "runs"
    config_path.write_text(
        dedent(
            f"""
            [run]
            spec = "causal_lm"
            log_dir = "{run_dir}"
            id = "parquet-smoke"
            wandb_mode = "disabled"

            [model]
            attention_type = "gqa"
            vocab_size = 50281
            num_layers = 1
            embedding_dim = 32
            num_heads = 4
            num_kv_heads = 2
            context_length = 8
            feedforward_dim = 64

            [training]
            device = "cpu"
            batch_size = 2
            learning_rate = 0.001
            lr_warmup_steps = 1
            lr_schedule = "linear_warmup"
            max_steps = 2
            log_every = 1

            [compile]
            enable = false

            [data]
            snapshot_path = "{snapshot_dir}"
            plan_path = "{plan_dir}"
            """
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["mirrorshift-train", f"--job.config_file={config_path}"],
    )

    assert main() == 0

    output_run_dir = run_dir / "parquet-smoke"
    run_manifest = output_run_dir / "manifest.json"

    assert run_manifest.exists()
    assert not (output_run_dir / "data").exists()

    manifest_payload = json.loads(run_manifest.read_text())
    assert manifest_payload["data_snapshot_path"] == str(snapshot_dir)
    assert manifest_payload["data_plan_path"] == str(plan_dir)
    assert manifest_payload["dataset_size"] > 0
