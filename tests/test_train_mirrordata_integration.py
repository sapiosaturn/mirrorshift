import json
import sys
from pathlib import Path
from textwrap import dedent

import pyarrow as pa
import pyarrow.parquet as pq

from mirrorshift.train import main


def _write_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path, row_group_size=2)


def test_main_trains_from_parquet_via_mirrordata(tmp_path, monkeypatch) -> None:
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

    config_path = tmp_path / "train.toml"
    run_dir = tmp_path / "runs"
    config_path.write_text(
        dedent(
            f"""
            [job]
            print_config = false

            [run]
            dataset = "{parquet_path}"
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
            compile = false
            log_every = 1

            [data]
            input_format = "parquet"
            max_documents = 4
            max_tokens_per_shard = 256
            shuffle = true
            shuffle_seed = 7
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
    snapshot_manifest = output_run_dir / "data" / "snapshot" / "manifest.json"
    plan_manifest = output_run_dir / "data" / "plan" / "plan.json"
    run_manifest = output_run_dir / "manifest.json"

    assert snapshot_manifest.exists()
    assert plan_manifest.exists()
    assert run_manifest.exists()

    manifest_payload = json.loads(run_manifest.read_text())
    assert manifest_payload["dataset"] == str(parquet_path)
    assert manifest_payload["dataset_size"] > 0
