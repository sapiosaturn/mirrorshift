from pathlib import Path
import sys
from textwrap import dedent

import pytest

from mirrorshift.config import (
    CheckpointConfig,
    ConfigManager,
    DataConfig,
    ModelConfig,
    Run,
    TrainingConfig,
    validate_checkpoint_config,
    validate_data_config,
    validate_model_config,
    validate_run_config,
    validate_training_config,
)
from mirrorshift.utils import get_lr_schedule


def test_config_manager_uses_default_toml_file() -> None:
    config = ConfigManager().parse_args([])
    assert config.model.vocab_size == 50281
    assert config.training.max_steps == 500


def test_config_manager_reads_current_sys_argv(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "argv.toml"
    config_path.write_text(
        dedent(
            """
            [training]
            max_steps = 9
            """
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mirrorshift-train", f"--job.config_file={config_path}"],
    )

    config = ConfigManager().parse_args()

    assert config.training.max_steps == 9


def test_config_manager_cli_overrides_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        dedent(
            """
            [training]
            max_steps = 40
            log_every = 8
            """
        )
    )
    config = ConfigManager().parse_args(
        [
            f"--job.config_file={config_path}",
            "--training.max_steps=7",
        ]
    )
    assert config.training.max_steps == 7
    assert config.training.log_every == 8


def test_config_manager_rejects_invalid_toml_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        dedent(
            """
            [model]
            fake_field = 1
            """
        )
    )
    with pytest.raises(ValueError, match="Invalid field names"):
        ConfigManager().parse_args([f"--job.config_file={config_path}"])


def test_validate_model_config_divisibility_guard() -> None:
    bad = ModelConfig(
        vocab_size=32,
        num_layers=1,
        num_kv_heads=3,
        embedding_dim=30,
        num_heads=4,
        context_length=8,
        feedforward_dim=64,
        attention_type="gqa",
    )
    with pytest.raises(ValueError, match="divisible"):
        validate_model_config(bad)


def test_validate_model_config_mla_requires_fields() -> None:
    bad = ModelConfig(attention_type="mla", q_lora_rank=None)
    with pytest.raises(ValueError, match="missing"):
        validate_model_config(bad)


def test_validate_training_config_invalid_device() -> None:
    config = TrainingConfig(
        device="metal",  # type: ignore[arg-type]
        batch_size=4,
        learning_rate=1e-3,
        lr_warmup_steps=0,
        lr_schedule="linear_warmup",
        max_steps=10,
        compile=False,
        log_every=2,
    )
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        validate_training_config(config)


def test_validate_run_config_empty_log_dir() -> None:
    with pytest.raises(ValueError, match="run.log_dir must be non-empty"):
        validate_run_config(Run(log_dir=""))


def test_validate_run_config_rejects_non_parquet_dataset() -> None:
    with pytest.raises(ValueError, match="parquet file"):
        validate_run_config(Run(dataset="mirrorshift/datasets/example_train.txt"))


def test_validate_run_config_empty_run_id_when_set() -> None:
    with pytest.raises(ValueError, match="run.id must be non-empty when provided"):
        validate_run_config(Run(id=""))


def test_validate_run_config_empty_wandb_project() -> None:
    with pytest.raises(ValueError, match="run.wandb_project must be non-empty"):
        validate_run_config(Run(wandb_project=""))


def test_validate_checkpoint_config_negative_keep_latest_k() -> None:
    with pytest.raises(ValueError, match="checkpoint.keep_latest_k must be >= 0"):
        validate_checkpoint_config(CheckpointConfig(keep_latest_k=-1))


def test_validate_checkpoint_config_requires_enable_for_load_step() -> None:
    with pytest.raises(ValueError, match="checkpoint.enable must be true"):
        validate_checkpoint_config(CheckpointConfig(load_step=-1))


def test_validate_data_config_rejects_non_positive_max_documents() -> None:
    with pytest.raises(ValueError, match="data.max_documents must be > 0"):
        validate_data_config(DataConfig(max_documents=0))


def test_config_manager_parses_data_toml_and_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "data.toml"
    config_path.write_text(
        dedent(
            """
            [data]
            text_column = "body"
            max_tokens_per_shard = 4096
            """
        )
    )
    config = ConfigManager().parse_args(
        [
            f"--job.config_file={config_path}",
            "--data.max_documents=16",
        ]
    )

    assert config.data.text_column == "body"
    assert config.data.max_tokens_per_shard == 4096
    assert config.data.max_documents == 16


def test_config_manager_parses_checkpoint_toml_and_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "checkpoint.toml"
    config_path.write_text(
        dedent(
            """
            [checkpoint]
            enable = true
            interval = 20
            keep_latest_k = 3
            """
        )
    )
    config = ConfigManager().parse_args(
        [
            f"--job.config_file={config_path}",
            "--checkpoint.load_step=-1",
        ]
    )

    assert config.checkpoint.enable is True
    assert config.checkpoint.interval == 20
    assert config.checkpoint.keep_latest_k == 3
    assert config.checkpoint.load_step == -1


def test_get_lr_schedule_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown schedule name"):
        get_lr_schedule(
            schedule="does_not_exist",  # type: ignore[arg-type]
            max_lr=1e-3,
            warmup_steps=0,
            total_steps=10,
        )


def test_get_lr_schedule_returns_positive_values() -> None:
    schedule = get_lr_schedule(
        schedule="wsd_cosine",
        max_lr=1e-3,
        warmup_steps=2,
        total_steps=10,
    )
    values = [schedule(step) for step in range(10)]
    assert all(value >= 0 for value in values)
    assert max(values) <= 1e-3
