from pathlib import Path
from textwrap import dedent

import pytest

from mirrorshift.config import (
    ConfigManager,
    ModelConfig,
    TrainingConfig,
    validate_model_config,
    validate_training_config,
)
from mirrorshift.utils import get_lr_schedule


def test_config_manager_uses_default_toml_file() -> None:
    config = ConfigManager().parse_args([])
    assert config.model.vocab_size == 50281
    assert config.training.max_steps == 500


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
