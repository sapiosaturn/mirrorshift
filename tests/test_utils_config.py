import json
from pathlib import Path

import pytest

from mirrorshift.utils import (
    ModelConfig,
    TrainingConfig,
    get_lr_schedule,
    read_model_config,
    read_training_config,
    validate_model_config,
    validate_training_config,
)


def build_model_config(attention_type: str = "gqa") -> dict[str, object]:
    config = {
        "vocab_size": 128,
        "num_layers": 2,
        "num_kv_heads": 2,
        "embedding_dim": 32,
        "num_heads": 4,
        "context_length": 16,
        "feedforward_dim": 64,
        "attention_type": attention_type,
    }
    if attention_type == "mla":
        config.update(
            {
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 8,
                "v_head_dim": 8,
            }
        )
    return config


def build_training_config() -> dict[str, object]:
    return {
        "device": "cpu",
        "batch_size": 4,
        "learning_rate": 1e-3,
        "lr_warmup_steps": 2,
        "lr_schedule": "wsd_linear",
        "max_steps": 20,
        "compile": False,
        "log_every": 5,
    }


def test_read_model_config_gqa(tmp_path: Path) -> None:
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(build_model_config("gqa")))
    parsed = read_model_config(config_path)
    assert parsed.attention_type == "gqa"
    assert parsed.embedding_dim == 32


def test_read_model_config_mla_requires_extra_fields(
    tmp_path: Path,
) -> None:
    broken = build_model_config("mla")
    del broken["q_lora_rank"]
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="missing"):
        read_model_config(config_path)


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


def test_read_training_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(build_training_config()))
    parsed = read_training_config(config_path)
    assert parsed.max_steps == 20
    assert parsed.log_every == 5


def test_read_training_config_missing_key_raises(
    tmp_path: Path,
) -> None:
    broken = build_training_config()
    del broken["max_steps"]
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Missing configuration options"):
        read_training_config(config_path)


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
