import logging
from pathlib import Path
import sys
from textwrap import dedent

import pytest

from mirrorshift.config import (
    ActivationCheckpointConfig,
    CheckpointConfig,
    CompileConfig,
    ConfigManager,
    DataConfig,
    DebugConfig,
    ModelConfig,
    ParallelismConfig,
    Run,
    TrainingConfig,
    validate_activation_checkpoint_config,
    validate_checkpoint_config,
    validate_compile_config,
    validate_data_config,
    validate_debug_config,
    validate_model_config,
    validate_parallelism_config,
    validate_run_config,
    validate_training_config,
)
from mirrorshift.utils import get_lr_schedule


def test_config_manager_uses_default_toml_file() -> None:
    config = ConfigManager().parse_args([])
    assert config.model.vocab_size == 50281
    assert config.training.max_steps == 500
    assert config.debug.seed is None
    assert config.debug.deterministic is False


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
        max_grad_norm=None,
        lr_warmup_steps=0,
        lr_schedule="linear_warmup",
        max_steps=10,
        log_every=2,
    )
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        validate_training_config(config)


def test_validate_debug_config_negative_seed() -> None:
    with pytest.raises(ValueError, match="debug.seed must be >= 0"):
        validate_debug_config(DebugConfig(seed=-1))


def test_validate_training_config_rejects_non_positive_max_grad_norm() -> None:
    with pytest.raises(ValueError, match="training.max_grad_norm must be > 0"):
        validate_training_config(TrainingConfig(max_grad_norm=0.0))


def test_validate_parallelism_config_invalid_degree() -> None:
    with pytest.raises(ValueError, match="parallelism.dp_replicate must be > 0"):
        validate_parallelism_config(ParallelismConfig(dp_replicate=0))


def test_validate_activation_checkpoint_config_invalid_selective_option() -> None:
    with pytest.raises(ValueError, match="selective_ac_option"):
        validate_activation_checkpoint_config(
            ActivationCheckpointConfig(mode="selective", selective_ac_option="0")
        )


def test_validate_compile_config_empty_backend() -> None:
    with pytest.raises(ValueError, match="compile.backend must be non-empty"):
        validate_compile_config(CompileConfig(backend=""))


def test_validate_run_config_empty_log_dir() -> None:
    with pytest.raises(ValueError, match="run.log_dir must be non-empty"):
        validate_run_config(Run(log_dir=""))


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


def test_validate_data_config_rejects_empty_snapshot_path() -> None:
    with pytest.raises(ValueError, match="data.snapshot_path must be non-empty"):
        validate_data_config(DataConfig(snapshot_path=""))


def test_config_manager_parses_data_toml_and_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "data.toml"
    config_path.write_text(
        dedent(
            """
            [data]
            snapshot_path = "snapshot-root"
            """
        )
    )
    config = ConfigManager().parse_args(
        [
            f"--job.config_file={config_path}",
            "--data.plan_path=plan-root",
        ]
    )

    assert config.data.snapshot_path == "snapshot-root"
    assert config.data.plan_path == "plan-root"


def test_config_manager_supports_top_level_use_fake_data_flag() -> None:
    config = ConfigManager().parse_args(["--use-fake-data"])

    assert config.data.use_fake_data is True


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


def test_config_manager_parses_debug_toml_and_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "debug.toml"
    config_path.write_text(
        dedent(
            """
            [debug]
            seed = 7
            """
        )
    )
    config = ConfigManager().parse_args(
        [
            f"--job.config_file={config_path}",
            "--debug.deterministic",
        ]
    )

    assert config.debug.seed == 7
    assert config.debug.deterministic is True


def test_config_manager_parses_parallelism_compile_and_ac(tmp_path: Path) -> None:
    config_path = tmp_path / "infra.toml"
    config_path.write_text(
        dedent(
            """
            [parallelism]
            dp_replicate = 1
            dp_shard = 1

            [activation_checkpoint]
            mode = "selective"
            selective_ac_option = "2"

            [compile]
            backend = "eager"
            """
        )
    )
    config = ConfigManager().parse_args(
        [
            f"--job.config_file={config_path}",
            "--compile.enable",
        ]
    )

    assert config.parallelism.dp_replicate == 1
    assert config.parallelism.dp_shard == 1
    assert config.activation_checkpoint.mode == "selective"
    assert config.activation_checkpoint.selective_ac_option == "2"
    assert config.compile.enable is True
    assert config.compile.backend == "eager"


def test_job_config_logs_resolved_config_unconditionally(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("mirrorshift.config.test")
    config = ConfigManager().parse_args([])

    with caplog.at_level(logging.INFO, logger=logger.name):
        config.maybe_log(logger)

    assert "Resolved config" in caplog.text


def test_job_config_to_dict_excludes_fake_data_by_default() -> None:
    config = ConfigManager().parse_args(["--use-fake-data"])

    assert "use_fake_data" not in config.to_dict()["data"]
    assert config.to_dict(include_transient=True)["data"]["use_fake_data"] is True


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
