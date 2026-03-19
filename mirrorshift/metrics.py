"""Metrics logging backends."""

import os
from pathlib import Path
from typing import Protocol

from mirrorshift.config import JobConfig


class MetricsLogger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpLogger:
    def log(self, metrics: dict[str, float], step: int) -> None:
        return None

    def close(self) -> None:
        return None


class WandBLogger:
    def __init__(self, config: JobConfig, run_id: str, run_dir: Path):
        import wandb

        self._run = wandb.init(
            project=config.run.wandb_project,
            entity=config.run.wandb_entity,
            name=run_id,
            mode=config.run.wandb_mode,
            dir=str(run_dir),
            config=config.to_dict(),
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def close(self) -> None:
        self._run.finish()


def build_metrics_logger(
    config: JobConfig, run_id: str, run_dir: Path, *, is_primary: bool = True
) -> MetricsLogger:
    # Never emit W&B data during pytest runs, even if run.wandb_mode=online.
    if not is_primary:
        return NoOpLogger()
    if "PYTEST_CURRENT_TEST" in os.environ:
        return NoOpLogger()
    if config.run.wandb_mode == "disabled":
        return NoOpLogger()
    return WandBLogger(config=config, run_id=run_id, run_dir=run_dir)
