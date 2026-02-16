"""Metrics logging backends."""

from pathlib import Path
from typing import Protocol

from mirrorshift.config import JobConfig


class MetricsLogger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None:
        ...

    def close(self) -> None:
        ...


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

