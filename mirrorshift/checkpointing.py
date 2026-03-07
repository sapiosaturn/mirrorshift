"""Minimal DCP checkpoint helpers for single-process training."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer

from mirrorshift.config.job_config import CheckpointConfig

LOGGER = logging.getLogger("mirrorshift.checkpointing")


@dataclass
class TrainState:
    step: int = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"step": torch.tensor(self.step, dtype=torch.int64)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.step = int(state_dict["step"].item())


class CheckpointManager:
    """Minimal synchronous DCP checkpoint manager."""

    def __init__(
        self,
        config: CheckpointConfig,
        *,
        run_dir: Path,
        model: torch.nn.Module,
        optimizer: Optimizer,
        train_state: TrainState,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.model = model
        self.optimizer = optimizer
        self.train_state = train_state
        self.checkpoint_dir = run_dir / config.folder

    def load(self) -> bool:
        if self.config.load_step is None:
            return False

        step = (
            self._find_latest_step()
            if self.config.load_step == -1
            else self.config.load_step
        )
        if step is None:
            raise FileNotFoundError(
                f"No checkpoints found in {self.checkpoint_dir} for requested load"
            )

        checkpoint_id = self._checkpoint_path(step)
        if not checkpoint_id.is_dir():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_id}")

        payload = self._load_payload()
        LOGGER.info("Loading checkpoint from %s", checkpoint_id)
        dcp.load(payload, checkpoint_id=str(checkpoint_id))
        self.model.load_state_dict(payload["model"])
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=payload["optimizer"],
        )
        self.train_state.load_state_dict(payload["train_state"])
        LOGGER.info("Loaded checkpoint at step=%d", self.train_state.step)
        return True

    def save(self, step: int, *, last_step: bool = False) -> None:
        if not self.config.enable or not self._should_save(step, last_step=last_step):
            return

        checkpoint_id = self._checkpoint_path(step)
        if checkpoint_id.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_id}")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving checkpoint to %s", checkpoint_id)
        dcp.save(self._save_payload(), checkpoint_id=str(checkpoint_id))
        self._purge_stale_checkpoints()

    def _save_payload(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": get_optimizer_state_dict(self.model, self.optimizer),
            "train_state": self.train_state.state_dict(),
        }

    def _load_payload(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": get_optimizer_state_dict(self.model, self.optimizer),
            "train_state": self.train_state.state_dict(),
        }

    def _should_save(self, step: int, *, last_step: bool) -> bool:
        return last_step or step % self.config.interval == 0

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"step-{step}"

    def _find_latest_step(self) -> int | None:
        candidates = self._list_checkpoint_steps()
        if not candidates:
            return None
        return candidates[-1]

    def _list_checkpoint_steps(self) -> list[int]:
        if not self.checkpoint_dir.is_dir():
            return []

        steps: list[int] = []
        for path in self.checkpoint_dir.iterdir():
            match = re.fullmatch(r"step-(\d+)", path.name)
            if match and path.is_dir() and (path / ".metadata").is_file():
                steps.append(int(match.group(1)))
        return sorted(steps)

    def _purge_stale_checkpoints(self) -> None:
        keep_latest_k = self.config.keep_latest_k
        if keep_latest_k == 0:
            return

        steps = self._list_checkpoint_steps()
        to_delete = steps[:-keep_latest_k]
        for step in to_delete:
            path = self._checkpoint_path(step)
            LOGGER.info("Removing stale checkpoint %s", path)
            shutil.rmtree(path, ignore_errors=True)
