"""Scheduling utilities."""

import math
from typing import Callable

from mirrorshift.config import ScheduleName


def get_lr_schedule(
    schedule: ScheduleName,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    decay_start_factor: float = 0.8,
    start_factor: float = 0.5,
) -> Callable[[int], float]:
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if max_lr <= 0:
        raise ValueError("max_lr must be > 0")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    decay_start_step = int(total_steps * decay_start_factor)
    if schedule == "linear_warmup":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            return max_lr

    elif schedule == "wsd_exponential":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            if step < decay_start_step:
                return max_lr
            progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
            decay = math.exp(-5 * progress)
            return max_lr * decay

    elif schedule == "wsd_linear":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            if step < decay_start_step:
                return max_lr
            progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
            return max_lr * (1 - progress)

    elif schedule == "wsd_cosine":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            if step < decay_start_step:
                return max_lr
            progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
            return max_lr * (1 + math.cos(math.pi * progress)) / 2

    else:
        raise ValueError(f"Unknown schedule name: {schedule}")
    return lr_schedule


__all__ = [
    "get_lr_schedule",
]
