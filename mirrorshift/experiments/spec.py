from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from mirrorshift.config import JobConfig, ModelConfig

ModelBuilder = Callable[[ModelConfig], torch.nn.Module]
DataBuilder = Callable[[JobConfig, Path, str], "TrainDataBundle"]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainDataBundle:
    train_loader: Any
    dataset_size: int
    vocab_size: int
    exact_resume: bool = False


@dataclass(frozen=True)
class TrainSpec:
    name: str
    build_model: ModelBuilder
    build_data: DataBuilder
    loss_fn: LossFunction
