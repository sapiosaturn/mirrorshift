from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset

from mirrorshift.utils import ModelConfig

ModelBuilder = Callable[[ModelConfig], torch.nn.Module]
DatasetBuilder = Callable[[str, int], Dataset]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainSpec:
    name: str
    build_model: ModelBuilder
    build_dataset: DatasetBuilder
    loss_fn: LossFunction
