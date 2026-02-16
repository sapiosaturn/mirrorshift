import torch
import torch.nn.functional as F

from mirrorshift.data import TiktokenTxtDataset
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.utils import ModelConfig

from .spec import TrainSpec


def build_default_model(model_config: ModelConfig) -> torch.nn.Module:
    return CausalTransformer(model_config=model_config)


def build_default_dataset(dataset_path: str, context_length: int) -> TiktokenTxtDataset:
    return TiktokenTxtDataset(dataset_path, sequence_length=context_length)


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


DEFAULT_TRAIN_SPEC = TrainSpec(
    name="causal_lm",
    build_model=build_default_model,
    build_dataset=build_default_dataset,
    loss_fn=causal_lm_loss,
)
