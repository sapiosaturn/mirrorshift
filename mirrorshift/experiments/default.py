import torch
import torch.nn.functional as F

from mirrordata import (
    CausalLMSequenceDataset,
    DeterministicBatchLoader,
)
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.config import JobConfig, ModelConfig

from .spec import TrainDataBundle, TrainSpec


def build_default_model(model_config: ModelConfig) -> torch.nn.Module:
    return CausalTransformer(model_config=model_config)


def build_default_data(config: JobConfig, run_dir, device: str) -> TrainDataBundle:
    _ = run_dir
    train_dataset = CausalLMSequenceDataset(
        config.data.snapshot_path,
        config.data.plan_path,
    )
    if train_dataset.sequence_length != config.model.context_length:
        raise ValueError(
            "Plan sequence length does not match model context length: "
            f"{train_dataset.sequence_length} vs {config.model.context_length}"
        )
    train_loader = DeterministicBatchLoader(
        train_dataset,
        global_batch_size=config.training.batch_size,
        device=device,
        drop_last=True,
        wrap=True,
    )
    return TrainDataBundle(
        train_loader=train_loader,
        dataset_size=len(train_dataset),
        vocab_size=train_dataset.get_vocab_size(),
        exact_resume=True,
    )


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


DEFAULT_TRAIN_SPEC = TrainSpec(
    name="causal_lm",
    build_model=build_default_model,
    build_data=build_default_data,
    loss_fn=causal_lm_loss,
)
