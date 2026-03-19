import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

from mirrordata import (
    CausalLMSequenceDataset,
    DeterministicBatchLoader,
)
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.config import JobConfig, ModelConfig
from mirrorshift.infra import RuntimeContext

from .spec import TrainDataBundle, TrainSpec

_FAKE_DATASET_SIZE = 1_000_000


class FakeTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, *, vocab_size: int, sequence_length: int, dataset_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.dataset_size = int(dataset_size)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(index)
        tokens = torch.randint(
            0,
            self.vocab_size,
            (self.sequence_length + 1,),
            generator=generator,
            dtype=torch.long,
        )
        return tokens[:-1], tokens[1:]

    def get_vocab_size(self) -> int:
        return self.vocab_size


def build_default_model(model_config: ModelConfig) -> torch.nn.Module:
    return CausalTransformer(model_config=model_config)


def build_default_data(
    config: JobConfig, run_dir, runtime_context: RuntimeContext
) -> TrainDataBundle:
    _ = run_dir
    if config.data.use_fake_data:
        train_dataset = FakeTokenDataset(
            vocab_size=config.model.vocab_size,
            sequence_length=config.model.context_length,
            dataset_size=_FAKE_DATASET_SIZE,
        )
        train_loader = DeterministicBatchLoader(
            train_dataset,
            global_batch_size=config.training.batch_size,
            world_size=runtime_context.batch_world_size,
            rank=runtime_context.batch_rank,
            device=runtime_context.device,
            drop_last=True,
            wrap=True,
        )
        return TrainDataBundle(
            train_loader=train_loader,
            dataset_size=len(train_dataset),
            vocab_size=train_dataset.get_vocab_size(),
            data_identity={
                "kind": "fake-data",
                "sequence_length": config.model.context_length,
                "vocab_size": config.model.vocab_size,
            },
            exact_resume=True,
        )

    snapshot_path = Path(config.data.snapshot_path)
    plan_path = Path(config.data.plan_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan path does not exist: {plan_path}")
    train_dataset = CausalLMSequenceDataset(
        str(snapshot_path),
        str(plan_path),
    )
    if train_dataset.sequence_length != config.model.context_length:
        raise ValueError(
            "Plan sequence length does not match model context length: "
            f"{train_dataset.sequence_length} vs {config.model.context_length}"
        )
    train_loader = DeterministicBatchLoader(
        train_dataset,
        global_batch_size=config.training.batch_size,
        world_size=runtime_context.batch_world_size,
        rank=runtime_context.batch_rank,
        device=runtime_context.device,
        drop_last=True,
        wrap=True,
    )
    return TrainDataBundle(
        train_loader=train_loader,
        dataset_size=len(train_dataset),
        vocab_size=train_dataset.get_vocab_size(),
        data_identity=train_dataset.identity().to_dict(),
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
