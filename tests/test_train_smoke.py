import torch
import pytest
from torch.utils.data import DataLoader, Dataset, RandomSampler

from mirrordata import DeterministicBatchLoader
from mirrorshift.config import ModelConfig, TrainingConfig
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.train import resolve_device, train


class TinyTokenDataset(Dataset):
    def __init__(self, vocab_size: int = 32, sequence_length: int = 8, token_count: int = 512):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.tokens = torch.arange(token_count, dtype=torch.long) % vocab_size

    def __len__(self) -> int:
        return len(self.tokens) - self.sequence_length - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[index : index + self.sequence_length]
        y = self.tokens[index + 1 : index + self.sequence_length + 1]
        return x, y

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(str(token) for token in token_ids)


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        num_layers=2,
        num_kv_heads=2,
        embedding_dim=32,
        num_heads=4,
        context_length=8,
        feedforward_dim=64,
        attention_type="gqa",
    )


def tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        device="cpu",
        batch_size=4,
        learning_rate=1e-3,
        lr_warmup_steps=1,
        lr_schedule="linear_warmup",
        max_steps=3,
        log_every=1,
    )


def test_resolve_device_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        resolve_device("unknown")


def test_train_cpu_smoke_single_digit_steps(tmp_path) -> None:
    dataset = TinyTokenDataset()
    model_config = tiny_model_config()
    training_config = tiny_training_config()
    model = CausalTransformer(model_config)
    initial_weight = model.lm_head.weight.detach().clone()

    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=RandomSampler(dataset),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    class DummyMetricsLogger:
        def __init__(self):
            self.logged_steps: list[int] = []

        def log(self, metrics: dict[str, float], step: int) -> None:
            assert set(metrics) == {
                "memory/max_active_gib",
                "memory/max_reserved_gib",
                "optimizer/lr",
                "throughput/tflops",
                "throughput/tokens_per_second_per_gpu",
                "timing/data_loading_seconds",
                "timing/end_to_end_seconds",
                "train/grad_norm",
                "train/loss",
                "train/max_loss",
                "train/n_tokens_seen",
            }
            self.logged_steps.append(step)

        def close(self) -> None:
            return None

    metrics_logger = DummyMetricsLogger()

    final_step = train(
        model=model,
        train_loader=loader,
        opt=opt,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=training_config,
        metrics_logger=metrics_logger,
    )

    assert final_step == training_config.max_steps
    assert metrics_logger.logged_steps[-1] == training_config.max_steps
    assert not torch.equal(initial_weight, model.lm_head.weight.detach())


def test_train_cpu_smoke_with_mirrordata_batch_loader() -> None:
    dataset = TinyTokenDataset()
    model_config = tiny_model_config()
    training_config = tiny_training_config()
    model = CausalTransformer(model_config)

    loader = DeterministicBatchLoader(
        dataset,
        global_batch_size=training_config.batch_size,
        device="cpu",
        wrap=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    class DummyMetricsLogger:
        def __init__(self):
            self.logged_steps: list[int] = []

        def log(self, metrics: dict[str, float], step: int) -> None:
            assert set(metrics) == {
                "memory/max_active_gib",
                "memory/max_reserved_gib",
                "optimizer/lr",
                "throughput/tflops",
                "throughput/tokens_per_second_per_gpu",
                "timing/data_loading_seconds",
                "timing/end_to_end_seconds",
                "train/grad_norm",
                "train/loss",
                "train/max_loss",
                "train/n_tokens_seen",
            }
            self.logged_steps.append(step)

        def close(self) -> None:
            return None

    metrics_logger = DummyMetricsLogger()

    final_step = train(
        model=model,
        train_loader=loader,
        opt=opt,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=training_config,
        metrics_logger=metrics_logger,
    )

    assert final_step == training_config.max_steps
    assert metrics_logger.logged_steps[-1] == training_config.max_steps
