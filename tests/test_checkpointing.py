from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from mirrordata import DeterministicBatchLoader
from mirrorshift.checkpointing import CheckpointManager, TrainState
from mirrorshift.config import CheckpointConfig, ModelConfig, TrainingConfig
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.train import train


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


class DummyMetricsLogger:
    def log(self, metrics: dict[str, float], step: int) -> None:
        assert "train/loss" in metrics
        assert step > 0

    def close(self) -> None:
        return None


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


def tiny_training_config(max_steps: int) -> TrainingConfig:
    return TrainingConfig(
        device="cpu",
        batch_size=4,
        learning_rate=1e-3,
        lr_warmup_steps=1,
        lr_schedule="linear_warmup",
        max_steps=max_steps,
        log_every=1,
    )


def build_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset),
    )


def build_deterministic_loader(dataset: Dataset, batch_size: int) -> DeterministicBatchLoader:
    return DeterministicBatchLoader(
        dataset,
        global_batch_size=batch_size,
        device="cpu",
        drop_last=True,
        wrap=False,
    )


def run_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataset: TinyTokenDataset,
) -> None:
    loader = build_loader(dataset, batch_size=4)
    x, y = next(iter(loader))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def assert_optimizer_state_equal(
    expected: dict[str, object],
    actual: dict[str, object],
) -> None:
    assert expected["param_groups"] == actual["param_groups"]
    expected_state = expected["state"]
    actual_state = actual["state"]
    assert expected_state.keys() == actual_state.keys()
    for param_id, expected_values in expected_state.items():
        actual_values = actual_state[param_id]
        assert expected_values.keys() == actual_values.keys()
        for key, expected_value in expected_values.items():
            actual_value = actual_values[key]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(expected_value, actual_value)
            else:
                assert expected_value == actual_value


def test_checkpoint_manager_round_trip_restores_model_optimizer_and_step(tmp_path) -> None:
    run_dir = tmp_path / "round-trip"
    run_dir.mkdir()

    dataset = TinyTokenDataset()
    model = CausalTransformer(tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_optimizer_step(model, optimizer, dataset)

    saved_model_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    saved_optimizer_state = optimizer.state_dict()

    train_state = TrainState(step=4)
    manager = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1),
        run_dir=run_dir,
        model=model,
        optimizer=optimizer,
        train_state=train_state,
    )
    manager.save(step=4, last_step=True)

    restored_model = CausalTransformer(tiny_model_config())
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_state = TrainState()
    restored_manager = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, load_step=-1),
        run_dir=run_dir,
        model=restored_model,
        optimizer=restored_optimizer,
        train_state=restored_state,
    )

    assert restored_manager.load() is True
    assert restored_state.step == 4
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, saved_model_state[key])
    assert_optimizer_state_equal(saved_optimizer_state, restored_optimizer.state_dict())


def test_checkpoint_manager_keep_latest_k_prunes_old_step_directories(tmp_path) -> None:
    run_dir = tmp_path / "retention"
    run_dir.mkdir()

    model = CausalTransformer(tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_state = TrainState()
    manager = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, keep_latest_k=2),
        run_dir=run_dir,
        model=model,
        optimizer=optimizer,
        train_state=train_state,
    )

    for step in range(1, 5):
        train_state.step = step
        manager.save(step=step)

    checkpoint_dir = run_dir / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["step-3", "step-4"]


def test_train_resumes_from_latest_checkpoint_on_cpu(tmp_path) -> None:
    dataset = TinyTokenDataset()
    model_config = tiny_model_config()
    initial_training_config = tiny_training_config(max_steps=2)
    resumed_training_config = replace(initial_training_config, max_steps=4)
    run_dir = tmp_path / "resume-run"
    run_dir.mkdir()

    first_model = CausalTransformer(model_config)
    first_optimizer = torch.optim.AdamW(
        first_model.parameters(),
        lr=initial_training_config.learning_rate,
    )
    first_state = TrainState()
    first_checkpointer = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, keep_latest_k=2),
        run_dir=run_dir,
        model=first_model,
        optimizer=first_optimizer,
        train_state=first_state,
    )

    final_step = train(
        model=first_model,
        train_loader=build_loader(dataset, batch_size=initial_training_config.batch_size),
        opt=first_optimizer,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=initial_training_config,
        metrics_logger=DummyMetricsLogger(),
        train_state=first_state,
        checkpointer=first_checkpointer,
    )
    assert final_step == 2

    resumed_model = CausalTransformer(model_config)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=resumed_training_config.learning_rate,
    )
    resumed_state = TrainState()
    resumed_checkpointer = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, keep_latest_k=2, load_step=-1),
        run_dir=run_dir,
        model=resumed_model,
        optimizer=resumed_optimizer,
        train_state=resumed_state,
    )

    assert resumed_checkpointer.load() is True
    assert resumed_state.step == 2

    final_resumed_step = train(
        model=resumed_model,
        train_loader=build_loader(dataset, batch_size=resumed_training_config.batch_size),
        opt=resumed_optimizer,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=resumed_training_config,
        metrics_logger=DummyMetricsLogger(),
        train_state=resumed_state,
        checkpointer=resumed_checkpointer,
    )

    assert final_resumed_step == 4
    assert resumed_state.step == 4
    checkpoint_dir = run_dir / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["step-3", "step-4"]


def test_train_returns_existing_step_when_already_complete() -> None:
    dataset = TinyTokenDataset()
    training_config = tiny_training_config(max_steps=2)
    train_state = TrainState(step=2)
    model = CausalTransformer(tiny_model_config())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    final_step = train(
        model=model,
        train_loader=build_loader(dataset, batch_size=training_config.batch_size),
        opt=optimizer,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=training_config,
        metrics_logger=DummyMetricsLogger(),
        train_state=train_state,
    )

    assert final_step == 2


def test_checkpoint_manager_restores_loader_state_for_exact_resume(tmp_path) -> None:
    dataset = TinyTokenDataset(token_count=2048)
    model_config = tiny_model_config()
    identity = {
        "snapshot_id": "tiny",
        "snapshot_fingerprint": "snapshot-fp",
        "plan_fingerprint": "plan-fp",
        "sequence_length": model_config.context_length,
        "dataset_size": len(dataset),
    }
    initial_state_dict = CausalTransformer(model_config).state_dict()
    full_training_config = tiny_training_config(max_steps=4)
    split_training_config = tiny_training_config(max_steps=2)
    resumed_training_config = tiny_training_config(max_steps=4)

    full_model = CausalTransformer(model_config)
    full_model.load_state_dict(initial_state_dict)
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=full_training_config.learning_rate)
    full_loader = build_deterministic_loader(dataset, batch_size=full_training_config.batch_size)
    train(
        model=full_model,
        train_loader=full_loader,
        opt=full_optimizer,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=full_training_config,
        metrics_logger=DummyMetricsLogger(),
    )

    run_dir = tmp_path / "resume-exact"
    run_dir.mkdir()
    split_model = CausalTransformer(model_config)
    split_model.load_state_dict(initial_state_dict)
    split_optimizer = torch.optim.AdamW(split_model.parameters(), lr=split_training_config.learning_rate)
    split_state = TrainState()
    split_loader = build_deterministic_loader(dataset, batch_size=split_training_config.batch_size)
    split_checkpointer = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, keep_latest_k=2),
        run_dir=run_dir,
        model=split_model,
        optimizer=split_optimizer,
        train_state=split_state,
        train_loader=split_loader,
        data_identity=identity,
    )
    train(
        model=split_model,
        train_loader=split_loader,
        opt=split_optimizer,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=split_training_config,
        metrics_logger=DummyMetricsLogger(),
        train_state=split_state,
        checkpointer=split_checkpointer,
    )

    resumed_model = CausalTransformer(model_config)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=resumed_training_config.learning_rate,
    )
    resumed_state = TrainState()
    resumed_loader = build_deterministic_loader(dataset, batch_size=resumed_training_config.batch_size)
    resumed_checkpointer = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, keep_latest_k=2, load_step=-1),
        run_dir=run_dir,
        model=resumed_model,
        optimizer=resumed_optimizer,
        train_state=resumed_state,
        train_loader=resumed_loader,
        data_identity=identity,
    )

    assert resumed_checkpointer.load() is True
    assert resumed_checkpointer.restored_loader_state is True
    assert resumed_state.step == 2
    train(
        model=resumed_model,
        train_loader=resumed_loader,
        opt=resumed_optimizer,
        loss_fn=lambda logits, targets: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ),
        device="cpu",
        training_config=resumed_training_config,
        metrics_logger=DummyMetricsLogger(),
        train_state=resumed_state,
        checkpointer=resumed_checkpointer,
        resume_batch_offset=0,
    )

    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key])
    assert resumed_state.step == 4


def test_checkpoint_manager_rejects_data_identity_mismatch(tmp_path) -> None:
    dataset = TinyTokenDataset(token_count=512)
    run_dir = tmp_path / "identity-mismatch"
    run_dir.mkdir()
    model = CausalTransformer(tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_state = TrainState(step=1)
    loader = build_deterministic_loader(dataset, batch_size=4)
    manager = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1),
        run_dir=run_dir,
        model=model,
        optimizer=optimizer,
        train_state=train_state,
        train_loader=loader,
        data_identity={
            "snapshot_id": "tiny",
            "snapshot_fingerprint": "snapshot-fp",
            "plan_fingerprint": "plan-fp",
            "sequence_length": 8,
            "dataset_size": len(dataset),
        },
    )
    manager.save(step=1, last_step=True)

    restored_model = CausalTransformer(tiny_model_config())
    restored = CheckpointManager(
        config=CheckpointConfig(enable=True, interval=1, load_step=-1),
        run_dir=run_dir,
        model=restored_model,
        optimizer=torch.optim.AdamW(restored_model.parameters(), lr=1e-3),
        train_state=TrainState(),
        train_loader=build_deterministic_loader(dataset, batch_size=4),
        data_identity={
            "snapshot_id": "tiny",
            "snapshot_fingerprint": "snapshot-fp",
            "plan_fingerprint": "different-plan-fp",
            "sequence_length": 8,
            "dataset_size": len(dataset),
        },
    )

    with pytest.raises(RuntimeError, match="data identity mismatch"):
        restored.load()
