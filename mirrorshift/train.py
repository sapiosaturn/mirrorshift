"""Step-based training entrypoint."""

import logging
import math
import time
from collections import deque
from typing import Callable, Iterator, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from mirrorshift.experiments import get_train_spec
from mirrorshift.config import (
    ConfigManager,
    JobConfig,
    TrainingConfig,
)
from mirrorshift.metrics import MetricsLogger, build_metrics_logger
from mirrorshift.run_manifest import create_run_artifacts, write_run_manifest
from mirrorshift.utils import get_lr_schedule

BatchType = Tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

LOGGER = logging.getLogger("mirrorshift.train")


def resolve_device(device_name: str) -> str:
    if device_name == "cpu":
        return "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        torch.set_float32_matmul_precision("high")
        return "cuda"
    raise ValueError("training.device must be 'cpu' or 'cuda'")


def iter_batches(train_loader: DataLoader) -> Iterator[BatchType]:
    while True:
        for batch in train_loader:
            yield batch


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    opt: optim.AdamW,
    loss_fn: LossFunction,
    device: str,
    training_config: TrainingConfig,
    metrics_logger: MetricsLogger,
) -> int:
    lr_schedule = get_lr_schedule(
        schedule=training_config.lr_schedule,
        max_lr=training_config.learning_rate,
        warmup_steps=training_config.lr_warmup_steps,
        total_steps=training_config.max_steps,
    )
    step_times = deque(maxlen=100)
    batch_iter = iter_batches(train_loader)

    model.train()
    for global_step in range(1, training_config.max_steps + 1):
        step_start_time = time.time()

        x, y = next(batch_iter)
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        lr = lr_schedule(global_step - 1)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        logits = model(x)
        loss_value = loss_fn(logits, y)
        loss_value.backward()
        opt.step()

        loss_scalar = loss_value.item()
        perplexity = math.exp(loss_scalar)
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)
        steps_per_second = 1.0 / avg_step_time

        metrics_logger.log(
            {
                "train/loss": loss_scalar,
                "train/perplexity": perplexity,
                "train/lr": lr,
                "perf/seconds_per_step": avg_step_time,
                "perf/steps_per_second": steps_per_second,
            },
            step=global_step,
        )

        if global_step == 1 or global_step % training_config.log_every == 0:
            LOGGER.info(
                "step=%d/%d loss=%.5f ppl=%.5f lr=%.2e sec_per_step=%.4f",
                global_step,
                training_config.max_steps,
                loss_scalar,
                perplexity,
                lr,
                avg_step_time,
            )
    return training_config.max_steps


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config: JobConfig = ConfigManager().parse_args()
    config.maybe_log(LOGGER)
    artifacts = create_run_artifacts(config)
    LOGGER.info("run_id=%s run_dir=%s", artifacts.run_id, artifacts.run_dir)

    train_spec = get_train_spec(config.run.spec)
    train_dataset = train_spec.build_dataset(config.run.dataset, config.model.context_length)

    if train_dataset.get_vocab_size() != config.model.vocab_size:
        raise ValueError(
            "Dataset vocab size does not match model config vocab size: "
            f"{train_dataset.get_vocab_size()} vs {config.model.vocab_size}"
        )
    if len(train_dataset) < config.training.batch_size:
        raise ValueError(
            "Dataset must have at least batch_size sequences. "
            f"len(dataset)={len(train_dataset)} batch_size={config.training.batch_size}"
        )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=RandomSampler(train_dataset),
    )
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty for the provided configuration")

    model = train_spec.build_model(config.model)
    device = resolve_device(config.training.device)
    model = model.to(device)
    if config.training.compile:
        model = torch.compile(model)

    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    metrics_logger = build_metrics_logger(
        config=config,
        run_id=artifacts.run_id,
        run_dir=artifacts.run_dir,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_run_manifest(
        artifacts=artifacts,
        config=config,
        dataset_size=len(train_dataset),
        trainable_params=trainable_params,
        device=device,
    )

    LOGGER.info(
        "dataset_size=%d trainable_params=%d device=%s max_steps=%d",
        len(train_dataset),
        trainable_params,
        device,
        config.training.max_steps,
    )
    LOGGER.info(
        "config_snapshot=%s run_manifest=%s",
        artifacts.config_snapshot_path,
        artifacts.manifest_path,
    )

    try:
        final_step = train(
            model=model,
            train_loader=train_loader,
            opt=opt,
            loss_fn=train_spec.loss_fn,
            device=device,
            training_config=config.training,
            metrics_logger=metrics_logger,
        )
    finally:
        metrics_logger.close()

    LOGGER.info("Training complete at step=%d", final_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
