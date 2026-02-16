"""Step-based training entrypoint."""

import argparse
import json
import logging
import math
import time
from collections import deque
from typing import Callable, Iterator, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from mirrorshift.experiments import get_train_spec, list_train_specs
from mirrorshift.utils import (
    TrainingConfig,
    config_to_dict,
    get_lr_schedule,
    read_model_config,
    read_training_config,
)

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
    raise ValueError("training_config.device must be 'cpu' or 'cuda'")


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
    writer: SummaryWriter,
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

        writer.add_scalar("train/loss", loss_scalar, global_step)
        writer.add_scalar("train/perplexity", perplexity, global_step)
        writer.add_scalar("train/lr", lr, global_step)
        writer.add_scalar("perf/seconds_per_step", avg_step_time, global_step)
        writer.add_scalar("perf/steps_per_second", steps_per_second, global_step)

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
    parser = argparse.ArgumentParser(description="Train a mirrorshift transformer model")
    parser.add_argument(
        "--model-config",
        type=str,
        default="mirrorshift/config/model_configs/small.json",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="mirrorshift/config/training_configs/small.json",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mirrorshift/datasets/coqa_stories.txt",
        help="Path to training text file",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="causal_lm",
        help=f"Experiment spec name ({', '.join(list_train_specs())})",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="TensorBoard log directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    model_config = read_model_config(args.model_config)
    training_config = read_training_config(args.training_config)
    LOGGER.info(
        "Resolved model config:\n%s",
        json.dumps(config_to_dict(model_config), indent=2, sort_keys=True),
    )
    LOGGER.info(
        "Resolved training config:\n%s",
        json.dumps(config_to_dict(training_config), indent=2, sort_keys=True),
    )

    train_spec = get_train_spec(args.spec)
    train_dataset = train_spec.build_dataset(args.dataset, model_config.context_length)

    if train_dataset.get_vocab_size() != model_config.vocab_size:
        raise ValueError(
            "Dataset vocab size does not match model config vocab size: "
            f"{train_dataset.get_vocab_size()} vs {model_config.vocab_size}"
        )
    if len(train_dataset) < training_config.batch_size:
        raise ValueError(
            "Dataset must have at least batch_size sequences. "
            f"len(dataset)={len(train_dataset)} batch_size={training_config.batch_size}"
        )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        sampler=RandomSampler(train_dataset),
    )
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty for the provided configuration")

    model = train_spec.build_model(model_config)
    device = resolve_device(training_config.device)
    model = model.to(device)
    if training_config.compile:
        model = torch.compile(model)

    opt = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    writer = SummaryWriter(log_dir=args.log_dir)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "dataset_size=%d trainable_params=%d device=%s max_steps=%d",
        len(train_dataset),
        trainable_params,
        device,
        training_config.max_steps,
    )

    final_step = train(
        model=model,
        train_loader=train_loader,
        opt=opt,
        loss_fn=train_spec.loss_fn,
        device=device,
        training_config=training_config,
        writer=writer,
    )
    writer.flush()
    writer.close()

    LOGGER.info("Training complete at step=%d", final_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
