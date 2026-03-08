"""Step-based training entrypoint."""

import logging
import math
import time
from collections import deque
from typing import Any, Callable, Iterator, Tuple

import torch
import torch.optim as optim

from mirrorshift.checkpointing import CheckpointManager, TrainState
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


def iter_batches(train_loader: Any) -> Iterator[BatchType]:
    if hasattr(train_loader, "next"):
        while True:
            yield train_loader.next()
    while True:
        for batch in train_loader:
            yield batch


def apply_resume_offset(train_loader: Any, resume_batch_offset: int) -> int:
    if resume_batch_offset <= 0:
        return 0

    load_state_dict = getattr(train_loader, "load_state_dict", None)
    global_batch_size = getattr(train_loader, "global_batch_size", None)
    if callable(load_state_dict) and global_batch_size is not None:
        load_state_dict({"global_sample_cursor": resume_batch_offset * int(global_batch_size)})
        return 0
    return resume_batch_offset


def log_data_preflight(*, dataset_size: int, batch_size: int, max_steps: int) -> None:
    required_sequences = batch_size * max_steps
    if dataset_size < required_sequences:
        LOGGER.warning(
            "Dataset will wrap during training: available_sequences=%d required_sequences=%d",
            dataset_size,
            required_sequences,
        )


def train(
    model: torch.nn.Module,
    train_loader: Any,
    opt: optim.AdamW,
    loss_fn: LossFunction,
    device: str,
    training_config: TrainingConfig,
    metrics_logger: MetricsLogger,
    train_state: TrainState | None = None,
    checkpointer: CheckpointManager | None = None,
    resume_batch_offset: int = 0,
) -> int:
    lr_schedule = get_lr_schedule(
        schedule=training_config.lr_schedule,
        max_lr=training_config.learning_rate,
        warmup_steps=training_config.lr_warmup_steps,
        total_steps=training_config.max_steps,
    )
    step_times = deque(maxlen=100)
    batch_iter = iter_batches(train_loader)
    active_train_state = train_state if train_state is not None else TrainState()

    model.train()
    if active_train_state.step >= training_config.max_steps:
        return active_train_state.step

    for _ in range(apply_resume_offset(train_loader, resume_batch_offset)):
        next(batch_iter)

    for global_step in range(active_train_state.step + 1, training_config.max_steps + 1):
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

        active_train_state.step = global_step
        if checkpointer is not None:
            checkpointer.save(
                global_step,
                last_step=global_step == training_config.max_steps,
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
    return active_train_state.step


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config: JobConfig = ConfigManager().parse_args()
    config.maybe_log(LOGGER)
    artifacts = create_run_artifacts(config)
    LOGGER.info("run_id=%s run_dir=%s", artifacts.run_id, artifacts.run_dir)
    device = resolve_device(config.training.device)

    train_spec = get_train_spec(config.run.spec)
    train_data = train_spec.build_data(config, artifacts.run_dir, device)

    if train_data.vocab_size != config.model.vocab_size:
        raise ValueError(
            "Dataset vocab size does not match model config vocab size: "
            f"{train_data.vocab_size} vs {config.model.vocab_size}"
        )
    if train_data.dataset_size < config.training.batch_size:
        raise ValueError(
            "Dataset must have at least batch_size sequences. "
            f"len(dataset)={train_data.dataset_size} batch_size={config.training.batch_size}"
        )
    train_loader = train_data.train_loader
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty for the provided configuration")
    log_data_preflight(
        dataset_size=train_data.dataset_size,
        batch_size=config.training.batch_size,
        max_steps=config.training.max_steps,
    )

    model = train_spec.build_model(config.model)
    model = model.to(device)
    if config.training.compile:
        model = torch.compile(model)

    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    train_state = TrainState()
    checkpointer = CheckpointManager(
        config=config.checkpoint,
        run_dir=artifacts.run_dir,
        model=model,
        optimizer=opt,
        train_state=train_state,
        train_loader=train_loader,
        data_identity=train_data.data_identity,
    )
    if config.checkpoint.load_step is not None:
        checkpointer.load()
        LOGGER.info("Resuming training from step=%d", train_state.step)

    metrics_logger = build_metrics_logger(
        config=config,
        run_id=artifacts.run_id,
        run_dir=artifacts.run_dir,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not artifacts.resumed:
        write_run_manifest(
            artifacts=artifacts,
            config=config,
            dataset_size=train_data.dataset_size,
            trainable_params=trainable_params,
            device=device,
        )
    else:
        LOGGER.info(
            "Reusing existing config snapshot=%s and run manifest=%s",
            artifacts.config_snapshot_path,
            artifacts.manifest_path,
        )

    LOGGER.info(
        "dataset_size=%d trainable_params=%d device=%s max_steps=%d",
        train_data.dataset_size,
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
            train_state=train_state,
            checkpointer=checkpointer,
            resume_batch_offset=(
                0
                if checkpointer.restored_loader_state
                else train_state.step if train_data.exact_resume else 0
            ),
        )
    finally:
        metrics_logger.close()

    LOGGER.info("Training complete at step=%d", final_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
