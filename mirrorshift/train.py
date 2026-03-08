"""Step-based training entrypoint."""

import logging
from typing import Any, Callable, Iterator, Tuple

import torch
import torch.optim as optim

from mirrorshift.checkpointing import CheckpointManager, TrainState
from mirrorshift.config import (
    ConfigManager,
    JobConfig,
    TrainingConfig,
)
from mirrorshift.experiments import get_train_spec
from mirrorshift.infra import (
    apply_model_infra,
    barrier_if_distributed,
    build_runtime_context,
    destroy_process_group_if_needed,
    synchronized_run_id,
)
from mirrorshift.metrics import MetricsLogger, build_metrics_logger
from mirrorshift.runtime import (
    build_meta_model,
    materialize_initialized_model,
    resolve_device,
    set_determinism,
)
from mirrorshift.run_manifest import create_run_artifacts, write_run_manifest
from mirrorshift.utils import get_lr_schedule

BatchType = Tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

LOGGER = logging.getLogger("mirrorshift.train")

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
    device: str | torch.device,
    training_config: TrainingConfig,
    metrics_logger: MetricsLogger,
    train_state: TrainState | None = None,
    checkpointer: CheckpointManager | None = None,
    resume_batch_offset: int = 0,
    is_primary: bool = True,
) -> int:
    lr_schedule = get_lr_schedule(
        schedule=training_config.lr_schedule,
        max_lr=training_config.learning_rate,
        warmup_steps=training_config.lr_warmup_steps,
        total_steps=training_config.max_steps,
    )
    batch_iter = iter_batches(train_loader)
    active_train_state = train_state if train_state is not None else TrainState()

    model.train()
    if active_train_state.step >= training_config.max_steps:
        return active_train_state.step

    for _ in range(apply_resume_offset(train_loader, resume_batch_offset)):
        next(batch_iter)

    for global_step in range(active_train_state.step + 1, training_config.max_steps + 1):
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

        metrics_logger.log(
            {"train/loss": loss_scalar},
            step=global_step,
        )

        active_train_state.step = global_step
        if checkpointer is not None:
            checkpointer.save(
                global_step,
                last_step=global_step == training_config.max_steps,
            )

        if is_primary and (global_step == 1 or global_step % training_config.log_every == 0):
            LOGGER.info(
                "step=%d/%d loss=%.5f",
                global_step,
                training_config.max_steps,
                loss_scalar,
            )
    return active_train_state.step


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config: JobConfig = ConfigManager().parse_args()
    base_device_type = resolve_device(config.training.device)
    runtime_context = build_runtime_context(base_device_type, config.parallelism)
    metrics_logger: MetricsLogger | None = None

    try:
        if runtime_context.is_primary:
            config.maybe_log(LOGGER)

        run_id = synchronized_run_id(config.run.id, runtime_context)
        if runtime_context.is_primary:
            artifacts = create_run_artifacts(
                config,
                resolved_run_id=run_id,
                write_files=True,
            )
        barrier_if_distributed(runtime_context)
        if not runtime_context.is_primary:
            artifacts = create_run_artifacts(
                config,
                resolved_run_id=run_id,
                write_files=False,
            )

        if runtime_context.is_primary:
            LOGGER.info("run_id=%s run_dir=%s", artifacts.run_id, artifacts.run_dir)

        set_determinism(str(runtime_context.device), config.debug)

        train_spec = get_train_spec(config.run.spec)
        train_data = train_spec.build_data(config, artifacts.run_dir, runtime_context)

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
        if runtime_context.is_primary:
            log_data_preflight(
                dataset_size=train_data.dataset_size,
                batch_size=config.training.batch_size,
                max_steps=config.training.max_steps,
            )

        model = build_meta_model(train_spec.build_model, config.model)
        model = apply_model_infra(
            model,
            runtime_context=runtime_context,
            parallelism_config=config.parallelism,
            activation_checkpoint_config=config.activation_checkpoint,
            compile_config=config.compile,
        )
        model = materialize_initialized_model(model, runtime_context.device)

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
            is_primary=runtime_context.is_primary,
            is_distributed=runtime_context.is_distributed,
        )
        if config.checkpoint.load_step is not None:
            checkpointer.load()
            if runtime_context.is_primary:
                LOGGER.info("Resuming training from step=%d", train_state.step)

        metrics_logger = build_metrics_logger(
            config=config,
            run_id=artifacts.run_id,
            run_dir=artifacts.run_dir,
            is_primary=runtime_context.is_primary,
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not artifacts.resumed and runtime_context.is_primary:
            write_run_manifest(
                artifacts=artifacts,
                config=config,
                dataset_size=train_data.dataset_size,
                trainable_params=trainable_params,
                device=str(runtime_context.device),
            )
        elif artifacts.resumed and runtime_context.is_primary:
            LOGGER.info(
                "Reusing existing config snapshot=%s and run manifest=%s",
                artifacts.config_snapshot_path,
                artifacts.manifest_path,
            )

        if runtime_context.is_primary:
            LOGGER.info(
                "dataset_size=%d trainable_params=%d device=%s max_steps=%d",
                train_data.dataset_size,
                trainable_params,
                runtime_context.device,
                config.training.max_steps,
            )
            LOGGER.info(
                "config_snapshot=%s run_manifest=%s",
                artifacts.config_snapshot_path,
                artifacts.manifest_path,
            )

        final_step = train(
            model=model,
            train_loader=train_loader,
            opt=opt,
            loss_fn=train_spec.loss_fn,
            device=runtime_context.device,
            training_config=config.training,
            metrics_logger=metrics_logger,
            train_state=train_state,
            checkpointer=checkpointer,
            resume_batch_offset=(
                0
                if checkpointer.restored_loader_state
                else train_state.step if train_data.exact_resume else 0
            ),
            is_primary=runtime_context.is_primary,
        )
    finally:
        if metrics_logger is not None:
            metrics_logger.close()
        destroy_process_group_if_needed(runtime_context)

    if runtime_context.is_primary:
        LOGGER.info("Training complete at step=%d", final_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
