"""Step-based training entrypoint."""

import logging
import time
from typing import Any, Callable, Iterator, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim

from mirrorshift.checkpointing import CheckpointManager, TrainState
from mirrorshift.config import (
    ConfigManager,
    JobConfig,
    ModelConfig,
    TrainingConfig,
)
from mirrorshift.experiments import get_train_spec
from mirrorshift.infra import (
    apply_model_infra,
    barrier_if_distributed,
    build_runtime_context,
    clip_grad_norm_,
    destroy_process_group_if_needed,
    get_grad_norm,
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
from mirrorshift.run_manifest import discard_staged_run_artifacts, finalize_run_artifacts
from mirrorshift.utils import get_lr_schedule

BatchType = Tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

LOGGER = logging.getLogger("mirrorshift.train")
_GIB_IN_BYTES = 1024**3

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


def estimate_num_flops_per_token(
    model: torch.nn.Module,
    *,
    seq_len: int,
) -> int:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "embedding" in name
    )
    model_config = getattr(model, "model_config", None)
    if isinstance(model_config, ModelConfig):
        if model_config.attention_type == "gqa":
            head_dims = 2 * (model_config.embedding_dim // model_config.num_heads)
        else:
            head_dims = (
                int(model_config.qk_nope_head_dim or 0)
                + int(model_config.qk_rope_head_dim or 0)
                + int(model_config.v_head_dim or 0)
            )
        attention_term = (
            6
            * model_config.num_layers
            * model_config.num_heads
            * head_dims
            * seq_len
        )
        return 6 * max(0, trainable_params - embedding_params) + attention_term
    return 6 * trainable_params


def compute_global_loss_metrics(
    loss_value: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, int]:
    device = loss_value.device
    local_token_count = torch.tensor(float(targets.numel()), device=device)
    local_loss_sum = loss_value.detach().float() * local_token_count
    local_avg_loss = local_loss_sum / local_token_count

    if dist.is_available() and dist.is_initialized():
        global_loss_sum = local_loss_sum.clone()
        global_token_count = local_token_count.clone()
        global_max_loss = local_avg_loss.clone()
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_max_loss, op=dist.ReduceOp.MAX)
        global_avg_loss = global_loss_sum / global_token_count
        return (
            float(global_avg_loss.item()),
            float(global_max_loss.item()),
            int(global_token_count.item()),
        )

    return (
        float(local_avg_loss.item()),
        float(local_avg_loss.item()),
        int(local_token_count.item()),
    )


def compute_batch_heterogeneity(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    per_token_loss = F.cross_entropy(
        logits.detach().float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.size(0), -1)
    per_sequence_loss = per_token_loss.mean(dim=1)
    local_sum = per_sequence_loss.sum()
    local_max = per_sequence_loss.max()
    local_count = torch.tensor(float(per_sequence_loss.numel()), device=logits.device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    global_mean = local_sum / local_count
    return float((local_max - global_mean).item())


def compute_grad_norm(
    model: torch.nn.Module,
    *,
    max_grad_norm: float | None = None,
) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if max_grad_norm is None:
        total_norm = get_grad_norm(parameters, foreach=True)
    else:
        total_norm = clip_grad_norm_(
            parameters,
            max_grad_norm,
            foreach=True,
        )
    return float(total_norm.item())


def get_peak_memory_gib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    max_active = torch.cuda.max_memory_allocated(device_index) / _GIB_IN_BYTES
    max_reserved = torch.cuda.max_memory_reserved(device_index) / _GIB_IN_BYTES
    return float(max_active), float(max_reserved)


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
    data_parallel_world_size: int = 1,
    n_tokens_seen_start: int = 0,
) -> int:
    lr_schedule = get_lr_schedule(
        schedule=training_config.lr_schedule,
        max_lr=training_config.learning_rate,
        warmup_steps=training_config.lr_warmup_steps,
        total_steps=training_config.max_steps,
    )
    batch_iter = iter_batches(train_loader)
    active_train_state = train_state if train_state is not None else TrainState()
    device = torch.device(device)
    num_flops_per_token: int | None = None
    n_tokens_seen = int(n_tokens_seen_start)

    model.train()
    if active_train_state.step >= training_config.max_steps:
        return active_train_state.step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(apply_resume_offset(train_loader, resume_batch_offset)):
        next(batch_iter)

    for global_step in range(active_train_state.step + 1, training_config.max_steps + 1):
        step_start_time = time.perf_counter()
        data_loading_start_time = step_start_time
        x, y = next(batch_iter)
        x = x.to(device)
        y = y.to(device)
        data_loading_time = time.perf_counter() - data_loading_start_time
        if num_flops_per_token is None:
            num_flops_per_token = estimate_num_flops_per_token(
                model,
                seq_len=y.size(1),
            )

        opt.zero_grad(set_to_none=True)
        lr = lr_schedule(global_step - 1)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        logits = model(x)
        loss_value = loss_fn(logits, y)
        batch_heterogeneity = compute_batch_heterogeneity(logits, y)
        loss_value.backward()
        grad_norm = compute_grad_norm(
            model,
            max_grad_norm=training_config.max_grad_norm,
        )
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end_to_end_time = time.perf_counter() - step_start_time

        global_avg_loss, global_max_loss, global_tokens_this_step = (
            compute_global_loss_metrics(loss_value, y)
        )
        n_tokens_seen += global_tokens_this_step
        tokens_per_second_per_gpu = (
            global_tokens_this_step / max(1, data_parallel_world_size)
        ) / max(end_to_end_time, 1e-9)
        tflops = (
            float(num_flops_per_token) * tokens_per_second_per_gpu / 1e12
            if num_flops_per_token is not None
            else 0.0
        )
        max_active_gib, max_reserved_gib = get_peak_memory_gib(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        metrics_logger.log(
            {
                "train/batch_heterogeneity": batch_heterogeneity,
                "train/loss": global_avg_loss,
                "train/max_loss": global_max_loss,
                "train/grad_norm": grad_norm,
                "train/n_tokens_seen": float(n_tokens_seen),
                "optimizer/lr": lr,
                "throughput/tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                "throughput/tflops": tflops,
                "timing/end_to_end_seconds": end_to_end_time,
                "timing/data_loading_seconds": data_loading_time,
                "memory/max_active_gib": max_active_gib,
                "memory/max_reserved_gib": max_reserved_gib,
            },
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
                "step=%d/%d loss=%.5f max_loss=%.5f batch_het=%.5f grad_norm=%.4f "
                "tps/gpu=%.2f tflops=%.4f end_to_end=%.4fs data_loading=%.4fs "
                "memory_active=%.2fGiB memory_reserved=%.2fGiB lr=%.2e "
                "n_tokens_seen=%d",
                global_step,
                training_config.max_steps,
                global_avg_loss,
                global_max_loss,
                batch_heterogeneity,
                grad_norm,
                tokens_per_second_per_gpu,
                tflops,
                end_to_end_time,
                data_loading_time,
                max_active_gib,
                max_reserved_gib,
                lr,
                n_tokens_seen,
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
    artifacts = None

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
        else:
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
        if runtime_context.is_primary and not artifacts.resumed:
            artifacts = finalize_run_artifacts(artifacts)
        barrier_if_distributed(runtime_context)

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
            data_parallel_world_size=runtime_context.batch_world_size,
            n_tokens_seen_start=(
                train_state.step
                * config.training.batch_size
                * config.model.context_length
            ),
        )
    finally:
        if metrics_logger is not None:
            metrics_logger.close()
        if runtime_context.is_primary:
            discard_staged_run_artifacts(artifacts)
        destroy_process_group_if_needed(runtime_context)

    if runtime_context.is_primary:
        LOGGER.info("Training complete at step=%d", final_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
