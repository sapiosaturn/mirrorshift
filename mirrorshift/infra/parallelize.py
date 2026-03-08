"""Simplified parallel/performance infrastructure derived from TorchTitan."""

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from mirrorshift.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
)
from mirrorshift.infra.activation_checkpoint import apply_activation_checkpointing
from mirrorshift.infra.compile import apply_compile
from mirrorshift.infra.discovery import (
    named_auxiliary_data_parallel_modules,
    named_transformer_blocks,
)
from mirrorshift.infra.distributed import RuntimeContext

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            set_gradient_divide_factor = getattr(module, "set_gradient_divide_factor", None)
            if callable(set_gradient_divide_factor):
                set_gradient_divide_factor(1.0)


def _resolve_reshard_after_forward(policy: str) -> bool:
    if policy == "always":
        return True
    if policy in {"never", "default"}:
        return False if policy == "never" else True
    raise ValueError(f"Unsupported reshard_after_forward policy: {policy}")


def apply_fsdp(
    model: nn.Module,
    runtime_context: RuntimeContext,
    parallelism_config: ParallelismConfig,
) -> nn.Module:
    dim_names = (
        ["dp_replicate", "fsdp"]
        if runtime_context.parallel_dims.dp_replicate_enabled
        else ["fsdp"]
    )
    dp_mesh = runtime_context.parallel_dims.get_mesh(dim_names)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[parallelism_config.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[parallelism_config.mixed_precision_reduce],
    )
    reshard_after_forward = _resolve_reshard_after_forward(
        parallelism_config.reshard_after_forward
    )
    for _, module in named_auxiliary_data_parallel_modules(model):
        fully_shard(
            module,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    for _, block in named_transformer_blocks(model):
        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    disable_fsdp_gradient_division(model)
    return model


def apply_ddp(
    model: nn.Module,
    runtime_context: RuntimeContext,
    parallelism_config: ParallelismConfig,
    compile_config: CompileConfig,
) -> nn.Module:
    if compile_config.enable:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"
    dp_mesh = runtime_context.parallel_dims.get_mesh("dp_replicate")
    replicate(
        model,
        device_mesh=dp_mesh,
        bucket_cap_mb=parallelism_config.bucket_cap_mb,
    )
    return model


def apply_model_infra(
    model: nn.Module,
    *,
    runtime_context: RuntimeContext,
    parallelism_config: ParallelismConfig,
    activation_checkpoint_config: ActivationCheckpointConfig,
    compile_config: CompileConfig,
) -> nn.Module:
    if activation_checkpoint_config.mode != "none":
        apply_activation_checkpointing(model, activation_checkpoint_config)

    if compile_config.enable:
        model = apply_compile(model, compile_config)

    if runtime_context.parallel_dims.dp_shard_enabled:
        model = apply_fsdp(model, runtime_context, parallelism_config)
    elif runtime_context.parallel_dims.dp_replicate_enabled:
        model = apply_ddp(model, runtime_context, parallelism_config, compile_config)

    return model
