"""Simplified activation checkpointing helpers derived from TorchTitan."""

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from mirrorshift.config import ActivationCheckpointConfig
from mirrorshift.infra.discovery import named_transformer_blocks, replace_transformer_block

_layer_sac_count = 0


def _safe_op(path: str) -> torch._ops.OpOverload | None:
    current: object = torch.ops
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    if isinstance(current, torch._ops.OpOverload):
        return current
    return None


def default_op_sac_save_list() -> set[torch._ops.OpOverload]:
    ops = [
        "aten.mm.default",
        "aten.linear.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_cudnn_attention.default",
        "aten._scaled_dot_product_attention_math.default",
        "aten._scaled_dot_product_fused_attention_overrideable.default",
        "_c10d_functional.reduce_scatter_tensor.default",
        "aten.max.default",
    ]
    resolved = {_safe_op(path) for path in ops}
    return {op for op in resolved if op is not None}


def _wrap_checkpoint(module: nn.Module, config: ActivationCheckpointConfig) -> nn.Module:
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=config.preserve_rng_state,
        determinism_check=config.determinism_check,
        early_stop=config.early_stop,
        debug=config.debug,
    )


def _apply_layer_selective_checkpointing(
    module: nn.Module, config: ActivationCheckpointConfig
) -> nn.Module:
    global _layer_sac_count
    _layer_sac_count += 1
    checkpoint_every = int(config.selective_ac_option)
    if _layer_sac_count % checkpoint_every == 0:
        return _wrap_checkpoint(module, config)
    return module


def _apply_op_selective_checkpointing(
    module: nn.Module, config: ActivationCheckpointConfig
) -> nn.Module:
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    op_sac_save_list = default_op_sac_save_list()
    mm_ops = tuple(op for op in (_safe_op("aten.mm.default"), _safe_op("aten.linear.default")) if op is not None)

    def _get_custom_policy(meta: defaultdict[str, int]):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func in mm_ops:
                meta[mm_count_key] += 1
            to_save = func in op_sac_save_list and not (
                func in mm_ops and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta: defaultdict[str, int] = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=config.preserve_rng_state,
        determinism_check=config.determinism_check,
        early_stop=config.early_stop,
        debug=config.debug,
    )


def apply_activation_checkpointing(
    model: nn.Module,
    config: ActivationCheckpointConfig,
) -> None:
    if config.mode == "none":
        return

    global _layer_sac_count
    _layer_sac_count = 0

    for block_name, block in list(named_transformer_blocks(model)):
        if config.mode == "full":
            wrapped = _wrap_checkpoint(block, config)
        elif config.mode == "selective":
            if config.selective_ac_option == "op":
                wrapped = _apply_op_selective_checkpointing(block, config)
            else:
                wrapped = _apply_layer_selective_checkpointing(block, config)
        else:
            raise ValueError(
                f"Unsupported activation checkpoint mode: {config.mode}"
            )
        replace_transformer_block(model, block_name, wrapped)
