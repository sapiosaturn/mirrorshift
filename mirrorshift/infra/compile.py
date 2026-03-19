"""Simplified TorchTitan-style compile helpers."""

import torch
import torch.nn as nn

from mirrorshift.config import CompileConfig
from mirrorshift.infra.discovery import (
    replace_transformer_block,
    try_get_transformer_block_container,
)


def apply_compile(model: nn.Module, config: CompileConfig) -> nn.Module:
    if not config.enable:
        return model

    found = try_get_transformer_block_container(model)
    if found is None:
        return torch.compile(model, backend=config.backend, fullgraph=config.fullgraph)

    _, container = found
    for block_name, block in list(container.named_children()):
        compiled_block = torch.compile(
            block, backend=config.backend, fullgraph=config.fullgraph
        )
        replace_transformer_block(model, block_name, compiled_block)
    return model
