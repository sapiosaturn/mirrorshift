"""Generic model structure discovery for compile/checkpoint/parallel infra."""

from collections.abc import Iterator

import torch.nn as nn

_BLOCK_CONTAINER_CANDIDATES = ("decoder_stack", "layers", "blocks")
_AUXILIARY_MODULE_CANDIDATES = (
    "embedding_layer",
    "tok_embeddings",
    "embedding",
    "lm_head",
    "output",
    "norm",
)


def try_get_transformer_block_container(
    model: nn.Module,
) -> tuple[str, nn.ModuleList | nn.ModuleDict] | None:
    for name in _BLOCK_CONTAINER_CANDIDATES:
        container = getattr(model, name, None)
        if isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            return name, container
    return None


def get_transformer_block_container(
    model: nn.Module,
) -> tuple[str, nn.ModuleList | nn.ModuleDict]:
    container = try_get_transformer_block_container(model)
    if container is None:
        raise ValueError(
            f"Could not find transformer block container on {type(model).__name__}. "
            f"Tried {_BLOCK_CONTAINER_CANDIDATES}."
        )
    return container


def named_transformer_blocks(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    found = try_get_transformer_block_container(model)
    if found is None:
        return
    _, container = found
    for name, module in container.named_children():
        yield name, module


def replace_transformer_block(model: nn.Module, block_name: str, module: nn.Module) -> None:
    _, container = get_transformer_block_container(model)
    container.register_module(block_name, module)


def named_auxiliary_data_parallel_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name in _AUXILIARY_MODULE_CANDIDATES:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            yield name, module
