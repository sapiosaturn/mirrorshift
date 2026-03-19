"""DTensor-aware gradient norm helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor


def _normalize_parameters(
    parameters: torch.Tensor | Iterable[torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters)


@torch.no_grad()
def get_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    parameters = _normalize_parameters(parameters)
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads,
        norm_type,
        error_if_nonfinite,
        foreach,
    )
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    return total_norm


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    parameters = _normalize_parameters(parameters)
    total_norm = get_grad_norm(
        parameters,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    torch.nn.utils.clip_grads_with_norm_(
        parameters,
        max_norm,
        total_norm,
        foreach,
    )
    return total_norm
