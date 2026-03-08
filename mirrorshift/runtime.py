"""Runtime helpers for device setup, determinism, and model initialization."""

import os
import random
from collections.abc import Callable

import torch
import torch.distributed as dist

from mirrorshift.config import DebugConfig, ModelConfig


def resolve_device(device_name: str) -> str:
    if device_name == "cpu":
        return "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        torch.set_float32_matmul_precision("high")
        return "cuda"
    raise ValueError("training.device must be 'cpu' or 'cuda'")


def set_determinism(device: str, debug_config: DebugConfig) -> None:
    seed = debug_config.seed
    if dist.is_initialized():
        seed_payload = [seed]
        if seed_payload[0] is None and dist.get_rank() == 0:
            seed_payload[0] = torch.seed()
        dist.broadcast_object_list(seed_payload, src=0)
        seed = int(seed_payload[0])
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    else:
        os.environ.pop("PYTHONHASHSEED", None)

    torch.use_deterministic_algorithms(debug_config.deterministic)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.deterministic = debug_config.deterministic
        torch.backends.cudnn.benchmark = not debug_config.deterministic
        if debug_config.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def build_meta_initialized_model(
    build_model: Callable[[ModelConfig], torch.nn.Module],
    model_config: ModelConfig,
    device: str | torch.device,
) -> torch.nn.Module:
    with torch.device("meta"):
        model = build_model(model_config)
    model.to_empty(device=device)
    init_weights = getattr(model, "init_weights", None)
    if not callable(init_weights):
        raise TypeError(
            f"{type(model).__name__} must implement init_weights() for meta-device materialization"
        )
    with torch.no_grad():
        init_weights()
    return model
