"""Simplified runtime/distributed context helpers for mirrorshift."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from mirrorshift.config import ParallelismConfig
from mirrorshift.infra.parallel_dims import ParallelDims
from mirrorshift.run_manifest import resolve_run_id


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    device_type: str
    world_size: int
    global_rank: int
    local_rank: int
    parallel_dims: ParallelDims

    @property
    def batch_world_size(self) -> int:
        return self.parallel_dims.batch_world_size

    @property
    def batch_rank(self) -> int:
        return self.global_rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.global_rank == 0


def _distributed_backend(device_type: str) -> str:
    if device_type in torch.distributed.Backend.default_device_backend_map:
        backend = torch.distributed.Backend.default_device_backend_map[device_type]
        if backend is not None:
            return backend
    return "gloo"


def build_runtime_context(
    device_type: str, parallelism_config: ParallelismConfig
) -> RuntimeContext:
    requested_world_size = (
        parallelism_config.dp_replicate * parallelism_config.dp_shard
    )
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if requested_world_size == 1:
        if env_world_size != 1:
            raise ValueError(
                "WORLD_SIZE indicates distributed launch but parallelism is configured for "
                f"single-process execution: env={env_world_size}"
            )
        device = torch.device(device_type)
        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
        return RuntimeContext(
            device=device,
            device_type=device_type,
            world_size=1,
            global_rank=0,
            local_rank=0,
            parallel_dims=parallel_dims,
        )

    if env_world_size != requested_world_size:
        raise ValueError(
            "Configured parallelism does not match WORLD_SIZE: "
            f"requested={requested_world_size} env={env_world_size}"
        )

    if not dist.is_initialized():
        dist.init_process_group(backend=_distributed_backend(device_type))

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_type)

    parallel_dims = ParallelDims(
        dp_replicate=parallelism_config.dp_replicate,
        dp_shard=parallelism_config.dp_shard,
        world_size=world_size,
    )
    parallel_dims.build_mesh(device.type)
    return RuntimeContext(
        device=device,
        device_type=device.type,
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        parallel_dims=parallel_dims,
    )


def barrier_if_distributed(runtime_context: RuntimeContext) -> None:
    if runtime_context.is_distributed:
        dist.barrier()


def synchronized_run_id(
    requested_run_id: str | None, runtime_context: RuntimeContext
) -> str:
    if not runtime_context.is_distributed:
        return resolve_run_id(requested_run_id)

    payload = [resolve_run_id(requested_run_id) if runtime_context.is_primary else None]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def destroy_process_group_if_needed(runtime_context: RuntimeContext) -> None:
    if runtime_context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
