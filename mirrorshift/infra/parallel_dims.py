"""Simplified data-parallel mesh helpers derived from TorchTitan."""

from dataclasses import dataclass, field

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    world_size: int

    _world_mesh: DeviceMesh | None = field(default=None, init=False)
    _meshes: dict[str, DeviceMesh] = field(default_factory=dict, init=False)
    _device_type: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.dp_replicate <= 0:
            raise ValueError("dp_replicate must be > 0")
        if self.dp_shard <= 0:
            raise ValueError("dp_shard must be > 0")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.dp_replicate * self.dp_shard != self.world_size:
            raise ValueError(
                "Invalid parallel dims: "
                f"dp_replicate({self.dp_replicate}) * dp_shard({self.dp_shard}) "
                f"!= world_size({self.world_size})"
            )

    @property
    def batch_world_size(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def dp_enabled(self) -> bool:
        return self.batch_world_size > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    def build_mesh(self, device_type: str) -> DeviceMesh | None:
        if self._world_mesh is not None:
            return self._world_mesh

        self._device_type = device_type
        if not self.dp_enabled:
            self._world_mesh = init_device_mesh(
                device_type,
                (1, 1),
                mesh_dim_names=("dp_replicate", "fsdp"),
            )
            self._meshes = {
                "dp_replicate": self._world_mesh["dp_replicate"],
                "fsdp": self._world_mesh["fsdp"],
            }
        elif self.dp_replicate_enabled and self.dp_shard_enabled:
            self._world_mesh = init_device_mesh(
                device_type,
                (self.dp_replicate, self.dp_shard),
                mesh_dim_names=("dp_replicate", "fsdp"),
            )
            self._meshes = {
                "dp_replicate": self._world_mesh["dp_replicate"],
                "fsdp": self._world_mesh["fsdp"],
            }
        elif self.dp_replicate_enabled:
            self._world_mesh = init_device_mesh(
                device_type,
                (self.dp_replicate,),
                mesh_dim_names=("dp_replicate",),
            )
            self._meshes = {"dp_replicate": self._world_mesh["dp_replicate"]}
        else:
            self._world_mesh = init_device_mesh(
                device_type,
                (self.dp_shard,),
                mesh_dim_names=("fsdp",),
            )
            self._meshes = {"fsdp": self._world_mesh["fsdp"]}

        return self._world_mesh

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        if self._world_mesh is None:
            raise RuntimeError("build_mesh() must be called before get_mesh()")

        if isinstance(dims, str):
            dims = [dims]
        if dims == ["dp_replicate", "fsdp"] or dims == ["fsdp", "dp_replicate"]:
            return self._world_mesh
        if len(dims) != 1:
            raise ValueError(f"Unsupported mesh dims request: {dims}")
        dim = dims[0]
        if dim not in self._meshes:
            raise ValueError(f"Mesh dim not available: {dim}")
        return self._meshes[dim]
