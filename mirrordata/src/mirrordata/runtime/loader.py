from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import CausalLMSequenceDataset


@dataclass(frozen=True)
class _RankSlice:
    count: int
    start_offset: int


def _compute_rank_slice(global_batch_size: int, world_size: int, rank: int) -> _RankSlice:
    q, r = divmod(global_batch_size, world_size)
    return _RankSlice(count=q + (1 if rank < r else 0), start_offset=rank * q + min(rank, r))


class DeterministicBatchLoader:
    def __init__(
        self,
        dataset: CausalLMSequenceDataset,
        *,
        global_batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        device: str | torch.device = "cpu",
        drop_last: bool = True,
        wrap: bool = False,
    ) -> None:
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range for world_size={world_size}")
        if global_batch_size < world_size:
            raise ValueError("global_batch_size must be >= world_size for deterministic rank slicing")

        self.dataset = dataset
        self.global_batch_size = int(global_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = torch.device(device)
        self.drop_last = bool(drop_last)
        self.wrap = bool(wrap)

        rank_slice = _compute_rank_slice(self.global_batch_size, self.world_size, self.rank)
        self.local_batch_size = rank_slice.count
        self.local_batch_offset = rank_slice.start_offset
        self.global_sample_cursor = 0

    def __len__(self) -> int:
        dataset_size = len(self.dataset)
        if self.drop_last:
            return dataset_size // self.global_batch_size
        return (dataset_size + self.global_batch_size - 1) // self.global_batch_size

    def state_dict(self) -> dict[str, int]:
        return {"global_sample_cursor": self.global_sample_cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.global_sample_cursor = int(state["global_sample_cursor"])

    def _global_indices_for_step(self) -> list[int]:
        dataset_size = len(self.dataset)
        if dataset_size == 0:
            raise StopIteration

        start = self.global_sample_cursor
        end = start + self.global_batch_size

        if not self.wrap:
            if self.drop_last and end > dataset_size:
                raise StopIteration
            if not self.drop_last and start >= dataset_size:
                raise StopIteration
            end = min(end, dataset_size)
            indices = list(range(start, end))
        else:
            indices = [(start + offset) % dataset_size for offset in range(self.global_batch_size)]

        self.global_sample_cursor = end if not self.wrap else start + self.global_batch_size
        return indices

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._global_indices_for_step()
        local_indices = indices[self.local_batch_offset : self.local_batch_offset + self.local_batch_size]
        if not local_indices:
            raise StopIteration

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for index in local_indices:
            x, y = self.dataset[index]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)
