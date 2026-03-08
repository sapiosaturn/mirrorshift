from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from mirrordata.planning import SequencePlan
from mirrordata.snapshot import TokenSnapshot
from mirrordata.tokenizers import TiktokenTokenizer


@dataclass(frozen=True)
class DatasetIdentity:
    snapshot_id: str
    snapshot_fingerprint: str
    plan_fingerprint: str
    sequence_length: int
    dataset_size: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "sequence_length": self.sequence_length,
            "dataset_size": self.dataset_size,
        }


class CausalLMSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        snapshot_path: str,
        plan_path: str,
    ) -> None:
        self.snapshot = TokenSnapshot.open(snapshot_path)
        self.plan = SequencePlan.open(plan_path)
        if self.snapshot.manifest.snapshot_id != self.plan.manifest.snapshot_id:
            raise ValueError(
                "snapshot_id mismatch between snapshot and plan: "
                f"{self.snapshot.manifest.snapshot_id} vs {self.plan.manifest.snapshot_id}"
            )
        self.sequence_length = self.plan.manifest.sequence_length
        self.tokenizer = TiktokenTokenizer(self.snapshot.manifest.tokenizer.name)

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.plan.sample_start(index)
        window = self.snapshot.read_window(start, self.sequence_length + 1)
        x = torch.tensor(window[:-1], dtype=torch.long)
        y = torch.tensor(window[1:], dtype=torch.long)
        return x, y

    def get_vocab_size(self) -> int:
        vocab_size = self.snapshot.manifest.tokenizer.vocab_size
        if vocab_size is None:
            raise ValueError("snapshot manifest does not define tokenizer vocab size")
        return int(vocab_size)

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def identity(self) -> DatasetIdentity:
        return DatasetIdentity(
            snapshot_id=self.snapshot.manifest.snapshot_id,
            snapshot_fingerprint=self.snapshot.manifest.fingerprint(),
            plan_fingerprint=self.plan.manifest.fingerprint(),
            sequence_length=self.sequence_length,
            dataset_size=len(self),
        )
