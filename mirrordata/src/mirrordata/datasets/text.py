from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mirrordata.tokenizers import TiktokenEncoding


class TiktokenTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Small bridge dataset for local text-file experiments.

    This preserves the current mirrorshift training path while the token-first
    snapshot runtime is designed in this package.
    """

    def __init__(
        self,
        file_path: str,
        sequence_length: int,
        tokenizer_name: str = "p50k_base",
    ) -> None:
        self.file_path = str(file_path)
        self.sequence_length = int(sequence_length)
        self.tokenizer = TiktokenEncoding(tokenizer_name)

        text = Path(self.file_path).read_text()
        self.tokens = self.tokenizer.encode(text)
        if len(self.tokens) <= self.sequence_length:
            raise ValueError(
                f"tokenized dataset is too small for sequence_length={self.sequence_length}: "
                f"{len(self.tokens)} tokens in {self.file_path}"
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.sequence_length - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.sequence_length]
        y = self.tokens[idx + 1 : idx + self.sequence_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.n_vocab


TiktokenTxtDataset = TiktokenTextDataset
