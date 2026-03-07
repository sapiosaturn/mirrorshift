from __future__ import annotations

from typing import Sequence

import tiktoken


class TiktokenEncoding:
    def __init__(self, name: str = "p50k_base") -> None:
        self.name = name
        self._encoding = tiktoken.get_encoding(name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._encoding.decode([int(token_id) for token_id in token_ids])

    @property
    def n_vocab(self) -> int:
        return self._encoding.n_vocab
