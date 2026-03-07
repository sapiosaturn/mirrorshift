from __future__ import annotations

from typing import Protocol, Sequence


class TokenizerBackend(Protocol):
    name: str

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...

    @property
    def n_vocab(self) -> int:
        ...
