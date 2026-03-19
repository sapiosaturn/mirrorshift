from __future__ import annotations

from typing import Protocol, Sequence

from mirrordata.snapshot.manifest import TokenizerManifest


class TokenizerBackend(Protocol):
    backend: str
    name: str

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...

    @property
    def n_vocab(self) -> int:
        ...

    @property
    def eos_token_id(self) -> int | None:
        ...

    def to_manifest(self) -> TokenizerManifest:
        ...
