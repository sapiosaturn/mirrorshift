from __future__ import annotations

from typing import Sequence

import tiktoken

from mirrordata.snapshot.manifest import TokenizerManifest


class TiktokenTokenizer:
    backend = "tiktoken"

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

    @property
    def eos_token_id(self) -> int | None:
        token_id = getattr(self._encoding, "eot_token", None)
        return int(token_id) if token_id is not None else None

    def to_manifest(self) -> TokenizerManifest:
        return TokenizerManifest(
            backend=self.backend,
            name=self.name,
            vocab_size=self.n_vocab,
            eos_token_id=self.eos_token_id,
        )


TiktokenEncoding = TiktokenTokenizer
