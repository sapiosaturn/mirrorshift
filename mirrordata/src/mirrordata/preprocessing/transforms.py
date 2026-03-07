from __future__ import annotations

import unicodedata
from typing import Sequence


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def append_eos(token_ids: Sequence[int], eos_token_id: int | None) -> list[int]:
    output = [int(token_id) for token_id in token_ids]
    if eos_token_id is not None:
        output.append(int(eos_token_id))
    return output
