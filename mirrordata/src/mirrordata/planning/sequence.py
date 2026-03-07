from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mirrordata.snapshot import TokenSnapshot


SEQUENCE_PLAN_FORMAT_VERSION = "1"


@dataclass(frozen=True)
class SequencePlanSpec:
    snapshot_path: str
    output_dir: str
    sequence_length: int
    stride: int | None = None
    shuffle: bool = True
    shuffle_seed: int = 0


@dataclass(frozen=True)
class SequencePlanManifest:
    format_version: str
    snapshot_id: str
    sequence_length: int
    stride: int
    num_samples: int
    shuffle: bool
    shuffle_seed: int
    sample_starts_path: str
    sample_order_path: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "snapshot_id": self.snapshot_id,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "num_samples": self.num_samples,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "sample_starts_path": self.sample_starts_path,
            "sample_order_path": self.sample_order_path,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SequencePlanManifest":
        return cls(
            format_version=str(data["format_version"]),
            snapshot_id=str(data["snapshot_id"]),
            sequence_length=int(data["sequence_length"]),
            stride=int(data["stride"]),
            num_samples=int(data["num_samples"]),
            shuffle=bool(data["shuffle"]),
            shuffle_seed=int(data["shuffle_seed"]),
            sample_starts_path=str(data["sample_starts_path"]),
            sample_order_path=str(data["sample_order_path"]),
            created_at=str(data["created_at"]),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "SequencePlanManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))


class SequencePlanBuilder:
    def __init__(self, spec: SequencePlanSpec) -> None:
        if spec.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.spec = spec

    def run(self) -> SequencePlanManifest:
        snapshot = TokenSnapshot.open(self.spec.snapshot_path)
        output_dir = Path(self.spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stride = self.spec.stride or self.spec.sequence_length
        window_length = self.spec.sequence_length + 1
        if snapshot.total_tokens < window_length:
            raise ValueError(
                f"snapshot has {snapshot.total_tokens} tokens, fewer than required window {window_length}"
            )

        sample_starts = np.arange(
            0,
            snapshot.total_tokens - window_length + 1,
            stride,
            dtype=np.uint64,
        )
        sample_order = np.arange(sample_starts.size, dtype=np.uint64)
        if self.spec.shuffle and sample_order.size > 0:
            rng = np.random.default_rng(self.spec.shuffle_seed)
            rng.shuffle(sample_order)

        starts_path = output_dir / "sample_starts.npy"
        order_path = output_dir / "sample_order.npy"
        np.save(starts_path, sample_starts)
        np.save(order_path, sample_order)

        manifest = SequencePlanManifest(
            format_version=SEQUENCE_PLAN_FORMAT_VERSION,
            snapshot_id=snapshot.manifest.snapshot_id,
            sequence_length=self.spec.sequence_length,
            stride=stride,
            num_samples=int(sample_starts.size),
            shuffle=self.spec.shuffle,
            shuffle_seed=self.spec.shuffle_seed,
            sample_starts_path=starts_path.name,
            sample_order_path=order_path.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"snapshot_path": str(Path(self.spec.snapshot_path))},
        )
        manifest.write_json(output_dir / "plan.json")
        return manifest


class SequencePlan:
    def __init__(self, root: str | Path, manifest: SequencePlanManifest | None = None) -> None:
        root_path = Path(root)
        self.root = root_path if root_path.is_dir() else root_path.parent
        self.manifest = manifest or SequencePlanManifest.read_json(self.root / "plan.json")
        self.sample_starts = np.load(self.root / self.manifest.sample_starts_path, mmap_mode="r")
        self.sample_order = np.load(self.root / self.manifest.sample_order_path, mmap_mode="r")

    @classmethod
    def open(cls, path: str | Path) -> "SequencePlan":
        input_path = Path(path)
        if input_path.is_dir():
            return cls(input_path)
        if input_path.name == "plan.json":
            return cls(input_path.parent, SequencePlanManifest.read_json(input_path))
        raise ValueError(f"expected plan directory or plan.json, got {input_path}")

    def __len__(self) -> int:
        return self.manifest.num_samples

    def sample_start(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ordered_index = int(self.sample_order[index])
        return int(self.sample_starts[ordered_index])


def build_sequence_plan(spec: SequencePlanSpec) -> SequencePlanManifest:
    return SequencePlanBuilder(spec).run()
