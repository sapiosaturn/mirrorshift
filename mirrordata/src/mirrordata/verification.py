from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mirrordata.planning import SequencePlanManifest
from mirrordata.snapshot import IndexReader, SnapshotManifest, TokenSnapshot


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(
    snapshot_path: str | Path,
    *,
    check_checksums: bool = True,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    snapshot_root = Path(snapshot_path)
    if snapshot_root.is_file():
        snapshot_root = snapshot_root.parent

    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.is_file():
        return False, [f"missing snapshot manifest: {manifest_path}"]

    manifest = SnapshotManifest.read_json(manifest_path)
    expected_begin = 0
    total_tokens = 0
    total_documents = 0
    token_dtype = np.dtype(manifest.token_dtype)

    for shard in manifest.shards:
        shard_token_path = snapshot_root / shard.token_path
        shard_index_path = snapshot_root / shard.index_path
        if not shard_token_path.is_file():
            errors.append(f"missing shard token file: {shard.token_path}")
            continue
        if not shard_index_path.is_file():
            errors.append(f"missing shard index file: {shard.index_path}")
            continue

        actual_tokens = shard_token_path.stat().st_size // token_dtype.itemsize
        if actual_tokens != shard.num_tokens:
            errors.append(
                f"shard token count mismatch for {shard.token_path}: "
                f"manifest={shard.num_tokens} actual={actual_tokens}"
            )
        if shard.token_offset_begin != expected_begin:
            errors.append(
                f"shard offset begin mismatch for {shard.token_path}: "
                f"expected={expected_begin} actual={shard.token_offset_begin}"
            )
        if shard.token_offset_end != shard.token_offset_begin + shard.num_tokens:
            errors.append(
                f"shard offset end mismatch for {shard.token_path}: "
                f"begin={shard.token_offset_begin} end={shard.token_offset_end} "
                f"num_tokens={shard.num_tokens}"
            )

        index_reader = IndexReader(shard_index_path)
        if index_reader.header.num_documents != shard.num_documents:
            errors.append(
                f"index document count mismatch for {shard.index_path}: "
                f"manifest={shard.num_documents} index={index_reader.header.num_documents}"
            )
        if index_reader.header.num_tokens != shard.num_tokens:
            errors.append(
                f"index token count mismatch for {shard.index_path}: "
                f"manifest={shard.num_tokens} index={index_reader.header.num_tokens}"
            )

        if check_checksums and shard.token_checksum_sha256 is not None:
            actual_checksum = _sha256_file(shard_token_path)
            if actual_checksum != shard.token_checksum_sha256:
                errors.append(
                    f"token checksum mismatch for {shard.token_path}: "
                    f"expected={shard.token_checksum_sha256} actual={actual_checksum}"
                )
        if check_checksums and shard.index_checksum_sha256 is not None:
            actual_checksum = _sha256_file(shard_index_path)
            if actual_checksum != shard.index_checksum_sha256:
                errors.append(
                    f"index checksum mismatch for {shard.index_path}: "
                    f"expected={shard.index_checksum_sha256} actual={actual_checksum}"
                )

        expected_begin = shard.token_offset_end
        total_tokens += shard.num_tokens
        total_documents += shard.num_documents

    if total_tokens != manifest.total_tokens:
        errors.append(
            f"snapshot total_tokens mismatch: manifest={manifest.total_tokens} summed={total_tokens}"
        )
    if total_documents != manifest.total_documents:
        errors.append(
            "snapshot total_documents mismatch: "
            f"manifest={manifest.total_documents} summed={total_documents}"
        )

    return len(errors) == 0, errors


def verify_plan(
    plan_path: str | Path,
    *,
    snapshot_path: str | Path | None = None,
    check_checksums: bool = True,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    plan_root = Path(plan_path)
    if plan_root.is_file():
        plan_root = plan_root.parent

    manifest_path = plan_root / "plan.json"
    if not manifest_path.is_file():
        return False, [f"missing plan manifest: {manifest_path}"]

    manifest = SequencePlanManifest.read_json(manifest_path)
    starts_path = plan_root / manifest.sample_starts_path
    order_path = plan_root / manifest.sample_order_path
    if not starts_path.is_file():
        errors.append(f"missing sample starts file: {manifest.sample_starts_path}")
    if not order_path.is_file():
        errors.append(f"missing sample order file: {manifest.sample_order_path}")
    if errors:
        return False, errors

    if check_checksums and manifest.sample_starts_checksum_sha256 is not None:
        actual_checksum = _sha256_file(starts_path)
        if actual_checksum != manifest.sample_starts_checksum_sha256:
            errors.append(
                f"sample starts checksum mismatch: expected={manifest.sample_starts_checksum_sha256} "
                f"actual={actual_checksum}"
            )
    if check_checksums and manifest.sample_order_checksum_sha256 is not None:
        actual_checksum = _sha256_file(order_path)
        if actual_checksum != manifest.sample_order_checksum_sha256:
            errors.append(
                f"sample order checksum mismatch: expected={manifest.sample_order_checksum_sha256} "
                f"actual={actual_checksum}"
            )

    sample_starts = np.load(starts_path, mmap_mode="r")
    sample_order = np.load(order_path, mmap_mode="r")
    if sample_starts.shape != (manifest.num_samples,):
        errors.append(
            f"sample starts shape mismatch: manifest={manifest.num_samples} actual={sample_starts.shape}"
        )
    if sample_order.shape != (manifest.num_samples,):
        errors.append(
            f"sample order shape mismatch: manifest={manifest.num_samples} actual={sample_order.shape}"
        )

    if manifest.num_samples > 0:
        ordered = np.asarray(sample_order)
        if ordered.min() < 0 or ordered.max() >= manifest.num_samples:
            errors.append("sample order contains out-of-range indices")
        unique_count = int(np.unique(ordered).size)
        if unique_count != manifest.num_samples:
            errors.append(
                f"sample order is not a permutation: unique={unique_count} expected={manifest.num_samples}"
            )

    resolved_snapshot_path: Path | None = None
    if snapshot_path is not None:
        resolved_snapshot_path = Path(snapshot_path)
    else:
        manifest_snapshot_path = manifest.metadata.get("snapshot_path")
        if manifest_snapshot_path:
            candidate = Path(manifest_snapshot_path)
            if candidate.exists():
                resolved_snapshot_path = candidate

    if resolved_snapshot_path is not None:
        snapshot = TokenSnapshot.open(resolved_snapshot_path)
        if snapshot.manifest.snapshot_id != manifest.snapshot_id:
            errors.append(
                f"snapshot_id mismatch: snapshot={snapshot.manifest.snapshot_id} plan={manifest.snapshot_id}"
            )
        if sample_starts.size > 0:
            max_valid_start = snapshot.total_tokens - (manifest.sequence_length + 1)
            if max_valid_start < 0:
                errors.append(
                    f"snapshot too small for plan window: tokens={snapshot.total_tokens} "
                    f"window={manifest.sequence_length + 1}"
                )
            else:
                starts = np.asarray(sample_starts)
                if starts.min() < 0 or starts.max() > max_valid_start:
                    errors.append(
                        f"sample starts out of range for snapshot: max_valid_start={max_valid_start}"
                    )

    return len(errors) == 0, errors
