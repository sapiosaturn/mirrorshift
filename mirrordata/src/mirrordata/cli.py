from __future__ import annotations

import argparse
from pathlib import Path

from mirrordata.planning import SequencePlanSpec, build_sequence_plan
from mirrordata.preprocessing import ParquetSnapshotConfig, build_snapshot_from_parquet
from mirrordata.snapshot import SnapshotManifest


def _cmd_prep_parquet(args: argparse.Namespace) -> int:
    manifest = build_snapshot_from_parquet(
        ParquetSnapshotConfig(
            input_paths=tuple(args.input_paths),
            output_dir=args.output_dir,
            snapshot_id=args.snapshot_id,
            dataset_name=args.dataset_name,
            split=args.split,
            text_column=args.text_column,
            tokenizer_name=args.tokenizer_name,
            max_tokens_per_shard=args.max_tokens_per_shard,
            min_document_tokens=args.min_document_tokens,
            max_documents=args.max_documents,
        )
    )
    print(f"wrote snapshot: {Path(args.output_dir) / 'manifest.json'}")
    print(f"documents={manifest.total_documents} tokens={manifest.total_tokens} shards={len(manifest.shards)}")
    return 0


def _cmd_build_plan(args: argparse.Namespace) -> int:
    manifest = build_sequence_plan(
        SequencePlanSpec(
            snapshot_path=args.snapshot_path,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            stride=args.stride,
            shuffle=not args.no_shuffle,
            shuffle_seed=args.shuffle_seed,
        )
    )
    print(f"wrote plan: {Path(args.output_dir) / 'plan.json'}")
    print(f"samples={manifest.num_samples} sequence_length={manifest.sequence_length} stride={manifest.stride}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    manifest = SnapshotManifest.read_json(args.manifest_path)
    print(f"snapshot_id={manifest.snapshot_id}")
    print(f"dataset_name={manifest.dataset_name}")
    print(f"split={manifest.split}")
    print(f"documents={manifest.total_documents}")
    print(f"tokens={manifest.total_tokens}")
    print(f"shards={len(manifest.shards)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirrordata")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep-parquet")
    prep.add_argument("input_paths", nargs="+")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--snapshot-id", required=True)
    prep.add_argument("--dataset-name", required=True)
    prep.add_argument("--split", default="train")
    prep.add_argument("--text-column", default="text")
    prep.add_argument("--tokenizer-name", default="p50k_base")
    prep.add_argument("--max-tokens-per-shard", type=int, default=50_000_000)
    prep.add_argument("--min-document-tokens", type=int, default=1)
    prep.add_argument("--max-documents", type=int)
    prep.set_defaults(func=_cmd_prep_parquet)

    plan = subparsers.add_parser("build-plan")
    plan.add_argument("--snapshot-path", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--sequence-length", type=int, required=True)
    plan.add_argument("--stride", type=int)
    plan.add_argument("--shuffle-seed", type=int, default=0)
    plan.add_argument("--no-shuffle", action="store_true")
    plan.set_defaults(func=_cmd_build_plan)

    info = subparsers.add_parser("info")
    info.add_argument("manifest_path")
    info.set_defaults(func=_cmd_info)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))
