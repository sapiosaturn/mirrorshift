# Snapshot Format

This document defines the initial implemented `mirrordata` snapshot format.

## Design Goals

- immutable dataset snapshots
- exact inventory of shards and metadata
- token-first runtime path
- compatible with mmap-friendly local files
- easy future support for global sample-order indices

## Runtime Principle

Training should prefer pretokenized snapshots built from parquet files, not
runtime text-file tokenization.

## Implemented Layout

```text
snapshot_root/
  manifest.json
  shards/
    train-tokens-00000.bin
    train-tokens-00000.idx
    train-tokens-00001.bin
    train-tokens-00001.idx
plan_root/
  plan.json
  sample_starts.npy
  sample_order.npy
```

## Manifest Ownership

`manifest.json` is the source of truth for:

- snapshot id and format version
- dataset name and split
- tokenizer backend and tokenizer name
- token dtype
- shard inventory
- total token count
- total document count
- preprocessing metadata

## Shards

The default shard payload is a contiguous raw token array.

- preferred payload: raw `.bin`
- token dtype: `uint32`
- paired `.idx` file stores document boundaries
- shard manifests also record absolute token offset ranges

`mirrordata` should avoid a more elaborate container format until there is a
real constraint forcing it.

## Ordering

Storage order and sample order are separate concerns.

The snapshot layer stores immutable token bytes. A separate plan directory stores:

- `sample_starts.npy`: all legal sequence starts for a given `sequence_length` and `stride`
- `sample_order.npy`: the deterministic access order over those starts
- `plan.json`: metadata describing the plan

This is the main mechanism that keeps deterministic resume simple without
relying on a runtime sampler.
