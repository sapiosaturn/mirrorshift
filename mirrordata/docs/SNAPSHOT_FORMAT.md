# Snapshot Format

This document defines the initial `mirrordata` snapshot direction.

## Design Goals

- immutable dataset snapshots
- exact inventory of shards and metadata
- token-first runtime path
- compatible with mmap-friendly local files
- easy future support for global sample-order indices

## Runtime Principle

Training should prefer pretokenized snapshots, not raw text files.

Raw text ingestion remains useful for:

- tiny local experiments
- smoke tests
- preprocessing inputs

It is not the long-term hot path.

## Proposed Layout

```text
snapshot_root/
  manifest.json
  shards/
    tokens-000000.bin
    tokens-000000.idx
    tokens-000001.bin
    tokens-000001.idx
  indices/
    train.sample_idx.npy
    train.shuffle_idx.seed-00042.npy
```

## Manifest Ownership

`manifest.json` is the source of truth for:

- snapshot id and format version
- tokenizer backend and tokenizer name
- token dtype
- shard inventory
- total token count
- total document count
- preprocessing metadata

## Shards

The default shard payload should be contiguous token arrays.

- preferred payload: raw `.bin`
- preferred token dtype: `uint32`
- paired `.idx` file stores document boundaries

`mirrordata` should avoid a more elaborate container format until there is a
real constraint forcing it.

## Ordering

Storage order and sample order should be separate concerns.

That means:

- shard bytes stay immutable
- shuffle lives in explicit order/index files
- resume state can be a single cursor into a known order

This is the main mechanism that keeps deterministic resume simple without
relying on a runtime sampler.
