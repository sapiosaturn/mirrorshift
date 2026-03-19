# mirrordata

`mirrordata` is the data package for `mirrorshift`.

Current goals:

- keep runtime training token-first
- keep snapshot format explicit and versioned
- keep preprocessing modular so heavier pipelines can plug in later
- support only `tiktoken` for now

The package intentionally separates:

- runtime readers/loaders and tokenizers
- snapshot metadata contracts
- preprocessing interfaces

That split lets `mirrorshift` depend on a narrow, stable API while future
preprocessing code can evolve independently.

## Current Capabilities

- build immutable token snapshots from parquet files with a `text` column
- write raw `uint32` token shards plus `.idx` document-boundary files
- write a versioned `manifest.json` snapshot inventory
- build sequence plans with precomputed sample starts and shuffled order files
- expose a deterministic map-style dataset and a resumable batch loader
- provide a small CLI:
  - `mirrordata prep-parquet`
  - `mirrordata build-plan`
  - `mirrordata info`

## Current Layout

```text
snapshot_root/
  manifest.json
  shards/
    train-tokens-00000.bin
    train-tokens-00000.idx
    ...

plan_root/
  plan.json
  sample_starts.npy
  sample_order.npy
```
