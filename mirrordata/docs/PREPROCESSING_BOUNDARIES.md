# Preprocessing Boundaries

`mirrordata` should own the runtime contracts and snapshot format, not every
possible preprocessing workflow.

## Package Responsibilities

`mirrordata` should provide:

- tokenizer backends used by the snapshot format
- document and token contracts
- manifest and shard metadata models
- writer interfaces for the `mirrordata` snapshot format
- runtime dataset/loader helpers that read `mirrordata` snapshots

## Future External Preprocessing Libraries

A separate preprocessing package should be able to plug in:

- document sources
- cleaning and filtering transforms
- packing and sampling policies
- deduplication
- quality scoring
- multi-process or distributed execution

As long as that package can emit `mirrordata` documents/tokens and call the
`mirrordata` writer interfaces, it should not need to know about
`mirrorshift`.

## Pipeline Shape

```text
DocumentSource
  -> TextTransform
  -> TokenizerBackend
  -> TokenTransform
  -> SnapshotWriter
```

## Immediate Scope

The first implementation stays intentionally small:

- `tiktoken` only
- parquet input with a `text` column
- snapshot manifest dataclasses
- raw `.bin` shard writing plus `.idx` document indices
- sequence-plan generation for causal LM training
- deterministic runtime dataset/loader APIs
