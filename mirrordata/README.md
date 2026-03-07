# mirrordata

`mirrordata` is the data package for `mirrorshift`.

Current goals:

- keep runtime training token-first
- keep snapshot format explicit and versioned
- keep preprocessing modular so heavier pipelines can plug in later
- support only `tiktoken` for now

The package intentionally separates:

- runtime datasets and tokenizers
- snapshot metadata contracts
- preprocessing interfaces

That split lets `mirrorshift` depend on a narrow, stable API while future
preprocessing code can evolve independently.
