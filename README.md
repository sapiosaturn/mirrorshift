# mirrorshift
Experimental decoder-only transformer repo in PyTorch.

Current focus is architecture experiments (GQA and MLA attention), a simple training loop, and local text generation sampling.

## Repository Structure

```text
mirrorshift/
  mirrordata/
    pyproject.toml
    src/mirrordata/
  __init__.py
  train.py
  utils.py
  modeling/
    __init__.py
    causal_transformers.py
    attention.py
    decoder_blocks.py
    ffn.py
  experiments/
    __init__.py
    spec.py
    default.py
  config/
    train_configs/
      small.toml
  datasets/
    example_train.parquet
    example_train_snapshot/
    example_train_plan_ctx64/
```

## Module Map

- `mirrorshift/train.py`: CLI entrypoint and end-to-end training loop.
- `mirrordata/`: local workspace package for data contracts, tokenizers, and dataset runtime scaffolding.
- `mirrorshift/config/`: unified dataclass schema + TOML/CLI config manager.
- `mirrorshift/utils.py`: LR schedule helpers.
- `mirrorshift/modeling/causal_transformers.py`: `CausalTransformer` and RoPE frequency precomputation.
- `mirrorshift/modeling/attention.py`: GQA and MLA attention blocks plus builder utility.
- `mirrorshift/modeling/decoder_blocks.py`: sequential and parallel decoder block variants.
- `mirrorshift/modeling/ffn.py`: feedforward block and activation helpers.
- `mirrorshift/experiments/spec.py`: experiment contract (`TrainSpec`) for model/data/loss composition.
- `mirrorshift/experiments/default.py`: default causal LM experiment wiring.
- `mirrorshift/config/train_configs/small.toml`: default unified run/model/training config.
- `mirrorshift/datasets/example_train.parquet`: sample parquet source corpus for preprocessing examples.
- `mirrorshift/datasets/example_train_snapshot/`: tracked preprocessed snapshot used by the default training config.
- `mirrorshift/datasets/example_train_plan_ctx64/`: tracked sequence plan for `context_length = 64`.

## Installation

### UV Workspace

```bash
git clone https://github.com/sapiosaturn/mirrorshift.git
cd mirrorshift
uv sync
```

This installs both workspace packages:

- `mirrorshift`
- `mirrordata`

## Training

### Package CLI

```bash
mirrorshift-train --job.config_file mirrorshift/config/train_configs/small.toml \
                  --training.max_steps 100 \
                  --training.log_every 10
```

The default config already points at prebuilt `mirrordata` artifacts.

### Preprocessing

Build a snapshot from parquet:

```bash
mirrordata prep-parquet mirrorshift/datasets/example_train.parquet \
  --output-dir /tmp/example-train-snapshot \
  --snapshot-id example-train \
  --dataset-name example-train
```

Build a plan for a specific context length:

```bash
mirrordata build-plan \
  --snapshot-path /tmp/example-train-snapshot \
  --output-dir /tmp/example-train-plan-ctx64 \
  --sequence-length 64
```

### Module Run

```bash
python3 -m mirrorshift.train --job.config_file mirrorshift/config/train_configs/small.toml \
                             --training.max_steps 100 \
                             --data.snapshot_path /tmp/example-train-snapshot \
                             --data.plan_path /tmp/example-train-plan-ctx64
```

## Monitoring

```bash
mirrorshift-train --job.config_file mirrorshift/config/train_configs/small.toml \
                  --run.wandb_mode online \
                  --run.wandb_project mirrorshift
```

## Run Artifacts

Each training invocation creates an immutable run directory:

```text
runs/<run_id>/
  config.json
  manifest.json
```

- `config.json`: resolved dataclass config snapshot.
- `manifest.json`: run metadata (snapshot/plan paths, device, params, argv, paths).
- metrics: logged to Weights & Biases.

Set a fixed run id with `--run.id <name>` or let mirrorshift auto-generate one.

## Programmatic Use

```python
from mirrorshift import CausalTransformer, ModelConfig

config = ModelConfig(
    vocab_size=50281,
    num_layers=2,
    num_kv_heads=4,
    embedding_dim=128,
    num_heads=8,
    context_length=64,
    feedforward_dim=384,
    attention_type="mla",
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=64,
)

model = CausalTransformer(model_config=config)
```

## Notes

- There is currently no distributed training module in this repository.
- This project is intended for CUDA-focused development; CPU fallback exists but will be slower.
