# mirrorshift
Experimental decoder-only transformer repo in PyTorch.

Current focus is architecture experiments (GQA and MLA attention), a simple training loop, and local text generation sampling.

## Repository Structure

```text
mirrorshift/
  __init__.py
  train.py
  data.py
  inference.py
  logging_and_metrics.py
  utils.py
  modeling/
    __init__.py
    causal_transformers.py
    attention.py
    decoder_blocks.py
    ffn.py
  config/
    model_configs/
      small.json
    training_configs/
      small.json
  datasets/
    coqa_stories.txt
```

## Module Map

- `mirrorshift/train.py`: CLI entrypoint and end-to-end training loop.
- `mirrorshift/data.py`: text dataset wrappers (`CharacterTxtDataset`, `TiktokenTxtDataset`).
- `mirrorshift/inference.py`: token sampling helpers (`top_p`, `min_p`, autoregressive sampling).
- `mirrorshift/logging_and_metrics.py`: Rich live UI for training progress + validation/sample display.
- `mirrorshift/utils.py`: config dataclasses, JSON config loaders, LR schedule helpers.
- `mirrorshift/modeling/causal_transformers.py`: `CausalTransformer` and RoPE frequency precomputation.
- `mirrorshift/modeling/attention.py`: GQA and MLA attention blocks plus builder utility.
- `mirrorshift/modeling/decoder_blocks.py`: sequential and parallel decoder block variants.
- `mirrorshift/modeling/ffn.py`: feedforward block and activation helpers.
- `mirrorshift/config/model_configs/small.json`: default tiny model config.
- `mirrorshift/config/training_configs/small.json`: default training config.
- `mirrorshift/datasets/coqa_stories.txt`: sample training corpus.

## Installation

### Editable Install

```bash
git clone https://github.com/sapiosaturn/mirrorshift.git
cd mirrorshift
python3 -m pip install -e .
```

### UV Workflow

```bash
git clone https://github.com/sapiosaturn/mirrorshift.git
cd mirrorshift
uv sync
source .venv/bin/activate
```

## Training

### Package CLI

```bash
mirrorshift-train --model-config mirrorshift/config/model_configs/small.json \
                  --training-config mirrorshift/config/training_configs/small.json \
                  --dataset mirrorshift/datasets/coqa_stories.txt
```

### Module Run

```bash
python3 -m mirrorshift.train --model-config mirrorshift/config/model_configs/small.json \
                             --training-config mirrorshift/config/training_configs/small.json \
                             --dataset mirrorshift/datasets/coqa_stories.txt
```

## Monitoring

```bash
tensorboard --logdir runs/
```

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
    attention_dropout_p=0.05,
    residual_dropout_p=0.05,
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
