# mirrorshift
personal repo for transformers code

- probably not well-organized or optimized, but getting there

## Contents

- `models.py`: Transformer implementations with GQA and MLA attention mechanisms, RoPE embeddings
- `train.py`: Training loop and logging logic
- `data.py`: Basic character-level tokenization and dataset handling
- `utils.py`: Configuration dataclasses and JSON loading
- `inference.py`: Text generation utilities
- `distributed.py`: Functions wrapping most of the distributed training related code
- `config/`: JSON files for model architecture and training parameters

## Installation

### Development Installation

Clone the repository and install it in development mode:

```bash
git clone https://github.com/sapiosaturn/mirrorshift.git
cd mirrorshift
pip install -e .
```

This will install the package in development mode, allowing you to make changes to the code.

### Using UV (Alternative)

If you prefer using UV package manager:

```bash
git clone https://github.com/sapiosaturn/mirrorshift.git
cd mirrorshift
uv sync && source .venv/bin/activate
```

## Usage

### Running as a Package

After installation, you can train a model using:

```bash
mirrorshift-train --model-config mirrorshift/config/model_configs/small.json \
                 --training-config mirrorshift/config/training_configs/small.json \
                 --dataset mirrorshift/datasets/coqa_stories.txt
```

### Running from the Repository

If you didn't install as a package, you can run directly with Python:

```bash
python -m mirrorshift.train --model-config mirrorshift/config/model_configs/small.json \
                           --training-config mirrorshift/config/training_configs/small.json \
                           --dataset mirrorshift/datasets/coqa_stories.txt
```

### Distributed Training

To run on multiple GPUs:

```bash
torchrun --nproc_per_node=<number of GPUs> -m mirrorshift.train
```

with any additional arguments as needed.

### Visualization

Visualize training with tensorboard (loss curves and text samples are saved):

```bash
tensorboard --logdir runs/
```

## Using in Your Python Code

You can also use the components directly in your Python code:

```python
from mirrorshift import CausalTransformer, ModelConfig

# Create a model config
config = ModelConfig(
    vocab_size=50000,
    num_layers=12,
    embedding_dim=768,
    num_heads=12,
    num_kv_heads=4,
    context_length=1024,
    feedforward_dim=3072,
    attention_dropout_p=0.1,
    residual_dropout_p=0.1,
    attention_type="gqa"
)

# Initialize a model
model = CausalTransformer(model_config=config)
```