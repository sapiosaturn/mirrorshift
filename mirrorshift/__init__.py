"""
Mirrorshift - Transformer implementations with various attention mechanisms

This package provides implementations of transformer models with GQA and MLA
attention mechanisms, along with utilities for training, inference, and data handling.
"""

__version__ = "0.1.0"

# Import main classes and functions for easier access
from mirrorshift.modeling.causal_transformers import (
    CausalTransformer,
)
from mirrorshift.utils import ModelConfig, TrainingConfig
from mirrorshift.inference import sample
from mirrorshift.logging_and_metrics import RichLogger

# Make these modules available for import
__all__ = [
    "CausalTransformer",
    "ModelConfig",
    "TrainingConfig",
    "RichLogger",
    "sample",
]