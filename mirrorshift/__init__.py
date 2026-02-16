"""
Mirrorshift - Transformer implementations with various attention mechanisms

This package provides implementations of transformer models with GQA and MLA
attention mechanisms, along with utilities for training and data handling.
"""

__version__ = "0.1.0"

# Import main classes and functions for easier access
from mirrorshift.modeling.causal_transformers import (
    CausalTransformer,
)
from mirrorshift.config import ConfigManager, JobConfig, ModelConfig, TrainingConfig

# Make these modules available for import
__all__ = [
    "CausalTransformer",
    "ConfigManager",
    "JobConfig",
    "ModelConfig",
    "TrainingConfig",
]
