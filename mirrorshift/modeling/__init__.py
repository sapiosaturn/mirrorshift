"""
MirrorShift Modeling Module

This module contains the core neural network components for transformer models,
including attention mechanisms, decoder blocks, feed-forward networks, and the
main CausalTransformer model.
"""

from .causal_transformers import CausalTransformer, precompute_freqs_cis
from .decoder_blocks import DecoderBlock, ParallelDecoderBlock
from .ffn import FFN, relu_square
from .attention import (
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    build_attention_block,
)

__all__ = [
    "CausalTransformer",
    "precompute_freqs_cis",
    "DecoderBlock", 
    "ParallelDecoderBlock",
    "FFN",
    "relu_square",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention", 
    "build_attention_block",
]
