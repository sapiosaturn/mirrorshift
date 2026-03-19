import os
import random

import torch

from mirrorshift.config import DebugConfig, ModelConfig
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.runtime import (
    build_meta_initialized_model,
    build_meta_model,
    materialize_initialized_model,
    set_determinism,
)


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        num_layers=1,
        num_kv_heads=2,
        embedding_dim=32,
        num_heads=4,
        context_length=8,
        feedforward_dim=64,
        attention_type="gqa",
    )


def test_set_determinism_reseeds_python_and_torch() -> None:
    set_determinism("cpu", DebugConfig(seed=123, deterministic=False))
    first_python = random.random()
    first_torch = torch.rand(4)

    set_determinism("cpu", DebugConfig(seed=123, deterministic=False))
    second_python = random.random()
    second_torch = torch.rand(4)

    assert first_python == second_python
    assert torch.equal(first_torch, second_torch)
    assert os.environ["PYTHONHASHSEED"] == str(123)


def test_set_determinism_toggles_deterministic_algorithms() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        set_determinism("cpu", DebugConfig(deterministic=True))
        assert torch.are_deterministic_algorithms_enabled() is True

        set_determinism("cpu", DebugConfig(deterministic=False))
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


def test_build_meta_initialized_model_materializes_weights() -> None:
    model = build_meta_initialized_model(CausalTransformer, tiny_model_config(), "cpu")

    assert model.embedding_layer.weight.device.type == "cpu"
    assert not model.embedding_layer.weight.is_meta

    x = torch.randint(0, 32, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 32)


def test_build_meta_model_stays_on_meta_until_materialized() -> None:
    model = build_meta_model(CausalTransformer, tiny_model_config())

    assert model.embedding_layer.weight.is_meta

    materialized = materialize_initialized_model(model, "cpu")
    assert materialized is model
    assert not model.embedding_layer.weight.is_meta
