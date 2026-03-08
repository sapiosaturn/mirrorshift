import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from mirrorshift.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ModelConfig,
    ParallelismConfig,
)
from mirrorshift.infra import RuntimeContext
from mirrorshift.infra.activation_checkpoint import apply_activation_checkpointing
from mirrorshift.infra.compile import apply_compile
from mirrorshift.infra.discovery import get_transformer_block_container
from mirrorshift.infra.distributed import build_runtime_context
from mirrorshift.infra.parallel_dims import ParallelDims
from mirrorshift.infra.parallelize import apply_ddp, apply_fsdp
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.runtime import materialize_initialized_model


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        num_layers=2,
        num_kv_heads=2,
        embedding_dim=32,
        num_heads=4,
        context_length=8,
        feedforward_dim=64,
        attention_type="gqa",
    )


@contextmanager
def single_rank_process_group():
    if dist.is_initialized():
        yield
        return

    path = Path(tempfile.mkdtemp()) / "pg"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _single_rank_runtime_context() -> RuntimeContext:
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    parallel_dims.build_mesh("cpu")
    return RuntimeContext(
        device=torch.device("cpu"),
        device_type="cpu",
        world_size=1,
        global_rank=0,
        local_rank=0,
        parallel_dims=parallel_dims,
    )


def test_parallel_dims_builds_degenerate_single_rank_mesh() -> None:
    with single_rank_process_group():
        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
        parallel_dims.build_mesh("cpu")

        assert parallel_dims.get_mesh("dp_replicate").size() == 1
        assert parallel_dims.get_mesh("fsdp").size() == 1
        assert parallel_dims.get_mesh(["dp_replicate", "fsdp"]).size() == 1


def test_build_runtime_context_rejects_distributed_env_without_parallelism(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process execution"):
        build_runtime_context("cpu", ParallelismConfig())


def test_get_transformer_block_container_finds_decoder_stack() -> None:
    model = CausalTransformer(tiny_model_config())
    name, container = get_transformer_block_container(model)

    assert name == "decoder_stack"
    assert len(container) == 2


def test_apply_activation_checkpointing_wraps_blocks() -> None:
    model = CausalTransformer(tiny_model_config())
    original_block = model.decoder_stack[0]

    apply_activation_checkpointing(
        model,
        ActivationCheckpointConfig(mode="full"),
    )

    wrapped_block = model.decoder_stack[0]
    assert wrapped_block is not original_block
    assert hasattr(wrapped_block, "_checkpoint_wrapped_module")


def test_apply_compile_compiles_each_transformer_block() -> None:
    model = CausalTransformer(tiny_model_config())

    compiled = apply_compile(
        model,
        CompileConfig(enable=True, backend="eager", fullgraph=True),
    )

    assert compiled is model
    assert type(model.decoder_stack[0]).__name__ == "OptimizedModule"


def test_apply_ddp_one_rank_smoke() -> None:
    with single_rank_process_group():
        runtime_context = _single_rank_runtime_context()
        model = CausalTransformer(tiny_model_config())

        apply_ddp(
            model,
            runtime_context,
            ParallelismConfig(bucket_cap_mb=50),
            CompileConfig(enable=False),
        )

        x = torch.randint(0, 32, (2, 8))
        y = model(x)
        y.sum().backward()

        assert y.shape == (2, 8, 32)


def test_apply_fsdp_one_rank_smoke() -> None:
    with single_rank_process_group():
        runtime_context = _single_rank_runtime_context()
        model = CausalTransformer(tiny_model_config())

        apply_fsdp(
            model,
            runtime_context,
            ParallelismConfig(),
        )

        x = torch.randint(0, 32, (2, 8))
        y = model(x)
        y.sum().backward()

        assert y.shape == (2, 8, 32)


def test_apply_fsdp_on_meta_model_materializes_after_wrapping() -> None:
    with single_rank_process_group():
        runtime_context = _single_rank_runtime_context()
        with torch.device("meta"):
            model = CausalTransformer(tiny_model_config())

        apply_fsdp(
            model,
            runtime_context,
            ParallelismConfig(),
        )
        materialize_initialized_model(model, "cpu")

        x = torch.randint(0, 32, (2, 8))
        y = model(x)
        y.sum().backward()

        assert y.shape == (2, 8, 32)
