import torch

from mirrorshift.modeling.attention import build_attention_block
from mirrorshift.modeling.causal_transformers import CausalTransformer, precompute_freqs_cis
from mirrorshift.modeling.decoder_blocks import DecoderBlock, ParallelDecoderBlock
from mirrorshift.modeling.ffn import FFN
from mirrorshift.config import ModelConfig


def build_model_config(attention_type: str) -> ModelConfig:
    common = {
        "vocab_size": 64,
        "num_layers": 2,
        "num_kv_heads": 2,
        "embedding_dim": 32,
        "num_heads": 4,
        "context_length": 16,
        "feedforward_dim": 64,
    }
    if attention_type == "gqa":
        return ModelConfig(attention_type="gqa", **common)
    return ModelConfig(
        attention_type="mla",
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        **common,
    )


def test_causal_transformer_forward_shape_gqa() -> None:
    model = CausalTransformer(build_model_config("gqa"))
    x = torch.randint(0, 64, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 64)


def test_causal_transformer_backward_single_step_mla() -> None:
    model = CausalTransformer(build_model_config("mla"))
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)


def test_decoder_block_forward_shape() -> None:
    config = build_model_config("gqa")
    attention = build_attention_block(config)
    block = DecoderBlock(
        attention_block=attention,
        ffn=FFN(model_dim=config.embedding_dim, feedforward_dim=config.feedforward_dim),
    )
    x = torch.randn(2, 8, config.embedding_dim)
    freqs = precompute_freqs_cis(config.embedding_dim // config.num_heads, config.context_length)
    out = block(x, freqs)
    assert out.shape == x.shape


def test_parallel_decoder_block_forward_shape() -> None:
    config = build_model_config("gqa")
    attention = build_attention_block(config)
    block = ParallelDecoderBlock(
        attention_block=attention,
        ffn=FFN(model_dim=config.embedding_dim, feedforward_dim=config.feedforward_dim),
    )
    x = torch.randn(2, 8, config.embedding_dim)
    freqs = precompute_freqs_cis(config.embedding_dim // config.num_heads, config.context_length)
    out = block(x, freqs)
    assert out.shape == x.shape
