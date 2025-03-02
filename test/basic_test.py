# placeholder tests for making sure forward passes are dimensionally correct
# also making sure they torch.compile

import torch
from mirrorshift.models import GroupedQueryAttention, MultiHeadLatentAttention, MLP, DecoderBlock, CausalTransformer
from mirrorshift.utils import ModelConfig

def precompute_freqs_cis(dim, end, theta = 10000.0):
    # theta variable is base theta, 10000 in original paper
    # theta ^ -2(i-1)/d
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    # imo more readable than ones_like
    freqs_cis = torch.polar(torch.ones(freqs.size()), freqs)
    # output dimensions are (end, dim//2)
    return freqs_cis

def test_gqa():
    mha_module = GroupedQueryAttention(
        embedding_dim=16,
        num_heads=4,
        num_kv_heads=2,
        context_length=4,
        attention_dropout_p=0,
        residual_dropout_p=0
    )
    mha_module = torch.compile(mha_module)
    # batch size of 5, seq len of 3, embedding dim of 16
    example_input = torch.randn((5, 3, 16))
    test_output = mha_module(example_input, precompute_freqs_cis(dim=4, end=4))
    assert test_output.size() == example_input.size()

def test_mlp():
    mlp_module = MLP(
        model_dim=8, 
        feedforward_dim=16
    )
    mlp_module = torch.compile(mlp_module)
    # batch size of 5, seq len of 3, embedding dim of 8
    example_input = torch.randn((5, 3, 8))
    test_output = mlp_module(example_input)
    assert test_output.size() == example_input.size()

def test_decoder_block_gqa():
    model_config = ModelConfig(
        vocab_size=12,
        num_layers=3,
        embedding_dim=8,
        num_heads=2,
        num_kv_heads=2,
        context_length=4,
        feedforward_dim=16,
        attention_dropout_p=0,
        residual_dropout_p=0,
        attention_type="gqa"
    )
    decoder_block_module = DecoderBlock(model_config=model_config)
    decoder_block_module = torch.compile(decoder_block_module)
    # batch size of 5, seq len of 3, embedding dim of 8
    example_input = torch.randn((5, 3, 8))
    test_output = decoder_block_module(example_input, precompute_freqs_cis(4, 4))
    assert test_output.size() == example_input.size()

def test_multi_head_latent_attention():
    # Make sure the rope dimension is even (for complex number representation)
    qk_rope_head_dim = 2  # Must be even for reshape to work properly
    mla_module = MultiHeadLatentAttention(
        embedding_dim=16,
        num_heads=4,
        num_kv_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=2,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=4,
        context_length=4,
        attention_dropout_p=0,
        residual_dropout_p=0
    )
    mla_module = torch.compile(mla_module)
    # batch size of 5, seq len of 3, embedding dim of 16
    example_input = torch.randn((5, 3, 16))
    test_output = mla_module(example_input, precompute_freqs_cis(dim=qk_rope_head_dim, end=4))
    assert test_output.size() == example_input.size()

def test_transformer_with_gqa():
    model_config = ModelConfig(
        vocab_size=12,
        num_layers=3,
        embedding_dim=8,
        num_heads=2,
        num_kv_heads=2,
        context_length=4,
        feedforward_dim=16,
        attention_dropout_p=0,
        residual_dropout_p=0,
        attention_type="gqa"
    )
    dec_only_transformer_module = CausalTransformer(model_config=model_config)
    dec_only_transformer_module = torch.compile(dec_only_transformer_module)
    # batch size of 5, seq len of 3
    example_input = torch.randint(low=0, high=12, size=(5, 3))
    test_output = dec_only_transformer_module(example_input)
    # output should be dims (batch_size, sequence_length, vocab_size)
    # predicted logits are output[:, -1, :]
    assert test_output.size() == torch.Size((5,3,12))

def test_decoder_block_mla():
    # Make sure the rope dimension is even (for complex number representation)
    qk_rope_head_dim = 2  # Must be even for reshape to work properly
    model_config = ModelConfig(
        vocab_size=12,
        num_layers=3,
        embedding_dim=16,
        num_heads=4,
        num_kv_heads=2,
        context_length=4,
        feedforward_dim=32,
        attention_dropout_p=0,
        residual_dropout_p=0,
        attention_type="mla",
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=2,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=4
    )
    decoder_block_module = DecoderBlock(model_config=model_config)
    decoder_block_module = torch.compile(decoder_block_module)
    # batch size of 5, seq len of 3, embedding dim of 16
    example_input = torch.randn((5, 3, 16))
    test_output = decoder_block_module(example_input, precompute_freqs_cis(dim=qk_rope_head_dim, end=4))
    assert test_output.size() == example_input.size()

def test_transformer_with_mla():
    # Make sure the rope dimension is even (for complex number representation)
    qk_rope_head_dim = 2  # Must be even for reshape to work properly
    model_config = ModelConfig(
        vocab_size=12,
        num_layers=3,
        embedding_dim=16,
        num_heads=4,
        num_kv_heads=2,
        context_length=4,
        feedforward_dim=32,
        attention_dropout_p=0,
        residual_dropout_p=0,
        attention_type="mla",
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=2,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=4
    )
    dec_only_transformer_module = CausalTransformer(model_config=model_config)
    dec_only_transformer_module = torch.compile(dec_only_transformer_module)
    # batch size of 5, seq len of 3
    example_input = torch.randint(low=0, high=12, size=(5, 3))
    test_output = dec_only_transformer_module(example_input)
    # output should be dims (batch_size, sequence_length, vocab_size)
    # predicted logits are output[:, -1, :]
    assert test_output.size() == torch.Size((5,3,12))

