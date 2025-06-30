import torch
import torch.nn as nn
import torch.nn.functional as F

from mirrorshift.modeling.decoder_blocks import DecoderBlock, ParallelDecoderBlock
from mirrorshift.modeling.attention import build_attention_block
from mirrorshift.modeling.ffn import FFN
from mirrorshift.utils import ModelConfig

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    # theta variable is base theta, 10000 in original paper
    # theta ^ -2(i-1)/d
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    # imo more readable than ones_like
    freqs_cis = torch.polar(torch.ones(freqs.size()), freqs)
    # output dimensions are (end, dim//2)
    return freqs_cis

class CausalTransformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig
    ):
        # vocab size is equal for input and output
        super().__init__()
        self.embedding_layer = nn.Embedding(model_config.vocab_size, model_config.embedding_dim)
        if model_config.attention_type == 'gqa':
            freqs_cis = precompute_freqs_cis(model_config.embedding_dim // model_config.num_heads, model_config.context_length)
        elif model_config.attention_type == 'mla':
            freqs_cis = precompute_freqs_cis(model_config.qk_rope_head_dim, model_config.context_length)
        else:
            raise ValueError(f"Unknown attention type: {model_config.attention_type}")
        self.register_buffer("freqs_cis", freqs_cis)

        self.decoder_stack = nn.ModuleList(
            [
                DecoderBlock(
                    attention_block=build_attention_block(model_config),
                    ffn=FFN(
                        model_dim=model_config.embedding_dim,
                        feedforward_dim=model_config.feedforward_dim,
                    )
                )
                for _ in range(model_config.num_layers)
            ]
        )
        self.lm_head = nn.Linear(model_config.embedding_dim, model_config.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is batch size, seq_len where each value is a token index
        output = self.embedding_layer(x)
        output = F.rms_norm(output, (output.size(-1),))
        for layer in self.decoder_stack:
            output = layer(output, self.freqs_cis)
        output = F.rms_norm(output, (output.size(-1),))
        output = self.lm_head(output)
        return output

