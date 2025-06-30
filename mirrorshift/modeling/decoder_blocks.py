import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: nn.Module,
        ffn: nn.Module
    ):
        super().__init__()
        self.attention_block = attention_block
        self.ff_block = ffn

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # should be straightforward, dimensions stay the same
        output = F.rms_norm(x, (x.size(-1),))  # last dim is embedding_dim
        output = output + self.attention_block(output, freqs_cis)
        output = F.rms_norm(output, (output.size(-1),))  # last dim is embedding_dim
        output = output + self.ff_block(output)
        return output
    
class ParallelDecoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: nn.Module,
        ffn: nn.Module
    ):
        super().__init__()
        self.attention_block = attention_block
        self.ff_block = ffn

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.rms_norm(x, (x.size(-1),))
        attn_output = self.attention_block(x, freqs_cis)
        ffn_output = self.ff_block(x)
        output = residual + attn_output + ffn_output
        return output