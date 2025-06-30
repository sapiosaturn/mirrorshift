import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

def relu_square(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()

class FFN(nn.Module):
    # position-wise FF layer
    def __init__(
        self,
        model_dim: int,
        feedforward_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = relu_square,
        gated: bool = False
    ):
        super().__init__()
        self.to_hidden = nn.Linear(model_dim, feedforward_dim, bias=False)
        self.from_hidden = nn.Linear(feedforward_dim, model_dim, bias=False)
        self.act_fn = act_fn
        self.gated = gated
        if self.gated:
            self.gate = nn.Linear(model_dim, feedforward_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dimensions of x are batch_size, seq_length, embedding_dim
        # which is also batch_size, seq_length, model_dim
        if self.gated:
            return self.from_hidden(self.act_fn(self.to_hidden(x) * self.gate(x)))
        else:
            return self.from_hidden(self.act_fn(self.to_hidden(x)))