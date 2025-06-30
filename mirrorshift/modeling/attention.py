import torch
import torch.nn as nn
import torch.nn.functional as F
from mirrorshift.utils import ModelConfig

class GroupedQueryAttention(nn.Module):
    # for grouped query attention, many queries are grouped together with a single key and value matrix
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        context_length: int,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.per_head_dim = self.embedding_dim // self.num_heads
        self.context_length = context_length

        # W_Q, W_K, W_V for all attention heads
        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(
            embedding_dim, self.num_kv_heads * self.per_head_dim, bias=False
        )
        self.V = nn.Linear(
            embedding_dim, self.num_kv_heads * self.per_head_dim, bias=False
        )
        self.output_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_heads, per_head_dim = x.size()
        # reshaping does the "division into d/2" subspaces
        # viewed as complex so you can simply multiply the complex number
        # x+iy by the precomputed cos theta + i sin theta, rotating vector (x,y) by theta
        x_complex = torch.view_as_complex(
            x.reshape(batch_size, seq_length, num_heads, per_head_dim // 2, 2)
        )
        freqs_cis = freqs_cis[:seq_length].view(
            1, seq_length, 1, per_head_dim // 2
        )  # flatten for multiply
        x_rotated = x_complex * freqs_cis
        # reshape back to normal
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch_size, seq_length, num_heads, per_head_dim)
        return x_out

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        # each q, k, v matrix is (batch size, seq_length, embedding_dim, embedding_dim) here
        # forward pass on the attention_matrices linear layer results in calculating queries, keys, values
        # corresponding to the tokens that are there
        # transposing here so next operations are done per attention head
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.per_head_dim)
        K = K.view(batch_size, seq_length, self.num_kv_heads, self.per_head_dim)
        V = V.view(
            batch_size, seq_length, self.num_kv_heads, self.per_head_dim
        ).transpose(1, 2)

        Q = F.rms_norm(Q, (Q.size(-1),))
        K = F.rms_norm(K, (K.size(-1),))
        Q = self.apply_rope(Q, freqs_cis).transpose(1, 2)
        K = self.apply_rope(K, freqs_cis).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )

        # transpose to move num_heads back to original dim and then recombine
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.embedding_dim)
        )
        output = self.output_proj(output)
        return output
    
class MultiHeadLatentAttention(nn.Module):
    # for MLA, queries, keys, and values are all projected down to a low-rank latent tensor
    # before being up-projected for the attention mechanism
    # one latent for queries and a joint latent for keys and values
    # this lets the joint latents be cached instead of the entire key and value pair
    # the reason head_dims are specified here is because deepseek tends to use
    # head_dim values greater than embedding_dim / num_heads
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        context_length: int,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim

        self.context_length = context_length

        # W_Q, W_K, W_V for all attention heads
        self.Q_a = nn.Linear(self.embedding_dim, self.q_lora_rank, bias=False)
        # note, num_heads * qk_head_dim does not have to be equal to embedding_dim
        self.Q_b = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # joint down-projection to kv_lora as well as decoupled part of key for rope
        # this could be two matrices but it's done jointly
        self.KV_a = nn.Linear(
            self.embedding_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        # upscales for the nope part of the key and for the value
        # this could be two matrices but it's done jointly
        self.KV_b = nn.Linear(
            self.kv_lora_rank,
            self.num_kv_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )
        self.output_proj = nn.Linear(
            self.num_heads * self.v_head_dim, embedding_dim, bias=False
        )
        nn.init.zeros_(self.output_proj.weight)

    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_heads, per_head_dim = x.size()
        # reshaping does the "division into d/2" subspaces
        # viewed as complex so you can simply multiply the complex number
        # x+iy by the precomputed cos theta + i sin theta, rotating vector (x,y) by theta
        x_complex = torch.view_as_complex(
            x.reshape(batch_size, seq_length, num_heads, per_head_dim // 2, 2)
        )
        freqs_cis = freqs_cis[:seq_length].view(
            1, seq_length, 1, per_head_dim // 2
        )  # flatten for multiply
        x_rotated = x_complex * freqs_cis
        # reshape back to normal
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch_size, seq_length, num_heads, per_head_dim)
        return x_out

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        q = self.Q_a(x)
        q = F.rms_norm(q, (q.size(-1),))  # low rank is normalized
        q = self.Q_b(q)

        q = q.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_rope = self.apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)

        # for keys, we want the up-projection for rope to be decoupled
        # since rope is position-sensitive, we can't cache rope-d keys
        # cache is not relevant for this code (yet) but explains why the rope-d
        # part of the keys are decoupled - refer to deepseek paper for more
        kv_latent_plus_rope = self.KV_a(x)
        kv_latent, k_rope = torch.split(
            kv_latent_plus_rope, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # k_rope here is common for all attn heads
        k_rope = k_rope.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        k_rope = k_rope.expand(
            batch_size, seq_length, self.num_kv_heads, self.qk_rope_head_dim
        )
        k_rope = self.apply_rope(k_rope, freqs_cis).transpose(1, 2)

        kv_latent = F.rms_norm(kv_latent, (kv_latent.size(-1),))
        kv = self.KV_b(kv_latent)
        kv = kv.view(
            batch_size,
            seq_length,
            self.num_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )

        # transpose to move num_heads back to original dim and then recombine
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.num_heads * self.v_head_dim)
        )
        output = self.output_proj(output)
        return output
    

def build_attention_block(model_config: ModelConfig) -> nn.Module:
    if model_config.attention_type == "gqa":
        return GroupedQueryAttention(
            embedding_dim=model_config.embedding_dim,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            context_length=model_config.context_length,
        )
    elif model_config.attention_type == 'mla':
        return MultiHeadLatentAttention(
            embedding_dim=model_config.embedding_dim,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            q_lora_rank=model_config.q_lora_rank,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            v_head_dim=model_config.v_head_dim,
            context_length=model_config.context_length,
        )
    else:
        raise ValueError(f"Unknown attention type: {model_config.attention_type}")