import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_O = nn.Linear(head_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V
        return self.out_drop(self.W_O(out))


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the (cos, sin) lookup table for rotary position embeddings.

    head_dim must be even. Returns two tensors of shape ``(max_seq_len, head_dim // 2)``.

    The i-th frequency is ``θ_i = base ** (-2i / head_dim)`` for ``i ∈ [0, head_dim/2)``;
    the angle for position ``m`` and pair ``i`` is ``θ_i · m``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x`` of shape ``(..., T, head_dim)``.

    Pairs are formed by even/odd dim indices: dim ``2i`` and dim ``2i+1`` rotate
    together by angle ``θ_i · pos``.
    """
    T = x.shape[-2]
    cos_t = cos[:T]  # (T, head_dim // 2)
    sin_t = sin[:T]
    # Broadcast over leading dims (e.g., batch and head)
    while cos_t.dim() < x.dim() - 1:
        cos_t = cos_t.unsqueeze(0)
        sin_t = sin_t.unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos_t - x_odd * sin_t
    rotated_odd = x_even * sin_t + x_odd * cos_t
    out = torch.stack([rotated_even, rotated_odd], dim=-1)
    return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        position_type: str = "learned",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.position_type = position_type

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

        if position_type == "rope":
            cos, sin = precompute_rope_cache(self.head_dim, max_seq_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.position_type == "rope":
            Q = apply_rope(Q, self.rope_cos, self.rope_sin)
            K = apply_rope(K, self.rope_cos, self.rope_sin)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_drop(self.W_O(out))
