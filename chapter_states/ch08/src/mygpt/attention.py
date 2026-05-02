import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    """Single-head causal self-attention with a registered causal mask and dropout."""

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


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention.

    Inputs:
        x: tensor of shape (B, T, embed_dim).

    Outputs:
        tensor of shape (B, T, embed_dim).

    Constructor arguments:
        embed_dim:    width of the input/output embedding axis (C).
        num_heads:    number of parallel heads (h). Must divide embed_dim.
        max_seq_len:  the largest sequence length the module is willing to
                      process. The causal mask is allocated once with this size
                      in __init__ and sliced down in forward.
        dropout:      probability of zeroing each entry in the attention weights
                      and in the output projection.

    Each head operates in head_dim = embed_dim // num_heads dimensions; the
    heads run in parallel via tensor reshape, and their outputs are
    concatenated along the channel axis before a final embed_dim x embed_dim
    output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
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

        # One C x C projection per role. After the reshape in forward,
        # the first head_dim output channels of W_Q go to head 0, the
        # next head_dim go to head 1, and so on.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        # (B, T, C) -> three (B, T, C) tensors
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split the C axis into (num_heads, head_dim) and move the head axis
        # next to the batch axis: (B, T, C) -> (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention, batched over (B, num_heads).
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, h, T, T)
        scores = scores + self.causal_mask[:T, :T]                   # broadcast (T,T) over (B,h)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V                                            # (B, h, T, head_dim)

        # Undo the reshape, concatenate heads back into the C axis,
        # apply the output projection.
        out = out.transpose(1, 2).contiguous().view(B, T, C)         # (B, T, C)
        return self.out_drop(self.W_O(out))
