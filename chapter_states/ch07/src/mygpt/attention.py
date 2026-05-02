import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    """Single-head causal self-attention with a registered causal mask and dropout.

    Inputs:
        x: tensor of shape (B, T, embed_dim).

    Outputs:
        tensor of shape (B, T, embed_dim).

    Constructor arguments:
        embed_dim:    width of the input/output embedding axis (C).
        head_dim:     width of the head's internal Q/K/V (d_h). For single-head
                      we set head_dim = embed_dim; multi-head (Chapter 8) sets
                      head_dim < embed_dim and runs several heads in parallel.
        max_seq_len:  the largest sequence length the module is willing to
                      process. The causal mask is allocated once with this size
                      in __init__ and sliced down in forward.
        dropout:      probability of zeroing each entry in the attention weights
                      and in the output projection. Default 0.0 reproduces the
                      Chapter 6 behaviour exactly.
    """

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

        # Causal mask: allocated once, sliced per-call. Buffer so it moves
        # with the module (to GPU, into checkpoints) without ever receiving
        # gradients.
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

        Q = self.W_Q(x)                                              # (B, T, head_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)                          # (B, T, T)
        weights = self.attn_drop(weights)
        out = weights @ V                                             # (B, T, head_dim)
        return self.out_drop(self.W_O(out))
