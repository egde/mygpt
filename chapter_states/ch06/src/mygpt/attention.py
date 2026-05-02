import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    """Single-head causal self-attention.

    Inputs:
        x: tensor of shape (B, T, embed_dim).

    Outputs:
        tensor of shape (B, T, embed_dim).

    Has three learnable projections (W_Q, W_K, W_V) of shape
    (embed_dim, head_dim) and a final output projection W_O of shape
    (head_dim, embed_dim). For single-head we set head_dim = embed_dim;
    for multi-head (Chapter 8) head_dim < embed_dim.
    """

    def __init__(self, embed_dim: int, head_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_O = nn.Linear(head_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.W_Q(x)                          # (B, T, head_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)                          # (B, T, T)
        out = weights @ V                                             # (B, T, head_dim)
        return self.W_O(out)
