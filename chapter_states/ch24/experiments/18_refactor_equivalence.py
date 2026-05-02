"""Experiment 18 — The refactored SingleHeadAttention with dropout=0 produces
identical output to a hand-coded Chapter-6 version.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mygpt import SingleHeadAttention, set_seed


class Ch6SingleHeadAttention(nn.Module):
    """The Chapter-6 version, locally for comparison."""

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
        Q = self.W_Q(x); K = self.W_K(x); V = self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return self.W_O(out)


def main() -> None:
    # Build both with the SAME seed-0 init order so their W_Q, W_K, W_V, W_O all match.
    set_seed(0)
    old = Ch6SingleHeadAttention(embed_dim=4, head_dim=4)

    set_seed(0)
    new = SingleHeadAttention(embed_dim=4, head_dim=4, max_seq_len=64, dropout=0.0)
    new.eval()

    # Same input
    set_seed(42)
    x = torch.randn(1, 4, 4)

    with torch.no_grad():
        out_old = old(x)
        out_new = new(x)

    print("OLD (Chapter 6):")
    print(out_old)
    print()
    print("NEW (Chapter 7):")
    print(out_new)
    print()
    print(f"identical:    {torch.equal(out_old, out_new)}")
    print(f"max abs diff: {(out_old - out_new).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
