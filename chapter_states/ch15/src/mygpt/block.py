import torch
import torch.nn as nn

from mygpt.attention import MultiHeadAttention
from mygpt.mlp import MLP
from mygpt.norm import LayerNorm


class TransformerBlock(nn.Module):
    """A single GPT-2 pre-norm transformer block.

    forward(x) computes:
        x = x + mha(ln1(x))      # residual around attention
        x = x + mlp(ln2(x))      # residual around MLP

    Inputs / outputs:
        (B, T, embed_dim) -> (B, T, embed_dim).

    Total parameters: 12 * embed_dim^2 + 9 * embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ln1 = LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
