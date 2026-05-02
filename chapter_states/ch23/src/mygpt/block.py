import torch.nn as nn

from mygpt.attention import MultiHeadAttention
from mygpt.mlp import MLP
from mygpt.norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=64, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ln1 = LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
