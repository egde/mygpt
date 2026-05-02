import torch.nn as nn

from mygpt.attention import MultiHeadAttention
from mygpt.mlp import MLP
from mygpt.norm import _make_norm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        max_seq_len=64,
        dropout=0.0,
        norm_type="layer",
        position_type="learned",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.position_type = position_type
        self.ln1 = _make_norm(embed_dim, norm_type)
        self.mha = MultiHeadAttention(
            embed_dim, num_heads, max_seq_len, dropout, position_type=position_type
        )
        self.ln2 = _make_norm(embed_dim, norm_type)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
