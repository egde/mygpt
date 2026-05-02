import torch
import torch.nn as nn
import torch.nn.functional as F

from mygpt.block import TransformerBlock
from mygpt.embedding import TokenEmbedding
from mygpt.norm import LayerNorm


class GPT(nn.Module):
    """Full GPT-2-style decoder-only transformer with weight-tied head."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len=64, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"input length T={T} exceeds max_seq_len={self.max_seq_len}")
        positions = torch.arange(T, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.embed_drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = x @ self.token_embedding.embedding.weight.T
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
