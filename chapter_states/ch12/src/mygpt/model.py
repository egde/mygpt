import torch
import torch.nn as nn

from mygpt.block import TransformerBlock
from mygpt.embedding import TokenEmbedding
from mygpt.norm import LayerNorm


class GPT(nn.Module):
    """The full GPT-2-style decoder-only transformer.

    Inputs:
        ids: long tensor of shape (B, T) with values in [0, vocab_size).

    Outputs:
        logits: float tensor of shape (B, T, vocab_size), unnormalised.

    Architecture:
        token_embedding   (V, C) parameters
      + position_embedding (max_seq_len, C) parameters
      → embed_drop
      → N x TransformerBlock(C, num_heads)
      → ln_f (final LayerNorm)
      → head (tied to token_embedding.embedding.weight; no extra params)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
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
        # No separate head: we reuse self.token_embedding.embedding.weight in forward.

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(T, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)  # (B, T, C)
        x = self.embed_drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        # Tied head: logits = x @ E^T, where E is the token-embedding matrix.
        logits = x @ self.token_embedding.embedding.weight.T  # (B, T, V)
        return logits
