import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Map a tensor of integer token ids to a tensor of dense embedding vectors.

    Inputs:
        token_ids: long tensor of arbitrary shape `(...)` containing values
            in `[0, vocab_size)`.

    Outputs:
        tensor of shape `(..., embed_dim)` with each id replaced by its
        learned `embed_dim`-vector.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)
