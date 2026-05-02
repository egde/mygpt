import torch
import torch.nn as nn


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear -> Dropout.

    Applied independently to each token's vector. The intermediate width
    is 4 * embed_dim, the canonical choice for transformer MLPs.

    Inputs:
        x: tensor of shape (B, T, embed_dim).
    Outputs:
        tensor of shape (B, T, embed_dim).

    Has 2 * (embed_dim * 4*embed_dim) + (4*embed_dim + embed_dim)
       = 8 * embed_dim^2 + 5 * embed_dim   parameters (with biases).
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))
