import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) / torch.sqrt(var + self.eps)
        return x_normed * self.weight + self.bias
