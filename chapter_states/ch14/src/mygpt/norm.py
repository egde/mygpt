import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Per-token layer normalisation with learnable scale and bias.

    Inputs:
        x: tensor of shape (..., embed_dim).

    Outputs:
        tensor of the same shape, with each (..., :)-slice normalised to
        mean 0 and (population) variance 1, then scaled by `weight` and
        shifted by `bias`.

    Parameters: weight (embed_dim,) and bias (embed_dim,), both
    initialised to ones and zeros respectively (so the module starts as
    a pure normalisation).
    """

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
