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


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Llama / Mistral default).

    Compared to LayerNorm:
      - drops the bias term,
      - drops the mean subtraction (no centring),
      - normalises by the root-mean-square of x along the last axis.

    Forward: ``x / sqrt(mean(x²) + eps) * weight``.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _make_norm(embed_dim: int, norm_type: str) -> nn.Module:
    """Norm-class selector. `norm_type` is 'layer' or 'rms'."""
    if norm_type == "layer":
        return LayerNorm(embed_dim)
    if norm_type == "rms":
        return RMSNorm(embed_dim)
    raise ValueError(f"unknown norm_type: {norm_type!r} (expected 'layer' or 'rms')")
