"""Experiment 05 — A minimal nn.Module: an affine map y = x W^T + b."""

import torch
import torch.nn as nn


class Affine(nn.Module):
    """y = x W^T + b — a single linear-then-shift layer.

    Parameters:
      weight: tensor of shape (out_features, in_features)
      bias:   tensor of shape (out_features,)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


def main() -> None:
    torch.manual_seed(0)

    m = Affine(in_features=3, out_features=2)
    print("Parameters in Affine(3, 2):")
    for name, p in m.named_parameters():
        print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
    print(f"Total: {sum(p.numel() for p in m.parameters())} parameters")

    x = torch.randn(4, 3)   # a "batch" of 4 vectors of length 3
    y = m(x)
    print(f"\nForward pass: input shape {tuple(x.shape)} -> output shape {tuple(y.shape)}")

    loss = y.sum()
    loss.backward()
    print(f"\nAfter backward, gradient shapes:")
    for name, p in m.named_parameters():
        print(f"  {name}.grad: shape={tuple(p.grad.shape)}")


if __name__ == "__main__":
    main()
