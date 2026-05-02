"""Experiment 21 — Residual connections stabilise deep stacks.

Stacks 30 randomly-initialised MLPs and tracks the std of activations
after each layer, both with and without residual connections.
"""

import torch

from mygpt import MLP, set_seed


def main() -> None:
    set_seed(0)
    C = 16
    n_layers = 30
    mlps = [MLP(embed_dim=C, dropout=0.0).eval() for _ in range(n_layers)]

    x0 = torch.randn(1, 8, C)
    print(f"input std: {x0.std().item():.4f}")
    print()

    print("WITHOUT residuals:")
    with torch.no_grad():
        x = x0.clone()
        for i, mlp in enumerate(mlps):
            x = mlp(x)
            if i in (0, 4, 9, 14, 19, 29):
                print(f"  after layer {i+1:2d}: std = {x.std().item():.6f}")

    print()
    print("WITH residuals (x = x + mlp(x)):")
    with torch.no_grad():
        x = x0.clone()
        for i, mlp in enumerate(mlps):
            x = x + mlp(x)
            if i in (0, 4, 9, 14, 19, 29):
                print(f"  after layer {i+1:2d}: std = {x.std().item():.6f}")


if __name__ == "__main__":
    main()
