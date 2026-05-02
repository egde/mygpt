"""Experiment 23 — LayerNorm reduces the §9.7 residual drift.

Stacks 30 randomly-initialised MLPs with residual connections, with and
without a LayerNorm before each MLP. With pre-LN, the residual stream's
scale grows more slowly across depth.
"""

import torch

from mygpt import MLP, LayerNorm, set_seed


def main() -> None:
    set_seed(0)
    C = 16
    n_layers = 30
    mlps = [MLP(embed_dim=C, dropout=0.0).eval() for _ in range(n_layers)]
    lns = [LayerNorm(embed_dim=C).eval() for _ in range(n_layers)]

    x0 = torch.randn(1, 8, C)
    print(f"input std: {x0.std().item():.4f}")
    print()

    print("WITH residuals + pre-LayerNorm (x = x + mlp(ln(x))):")
    with torch.no_grad():
        x = x0.clone()
        for i, (mlp, ln) in enumerate(zip(mlps, lns)):
            x = x + mlp(ln(x))
            if i in (0, 4, 9, 14, 19, 29):
                print(f"  after layer {i+1:2d}: std = {x.std().item():.6f}")


if __name__ == "__main__":
    main()
