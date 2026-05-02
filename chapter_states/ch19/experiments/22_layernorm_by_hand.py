"""Experiment 22 — Layer norm by hand on x = (1, 2, 3, 4)."""

import torch
import torch.nn as nn


def main() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    mean = x.mean()
    var = x.var(unbiased=False)         # divisor n, matches LayerNorm
    eps = 1e-5
    x_normed = (x - mean) / torch.sqrt(var + eps)

    print(f"x:           {x}")
    print(f"mean:        {mean.item():.6f}")
    print(f"var:         {var.item():.6f}")
    print(f"std:         {torch.sqrt(var).item():.6f}")
    print(f"x_normed:    {x_normed}")
    print(f"normed mean: {x_normed.mean().item():.6f}  (should be ~0)")
    print(f"normed std:  {x_normed.std(unbiased=False).item():.6f}  (should be ~1)")
    print()

    # Compare with torch's LayerNorm (with default gamma=1, beta=0)
    ln = nn.LayerNorm(4)
    ln.eval()
    out_torch = ln(x)
    print(f"nn.LayerNorm(4)(x):  {out_torch}")
    print(f"matches our by-hand: {torch.allclose(x_normed, out_torch, atol=1e-5)}")


if __name__ == "__main__":
    main()
