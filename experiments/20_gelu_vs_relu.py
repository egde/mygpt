"""Experiment 20 — GELU and ReLU compared on a few inputs.

GELU is smooth and slightly negative for moderately negative inputs;
ReLU is identically zero for all negative inputs and has a kink at 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    xs = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    print(f"x:    {xs}")
    print(f"gelu: {F.gelu(xs)}")
    print(f"relu: {F.relu(xs)}")
    print()

    g = nn.GELU()
    print(f"nn.GELU and F.gelu identical: {torch.equal(g(xs), F.gelu(xs))}")


if __name__ == "__main__":
    main()
