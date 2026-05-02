"""Experiment 14 — Causal self-attention by hand on a 3x2 input.

Uses identity projections (Q = K = V = X) so the scores are X X^T and
the math is checkable on paper. Compare the printed weights and output
to the by-hand calculation in §6.8.
"""

import math

import torch
import torch.nn.functional as F


def main() -> None:
    X = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    T, d = X.shape
    print(f"X (T={T}, C={d}):")
    print(X)
    print()

    # Step 1: scores = X X^T
    scores = X @ X.T
    print("scores = X X^T:")
    print(scores)
    print()

    # Step 2: scaled by 1/sqrt(d)
    scaled = scores / math.sqrt(d)
    print(f"scaled by 1/sqrt({d}) = 1/{math.sqrt(d):.4f}:")
    print(scaled)
    print()

    # Step 3: causal mask
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    masked = scaled + mask
    print("scaled + causal mask:")
    print(masked)
    print()

    # Step 4: softmax row-wise
    weights = F.softmax(masked, dim=-1)
    print("attention weights (lower-triangular, rows sum to 1):")
    print(weights)
    print(f"row sums: {weights.sum(dim=-1)}")
    print()

    # Step 5: output = weights @ V (V = X)
    out = weights @ X
    print("attention output = weights @ V:")
    print(out)


if __name__ == "__main__":
    main()
