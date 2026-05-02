"""Experiment 13 — Softmax by hand and by torch."""

import math

import torch
import torch.nn.functional as F


def main() -> None:
    z = torch.tensor([1.0, 2.0, 3.0])
    print(f"z = {z}")

    # By hand
    exp_z = torch.exp(z)
    print(f"exp(z) = {exp_z}")
    total = exp_z.sum().item()
    print(f"sum of exp(z) = {total:.4f}")
    by_hand = exp_z / total
    print(f"softmax(z) by hand = {by_hand}")
    print()

    # By torch
    by_torch = F.softmax(z, dim=-1)
    print(f"softmax(z) by torch = {by_torch}")
    print(f"sum by torch = {by_torch.sum().item():.4f}")
    print(f"identical: {torch.allclose(by_hand, by_torch)}")


if __name__ == "__main__":
    main()
